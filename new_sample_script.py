import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem

# 导入你项目中的必要模块
from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.frameworks.markov_bridge import MarkovBridge
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
# 导入所需的 "Dummy" 和其他辅助类
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.features.extra_features_molecular import ExtraMolecularFeatures
from src.metrics.molecular_metrics_discrete import DummyTrainMolecularMetricsDiscrete
from src.metrics.sampling_metrics import DummySamplingMolecularMetrics


def main(args):
    """
    主采样函数
    """
    # 1. 初始化和设置
    torch_device = torch.device(args.device)
    set_deterministic(args.sampling_seed)

    config_args = parse_yaml_config(
        argparse.Namespace(config=open(args.config, 'r')))

    for key, value in vars(args).items():
        if value is not None:
            setattr(config_args, key, value)
    args = config_args

    # 2. 准备输出路径
    data_root = os.path.join(args.data, args.dataset)
    checkpoint_name = os.path.basename(args.checkpoint).replace('.ckpt', '')

    os.makedirs(args.output_dir, exist_ok=True)
    table_name = f'{checkpoint_name}_mode={args.mode}_n={args.n_samples}_seed={args.sampling_seed}_reranked.csv'
    table_path = os.path.join(args.output_dir, table_name)

    print(f"开始采样，结果将保存至: {table_path}")
    print("注意: 检查点必须是通过表征对齐训练的，否则重排会失败。")

    # 3. 准备模型初始化所需的参数
    print("正在准备模型初始化参数...")
    datamodule = RetroBridgeDataModule(
        data_root=data_root, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, extra_nodes=args.extra_nodes, evaluation=True, swap=args.swap,
    )
    dataset_infos = RetroBridgeDatasetInfos(datamodule)

    extra_features = ExtraFeatures(
        args.extra_features, dataset_info=dataset_infos) if args.extra_features is not None else DummyExtraFeatures()
    domain_features = ExtraMolecularFeatures(
        dataset_infos=dataset_infos) if args.extra_molecular_features else DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, extra_features=extra_features,
        domain_features=domain_features, use_context=args.use_context,
    )

    model_kwargs = {
        'chains_dir': os.path.join(args.logs, 'chains'), 'graphs_dir': os.path.join(args.logs, 'graphs'),
        'checkpoints_dir': args.checkpoints, 'transition': args.transition, 'dataset_infos': dataset_infos,
        'train_metrics': DummyTrainMolecularMetricsDiscrete(), 'sampling_metrics': DummySamplingMolecularMetrics(),
        'visualization_tools': None, 'extra_features': extra_features, 'domain_features': domain_features,
        'log_every_steps': args.log_every_steps, 'sample_every_val': args.sample_every_val,
        'samples_to_generate': args.samples_to_generate, 'samples_to_save': args.samples_to_save,
        'samples_per_input': args.samples_per_input, 'chains_to_save': args.chains_to_save,
        'number_chain_steps_to_save': args.number_chain_steps_to_save,
    }

    # 4. 加载模型
    print("正在加载模型...")
    model = MarkovBridge.load_from_checkpoint(
        args.checkpoint, map_location=torch_device, strict=False, **model_kwargs
    )
    model.eval().to(torch_device)
    if args.n_steps:
        model.T = args.n_steps
    print("模型加载完毕。")

    # <--- MODIFICATION START: 加载用于重排的模块和数据 --->
    print("正在加载用于重排的 Embeddings 和对齐模块...")
    alignment_mlp2 = model.alignment_mlp2.to(torch_device)

    # 根据采样模式选择对应的 embedding 文件
    embedding_filename = f'rxn_encoded_prod_uspto50k_{args.mode}.pt'
    embedding_path = os.path.join('embeddings', embedding_filename)
    if not os.path.exists(embedding_path):
        print(f"警告: 未找到 {embedding_path}。将回退使用训练集的 embedding。")
        embedding_path = 'embeddings/rxn_encoded_prod_uspto50k_train.pt'

    prod_embeddings = torch.load(embedding_path, map_location=torch_device)
    print(f"已加载 {embedding_path}")
    # <--- MODIFICATION END --->

    # 5. 准备数据集加载器
    dataloader = datamodule.test_dataloader(
    ) if args.mode == 'test' else datamodule.val_dataloader()

    # 6. 执行采样循环
    group_size = args.n_samples
    all_results = []

    # 可选限制：按批或按样本数限制处理范围
    max_batches = getattr(args, 'limit_batches', None)
    max_items = getattr(args, 'limit_items', None)
    if max_batches is not None and max_items is not None:
        raise ValueError('limit_batches 与 limit_items 不能同时使用')

    written = 0

    for i, data in enumerate(tqdm(dataloader, desc="正在处理批次")):
        if max_batches is not None and i >= max_batches:
            break
        data = data.to(torch_device)
        bs = len(data.batch.unique())

        batch_groups_mols = []
        batch_groups_y_reps = []  # <--- MODIFICATION: 存储y表征 --->
        ground_truth = []
        input_products = []

        for sample_idx in range(group_size):
            # <--- MODIFICATION: 接收 final_y_reps --->
            pred_molecule_list, true_molecule_list, products_list, _, nlls, ells, final_y_reps = model.sample_batch(
                data=data, batch_id=i, batch_size=bs, save_final=0, keep_chain=0,
                number_chain_steps_to_save=1, sample_idx=sample_idx, save_true_reactants=True,
                use_one_hot=False,
            )

            batch_groups_mols.append(pred_molecule_list)
            # <--- MODIFICATION: 存储y表征 --->
            batch_groups_y_reps.append(final_y_reps.cpu())

            if sample_idx == 0:
                ground_truth.extend(true_molecule_list)
                input_products.extend(products_list)

        # 7. 处理并整理当前批次的结果
        original_indices = data.idx.cpu()

        bs_to_write = bs
        if max_items is not None:
            remaining = max(0, max_items - written)
            if remaining == 0:
                break
            bs_to_write = min(bs, remaining)

        for mol_idx_in_batch in range(bs_to_write):
            true_mol_data = ground_truth[mol_idx_in_batch]
            product_mol_data = input_products[mol_idx_in_batch]

            true_mol_rdkit, _ = build_molecule(
                true_mol_data[0], true_mol_data[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True
            )
            true_smi = Chem.MolToSmiles(
                true_mol_rdkit) if true_mol_rdkit else "INVALID"

            product_mol_rdkit = build_molecule(
                product_mol_data[0], product_mol_data[1], dataset_infos.atom_decoder
            )
            product_smi = Chem.MolToSmiles(
                product_mol_rdkit) if product_mol_rdkit else "INVALID"

            # <--- MODIFICATION START: 计算新的重排分数 --->
            prod_idx = original_indices[mol_idx_in_batch]
            prod_emb = prod_embeddings[prod_idx].unsqueeze(0)
            # <--- MODIFICATION END --->

            for sample_idx in range(group_size):
                pred_mol_data = batch_groups_mols[sample_idx][mol_idx_in_batch]

                pred_mol_rdkit, _ = build_molecule(
                    pred_mol_data[0], pred_mol_data[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True
                )
                pred_smi = Chem.MolToSmiles(
                    pred_mol_rdkit) if pred_mol_rdkit else "INVALID"

                # <--- MODIFICATION START: 计算新的重排分数 --->
                with torch.no_grad():
                    reactant_y_rep = batch_groups_y_reps[sample_idx][mol_idx_in_batch].unsqueeze(
                        0).to(torch_device)
                    aligned_reactant_emb = alignment_mlp2(reactant_y_rep)
                    similarity_score = F.cosine_similarity(
                        aligned_reactant_emb, prod_emb).item()
                # <--- MODIFICATION END --->

                all_results.append({
                    'product': product_smi,
                    'pred': pred_smi,
                    'true': true_smi,
                    'score': similarity_score  # <--- MODIFICATION: 使用新的分数 --->
                })
                written += 1

        if max_items is not None and written >= max_items:
            break

    # 8. 保存最终结果到 CSV
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(table_path, index=False)
    print(f"采样完成！结果已保存到 {table_path}")


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser(
        description="使用 RetroBridge 模型进行采样并基于表征相似度进行重排")

    # --- 核心参数 ---
    parser.add_argument('--checkpoint', type=str,
                        required=True, help='要加载的模型检查点文件路径')
    parser.add_argument('--n_samples', type=int,
                        default=10, help='每个产物分子的采样次数')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'val'], help='使用哪个数据集进行采样 (test/val)')

    # --- 可选高级参数 ---
    parser.add_argument(
        '--config', type=str, default='configs/retrobridge.yaml', help='模型和数据的基础配置文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='samples_reranked', help='保存重排后采样结果的目录')
    parser.add_argument('--batch_size', type=int,
                        default=None, help='推理时的批处理大小 (默认使用配置文件中的值)')
    parser.add_argument('--n_steps', type=int, default=500, help='扩散/桥接过程的步数')
    parser.add_argument('--sampling_seed', type=int,
                        default=42, help='采样过程的随机种子')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available()
                        else 'cpu', help='运行设备的名称 (例如 "cuda:0" 或 "cpu")')

    # 添加一些从配置文件中读取但在加载模型时可能需要的参数的默认值
    parser.add_argument('--data', default='data', help=argparse.SUPPRESS)
    parser.add_argument('--dataset', default='uspto50k',
                        help=argparse.SUPPRESS)
    parser.add_argument('--logs', default='logs', help=argparse.SUPPRESS)
    parser.add_argument(
        '--checkpoints', default='checkpoints', help=argparse.SUPPRESS)

    # 可选限制参数：只处理前若干个 batch 或前若干条样本
    parser.add_argument('--limit_batches', type=int, default=None,
                        help='仅处理前 N 个 batch（与 limit_items 互斥）')
    parser.add_argument('--limit_items', type=int, default=None,
                        help='仅处理前 N 个样本（与 limit_batches 互斥）')

    parsed_args = parser.parse_args()
    main(parsed_args)
