import argparse
import os
import torch
from tqdm import tqdm
from rdkit import Chem

# 导入您项目中的必要模块
from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.frameworks.markov_bridge import MarkovBridge
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
from src.features.extra_features import DummyExtraFeatures, ExtraFeatures
from src.features.extra_features_molecular import ExtraMolecularFeatures
from src.metrics.molecular_metrics_discrete import DummyTrainMolecularMetricsDiscrete
from src.metrics.sampling_metrics import DummySamplingMolecularMetrics


def main(args):
    """
    主函数：执行采样并保存包含完整表征轨迹的富数据文件。
    """
    # 1. 初始化和设置
    torch_device = torch.device(args.device)
    set_deterministic(args.sampling_seed)

    # 从 YAML 文件加载配置，并允许命令行参数覆盖
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
    output_filename = f'{checkpoint_name}_mode={args.mode}_n={args.n_samples}_seed={args.sampling_seed}_trajectories.pt'
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"开始生成轨迹数据，结果将保存至: {output_path}")

    # 3. 加载模型所需的所有依赖项
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

    # 5. 准备数据集加载器
    dataloader = datamodule.test_dataloader(
    ) if args.mode == 'test' else datamodule.val_dataloader()

    # 6. 执行采样循环并构建结果列表
    group_size = args.n_samples
    results_list = []  # 这是我们将要保存的主要数据结构

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

        # 初始化用于收集一个批次内所有样本数据的容器
        batch_mol_samples = [[] for _ in range(bs)]
        batch_traj_samples = [[] for _ in range(bs)]

        # 运行 N 次采样
        for sample_idx in range(group_size):
            # 调用修改后的 sample_batch，接收 trajectories
            pred_mol_list, true_mol_list, prod_mol_list, _, _, _, trajectories = model.sample_batch(
                data=data, batch_id=i, batch_size=bs, save_final=0, keep_chain=0,
                number_chain_steps_to_save=1, sample_idx=sample_idx, save_true_reactants=True,
                use_one_hot=False,
            )
            # trajectories shape: [bs, T, 2, hidden_dim]

            # 将该次采样的结果分配到每个输入分子上
            for mol_idx in range(bs):
                batch_mol_samples[mol_idx].append(pred_mol_list[mol_idx])
                # 将轨迹移动到 CPU 以节省 GPU 显存，因为后续处理不需要在 GPU 上进行
                batch_traj_samples[mol_idx].append(trajectories[mol_idx].cpu())

        # 7. 为当前批次的所有分子构建结构化的数据字典
        bs_to_write = bs
        if max_items is not None:
            remaining = max(0, max_items - written)
            if remaining == 0:
                break
            bs_to_write = min(bs, remaining)

        for mol_idx in range(bs_to_write):
            true_mol_data = true_mol_list[mol_idx]
            prod_mol_data = prod_mol_list[mol_idx]

            # 将 RDKit 分子对象转换为 SMILES 字符串
            true_mol_rdkit, _ = build_molecule(
                true_mol_data[0], true_mol_data[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True)
            true_smi = Chem.MolToSmiles(
                true_mol_rdkit) if true_mol_rdkit else "INVALID"

            prod_mol_rdkit = build_molecule(
                prod_mol_data[0], prod_mol_data[1], dataset_infos.atom_decoder)
            prod_smi = Chem.MolToSmiles(
                prod_mol_rdkit) if prod_mol_rdkit else "INVALID"

            # 创建该产物的主要条目
            product_data_entry = {
                'product_smiles': prod_smi,
                'true_reactant_smiles': true_smi,
                'samples': []
            }

            # 填充该产物下的 N 个采样结果
            for sample_idx in range(group_size):
                pred_mol_data = batch_mol_samples[mol_idx][sample_idx]
                pred_mol_rdkit, _ = build_molecule(
                    pred_mol_data[0], pred_mol_data[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True)
                pred_smi = Chem.MolToSmiles(
                    pred_mol_rdkit) if pred_mol_rdkit else "INVALID"

                sample_entry = {
                    'predicted_reactant_smiles': pred_smi,
                    # shape: [T, 2, hidden_dim]
                    'trajectory': batch_traj_samples[mol_idx][sample_idx]
                }
                product_data_entry['samples'].append(sample_entry)

            results_list.append(product_data_entry)
            written += 1

        if max_items is not None and written >= max_items:
            break

    # 8. 保存最终的包含所有信息的列表到 .pt 文件
    torch.save(results_list, output_path)
    print(f"轨迹生成完成！所有数据已保存到 {output_path}")
    print("\n下一步: 使用 analyze_trajectories.py 脚本来处理这个文件并生成用于评估的 CSV。")


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser(
        description="使用 RetroBridge 模型进行采样并保存完整的表征轨迹")

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
                        default='trajectory_data', help='保存轨迹数据的目录')
    parser.add_argument('--batch_size', type=int,
                        default=None, help='推理时的批处理大小 (默认使用配置文件中的值)')
    parser.add_argument('--n_steps', type=int, default=500, help='扩散/桥接过程的步数')
    parser.add_argument('--sampling_seed', type=int,
                        default=42, help='采样过程的随机种子')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available()
                        else 'cpu', help='运行设备的名称 (例如 "cuda:0" 或 "cpu")')

    # --- 隐藏的、从配置文件读取的参数，为 argparse 提供默认值 ---
    parser.add_argument('--data', default='data', help=argparse.SUPPRESS)
    parser.add_argument('--dataset', default='uspto50k',
                        help=argparse.SUPPRESS)
    parser.add_argument('--logs', default='logs', help=argparse.SUPPRESS)
    parser.add_argument(
        '--checkpoints', default='checkpoints', help=argparse.SUPPRESS)

    # 可选限制参数：只处理前若干个 batch 或前若干条样本（与上面的参数互斥）
    parser.add_argument('--limit_batches', type=int, default=None,
                        help='仅处理前 N 个 batch')
    parser.add_argument('--limit_items', type=int, default=None,
                        help='仅处理前 N 个样本')

    parsed_args = parser.parse_args()
    main(parsed_args)
