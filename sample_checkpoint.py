#!/usr/bin/env python
"""
使用指定checkpoint在测试集前500条数据上进行采样的脚本
每条数据采样100次
参考train.py的实现方式
"""
import argparse
import os
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.frameworks.markov_bridge import MarkovBridge
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
from src.frameworks.noise_schedule import PredefinedNoiseScheduleDiscrete, InterpolationTransition
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.metrics.sampling_metrics import SamplingMolecularMetrics
from src.features.extra_features import ExtraFeatures
from src.features.extra_features_molecular import ExtraMolecularFeatures

from rdkit import Chem
import pytorch_lightning as pl


def create_model_from_args(args, config, dataset_infos, datamodule, extra_features, domain_features):
    """创建模型实例，参考train.py的实现"""
    
    # 创建目录
    experiment = f"{args.model}_{args.dataset}"
    chains_dir = os.path.join(args.logs, f'{args.dataset}_chains')
    graphs_dir = os.path.join(args.logs, f'{args.dataset}_graphs')
    checkpoints_dir = args.checkpoints
    os.makedirs(chains_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 初始化噪声调度和转换
    if config['transition'] is None:
        noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule=config['diffusion_noise_schedule'],
            timesteps=config['diffusion_steps']
        )
        transition = InterpolationTransition(
            x_classes=dataset_infos.output_dims['X'],
            e_classes=dataset_infos.output_dims['E'],
            y_classes=dataset_infos.output_dims['y']
        )
    else:
        transition = config['transition']
    
    # 初始化指标
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(
        dataset_infos, datamodule.train_smiles)
    
    # 设置可视化工具为None
    visualization_tools = None
    
    # 创建模型
    model = MarkovBridge(
        experiment_name=experiment,
        chains_dir=chains_dir,
        graphs_dir=graphs_dir,
        checkpoints_dir=checkpoints_dir,
        diffusion_steps=config['diffusion_steps'],
        diffusion_noise_schedule=config['diffusion_noise_schedule'],
        transition=transition,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        n_layers=config['n_layers'],
        hidden_mlp_dims=config['hidden_mlp_dims'],
        hidden_dims=config['hidden_dims'],
        lambda_train=config['lambda_train'],
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
        use_context=config['use_context'],
        log_every_steps=config['log_every_steps'],
        sample_every_val=config['sample_every_val'],
        samples_to_generate=config['samples_to_generate'],
        samples_to_save=config['samples_to_save'],
        samples_per_input=config['samples_per_input'],
        chains_to_save=config['chains_to_save'],
        number_chain_steps_to_save=config['number_chain_steps_to_save'],
        fix_product_nodes=config['fix_product_nodes'],
        loss_type=config['loss_type'],
    )
    
    return model


def main(args):
    # 设置设备
    torch_device = 'cuda:0' if args.device == 'gpu' else 'cpu'
    print(f"使用设备: {torch_device}")
    
    # 设置路径
    data_root = os.path.join(args.data, args.dataset)
    checkpoint_name = os.path.basename(args.checkpoint).replace('.ckpt', '')
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    output_file = f'{checkpoint_name}_top500_{args.n_samples}samples.csv'
    output_path = os.path.join(output_dir, output_file)
    
    print(f"采样结果将保存到: {output_path}")
    
    # 读取配置
    if isinstance(args.config, str):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # 如果是文件对象（来自argparse.FileType）
        with open(args.config.name, 'r') as f:
            config = yaml.safe_load(f)
    
    # 首先准备数据模块和dataset_infos
    print("正在准备数据模块...")
    datamodule = RetroBridgeDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        extra_nodes=args.extra_nodes,
        evaluation=False,
        swap=args.swap,
    )
    dataset_infos = RetroBridgeDatasetInfos(datamodule)
    
    # 计算输入输出维度
    extra_features = ExtraFeatures(config['extra_features'], dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
        use_context=config['use_context'],
    )
    
    # 确保dataset_infos已经正确初始化
    print(f"Output dims: {dataset_infos.output_dims}")
    
    # 创建模型
    print("正在创建模型...")
    model = create_model_from_args(args, config, dataset_infos, datamodule, extra_features, domain_features)
    
    # 加载checkpoint权重
    print("正在加载checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=torch_device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(torch_device)
    model.eval()
    
    # 设置模型参数
    model.visualization_tools = None
    model.T = args.n_steps if args.n_steps is not None else config['diffusion_steps']
    
    # 设置随机种子
    set_deterministic(args.sampling_seed)
    
    # 读取测试数据，只取前500条
    print("正在加载测试数据...")
    test_df = pd.read_csv(os.path.join(data_root, f'{args.dataset}_test.csv'))
    test_df = test_df.head(500)  # 只取前500条
    print(f"使用前 {len(test_df)} 条测试数据")
    
    # 准备保存结果
    all_results = []
    
    # 处理每个批次
    dataloader = datamodule.test_dataloader()
    
    for batch_idx, data in enumerate(tqdm(dataloader, desc="处理批次")):
        if batch_idx * args.batch_size >= 500:
            break
            
        # 获取当前批次的实际大小
        bs = len(data.batch.unique())
        
        # 为每条数据采样多次
        for sample_idx in tqdm(range(args.n_samples), desc=f"批次 {batch_idx+1} 采样", leave=False):
            data = data.to(torch_device)
            
            # 执行采样
            pred_molecule_list, true_molecule_list, products_list, scores, nlls, ells = model.sample_batch(
                data=data,
                batch_id=batch_idx,
                batch_size=bs,
                save_final=0,
                keep_chain=0,
                number_chain_steps_to_save=1,
                sample_idx=sample_idx,
                save_true_reactants=True,
                use_one_hot=args.use_one_hot,
            )
            
            # 处理每个分子
            for mol_idx in range(bs):
                global_idx = batch_idx * args.batch_size + mol_idx
                
                if global_idx >= 500:
                    break
                
                # 构建产物分子
                product_mol = build_molecule(
                    products_list[mol_idx][0], 
                    products_list[mol_idx][1], 
                    dataset_infos.atom_decoder
                )
                product_smi = Chem.MolToSmiles(product_mol)
                
                # 构建真实反应物分子
                true_mol, true_n_dummy_atoms = build_molecule(
                    true_molecule_list[mol_idx][0], 
                    true_molecule_list[mol_idx][1], 
                    dataset_infos.atom_decoder, 
                    return_n_dummy_atoms=True
                )
                true_smi = Chem.MolToSmiles(true_mol)
                
                # 构建预测反应物分子
                pred_mol, n_dummy_atoms = build_molecule(
                    pred_molecule_list[mol_idx][0], 
                    pred_molecule_list[mol_idx][1], 
                    dataset_infos.atom_decoder, 
                    return_n_dummy_atoms=True
                )
                pred_smi = Chem.MolToSmiles(pred_mol)
                
                # 保存结果
                result = {
                    'id': test_df.iloc[global_idx]['id'],
                    'class': test_df.iloc[global_idx]['class'],
                    'sample_idx': sample_idx,
                    'product': product_smi,
                    'pred': pred_smi,
                    'true': true_smi,
                    'score': scores[mol_idx],
                    'true_n_atoms': RetroBridgeDatasetInfos.max_n_dummy_nodes - true_n_dummy_atoms,
                    'pred_n_atoms': RetroBridgeDatasetInfos.max_n_dummy_nodes - n_dummy_atoms,
                    'nll': nlls[mol_idx],
                    'ell': ells[mol_idx],
                }
                all_results.append(result)
        
        # 每处理完一个批次就保存一次结果
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False)
            print(f"已保存 {len(all_results)} 条结果到 {output_path}")
    
    print(f"采样完成！总共生成了 {len(all_results)} 条样本")
    print(f"结果保存在: {output_path}")


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser(description='使用指定checkpoint进行采样')
    
    # 添加配置文件参数
    parser.add_argument('--config', type=argparse.FileType(mode='r'), required=True,
                       help='配置文件路径')
    
    # 基本参数
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='checkpoint文件路径')
    parser.add_argument('--output_dir', type=str, default='sampling_results',
                       help='输出目录')
    parser.add_argument('--data', type=str, default='datasets',
                       help='数据根目录')
    parser.add_argument('--dataset', type=str, default='uspto50k',
                       help='数据集名称')
    parser.add_argument('--model', type=str, default='RGR',
                       help='模型名称')
    parser.add_argument('--logs', type=str, default='logs',
                       help='日志目录')
    parser.add_argument('--checkpoints', type=str, default='checkpoints',
                       help='checkpoint目录')
    
    # 模型参数
    parser.add_argument('--extra_nodes', type=int, default=10,
                       help='额外的虚拟节点数')
    parser.add_argument('--swap', action='store_true',
                       help='是否交换产物和反应物')
    parser.add_argument('--use_one_hot', action='store_true',
                       help='是否使用one-hot编码')
    
    # 采样参数
    parser.add_argument('--n_samples', type=int, default=100,
                       help='每条数据的采样次数')
    parser.add_argument('--n_steps', type=int, default=None,
                       help='扩散步数')
    parser.add_argument('--sampling_seed', type=int, default=42,
                       help='随机种子')
    
    # 运行参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'],
                       help='使用的设备')
    
    # 使用parse_yaml_config解析参数
    args = parse_yaml_config(parser.parse_args())
    
    main(args)
