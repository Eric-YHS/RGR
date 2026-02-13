# RGR

本仓库整理并开源了 RGR 实例化相关的训练与采样代码。

## 环境

建议使用 Python 3.9 + conda-forge RDKit：

```bash
conda create -n rgr python=3.9 rdkit=2023.09.5 -c conda-forge -y
conda activate rgr
pip install -r requirements.txt
```

## 数据（USPTO-50K）

默认数据目录为 `data/uspto50k/`。首次运行训练/脚本时，会自动下载 USPTO-50K 的 split CSV 到 `data/uspto50k/raw/`。

## 生成对齐用 teacher embeddings

训练图级对齐前，需要先生成 teacher embeddings（并在训练时自动进行 PCA(whiten)+L2 缓存为 64 维对齐目标）：

```bash
python scripts/build_alignment_embeddings.py \
  --data_root data/uspto50k \
  --out_dir embeddings \
  --fp morgan \
  --radius 2 \
  --n_bits 512
```

该脚本会为 train/val/test 生成并保存 product/reactants 的图级 embedding（`.pt`），供训练时读取。

## 训练

```bash
python train.py --config configs/rgr.yaml --model RGR
```

默认产物目录：
- checkpoints：`checkpoints/<run_name>/...`
- logs：`logs/{chains,graphs,lightning_logs}/<run_name>/...`

## 采样

```bash
python sample.py \
  --config configs/rgr.yaml \
  --checkpoint <path/to/ckpt> \
  --samples samples \
  --model RGR \
  --mode test \
  --n_samples 10 \
  --n_steps 500 \
  --sampling_seed 1
```

## 许可证

见 `LICENSE.txt`（CC BY-NC 4.0）。
