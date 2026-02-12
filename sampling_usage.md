# 采样与评测说明

## 1) 训练得到 checkpoint

先按 `README.md` 完成训练，得到类似 `checkpoints/<run_name>/.../*.ckpt` 的模型文件。

## 2) 采样（推荐）

使用 `sample.py`：

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

输出 CSV 默认在 `samples/<dataset>_<mode>/` 下。

## 3) 采样（便捷脚本）

如果你希望用一个更短的命令调用 `sample_checkpoint.py`，可以使用：

```bash
python run_sampling.py --checkpoint <path/to/ckpt>
```

可选参数（示例）：

```bash
python run_sampling.py \
  --checkpoint <path/to/ckpt> \
  --config configs/rgr.yaml \
  --output_dir sampling_results_top500 \
  --n_samples 100 \
  --batch_size 16 \
  --device gpu \
  --sampling_seed 42
```

