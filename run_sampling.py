#!/usr/bin/env python
"""
采样便捷脚本（封装 sample_checkpoint.py）。

示例：
  python run_sampling.py --checkpoint path/to/model.ckpt
"""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sampling from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .ckpt file.")
    parser.add_argument("--config", type=str, default="configs/rgr.yaml", help="YAML config path.")
    parser.add_argument("--output_dir", type=str, default="sampling_results_top500", help="Output directory.")
    parser.add_argument("--n_samples", type=int, default=100, help="Samples per product.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device.")
    parser.add_argument("--sampling_seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cmd = [
        sys.executable,
        "sample_checkpoint.py",
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--output_dir",
        args.output_dir,
        "--n_samples",
        str(args.n_samples),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--sampling_seed",
        str(args.sampling_seed),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
