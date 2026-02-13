# RGR

This repository contains training and sampling code for the RGR instantiation.

## Environment

Recommended: Python 3.9 + conda-forge RDKit.

```bash
conda create -n rgr python=3.9 rdkit=2023.09.5 -c conda-forge -y
conda activate rgr
pip install -r requirements.txt
```

## Data (USPTO-50K)

Default dataset root: `data/uspto50k/`.

On the first run, USPTO-50K split CSV files will be downloaded to `data/uspto50k/raw/` automatically.

## Build teacher embeddings

Before training, generate teacher embeddings (saved as `.pt` files under `embeddings/`):

```bash
python scripts/build_alignment_embeddings.py \
  --data_root data/uspto50k \
  --out_dir embeddings \
  --fp morgan \
  --radius 2 \
  --n_bits 512
```

## Training

```bash
python train.py --config configs/rgr.yaml --model RGR
```

Outputs:
- checkpoints: `checkpoints/<run_name>/...`
- logs: `logs/{chains,graphs,lightning_logs}/<run_name>/...`

## Sampling

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

## License

See `LICENSE.txt` (CC BY-NC 4.0).
