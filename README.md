# RGR

[![CI](https://img.shields.io/github/actions/workflow/status/Eric-YHS/RGR/ci.yml?branch=main&logo=githubactions&logoColor=white&label=CI)](https://github.com/Eric-YHS/RGR/actions/workflows/ci.yml)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](#environment)

Training and sampling code for the RGR instantiation.

It builds on the RetroBridge Markov-bridge graph diffusion stack (`src/frameworks`,
`src/models`) and adds a representation-alignment branch: teacher embeddings are built offline with
`scripts/build_alignment_embeddings.py`, reduced to 64-D (whitened PCA + L2 normalisation) by
`src/alignment/pca_utils.py`, and matched by two alignment MLP heads inside `src/frameworks/markov_bridge.py`
(`compute_alignment_loss`, cosine similarity, with a `WeightScheduler` ramping the loss weight up to 0.6).
Note that `markov_bridge.py` loads them from the hard-coded relative path `embeddings/`, so training has to
be started from the repository root after running the embedding script.

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
