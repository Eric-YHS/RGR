import os
import pickle
from typing import Dict, Optional

import numpy as np
import torch


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def _try_load(path: str):
    if os.path.exists(path):
        return torch.load(path, map_location='cpu')
    return None


def _save_tensor(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(torch.from_numpy(arr).float(), path)


def prepare_and_cache_embeddings(
    emb_dir: str,
    n_components: int = 64,
    force_recompute: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Prepare PCA(whiten)->L2 embeddings for product and react across train/val.

    - Fits PCA separately on train product and train react (512->64, whiten=True)
    - Transforms train/val with the trained PCA, then L2-normalizes
    - Caches PCA params (pickle) and transformed tensors (pt)

    Returns a dict with keys:
        'prod_train', 'react_train', 'prod_val', 'react_val'
    """
    # Input files
    prod_train_fp = os.path.join(emb_dir, 'rxn_encoded_prod_uspto50k_train.pt')
    react_train_fp = os.path.join(emb_dir, 'rxn_encoded_react_tensor.pt')
    prod_val_fp = os.path.join(emb_dir, 'rxn_encoded_prod_uspto50k_val.pt')
    react_val_fp = os.path.join(emb_dir, 'rxn_encoded_react_tensor_val.pt')

    # Output files
    prod_train_out = os.path.join(emb_dir, 'prod_64_train.pt')
    react_train_out = os.path.join(emb_dir, 'react_64_train.pt')
    prod_val_out = os.path.join(emb_dir, 'prod_64_val.pt')
    react_val_out = os.path.join(emb_dir, 'react_64_val.pt')

    pca_prod_pkl = os.path.join(emb_dir, 'pca_prod.pkl')
    pca_react_pkl = os.path.join(emb_dir, 'pca_react.pkl')

    # Fast path: load cached
    if (not force_recompute and
        all(os.path.exists(fp) for fp in [prod_train_out, react_train_out, prod_val_out, react_val_out]) and
            all(os.path.exists(fp) for fp in [pca_prod_pkl, pca_react_pkl])):
        return {
            'prod_train': torch.load(prod_train_out, map_location='cpu'),
            'react_train': torch.load(react_train_out, map_location='cpu'),
            'prod_val': torch.load(prod_val_out, map_location='cpu'),
            'react_val': torch.load(react_val_out, map_location='cpu'),
        }

    # Lazy import sklearn only when recomputing
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise RuntimeError(
            'scikit-learn is required to compute PCA embeddings. '
            'Please install scikit-learn and retry.'
        ) from e

    missing = [fp for fp in [prod_train_fp, react_train_fp, prod_val_fp, react_val_fp] if not os.path.exists(fp)]
    if missing:
        missing_str = '\n'.join([f'  - {m}' for m in missing])
        raise RuntimeError(
            'Missing raw teacher embedding files required for PCA caching:\n'
            f'{missing_str}\n\n'
            'Generate them first, e.g.:\n'
            '  python scripts/build_alignment_embeddings.py --data_root data/uspto50k --out_dir embeddings\n'
        )

    # Load originals
    prod_train_np = torch.load(prod_train_fp, map_location='cpu').numpy()
    react_train_np = torch.load(react_train_fp, map_location='cpu').numpy()
    prod_val_np = torch.load(prod_val_fp, map_location='cpu').numpy()
    react_val_np = torch.load(react_val_fp, map_location='cpu').numpy()

    # Product PCA
    pca_prod = PCA(n_components=n_components, whiten=True, svd_solver='auto', random_state=0)
    prod_train_pca = pca_prod.fit_transform(prod_train_np)
    prod_val_pca = pca_prod.transform(prod_val_np)

    # React PCA
    pca_react = PCA(n_components=n_components, whiten=True, svd_solver='auto', random_state=0)
    react_train_pca = pca_react.fit_transform(react_train_np)
    react_val_pca = pca_react.transform(react_val_np)

    # L2 normalize
    prod_train_pca = _l2_normalize(prod_train_pca)
    prod_val_pca = _l2_normalize(prod_val_pca)
    react_train_pca = _l2_normalize(react_train_pca)
    react_val_pca = _l2_normalize(react_val_pca)

    # Save PCA params
    with open(pca_prod_pkl, 'wb') as f:
        pickle.dump(pca_prod, f)
    with open(pca_react_pkl, 'wb') as f:
        pickle.dump(pca_react, f)

    # Save tensors
    _save_tensor(prod_train_out, prod_train_pca)
    _save_tensor(prod_val_out, prod_val_pca)
    _save_tensor(react_train_out, react_train_pca)
    _save_tensor(react_val_out, react_val_pca)

    return {
        'prod_train': torch.from_numpy(prod_train_pca).float(),
        'react_train': torch.from_numpy(react_train_pca).float(),
        'prod_val': torch.from_numpy(prod_val_pca).float(),
        'react_val': torch.from_numpy(react_val_pca).float(),
    }
