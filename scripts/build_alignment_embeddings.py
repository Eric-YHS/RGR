import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


DOWNLOAD_URL_TEMPLATE = "https://zenodo.org/record/8114657/files/{fname}?download=1"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    csv_name: str


SPLITS = [
    SplitSpec(name="train", csv_name="uspto50k_train.csv"),
    SplitSpec(name="val", csv_name="uspto50k_val.csv"),
    SplitSpec(name="test", csv_name="uspto50k_test.csv"),
]


def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    try:
        import urllib.request

        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dst_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {url} to {dst_path}. "
            "Please download the file manually and retry."
        ) from e


def _ensure_uspto50k_csvs(data_root: str) -> None:
    raw_dir = os.path.join(data_root, "raw")
    for spec in SPLITS:
        csv_path = os.path.join(raw_dir, spec.csv_name)
        url = DOWNLOAD_URL_TEMPLATE.format(fname=spec.csv_name)
        _download(url, csv_path)


def _iter_reactions(csv_path: str) -> Iterable[Tuple[str, str]]:
    table = pd.read_csv(csv_path)
    # Official USPTO-50K split files use this column name.
    col = "reactants>reagents>production"
    if col not in table.columns:
        raise KeyError(f"Column {col!r} not found in {csv_path}")
    for rxn in table[col].astype(str).tolist():
        parts = rxn.split(">")
        if len(parts) != 3:
            yield "", ""
            continue
        reactants_smi, _, product_smi = parts
        yield reactants_smi, product_smi


def _mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Remove atom-map numbers to make fingerprints map-invariant.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def _morgan_fp_bits(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def _save_tensor(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(torch.from_numpy(arr).float(), path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build graph-level teacher embeddings for graph-level alignment (USPTO-50K)."
    )
    parser.add_argument("--data_root", type=str, default="data/uspto50k", help="Dataset root dir.")
    parser.add_argument("--out_dir", type=str, default="embeddings", help="Output directory.")
    parser.add_argument("--fp", type=str, default="morgan", choices=["morgan"], help="Fingerprint type.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius.")
    parser.add_argument("--n_bits", type=int, default=512, help="Fingerprint size.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files.",
    )
    args = parser.parse_args()

    _ensure_uspto50k_csvs(args.data_root)
    raw_dir = os.path.join(args.data_root, "raw")

    for spec in SPLITS:
        csv_path = os.path.join(raw_dir, spec.csv_name)
        out_prod = os.path.join(args.out_dir, f"rxn_encoded_prod_uspto50k_{spec.name}.pt")
        out_react = (
            os.path.join(args.out_dir, "rxn_encoded_react_tensor.pt")
            if spec.name == "train"
            else os.path.join(args.out_dir, f"rxn_encoded_react_tensor_{spec.name}.pt")
        )

        if (not args.overwrite) and os.path.exists(out_prod) and os.path.exists(out_react):
            print(f"[skip] {spec.name}: {out_prod} / {out_react}")
            continue

        react_fps = []
        prod_fps = []
        for react_smi, prod_smi in tqdm(list(_iter_reactions(csv_path)), desc=f"Embedding {spec.name}"):
            react_fps.append(_morgan_fp_bits(react_smi, radius=args.radius, n_bits=args.n_bits))
            prod_fps.append(_morgan_fp_bits(prod_smi, radius=args.radius, n_bits=args.n_bits))

        react_arr = np.stack(react_fps, axis=0)
        prod_arr = np.stack(prod_fps, axis=0)

        _save_tensor(out_react, react_arr)
        _save_tensor(out_prod, prod_arr)
        print(f"[ok] {spec.name}: saved {out_react} and {out_prod}")


if __name__ == "__main__":
    main()

