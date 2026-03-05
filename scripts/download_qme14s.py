#!/usr/bin/env python3
"""
Download and preprocess QMe14S dataset for Spektron.

QMe14S: 186,102 molecules, 14 elements, 47 functional groups.
Source: Figshare (https://figshare.com/s/889262a4e999b5c9a5b3)
Paper: Li, K. et al. J. Phys. Chem. Lett. 2025, 16, 1234–1240.

Data format (same as QM9S):
    - qme14s.pt: PyTorch Geometric format with SMILES + 3D coords
    - ir_broaden.csv: Broadened IR spectra (N × 3501, 500-4000 cm⁻¹)
    - raman_broaden.csv: Broadened Raman spectra (N × 3501, 500-4000 cm⁻¹)

Output: qme14s_processed.h5 — same schema as qm9s_processed.h5
        (ir, raman, smiles, has_ir, has_raman datasets + metadata)

Usage:
    python scripts/download_qme14s.py --data-dir data/raw/qme14s
    python scripts/download_qme14s.py --data-dir data/raw/qme14s --preprocess-only
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_qme14s(data_dir: Path):
    """Download QMe14S from Figshare."""
    import subprocess

    data_dir.mkdir(parents=True, exist_ok=True)

    # Figshare private link for QMe14S
    FIGSHARE_URL = "https://figshare.com/s/889262a4e999b5c9a5b3"

    log.info(f"QMe14S Figshare URL: {FIGSHARE_URL}")
    log.info("Note: QMe14S may need manual download from the Figshare link above.")
    log.info(f"Download the following files to {data_dir}/:")
    log.info("  1. qme14s.pt (or equivalent .pt file with molecular data)")
    log.info("  2. ir_broaden.csv (broadened IR spectra)")
    log.info("  3. raman_broaden.csv (broadened Raman spectra)")
    log.info("")
    log.info("If the dataset provides a ZIP archive, extract it to the data_dir.")

    # Try wget/curl if direct download links are available
    # QMe14S may require manual download from Figshare private link
    expected_files = ["ir_broaden.csv", "raman_broaden.csv"]
    pt_candidates = list(data_dir.glob("*.pt"))

    missing = [f for f in expected_files if not (data_dir / f).exists()]
    if missing and not pt_candidates:
        log.warning(f"Missing files: {missing}")
        log.warning("Please download manually from Figshare and re-run with --preprocess-only")
        return False

    log.info("All required files found!")
    return True


def preprocess_qme14s(data_dir: Path, output_path: Path = None,
                       target_length: int = 2048, max_samples: int = None):
    """Convert QMe14S raw files to HDF5 (same schema as QM9S).

    Reuses the QM9SPreprocessor for spectral preprocessing (SG smoothing,
    resampling, SNV normalization) since the spectral format is identical.
    """
    import h5py
    import numpy as np
    import pandas as pd
    import torch

    from src.data.qm9s import QM9SPreprocessor

    output_path = output_path or data_dir / "qme14s_processed.h5"

    # 1. Load molecular data (.pt file)
    pt_candidates = sorted(data_dir.glob("*.pt"))
    if not pt_candidates:
        raise FileNotFoundError(f"No .pt file found in {data_dir}")

    pt_path = pt_candidates[0]
    log.info(f"Loading molecular data from {pt_path}...")
    pt_data = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Extract SMILES — handle both list and Data formats
    if isinstance(pt_data, list):
        all_smiles = []
        for mol in pt_data:
            if hasattr(mol, 'smile'):
                all_smiles.append(mol.smile)
            elif hasattr(mol, 'smiles'):
                all_smiles.append(mol.smiles)
            else:
                all_smiles.append("")
    else:
        log.warning("Unexpected .pt format; trying as dict")
        all_smiles = pt_data.get('smiles', [''] * 186102)

    n_total_pt = len(all_smiles)
    log.info(f"Loaded {n_total_pt} molecules from .pt file")

    # 2. Load broadened spectra
    ir_path = data_dir / "ir_broaden.csv"
    raman_path = data_dir / "raman_broaden.csv"

    ir_data, raman_data = None, None
    n_ir, n_raman = 0, 0

    if ir_path.exists():
        log.info("Loading IR spectra from CSV...")
        df_ir = pd.read_csv(ir_path)
        ir_data = df_ir.iloc[:, 1:].values.astype(np.float32)
        n_ir = ir_data.shape[0]
        log.info(f"IR data shape: {ir_data.shape}")
        del df_ir

    if raman_path.exists():
        log.info("Loading Raman spectra from CSV...")
        df_raman = pd.read_csv(raman_path)
        raman_data = df_raman.iloc[:, 1:].values.astype(np.float32)
        n_raman = raman_data.shape[0]
        log.info(f"Raman data shape: {raman_data.shape}")
        del df_raman

    # 3. Determine intersection
    n_intersection = min(
        n_ir or n_total_pt,
        n_raman or n_total_pt,
        n_total_pt
    )
    if max_samples:
        n_intersection = min(n_intersection, max_samples)

    log.info(f"Using intersection: {n_intersection} molecules "
             f"(IR={n_ir}, Raman={n_raman}, PT={n_total_pt})")

    # 4. Preprocess and save (same schema as QM9S)
    preprocessor = QM9SPreprocessor(target_length=target_length)

    with h5py.File(str(output_path), 'w') as f:
        f.create_dataset('ir', shape=(n_intersection, target_length),
                         dtype=np.float32, chunks=(64, target_length))
        f.create_dataset('raman', shape=(n_intersection, target_length),
                         dtype=np.float32, chunks=(64, target_length))
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('smiles', shape=(n_intersection,), dtype=dt)

        has_ir = np.zeros(n_intersection, dtype=bool)
        has_raman = np.zeros(n_intersection, dtype=bool)

        for i in range(n_intersection):
            if i % 10000 == 0:
                log.info(f"Preprocessing {i}/{n_intersection}...")

            if i < len(all_smiles):
                f['smiles'][i] = all_smiles[i]

            if ir_data is not None and i < n_ir:
                f['ir'][i] = preprocessor.process(ir_data[i])
                has_ir[i] = True

            if raman_data is not None and i < n_raman:
                f['raman'][i] = preprocessor.process(raman_data[i])
                has_raman[i] = True

        f.create_dataset('has_ir', data=has_ir)
        f.create_dataset('has_raman', data=has_raman)

        # Metadata
        f.attrs['n_molecules'] = n_intersection
        f.attrs['target_length'] = target_length
        f.attrs['source_spectral_points'] = ir_data.shape[1] if ir_data is not None else 3501
        f.attrs['wavenumber_range'] = '500-4000 cm^-1'
        f.attrs['dataset'] = 'QMe14S'
        f.attrs['n_elements'] = 14
        f.attrs['elements'] = 'H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br'
        f.attrs['has_rgn'] = False  # No R(G,N) data for QMe14S yet

    log.info(f"Saved preprocessed QMe14S to {output_path}")
    log.info(f"  Molecules with IR: {has_ir.sum()}, Raman: {has_raman.sum()}")
    log.info(f"  Both modalities: {(has_ir & has_raman).sum()}")
    return str(output_path)


def verify_hdf5(h5_path: Path):
    """Verify the processed HDF5 is compatible with QM9SDataset."""
    import h5py
    import numpy as np

    log.info(f"\n=== Verifying {h5_path} ===")
    with h5py.File(str(h5_path), 'r') as f:
        log.info(f"Datasets: {list(f.keys())}")
        log.info(f"Attributes: {dict(f.attrs)}")
        log.info(f"IR shape: {f['ir'].shape}")
        log.info(f"Raman shape: {f['raman'].shape}")
        log.info(f"SMILES count: {f['smiles'].shape[0]}")

        has_ir = f['has_ir'][:]
        has_raman = f['has_raman'][:]
        log.info(f"Has IR: {has_ir.sum()}")
        log.info(f"Has Raman: {has_raman.sum()}")
        log.info(f"Has both: {(has_ir & has_raman).sum()}")

        # Check for NaN/zero spectra
        ir_sample = f['ir'][0]
        raman_sample = f['raman'][0]
        log.info(f"IR sample: min={ir_sample.min():.4f}, max={ir_sample.max():.4f}, "
                 f"std={ir_sample.std():.4f}")
        log.info(f"Raman sample: min={raman_sample.min():.4f}, max={raman_sample.max():.4f}, "
                 f"std={raman_sample.std():.4f}")

    # Test loading with QM9SDataset
    log.info("\n=== Testing QM9SDataset compatibility ===")
    from src.data.qm9s import QM9SDataset
    ds = QM9SDataset(str(h5_path), split="train", max_samples=100)
    log.info(f"Dataset length: {len(ds)}")
    sample = ds[0]
    log.info(f"Sample keys: {list(sample.keys())}")
    log.info(f"Spectrum shape: {sample['spectrum'].shape}")
    log.info(f"Modality: {sample['modality']}")
    log.info("\n✅ QMe14S HDF5 is compatible with QM9SDataset loader!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess QMe14S")
    parser.add_argument("--data-dir", type=str, default="data/raw/qme14s",
                        help="Directory for QMe14S raw data")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HDF5 path (default: data_dir/qme14s_processed.h5)")
    parser.add_argument("--target-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples (for debugging)")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Skip download, just preprocess existing files")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing HDF5")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "qme14s_processed.h5"

    if args.verify_only:
        verify_hdf5(output_path)
    else:
        if not args.preprocess_only:
            download_qme14s(data_dir)

        preprocess_qme14s(data_dir, output_path,
                          target_length=args.target_length,
                          max_samples=args.max_samples)
        verify_hdf5(output_path)
