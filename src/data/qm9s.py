"""
QM9S Dataset Loader for Spektron.

QM9S: ~130K molecules with DFT-computed IR, Raman, and UV-Vis spectra.
Source: Figshare (DOI: 10.6084/m9.figshare.24235333.v3)
Paper: Wang, Zou et al., Nature Computational Science (2023)

Data files:
    - qm9s.pt: PyTorch Geometric format with SMILES + 3D coords (129,817 molecules)
    - ir_broaden.csv: Broadened IR spectra (103,991 × 3501, 500-4000 cm⁻¹)
    - raman_broaden.csv: Broadened Raman spectra (129,817 × 3501, 500-4000 cm⁻¹)

Note: IR CSV has fewer rows than Raman (103,991 vs 129,817).
We use the intersection (first 103,991) when both modalities are needed.

Standard splits (Vib2Mol): 110,992 train / 5,842 val / 12,982 test
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


class QM9SPreprocessor:
    """Preprocessing pipeline for QM9S spectra.

    Pipeline: raw 3501-pt spectrum → NaN check → SG filter → resample to 2048 → SNV normalize
    """

    def __init__(self, target_length: int = 2048,
                 sg_window: int = 7, sg_order: int = 3,
                 normalize: str = "snv"):
        self.target_length = target_length
        self.sg_window = sg_window
        self.sg_order = sg_order
        self.normalize = normalize

    def process(self, spectrum: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline for a single spectrum.

        Args:
            spectrum: (3501,) raw broadened spectrum

        Returns:
            processed: (target_length,) preprocessed spectrum
        """
        # 0. NaN/Inf check — replace with zeros
        if not np.all(np.isfinite(spectrum)):
            nan_count = np.sum(~np.isfinite(spectrum))
            log.warning(f"Found {nan_count} NaN/Inf values in spectrum, replacing with 0")
            spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Savitzky-Golay smoothing
        if self.sg_window > 0 and len(spectrum) >= self.sg_window:
            spectrum = savgol_filter(spectrum, self.sg_window, self.sg_order)

        # 2. Resample to target length
        if len(spectrum) != self.target_length:
            x_old = np.linspace(0, 1, len(spectrum))
            x_new = np.linspace(0, 1, self.target_length)
            f = interp1d(x_old, spectrum, kind='cubic')
            spectrum = f(x_new)

        # 3. Normalize
        if self.normalize == "snv":
            mean = spectrum.mean()
            std = spectrum.std()
            if std > 1e-10:
                spectrum = (spectrum - mean) / std
            else:
                spectrum = spectrum - mean
        elif self.normalize == "minmax":
            mn, mx = spectrum.min(), spectrum.max()
            if mx - mn > 1e-10:
                spectrum = (spectrum - mn) / (mx - mn)

        return spectrum.astype(np.float32)


# ============================================================
# HDF5 Preprocessing (run once, train from cache)
# ============================================================

def preprocess_qm9s_to_hdf5(data_dir: str, output_path: str = None,
                              target_length: int = 2048,
                              max_samples: int = None,
                              rgn_path: str = None) -> str:
    """Convert raw QM9S CSVs to an efficient HDF5 cache.

    Loads SMILES from qm9s.pt, spectra from CSVs, preprocesses, and saves
    to a single HDF5 file. Run this once; subsequent training loads from HDF5.

    Optionally includes R(G,N) symmetry metadata from pre-computed point
    group analysis (experiments/results/qm9_rgn_data.npz).

    Args:
        data_dir: Path to directory containing qm9s.pt, ir_broaden.csv, raman_broaden.csv
        output_path: Output HDF5 path (default: data_dir/qm9s_processed.h5)
        target_length: Resample to this length
        max_samples: Limit samples (for debugging)
        rgn_path: Path to qm9_rgn_data.npz (default: auto-detect)

    Returns:
        Path to the created HDF5 file
    """
    import h5py
    import pandas as pd

    data_dir = Path(data_dir)
    output_path = output_path or str(data_dir / "qm9s_processed.h5")

    # 1. Load SMILES from qm9s.pt
    pt_path = data_dir / "qm9s.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"qm9s.pt not found in {data_dir}")

    log.info("Loading qm9s.pt for SMILES...")
    pt_data = torch.load(pt_path, map_location='cpu', weights_only=False)
    all_smiles = [mol.smile for mol in pt_data]
    n_total_pt = len(all_smiles)
    log.info(f"Loaded {n_total_pt} SMILES from qm9s.pt")

    # 2. Load IR spectra
    ir_path = data_dir / "ir_broaden.csv"
    raman_path = data_dir / "raman_broaden.csv"

    ir_data = None
    raman_data = None
    n_ir = 0
    n_raman = 0

    if ir_path.exists():
        log.info("Loading IR spectra from CSV...")
        df_ir = pd.read_csv(ir_path)
        ir_data = df_ir.iloc[:, 1:].values.astype(np.float32)
        n_ir = ir_data.shape[0]
        log.info(f"IR data: {ir_data.shape}")
        del df_ir

    if raman_path.exists():
        log.info("Loading Raman spectra from CSV...")
        df_raman = pd.read_csv(raman_path)
        raman_data = df_raman.iloc[:, 1:].values.astype(np.float32)
        n_raman = raman_data.shape[0]
        log.info(f"Raman data: {raman_data.shape}")
        del df_raman

    # 3. Determine intersection size
    # IR has fewer rows (103,991) than Raman (129,817) and qm9s.pt (129,817)
    # CSV row i corresponds to qm9s.pt molecule i (same ordering)
    n_intersection = min(n_ir or n_total_pt, n_raman or n_total_pt, n_total_pt)
    if max_samples:
        n_intersection = min(n_intersection, max_samples)

    log.info(f"Using intersection of {n_intersection} molecules "
             f"(IR={n_ir}, Raman={n_raman}, PT={n_total_pt})")

    # 4. Preprocess and save
    preprocessor = QM9SPreprocessor(target_length=target_length)

    with h5py.File(output_path, 'w') as f:
        # Datasets
        f.create_dataset('ir', shape=(n_intersection, target_length),
                         dtype=np.float32, chunks=(64, target_length))
        f.create_dataset('raman', shape=(n_intersection, target_length),
                         dtype=np.float32, chunks=(64, target_length))
        # Store SMILES as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('smiles', shape=(n_intersection,), dtype=dt)

        # Track which modalities are available per molecule
        has_ir = np.zeros(n_intersection, dtype=bool)
        has_raman = np.zeros(n_intersection, dtype=bool)

        for i in range(n_intersection):
            if i % 10000 == 0:
                log.info(f"Preprocessing {i}/{n_intersection}...")

            f['smiles'][i] = all_smiles[i]

            if ir_data is not None and i < n_ir:
                f['ir'][i] = preprocessor.process(ir_data[i])
                has_ir[i] = True

            if raman_data is not None and i < n_raman:
                f['raman'][i] = preprocessor.process(raman_data[i])
                has_raman[i] = True

        f.create_dataset('has_ir', data=has_ir)
        f.create_dataset('has_raman', data=has_raman)

        # R(G,N) symmetry metadata (if available)
        if rgn_path is None:
            # Auto-detect from standard location
            rgn_candidates = [
                Path("experiments/results/qm9_rgn_data.npz"),
                data_dir / "qm9_rgn_data.npz",
            ]
            for c in rgn_candidates:
                if c.exists():
                    rgn_path = str(c)
                    break

        if rgn_path and Path(rgn_path).exists():
            log.info(f"Loading R(G,N) metadata from {rgn_path}...")
            rgn = np.load(rgn_path, allow_pickle=True)
            # R(G,N) data has 130,831 entries; slice to intersection
            f.create_dataset('R_gn', data=rgn['R'][:n_intersection].astype(np.float32))
            f.create_dataset('n_atoms', data=rgn['N'][:n_intersection].astype(np.int32))
            f.create_dataset('n_vib_modes', data=rgn['d'][:n_intersection].astype(np.int32))
            f.create_dataset('n_silent', data=rgn['n_silent'][:n_intersection].astype(np.int32))
            f.create_dataset('is_centrosymmetric',
                             data=rgn['is_centrosymmetric'][:n_intersection])
            f.attrs['has_rgn'] = True
            log.info(f"  R(G,N) mean={rgn['R'][:n_intersection].mean():.4f}, "
                     f"centrosymmetric={rgn['is_centrosymmetric'][:n_intersection].sum()}")
        else:
            f.attrs['has_rgn'] = False
            log.info("No R(G,N) data found, skipping symmetry metadata")

        # Metadata
        f.attrs['n_molecules'] = n_intersection
        f.attrs['target_length'] = target_length
        f.attrs['source_spectral_points'] = 3501
        f.attrs['wavenumber_range'] = '500-4000 cm^-1'

    log.info(f"Saved preprocessed QM9S to {output_path}")
    log.info(f"  Molecules with IR: {has_ir.sum()}, Raman: {has_raman.sum()}")
    return output_path


# ============================================================
# Dataset classes (load from HDF5)
# ============================================================

class QM9SDataset(Dataset):
    """QM9S dataset loading from preprocessed HDF5.

    Each sample alternates between IR and Raman modalities.
    Length = n_molecules * n_modalities.
    """

    def __init__(self, h5_path: str, split: str = "train",
                 modalities: List[str] = None,
                 max_samples: int = None,
                 seed: int = 42):
        """
        Args:
            h5_path: Path to qm9s_processed.h5
            split: "train", "val", or "test"
            modalities: List of modalities (default: ["ir", "raman"])
            max_samples: Limit number of molecules (for debugging)
            seed: Random seed for train/val/test split
        """
        import h5py

        self.h5_path = h5_path
        self.split = split
        self.modalities = modalities or ["ir", "raman"]

        # Read metadata and create split
        with h5py.File(h5_path, 'r') as f:
            total = f.attrs['n_molecules']
            self._has_ir = f['has_ir'][:]
            self._has_raman = f['has_raman'][:]
            self._has_rgn = f.attrs.get('has_rgn', False)
            if self._has_rgn:
                self._R_gn = f['R_gn'][:]
            else:
                self._R_gn = None

        self._create_split(total, seed, max_samples)

        # HDF5 file handle (opened lazily, one per worker via PID check)
        self._h5 = None
        self._h5_pid = None

    def _create_split(self, total: int, seed: int, max_samples: Optional[int]):
        """Create reproducible train/val/test split (85.5/4.5/10%)."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(total)

        n_test = min(12982, int(total * 0.10))
        n_val = min(5842, int(total * 0.045))

        if self.split == "train":
            self.indices = indices[:total - n_val - n_test]
        elif self.split == "val":
            self.indices = indices[total - n_val - n_test:total - n_test]
        elif self.split == "test":
            self.indices = indices[total - n_test:]
        else:
            self.indices = indices

        if max_samples is not None:
            self.indices = self.indices[:max_samples]

        log.info(f"QM9S split '{self.split}': {len(self.indices)} molecules")

    def _get_h5(self):
        """Get HDF5 file handle (lazy, fork-safe via PID check)."""
        import os
        import h5py
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
            self._h5 = h5py.File(self.h5_path, 'r', swmr=True)
            self._h5_pid = pid
        return self._h5

    def __len__(self) -> int:
        return len(self.indices) * len(self.modalities)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self._get_h5()

        # Map flat index to (molecule, modality)
        mol_idx = idx // len(self.modalities)
        mod_idx = idx % len(self.modalities)
        real_idx = int(self.indices[mol_idx])
        modality = self.modalities[mod_idx]

        # Get spectrum
        if modality == "ir" and self._has_ir[real_idx]:
            spectrum = f['ir'][real_idx]
        elif modality == "raman" and self._has_raman[real_idx]:
            spectrum = f['raman'][real_idx]
        else:
            # Fallback: use other modality if available, else zeros
            if modality == "ir" and self._has_raman[real_idx]:
                spectrum = f['raman'][real_idx]
                modality = "raman"
            elif modality == "raman" and self._has_ir[real_idx]:
                spectrum = f['ir'][real_idx]
                modality = "ir"
            else:
                spectrum = np.zeros(f['ir'].shape[1], dtype=np.float32)

        smiles = f['smiles'][real_idx]
        if isinstance(smiles, bytes):
            smiles = smiles.decode('utf-8')

        domain_map = {"ir": 1, "raman": 2, "nir": 0}

        result = {
            "spectrum": torch.tensor(spectrum, dtype=torch.float32),
            "smiles": smiles,
            "modality": modality,
            "domain": modality.upper(),
            "instrument_id": torch.tensor(domain_map.get(modality, 3), dtype=torch.long),
            "sample_id": torch.tensor(real_idx, dtype=torch.long),
        }

        # Add R(G,N) symmetry metadata if available
        if self._R_gn is not None:
            result["R_gn"] = torch.tensor(self._R_gn[real_idx], dtype=torch.float32)

        return result

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


class QM9SPairedDataset(Dataset):
    """QM9S dataset returning paired IR+Raman spectra for the same molecule.

    Used for contrastive learning (same molecule, different modality → similar z_chem)
    and modal complementarity experiments (E2).

    Only includes molecules that have BOTH IR and Raman spectra.
    """

    def __init__(self, h5_path: str, split: str = "train",
                 max_samples: int = None, seed: int = 42):
        import h5py

        self.h5_path = h5_path
        self.split = split

        with h5py.File(h5_path, 'r') as f:
            total = f.attrs['n_molecules']
            has_ir = f['has_ir'][:]
            has_raman = f['has_raman'][:]

        # Only molecules with both modalities
        both_mask = has_ir & has_raman
        all_paired_indices = np.where(both_mask)[0]

        # Split
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(all_paired_indices))
        n_test = min(12982, int(len(all_paired_indices) * 0.10))
        n_val = min(5842, int(len(all_paired_indices) * 0.045))

        if split == "train":
            split_perm = perm[:len(perm) - n_val - n_test]
        elif split == "val":
            split_perm = perm[len(perm) - n_val - n_test:len(perm) - n_test]
        elif split == "test":
            split_perm = perm[len(perm) - n_test:]
        else:
            split_perm = perm

        self.indices = all_paired_indices[split_perm]
        if max_samples:
            self.indices = self.indices[:max_samples]

        log.info(f"QM9S paired split '{split}': {len(self.indices)} molecules (both IR+Raman)")
        self._h5 = None
        self._h5_pid = None

    def _get_h5(self):
        """Get HDF5 file handle (lazy, fork-safe via PID check)."""
        import os
        import h5py
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
            self._h5 = h5py.File(self.h5_path, 'r', swmr=True)
            self._h5_pid = pid
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self._get_h5()
        real_idx = int(self.indices[idx])

        ir_spectrum = f['ir'][real_idx]
        raman_spectrum = f['raman'][real_idx]
        smiles = f['smiles'][real_idx]
        if isinstance(smiles, bytes):
            smiles = smiles.decode('utf-8')

        return {
            "ir_spectrum": torch.tensor(ir_spectrum, dtype=torch.float32),
            "raman_spectrum": torch.tensor(raman_spectrum, dtype=torch.float32),
            "smiles": smiles,
            "sample_id": torch.tensor(real_idx, dtype=torch.long),
            "domain": ["IR", "RAMAN"],
            "instrument_id": torch.tensor([1, 2], dtype=torch.long),
        }

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


# ============================================================
# DataLoader builders
# ============================================================

def build_qm9s_loaders(h5_path: str, batch_size: int = 64,
                        num_workers: int = 2,
                        modalities: List[str] = None,
                        max_samples: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders for QM9S.

    Args:
        h5_path: Path to qm9s_processed.h5 (run preprocess_qm9s_to_hdf5 first)
        batch_size: Batch size
        num_workers: Data loading workers
        modalities: Which modalities to include
        max_samples: Limit samples per split (for debugging)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    def _collate(batch):
        """Custom collate that handles string fields."""
        result = {}
        for key in batch[0]:
            vals = [b[key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals)
            else:
                result[key] = vals
        return result

    loaders = []
    for split in ["train", "val", "test"]:
        dataset = QM9SDataset(
            h5_path, split=split, modalities=modalities,
            max_samples=max_samples,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            collate_fn=_collate,
            persistent_workers=(num_workers > 0),
        )
        loaders.append(loader)

    return tuple(loaders)
