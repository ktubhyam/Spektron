"""
Cross-Spectral Prediction Dataset for Spektron.

Wraps QM9SDataset to provide paired IR/Raman spectra for cross-modal
prediction: predict Raman from IR, or IR from Raman.

Only includes molecules that have BOTH IR and Raman spectra available.
Uses the same preprocessing pipeline (SG filter -> resample 2048 -> SNV)
and the same reproducible train/val/test splits as the base QM9SDataset.
"""
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


class CrossSpectralDataset(Dataset):
    """Dataset for cross-spectral prediction (IR <-> Raman).

    Each sample returns the source spectrum, target spectrum, and molecule
    index. The direction (ir->raman or raman->ir) is set at construction.

    Only includes molecules that have BOTH IR and Raman spectra in the
    preprocessed HDF5 file.

    Splits are constructed identically to QM9SDataset._create_split so
    that train/val/test molecule assignments are consistent across tasks.

    Attributes:
        direction: Prediction direction, either "ir2raman" or "raman2ir".
        indices: Array of HDF5 molecule indices for this split.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        direction: Literal["ir2raman", "raman2ir"] = "ir2raman",
        max_samples: Optional[int] = None,
        seed: int = 42,
        augment_source: bool = False,
    ) -> None:
        """Initialize the cross-spectral dataset.

        Args:
            h5_path: Path to qm9s_processed.h5 (from preprocess_qm9s_to_hdf5).
            split: One of "train", "val", or "test".
            direction: Prediction direction. "ir2raman" means the model
                receives an IR spectrum and must predict the Raman spectrum.
                "raman2ir" is the reverse.
            max_samples: Limit number of molecules (for debugging).
            seed: Random seed for train/val/test split (must match QM9SDataset).
            augment_source: Whether to apply spectral augmentation to the
                source spectrum during training. Target is never augmented.
        """
        import h5py

        self.h5_path = h5_path
        self.split = split
        self.direction = direction
        self.augment_source = augment_source and (split == "train")

        if direction == "ir2raman":
            self.source_key = "ir"
            self.target_key = "raman"
            self.source_domain = "IR"
            self.target_domain = "RAMAN"
            self.source_instrument_id = 1
            self.target_instrument_id = 2
        elif direction == "raman2ir":
            self.source_key = "raman"
            self.target_key = "ir"
            self.source_domain = "RAMAN"
            self.target_domain = "IR"
            self.source_instrument_id = 2
            self.target_instrument_id = 1
        else:
            raise ValueError(
                f"Invalid direction '{direction}'. "
                f"Must be 'ir2raman' or 'raman2ir'."
            )

        # Read metadata and filter to molecules with both modalities
        with h5py.File(h5_path, "r") as f:
            total = int(f.attrs["n_molecules"])
            has_ir = f["has_ir"][:]
            has_raman = f["has_raman"][:]

        # Only molecules that have BOTH IR and Raman spectra
        both_mask = has_ir & has_raman
        paired_indices = np.where(both_mask)[0]
        n_paired = len(paired_indices)
        log.info(
            f"CrossSpectral: {n_paired}/{total} molecules have both IR and Raman"
        )

        # Create split using the SAME logic as QM9SDataset._create_split
        # so molecule assignments are consistent across tasks.
        # The split is applied to ALL molecules first (same as QM9SDataset),
        # then filtered to paired-only.
        rng = np.random.RandomState(seed)
        all_indices = rng.permutation(total)

        n_test = min(12982, int(total * 0.10))
        n_val = min(5842, int(total * 0.045))

        if split == "train":
            split_indices = set(all_indices[: total - n_val - n_test].tolist())
        elif split == "val":
            split_indices = set(
                all_indices[total - n_val - n_test : total - n_test].tolist()
            )
        elif split == "test":
            split_indices = set(all_indices[total - n_test :].tolist())
        else:
            split_indices = set(all_indices.tolist())

        # Intersect with paired molecules
        self.indices = np.array(
            [idx for idx in paired_indices if idx in split_indices],
            dtype=np.int64,
        )

        if max_samples is not None:
            self.indices = self.indices[:max_samples]

        log.info(
            f"CrossSpectral split '{split}' ({direction}): "
            f"{len(self.indices)} paired molecules"
        )

        # Lazy augmentor
        self._augmentor = None
        if self.augment_source:
            from .qm9s import SpectralAugmentor

            self._augmentor = SpectralAugmentor()

        # HDF5 handle (lazy, fork-safe)
        self._h5 = None
        self._h5_pid = None

    def _get_h5(self):
        """Get HDF5 file handle (lazy open, fork-safe via PID check)."""
        import os
        import h5py

        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                try:
                    self._h5.close()
                except Exception:
                    pass
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            self._h5_pid = pid
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single cross-spectral sample.

        Returns:
            Dictionary with keys:
                - source_spectrum: (n_channels,) float32 tensor (input to model)
                - target_spectrum: (n_channels,) float32 tensor (prediction target)
                - molecule_idx: scalar long tensor (HDF5 row index)
                - smiles: str SMILES for the molecule
                - source_domain: str ("IR" or "RAMAN")
                - target_domain: str ("IR" or "RAMAN")
                - source_instrument_id: scalar long tensor
                - target_instrument_id: scalar long tensor
        """
        f = self._get_h5()
        real_idx = int(self.indices[idx])

        source_spectrum = f[self.source_key][real_idx]  # (n_channels,) float32
        target_spectrum = f[self.target_key][real_idx]  # (n_channels,) float32

        # Augment source only (never the target)
        if self._augmentor is not None:
            source_spectrum = self._augmentor(source_spectrum)

        smiles = f["smiles"][real_idx]
        if isinstance(smiles, bytes):
            smiles = smiles.decode("utf-8")

        return {
            "source_spectrum": torch.tensor(source_spectrum, dtype=torch.float32),
            "target_spectrum": torch.tensor(target_spectrum, dtype=torch.float32),
            "molecule_idx": torch.tensor(real_idx, dtype=torch.long),
            "smiles": smiles,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "source_instrument_id": torch.tensor(
                self.source_instrument_id, dtype=torch.long
            ),
            "target_instrument_id": torch.tensor(
                self.target_instrument_id, dtype=torch.long
            ),
        }

    def __del__(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def _cross_spectral_collate(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Custom collate function that handles string fields."""
    result: Dict = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result


def build_cross_spectral_loaders(
    h5_path: str,
    direction: Literal["ir2raman", "raman2ir"] = "ir2raman",
    batch_size: int = 64,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    seed: int = 42,
    augment_source: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for cross-spectral prediction.

    Args:
        h5_path: Path to qm9s_processed.h5.
        direction: "ir2raman" or "raman2ir".
        batch_size: Batch size for all splits.
        num_workers: Number of DataLoader workers.
        max_samples: Limit samples per split (for debugging).
        seed: Random seed for splitting.
        augment_source: Whether to augment the source spectrum during training.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    loaders = []
    for split in ["train", "val", "test"]:
        dataset = CrossSpectralDataset(
            h5_path=h5_path,
            split=split,
            direction=direction,
            max_samples=max_samples,
            seed=seed,
            augment_source=augment_source if split == "train" else False,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            collate_fn=_cross_spectral_collate,
            persistent_workers=(num_workers > 0),
        )
        loaders.append(loader)

    return tuple(loaders)
