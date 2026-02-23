"""
Training and evaluation script for the minimal CNN-Transformer-VIB
spectrum-to-structure identification model.

Usage:
    python experiments/train_minimal.py [--n_molecules N] [--epochs E] [--batch_size B]

Workflow:
    1. Load broadened IR + Raman CSVs from QM9S
    2. Find molecules with both IR and Raman spectra
    3. Create train/val/test splits (80/10/10)
    4. Train with: contrastive (InfoNCE) + classification + VIB KL loss
    5. Evaluate: top-1, top-5, top-10 retrieval accuracy

For quick testing on laptop CPU, use --n_molecules 5000 (default).
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_model import SpectrumIdentifier, InfoNCELoss, ClassificationLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Data Loading
# ============================================================

def load_qm9s_spectra(
    ir_path: str,
    raman_path: str,
    n_molecules: int = 5000,
    seed: int = 42,
) -> dict:
    """Load QM9S broadened IR and Raman spectra from CSVs.

    The CSVs have format:
        - Header row: ,500.0,501.0,...,4000.0  (3501 wavenumber columns)
        - Data rows: mol_idx,val1,val2,...,val3501

    Only molecules present in BOTH files are used.
    We load a random subset of n_molecules for speed.

    Args:
        ir_path: path to ir_broaden.csv
        raman_path: path to raman_broaden.csv
        n_molecules: how many molecules to load (for speed)
        seed: random seed for subset selection

    Returns:
        dict with keys: ir_spectra, raman_spectra, mol_indices, wavenumbers
    """
    logger.info(f"Loading QM9S spectra (requesting {n_molecules} molecules)...")

    # --- Strategy: read line by line to avoid loading 6GB+ into memory ---
    # First pass: collect molecule indices from both files

    logger.info("Scanning IR file for molecule indices...")
    ir_indices = set()
    with open(ir_path, "r") as f:
        header = f.readline()  # skip header
        for line in f:
            idx = int(line.split(",", 1)[0])
            ir_indices.add(idx)
    logger.info(f"  IR file contains {len(ir_indices)} molecules")

    logger.info("Scanning Raman file for molecule indices...")
    raman_indices = set()
    with open(raman_path, "r") as f:
        header_line = f.readline()  # skip header
        for line in f:
            idx = int(line.split(",", 1)[0])
            raman_indices.add(idx)
    logger.info(f"  Raman file contains {len(raman_indices)} molecules")

    # Find common molecules
    common = sorted(ir_indices & raman_indices)
    logger.info(f"  Molecules in both: {len(common)}")

    # Select random subset
    rng = np.random.RandomState(seed)
    if n_molecules < len(common):
        selected = set(rng.choice(common, size=n_molecules, replace=False))
    else:
        selected = set(common)
        n_molecules = len(common)

    logger.info(f"  Selected {len(selected)} molecules for loading")

    # Parse wavenumber grid from header
    wavenumber_strs = header.strip().split(",")[1:]  # skip first empty column
    wavenumbers = np.array([float(w) for w in wavenumber_strs], dtype=np.float32)
    n_points = len(wavenumbers)
    logger.info(f"  Spectrum length: {n_points} points ({wavenumbers[0]:.0f}-{wavenumbers[-1]:.0f} cm^-1)")

    # Second pass: load selected spectra
    # Build index -> position mapping for output arrays
    selected_sorted = sorted(selected)
    idx_to_pos = {idx: pos for pos, idx in enumerate(selected_sorted)}

    ir_spectra = np.zeros((len(selected_sorted), n_points), dtype=np.float32)
    raman_spectra = np.zeros((len(selected_sorted), n_points), dtype=np.float32)

    logger.info("Loading IR spectra...")
    loaded_ir = 0
    with open(ir_path, "r") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.split(",", 1)
            idx = int(parts[0])
            if idx in selected:
                pos = idx_to_pos[idx]
                values = np.fromstring(parts[1], sep=",", dtype=np.float32)
                ir_spectra[pos, :len(values)] = values[:n_points]
                loaded_ir += 1
    logger.info(f"  Loaded {loaded_ir} IR spectra")

    logger.info("Loading Raman spectra...")
    loaded_raman = 0
    with open(raman_path, "r") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.split(",", 1)
            idx = int(parts[0])
            if idx in selected:
                pos = idx_to_pos[idx]
                values = np.fromstring(parts[1], sep=",", dtype=np.float32)
                raman_spectra[pos, :len(values)] = values[:n_points]
                loaded_raman += 1
    logger.info(f"  Loaded {loaded_raman} Raman spectra")

    return {
        "ir_spectra": ir_spectra,       # (N, 3501)
        "raman_spectra": raman_spectra, # (N, 3501)
        "mol_indices": np.array(selected_sorted, dtype=np.int64),  # original QM9S IDs
        "wavenumbers": wavenumbers,     # (3501,)
    }


# ============================================================
# Dataset
# ============================================================

class QM9SDataset(Dataset):
    """Dataset for QM9S IR+Raman spectra with augmentation.

    Each sample is one molecule. Since each molecule has exactly one
    IR + one Raman spectrum, we create positive pairs via augmentation
    (Gaussian noise + random scaling + baseline shift).
    """

    def __init__(
        self,
        ir_spectra: np.ndarray,
        raman_spectra: np.ndarray,
        labels: np.ndarray,
        augment: bool = True,
        noise_std: float = 0.01,
        scale_range: tuple = (0.95, 1.05),
        shift_std: float = 0.005,
    ):
        """
        Args:
            ir_spectra: (N, L) IR spectra (already normalized)
            raman_spectra: (N, L) Raman spectra (already normalized)
            labels: (N,) integer labels (0..N-1 for classification)
            augment: whether to apply random augmentation
            noise_std: std of Gaussian noise (relative to spectrum std)
            scale_range: range for random multiplicative scaling
            shift_std: std for additive baseline shift
        """
        self.ir = torch.from_numpy(ir_spectra).float()
        self.raman = torch.from_numpy(raman_spectra).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_std = shift_std

    def __len__(self):
        return len(self.labels)

    def _augment_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to a spectrum."""
        if not self.augment:
            return spectrum

        aug = spectrum.clone()

        # Gaussian noise (relative to per-spectrum std)
        spec_std = aug.std() + 1e-8
        aug = aug + torch.randn_like(aug) * self.noise_std * spec_std

        # Random scaling
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        aug = aug * scale

        # Random baseline shift
        shift = torch.randn(1).item() * self.shift_std * spec_std
        aug = aug + shift

        return aug

    def __getitem__(self, idx):
        ir = self.ir[idx]
        raman = self.raman[idx]
        label = self.labels[idx]

        # Create augmented view for contrastive learning
        ir_aug = self._augment_spectrum(ir)
        raman_aug = self._augment_spectrum(raman)

        return {
            "ir": ir,
            "raman": raman,
            "ir_aug": ir_aug,
            "raman_aug": raman_aug,
            "label": label,
        }


# ============================================================
# Normalization
# ============================================================

def normalize_spectra(spectra: np.ndarray) -> np.ndarray:
    """Per-spectrum normalization: subtract mean, divide by std.

    This is a simple but effective normalization for spectral data.
    More sophisticated approaches (SNV, MSC) can be used later.
    """
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True) + 1e-8
    return (spectra - mean) / std


# ============================================================
# Evaluation: Retrieval Accuracy
# ============================================================

@torch.no_grad()
def compute_retrieval_accuracy(
    model: SpectrumIdentifier,
    dataloader: DataLoader,
    device: torch.device,
    ks: tuple = (1, 5, 10),
) -> dict:
    """Compute top-k retrieval accuracy via augmented-to-original matching.

    For each molecule, embed both the clean and augmented spectra.
    Then for each augmented embedding (query), find the nearest
    clean embedding (gallery) and check if it's the same molecule.

    This measures whether the model produces consistent, identity-
    preserving embeddings under realistic spectral perturbations.

    Args:
        model: trained model
        dataloader: DataLoader for the evaluation set (must have augmentation enabled)
        device: torch device
        ks: tuple of k values for top-k accuracy

    Returns:
        dict mapping k -> accuracy (0-1)
    """
    model.eval()

    clean_embeddings = []
    aug_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            ir = batch["ir"].to(device)
            raman = batch["raman"].to(device)
            ir_aug = batch["ir_aug"].to(device)
            raman_aug = batch["raman_aug"].to(device)

            # Clean embeddings (use mu directly, no sampling)
            out_clean = model(ir, raman)
            z_clean = F.normalize(out_clean["chem_mu"], dim=-1)
            clean_embeddings.append(z_clean.cpu())

            # Augmented embeddings
            out_aug = model(ir_aug, raman_aug)
            z_aug = F.normalize(out_aug["chem_mu"], dim=-1)
            aug_embeddings.append(z_aug.cpu())

    clean_embs = torch.cat(clean_embeddings, dim=0)  # (N, D)
    aug_embs = torch.cat(aug_embeddings, dim=0)       # (N, D)
    N = clean_embs.size(0)

    # Similarity: each augmented query vs all clean gallery entries
    sim_matrix = torch.mm(aug_embs, clean_embs.t())  # (N, N)

    # For each augmented query, rank clean gallery by similarity
    _, sorted_indices = sim_matrix.sort(dim=1, descending=True)

    # Ground truth: augmented[i] should match clean[i]
    gt_labels = torch.arange(N)

    results = {}
    for k in ks:
        top_k_indices = sorted_indices[:, :k]  # (N, k)
        correct = (top_k_indices == gt_labels.unsqueeze(1)).any(dim=1)
        accuracy = correct.float().mean().item()
        results[k] = accuracy

    return results


@torch.no_grad()
def compute_train_accuracy(
    model: SpectrumIdentifier,
    classifier,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute classification accuracy on the training set.

    Uses the classifier head to check if the model can correctly
    classify training molecules it has seen.
    """
    model.eval()
    classifier.eval()

    correct = 0
    total = 0

    for batch in dataloader:
        ir = batch["ir"].to(device)
        raman = batch["raman"].to(device)
        labels = batch["label"].to(device)

        out = model(ir, raman)
        z_chem = out["chem_mu"]  # Use mean (no sampling)

        # Get logits from classifier (prototypes is a Parameter, not Linear)
        z_norm = F.normalize(z_chem, dim=-1)
        w_norm = F.normalize(classifier.prototypes, dim=-1)
        logits = torch.mm(z_norm, w_norm.t()) / classifier.temperature

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_modality_ablation(
    model: SpectrumIdentifier,
    dataloader: DataLoader,
    device: torch.device,
    ks: tuple = (1, 5, 10),
) -> dict:
    """Test IR-only vs Raman-only vs IR+Raman retrieval.

    For IR-only: zero out Raman input.
    For Raman-only: zero out IR input.
    For IR+Raman: use both (standard).
    """
    model.eval()
    results = {}

    for mode in ["IR+Raman", "IR_only", "Raman_only"]:
        clean_embs_list = []
        aug_embs_list = []

        for batch in dataloader:
            ir = batch["ir"].to(device)
            raman = batch["raman"].to(device)
            ir_aug = batch["ir_aug"].to(device)
            raman_aug = batch["raman_aug"].to(device)

            if mode == "IR_only":
                raman = torch.zeros_like(raman)
                raman_aug = torch.zeros_like(raman_aug)
            elif mode == "Raman_only":
                ir = torch.zeros_like(ir)
                ir_aug = torch.zeros_like(ir_aug)

            out_clean = model(ir, raman)
            z_clean = F.normalize(out_clean["chem_mu"], dim=-1)
            clean_embs_list.append(z_clean.cpu())

            out_aug = model(ir_aug, raman_aug)
            z_aug = F.normalize(out_aug["chem_mu"], dim=-1)
            aug_embs_list.append(z_aug.cpu())

        clean = torch.cat(clean_embs_list, 0)
        aug = torch.cat(aug_embs_list, 0)
        N = clean.size(0)

        sim = torch.mm(aug, clean.t())
        _, sorted_idx = sim.sort(dim=1, descending=True)
        gt = torch.arange(N)

        mode_results = {}
        for k in ks:
            top_k = sorted_idx[:, :k]
            correct = (top_k == gt.unsqueeze(1)).any(dim=1)
            mode_results[k] = correct.float().mean().item()
        results[mode] = mode_results

    return results


@torch.no_grad()
def compute_retrieval_accuracy_crossmodal(
    model: SpectrumIdentifier,
    gallery_loader: DataLoader,
    query_loader: DataLoader,
    device: torch.device,
    ks: tuple = (1, 5, 10),
) -> dict:
    """Cross-set retrieval: query from test set, gallery from train set.

    This is a more realistic evaluation: given a test spectrum, find
    the most similar molecule in the training database.

    For this to work, the test set must contain molecules also present
    in the gallery (or we accept 0% accuracy for truly new molecules).
    Since we split by molecule, this tests generalization of embeddings.
    """
    model.eval()

    # Build gallery
    gallery_embs = []
    gallery_labels = []
    for batch in gallery_loader:
        ir = batch["ir"].to(device)
        raman = batch["raman"].to(device)
        z = model.get_embedding(ir, raman)
        gallery_embs.append(z.cpu())
        gallery_labels.append(batch["label"])
    gallery_embs = F.normalize(torch.cat(gallery_embs, 0), dim=-1)
    gallery_labels = torch.cat(gallery_labels, 0)

    # Query
    query_embs = []
    query_labels = []
    for batch in query_loader:
        ir = batch["ir"].to(device)
        raman = batch["raman"].to(device)
        z = model.get_embedding(ir, raman)
        query_embs.append(z.cpu())
        query_labels.append(batch["label"])
    query_embs = F.normalize(torch.cat(query_embs, 0), dim=-1)
    query_labels = torch.cat(query_labels, 0)

    # Similarity: (n_query, n_gallery)
    sim = torch.mm(query_embs, gallery_embs.t())
    _, sorted_idx = sim.sort(dim=1, descending=True)

    results = {}
    for k in ks:
        top_k = sorted_idx[:, :k]
        top_k_labels = gallery_labels[top_k]
        correct = (top_k_labels == query_labels.unsqueeze(1)).any(dim=1)
        results[k] = correct.float().mean().item()

    return results


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(
    model: SpectrumIdentifier,
    classifier: ClassificationLoss,
    contrastive_loss_fn: InfoNCELoss,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    beta_kl: float = 1e-4,
    lambda_cls: float = 1.0,
    lambda_contrast: float = 0.5,
) -> dict:
    """Train for one epoch.

    Loss = lambda_cls * L_classification
         + lambda_contrast * L_contrastive
         + beta_kl * L_kl

    Args:
        model: the SpectrumIdentifier
        classifier: ClassificationLoss module
        contrastive_loss_fn: InfoNCELoss
        optimizer: optimizer
        dataloader: training DataLoader
        device: torch device
        epoch: current epoch number
        beta_kl: weight for VIB KL divergence
        lambda_cls: weight for classification loss
        lambda_contrast: weight for contrastive loss

    Returns:
        dict with average losses
    """
    model.train()
    classifier.train()

    total_loss = 0.0
    total_cls = 0.0
    total_contrast = 0.0
    total_kl = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        ir = batch["ir"].to(device)
        raman = batch["raman"].to(device)
        ir_aug = batch["ir_aug"].to(device)
        raman_aug = batch["raman_aug"].to(device)
        labels = batch["label"].to(device)

        # Forward pass on original
        out = model(ir, raman)
        z_chem = out["z_chem"]

        # Forward pass on augmented (for contrastive loss)
        out_aug = model(ir_aug, raman_aug)
        z_chem_aug = out_aug["z_chem"]

        # Classification loss (prototype-based)
        loss_cls = classifier(z_chem, labels)

        # Contrastive loss (InfoNCE between original and augmented)
        loss_contrast = contrastive_loss_fn(z_chem, z_chem_aug)

        # KL divergence (VIB regularization)
        loss_kl = out["kl_total"] + out_aug["kl_total"]

        # Total loss
        loss = (
            lambda_cls * loss_cls
            + lambda_contrast * loss_contrast
            + beta_kl * loss_kl
        )

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_contrast += loss_contrast.item()
        total_kl += loss_kl.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"loss={loss.item():.4f} cls={loss_cls.item():.4f} "
                f"contrast={loss_contrast.item():.4f} kl={loss_kl.item():.2f}"
            )

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls / n_batches,
        "contrast_loss": total_contrast / n_batches,
        "kl_loss": total_kl / n_batches,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train minimal CNN-Transformer-VIB for spectrum identification"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "raw" / "qm9s"),
        help="Directory containing ir_broaden.csv and raman_broaden.csv",
    )
    parser.add_argument("--n_molecules", type=int, default=5000,
                        help="Number of molecules to use (default: 5000)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Embedding dimension (default: 128)")
    parser.add_argument("--z_chem_dim", type=int, default=64,
                        help="Chemistry latent dimension (default: 64)")
    parser.add_argument("--z_inst_dim", type=int, default=32,
                        help="Instrument latent dimension (default: 32)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of transformer layers (default: 2)")
    parser.add_argument("--beta_kl", type=float, default=1e-4,
                        help="KL loss weight (default: 1e-4)")
    parser.add_argument("--lambda_cls", type=float, default=1.0,
                        help="Classification loss weight (default: 1.0)")
    parser.add_argument("--lambda_contrast", type=float, default=0.5,
                        help="Contrastive loss weight (default: 0.5)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature (default: 0.07)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader num_workers (default: 0)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save checkpoints (default: experiments/checkpoints)")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.save_dir is None:
        args.save_dir = str(Path(__file__).resolve().parent / "checkpoints")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Load Data ----
    ir_path = os.path.join(args.data_dir, "ir_broaden.csv")
    raman_path = os.path.join(args.data_dir, "raman_broaden.csv")

    if not os.path.exists(ir_path):
        raise FileNotFoundError(f"IR CSV not found: {ir_path}")
    if not os.path.exists(raman_path):
        raise FileNotFoundError(f"Raman CSV not found: {raman_path}")

    data = load_qm9s_spectra(
        ir_path, raman_path,
        n_molecules=args.n_molecules,
        seed=args.seed,
    )

    ir_spectra = data["ir_spectra"]      # (N, 3501)
    raman_spectra = data["raman_spectra"]  # (N, 3501)
    mol_indices = data["mol_indices"]     # (N,) original QM9S molecule IDs
    N = len(mol_indices)

    logger.info(f"Loaded {N} molecules, spectrum length {ir_spectra.shape[1]}")

    # Normalize
    ir_spectra = normalize_spectra(ir_spectra)
    raman_spectra = normalize_spectra(raman_spectra)
    logger.info("Spectra normalized (zero-mean, unit-variance per sample)")

    # ---- Train/Val/Test Split ----
    # Assign labels 0..N-1 (each molecule is its own class)
    labels = np.arange(N, dtype=np.int64)

    # Shuffle and split 80/10/10
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(N)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    logger.info(f"Split: train={n_train}, val={n_val}, test={n_test}")

    # Re-label training molecules to 0..n_train-1 for the classifier
    # (the classifier only needs to classify training molecules)
    train_label_map = {orig: new for new, orig in enumerate(train_idx)}
    train_labels = np.array([train_label_map[i] for i in train_idx], dtype=np.int64)

    # For val/test: keep original indices (they won't be used for classification,
    # only for retrieval where we match against the training gallery)
    # But for within-set retrieval, we need unique labels
    val_labels = np.arange(len(val_idx), dtype=np.int64)
    test_labels = np.arange(len(test_idx), dtype=np.int64)

    # Create datasets
    train_dataset = QM9SDataset(
        ir_spectra[train_idx], raman_spectra[train_idx],
        train_labels, augment=True,
    )
    val_dataset = QM9SDataset(
        ir_spectra[val_idx], raman_spectra[val_idx],
        val_labels, augment=True,  # Need augmentation for aug-to-clean retrieval eval
    )
    test_dataset = QM9SDataset(
        ir_spectra[test_idx], raman_spectra[test_idx],
        test_labels, augment=True,  # Need augmentation for aug-to-clean retrieval eval
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # ---- Build Model ----
    model = SpectrumIdentifier(
        input_length=ir_spectra.shape[1],
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_layers,
        ff_dim=args.embed_dim * 2,
        z_chem_dim=args.z_chem_dim,
        z_inst_dim=args.z_inst_dim,
        dropout=0.1,
    ).to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"CNN patches per spectrum: {model.ir_tokenizer.n_patches}")

    # Classifier head (prototype-based)
    classifier = ClassificationLoss(
        embed_dim=args.z_chem_dim,
        n_classes=n_train,
        temperature=args.temperature,
    ).to(device)

    # Contrastive loss
    contrastive_fn = InfoNCELoss(temperature=args.temperature)

    # Optimizer
    all_params = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ---- Training ----
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)

    best_val_top1 = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            classifier=classifier,
            contrastive_loss_fn=contrastive_fn,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            beta_kl=args.beta_kl,
            lambda_cls=args.lambda_cls,
            lambda_contrast=args.lambda_contrast,
        )

        scheduler.step()

        elapsed = time.time() - t0

        # Log training metrics
        logger.info(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) -- "
            f"loss={train_metrics['loss']:.4f} "
            f"cls={train_metrics['cls_loss']:.4f} "
            f"contrast={train_metrics['contrast_loss']:.4f} "
            f"kl={train_metrics['kl_loss']:.2f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Evaluate every 5 epochs (or last epoch)
        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            # Training classification accuracy
            train_acc = compute_train_accuracy(
                model, classifier, train_loader, device
            )
            logger.info(f"  Train classification accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")

            # Augmented-to-clean retrieval on validation set
            val_acc = compute_retrieval_accuracy(
                model, val_loader, device, ks=(1, 5, 10)
            )
            logger.info(
                f"  Val retrieval (aug->clean): "
                f"top-1={val_acc[1]:.4f}  top-5={val_acc[5]:.4f}  top-10={val_acc[10]:.4f}"
            )

            # Save best model (or save first checkpoint unconditionally)
            if val_acc[1] > best_val_top1 or epoch == 1:
                best_val_top1 = max(best_val_top1, val_acc[1])
                best_epoch = epoch
                ckpt_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "classifier_state": classifier.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_top1": val_acc[1],
                    "train_acc": train_acc,
                    "args": vars(args),
                }, ckpt_path)
                logger.info(f"  Saved best model (top-1={best_val_top1:.4f}) to {ckpt_path}")

    # ---- Final Evaluation ----
    logger.info("=" * 60)
    logger.info("Final Evaluation")
    logger.info("=" * 60)

    # Load best model
    ckpt_path = os.path.join(args.save_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded best model from epoch {ckpt['epoch']}")

    # Training classification accuracy
    train_acc = compute_train_accuracy(model, classifier, train_loader, device)
    logger.info(f"Train classification accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")

    # Augmented-to-clean retrieval on test set
    test_acc = compute_retrieval_accuracy(
        model, test_loader, device, ks=(1, 5, 10, 20, 50)
    )
    logger.info("Test set retrieval (aug->clean):")
    for k, acc in sorted(test_acc.items()):
        logger.info(f"  top-{k}: {acc:.4f} ({acc*100:.2f}%)")

    # Modality ablation: IR-only vs Raman-only vs IR+Raman
    logger.info("\n--- Modality Ablation (test set) ---")
    modality_results = compute_modality_ablation(
        model, test_loader, device, ks=(1, 5, 10)
    )
    for mode, accs in modality_results.items():
        logger.info(f"  {mode:12s}: top-1={accs[1]:.4f}  top-5={accs[5]:.4f}  top-10={accs[10]:.4f}")

    # ---- Embedding Statistics ----
    logger.info("\n--- Embedding Statistics ---")
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch in test_loader:
            ir = batch["ir"].to(device)
            raman = batch["raman"].to(device)
            out = model(ir, raman)
            all_z.append(out["chem_mu"].cpu())
    all_z = torch.cat(all_z, dim=0)
    logger.info(f"z_chem shape: {all_z.shape}")
    logger.info(f"z_chem mean: {all_z.mean().item():.4f}")
    logger.info(f"z_chem std: {all_z.std().item():.4f}")
    logger.info(f"z_chem norm (avg): {all_z.norm(dim=-1).mean().item():.4f}")

    # Pairwise similarity distribution
    z_norm = F.normalize(all_z, dim=-1)
    sim = torch.mm(z_norm, z_norm.t())
    sim.fill_diagonal_(0)
    logger.info(f"Pairwise cosine sim: mean={sim.mean().item():.4f}, "
                f"std={sim.std().item():.4f}, "
                f"max={sim.max().item():.4f}, "
                f"min={sim.min().item():.4f}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Best validation top-1 accuracy: {best_val_top1:.4f} (epoch {best_epoch})")
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
