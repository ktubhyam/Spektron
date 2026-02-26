#!/usr/bin/env python3
"""
Spektron: Run Classical Baselines + Spektron Fine-tuning Test

Tasks:
1. Run PLS, PDS, SBC, DS on corn m5->mp6 transfer (moisture)
2. Run minimal Spektron fine-tuning (random init, no pretrain)
3. Save results and print comparison table

Usage:
    python scripts/run_baselines.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

from src.evaluation.baselines import (
    PLSCalibration, PDS, SBC, DS,
    compute_metrics, run_baseline_comparison, print_results_table
)
from src.data.datasets import SpectralPreprocessor
from src.config import SpectralFMConfig
from src.models.spectral_fm import SpectralFM
from src.models.heads import RegressionHead


def load_corn_data(data_dir: str, n_transfer: int = 30, seed: int = 42):
    """Load corn dataset for m5→mp6 transfer experiment.

    Args:
        data_dir: Root data directory
        n_transfer: Number of transfer samples
        seed: Random seed for reproducibility

    Returns:
        dict with train/test splits for source/target instruments
    """
    corn_dir = Path(data_dir) / "processed" / "corn"

    # Load raw data
    m5_spectra = np.load(corn_dir / "m5_spectra.npy")      # (80, 700)
    mp6_spectra = np.load(corn_dir / "mp6_spectra.npy")    # (80, 700)
    properties = np.load(corn_dir / "properties.npy")       # (80, 4)
    wavelengths = np.load(corn_dir / "wavelengths.npy")    # (700,)

    # Moisture is column 0
    moisture = properties[:, 0]

    # Create train/test split
    rng = np.random.RandomState(seed)
    n_total = len(moisture)
    indices = rng.permutation(n_total)

    # Use n_transfer samples for training, rest for testing
    train_idx = indices[:n_transfer]
    test_idx = indices[n_transfer:]

    return {
        "X_source_train": m5_spectra[train_idx],   # m5 = source
        "X_target_train": mp6_spectra[train_idx],  # mp6 = target
        "X_source_test": m5_spectra[test_idx],
        "X_target_test": mp6_spectra[test_idx],
        "y_train": moisture[train_idx],
        "y_test": moisture[test_idx],
        "wavelengths": wavelengths,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def run_spectral_fm_finetune(data: dict, n_epochs: int = 30, lr: float = 1e-3,
                              device: str = "cpu") -> dict:
    """Run minimal Spektron fine-tuning test (random init, no pretrain).

    This validates the training loop works end-to-end.
    """
    print("\n--- Running Spektron Fine-tuning Test ---")

    # Preprocess spectra to 2048 channels
    preprocessor = SpectralPreprocessor(target_length=2048)

    X_train = np.stack([
        preprocessor.process(s, data["wavelengths"])["normalized"]
        for s in data["X_target_train"]
    ])
    X_test = np.stack([
        preprocessor.process(s, data["wavelengths"])["normalized"]
        for s in data["X_target_test"]
    ])

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(data["y_train"], dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(data["y_test"], dtype=torch.float32).to(device)

    # Create model (random init)
    cfg = SpectralFMConfig()
    model = SpectralFM(cfg).to(device)
    regression_head = RegressionHead(d_input=cfg.vib.z_chem_dim, n_targets=1).to(device)

    # Optimizer
    params = list(model.parameters()) + list(regression_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Training loop
    model.train()
    regression_head.train()

    batch_size = min(16, len(X_train_t))
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X_train_t[idx]
            y_batch = y_train_t[idx]

            optimizer.zero_grad()

            # Forward
            enc_out = model.encode(x_batch, domain="NIR")
            z_chem = enc_out["z_chem"]
            pred = regression_head(z_chem).squeeze(-1)

            # Loss
            loss = torch.nn.functional.mse_loss(pred, y_batch)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss = {avg_loss:.4f}")

    # Evaluate
    model.eval()
    regression_head.eval()

    with torch.no_grad():
        enc_out = model.encode(X_test_t, domain="NIR")
        z_chem = enc_out["z_chem"]
        y_pred = regression_head(z_chem).squeeze(-1).cpu().numpy()

    y_test_np = y_test_t.cpu().numpy()
    metrics = compute_metrics(y_test_np, y_pred)

    print(f"  Final: R² = {metrics['r2']:.4f}, RMSEP = {metrics['rmsep']:.4f}")

    return metrics


def main():
    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    experiments_dir = project_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\n=== Loading Corn Dataset ===")
    n_transfer = 30
    data = load_corn_data(data_dir, n_transfer=n_transfer, seed=42)
    print(f"  Training samples (transfer): {data['n_train']}")
    print(f"  Test samples: {data['n_test']}")
    print(f"  Spectral channels: {data['X_source_train'].shape[1]}")
    print(f"  Moisture range: [{data['y_train'].min():.2f}, {data['y_train'].max():.2f}]")

    # Run classical baselines
    print("\n=== Running Classical Baselines (m5 → mp6) ===")
    baseline_results = run_baseline_comparison(
        X_source_train=data["X_source_train"],
        X_target_train=data["X_target_train"],
        X_source_test=data["X_source_test"],
        X_target_test=data["X_target_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
    )

    # Print baseline results
    print_results_table(baseline_results, "Classical Calibration Transfer Methods")

    # Save baseline results
    baseline_output = {
        "experiment": "corn_m5_to_mp6_moisture",
        "n_transfer": n_transfer,
        "n_test": data["n_test"],
        "timestamp": datetime.now().isoformat(),
        "results": baseline_results,
    }

    with open(experiments_dir / "baselines_corn.json", "w") as f:
        json.dump(baseline_output, f, indent=2)
    print(f"\nBaseline results saved to experiments/baselines_corn.json")

    # Run Spektron fine-tuning test
    spectral_fm_metrics = run_spectral_fm_finetune(data, n_epochs=30, device=device)

    # Save fine-tuning results
    finetune_output = {
        "experiment": "spectral_fm_finetune_random_init",
        "dataset": "corn_m5_to_mp6_moisture",
        "n_transfer": n_transfer,
        "n_test": data["n_test"],
        "pretrained": False,
        "n_epochs": 30,
        "timestamp": datetime.now().isoformat(),
        "results": spectral_fm_metrics,
    }

    with open(experiments_dir / "finetune_test.json", "w") as f:
        json.dump(finetune_output, f, indent=2)
    print(f"Fine-tuning results saved to experiments/finetune_test.json")

    # Print comparison table
    print("\n" + "=" * 70)
    print("SUMMARY: All Methods Comparison (Corn m5 → mp6, Moisture)")
    print("=" * 70)
    print(f"{'Method':<20} {'R²':>10} {'RMSEP':>10} {'RPD':>8}")
    print("-" * 70)

    for method, metrics in baseline_results.items():
        print(f"{method:<20} {metrics['r2']:>10.4f} {metrics['rmsep']:>10.4f} {metrics['rpd']:>8.2f}")

    print(f"{'Spektron (no PT)':<20} {spectral_fm_metrics['r2']:>10.4f} "
          f"{spectral_fm_metrics['rmsep']:>10.4f} {spectral_fm_metrics['rpd']:>8.2f}")
    print("-" * 70)

    print("\n✓ All experiments completed successfully!")


if __name__ == "__main__":
    main()
