#!/usr/bin/env python3
"""
Spektron: Run Fine-tuning Validation Test

Validates the Spektron training loop works end-to-end.
Uses random initialization (no pretraining) to test the pipeline.

Usage:
    python scripts/run_finetune_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

from src.evaluation.baselines import compute_metrics
from src.data.datasets import SpectralPreprocessor
from src.config import SpectralFMConfig, get_light_config
from src.models.spectral_fm import SpectralFM
from src.models.heads import RegressionHead


def load_corn_data(data_dir: str, n_transfer: int = 30, seed: int = 42):
    """Load corn dataset for m5→mp6 transfer experiment."""
    corn_dir = Path(data_dir) / "processed" / "corn"

    m5_spectra = np.load(corn_dir / "m5_spectra.npy")
    mp6_spectra = np.load(corn_dir / "mp6_spectra.npy")
    properties = np.load(corn_dir / "properties.npy")
    wavelengths = np.load(corn_dir / "wavelengths.npy")

    moisture = properties[:, 0]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(moisture))
    train_idx = indices[:n_transfer]
    test_idx = indices[n_transfer:]

    return {
        "X_target_train": mp6_spectra[train_idx],
        "X_target_test": mp6_spectra[test_idx],
        "y_train": moisture[train_idx],
        "y_test": moisture[test_idx],
        "wavelengths": wavelengths,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def run_spectral_fm_finetune(data: dict, n_epochs: int = 20, lr: float = 1e-3,
                              device: str = "cpu") -> dict:
    """Run minimal Spektron fine-tuning test."""
    print("\n--- Running Spektron Fine-tuning Test ---")

    # Preprocess spectra to 2048 channels
    preprocessor = SpectralPreprocessor(target_length=2048)

    print("  Preprocessing spectra...")
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

    # Create model with lightweight config for faster CPU testing
    print("  Creating model (lightweight config)...")
    cfg = get_light_config()
    model = SpectralFM(cfg).to(device)
    regression_head = RegressionHead(d_input=cfg.vib.z_chem_dim, n_targets=1).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Optimizer
    params = list(model.parameters()) + list(regression_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Training loop
    model.train()
    regression_head.train()

    batch_size = min(16, len(X_train_t))
    print(f"  Training for {n_epochs} epochs (batch_size={batch_size})...")

    for epoch in range(n_epochs):
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
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss = {avg_loss:.4f}")

    # Evaluate
    print("  Evaluating...")
    model.eval()
    regression_head.eval()

    with torch.no_grad():
        enc_out = model.encode(X_test_t, domain="NIR")
        z_chem = enc_out["z_chem"]
        y_pred = regression_head(z_chem).squeeze(-1).cpu().numpy()

    y_test_np = y_test_t.cpu().numpy()
    metrics = compute_metrics(y_test_np, y_pred)

    print(f"  Final: R² = {metrics['r2']:.4f}, RMSEP = {metrics['rmsep']:.4f}, RPD = {metrics['rpd']:.2f}")

    return metrics


def main():
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    experiments_dir = project_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\n=== Loading Corn Dataset ===")
    n_transfer = 30
    data = load_corn_data(data_dir, n_transfer=n_transfer, seed=42)
    print(f"  Training samples: {data['n_train']}")
    print(f"  Test samples: {data['n_test']}")

    # Run fine-tuning test
    metrics = run_spectral_fm_finetune(data, n_epochs=20, device=device)

    # Save results
    output = {
        "experiment": "spectral_fm_finetune_random_init",
        "dataset": "corn_mp6_moisture",
        "n_transfer": n_transfer,
        "n_test": data["n_test"],
        "pretrained": False,
        "n_epochs": 20,
        "timestamp": datetime.now().isoformat(),
        "results": metrics,
    }

    with open(experiments_dir / "finetune_test.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to experiments/finetune_test.json")

    # Load and compare with baselines
    baselines_path = experiments_dir / "baselines_corn.json"
    if baselines_path.exists():
        with open(baselines_path) as f:
            baselines = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON: All Methods (Corn, Moisture)")
        print("=" * 70)
        print(f"{'Method':<20} {'R²':>10} {'RMSEP':>10} {'RPD':>8}")
        print("-" * 70)

        for method, m in baselines["results"].items():
            print(f"{method:<20} {m['r2']:>10.4f} {m['rmsep']:>10.4f} {m['rpd']:>8.2f}")

        print(f"{'Spektron (no PT)':<20} {metrics['r2']:>10.4f} {metrics['rmsep']:>10.4f} {metrics['rpd']:>8.2f}")
        print("-" * 70)

    print("\n✓ Fine-tuning test completed!")


if __name__ == "__main__":
    main()
