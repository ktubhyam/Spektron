#!/usr/bin/env python3
"""
Spektron: Run Classical Baselines Only (Fast)

Tasks:
1. Run PLS, PDS, SBC, DS on corn m5→mp6 transfer (moisture)
2. Save results to experiments/baselines_corn.json

Usage:
    python scripts/run_baselines_only.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from datetime import datetime

from src.evaluation.baselines import (
    PLSCalibration, PDS, SBC, DS,
    compute_metrics, run_baseline_comparison, print_results_table
)


def load_corn_data(data_dir: str, n_transfer: int = 30, seed: int = 42):
    """Load corn dataset for m5→mp6 transfer experiment."""
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


def main():
    # Paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    experiments_dir = project_dir / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    # Load data
    print("=== Loading Corn Dataset ===")
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

    print("\n✓ Baseline experiments completed!")


if __name__ == "__main__":
    main()
