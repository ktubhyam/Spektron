#!/usr/bin/env python3
"""
E2 classical baseline: PLS2 cross-spectral prediction (IR <-> Raman).

Trains PLS2 (multivariate PLS) to predict target spectra from source
spectra and evaluates on the same test split as E2.

Usage:
    python experiments/e2_pls_baseline.py --h5-path data/raw/qm9s/qm9s_processed.h5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression

from src.data.cross_spectral import CrossSpectralDataset
from src.evaluation.cross_spectral_metrics import compute_cross_spectral_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

N_TRAIN_SAMPLES = 10000   # Subset for PLS fitting (full 85K is too slow)
N_COMPONENTS = 30


def load_split_arrays(
    h5_path: str, direction: str, split: str, max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load source and target arrays for a split."""
    ds = CrossSpectralDataset(
        h5_path, split=split, direction=direction, max_samples=max_samples,
    )
    sources, targets = [], []
    for i in range(len(ds)):
        item = ds[i]
        sources.append(item["source_spectrum"].numpy())
        targets.append(item["target_spectrum"].numpy())
    return np.array(sources, dtype=np.float32), np.array(targets, dtype=np.float32)


def run_pls_baseline(direction: str, h5_path: str, n_components: int) -> dict:
    """Train PLS2 and evaluate cross-spectral prediction."""
    log.info(f"\n  Direction: {direction}, n_components={n_components}")

    t0 = time.time()
    log.info(f"  Loading {N_TRAIN_SAMPLES} training samples...")
    X_train, Y_train = load_split_arrays(
        h5_path, direction, "train", max_samples=N_TRAIN_SAMPLES
    )
    log.info(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")

    pls = PLSRegression(n_components=n_components, max_iter=500)
    log.info(f"  Fitting PLS2 ({n_components} components)...")
    pls.fit(X_train, Y_train)
    elapsed_fit = time.time() - t0

    log.info(f"  Loading test split...")
    X_test, Y_test = load_split_arrays(h5_path, direction, "test")
    log.info(f"  X_test: {X_test.shape}")

    log.info(f"  Predicting on test set...")
    Y_pred = pls.predict(X_test).astype(np.float32)
    elapsed_pred = time.time() - t0 - elapsed_fit

    pred_t = torch.tensor(Y_pred)
    target_t = torch.tensor(Y_test)
    metrics = compute_cross_spectral_metrics(pred_t, target_t)
    means = {k: float(v.mean().item()) for k, v in metrics.items()}

    log.info(f"  MSE={means['mse']:.6f} Cosine={means['cosine_similarity']:.4f} "
             f"PeakRecall={means['peak_recall']:.4f}")
    log.info(f"  Fit: {elapsed_fit:.1f}s  Predict: {elapsed_pred:.1f}s")

    return {
        "direction": direction,
        "n_components": n_components,
        "n_train_samples": N_TRAIN_SAMPLES,
        "n_test_samples": len(X_test),
        "fit_time_sec": elapsed_fit,
        **means,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 PLS cross-spectral baseline")
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
    parser.add_argument("--directions", nargs="+", default=["ir2raman", "raman2ir"])
    parser.add_argument("--output", type=str,
                        default="experiments/results/e2_pls_baseline.json")
    args = parser.parse_args()

    results = {}
    for direction in args.directions:
        results[direction] = run_pls_baseline(direction, args.h5_path, args.n_components)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  E2 PLS Baseline (n_components={args.n_components}, "
          f"n_train={N_TRAIN_SAMPLES})")
    print(f"{'='*70}")
    print(f"  {'Direction':<15} {'MSE':>12} {'Cosine':>10} {'PeakRecall':>12}")
    print(f"  {'-'*50}")
    for d, r in results.items():
        print(f"  {d:<15} {r['mse']:>12.6f} {r['cosine_similarity']:>10.4f} "
              f"{r['peak_recall']:>12.4f}")
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "E2: PLS cross-spectral baseline",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {"n_components": args.n_components,
                       "n_train_samples": N_TRAIN_SAMPLES},
            "results": results,
        }, f, indent=2)
    log.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
