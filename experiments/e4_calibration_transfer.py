#!/usr/bin/env python3
"""
E4: Calibration Transfer

Fine-tunes pretrained architectures on Corn (80x3x700) and Tablet (655x2x650)
datasets for calibration transfer. Compares against classical baselines
(PDS, SBC, DS, CCA, di-PLS, PLS).

Honest about DFT mid-IR → experimental NIR domain gap.
A negative result here is fine if framed properly.

Usage:
    python experiments/e4_calibration_transfer.py \
        --checkpoint checkpoints/best_pretrain.pt

    # Quick test
    python experiments/e4_calibration_transfer.py --checkpoint ... --quick
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import copy
import json
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import get_dlinoss_config, SpektronConfig
from src.models.spektron import Spektron
from src.training.trainer import FinetuneTrainer
from src.evaluation.baselines import run_baseline_comparison
from src.data.datasets import (
    SpectralPreprocessor, CornDataset, TabletDataset,
    CalibrationTransferDataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "n_samples": int(len(y_true)),
    }


def run_corn_experiment(
    model: Spektron,
    config: SpektronConfig,
    data_dir: str,
    n_transfer_list: list,
    n_epochs: int = 100,
    lr: float = 1e-4,
    patience: int = 15,
    seeds: int = 3,
) -> dict:
    """Run calibration transfer on Corn dataset."""
    log.info("\n--- Corn Dataset ---")

    data_dir = Path(data_dir)
    corn_spectra = {
        inst: np.load(data_dir / f"processed/corn/{inst}_spectra.npy")
        for inst in ["m5", "mp5", "mp6"]
    }
    corn_props = np.load(data_dir / "processed/corn/properties.npy")
    corn_wavelengths = np.load(data_dir / "processed/corn/wavelengths.npy")

    preprocessor = SpectralPreprocessor(target_length=config.n_channels)

    results = {}
    instrument_pairs = [("m5", "mp5"), ("m5", "mp6"), ("mp5", "mp6")]

    for source_inst, target_inst in instrument_pairs:
        pair_key = f"{source_inst}→{target_inst}"
        log.info(f"\n  Transfer: {pair_key}")
        results[pair_key] = {}

        for n_transfer in n_transfer_list:
            if n_transfer > len(corn_props):
                continue

            seed_results = []
            for seed in range(seeds):
                model_copy = copy.deepcopy(model)

                # Create transfer dataset
                transfer_ds = CalibrationTransferDataset(
                    corn_spectra[source_inst],
                    corn_spectra[target_inst],
                    corn_props[:, 0],  # moisture
                    n_transfer=n_transfer,
                    preprocessor=preprocessor,
                )
                transfer_loader = DataLoader(
                    transfer_ds, batch_size=min(n_transfer, 16), shuffle=True,
                )

                # Fine-tune
                trainer = FinetuneTrainer(
                    model_copy, config, use_wandb=False,
                    run_name=f"e4_corn_{pair_key}_n{n_transfer}_s{seed}",
                )
                trainer.finetune(
                    transfer_loader, n_epochs=n_epochs,
                    lr=lr, patience=patience, freeze_backbone=True,
                )

                # Evaluate on full dataset
                model_copy.eval()
                with torch.no_grad():
                    all_target = []
                    for spec in corn_spectra[target_inst]:
                        processed = preprocessor.process(spec, corn_wavelengths)
                        all_target.append(processed["normalized"])
                    all_target_tensor = torch.tensor(
                        np.array(all_target), dtype=torch.float32,
                    ).to(config.device)

                    output = model_copy.predict(all_target_tensor, mc_samples=10)

                preds = output["prediction"].cpu().numpy()
                metrics = compute_metrics(corn_props[:, 0], preds)
                seed_results.append(metrics)

                del model_copy

            # Aggregate over seeds
            r2s = [r["r2"] for r in seed_results]
            rmses = [r["rmse"] for r in seed_results]
            results[pair_key][n_transfer] = {
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "n_seeds": seeds,
            }
            log.info(f"    N={n_transfer}: R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}")

    return results


def run_baselines(data_dir: str) -> dict:
    """Run classical baseline methods on Corn and Tablet."""
    log.info("\n--- Classical Baselines ---")

    data_dir = Path(data_dir)
    corn_spectra = {
        inst: np.load(data_dir / f"processed/corn/{inst}_spectra.npy")
        for inst in ["m5", "mp5", "mp6"]
    }
    corn_props = np.load(data_dir / "processed/corn/properties.npy")

    n = len(corn_props)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_train = int(n * 0.75)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    results = {}
    for source_inst, target_inst in [("m5", "mp5"), ("m5", "mp6"), ("mp5", "mp6")]:
        pair_key = f"{source_inst}→{target_inst}"
        log.info(f"  {pair_key}")

        baseline_results = run_baseline_comparison(
            corn_spectra[source_inst][train_idx],
            corn_spectra[target_inst][train_idx],
            corn_spectra[source_inst][test_idx],
            corn_spectra[target_inst][test_idx],
            corn_props[train_idx, 0],
            corn_props[test_idx, 0],
        )

        for method, metrics in baseline_results.items():
            log.info(f"    {method}: R²={metrics.get('r2', 'N/A')}")

        results[pair_key] = baseline_results

    return results


def main():
    parser = argparse.ArgumentParser(
        description="E4: Calibration Transfer")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e4_calibration_transfer.json")
    args = parser.parse_args()

    if args.quick:
        args.n_epochs = 10
        args.seeds = 1

    n_transfer_list = [5, 10, 20, 30, 50] if not args.quick else [10, 30]

    # Load model
    config = get_dlinoss_config()
    model = Spektron(config)

    if args.checkpoint:
        log.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                model_state[k[6:]] = v
            else:
                model_state[k] = v
        model.load_state_dict(model_state, strict=False)
    else:
        log.warning("No checkpoint provided — using random initialization")

    model.to(config.device)

    all_results = {
        "experiment": "E4: Calibration Transfer",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": args.checkpoint,
    }

    # Baselines
    all_results["baselines"] = run_baselines(args.data_dir)

    # Spektron fine-tuning
    all_results["spektron_corn"] = run_corn_experiment(
        model, config, args.data_dir,
        n_transfer_list=n_transfer_list,
        n_epochs=args.n_epochs,
        lr=args.lr,
        patience=args.patience,
        seeds=args.seeds,
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"  E4 CALIBRATION TRANSFER RESULTS")
    print(f"{'='*70}")

    print("\n  BASELINES:")
    for pair, methods in all_results["baselines"].items():
        print(f"    {pair}:")
        for method, metrics in methods.items():
            print(f"      {method}: R²={metrics.get('r2', 'N/A')}")

    print("\n  SPEKTRON (fine-tuned):")
    for pair, n_results in all_results["spektron_corn"].items():
        print(f"    {pair}:")
        for n, metrics in n_results.items():
            print(f"      N={n}: R²={metrics['r2_mean']:.4f}±{metrics['r2_std']:.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
