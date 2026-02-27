#!/usr/bin/env python3
"""
E3: Transfer Function Analysis

Computes H(z) for each oscillator in a trained D-LinOSS model.
Generates:
1. Filter bank heatmap (learned bandpass filters per layer)
2. Pole-zero plots (eigenvalues on complex plane)
3. Per-layer frequency coverage (aggregate |H| per layer)
4. Statistical comparison: learned vs random initialization

Usage:
    python experiments/e3_transfer_function.py --checkpoint checkpoints/best_pretrain.pt

    # Also compare against random (untrained) model
    python experiments/e3_transfer_function.py --checkpoint ... --compare-random
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time

import numpy as np
import torch

from src.config import get_dlinoss_config, get_benchmark_config
from src.models.spektron import Spektron
from src.analysis.transfer_function import (
    extract_layer_responses,
    plot_filter_bank,
    plot_pole_zero,
    plot_layer_frequency_coverage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# Known functional group absorption bands (cm^-1)
REFERENCE_BANDS = {
    "O-H stretch": 3400,
    "N-H stretch": 3300,
    "C-H stretch": 2900,
    "C≡N stretch": 2200,
    "C=O stretch": 1700,
    "C=C stretch": 1600,
    "C-O stretch": 1100,
    "C-H bend": 1450,
    "Fingerprint": 900,
}


def analyze_model(model, figures_dir: str, prefix: str = "trained",
                  n_freq: int = 1024):
    """Run full transfer function analysis on a D-LinOSS model."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(model.backbone.layers)
    has_bwd = hasattr(model.backbone, "bwd_layers") and model.backbone.bidirectional

    results = {"layers": []}

    for layer_idx in range(n_layers):
        for direction in ["fwd"] + (["bwd"] if has_bwd else []):
            log.info(f"  Analyzing layer {layer_idx} ({direction})...")

            omega, H_mag, res_freq, damping = extract_layer_responses(
                model, layer_idx=layer_idx, direction=direction, n_freq=n_freq,
            )

            # Filter bank heatmap
            save_path = figures_dir / f"{prefix}_filterbank_L{layer_idx}_{direction}.pdf"
            plot_filter_bank(omega, H_mag, save_path=str(save_path))
            log.info(f"    Saved: {save_path}")

            # Statistics
            layer_stats = {
                "layer": layer_idx,
                "direction": direction,
                "n_oscillators": int(H_mag.shape[0]),
                "resonant_freq_mean": float(np.mean(res_freq)),
                "resonant_freq_std": float(np.std(res_freq)),
                "resonant_freq_min": float(np.min(res_freq)),
                "resonant_freq_max": float(np.max(res_freq)),
                "damping_mean": float(np.mean(damping)),
                "damping_std": float(np.std(damping)),
                "damping_min": float(np.min(damping)),
                "damping_max": float(np.max(damping)),
            }

            # Map resonant frequencies to wavenumber range
            wn_min, wn_max = 500, 4000
            res_wn = wn_min + (res_freq / np.pi) * (wn_max - wn_min)
            layer_stats["resonant_wavenumber_mean"] = float(np.mean(res_wn))
            layer_stats["resonant_wavenumber_std"] = float(np.std(res_wn))

            # Count oscillators near known bands
            band_counts = {}
            for band_name, band_wn in REFERENCE_BANDS.items():
                n_near = int(np.sum(np.abs(res_wn - band_wn) < 100))
                band_counts[band_name] = n_near
            layer_stats["oscillators_near_bands"] = band_counts

            results["layers"].append(layer_stats)

    # Pole-zero plot (all layers)
    save_path = figures_dir / f"{prefix}_pole_zero.pdf"
    plot_pole_zero(model, save_path=str(save_path))
    log.info(f"  Saved: {save_path}")

    # Per-layer frequency coverage
    save_path = figures_dir / f"{prefix}_layer_coverage.pdf"
    plot_layer_frequency_coverage(model, save_path=str(save_path))
    log.info(f"  Saved: {save_path}")

    return results


def compare_to_random(trained_results: dict, random_results: dict) -> dict:
    """Statistical comparison between trained and random model."""
    comparison = {}

    for trained_layer in trained_results["layers"]:
        layer_idx = trained_layer["layer"]
        direction = trained_layer["direction"]
        key = f"L{layer_idx}_{direction}"

        # Find matching random layer
        random_layer = None
        for rl in random_results["layers"]:
            if rl["layer"] == layer_idx and rl["direction"] == direction:
                random_layer = rl
                break

        if random_layer is None:
            continue

        comparison[key] = {
            "trained_freq_mean": trained_layer["resonant_freq_mean"],
            "random_freq_mean": random_layer["resonant_freq_mean"],
            "trained_freq_std": trained_layer["resonant_freq_std"],
            "random_freq_std": random_layer["resonant_freq_std"],
            "trained_damping_mean": trained_layer["damping_mean"],
            "random_damping_mean": random_layer["damping_mean"],
            "freq_shift": (trained_layer["resonant_freq_mean"]
                           - random_layer["resonant_freq_mean"]),
            "damping_shift": (trained_layer["damping_mean"]
                              - random_layer["damping_mean"]),
        }

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="E3: Transfer Function Analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained D-LinOSS checkpoint")
    parser.add_argument("--compare-random", action="store_true",
                        help="Also analyze random (untrained) model")
    parser.add_argument("--n-freq", type=int, default=1024,
                        help="Number of frequency evaluation points")
    parser.add_argument("--figures-dir", type=str,
                        default="figures/e3_transfer_function")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e3_transfer_function.json")
    args = parser.parse_args()

    # Load trained model
    log.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    config = ckpt.get("config", get_dlinoss_config())
    model = Spektron(config)

    # Handle SpektronForPretraining wrapper
    state_dict = ckpt["model_state_dict"]
    model_state = {}
    for k, v in state_dict.items():
        # Strip "model." prefix if saved from SpektronForPretraining
        if k.startswith("model."):
            model_state[k[6:]] = v
        else:
            model_state[k] = v
    model.load_state_dict(model_state, strict=False)
    model.eval()

    log.info("Analyzing trained model...")
    trained_results = analyze_model(
        model, args.figures_dir, prefix="trained", n_freq=args.n_freq,
    )

    all_results = {
        "experiment": "E3: Transfer Function Analysis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": args.checkpoint,
        "trained": trained_results,
    }

    # Compare to random
    if args.compare_random:
        log.info("Analyzing random (untrained) model...")
        random_model = Spektron(config)
        random_model.eval()

        random_results = analyze_model(
            random_model, args.figures_dir, prefix="random", n_freq=args.n_freq,
        )
        all_results["random"] = random_results
        all_results["comparison"] = compare_to_random(
            trained_results, random_results,
        )

    # Print summary
    print(f"\n{'='*70}")
    print(f"  E3 TRANSFER FUNCTION ANALYSIS")
    print(f"{'='*70}")
    for layer in trained_results["layers"]:
        print(f"\n  Layer {layer['layer']} ({layer['direction']}):")
        print(f"    Oscillators: {layer['n_oscillators']}")
        print(f"    Resonant freq: {layer['resonant_freq_mean']:.4f} "
              f"+/- {layer['resonant_freq_std']:.4f} rad")
        print(f"    Wavenumber: {layer['resonant_wavenumber_mean']:.0f} "
              f"+/- {layer['resonant_wavenumber_std']:.0f} cm^-1")
        print(f"    Damping: {layer['damping_mean']:.4f} "
              f"+/- {layer['damping_std']:.4f}")
        print(f"    Near known bands: {layer['oscillators_near_bands']}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved to {output_path}")
    log.info(f"Figures saved to {args.figures_dir}/")


if __name__ == "__main__":
    main()
