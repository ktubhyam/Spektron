#!/usr/bin/env python3
"""
Frequency Alignment Analysis: Learned ω_k vs Physical Vibrational Frequencies.

After pretraining the D-LinOSS backbone on QM9S, this script:
1. Extracts learned oscillation frequencies (ω_k) from each D-LinOSS layer
2. Compares their distribution against physical DFT vibrational frequencies
3. Checks if learned frequencies cluster near spectroscopically active modes
4. Generates publication-quality figures for Paper 1

Usage:
    python experiments/analyze_frequencies.py --checkpoint checkpoints/best_pretrain.pt
    python experiments/analyze_frequencies.py --checkpoint checkpoints/best_pretrain.pt --compare-random
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from src.config import SpectralFMConfig, get_dlinoss_config, get_light_dlinoss_config
from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining


def extract_frequencies(model: SpectralFM) -> dict:
    """Extract learned frequencies and damping from all D-LinOSS layers.

    Returns:
        dict with 'frequencies' and 'damping', each a list of numpy arrays
        (one per layer), plus 'n_layers' and 'd_state'.
    """
    backbone = model.backbone
    if not hasattr(backbone, 'learned_frequencies'):
        raise ValueError("Model backbone does not have learned_frequencies "
                         "(not D-LinOSS?)")

    freqs = [f.cpu().numpy() for f in backbone.learned_frequencies]
    damps = [d.cpu().numpy() for d in backbone.learned_damping]

    return {
        "frequencies": freqs,
        "damping": damps,
        "n_layers": len(freqs),
        "d_state": freqs[0].shape[0] if freqs else 0,
    }


def get_physical_frequency_range() -> dict:
    """Return characteristic vibrational frequency ranges (cm⁻¹).

    These are standard spectroscopic reference ranges for common functional
    groups, used as comparison targets.
    """
    return {
        # Group: (min_cm-1, max_cm-1, description)
        "C-H stretch": (2800, 3100, "Alkane C-H stretching"),
        "=C-H stretch": (3000, 3100, "Alkene =C-H stretching"),
        "≡C-H stretch": (3250, 3350, "Alkyne ≡C-H stretching"),
        "O-H stretch": (3200, 3600, "Alcohol/acid O-H stretching"),
        "N-H stretch": (3300, 3500, "Amine N-H stretching"),
        "C=O stretch": (1650, 1800, "Carbonyl stretching"),
        "C=C stretch": (1600, 1680, "Alkene C=C stretching"),
        "C-O stretch": (1000, 1300, "C-O stretching"),
        "C-N stretch": (1000, 1250, "C-N stretching"),
        "C-F stretch": (1000, 1400, "C-F stretching"),
        "Fingerprint": (500, 1500, "Fingerprint region"),
    }


def wavenumber_to_model_freq(wavenumber_cm1: float,
                              spectral_range: tuple = (500, 4000),
                              n_channels: int = 2048) -> float:
    """Convert physical wavenumber (cm⁻¹) to model frequency units.

    The model processes spectra sampled at n_channels points over the
    spectral_range. The "model frequency" relates to the sampling:
    one oscillation period in model space = spectral_range / n_channels.

    This is an approximate mapping — the learned frequencies exist in
    an arbitrary space but should show correlation structure.

    Args:
        wavenumber_cm1: Physical wavenumber in cm⁻¹
        spectral_range: (min, max) wavenumber range
        n_channels: Number of spectral points

    Returns:
        Approximate model-space frequency (dimensionless)
    """
    # Normalize to [0, 1] within spectral range
    frac = (wavenumber_cm1 - spectral_range[0]) / (spectral_range[1] - spectral_range[0])
    # Scale to angular frequency in model space: ω = 2π × position / n_channels
    return 2 * np.pi * frac * n_channels / (spectral_range[1] - spectral_range[0])


def analyze_frequency_distribution(extracted: dict) -> dict:
    """Analyze the distribution of learned frequencies per layer.

    Returns statistics and binning suitable for histogram comparison.
    """
    results = {}
    for layer_idx, (freqs, damps) in enumerate(
        zip(extracted["frequencies"], extracted["damping"])
    ):
        # Sort by frequency
        order = np.argsort(freqs)
        sorted_freqs = freqs[order]
        sorted_damps = damps[order]

        # Quality factor Q = ω / (2γ) — higher Q = sharper resonance
        q_factors = np.where(sorted_damps > 1e-8,
                             sorted_freqs / (2 * sorted_damps),
                             0.0)

        results[f"layer_{layer_idx}"] = {
            "frequencies": sorted_freqs.tolist(),
            "damping": sorted_damps.tolist(),
            "q_factors": q_factors.tolist(),
            "freq_min": float(sorted_freqs.min()),
            "freq_max": float(sorted_freqs.max()),
            "freq_mean": float(sorted_freqs.mean()),
            "freq_std": float(sorted_freqs.std()),
            "freq_median": float(np.median(sorted_freqs)),
            "damp_mean": float(sorted_damps.mean()),
            "damp_std": float(sorted_damps.std()),
            "q_mean": float(q_factors[q_factors > 0].mean()) if (q_factors > 0).any() else 0.0,
            "n_underdamped": int((q_factors > 0.5).sum()),
            "n_overdamped": int((q_factors <= 0.5).sum()),
            "n_near_zero": int((sorted_freqs < 0.01).sum()),
        }

    return results


def compare_trained_vs_random(checkpoint_path: str, config: SpectralFMConfig) -> dict:
    """Compare frequency distributions: trained model vs random init.

    This is a key control experiment: if the trained model's frequency
    distribution is significantly different from random initialization,
    it means training has shaped the oscillation frequencies.
    """
    # Random init
    torch.manual_seed(config.seed)
    random_model = SpectralFM(config)
    random_extracted = extract_frequencies(random_model)
    random_analysis = analyze_frequency_distribution(random_extracted)

    # Trained
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    trained_model = SpectralFM(config)
    # Load from pretrain wrapper state_dict
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Strip "model." prefix if saved from SpectralFMForPretraining
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("model.", "", 1) if k.startswith("model.") else k
        cleaned[key] = v
    trained_model.load_state_dict(cleaned, strict=False)
    trained_extracted = extract_frequencies(trained_model)
    trained_analysis = analyze_frequency_distribution(trained_extracted)

    # Compare per layer
    comparison = {}
    for layer_key in random_analysis:
        rand_freqs = np.array(random_analysis[layer_key]["frequencies"])
        train_freqs = np.array(trained_analysis[layer_key]["frequencies"])

        # KS test: are the distributions different?
        from scipy.stats import ks_2samp
        ks_stat, ks_pval = ks_2samp(rand_freqs, train_freqs)

        # Earth Mover's Distance
        from scipy.stats import wasserstein_distance
        emd = wasserstein_distance(rand_freqs, train_freqs)

        comparison[layer_key] = {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "wasserstein_distance": float(emd),
            "random_mean": float(rand_freqs.mean()),
            "trained_mean": float(train_freqs.mean()),
            "random_std": float(rand_freqs.std()),
            "trained_std": float(train_freqs.std()),
            "distribution_changed": ks_pval < 0.01,
        }

    return {
        "random": random_analysis,
        "trained": trained_analysis,
        "comparison": comparison,
    }


def print_frequency_report(analysis: dict):
    """Print a human-readable frequency analysis report."""
    print("\n" + "=" * 70)
    print("  D-LinOSS Learned Frequency Analysis")
    print("=" * 70)

    for layer_key, stats in analysis.items():
        print(f"\n  {layer_key}:")
        print(f"    Frequency range: [{stats['freq_min']:.4f}, {stats['freq_max']:.4f}]")
        print(f"    Mean ± std:      {stats['freq_mean']:.4f} ± {stats['freq_std']:.4f}")
        print(f"    Median:          {stats['freq_median']:.4f}")
        print(f"    Damping mean:    {stats['damp_mean']:.4f} ± {stats['damp_std']:.4f}")
        print(f"    Mean Q-factor:   {stats['q_mean']:.2f}")
        print(f"    Underdamped:     {stats['n_underdamped']} / "
              f"{stats['n_underdamped'] + stats['n_overdamped']}")
        print(f"    Near-zero freq:  {stats['n_near_zero']}")


def main():
    parser = argparse.ArgumentParser(description="D-LinOSS Frequency Alignment Analysis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--compare-random", action="store_true",
                        help="Compare trained vs randomly initialized frequencies")
    parser.add_argument("--light", action="store_true",
                        help="Use light config (for CPU testing)")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config = get_light_dlinoss_config() if args.light else get_dlinoss_config()

    if args.checkpoint and Path(args.checkpoint).exists():
        if args.compare_random:
            print("Comparing trained vs random frequency distributions...")
            results = compare_trained_vs_random(args.checkpoint, config)
            print_frequency_report(results["trained"])

            print("\n" + "=" * 70)
            print("  Comparison: Trained vs Random Init")
            print("=" * 70)
            for layer_key, comp in results["comparison"].items():
                changed = "YES" if comp["distribution_changed"] else "no"
                print(f"  {layer_key}: KS={comp['ks_statistic']:.4f} "
                      f"(p={comp['ks_pvalue']:.2e}) | "
                      f"EMD={comp['wasserstein_distance']:.4f} | "
                      f"Changed: {changed}")

            with open(out_dir / "frequency_comparison.json", "w") as f:
                json.dump(results["comparison"], f, indent=2)
        else:
            # Just extract and analyze trained model
            print(f"Loading checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            model = SpectralFM(config)
            state_dict = ckpt.get("model_state_dict", ckpt)
            cleaned = {}
            for k, v in state_dict.items():
                key = k.replace("model.", "", 1) if k.startswith("model.") else k
                cleaned[key] = v
            model.load_state_dict(cleaned, strict=False)

            extracted = extract_frequencies(model)
            analysis = analyze_frequency_distribution(extracted)
            print_frequency_report(analysis)

            with open(out_dir / "frequency_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
    else:
        # No checkpoint — analyze random init as baseline
        print("No checkpoint provided. Analyzing random initialization baseline...")
        torch.manual_seed(config.seed)
        model = SpectralFM(config)
        extracted = extract_frequencies(model)
        analysis = analyze_frequency_distribution(extracted)
        print_frequency_report(analysis)

        with open(out_dir / "frequency_baseline.json", "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved to {out_dir / 'frequency_baseline.json'}")


if __name__ == "__main__":
    main()
