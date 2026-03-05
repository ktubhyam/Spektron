#!/usr/bin/env python3
"""
E3: Transfer Function Analysis

Computes H(z) for each oscillator in trained D-LinOSS models.

Analyses:
1. Multi-seed: aggregates statistics across all trained checkpoints.
2. Random ensemble: averages 20 random initializations (not just 1).
3. B,C-weighted frequencies: true effective response weighting.
4. KS tests + Cohen's d: per-layer statistical tests with effect sizes.
5. Spectral variance: per-wavenumber QM9S variance vs oscillator density
   (shown for BOTH trained and random to distinguish training vs init).
6. Layer specialization: inter-layer frequency diversity index.
7. Filter peak locations: max|H(ω)| vs pole locations.
8. B,C coupling concentration: do trained models create specialist oscillators?

Usage:
    python experiments/e3_transfer_function.py \\
        --checkpoints checkpoints/e1_dlinoss_s42/best_pretrain.pt \\
                      checkpoints/e1_dlinoss_s43/best_pretrain.pt \\
                      checkpoints/e1_dlinoss_s44/best_pretrain.pt \\
        --h5-path data/raw/qm9s/qm9s_processed.h5 \\
        --n-random 20
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import get_dlinoss_config
from src.models.spektron import Spektron
from src.analysis.transfer_function import (
    extract_layer_responses,
    compute_bc_weights,
    plot_filter_bank,
    plot_pole_zero,
    plot_layer_frequency_coverage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

WN_MIN, WN_MAX = 500.0, 4000.0

REFERENCE_BANDS = {
    "O-H": 3400, "N-H": 3300, "C-H": 2900, "C≡N": 2200,
    "C=O": 1700, "C=C": 1600, "C-H bend": 1450,
    "C-O": 1100, "Fingerprint": 900,
}


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyze_one_model(model: Spektron, n_freq: int = 1024) -> dict:
    """Extract per-layer frequency, damping, B,C-coupling, and filter peaks."""
    backbone = model.backbone
    n_layers = len(backbone.layers)
    has_bwd = hasattr(backbone, "bwd_layers") and backbone.bidirectional

    layers_data = {}
    for layer_idx in range(n_layers):
        for direction in ["fwd"] + (["bwd"] if has_bwd else []):
            key = f"L{layer_idx}_{direction}"

            omega, H_mag, res_freq, damping = extract_layer_responses(
                model, layer_idx=layer_idx, direction=direction, n_freq=n_freq,
            )

            # B, C coupling weights: w_p = ||B[p]||_F * ||C[:,p]||_F
            layer_list = backbone.layers if direction == "fwd" else backbone.bwd_layers
            bc_w = compute_bc_weights(layer_list[layer_idx].layer)  # (P,)

            # Pole-based wavenumber mapping
            res_wn = WN_MIN + (res_freq / np.pi) * (WN_MAX - WN_MIN)  # (P,)

            # Filter peak wavenumber: where max|H(ω)| occurs per oscillator
            # This can differ from the pole angle when damping is high
            peak_omega_idx = np.argmax(H_mag, axis=1)  # (P,)
            peak_wn = WN_MIN + (omega[peak_omega_idx] / np.pi) * (WN_MAX - WN_MIN)

            # B,C-weighted statistics
            w_sum = bc_w.sum()
            if w_sum > 0:
                wt_wn_mean = float(np.dot(bc_w, res_wn) / w_sum)
                wt_wn_std = float(np.sqrt(np.dot(bc_w, (res_wn - wt_wn_mean) ** 2) / w_sum))
                wt_peak_mean = float(np.dot(bc_w, peak_wn) / w_sum)
            else:
                wt_wn_mean = float(np.mean(res_wn))
                wt_wn_std = float(np.std(res_wn))
                wt_peak_mean = float(np.mean(peak_wn))

            # B,C concentration: coefficient of variation (specialist vs generalist)
            bc_cv = float(np.std(bc_w) / (np.mean(bc_w) + 1e-12))

            layers_data[key] = {
                "layer_idx": layer_idx,
                "direction": direction,
                "n_oscillators": int(H_mag.shape[0]),
                # Raw arrays
                "res_freq": res_freq,
                "res_wn": res_wn,
                "peak_wn": peak_wn,
                "damping": damping,
                "bc_weights": bc_w,
                "H_mag": H_mag,
                "omega": omega,
                # Summaries
                "wn_mean": float(np.mean(res_wn)),
                "wn_std": float(np.std(res_wn)),
                "weighted_wn_mean": wt_wn_mean,
                "weighted_wn_std": wt_wn_std,
                "peak_wn_mean": float(np.mean(peak_wn)),
                "weighted_peak_wn_mean": wt_peak_mean,
                "damping_mean": float(np.mean(damping)),
                "damping_std": float(np.std(damping)),
                "bc_weight_mean": float(np.mean(bc_w)),
                "bc_weight_std": float(np.std(bc_w)),
                "bc_weight_cv": bc_cv,
            }

    return layers_data


def _load_model(ckpt_path: str, config=None) -> tuple[Spektron, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config or ckpt.get("config", get_dlinoss_config())
    model = Spektron(cfg)
    state_dict = ckpt["model_state_dict"]
    model_state = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Aggregate across seeds / random inits
# ---------------------------------------------------------------------------

def aggregate_seed_results(per_seed: list[dict]) -> dict:
    """Pool raw arrays and compute summaries across seeds or random inits."""
    keys = list(per_seed[0].keys())
    agg = {}

    for key in keys:
        all_wn = np.concatenate([s[key]["res_wn"] for s in per_seed])
        all_freq = np.concatenate([s[key]["res_freq"] for s in per_seed])
        all_peak_wn = np.concatenate([s[key]["peak_wn"] for s in per_seed])
        all_damping = np.concatenate([s[key]["damping"] for s in per_seed])
        all_bc = np.concatenate([s[key]["bc_weights"] for s in per_seed])

        w_sum = all_bc.sum()
        if w_sum > 0:
            wt_wn_mean = float(np.dot(all_bc, all_wn) / w_sum)
            wt_wn_std = float(np.sqrt(np.dot(all_bc, (all_wn - wt_wn_mean) ** 2) / w_sum))
            wt_peak_mean = float(np.dot(all_bc, all_peak_wn) / w_sum)
        else:
            wt_wn_mean = float(np.mean(all_wn))
            wt_wn_std = float(np.std(all_wn))
            wt_peak_mean = float(np.mean(all_peak_wn))

        seed_wn = [s[key]["wn_mean"] for s in per_seed]
        seed_peak = [s[key]["peak_wn_mean"] for s in per_seed]
        seed_damp = [s[key]["damping_mean"] for s in per_seed]
        seed_bc_cv = [s[key]["bc_weight_cv"] for s in per_seed]

        agg[key] = {
            "layer_idx": per_seed[0][key]["layer_idx"],
            "direction": per_seed[0][key]["direction"],
            "n_oscillators": per_seed[0][key]["n_oscillators"],
            "n_seeds": len(per_seed),
            # Pooled raw (for KS tests, spectral entropy)
            "res_freq": all_freq,
            "res_wn": all_wn,
            "peak_wn": all_peak_wn,
            "damping": all_damping,
            "bc_weights": all_bc,
            # Summaries
            "wn_mean": float(np.mean(seed_wn)),
            "wn_sem": float(np.std(seed_wn)),
            "peak_wn_mean": float(np.mean(seed_peak)),
            "peak_wn_sem": float(np.std(seed_peak)),
            "weighted_wn_mean": wt_wn_mean,
            "weighted_wn_std": wt_wn_std,
            "weighted_peak_wn_mean": wt_peak_mean,
            "damping_mean": float(np.mean(seed_damp)),
            "damping_sem": float(np.std(seed_damp)),
            "bc_weight_cv_mean": float(np.mean(seed_bc_cv)),
            "bc_weight_cv_sem": float(np.std(seed_bc_cv)),
            # Per-seed list for Cohen's d comparison
            "bc_weight_cv_list": [float(x) for x in seed_bc_cv],
        }

    return agg


# ---------------------------------------------------------------------------
# Random ensemble
# ---------------------------------------------------------------------------

def analyze_random_ensemble(
    config, n_random: int = 20, n_freq: int = 1024, base_seed: int = 1000,
) -> dict:
    log.info(f"Analyzing {n_random} random initializations...")
    per_init = []
    for i in range(n_random):
        torch.manual_seed(base_seed + i)
        np.random.seed(base_seed + i)
        m = Spektron(config)
        m.eval()
        per_init.append(analyze_one_model(m, n_freq=n_freq))
        if (i + 1) % 5 == 0:
            log.info(f"  Random init {i+1}/{n_random} done")
    return aggregate_seed_results(per_init)


# ---------------------------------------------------------------------------
# Layer specialization
# ---------------------------------------------------------------------------

def compute_layer_specialization(agg: dict) -> dict:
    """Compute inter-layer frequency diversity index.

    For each direction (fwd/bwd), collects the per-layer weighted WN mean
    and computes the std across layers. Higher std = more layer specialization
    (different layers tuned to different frequency bands).

    Also computes the range (max-min) per direction.
    """
    for direction in ["fwd", "bwd"]:
        keys = sorted(
            [k for k in agg if k.endswith(f"_{direction}")],
            key=lambda k: agg[k]["layer_idx"],
        )
        if not keys:
            continue

        wn_means = np.array([agg[k]["weighted_wn_mean"] for k in keys])
        peak_means = np.array([agg[k]["weighted_peak_wn_mean"] for k in keys])

    result = {}
    for direction in ["fwd", "bwd"]:
        keys = sorted(
            [k for k in agg if k.endswith(f"_{direction}")],
            key=lambda k: agg[k]["layer_idx"],
        )
        if not keys:
            continue
        wn_means = np.array([agg[k]["weighted_wn_mean"] for k in keys])
        peak_means = np.array([agg[k]["weighted_peak_wn_mean"] for k in keys])
        result[direction] = {
            "layer_wn_means": wn_means.tolist(),
            "layer_peak_means": peak_means.tolist(),
            "wn_diversity_std": float(np.std(wn_means)),
            "wn_diversity_range": float(np.max(wn_means) - np.min(wn_means)),
            "peak_diversity_std": float(np.std(peak_means)),
            "peak_diversity_range": float(np.max(peak_means) - np.min(peak_means)),
        }

    return result


# ---------------------------------------------------------------------------
# KS tests + effect sizes
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d effect size."""
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
        / (n_a + n_b - 2)
    )
    return float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-12))


def run_ks_tests(trained_agg: dict, random_agg: dict) -> dict:
    """Per-layer KS tests (unweighted + B,C-weighted) with effect sizes.

    Reports:
    - KS D statistic and p-value for frequency distribution
    - Cohen's d for damping distribution (primary finding)
    - WN shift in cm^-1 (actual effect size, not just significance)
    - Filter-peak KS test (independent of pole analysis)
    """
    from scipy.stats import ks_2samp

    results = {}
    rng = np.random.default_rng(42)

    for key in trained_agg:
        if key not in random_agg:
            continue

        t = trained_agg[key]
        r = random_agg[key]

        # --- Frequency: unweighted pole-based ---
        D_raw, p_raw = ks_2samp(t["res_wn"], r["res_wn"])
        wn_shift = t["wn_mean"] - r["wn_mean"]

        # --- Frequency: B,C-weighted (resample to equalize N) ---
        n_samp = 2048
        t_w = t["bc_weights"] / (t["bc_weights"].sum() + 1e-12)
        r_w = r["bc_weights"] / (r["bc_weights"].sum() + 1e-12)
        t_samp = rng.choice(t["res_wn"], size=n_samp, p=t_w)
        r_samp = rng.choice(r["res_wn"], size=n_samp, p=r_w)
        D_wt, p_wt = ks_2samp(t_samp, r_samp)
        wn_shift_wt = float(np.mean(t_samp) - np.mean(r_samp))

        # --- Filter peak distribution ---
        D_peak, p_peak = ks_2samp(t["peak_wn"], r["peak_wn"])

        # --- Damping: Cohen's d (primary finding) ---
        d_damp = _cohens_d(t["damping"], r["damping"])
        damp_shift = t["damping_mean"] - r["damping_mean"]
        damp_ratio = t["damping_mean"] / (r["damping_mean"] + 1e-12)

        # --- BC concentration: Cohen's d (compare per-seed/per-init distributions) ---
        d_bc_cv = _cohens_d(
            np.array(t["bc_weight_cv_list"]),
            np.array(r["bc_weight_cv_list"]),
        )

        results[key] = {
            # Frequency (pole-based)
            "ks_D_unweighted": float(D_raw),
            "ks_p_unweighted": float(p_raw),
            "wn_shift_cm1": float(wn_shift),
            # Frequency (B,C-weighted)
            "ks_D_weighted": float(D_wt),
            "ks_p_weighted": float(p_wt),
            "wn_shift_weighted_cm1": float(wn_shift_wt),
            # Filter peaks
            "ks_D_peak": float(D_peak),
            "ks_p_peak": float(p_peak),
            # Damping
            "cohens_d_damping": float(d_damp),
            "damping_shift": float(damp_shift),
            "damping_ratio": float(damp_ratio),
            # BC concentration
            "cohens_d_bc_cv": float(d_bc_cv),
        }

    return results


# ---------------------------------------------------------------------------
# Spectral variance
# ---------------------------------------------------------------------------

def compute_spectral_variance(h5_path: str, n_sample: int = 10000) -> dict:
    """Per-wavenumber variance across QM9S IR spectra.

    Returns variance array along with correlation against oscillator
    density for BOTH trained and random (caller computes correlations).
    """
    import h5py

    log.info(f"Computing spectral variance from {h5_path} (n={n_sample})...")
    with h5py.File(h5_path, "r") as f:
        has_ir = f["has_ir"][:]
        ir_idx = np.where(has_ir)[0]
        rng = np.random.default_rng(42)
        sel = np.sort(rng.choice(len(ir_idx), size=min(n_sample, len(ir_idx)), replace=False))
        spectra = np.stack([f["ir"][ir_idx[i]] for i in sel])  # (N, 2048)

    variance = np.var(spectra, axis=0)   # (2048,)
    var_norm = variance / (variance.max() + 1e-12)
    wavenumber = np.linspace(WN_MIN, WN_MAX, 2048)

    log.info(f"  Peak variance at {wavenumber[np.argmax(variance)]:.0f} cm^-1")
    return {
        "wavenumber": wavenumber,
        "variance": variance,
        "variance_normalized": var_norm,
        "n_spectra": int(len(sel)),
        "peak_variance_wavenumber": float(wavenumber[np.argmax(variance)]),
    }


def _oscillator_density_on_grid(wn_samples: np.ndarray, bc_weights: np.ndarray,
                                 wavenumber_grid: np.ndarray) -> np.ndarray:
    """Interpolate B,C-weighted oscillator density onto a wavenumber grid."""
    w = bc_weights / (bc_weights.sum() + 1e-12)
    n_bins = len(wavenumber_grid) - 1
    counts, _ = np.histogram(wn_samples, bins=n_bins,
                              range=(WN_MIN, WN_MAX), weights=w * len(wn_samples),
                              density=True)
    return counts


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_frequency_distributions(
    trained_agg: dict, random_agg: dict, ks_results: dict, save_path: str,
) -> None:
    """Histograms: trained vs random, weighted vs unweighted, per layer."""
    keys = sorted(trained_agg.keys())
    n = len(keys)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), squeeze=False)
    bins = np.linspace(WN_MIN, WN_MAX, 60)

    for row, key in enumerate(keys):
        t = trained_agg[key]
        r = random_agg.get(key)
        ks = ks_results.get(key, {})
        label = f"L{t['layer_idx']} {t['direction']}"

        for col, (weighted, title_suffix) in enumerate([
            (False, "pole-based"),
            (True, "B,C-weighted"),
        ]):
            ax = axes[row, col]
            if weighted:
                t_w = t["bc_weights"] / (t["bc_weights"].sum() + 1e-12)
                ax.hist(t["res_wn"], bins=bins, weights=t_w * len(t["res_wn"]),
                        density=True, alpha=0.6, color="C0", label="Trained")
                if r is not None:
                    r_w = r["bc_weights"] / (r["bc_weights"].sum() + 1e-12)
                    ax.hist(r["res_wn"], bins=bins, weights=r_w * len(r["res_wn"]),
                            density=True, alpha=0.6, color="C3", label="Random")
                D = ks.get("ks_D_weighted", float("nan"))
                p = ks.get("ks_p_weighted", float("nan"))
                shift = ks.get("wn_shift_weighted_cm1", float("nan"))
            else:
                ax.hist(t["res_wn"], bins=bins, density=True, alpha=0.6,
                        color="C0", label="Trained")
                if r is not None:
                    ax.hist(r["res_wn"], bins=bins, density=True, alpha=0.6,
                            color="C3", label="Random")
                D = ks.get("ks_D_unweighted", float("nan"))
                p = ks.get("ks_p_unweighted", float("nan"))
                shift = ks.get("wn_shift_cm1", float("nan"))

            for nu in REFERENCE_BANDS.values():
                ax.axvline(nu, color="gray", lw=0.5, ls=":", alpha=0.4)
            ax.set_title(
                f"{label} — {title_suffix}  KS D={D:.3f}  Δ={shift:+.0f} cm⁻¹",
                fontsize=9)
            ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=8)
            ax.invert_xaxis()

    fig.suptitle("Oscillator Frequency Distributions: Trained vs Random", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {save_path}")


def plot_damping_and_concentration(
    trained_agg: dict, random_agg: dict, ks_results: dict, save_path: str,
) -> None:
    """Two panels: (1) damping per layer with Cohen's d, (2) BC concentration CV."""
    keys = sorted(trained_agg.keys())
    x = np.arange(len(keys))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Panel 1: Damping
    ax = axes[0]
    t_damp = [trained_agg[k]["damping_mean"] for k in keys]
    t_derr = [trained_agg[k]["damping_sem"] for k in keys]
    r_damp = [random_agg[k]["damping_mean"] for k in keys if k in random_agg]
    r_derr = [random_agg[k]["damping_sem"] for k in keys if k in random_agg]
    ax.bar(x - width / 2, t_damp, width, yerr=t_derr, label="Trained",
           color="C0", alpha=0.85, capsize=3)
    if r_damp:
        ax.bar(x + width / 2, r_damp, width, yerr=r_derr, label="Random (20 inits)",
               color="C3", alpha=0.85, capsize=3)

    # Annotate Cohen's d
    for i, key in enumerate(keys):
        d = ks_results.get(key, {}).get("cohens_d_damping", float("nan"))
        ratio = ks_results.get(key, {}).get("damping_ratio", float("nan"))
        if not np.isnan(d):
            ax.text(i, max(t_damp[i], r_damp[i] if r_damp else 0) + 0.002,
                    f"d={d:.2f}\n×{ratio:.2f}", ha="center", va="bottom",
                    fontsize=7, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean damping γ", fontsize=11)
    ax.set_title("Damping: Trained vs Random  (Cohen's d annotated)", fontsize=11)
    ax.legend(fontsize=9)

    # Panel 2: BC coupling concentration (CV)
    ax2 = axes[1]
    t_cv = [trained_agg[k]["bc_weight_cv_mean"] for k in keys]
    t_cv_err = [trained_agg[k]["bc_weight_cv_sem"] for k in keys]
    r_cv = [random_agg[k]["bc_weight_cv_mean"] for k in keys if k in random_agg]
    r_cv_err = [random_agg[k]["bc_weight_cv_sem"] for k in keys if k in random_agg]
    ax2.bar(x - width / 2, t_cv, width, yerr=t_cv_err, label="Trained",
            color="C0", alpha=0.85, capsize=3)
    if r_cv:
        ax2.bar(x + width / 2, r_cv, width, yerr=r_cv_err, label="Random",
                color="C3", alpha=0.85, capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("B,C coupling CV (std/mean)", fontsize=11)
    ax2.set_title("Coupling Concentration: do trained oscillators specialise?", fontsize=11)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {save_path}")


def plot_layer_specialization(
    trained_spec: dict, random_spec: dict, save_path: str,
) -> None:
    """Per-layer weighted frequency means, showing inter-layer diversity."""
    directions = [d for d in ["fwd", "bwd"] if d in trained_spec]
    fig, axes = plt.subplots(1, len(directions), figsize=(7 * len(directions), 5),
                             squeeze=False)

    for col, direction in enumerate(directions):
        ax = axes[0, col]
        t = trained_spec[direction]
        r = random_spec.get(direction, {})

        n_layers = len(t["layer_wn_means"])
        layers = np.arange(n_layers)

        ax.plot(layers, t["layer_wn_means"], "o-", color="C0", lw=2,
                ms=8, label=f"Trained (diversity std={t['wn_diversity_std']:.1f} cm⁻¹)")
        ax.plot(layers, t["layer_peak_means"], "s--", color="C0", lw=1.5,
                ms=7, alpha=0.6, label="Trained (filter peaks)")
        if r:
            ax.plot(layers, r["layer_wn_means"], "o-", color="C3", lw=2,
                    ms=8, label=f"Random (diversity std={r['wn_diversity_std']:.1f} cm⁻¹)")
            ax.plot(layers, r["layer_peak_means"], "s--", color="C3", lw=1.5,
                    ms=7, alpha=0.6, label="Random (filter peaks)")

        for nu in [1700, 1600, 2900]:  # C=O, C=C, C-H
            ax.axhline(nu, color="gray", lw=0.7, ls=":", alpha=0.5)

        ax.set_xlabel("Layer index", fontsize=12)
        ax.set_ylabel("Weighted resonant wavenumber (cm⁻¹)", fontsize=11)
        ax.set_title(f"Layer Frequency Specialisation ({direction})\n"
                     f"Trained diversity ÷ Random diversity = "
                     f"{t['wn_diversity_std'] / (r.get('wn_diversity_std', 1) + 1e-12):.1f}×",
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{i}" for i in layers])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {save_path}")


def plot_spectral_variance_vs_oscillators(
    spec_var: dict, trained_agg: dict, random_agg: dict, save_path: str,
) -> None:
    """Overlay QM9S variance + trained AND random oscillator density.

    Honest comparison: if random also aligns with variance peak,
    it's an architecture property, not a trained behavior.
    """
    wn = spec_var["wavenumber"]
    var_norm = spec_var["variance_normalized"]

    all_t_wn = np.concatenate([v["res_wn"] for v in trained_agg.values()])
    all_t_bc = np.concatenate([v["bc_weights"] for v in trained_agg.values()])
    all_r_wn = np.concatenate([v["res_wn"] for v in random_agg.values()])
    all_r_bc = np.concatenate([v["bc_weights"] for v in random_agg.values()])

    # Compute correlations
    grid = np.array(wn)
    t_dens = _oscillator_density_on_grid(all_t_wn, all_t_bc, grid)
    r_dens = _oscillator_density_on_grid(all_r_wn, all_r_bc, grid)
    r_t = np.corrcoef(var_norm[:-1], t_dens)[0, 1]
    r_r = np.corrcoef(var_norm[:-1], r_dens)[0, 1]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.fill_between(wn, var_norm, alpha=0.25, color="C1")
    ax1.plot(wn, var_norm, color="C1", lw=1.0, alpha=0.8,
             label="QM9S spectral variance (norm.)")
    ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax1.set_ylabel("Normalized spectral variance", fontsize=11, color="C1")
    ax1.tick_params(axis="y", labelcolor="C1")
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    bins = np.linspace(WN_MIN, WN_MAX, 80)
    t_w = all_t_bc / (all_t_bc.sum() + 1e-12)
    r_w = all_r_bc / (all_r_bc.sum() + 1e-12)
    ax2.hist(all_t_wn, bins=bins, weights=t_w * len(all_t_wn), density=True,
             alpha=0.55, color="C0", label=f"Trained (r={r_t:.2f})")
    ax2.hist(all_r_wn, bins=bins, weights=r_w * len(all_r_wn), density=True,
             alpha=0.35, color="C3", label=f"Random (r={r_r:.2f})")
    ax2.set_ylabel("Oscillator density (B,C-weighted)", fontsize=11, color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")

    for nu in REFERENCE_BANDS.values():
        ax1.axvline(nu, color="gray", lw=0.5, ls="--", alpha=0.3)

    ax1.set_title(
        f"QM9S Spectral Variance vs Oscillator Density  "
        f"[Trained r={r_t:.3f}  |  Random r={r_r:.3f}]",
        fontsize=12,
    )
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {save_path}")


def plot_existing_figures(model: Spektron, figures_dir: Path,
                          prefix: str, n_freq: int) -> None:
    n_layers = len(model.backbone.layers)
    has_bwd = hasattr(model.backbone, "bwd_layers") and model.backbone.bidirectional
    for layer_idx in range(n_layers):
        for direction in ["fwd"] + (["bwd"] if has_bwd else []):
            omega, H_mag, _, _ = extract_layer_responses(
                model, layer_idx=layer_idx, direction=direction, n_freq=n_freq,
            )
            sp = figures_dir / f"{prefix}_filterbank_L{layer_idx}_{direction}.pdf"
            plot_filter_bank(omega, H_mag, save_path=str(sp))
    plot_pole_zero(model, save_path=str(figures_dir / f"{prefix}_pole_zero.pdf"))
    plot_layer_frequency_coverage(
        model, save_path=str(figures_dir / f"{prefix}_layer_coverage.pdf"))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialise(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj


_STRIP = {"res_freq", "res_wn", "peak_wn", "damping", "bc_weights", "H_mag", "omega"}


def _summary_only(agg: dict) -> dict:
    return {
        key: {k: _serialise(v) for k, v in val.items() if k not in _STRIP}
        for key, val in agg.items()
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E3: Transfer Function Analysis")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--n-freq", type=int, default=1024)
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--figures-dir", type=str,
                        default="figures/e3_transfer_function")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e3_transfer_function.json")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Trained checkpoints
    log.info(f"Loading {len(args.checkpoints)} checkpoint(s)...")
    per_seed, config = [], None
    for i, ckpt_path in enumerate(args.checkpoints):
        log.info(f"  {i+1}/{len(args.checkpoints)}: {ckpt_path}")
        model, config = _load_model(ckpt_path, config)
        if i == 0:
            plot_existing_figures(model, figures_dir, f"trained_s{i}", args.n_freq)
        per_seed.append(analyze_one_model(model, n_freq=args.n_freq))

    trained_agg = aggregate_seed_results(per_seed)
    log.info(f"Trained aggregate done ({len(per_seed)} seeds).")

    # 2. Random ensemble
    random_agg = analyze_random_ensemble(config, n_random=args.n_random,
                                         n_freq=args.n_freq)
    torch.manual_seed(9999)
    plot_existing_figures(Spektron(config).eval(), figures_dir,
                          "random_example", args.n_freq)

    # 3+4. KS tests + effect sizes
    log.info("Running KS tests + effect sizes...")
    ks_results = run_ks_tests(trained_agg, random_agg)

    # 5. Layer specialization
    trained_spec = compute_layer_specialization(trained_agg)
    random_spec = compute_layer_specialization(random_agg)

    # 6. Spectral variance
    spec_var = None
    if args.h5_path:
        try:
            spec_var = compute_spectral_variance(args.h5_path)
        except Exception as e:
            log.warning(f"Spectral variance failed: {e}")

    # Figures
    log.info("Generating figures...")
    plot_frequency_distributions(trained_agg, random_agg, ks_results,
                                 str(figures_dir / "freq_distributions.pdf"))
    plot_damping_and_concentration(trained_agg, random_agg, ks_results,
                                   str(figures_dir / "damping_and_concentration.pdf"))
    plot_layer_specialization(trained_spec, random_spec,
                              str(figures_dir / "layer_specialization.pdf"))
    if spec_var is not None:
        plot_spectral_variance_vs_oscillators(
            spec_var, trained_agg, random_agg,
            str(figures_dir / "spectral_variance_vs_oscillators.pdf"))

    # Summary
    print(f"\n{'='*78}")
    print(f"  E3: {len(per_seed)} seeds · {args.n_random} random inits")
    print(f"{'='*78}")
    print(f"\n  PRIMARY FINDING — DAMPING (Cohen's d, ratio vs random):")
    print(f"  {'Layer':<10} {'γ trained':>12} {'γ random':>12} {'Ratio':>8} {'Cohen d':>9}")
    print(f"  {'-'*56}")
    for key in sorted(ks_results.keys()):
        ks = ks_results[key]
        t = trained_agg[key]
        r = random_agg[key]
        print(f"  {key:<10} {t['damping_mean']:>12.4f} {r['damping_mean']:>12.4f} "
              f"{ks['damping_ratio']:>8.2f}× {ks['cohens_d_damping']:>9.3f}")

    print(f"\n  LAYER SPECIALISATION (inter-layer frequency diversity):")
    for direction in ["fwd", "bwd"]:
        if direction in trained_spec and direction in random_spec:
            t_div = trained_spec[direction]["wn_diversity_std"]
            r_div = random_spec[direction]["wn_diversity_std"]
            ratio = t_div / (r_div + 1e-12)
            print(f"  {direction}: trained std={t_div:.1f} cm⁻¹  random std={r_div:.1f} cm⁻¹  "
                  f"ratio={ratio:.1f}×")

    print(f"\n  B,C COUPLING CONCENTRATION (specialist oscillators):")
    print(f"  {'Layer':<10} {'T bc_cv':>10} {'R bc_cv':>10} {'Cohen d':>9}")
    print(f"  {'-'*44}")
    for key in sorted(ks_results.keys()):
        t = trained_agg[key]
        r = random_agg[key]
        d = ks_results[key]["cohens_d_bc_cv"]
        print(f"  {key:<10} {t['bc_weight_cv_mean']:>10.3f} {r['bc_weight_cv_mean']:>10.3f} "
              f"{d:>9.3f}")

    if spec_var is not None:
        all_t_wn = np.concatenate([v["res_wn"] for v in trained_agg.values()])
        all_t_bc = np.concatenate([v["bc_weights"] for v in trained_agg.values()])
        all_r_wn = np.concatenate([v["res_wn"] for v in random_agg.values()])
        all_r_bc = np.concatenate([v["bc_weights"] for v in random_agg.values()])
        t_dens = _oscillator_density_on_grid(
            all_t_wn, all_t_bc, spec_var["wavenumber"])
        r_dens = _oscillator_density_on_grid(
            all_r_wn, all_r_bc, spec_var["wavenumber"])
        r_t = np.corrcoef(spec_var["variance_normalized"][:-1], t_dens)[0, 1]
        r_r = np.corrcoef(spec_var["variance_normalized"][:-1], r_dens)[0, 1]
        print(f"\n  SPECTRAL VARIANCE ALIGNMENT:")
        print(f"  Trained r={r_t:.3f}  Random r={r_r:.3f}  "
              f"Δ={r_t - r_r:+.3f}  "
              f"({'trained stronger' if r_t > r_r else 'no trained advantage'})")
    print()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "E3: Transfer Function Analysis (v3)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoints": args.checkpoints,
        "n_seeds": len(per_seed),
        "n_random": args.n_random,
        "trained": _summary_only(trained_agg),
        "random": _summary_only(random_agg),
        "ks_tests": _serialise(ks_results),
        "layer_specialization": {
            "trained": _serialise(trained_spec),
            "random": _serialise(random_spec),
        },
    }
    if spec_var is not None:
        results["spectral_variance"] = {
            k: _serialise(v) for k, v in spec_var.items()
            if k != "wavenumber"
        }
        results["spectral_variance"]["peak_wavenumber"] = spec_var["peak_variance_wavenumber"]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results → {output_path}")
    log.info(f"Figures → {figures_dir}/")


if __name__ == "__main__":
    main()
