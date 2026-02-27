#!/usr/bin/env python3
"""
E1: Symmetry-Stratified Evaluation

Validates Theorem 1 (Information Completeness): molecules with lower R(G,N)
— meaning more symmetry-silent modes — should be harder to reconstruct from
spectra alone, because the spectral inverse map has a larger quotient space.

Protocol:
1. Load pretrained Spektron checkpoint
2. Run MSRP evaluation on QM9S test set
3. Stratify results by:
   a) R(G,N) bins (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
   b) Point group (C1, Cs, C2v, C2h, D2h, ...)
   c) Centrosymmetric vs non-centrosymmetric
4. Compute per-stratum: MSRP loss, variance, z_chem entropy
5. Test Spearman correlation: R(G,N) vs reconstruction error
6. Statistical tests: Mann-Whitney U, Kruskal-Wallis

Output:
- experiments/results/e1_symmetry_stratification.json
- figures/e1_rgn_vs_msrp.pdf
- figures/e1_pointgroup_boxplot.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, mannwhitneyu, kruskal

from src.config import get_dlinoss_config
from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
from src.data.qm9s import build_qm9s_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def load_pretrained(checkpoint_path: str, device: str = "cuda"):
    """Load pretrained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = SpectralFM(config)
    pretrain_model = SpectralFMForPretraining(model, config)

    # Handle DataParallel state dict (keys prefixed with 'module.')
    state_dict = ckpt["model_state_dict"]
    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v
    pretrain_model.load_state_dict(clean_state)

    pretrain_model.to(device)
    pretrain_model.eval()
    log.info(f"Loaded checkpoint from step {ckpt['step']}")
    return pretrain_model, config


@torch.no_grad()
def evaluate_per_sample(model, config, test_loader, device="cuda",
                        n_mask_trials: int = 5):
    """Evaluate MSRP loss per sample with multiple mask trials.

    Returns:
        dict with per-sample metrics: msrp_loss, z_chem, z_inst, mol_idx, domain
    """
    model.eval()
    all_results = defaultdict(list)

    for batch_idx, batch in enumerate(test_loader):
        spectrum = batch["spectrum"].to(device)
        instrument_id = batch.get("instrument_id")
        if instrument_id is not None:
            instrument_id = instrument_id.to(device)

        domain = batch.get("domain", "IR")
        if isinstance(domain, list):
            _dmap = {"NIR": 0, "IR": 1, "RAMAN": 2, "UNKNOWN": 3}
            domain_tensor = torch.tensor(
                [_dmap.get(d, 3) for d in domain],
                dtype=torch.long, device=device,
            )
        else:
            domain_tensor = domain

        B = spectrum.size(0)
        seq_len = config.seq_len

        # Multiple mask trials for robust loss estimate
        sample_losses = torch.zeros(B, device=device)
        for trial in range(n_mask_trials):
            mask_patch_size = config.pretrain.mask_patch_size
            if config.use_raw_embedding:
                mask_patch_size = max(mask_patch_size, config.patch_size)

            mask = torch.zeros(B, seq_len, device=device)
            mask_ratio = config.pretrain.mask_ratio
            n_mask = int(seq_len * mask_ratio)

            for i in range(B):
                n_blocks = max(1, n_mask // mask_patch_size)
                for _ in range(n_blocks):
                    start = torch.randint(0, max(1, seq_len - mask_patch_size + 1), (1,))
                    end = min(start.item() + mask_patch_size, seq_len)
                    mask[i, start:end] = 1

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=torch.cuda.is_available()):
                output = model.model.pretrain_forward(
                    spectrum, mask, domain_tensor, instrument_id
                )

            target = model.model._create_reconstruction_target(spectrum)
            mask_expanded = mask.unsqueeze(-1)
            # Per-sample MSRP loss
            diff = (output["reconstruction"] - target) * mask_expanded
            per_sample = (diff ** 2).sum(dim=(1, 2)) / (
                mask.sum(dim=1) * target.size(-1) + 1e-8
            )
            sample_losses += per_sample

        sample_losses /= n_mask_trials

        # Also extract latent representations
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=torch.cuda.is_available()):
            enc = model.model.encode(spectrum, domain_tensor, instrument_id)

        all_results["msrp_loss"].append(sample_losses.cpu().numpy())
        all_results["z_chem"].append(enc["z_chem"].cpu().numpy())

        # Store metadata
        if "mol_idx" in batch:
            all_results["mol_idx"].append(batch["mol_idx"].numpy())
        if "domain" in batch:
            all_results["domain_str"].extend(
                batch["domain"] if isinstance(batch["domain"], list)
                else [batch["domain"]] * B
            )

        if (batch_idx + 1) % 50 == 0:
            log.info(f"  Evaluated {(batch_idx + 1) * B} samples...")

    # Concatenate
    return {
        "msrp_loss": np.concatenate(all_results["msrp_loss"]),
        "z_chem": np.concatenate(all_results["z_chem"]),
        "mol_idx": np.concatenate(all_results["mol_idx"]) if all_results["mol_idx"] else None,
        "domain_str": all_results["domain_str"] if all_results["domain_str"] else None,
    }


def load_symmetry_metadata(rgn_path: str, pg_path: str = None):
    """Load R(G,N) and point group metadata."""
    data = np.load(rgn_path)
    result = {
        "R": data["R"],
        "N": data["N"],
        "d": data["d"],
        "n_silent": data["n_silent"],
        "is_centro": data["is_centrosymmetric"],
    }

    if pg_path and Path(pg_path).exists():
        with open(pg_path) as f:
            pg_data = json.load(f)
        result["pg_counts"] = pg_data.get("point_group_counts", {})
        if "point_groups" in pg_data:
            result["point_groups"] = pg_data["point_groups"]

    return result


def stratify_by_rgn(msrp_losses, R_values, bins=None):
    """Stratify reconstruction errors by R(G,N) bins."""
    if bins is None:
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.01]

    results = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (R_values >= lo) & (R_values < hi)
        n = mask.sum()
        if n > 0:
            losses = msrp_losses[mask]
            bin_label = f"[{lo:.2f}, {hi:.2f})"
            results[bin_label] = {
                "n": int(n),
                "mean_msrp": float(np.mean(losses)),
                "std_msrp": float(np.std(losses)),
                "median_msrp": float(np.median(losses)),
                "p25_msrp": float(np.percentile(losses, 25)),
                "p75_msrp": float(np.percentile(losses, 75)),
                "mean_R": float(np.mean(R_values[mask])),
            }

    return results


def stratify_by_centrosymmetry(msrp_losses, is_centro):
    """Compare centro vs non-centro reconstruction quality."""
    centro = is_centro.astype(bool)

    if centro.sum() == 0 or (~centro).sum() == 0:
        return {"error": "Insufficient samples in one group"}

    centro_losses = msrp_losses[centro]
    noncentro_losses = msrp_losses[~centro]

    stat, p_value = mannwhitneyu(
        centro_losses, noncentro_losses, alternative="greater"
    )

    return {
        "centrosymmetric": {
            "n": int(centro.sum()),
            "mean_msrp": float(np.mean(centro_losses)),
            "std_msrp": float(np.std(centro_losses)),
            "median_msrp": float(np.median(centro_losses)),
        },
        "non_centrosymmetric": {
            "n": int((~centro).sum()),
            "mean_msrp": float(np.mean(noncentro_losses)),
            "std_msrp": float(np.std(noncentro_losses)),
            "median_msrp": float(np.median(noncentro_losses)),
        },
        "mann_whitney_u": {
            "statistic": float(stat),
            "p_value": float(p_value),
            "hypothesis": "centrosymmetric > non-centrosymmetric (one-sided)",
        },
    }


def compute_correlations(msrp_losses, R_values, N_values, n_silent):
    """Compute correlation between molecular properties and reconstruction error."""
    results = {}

    # R(G,N) vs MSRP
    rho, p = spearmanr(R_values, msrp_losses)
    results["R_vs_msrp"] = {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "interpretation": (
            "Negative rho = lower R(G,N) → higher MSRP → harder to reconstruct. "
            "This validates Theorem 1."
        ),
    }

    # N_atoms vs MSRP
    rho, p = spearmanr(N_values, msrp_losses)
    results["N_vs_msrp"] = {
        "spearman_rho": float(rho),
        "p_value": float(p),
    }

    # n_silent vs MSRP
    rho, p = spearmanr(n_silent, msrp_losses)
    results["n_silent_vs_msrp"] = {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "interpretation": (
            "Positive rho = more silent modes → higher MSRP. "
            "Silent modes are unobservable by spectroscopy."
        ),
    }

    return results


def generate_figures(results, figures_dir="figures"):
    """Generate publication figures for E1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping figures")
        return

    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Publication style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # Fig E1a: R(G,N) bins vs MSRP (bar chart with error bars)
    rgn_bins = results.get("rgn_stratification", {})
    if rgn_bins:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        labels = list(rgn_bins.keys())
        means = [rgn_bins[k]["mean_msrp"] for k in labels]
        stds = [rgn_bins[k]["std_msrp"] for k in labels]
        ns = [rgn_bins[k]["n"] for k in labels]

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color="#2166AC",
                      alpha=0.8, edgecolor="white")

        # Annotate with sample counts
        for i, (bar, n) in enumerate(zip(bars, ns)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.001,
                    f"n={n}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("R(G,N) bin")
        ax.set_ylabel("MSRP Loss (lower = better)")
        ax.set_title("E1: Reconstruction Error by Information Completeness Ratio")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "e1_rgn_vs_msrp.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "e1_rgn_vs_msrp.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        log.info(f"Saved e1_rgn_vs_msrp.pdf")

    # Fig E1b: Centro vs non-centro comparison
    centro = results.get("centrosymmetry", {})
    if "centrosymmetric" in centro and "non_centrosymmetric" in centro:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        groups = ["Non-centro", "Centro"]
        means = [
            centro["non_centrosymmetric"]["mean_msrp"],
            centro["centrosymmetric"]["mean_msrp"],
        ]
        stds = [
            centro["non_centrosymmetric"]["std_msrp"],
            centro["centrosymmetric"]["std_msrp"],
        ]
        colors = ["#2166AC", "#B2182B"]

        ax.bar(groups, means, yerr=stds, capsize=5, color=colors,
               alpha=0.8, edgecolor="white", width=0.5)

        p_val = centro.get("mann_whitney_u", {}).get("p_value", 1.0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        y_max = max(m + s for m, s in zip(means, stds))
        ax.plot([0, 1], [y_max * 1.05, y_max * 1.05], "k-", linewidth=1)
        ax.text(0.5, y_max * 1.07, sig, ha="center", fontsize=12)

        ax.set_ylabel("MSRP Loss")
        ax.set_title("Centro vs Non-centrosymmetric")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "e1_centrosymmetry.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "e1_centrosymmetry.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        log.info(f"Saved e1_centrosymmetry.pdf")

    # Fig E1c: Scatter plot R(G,N) vs MSRP with correlation
    corr = results.get("correlations", {}).get("R_vs_msrp", {})
    scatter_data = results.get("_scatter_data", None)
    if scatter_data is not None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        R = scatter_data["R"]
        msrp = scatter_data["msrp"]

        # Subsample for plotting clarity
        n_plot = min(5000, len(R))
        idx = np.random.RandomState(42).choice(len(R), n_plot, replace=False)

        ax.scatter(R[idx], msrp[idx], s=4, alpha=0.3, c="#2166AC", edgecolors="none")

        rho = corr.get("spearman_rho", 0)
        p = corr.get("p_value", 1)
        ax.set_xlabel("R(G,N)")
        ax.set_ylabel("MSRP Loss")
        ax.set_title(f"E1: R(G,N) vs Reconstruction Error\n"
                     f"Spearman ρ = {rho:.3f}, p = {p:.2e}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "e1_scatter_rgn_msrp.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "e1_scatter_rgn_msrp.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        log.info(f"Saved e1_scatter_rgn_msrp.pdf")


def main():
    parser = argparse.ArgumentParser(description="E1: Symmetry-Stratified Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--h5-path", type=str,
                        default="data/raw/qm9s/qm9s_processed.h5",
                        help="Path to QM9S HDF5")
    parser.add_argument("--rgn-path", type=str,
                        default="experiments/results/qm9_rgn_data.npz",
                        help="Path to R(G,N) data")
    parser.add_argument("--pg-path", type=str,
                        default="experiments/results/qm9_point_groups.json",
                        help="Path to point group data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-mask-trials", type=int, default=5,
                        help="Number of mask trials per sample for robust estimate")
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e1_symmetry_stratification.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    log.info(f"Loading checkpoint: {args.checkpoint}")
    pretrain_model, config = load_pretrained(args.checkpoint, device)

    # Load data
    log.info(f"Loading QM9S test set from {args.h5_path}")
    _, _, test_loader = build_qm9s_loaders(
        args.h5_path, batch_size=args.batch_size, num_workers=4,
    )
    log.info(f"Test set: {len(test_loader.dataset)} samples")

    # Load symmetry metadata
    log.info("Loading symmetry metadata...")
    sym_meta = load_symmetry_metadata(args.rgn_path, args.pg_path)

    # Evaluate per sample
    log.info(f"Evaluating with {args.n_mask_trials} mask trials per sample...")
    t0 = time.time()
    eval_results = evaluate_per_sample(
        pretrain_model, config, test_loader, device,
        n_mask_trials=args.n_mask_trials,
    )
    log.info(f"Evaluation done in {time.time() - t0:.0f}s")

    msrp_losses = eval_results["msrp_loss"]
    mol_idx = eval_results["mol_idx"]

    # Map eval indices to symmetry metadata
    if mol_idx is not None:
        R_values = sym_meta["R"][mol_idx]
        N_values = sym_meta["N"][mol_idx]
        n_silent = sym_meta["n_silent"][mol_idx]
        is_centro = sym_meta["is_centro"][mol_idx]
    else:
        # Fallback: assume test set ordering matches
        n = len(msrp_losses)
        R_values = sym_meta["R"][:n]
        N_values = sym_meta["N"][:n]
        n_silent = sym_meta["n_silent"][:n]
        is_centro = sym_meta["is_centro"][:n]

    # Analyses
    log.info("Running stratified analyses...")
    results = {
        "experiment": "E1_symmetry_stratification",
        "n_samples": len(msrp_losses),
        "checkpoint": args.checkpoint,
        "n_mask_trials": args.n_mask_trials,
        "overall_msrp": {
            "mean": float(np.mean(msrp_losses)),
            "std": float(np.std(msrp_losses)),
            "median": float(np.median(msrp_losses)),
        },
    }

    # 1. R(G,N) stratification
    results["rgn_stratification"] = stratify_by_rgn(msrp_losses, R_values)

    # 2. Centrosymmetry analysis
    results["centrosymmetry"] = stratify_by_centrosymmetry(msrp_losses, is_centro)

    # 3. Correlations
    results["correlations"] = compute_correlations(
        msrp_losses, R_values, N_values, n_silent
    )

    # 4. Domain-wise analysis (IR vs Raman)
    domain_strs = eval_results.get("domain_str")
    if domain_strs:
        domain_results = {}
        for domain in set(domain_strs):
            mask = np.array([d == domain for d in domain_strs])
            if mask.sum() > 0:
                domain_results[domain] = {
                    "n": int(mask.sum()),
                    "mean_msrp": float(np.mean(msrp_losses[mask])),
                    "std_msrp": float(np.std(msrp_losses[mask])),
                }
        results["domain_stratification"] = domain_results

    # Print summary
    print("\n" + "=" * 70)
    print("E1: SYMMETRY-STRATIFIED EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall MSRP: {results['overall_msrp']['mean']:.4f} "
          f"(+/- {results['overall_msrp']['std']:.4f})")

    print("\nR(G,N) Stratification:")
    for bin_label, stats in results["rgn_stratification"].items():
        print(f"  {bin_label}: MSRP={stats['mean_msrp']:.4f} "
              f"(n={stats['n']})")

    corr = results["correlations"]["R_vs_msrp"]
    print(f"\nSpearman(R, MSRP): rho={corr['spearman_rho']:.4f}, "
          f"p={corr['p_value']:.2e}")

    centro = results["centrosymmetry"]
    if "centrosymmetric" in centro:
        print(f"\nCentrosymmetric: {centro['centrosymmetric']['mean_msrp']:.4f}")
        print(f"Non-centrosymmetric: {centro['non_centrosymmetric']['mean_msrp']:.4f}")
        mw = centro["mann_whitney_u"]
        print(f"Mann-Whitney U p={mw['p_value']:.2e}")

    # Save results (store scatter data for figures but not in JSON)
    results["_scatter_data"] = {"R": R_values, "msrp": msrp_losses}
    generate_figures(results, args.figures_dir)

    # Remove non-serializable data before saving JSON
    results.pop("_scatter_data", None)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
