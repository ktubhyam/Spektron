#!/usr/bin/env python3
"""
E2: Modal Complementarity

Validates Theorem 2: for centrosymmetric molecules, IR and Raman observe
disjoint mode sets (mutual exclusion rule). Combined IR+Raman should
strictly increase information content and reconstruction quality.

Protocol:
1. Load pretrained Spektron checkpoint
2. Evaluate reconstruction on QM9S test set with three input conditions:
   a) IR-only (mask all Raman)
   b) Raman-only (mask all IR)
   c) IR + Raman combined
3. Compute per-molecule: MSRP loss for each condition
4. Stratify by centrosymmetry:
   - For centrosymmetric: combined should be significantly better (mutual exclusion)
   - For non-centrosymmetric: improvement should be smaller (modes overlap)
5. Compute complementarity gain: delta = MSRP(single) - MSRP(combined)
6. Statistical tests: paired Wilcoxon signed-rank, effect size (Cohen's d)

Output:
- experiments/results/e2_modal_complementarity.json
- figures/e2_complementarity_centro.pdf
- figures/e2_complementarity_heatmap.pdf
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
from scipy.stats import wilcoxon, mannwhitneyu

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

    state_dict = ckpt["model_state_dict"]
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    pretrain_model.load_state_dict(clean_state)
    pretrain_model.to(device)
    pretrain_model.eval()
    log.info(f"Loaded checkpoint from step {ckpt['step']}")
    return pretrain_model, config


@torch.no_grad()
def evaluate_modality(model, config, test_loader, modality="combined",
                      device="cuda", n_mask_trials=3):
    """Evaluate reconstruction quality for a specific modality.

    For QM9S, each sample has a domain label ('IR' or 'Raman').
    We filter the test set to get IR-only, Raman-only, or use all samples.

    For 'combined' analysis at the molecule level, we average per-molecule losses.

    Args:
        modality: 'IR', 'Raman', or 'combined'
    """
    model.eval()
    all_losses = []
    all_mol_idx = []
    all_domains = []

    for batch in test_loader:
        spectrum = batch["spectrum"].to(device)
        instrument_id = batch.get("instrument_id")
        if instrument_id is not None:
            instrument_id = instrument_id.to(device)

        domain = batch.get("domain", "IR")
        if isinstance(domain, list):
            domain_list = domain
        else:
            domain_list = [domain] * spectrum.size(0)

        # Filter by modality
        if modality in ("IR", "Raman"):
            keep = [i for i, d in enumerate(domain_list) if d == modality]
            if not keep:
                continue
            keep_idx = torch.tensor(keep, device=device)
            spectrum = spectrum[keep_idx]
            if instrument_id is not None:
                instrument_id = instrument_id[keep_idx]
            domain_list = [domain_list[i] for i in keep]
            if "mol_idx" in batch:
                mol_idx_batch = batch["mol_idx"].numpy()[keep]
            else:
                mol_idx_batch = None
        else:
            mol_idx_batch = batch.get("mol_idx")
            if mol_idx_batch is not None:
                mol_idx_batch = mol_idx_batch.numpy()

        B = spectrum.size(0)
        if B == 0:
            continue

        seq_len = config.seq_len

        _dmap = {"NIR": 0, "IR": 1, "RAMAN": 2, "UNKNOWN": 3}
        domain_tensor = torch.tensor(
            [_dmap.get(d, 3) for d in domain_list],
            dtype=torch.long, device=device,
        )

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
            diff = (output["reconstruction"] - target) * mask_expanded
            per_sample = (diff ** 2).sum(dim=(1, 2)) / (
                mask.sum(dim=1) * target.size(-1) + 1e-8
            )
            sample_losses += per_sample

        sample_losses /= n_mask_trials
        all_losses.append(sample_losses.cpu().numpy())
        all_domains.extend(domain_list)
        if mol_idx_batch is not None:
            all_mol_idx.append(mol_idx_batch)

    return {
        "losses": np.concatenate(all_losses) if all_losses else np.array([]),
        "mol_idx": np.concatenate(all_mol_idx) if all_mol_idx else None,
        "domains": all_domains,
    }


def compute_complementarity(ir_results, raman_results, combined_results,
                            sym_meta):
    """Compute complementarity metrics stratified by centrosymmetry.

    For molecules that appear in both IR and Raman test sets, compare:
    - min(MSRP_IR, MSRP_Raman) vs MSRP_combined
    """
    # Build per-molecule loss maps
    ir_mol_loss = {}
    if ir_results["mol_idx"] is not None:
        for idx, loss in zip(ir_results["mol_idx"], ir_results["losses"]):
            ir_mol_loss[int(idx)] = loss

    raman_mol_loss = {}
    if raman_results["mol_idx"] is not None:
        for idx, loss in zip(raman_results["mol_idx"], raman_results["losses"]):
            raman_mol_loss[int(idx)] = loss

    # Combined: average IR and Raman losses per molecule
    combined_mol_loss = {}
    if combined_results["mol_idx"] is not None:
        mol_losses = defaultdict(list)
        for idx, loss in zip(combined_results["mol_idx"], combined_results["losses"]):
            mol_losses[int(idx)].append(loss)
        for mol_id, losses in mol_losses.items():
            combined_mol_loss[mol_id] = np.mean(losses)

    # Find molecules with all three
    common_mols = set(ir_mol_loss.keys()) & set(raman_mol_loss.keys()) & set(combined_mol_loss.keys())
    log.info(f"Molecules with IR+Raman+Combined: {len(common_mols)}")

    if len(common_mols) < 10:
        log.warning("Too few common molecules for complementarity analysis")
        return {"error": "Insufficient common molecules"}

    common_list = sorted(common_mols)
    ir_losses = np.array([ir_mol_loss[m] for m in common_list])
    raman_losses = np.array([raman_mol_loss[m] for m in common_list])
    comb_losses = np.array([combined_mol_loss[m] for m in common_list])
    best_single = np.minimum(ir_losses, raman_losses)

    # Get centrosymmetry
    is_centro = sym_meta["is_centro"][np.array(common_list)]

    # Complementarity gain: how much does combining help vs best single?
    gain = best_single - comb_losses  # positive = combined is better

    results = {
        "n_common_molecules": len(common_list),
        "overall": {
            "ir_mean": float(np.mean(ir_losses)),
            "raman_mean": float(np.mean(raman_losses)),
            "combined_mean": float(np.mean(comb_losses)),
            "best_single_mean": float(np.mean(best_single)),
            "gain_mean": float(np.mean(gain)),
            "gain_std": float(np.std(gain)),
            "fraction_improved": float(np.mean(gain > 0)),
        },
    }

    centro = is_centro.astype(bool)
    for label, mask in [("centrosymmetric", centro), ("non_centrosymmetric", ~centro)]:
        if mask.sum() < 5:
            results[label] = {"n": int(mask.sum()), "error": "Too few samples"}
            continue

        g = gain[mask]
        results[label] = {
            "n": int(mask.sum()),
            "ir_mean": float(np.mean(ir_losses[mask])),
            "raman_mean": float(np.mean(raman_losses[mask])),
            "combined_mean": float(np.mean(comb_losses[mask])),
            "gain_mean": float(np.mean(g)),
            "gain_std": float(np.std(g)),
            "gain_median": float(np.median(g)),
            "fraction_improved": float(np.mean(g > 0)),
        }

        # Wilcoxon signed-rank test: is gain > 0?
        try:
            stat, p = wilcoxon(g, alternative="greater")
            results[label]["wilcoxon_p"] = float(p)
            results[label]["wilcoxon_stat"] = float(stat)
        except ValueError:
            results[label]["wilcoxon_p"] = None

        # Cohen's d effect size
        if np.std(g) > 1e-10:
            results[label]["cohens_d"] = float(np.mean(g) / np.std(g))

    # Compare gains: centro vs non-centro
    if centro.sum() > 5 and (~centro).sum() > 5:
        stat, p = mannwhitneyu(
            gain[centro], gain[~centro], alternative="greater"
        )
        results["gain_comparison"] = {
            "hypothesis": "centro gain > non-centro gain (Theorem 2: mutual exclusion → greater complementarity)",
            "mann_whitney_u": float(stat),
            "p_value": float(p),
        }

    return results


def generate_figures(results, figures_dir="figures"):
    """Generate publication figures for E2."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping figures")
        return

    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    comp = results.get("complementarity", {})
    if "error" in comp:
        return

    # Fig E2a: Grouped bar chart — IR vs Raman vs Combined, stratified by centro
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    groups = []
    ir_vals, raman_vals, comb_vals = [], [], []

    for label in ["non_centrosymmetric", "centrosymmetric"]:
        data = comp.get(label, {})
        if "ir_mean" in data:
            groups.append("Non-centro" if label == "non_centrosymmetric" else "Centro")
            ir_vals.append(data["ir_mean"])
            raman_vals.append(data["raman_mean"])
            comb_vals.append(data["combined_mean"])

    if groups:
        x = np.arange(len(groups))
        width = 0.25
        ax.bar(x - width, ir_vals, width, label="IR only", color="#EF8A62", alpha=0.8)
        ax.bar(x, raman_vals, width, label="Raman only", color="#67A9CF", alpha=0.8)
        ax.bar(x + width, comb_vals, width, label="IR + Raman", color="#2166AC", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel("MSRP Loss (lower = better)")
        ax.set_title("E2: Modal Complementarity by Centrosymmetry")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "e2_complementarity_bars.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "e2_complementarity_bars.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        log.info("Saved e2_complementarity_bars.pdf")

    # Fig E2b: Gain distribution — centro vs non-centro
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for label, color, name in [
        ("centrosymmetric", "#B2182B", "Centro"),
        ("non_centrosymmetric", "#2166AC", "Non-centro"),
    ]:
        data = comp.get(label, {})
        if "gain_mean" in data:
            # Plot gain distribution as a summary point with error bar
            ax.errorbar(
                [name], [data["gain_mean"]], yerr=[data["gain_std"]],
                fmt="o", color=color, capsize=6, markersize=10, linewidth=2,
            )
            n = data.get("n", "?")
            p = data.get("wilcoxon_p", None)
            p_str = f"p={p:.2e}" if p is not None else ""
            ax.annotate(f"n={n}\n{p_str}", (name, data["gain_mean"]),
                       textcoords="offset points", xytext=(30, 0),
                       fontsize=8, ha="left")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Complementarity Gain (best_single - combined MSRP)")
    ax.set_title("E2: Complementarity Gain\n(Positive = combined helps)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "e2_complementarity_gain.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "e2_complementarity_gain.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("Saved e2_complementarity_gain.pdf")


def main():
    parser = argparse.ArgumentParser(description="E2: Modal Complementarity")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--h5-path", type=str,
                        default="data/raw/qm9s/qm9s_processed.h5")
    parser.add_argument("--rgn-path", type=str,
                        default="experiments/results/qm9_rgn_data.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-mask-trials", type=int, default=3)
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e2_modal_complementarity.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    pretrain_model, config = load_pretrained(args.checkpoint, device)

    # Load data
    _, _, test_loader = build_qm9s_loaders(
        args.h5_path, batch_size=args.batch_size, num_workers=4,
    )
    log.info(f"Test set: {len(test_loader.dataset)} samples")

    # Load symmetry metadata
    sym_meta = np.load(args.rgn_path)

    # Evaluate each modality
    log.info("Evaluating IR-only...")
    t0 = time.time()
    ir_results = evaluate_modality(
        pretrain_model, config, test_loader, "IR", device, args.n_mask_trials
    )
    log.info(f"  IR: {len(ir_results['losses'])} samples, "
             f"mean MSRP={np.mean(ir_results['losses']):.4f}")

    log.info("Evaluating Raman-only...")
    raman_results = evaluate_modality(
        pretrain_model, config, test_loader, "Raman", device, args.n_mask_trials
    )
    log.info(f"  Raman: {len(raman_results['losses'])} samples, "
             f"mean MSRP={np.mean(raman_results['losses']):.4f}")

    log.info("Evaluating combined...")
    combined_results = evaluate_modality(
        pretrain_model, config, test_loader, "combined", device, args.n_mask_trials
    )
    log.info(f"  Combined: {len(combined_results['losses'])} samples, "
             f"mean MSRP={np.mean(combined_results['losses']):.4f}")

    elapsed = time.time() - t0
    log.info(f"All evaluations done in {elapsed:.0f}s")

    # Complementarity analysis
    log.info("Computing complementarity metrics...")
    comp_results = compute_complementarity(
        ir_results, raman_results, combined_results, sym_meta
    )

    # Assemble output
    results = {
        "experiment": "E2_modal_complementarity",
        "checkpoint": args.checkpoint,
        "n_mask_trials": args.n_mask_trials,
        "modality_summary": {
            "IR": {
                "n_samples": len(ir_results["losses"]),
                "mean_msrp": float(np.mean(ir_results["losses"])) if len(ir_results["losses"]) > 0 else None,
            },
            "Raman": {
                "n_samples": len(raman_results["losses"]),
                "mean_msrp": float(np.mean(raman_results["losses"])) if len(raman_results["losses"]) > 0 else None,
            },
            "Combined": {
                "n_samples": len(combined_results["losses"]),
                "mean_msrp": float(np.mean(combined_results["losses"])) if len(combined_results["losses"]) > 0 else None,
            },
        },
        "complementarity": comp_results,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("E2: MODAL COMPLEMENTARITY RESULTS")
    print("=" * 70)
    for mod, data in results["modality_summary"].items():
        if data["mean_msrp"] is not None:
            print(f"  {mod:>10}: MSRP={data['mean_msrp']:.4f} (n={data['n_samples']})")

    if "error" not in comp_results:
        print(f"\nComplementarity ({comp_results['n_common_molecules']} molecules):")
        for label in ["centrosymmetric", "non_centrosymmetric"]:
            d = comp_results.get(label, {})
            if "gain_mean" in d:
                p = d.get("wilcoxon_p", "N/A")
                print(f"  {label}: gain={d['gain_mean']:.4f} +/- {d['gain_std']:.4f}, "
                      f"improved={d['fraction_improved']:.1%}, p={p}")

        gc = comp_results.get("gain_comparison", {})
        if gc:
            print(f"\nGain comparison (centro vs non-centro): p={gc['p_value']:.2e}")

    # Generate figures
    generate_figures(results, args.figures_dir)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
