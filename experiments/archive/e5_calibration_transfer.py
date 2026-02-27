#!/usr/bin/env python3
"""
E5: Calibration Transfer — Sample Efficiency & Comparison

The central experiment for the paper. Demonstrates that Spektron's
pretrained representations enable calibration transfer with far fewer
labeled samples than classical methods.

Protocol:
1. Load pretrained Spektron checkpoint
2. For each dataset (corn, tablet) and instrument pair:
   a) Freeze backbone, fine-tune LoRA + regression head
   b) Sweep N_transfer = [5, 10, 20, 30, 50] with 5 random seeds
   c) Evaluate R², RMSEP, RPD on held-out test set
3. Compare against classical baselines:
   - No Transfer (PLS on source, test on target)
   - PDS + PLS (Wang et al. 1991)
   - SBC + PLS (slope/bias correction)
   - DS + PLS (direct standardization)
   - CCA + PLS (canonical correlation)
   - di-PLS (domain-invariant PLS, Nikzad-Langerodi 2021)
4. Optionally: TTT zero-shot + TTT+LoRA hybrid
5. Generate sample efficiency curves (Fig 5 in paper)

Target: Beat LoRA-CT (R² = 0.952 on corn moisture) with <= 10 samples.

Output:
- experiments/results/e5_calibration_transfer.json
- figures/e5_sample_efficiency_corn.pdf
- figures/e5_sample_efficiency_tablet.pdf
- figures/e5_comparison_table.tex
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import SpectralFMConfig
from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
from src.data.datasets import SpectralPreprocessor
from src.evaluation.baselines import (
    PLSCalibration, PDS, SBC, DS, CCA, DiPLS,
    compute_metrics, run_baseline_comparison, print_results_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ============================================================
# Data Loading
# ============================================================

def load_corn_data(data_dir: str, target_length: int = 2048):
    """Load and preprocess corn dataset.

    Returns dict with per-instrument spectra and properties.
    """
    corn_dir = Path(data_dir) / "processed" / "corn"
    prep = SpectralPreprocessor(target_length=target_length)

    wavelengths = np.load(corn_dir / "wavelengths.npy")
    properties = np.load(corn_dir / "properties.npy")  # (80, 4)

    instruments = {}
    for inst in ["m5", "mp5", "mp6"]:
        raw = np.load(corn_dir / f"{inst}_spectra.npy")  # (80, 700)
        processed = np.array([
            prep.process(s, wavelengths)["normalized"] for s in raw
        ])
        instruments[inst] = processed

    return {
        "instruments": instruments,
        "properties": properties,
        "property_names": ["moisture", "oil", "protein", "starch"],
        "n_samples": 80,
    }


def load_tablet_data(data_dir: str, target_length: int = 2048):
    """Load and preprocess tablet dataset."""
    tablet_dir = Path(data_dir) / "processed" / "tablet"
    prep = SpectralPreprocessor(target_length=target_length)

    def preprocess_array(arr):
        return np.array([prep.process(s)["normalized"] for s in arr])

    cal_1 = preprocess_array(np.load(tablet_dir / "calibrate_1.npy"))
    cal_2 = preprocess_array(np.load(tablet_dir / "calibrate_2.npy"))
    cal_Y = np.load(tablet_dir / "calibrate_Y.npy")

    test_1 = preprocess_array(np.load(tablet_dir / "test_1.npy"))
    test_2 = preprocess_array(np.load(tablet_dir / "test_2.npy"))
    test_Y = np.load(tablet_dir / "test_Y.npy")

    val_1 = preprocess_array(np.load(tablet_dir / "validate_1.npy"))
    val_2 = preprocess_array(np.load(tablet_dir / "validate_2.npy"))
    val_Y = np.load(tablet_dir / "validate_Y.npy")

    return {
        "calibrate": {"spec_1": cal_1, "spec_2": cal_2, "Y": cal_Y},
        "test": {"spec_1": test_1, "spec_2": test_2, "Y": test_Y},
        "validate": {"spec_1": val_1, "spec_2": val_2, "Y": val_Y},
        "property_names": ["active_ingredient", "weight", "hardness"],
    }


# ============================================================
# Spektron Fine-tuning
# ============================================================

def load_pretrained_model(checkpoint_path: str, device: str = "cuda"):
    """Load pretrained Spektron and return the base SpectralFM model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = SpectralFM(config)
    pretrain_model = SpectralFMForPretraining(model, config)

    state_dict = ckpt["model_state_dict"]
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    pretrain_model.load_state_dict(clean_state)

    log.info(f"Loaded checkpoint from step {ckpt['step']}")
    return model, config


def finetune_and_evaluate(model, config, source_train, target_train, y_train,
                          target_test, y_test, device="cuda",
                          n_epochs=100, lr=1e-4, patience=20,
                          freeze_backbone=True):
    """Fine-tune Spektron for calibration transfer and evaluate.

    Uses the source instrument spectra for encoding (pretrained representations)
    and the regression head for prediction.

    Args:
        model: SpectralFM model (will be deep-copied)
        source_train: (N_train, L) source instrument spectra
        target_train: (N_train, L) target instrument spectra (for TTT, unused in basic LoRA)
        y_train: (N_train,) property values
        target_test: (N_test, L) target instrument test spectra
        y_test: (N_test,) test property values
    """
    m = copy.deepcopy(model)
    m.to(device)

    if freeze_backbone:
        m.freeze_backbone()

    # Build data loaders
    train_X = torch.tensor(source_train, dtype=torch.float32)
    train_Y = torch.tensor(y_train, dtype=torch.float32)
    test_X = torch.tensor(target_test, dtype=torch.float32)
    test_Y = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_ds, batch_size=min(16, len(train_ds)),
                              shuffle=True, drop_last=False)

    # Optimizer
    params = [p for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    # Training loop
    for epoch in range(n_epochs):
        m.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            enc = m.encode(batch_X, "NIR")
            pred = m.regression_head(enc["z_chem"]).squeeze(-1)
            loss = F.mse_loss(pred, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= max(n_batches, 1)

        # Early stopping on training loss (small datasets have no val split)
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best
    if best_state is not None:
        m.load_state_dict(best_state)

    # Evaluate on test set
    m.eval()
    with torch.no_grad():
        test_X_dev = test_X.to(device)
        enc = m.encode(test_X_dev, "NIR")
        preds = m.regression_head(enc["z_chem"]).squeeze(-1).cpu().numpy()

    metrics = compute_metrics(y_test, preds)
    m.unfreeze_all()

    return metrics, preds


def finetune_with_ttt(model, config, target_unlabeled, source_train,
                      target_train, y_train, target_test, y_test,
                      device="cuda", ttt_steps=20, ttt_lr=1e-4,
                      n_epochs=100, lr=1e-4, patience=20):
    """TTT + LoRA fine-tuning: first adapt with TTT, then fine-tune."""
    m = copy.deepcopy(model)
    m.to(device)

    # Step 1: TTT on unlabeled target spectra
    target_tensor = torch.tensor(target_unlabeled, dtype=torch.float32).to(device)
    m.test_time_train(target_tensor, n_steps=ttt_steps, lr=ttt_lr,
                      mask_ratio=config.ttt.mask_ratio)

    # Step 2: LoRA fine-tune
    m.freeze_backbone()
    train_X = torch.tensor(source_train, dtype=torch.float32)
    train_Y = torch.tensor(y_train, dtype=torch.float32)
    test_X = torch.tensor(target_test, dtype=torch.float32)

    train_ds = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_ds, batch_size=min(16, len(train_ds)),
                              shuffle=True, drop_last=False)

    params = [p for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        m.train()
        epoch_loss = 0.0
        n = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            enc = m.encode(bx, "NIR")
            pred = m.regression_head(enc["z_chem"]).squeeze(-1)
            loss = F.mse_loss(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        epoch_loss /= max(n, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        m.load_state_dict(best_state)

    m.eval()
    with torch.no_grad():
        enc = m.encode(test_X.to(device), "NIR")
        preds = m.regression_head(enc["z_chem"]).squeeze(-1).cpu().numpy()

    metrics = compute_metrics(y_test, preds)
    m.unfreeze_all()
    return metrics, preds


# ============================================================
# Experiment Runner
# ============================================================

def run_corn_experiment(model, config, corn_data, device="cuda",
                        n_transfer_list=None, n_seeds=5, use_ttt=False):
    """Run calibration transfer on corn dataset.

    Tests all instrument pairs: m5→mp5, m5→mp6, mp5→mp6, etc.
    """
    if n_transfer_list is None:
        n_transfer_list = [5, 10, 20, 30, 50]

    instruments = corn_data["instruments"]
    properties = corn_data["properties"]
    prop_names = corn_data["property_names"]

    results = {}

    # All directional pairs
    inst_pairs = [
        ("m5", "mp5"), ("m5", "mp6"), ("mp5", "mp6"),
        ("mp5", "m5"), ("mp6", "m5"), ("mp6", "mp5"),
    ]

    for source_inst, target_inst in inst_pairs:
        for prop_idx, prop_name in enumerate(prop_names):
            exp_key = f"corn_{source_inst}_to_{target_inst}_{prop_name}"
            log.info(f"\n{'='*60}")
            log.info(f"Experiment: {exp_key}")
            log.info(f"{'='*60}")

            source_spectra = instruments[source_inst]  # (80, 2048)
            target_spectra = instruments[target_inst]
            y = properties[:, prop_idx]

            # Train/test split (60/20 for transfer, 20 for test)
            rng = np.random.RandomState(42)
            n = len(y)
            idx = rng.permutation(n)
            n_pool = 60  # max transfer pool
            n_test = n - n_pool
            pool_idx = idx[:n_pool]
            test_idx = idx[n_pool:]

            # Classical baselines (at max n_transfer)
            max_n = min(n_pool, max(n_transfer_list))
            bl_train_idx = pool_idx[:max_n]

            baselines = run_baseline_comparison(
                source_spectra[bl_train_idx],
                target_spectra[bl_train_idx],
                source_spectra[test_idx],
                target_spectra[test_idx],
                y[bl_train_idx],
                y[test_idx],
            )
            log.info("Classical baselines:")
            for method, metrics in baselines.items():
                log.info(f"  {method}: R²={metrics['r2']:.4f}, "
                         f"RMSEP={metrics['rmsep']:.4f}")

            # Spektron sweep over n_transfer
            sweep = {}
            for n_transfer in n_transfer_list:
                seed_results = []
                for seed in range(n_seeds):
                    rng_seed = np.random.RandomState(42 + seed)
                    transfer_idx = rng_seed.choice(pool_idx, n_transfer, replace=False)

                    metrics, _ = finetune_and_evaluate(
                        model, config,
                        source_spectra[transfer_idx],
                        target_spectra[transfer_idx],
                        y[transfer_idx],
                        target_spectra[test_idx],
                        y[test_idx],
                        device=device,
                    )
                    seed_results.append(metrics)

                # Aggregate over seeds
                sweep[n_transfer] = {
                    "r2_mean": float(np.mean([r["r2"] for r in seed_results])),
                    "r2_std": float(np.std([r["r2"] for r in seed_results])),
                    "rmsep_mean": float(np.mean([r["rmsep"] for r in seed_results])),
                    "rmsep_std": float(np.std([r["rmsep"] for r in seed_results])),
                    "rpd_mean": float(np.mean([r["rpd"] for r in seed_results])),
                    "rpd_std": float(np.std([r["rpd"] for r in seed_results])),
                }
                log.info(f"  N={n_transfer}: R²={sweep[n_transfer]['r2_mean']:.4f} "
                         f"(+/- {sweep[n_transfer]['r2_std']:.4f})")

            # TTT sweep (optional)
            ttt_sweep = {}
            if use_ttt:
                for n_transfer in n_transfer_list:
                    seed_results = []
                    for seed in range(n_seeds):
                        rng_seed = np.random.RandomState(42 + seed)
                        transfer_idx = rng_seed.choice(pool_idx, n_transfer, replace=False)

                        metrics, _ = finetune_with_ttt(
                            model, config,
                            target_spectra[pool_idx],  # unlabeled target for TTT
                            source_spectra[transfer_idx],
                            target_spectra[transfer_idx],
                            y[transfer_idx],
                            target_spectra[test_idx],
                            y[test_idx],
                            device=device,
                        )
                        seed_results.append(metrics)

                    ttt_sweep[n_transfer] = {
                        "r2_mean": float(np.mean([r["r2"] for r in seed_results])),
                        "r2_std": float(np.std([r["r2"] for r in seed_results])),
                        "rmsep_mean": float(np.mean([r["rmsep"] for r in seed_results])),
                        "rmsep_std": float(np.std([r["rmsep"] for r in seed_results])),
                    }

            results[exp_key] = {
                "baselines": baselines,
                "spektron_sweep": {str(k): v for k, v in sweep.items()},
                "ttt_sweep": {str(k): v for k, v in ttt_sweep.items()} if ttt_sweep else None,
            }

    return results


def run_tablet_experiment(model, config, tablet_data, device="cuda",
                          n_transfer_list=None, n_seeds=5, use_ttt=False):
    """Run calibration transfer on tablet dataset.

    Tablet has explicit calibrate/test/validate splits and 2 instruments.
    """
    if n_transfer_list is None:
        n_transfer_list = [5, 10, 20, 30, 50]

    cal = tablet_data["calibrate"]
    test = tablet_data["test"]
    prop_names = tablet_data["property_names"]

    results = {}

    # Direction: spec_1 → spec_2 and spec_2 → spec_1
    for source_key, target_key in [("spec_1", "spec_2"), ("spec_2", "spec_1")]:
        for prop_idx, prop_name in enumerate(prop_names):
            exp_key = f"tablet_{source_key}_to_{target_key}_{prop_name}"
            log.info(f"\n{'='*60}")
            log.info(f"Experiment: {exp_key}")
            log.info(f"{'='*60}")

            source_cal = cal[source_key]   # (155, 2048)
            target_cal = cal[target_key]
            y_cal = cal["Y"][:, prop_idx]

            source_test = test[source_key]  # (460, 2048)
            target_test_sp = test[target_key]
            y_test = test["Y"][:, prop_idx]

            # Classical baselines
            max_n = min(len(y_cal), max(n_transfer_list))
            rng = np.random.RandomState(42)
            bl_idx = rng.choice(len(y_cal), max_n, replace=False)

            baselines = run_baseline_comparison(
                source_cal[bl_idx], target_cal[bl_idx],
                source_test, target_test_sp,
                y_cal[bl_idx], y_test,
            )
            log.info("Classical baselines:")
            for method, metrics in baselines.items():
                log.info(f"  {method}: R²={metrics['r2']:.4f}")

            # Spektron sweep
            sweep = {}
            for n_transfer in n_transfer_list:
                seed_results = []
                for seed in range(n_seeds):
                    rng_seed = np.random.RandomState(42 + seed)
                    idx = rng_seed.choice(len(y_cal), n_transfer, replace=False)

                    metrics, _ = finetune_and_evaluate(
                        model, config,
                        source_cal[idx], target_cal[idx], y_cal[idx],
                        target_test_sp, y_test,
                        device=device,
                    )
                    seed_results.append(metrics)

                sweep[n_transfer] = {
                    "r2_mean": float(np.mean([r["r2"] for r in seed_results])),
                    "r2_std": float(np.std([r["r2"] for r in seed_results])),
                    "rmsep_mean": float(np.mean([r["rmsep"] for r in seed_results])),
                    "rmsep_std": float(np.std([r["rmsep"] for r in seed_results])),
                    "rpd_mean": float(np.mean([r["rpd"] for r in seed_results])),
                    "rpd_std": float(np.std([r["rpd"] for r in seed_results])),
                }
                log.info(f"  N={n_transfer}: R²={sweep[n_transfer]['r2_mean']:.4f} "
                         f"(+/- {sweep[n_transfer]['r2_std']:.4f})")

            results[exp_key] = {
                "baselines": baselines,
                "spektron_sweep": {str(k): v for k, v in sweep.items()},
            }

    return results


# ============================================================
# Visualization
# ============================================================

def generate_sample_efficiency_figure(results, dataset_name, figures_dir="figures"):
    """Generate sample efficiency curve for one dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available")
        return

    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Find the primary experiment (e.g., corn_m5_to_mp6_moisture)
    primary_keys = [k for k in results if "moisture" in k or "active_ingredient" in k]
    if not primary_keys:
        primary_keys = list(results.keys())[:1]

    for exp_key in primary_keys:
        exp = results[exp_key]
        sweep = exp.get("spektron_sweep", {})
        baselines = exp.get("baselines", {})

        if not sweep:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

        # Spektron curve
        ns = sorted([int(k) for k in sweep.keys()])
        means = [sweep[str(n)]["r2_mean"] for n in ns]
        stds = [sweep[str(n)]["r2_std"] for n in ns]
        ax.errorbar(ns, means, yerr=stds, label="Spektron (LoRA)",
                    color="#2166AC", marker="o", capsize=3, linewidth=2)

        # TTT curve
        ttt_sweep = exp.get("ttt_sweep")
        if ttt_sweep:
            ns_ttt = sorted([int(k) for k in ttt_sweep.keys()])
            ttt_means = [ttt_sweep[str(n)]["r2_mean"] for n in ns_ttt]
            ttt_stds = [ttt_sweep[str(n)]["r2_std"] for n in ns_ttt]
            ax.errorbar(ns_ttt, ttt_means, yerr=ttt_stds,
                        label="Spektron (TTT+LoRA)", color="#053061",
                        marker="s", capsize=3, linewidth=2, linestyle="--")

        # Baseline horizontal lines
        baseline_colors = {
            "DS": "#B2182B", "PDS": "#EF8A62", "SBC": "#67A9CF",
            "CCA": "#D6604D", "di-PLS": "#FDDBC7", "No_Transfer": "#999999",
            "Target_Direct": "#1B7837",
        }
        for method, metrics in baselines.items():
            r2 = metrics.get("r2", float("nan"))
            if np.isnan(r2):
                continue
            color = baseline_colors.get(method, "#666666")
            if method == "Target_Direct":
                ax.axhline(y=r2, color=color, linestyle="-.", alpha=0.5, linewidth=1)
                ax.text(max(ns) * 0.95, r2 + 0.02, "Upper bound",
                        fontsize=8, color=color, ha="right")
            elif method != "No_Transfer":
                ax.axhline(y=r2, color=color, linestyle=":", alpha=0.7, linewidth=1)
                ax.text(max(ns) * 0.95, r2 + 0.01, method,
                        fontsize=8, color=color, ha="right", va="bottom")

        ax.set_xlabel("Number of transfer samples")
        ax.set_ylabel("R²")
        title = exp_key.replace("_", " ").title()
        ax.set_title(f"Sample Efficiency: {title}")
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = f"e5_sample_efficiency_{exp_key}"
        fig.savefig(fig_dir / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / f"{fname}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        log.info(f"Saved {fname}.pdf")


def generate_latex_table(all_results, output_path="figures/e5_comparison_table.tex"):
    """Generate LaTeX comparison table."""
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Calibration transfer results (R\textsuperscript{2}) at N=20 transfer samples. "
        r"Mean $\pm$ std over 5 random seeds. Best in \textbf{bold}.}",
        r"\label{tab:calibration_transfer}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Dataset & Property & No Transfer & SBC & PDS & DS & CCA & di-PLS & Spektron & Spektron+TTT \\",
        r"\midrule",
    ]

    for exp_key, exp in all_results.items():
        baselines = exp.get("baselines", {})
        sweep = exp.get("spektron_sweep", {})
        ttt = exp.get("ttt_sweep", {})

        # Extract dataset and property
        parts = exp_key.split("_")
        dataset = parts[0]
        prop = parts[-1]

        # Collect R² values
        vals = {}
        for method in ["No_Transfer", "SBC", "PDS", "DS", "CCA", "di-PLS"]:
            v = baselines.get(method, {}).get("r2", float("nan"))
            vals[method] = v

        # Spektron at N=20
        sp20 = sweep.get("20", {})
        vals["Spektron"] = sp20.get("r2_mean", float("nan"))
        sp20_std = sp20.get("r2_std", 0)

        ttt20 = ttt.get("20", {}) if ttt else {}
        vals["Spektron+TTT"] = ttt20.get("r2_mean", float("nan"))

        # Find best
        best_val = max(v for v in vals.values() if not np.isnan(v))

        # Format row
        cells = [f"{dataset}", f"{prop}"]
        for method in ["No_Transfer", "SBC", "PDS", "DS", "CCA", "di-PLS", "Spektron", "Spektron+TTT"]:
            v = vals.get(method, float("nan"))
            if np.isnan(v):
                cells.append("---")
            elif method == "Spektron":
                cell = f"{v:.3f}$\\pm${sp20_std:.3f}"
                if abs(v - best_val) < 1e-4:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            else:
                cell = f"{v:.3f}"
                if abs(v - best_val) < 1e-4:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)

        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"LaTeX table saved to {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="E5: Calibration Transfer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-transfer", type=str, default="5,10,20,30,50",
                        help="Comma-separated list of transfer sample counts")
    parser.add_argument("--use-ttt", action="store_true",
                        help="Also run TTT+LoRA experiments")
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["corn", "tablet", "both"])
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e5_calibration_transfer.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_transfer_list = [int(x) for x in args.n_transfer.split(",")]

    # Load model
    model, config = load_pretrained_model(args.checkpoint, device)

    all_results = {}

    # Corn experiments
    if args.dataset in ("corn", "both"):
        log.info("Loading corn dataset...")
        corn_data = load_corn_data(args.data_dir, config.n_channels)
        log.info(f"Corn: {corn_data['n_samples']} samples, "
                 f"{len(corn_data['instruments'])} instruments")

        corn_results = run_corn_experiment(
            model, config, corn_data, device,
            n_transfer_list, args.n_seeds, args.use_ttt,
        )
        all_results.update(corn_results)
        generate_sample_efficiency_figure(corn_results, "corn", args.figures_dir)

    # Tablet experiments
    if args.dataset in ("tablet", "both"):
        log.info("Loading tablet dataset...")
        tablet_data = load_tablet_data(args.data_dir, config.n_channels)
        log.info(f"Tablet: calibrate={len(tablet_data['calibrate']['Y'])}, "
                 f"test={len(tablet_data['test']['Y'])}")

        tablet_results = run_tablet_experiment(
            model, config, tablet_data, device,
            n_transfer_list, args.n_seeds, args.use_ttt,
        )
        all_results.update(tablet_results)
        generate_sample_efficiency_figure(tablet_results, "tablet", args.figures_dir)

    # Generate LaTeX table
    generate_latex_table(all_results, f"{args.figures_dir}/e5_comparison_table.tex")

    # Print summary
    print("\n" + "=" * 80)
    print("E5: CALIBRATION TRANSFER SUMMARY")
    print("=" * 80)
    for exp_key, exp in all_results.items():
        sweep = exp.get("spektron_sweep", {})
        baselines = exp.get("baselines", {})
        print(f"\n{exp_key}:")
        for method in ["No_Transfer", "PDS", "DS", "CCA", "di-PLS", "Target_Direct"]:
            if method in baselines:
                print(f"  {method:>15}: R²={baselines[method]['r2']:.4f}")
        for n_str, data in sorted(sweep.items(), key=lambda x: int(x[0])):
            print(f"  Spektron (N={n_str:>3}): R²={data['r2_mean']:.4f} "
                  f"(+/- {data['r2_std']:.4f})")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "E5_calibration_transfer",
            "checkpoint": args.checkpoint,
            "n_seeds": args.n_seeds,
            "n_transfer_list": n_transfer_list,
            "results": all_results,
        }, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
