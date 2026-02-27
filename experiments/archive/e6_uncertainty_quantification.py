#!/usr/bin/env python3
"""
E6: Uncertainty Quantification via Conformal Prediction

Provides distribution-free prediction intervals for calibration transfer
predictions using split conformal prediction + MC Dropout.

Protocol:
1. Load pretrained and fine-tuned Spektron model
2. On calibration set, compute nonconformity scores:
   |y_i - hat{y}_i| / sigma_i  (normalized residuals)
3. For target confidence levels (90%, 95%, 99%):
   - Compute conformal quantile q_alpha from calibration scores
   - Prediction interval: [hat{y} - q_alpha * sigma, hat{y} + q_alpha * sigma]
4. Evaluate on test set:
   - Empirical coverage (should match target)
   - Average interval width (narrower = better)
   - Conditional coverage by property range
5. Compare uncertainty methods:
   - MC Dropout (epistemic)
   - Conformal prediction (distribution-free)
   - MC Dropout + Conformal (hybrid)

Output:
- experiments/results/e6_uncertainty_quantification.json
- figures/e6_calibration_plot.pdf
- figures/e6_interval_width.pdf
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
from src.evaluation.baselines import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ============================================================
# MC Dropout Uncertainty
# ============================================================

@torch.no_grad()
def mc_dropout_predictions(model, spectra, device="cuda",
                           mc_samples=30, batch_size=64):
    """Get predictions with MC Dropout uncertainty estimates.

    Returns:
        means: (N,) mean predictions
        stds: (N,) epistemic uncertainty (std of MC samples)
        all_preds: (mc_samples, N) all predictions
    """
    model.to(device)
    model.train()  # Enable dropout

    n = len(spectra)
    all_preds = np.zeros((mc_samples, n))

    for mc in range(mc_samples):
        preds = []
        for i in range(0, n, batch_size):
            batch = torch.tensor(spectra[i:i+batch_size],
                                 dtype=torch.float32).to(device)
            enc = model.encode(batch, "NIR")
            z = model.mc_dropout(enc["z_chem"])
            pred = model.regression_head(z).squeeze(-1)
            preds.append(pred.cpu().numpy())
        all_preds[mc] = np.concatenate(preds)

    model.eval()

    means = all_preds.mean(axis=0)
    stds = all_preds.std(axis=0)
    return means, stds, all_preds


# ============================================================
# Conformal Prediction
# ============================================================

def compute_conformal_scores(y_true, y_pred, y_std):
    """Compute nonconformity scores for conformal prediction.

    Score: |y - hat{y}| / sigma  (normalized absolute residual)
    """
    residuals = np.abs(y_true - y_pred)
    # Avoid division by zero
    sigma = np.maximum(y_std, 1e-8)
    return residuals / sigma


def conformal_quantile(scores, alpha):
    """Compute the (1-alpha) quantile for split conformal prediction.

    Uses the correction factor (n+1)/n for finite-sample validity.
    """
    n = len(scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)
    return np.quantile(scores, level)


def conformal_prediction_intervals(y_pred, y_std, q_alpha):
    """Compute prediction intervals using conformal quantile.

    Interval: [hat{y} - q * sigma, hat{y} + q * sigma]
    """
    sigma = np.maximum(y_std, 1e-8)
    lower = y_pred - q_alpha * sigma
    upper = y_pred + q_alpha * sigma
    return lower, upper


def evaluate_coverage(y_true, lower, upper):
    """Evaluate empirical coverage and interval metrics."""
    covered = (y_true >= lower) & (y_true <= upper)
    widths = upper - lower

    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "max_width": float(np.max(widths)),
        "min_width": float(np.min(widths)),
        "width_std": float(np.std(widths)),
    }


# ============================================================
# Full UQ Pipeline
# ============================================================

def run_uq_experiment(model, config, source_cal, target_cal, y_cal,
                      target_test, y_test, device="cuda",
                      mc_samples=30, alphas=None,
                      n_transfer=30, seed=42):
    """Run full UQ pipeline.

    Steps:
    1. Fine-tune on n_transfer samples
    2. MC Dropout predictions on calibration holdout
    3. Compute conformal scores
    4. Apply to test set
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.10]  # 99%, 95%, 90% confidence

    # Split calibration into fine-tune and conformal calibration
    rng = np.random.RandomState(seed)
    n = len(y_cal)
    idx = rng.permutation(n)
    n_ft = min(n_transfer, n // 2)
    ft_idx = idx[:n_ft]
    conf_idx = idx[n_ft:]

    # Fine-tune
    m = copy.deepcopy(model)
    m.to(device)
    m.freeze_backbone()

    train_X = torch.tensor(source_cal[ft_idx], dtype=torch.float32)
    train_Y = torch.tensor(y_cal[ft_idx], dtype=torch.float32)

    train_ds = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_ds, batch_size=min(16, len(train_ds)),
                              shuffle=True)

    params = [p for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

    best_loss = float("inf")
    best_state = None

    for epoch in range(100):
        m.train()
        epoch_loss = 0.0
        nb = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            enc = m.encode(bx, "NIR")
            pred = m.regression_head(enc["z_chem"]).squeeze(-1)
            loss = F.mse_loss(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        epoch_loss /= max(nb, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in m.state_dict().items()}

    if best_state:
        m.load_state_dict(best_state)

    # MC Dropout on conformal calibration set
    log.info("  Computing MC Dropout on calibration holdout...")
    cal_means, cal_stds, _ = mc_dropout_predictions(
        m, target_cal[conf_idx], device, mc_samples
    )
    y_cal_holdout = y_cal[conf_idx]

    # Compute nonconformity scores
    scores = compute_conformal_scores(y_cal_holdout, cal_means, cal_stds)

    # MC Dropout on test set
    log.info("  Computing MC Dropout on test set...")
    test_means, test_stds, test_all_preds = mc_dropout_predictions(
        m, target_test, device, mc_samples
    )

    # Point prediction metrics
    point_metrics = compute_metrics(y_test, test_means)
    log.info(f"  Point prediction: R²={point_metrics['r2']:.4f}, "
             f"RMSEP={point_metrics['rmsep']:.4f}")

    # Conformal prediction at each alpha level
    conformal_results = {}
    for alpha in alphas:
        q = conformal_quantile(scores, alpha)
        lower, upper = conformal_prediction_intervals(test_means, test_stds, q)
        coverage_metrics = evaluate_coverage(y_test, lower, upper)

        conf_level = f"{100*(1-alpha):.0f}%"
        conformal_results[conf_level] = {
            "alpha": float(alpha),
            "target_coverage": float(1 - alpha),
            "q_alpha": float(q),
            **coverage_metrics,
        }
        log.info(f"  {conf_level}: coverage={coverage_metrics['coverage']:.3f} "
                 f"(target={1-alpha:.2f}), width={coverage_metrics['mean_width']:.4f}")

    # MC Dropout-only intervals (no conformal correction)
    mc_results = {}
    for z_score, conf_name in [(1.0, "68%"), (1.96, "95%"), (2.576, "99%")]:
        lower = test_means - z_score * test_stds
        upper = test_means + z_score * test_stds
        mc_results[conf_name] = evaluate_coverage(y_test, lower, upper)

    # Conditional coverage: split by property quantiles
    conditional_results = {}
    for q_low, q_high, label in [
        (0, 0.25, "Q1"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"), (0.75, 1.0, "Q4")
    ]:
        lo = np.quantile(y_test, q_low)
        hi = np.quantile(y_test, q_high)
        mask = (y_test >= lo) & (y_test <= hi)
        if mask.sum() < 5:
            continue
        q95 = conformal_quantile(scores, 0.05)
        lower, upper = conformal_prediction_intervals(
            test_means[mask], test_stds[mask], q95
        )
        conditional_results[label] = evaluate_coverage(y_test[mask], lower, upper)
        conditional_results[label]["n"] = int(mask.sum())
        conditional_results[label]["y_range"] = [float(lo), float(hi)]

    m.unfreeze_all()

    return {
        "n_finetune": n_ft,
        "n_conformal_cal": len(conf_idx),
        "n_test": len(y_test),
        "mc_samples": mc_samples,
        "point_metrics": point_metrics,
        "conformal": conformal_results,
        "mc_dropout_only": mc_results,
        "conditional_coverage": conditional_results,
        "calibration_scores_summary": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "p95": float(np.percentile(scores, 95)),
        },
    }


# ============================================================
# Visualization
# ============================================================

def generate_figures(results, figures_dir="figures"):
    """Generate UQ figures."""
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

    # Fig E6a: Coverage calibration plot
    for exp_key, exp_data in results.items():
        conformal = exp_data.get("conformal", {})
        mc_only = exp_data.get("mc_dropout_only", {})

        if not conformal:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

        # Conformal coverage
        targets_conf = [conformal[k]["target_coverage"] for k in sorted(conformal.keys())]
        actual_conf = [conformal[k]["coverage"] for k in sorted(conformal.keys())]

        ax.plot(targets_conf, actual_conf, "o-", color="#2166AC",
                label="Conformal", linewidth=2, markersize=8)

        # MC Dropout coverage
        mc_target_map = {"68%": 0.68, "95%": 0.95, "99%": 0.99}
        targets_mc = [mc_target_map[k] for k in sorted(mc_only.keys()) if k in mc_target_map]
        actual_mc = [mc_only[k]["coverage"] for k in sorted(mc_only.keys()) if k in mc_target_map]
        if targets_mc:
            ax.plot(targets_mc, actual_mc, "s--", color="#B2182B",
                    label="MC Dropout", linewidth=2, markersize=8)

        # Perfect calibration line
        ax.plot([0.5, 1], [0.5, 1], "k--", alpha=0.4, linewidth=1,
                label="Perfect calibration")

        ax.set_xlabel("Target coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"E6: Uncertainty Calibration\n{exp_key}")
        ax.legend(fontsize=9)
        ax.set_xlim(0.6, 1.02)
        ax.set_ylim(0.6, 1.02)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = f"e6_calibration_{exp_key}"
        fig.savefig(fig_dir / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / f"{fname}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        log.info(f"Saved {fname}.pdf")

    # Fig E6b: Conditional coverage
    for exp_key, exp_data in results.items():
        cond = exp_data.get("conditional_coverage", {})
        if not cond:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
        quartiles = sorted(cond.keys())
        coverages = [cond[q]["coverage"] for q in quartiles]
        widths = [cond[q]["mean_width"] for q in quartiles]

        x = np.arange(len(quartiles))
        ax.bar(x, coverages, color="#2166AC", alpha=0.8, width=0.6)
        ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.7,
                    label="Target (95%)")

        ax.set_xticks(x)
        ax.set_xticklabels(quartiles)
        ax.set_xlabel("Property quartile")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"E6: Conditional Coverage\n{exp_key}")
        ax.legend()
        ax.set_ylim(0.7, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        fname = f"e6_conditional_{exp_key}"
        fig.savefig(fig_dir / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / f"{fname}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        log.info(f"Saved {fname}.pdf")


# ============================================================
# Main
# ============================================================

def load_pretrained_model(checkpoint_path, device):
    """Load pretrained model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = SpectralFM(config)
    pretrain_model = SpectralFMForPretraining(model, config)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    pretrain_model.load_state_dict(state_dict)
    log.info(f"Loaded checkpoint from step {ckpt['step']}")
    return model, config


def main():
    parser = argparse.ArgumentParser(description="E6: Uncertainty Quantification")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--n-transfer", type=int, default=30)
    parser.add_argument("--dataset", type=str, default="corn",
                        choices=["corn", "tablet"])
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e6_uncertainty_quantification.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_pretrained_model(args.checkpoint, device)

    # Load data
    prep = SpectralPreprocessor(target_length=config.n_channels)
    all_results = {}

    if args.dataset == "corn":
        corn_dir = Path(args.data_dir) / "processed" / "corn"
        wavelengths = np.load(corn_dir / "wavelengths.npy")
        properties = np.load(corn_dir / "properties.npy")

        instruments = {}
        for inst in ["m5", "mp5", "mp6"]:
            raw = np.load(corn_dir / f"{inst}_spectra.npy")
            instruments[inst] = np.array([
                prep.process(s, wavelengths)["normalized"] for s in raw
            ])

        # Primary experiment: m5 → mp6, moisture
        rng = np.random.RandomState(42)
        n = 80
        idx = rng.permutation(n)
        cal_idx = idx[:60]
        test_idx = idx[60:]

        log.info("Running UQ on corn m5→mp6 moisture...")
        result = run_uq_experiment(
            model, config,
            instruments["m5"][cal_idx], instruments["mp6"][cal_idx],
            properties[cal_idx, 0],
            instruments["mp6"][test_idx], properties[test_idx, 0],
            device, args.mc_samples,
            n_transfer=args.n_transfer,
        )
        all_results["corn_m5_mp6_moisture"] = result

    elif args.dataset == "tablet":
        tablet_dir = Path(args.data_dir) / "processed" / "tablet"

        def preprocess_array(arr):
            return np.array([prep.process(s)["normalized"] for s in arr])

        cal_1 = preprocess_array(np.load(tablet_dir / "calibrate_1.npy"))
        cal_2 = preprocess_array(np.load(tablet_dir / "calibrate_2.npy"))
        cal_Y = np.load(tablet_dir / "calibrate_Y.npy")
        test_1 = preprocess_array(np.load(tablet_dir / "test_1.npy"))
        test_2 = preprocess_array(np.load(tablet_dir / "test_2.npy"))
        test_Y = np.load(tablet_dir / "test_Y.npy")

        log.info("Running UQ on tablet spec_1→spec_2 active ingredient...")
        result = run_uq_experiment(
            model, config,
            cal_1, cal_2, cal_Y[:, 0],
            test_2, test_Y[:, 0],
            device, args.mc_samples,
            n_transfer=args.n_transfer,
        )
        all_results["tablet_s1_s2_active"] = result

    # Print summary
    print("\n" + "=" * 70)
    print("E6: UNCERTAINTY QUANTIFICATION RESULTS")
    print("=" * 70)
    for exp_key, data in all_results.items():
        print(f"\n{exp_key}:")
        pm = data["point_metrics"]
        print(f"  Point prediction: R²={pm['r2']:.4f}, RMSEP={pm['rmsep']:.4f}")
        print(f"  Conformal prediction:")
        for level, metrics in data["conformal"].items():
            print(f"    {level}: coverage={metrics['coverage']:.3f} "
                  f"(target={metrics['target_coverage']:.2f}), "
                  f"width={metrics['mean_width']:.4f}")

    # Generate figures
    generate_figures(all_results, args.figures_dir)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "E6_uncertainty_quantification",
            "checkpoint": args.checkpoint,
            "mc_samples": args.mc_samples,
            "results": all_results,
        }, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
