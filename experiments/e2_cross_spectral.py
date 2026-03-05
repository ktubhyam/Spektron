#!/usr/bin/env python3
"""
E2: Cross-Spectral Prediction (IR <-> Raman)

THE novel contribution — predicts Raman from IR and vice versa.
Zero prior work exists for this task (confirmed across 6 major surveys).

PRIMARY VALIDATION: QM9S dataset with real paired IR-Raman spectra.
99.93% of QM9 molecules are non-centrosymmetric, so mutual exclusion barely
applies and IR↔Raman correlation is chemically meaningful.

Trains all 5 backbone architectures on cross-spectral prediction,
evaluates with spectral MSE, cosine similarity, peak recall,
peak intensity correlation, and SID.

Usage:
    python experiments/e2_cross_spectral.py --h5-path data/raw/qm9s/qm9s_processed.h5

    # Quick test
    python experiments/e2_cross_spectral.py --h5-path ... --quick

    # Single direction
    python experiments/e2_cross_spectral.py --h5-path ... --direction ir2raman
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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.config import get_benchmark_config
from src.models.spektron import Spektron
from src.models.heads import CrossSpectralHead
from src.data.cross_spectral import build_cross_spectral_loaders
from src.evaluation.cross_spectral_metrics import compute_cross_spectral_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

BACKBONES = ["dlinoss", "mamba", "transformer", "cnn1d", "s4d"]
SKIP_BACKBONES: set[str] = set()  # All 5 architectures active; pure-PyTorch Mamba (no mamba-ssm)
DIRECTIONS = ["ir2raman", "raman2ir"]


def train_cross_spectral(
    backbone: str,
    direction: str,
    seed: int,
    h5_path: str,
    max_steps: int,
    batch_size: int,
    num_workers: int,
    lr: float = 3e-4,
) -> dict:
    """Train one architecture on cross-spectral prediction and evaluate."""
    log.info(f"\n{'='*60}")
    log.info(f"  {backbone.upper()} | {direction} | seed={seed}")
    log.info(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    config = get_benchmark_config(backbone=backbone, seed=seed)
    device = config.device

    # Data
    train_loader, val_loader, test_loader = build_cross_spectral_loaders(
        h5_path, direction=direction,
        batch_size=batch_size, num_workers=num_workers,
        augment_source=True,
    )

    # Model: encoder + cross-spectral head
    encoder = Spektron(config)
    cross_head = CrossSpectralHead(
        d_model=config.d_model,
        n_channels=config.n_channels,
    )

    encoder.to(device)
    cross_head.to(device)

    # Optimizer
    all_params = list(encoder.parameters()) + list(cross_head.parameters())
    optimizer = AdamW(all_params, lr=lr, weight_decay=0.01)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=min(1000, max_steps // 10))
    cosine = CosineAnnealingLR(optimizer,
                               T_max=max_steps - min(1000, max_steps // 10),
                               eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[min(1000, max_steps // 10)])

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16

    # Training loop
    step = 0
    best_val_mse = float("inf")
    best_state = None
    t0 = time.time()

    encoder.train()
    cross_head.train()

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            source = batch["source_spectrum"].to(device)
            target = batch["target_spectrum"].to(device)
            domain = batch["source_domain"]
            if isinstance(domain, list):
                _dmap = {"NIR": 0, "IR": 1, "RAMAN": 2, "UNKNOWN": 3}
                domain = torch.tensor(
                    [_dmap.get(d, 3) for d in domain],
                    dtype=torch.long, device=device,
                )

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                enc_out = encoder.encode(source, domain)
                pred = cross_head(enc_out["tokens"])
                loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 200 == 0:
                log.info(f"  Step {step}/{max_steps} | Loss: {loss.item():.6f}")

            # Validate every 1000 steps
            if step % max(500, max_steps // 10) == 0:
                val_mse = _evaluate(encoder, cross_head, val_loader, device,
                                    use_amp, amp_dtype)
                log.info(f"  Val MSE: {val_mse:.6f}")
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_state = {
                        "encoder": {k: v.cpu().clone()
                                    for k, v in encoder.state_dict().items()},
                        "cross_head": {k: v.cpu().clone()
                                       for k, v in cross_head.state_dict().items()},
                    }
                encoder.train()
                cross_head.train()

    elapsed = time.time() - t0

    # Restore best model
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        cross_head.load_state_dict(best_state["cross_head"])

    # Final evaluation on test set
    test_metrics = _evaluate_full(encoder, cross_head, test_loader, device,
                                  use_amp, amp_dtype)

    per_mol = test_metrics.pop("per_molecule", {})
    result = {
        "backbone": backbone,
        "direction": direction,
        "seed": seed,
        "train_time_sec": elapsed,
        "best_val_mse": best_val_mse,
        "steps": max_steps,
        **{f"test_{k}": float(v) for k, v in test_metrics.items()},
        "per_molecule": per_mol,
    }

    log.info(f"  Test MSE={test_metrics['mse']:.6f}, "
             f"Cosine={test_metrics['cosine_similarity']:.4f}, "
             f"PeakRecall={test_metrics['peak_recall']:.4f}, "
             f"NoPeakFrac={per_mol.get('frac_no_target_peaks', float('nan')):.3f}")

    return result


@torch.no_grad()
def _evaluate(encoder, cross_head, loader, device, use_amp, amp_dtype):
    """Quick MSE evaluation."""
    encoder.eval()
    cross_head.eval()
    total_mse = 0.0
    n = 0
    for batch in loader:
        source = batch["source_spectrum"].to(device)
        target = batch["target_spectrum"].to(device)
        domain = batch["source_domain"]
        if isinstance(domain, list):
            _dmap = {"NIR": 0, "IR": 1, "RAMAN": 2, "UNKNOWN": 3}
            domain = torch.tensor(
                [_dmap.get(d, 3) for d in domain],
                dtype=torch.long, device=device,
            )
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            enc_out = encoder.encode(source, domain)
            pred = cross_head(enc_out["tokens"])
        total_mse += F.mse_loss(pred, target, reduction="sum").item()
        n += source.size(0)
    return total_mse / max(n, 1)


@torch.no_grad()
def _evaluate_full(encoder, cross_head, loader, device, use_amp, amp_dtype):
    """Full evaluation with all cross-spectral metrics and per-molecule analysis."""
    encoder.eval()
    cross_head.eval()

    all_pred = []
    all_target = []
    all_mol_idx = []
    all_smiles = []

    for batch in loader:
        source = batch["source_spectrum"].to(device)
        target = batch["target_spectrum"].to(device)
        mol_idx = batch.get("molecule_idx")
        smiles = batch.get("smiles", [])
        domain = batch["source_domain"]
        if isinstance(domain, list):
            _dmap = {"NIR": 0, "IR": 1, "RAMAN": 2, "UNKNOWN": 3}
            domain = torch.tensor(
                [_dmap.get(d, 3) for d in domain],
                dtype=torch.long, device=device,
            )
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            enc_out = encoder.encode(source, domain)
            pred = cross_head(enc_out["tokens"])
        all_pred.append(pred.float().cpu())
        all_target.append(target.float().cpu())
        if mol_idx is not None:
            all_mol_idx.append(mol_idx.cpu())
        if isinstance(smiles, (list, tuple)):
            all_smiles.extend(smiles)

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    per_sample = compute_cross_spectral_metrics(all_pred, all_target)
    means = {k: float(v.mean().item()) for k, v in per_sample.items()}

    # Per-molecule analysis
    per_mol: dict = {}
    if all_mol_idx:
        from src.evaluation.cross_spectral_metrics import detect_peaks as _detect_peaks
        mol_idx_np = torch.cat(all_mol_idx).numpy()
        mse_np = per_sample["mse"].numpy()
        cos_np = per_sample["cosine_similarity"].numpy()
        target_np = all_target.numpy()

        # Count target peaks per molecule (distinguishes true low recall from
        # the peak_recall=1.0 artefact when there are no target peaks at all)
        n_peaks_target = np.array(
            [len(_detect_peaks(target_np[i])) for i in range(len(target_np))],
            dtype=np.int32,
        )
        has_peaks_mask = n_peaks_target > 0
        n_no_peaks = int(np.sum(~has_peaks_mask))

        # Percentile breakdown
        mse_pcts = {
            f"mse_p{p}": float(np.percentile(mse_np, p))
            for p in (10, 25, 50, 75, 90, 99)
        }

        # Best / worst 20 by MSE (indices into the test set)
        order = np.argsort(mse_np)
        def _mol_record(i: int) -> dict:
            return {
                "mol_idx": int(mol_idx_np[i]),
                "smiles": all_smiles[i] if i < len(all_smiles) else "",
                "mse": float(mse_np[i]),
                "cosine": float(cos_np[i]),
                "n_target_peaks": int(n_peaks_target[i]),
            }
        best_20 = [_mol_record(i) for i in order[:20]]
        worst_20 = [_mol_record(i) for i in order[-20:]]

        per_mol = {
            "n_samples": int(len(mse_np)),
            "n_no_target_peaks": n_no_peaks,
            "frac_no_target_peaks": float(n_no_peaks / max(len(mse_np), 1)),
            **mse_pcts,
            "best_20": best_20,
            "worst_20": worst_20,
        }

    return {**means, "per_molecule": per_mol}


def _is_e2_completed(
    results: list, backbone: str, direction: str, seed: int, target_steps: int,
) -> bool:
    for r in results:
        if (r["backbone"] == backbone and r["direction"] == direction
                and r["seed"] == seed and r.get("error") is None
                and r.get("steps", 0) >= target_steps):
            return True
    return False


def _save_e2_results(
    all_results: list, identity_baselines: dict, args: object, output_path: str,
) -> None:
    """Save results incrementally — safe to call after every completed run."""
    done_bbs = list(dict.fromkeys(r["backbone"] for r in all_results))
    done_dirs = list(dict.fromkeys(r["direction"] for r in all_results))

    summary: dict = {}
    for direction in done_dirs:
        summary[direction] = {}
        for backbone in done_bbs:
            runs = [r for r in all_results
                    if r["backbone"] == backbone and r["direction"] == direction
                    and r.get("error") is None]
            if not runs:
                continue
            mses = [r["test_mse"] for r in runs]
            cosines = [r["test_cosine_similarity"] for r in runs]
            recalls = [r["test_peak_recall"] for r in runs]
            pm_list = [r.get("per_molecule", {}) for r in runs]
            pm_p50 = [pm.get("mse_p50", float("nan")) for pm in pm_list]
            pm_p90 = [pm.get("mse_p90", float("nan")) for pm in pm_list]
            no_peak_frac = [pm.get("frac_no_target_peaks", float("nan"))
                            for pm in pm_list]
            summary[direction][backbone] = {
                "n_runs": len(runs),
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "cosine_mean": float(np.mean(cosines)),
                "cosine_std": float(np.std(cosines)),
                "peak_recall_mean": float(np.mean(recalls)),
                "peak_recall_std": float(np.std(recalls)),
                "mse_p50_mean": float(np.nanmean(pm_p50)),
                "mse_p90_mean": float(np.nanmean(pm_p90)),
                "frac_no_target_peaks": float(np.nanmean(no_peak_frac)),
            }

    significance: dict = {}
    for direction in done_dirs:
        significance[direction] = {}
        bb_list = [b for b in done_bbs if b in summary.get(direction, {})]
        for i, bb_a in enumerate(bb_list):
            for bb_b in bb_list[i + 1:]:
                runs_a = [r for r in all_results
                          if r["backbone"] == bb_a and r["direction"] == direction
                          and r.get("error") is None]
                runs_b = [r for r in all_results
                          if r["backbone"] == bb_b and r["direction"] == direction
                          and r.get("error") is None]
                mses_a = np.array([r["test_mse"] for r in runs_a])
                mses_b = np.array([r["test_mse"] for r in runs_b])
                if len(mses_a) >= 2 and len(mses_b) >= 2:
                    from scipy.stats import ttest_ind
                    t_stat, p_val = ttest_ind(mses_a, mses_b, equal_var=False)
                    pooled_std = np.sqrt(
                        ((len(mses_a) - 1) * mses_a.var(ddof=1)
                         + (len(mses_b) - 1) * mses_b.var(ddof=1))
                        / (len(mses_a) + len(mses_b) - 2)
                    )
                    cohens_d = float((mses_a.mean() - mses_b.mean())
                                     / (pooled_std + 1e-12))
                    significance[direction][f"{bb_a}_vs_{bb_b}"] = {
                        "t_stat": float(t_stat),
                        "p_value": float(p_val),
                        "cohens_d": cohens_d,
                        "mse_diff": float(mses_a.mean() - mses_b.mean()),
                    }

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({
            "experiment": "E2: Cross-Spectral Prediction",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "max_steps": args.max_steps,
                "seeds": args.seeds,
                "lr": args.lr,
            },
            "identity_baselines": identity_baselines,
            "summary": summary,
            "significance": significance,
            "runs": all_results,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="E2: Cross-Spectral Prediction")
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="all",
                        help="Single backbone or 'all'")
    parser.add_argument("--direction", type=str, default="both",
                        choices=["ir2raman", "raman2ir", "both"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing results and start fresh")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e2_cross_spectral.json")
    args = parser.parse_args()

    if args.quick:
        args.max_steps = 500
        args.seeds = 1
        args.batch_size = 8

    backbones = BACKBONES if args.backbone == "all" else [args.backbone]
    backbones = [b for b in backbones if b not in SKIP_BACKBONES]
    directions = DIRECTIONS if args.direction == "both" else [args.direction]

    # Resume: load existing completed runs from JSON
    all_results: list = []
    identity_baselines: dict = {}
    if not args.no_resume:
        p = Path(args.output)
        if p.exists():
            try:
                with open(p) as f:
                    existing = json.load(f)
                all_results = existing.get("runs", [])
                identity_baselines = existing.get("identity_baselines", {})
                if all_results:
                    completed = [
                        (r["backbone"], r["direction"], r["seed"])
                        for r in all_results if r.get("error") is None
                    ]
                    log.info(f"Resuming: {len(completed)} completed runs: {completed}")
            except (json.JSONDecodeError, KeyError):
                pass

    # Identity-copy baseline: fast, compute once per direction
    for direction in directions:
        if direction in identity_baselines:
            continue
        log.info(f"Computing identity-copy baseline for {direction}...")
        _, _, test_loader = build_cross_spectral_loaders(
            args.h5_path, direction=direction,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        all_source, all_target = [], []
        for batch in test_loader:
            all_source.append(batch["source_spectrum"])
            all_target.append(batch["target_spectrum"])
        all_source = torch.cat(all_source, dim=0)
        all_target = torch.cat(all_target, dim=0)
        id_metrics = compute_cross_spectral_metrics(all_source, all_target)
        identity_baselines[direction] = {k: v.mean().item() for k, v in id_metrics.items()}
        log.info(f"  Identity: MSE={identity_baselines[direction]['mse']:.6f}, "
                 f"Cosine={identity_baselines[direction]['cosine_similarity']:.4f}")

    for direction in directions:
        for backbone in backbones:
            for seed_i in range(args.seeds):
                seed = 42 + seed_i
                if _is_e2_completed(all_results, backbone, direction, seed,
                                    args.max_steps):
                    log.info(f"SKIP {backbone} {direction} seed={seed} "
                             f"(already completed at {args.max_steps} steps)")
                    continue
                try:
                    result = train_cross_spectral(
                        backbone=backbone,
                        direction=direction,
                        seed=seed,
                        h5_path=args.h5_path,
                        max_steps=args.max_steps,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        lr=args.lr,
                    )
                    all_results.append(result)
                except Exception as exc:
                    log.error(f"FAILED {backbone} {direction} seed={seed}: {exc}")
                    all_results.append({
                        "backbone": backbone, "direction": direction, "seed": seed,
                        "error": str(exc), "steps": 0,
                        "test_mse": float("nan"),
                        "test_cosine_similarity": float("nan"),
                        "test_peak_recall": float("nan"),
                        "per_molecule": {},
                    })
                _save_e2_results(all_results, identity_baselines, args, args.output)
                log.info(f"Saved incremental results to {args.output}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"  E2 CROSS-SPECTRAL PREDICTION RESULTS")
    print(f"{'='*80}")
    for direction in directions:
        print(f"\n  Direction: {direction}")
        if direction in identity_baselines:
            ib = identity_baselines[direction]
            print(f"  Identity baseline: MSE={ib['mse']:.6f}, "
                  f"Cosine={ib['cosine_similarity']:.4f}")
        good_runs = [r for r in all_results
                     if r["direction"] == direction and r.get("error") is None]
        bbs_done = list(dict.fromkeys(r["backbone"] for r in good_runs))
        print(f"  {'Backbone':<15} {'MSE':>15} {'Cosine':>15} {'PeakRecall':>15} "
              f"{'NoPeakFrac':>12}")
        print(f"  {'-'*70}")
        for bb in bbs_done:
            runs = [r for r in good_runs if r["backbone"] == bb]
            mses = [r["test_mse"] for r in runs]
            cosines = [r["test_cosine_similarity"] for r in runs]
            recalls = [r["test_peak_recall"] for r in runs]
            pm_list = [r.get("per_molecule", {}) for r in runs]
            no_peak_frac = float(np.nanmean(
                [pm.get("frac_no_target_peaks", float("nan")) for pm in pm_list]))
            mse_str = (f"{np.mean(mses):.6f}+/-{np.std(mses):.6f}"
                       if len(mses) > 1 else f"{mses[0]:.6f}")
            cos_str = (f"{np.mean(cosines):.4f}+/-{np.std(cosines):.4f}"
                       if len(cosines) > 1 else f"{cosines[0]:.4f}")
            pr_str = (f"{np.mean(recalls):.4f}+/-{np.std(recalls):.4f}"
                      if len(recalls) > 1 else f"{recalls[0]:.4f}")
            print(f"  {bb:<15} {mse_str:>15} {cos_str:>15} {pr_str:>15} "
                  f"{no_peak_frac:>12.3f}")

        sig_runs = [r for r in all_results if r["direction"] == direction
                    and r.get("error") is None]
        sig_bbs = list(dict.fromkeys(r["backbone"] for r in sig_runs))
        pairs_printed = False
        for i, bb_a in enumerate(sig_bbs):
            for bb_b in sig_bbs[i + 1:]:
                runs_a = [r for r in sig_runs if r["backbone"] == bb_a]
                runs_b = [r for r in sig_runs if r["backbone"] == bb_b]
                if len(runs_a) >= 2 and len(runs_b) >= 2:
                    if not pairs_printed:
                        print(f"\n  Pairwise significance (Welch t-test on seed MSEs):")
                        pairs_printed = True
                    from scipy.stats import ttest_ind
                    mses_a = np.array([r["test_mse"] for r in runs_a])
                    mses_b = np.array([r["test_mse"] for r in runs_b])
                    _, p_val = ttest_ind(mses_a, mses_b, equal_var=False)
                    pooled_std = np.sqrt(
                        ((len(mses_a) - 1) * mses_a.var(ddof=1)
                         + (len(mses_b) - 1) * mses_b.var(ddof=1))
                        / (len(mses_a) + len(mses_b) - 2)
                    )
                    d = (mses_a.mean() - mses_b.mean()) / (pooled_std + 1e-12)
                    sig = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                           else "*" if p_val < 0.05 else "ns")
                    print(f"    {bb_a}_vs_{bb_b:<25} p={p_val:.3e} {sig:>3}  "
                          f"d={d:+.2f}  ΔMSE={mses_a.mean()-mses_b.mean():+.6f}")
    print()

    log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
