#!/usr/bin/env python3
"""
E2: Cross-Spectral Prediction (IR <-> Raman)

THE novel contribution — predicts Raman from IR and vice versa.
Zero prior work exists for this task (confirmed across 6 major surveys).

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

    result = {
        "backbone": backbone,
        "direction": direction,
        "seed": seed,
        "train_time_sec": elapsed,
        "best_val_mse": best_val_mse,
        "steps": max_steps,
        **{f"test_{k}": float(v) for k, v in test_metrics.items()},
    }

    log.info(f"  Test MSE={test_metrics['mse']:.6f}, "
             f"Cosine={test_metrics['cosine_similarity']:.4f}, "
             f"PeakRecall={test_metrics['peak_recall']:.4f}")

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
    """Full evaluation with all cross-spectral metrics."""
    encoder.eval()
    cross_head.eval()

    all_pred = []
    all_target = []

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
        all_pred.append(pred.float().cpu())
        all_target.append(target.float().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    metrics = compute_cross_spectral_metrics(all_pred, all_target)

    return {k: v.mean().item() for k, v in metrics.items()}


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
    parser.add_argument("--output", type=str,
                        default="experiments/results/e2_cross_spectral.json")
    args = parser.parse_args()

    if args.quick:
        args.max_steps = 500
        args.seeds = 1
        args.batch_size = 8

    backbones = BACKBONES if args.backbone == "all" else [args.backbone]
    directions = DIRECTIONS if args.direction == "both" else [args.direction]

    all_results = []
    for direction in directions:
        for backbone in backbones:
            for seed in range(args.seeds):
                result = train_cross_spectral(
                    backbone=backbone,
                    direction=direction,
                    seed=42 + seed,
                    h5_path=args.h5_path,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    lr=args.lr,
                )
                all_results.append(result)

    # Identity-copy baseline: how well does "just copy source" do?
    # (if the model can't beat this, it's useless)
    identity_baselines = {}
    for direction in directions:
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
        identity_metrics = compute_cross_spectral_metrics(all_source, all_target)
        identity_baselines[direction] = {
            k: v.mean().item() for k, v in identity_metrics.items()
        }
        log.info(f"  Identity baseline: MSE={identity_baselines[direction]['mse']:.6f}, "
                 f"Cosine={identity_baselines[direction]['cosine_similarity']:.4f}")

    # Aggregate
    summary = {}
    for direction in directions:
        summary[direction] = {}
        for backbone in backbones:
            runs = [r for r in all_results
                    if r["backbone"] == backbone and r["direction"] == direction]
            mses = [r["test_mse"] for r in runs]
            cosines = [r["test_cosine_similarity"] for r in runs]
            recalls = [r["test_peak_recall"] for r in runs]
            summary[direction][backbone] = {
                "n_runs": len(runs),
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "cosine_mean": float(np.mean(cosines)),
                "cosine_std": float(np.std(cosines)),
                "peak_recall_mean": float(np.mean(recalls)),
                "peak_recall_std": float(np.std(recalls)),
            }

    # Print summary
    print(f"\n{'='*80}")
    print(f"  E2 CROSS-SPECTRAL PREDICTION RESULTS")
    print(f"{'='*80}")
    for direction in directions:
        print(f"\n  Direction: {direction}")
        ib = identity_baselines[direction]
        print(f"  Identity baseline: MSE={ib['mse']:.6f}, "
              f"Cosine={ib['cosine_similarity']:.4f}")
        print(f"  {'Backbone':<15} {'MSE':>15} {'Cosine':>15} {'PeakRecall':>15}")
        print(f"  {'-'*60}")
        for bb, s in summary[direction].items():
            mse_str = f"{s['mse_mean']:.6f}+/-{s['mse_std']:.6f}"
            cos_str = f"{s['cosine_mean']:.4f}+/-{s['cosine_std']:.4f}"
            pr_str = f"{s['peak_recall_mean']:.4f}+/-{s['peak_recall_std']:.4f}"
            print(f"  {bb:<15} {mse_str:>15} {cos_str:>15} {pr_str:>15}")
    print()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
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
            "runs": all_results,
        }, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
