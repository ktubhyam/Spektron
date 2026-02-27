#!/usr/bin/env python3
"""
E1: Architecture Benchmark on QM9S

Compares D-LinOSS vs Mamba vs Transformer vs 1D CNN vs S4D (+ PLS baseline)
on masked spectral reconstruction using QM9S (130K molecules, 2048 pts).

All architectures are param-matched at ~2M backbone params.
Runs 3 seeds per architecture, reports mean +/- std.

Usage:
    # Run all architectures (GPU)
    python experiments/e1_benchmark.py --h5-path data/raw/qm9s/qm9s_processed.h5

    # Quick test (CPU, 1 seed, 500 steps)
    python experiments/e1_benchmark.py --h5-path ... --quick

    # Single architecture
    python experiments/e1_benchmark.py --h5-path ... --backbone dlinoss --seeds 1
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

from src.config import get_benchmark_config
from src.models.spektron import Spektron, SpektronForPretraining
from src.data.qm9s import build_qm9s_loaders
from src.training.trainer import PretrainTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

BACKBONES = ["dlinoss", "mamba", "transformer", "cnn1d", "s4d"]


def count_backbone_params(model: Spektron) -> int:
    """Count only backbone parameters (not embedding, MoE, transformer, heads)."""
    return sum(p.numel() for p in model.backbone.parameters())


def run_single(backbone: str, seed: int, h5_path: str,
               max_steps: int, batch_size: int, num_workers: int,
               quick: bool) -> dict:
    """Train one architecture with one seed and return metrics."""
    log.info(f"\n{'='*60}")
    log.info(f"  {backbone.upper()} | seed={seed}")
    log.info(f"{'='*60}")

    config = get_benchmark_config(backbone=backbone, seed=seed)
    if quick:
        config.pretrain.max_steps = max_steps
        config.pretrain.batch_size = batch_size
        config.pretrain.warmup_steps = 50

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    train_loader, val_loader, _ = build_qm9s_loaders(
        h5_path, batch_size=config.pretrain.batch_size,
        num_workers=num_workers,
    )

    # Model
    model = Spektron(config)
    pretrain_model = SpektronForPretraining(model, config)

    backbone_params = count_backbone_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Backbone params: {backbone_params:,}")
    log.info(f"  Total params: {total_params:,}")

    # Train
    run_name = f"e1_{backbone}_s{seed}_{int(time.time())}"
    trainer = PretrainTrainer(
        pretrain_model, config, train_loader,
        val_loader=val_loader,
        use_wandb=False,
        run_name=run_name,
    )

    t0 = time.time()
    history = trainer.train(
        max_steps=config.pretrain.max_steps,
        log_every=100,
        val_every=max(500, config.pretrain.max_steps // 10),
        save_every=config.pretrain.max_steps,  # save only at end
    )
    elapsed = time.time() - t0

    # Extract final metrics
    result = {
        "backbone": backbone,
        "seed": seed,
        "backbone_params": backbone_params,
        "total_params": total_params,
        "train_time_sec": elapsed,
        "final_train_loss": history[-1]["total"] if history else None,
        "final_val_loss": trainer.best_val_loss if trainer.best_val_loss < float("inf") else None,
        "best_val_loss": trainer.best_val_loss,
        "steps": config.pretrain.max_steps,
    }

    # Memory usage
    if torch.cuda.is_available():
        result["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()

    log.info(f"  Done: val_loss={result['best_val_loss']:.4f}, "
             f"time={elapsed:.0f}s, params={backbone_params:,}")

    return result


def main():
    parser = argparse.ArgumentParser(description="E1: Architecture Benchmark")
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="all",
                        help="Single backbone or 'all'")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (500 steps, 1 seed)")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e1_benchmark.json")
    args = parser.parse_args()

    if args.quick:
        args.max_steps = 500
        args.seeds = 1
        args.batch_size = 8

    backbones = BACKBONES if args.backbone == "all" else [args.backbone]

    all_results = []
    for backbone in backbones:
        for seed in range(args.seeds):
            result = run_single(
                backbone=backbone,
                seed=42 + seed,
                h5_path=args.h5_path,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                quick=args.quick,
            )
            all_results.append(result)

    # Aggregate by backbone
    summary = {}
    for backbone in backbones:
        runs = [r for r in all_results if r["backbone"] == backbone]
        val_losses = [r["best_val_loss"] for r in runs
                      if r["best_val_loss"] is not None]
        summary[backbone] = {
            "n_runs": len(runs),
            "backbone_params": runs[0]["backbone_params"],
            "total_params": runs[0]["total_params"],
            "best_val_loss_mean": float(np.mean(val_losses)) if val_losses else None,
            "best_val_loss_std": float(np.std(val_losses)) if val_losses else None,
            "train_time_mean_sec": float(np.mean([r["train_time_sec"] for r in runs])),
        }

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  E1 BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  {'Backbone':<15} {'Params':>10} {'Val Loss':>15} {'Time(s)':>10}")
    print(f"  {'-'*55}")
    for bb, s in summary.items():
        vl = f"{s['best_val_loss_mean']:.4f}+/-{s['best_val_loss_std']:.4f}" \
            if s["best_val_loss_mean"] is not None else "N/A"
        print(f"  {bb:<15} {s['backbone_params']:>10,} {vl:>15} "
              f"{s['train_time_mean_sec']:>10.0f}")
    print()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "E1: Architecture Benchmark",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {"max_steps": args.max_steps, "seeds": args.seeds},
            "summary": summary,
            "runs": all_results,
        }, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
