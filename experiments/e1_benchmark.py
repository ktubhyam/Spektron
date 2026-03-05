#!/usr/bin/env python3
"""
E1: Architecture Benchmark on QM9S

Compares D-LinOSS vs Mamba vs Transformer vs 1D CNN vs S4D (+ PLS baseline)
on masked spectral reconstruction using QM9S (130K molecules, 2048 pts).

All architectures are param-matched at ~2M backbone params.
Runs 3 seeds per architecture, reports mean +/- std.

CROSS-VALIDATION: For headline results (D-LinOSS vs Transformer), run 3-fold
stratified cross-validation by executing with --fold-id 0, 1, 2 separately.
Each fold uses disjoint train/val/test splits. Aggregate results across folds.

Usage:
    # Run all architectures (GPU), default fold 0
    python experiments/e1_benchmark.py --h5-path data/raw/qm9s/qm9s_processed.h5

    # 3-fold cross-validation on headline models (run 3×)
    python experiments/e1_benchmark.py --h5-path ... --backbone dlinoss --fold-id 0
    python experiments/e1_benchmark.py --h5-path ... --backbone dlinoss --fold-id 1
    python experiments/e1_benchmark.py --h5-path ... --backbone dlinoss --fold-id 2

    # Quick test (CPU, 1 seed, 500 steps)
    python experiments/e1_benchmark.py --h5-path ... --quick

    # Single architecture
    python experiments/e1_benchmark.py --h5-path ... --backbone dlinoss --seeds 1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
import json
import logging
import os
import time
import traceback

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

SKIP_BACKBONES: set[str] = set()  # All 5 architectures active; pure-PyTorch Mamba (no mamba-ssm)


def count_backbone_params(model: Spektron) -> int:
    """Count only backbone parameters (not embedding, MoE, transformer, heads)."""
    return sum(p.numel() for p in model.backbone.parameters())


def load_existing_results(output_path: str) -> list:
    """Load existing results for resume support."""
    p = Path(output_path)
    if p.exists():
        try:
            with open(p) as f:
                data = json.load(f)
            return data.get("runs", [])
        except (json.JSONDecodeError, KeyError):
            return []
    return []


def is_completed(existing_results: list, backbone: str, seed: int,
                 fold_id: int, target_steps: int) -> bool:
    """Check if a (backbone, seed, fold) tuple has already been successfully completed.

    Only counts as completed if:
    - No error field present
    - Steps match target (avoids smoke-test results blocking real runs)
    - Fold ID matches (critical for CV)
    """
    for r in existing_results:
        if (r["backbone"] == backbone
                and r["seed"] == seed
                and r.get("fold_id") == fold_id
                and r.get("error") is None
                and r.get("steps", 0) >= target_steps):
            return True
    return False


def _significance_tests(all_results: list, backbones: list) -> dict:
    """Welch t-test + Cohen's d between each pair of backbones (val losses)."""
    from scipy.stats import ttest_ind
    by_bb = {
        bb: np.array([r["best_val_loss"] for r in all_results
                      if r["backbone"] == bb and r.get("error") is None
                      and r.get("best_val_loss") is not None])
        for bb in backbones
    }
    results = {}
    bb_list = [b for b in backbones if len(by_bb.get(b, [])) >= 2]
    for i, bb_a in enumerate(bb_list):
        for bb_b in bb_list[i + 1:]:
            a, b = by_bb[bb_a], by_bb[bb_b]
            t, p = ttest_ind(a, b, equal_var=False)
            pooled_std = np.sqrt(
                ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2)
            )
            d = float((a.mean() - b.mean()) / (pooled_std + 1e-12))
            results[f"{bb_a}_vs_{bb_b}"] = {
                "t_stat": float(t),
                "p_value": float(p),
                "cohens_d": d,
                "delta_val_loss": float(a.mean() - b.mean()),
            }
    return results


def save_results(all_results: list, backbones: list, args, output_path: str):
    """Save results incrementally after each run."""
    summary = {}
    for backbone in backbones:
        # Filter to successful runs only for summary stats
        runs = [r for r in all_results
                if r["backbone"] == backbone and r.get("error") is None]
        if not runs:
            continue
        val_losses = [r["best_val_loss"] for r in runs
                      if r.get("best_val_loss") is not None]
        gpu_mbs = [r["peak_gpu_mb"] for r in runs if r.get("peak_gpu_mb")]
        summary[backbone] = {
            "n_runs": len(runs),
            "backbone_params": runs[0]["backbone_params"],
            "total_params": runs[0]["total_params"],
            "best_val_loss_mean": float(np.mean(val_losses)) if val_losses else None,
            "best_val_loss_std": float(np.std(val_losses)) if val_losses else None,
            "train_time_mean_sec": float(np.mean([r["train_time_sec"] for r in runs])),
            "peak_gpu_mb_mean": float(np.mean(gpu_mbs)) if gpu_mbs else None,
        }

    significance = _significance_tests(all_results, backbones)

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({
            "experiment": "E1: Architecture Benchmark",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {"max_steps": args.max_steps, "seeds": args.seeds,
                        "batch_size": args.batch_size},
            "summary": summary,
            "significance": significance,
            "runs": all_results,
        }, f, indent=2)


def run_single(backbone: str, seed: int, h5_path: str,
               max_steps: int, batch_size: int, num_workers: int,
               quick: bool, fold_id: int = 0) -> dict:
    """Train one architecture with one seed and return metrics."""
    log.info(f"\n{'='*60}")
    log.info(f"  {backbone.upper()} | seed={seed}")
    log.info(f"{'='*60}")

    config = get_benchmark_config(backbone=backbone, seed=seed)
    # Fold-aware split seed: fold_id modulates the split generation
    config.data_seed = seed + fold_id * 1000  # Different fold → different split
    config.pretrain.max_steps = max_steps
    config.pretrain.batch_size = batch_size
    if quick:
        config.pretrain.warmup_steps = 50
        config.pretrain.grad_accumulation_steps = 1

    # Per-run checkpoint directory to avoid collisions
    config.checkpoint_dir = f"checkpoints/e1_{backbone}_s{seed}_f{fold_id}"

    # Memory-constrained architectures: force single GPU via trainer flag.
    # Mamba: custom CUDA kernels deadlock with DataParallel.
    # S4D: d_state=64 still uses significant memory; single GPU safer.
    # Both use batch=2 with grad_accum to maintain effective batch size.
    _force_single = backbone in ("mamba", "s4d")
    if _force_single:
        config.pretrain.batch_size = 2
        config.pretrain.grad_accumulation_steps = max(1, (batch_size * 4) // 2)
        log.info(f"  {backbone}: single GPU forced, batch={config.pretrain.batch_size}, "
                 f"grad_accum={config.pretrain.grad_accumulation_steps}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data (use fold-aware seed for reproducible split variation)
    train_loader, val_loader, _ = build_qm9s_loaders(
        h5_path, batch_size=config.pretrain.batch_size,
        num_workers=num_workers,
        seed=seed,
        fold_id=fold_id,
    )

    # Model
    model = Spektron(config)
    pretrain_model = SpektronForPretraining(model, config)

    backbone_params = count_backbone_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Backbone params: {backbone_params:,}")
    log.info(f"  Total params: {total_params:,}")
    log.info(f"  Checkpoints: {config.checkpoint_dir}")

    # Train
    run_name = f"e1_{backbone}_s{seed}_{int(time.time())}"
    trainer = PretrainTrainer(
        pretrain_model, config, train_loader,
        val_loader=val_loader,
        use_wandb=False,
        run_name=run_name,
        force_single_gpu=_force_single,
    )

    t0 = time.time()
    history = trainer.train(
        max_steps=config.pretrain.max_steps,
        log_every=100,
        val_every=max(500, config.pretrain.max_steps // 10),
        save_every=config.pretrain.max_steps,  # save only at end
    )
    elapsed = time.time() - t0

    # Read val curve from the JSONL log written by ExperimentLogger
    val_curve = []
    try:
        jsonl_path = Path(config.log_dir) / f"{run_name}.jsonl"
        if jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                entry = json.loads(line)
                if "val/loss" in entry and "_step" in entry:
                    val_curve.append({
                        "step": entry["_step"],
                        "val_loss": entry["val/loss"],
                        "val_msrp": entry.get("val/msrp"),
                    })
    except Exception as exc:
        log.warning(f"  Could not read val curve: {exc}")

    # Extract final metrics
    result = {
        "backbone": backbone,
        "seed": seed,
        "fold_id": fold_id,
        "backbone_params": backbone_params,
        "total_params": total_params,
        "train_time_sec": elapsed,
        "final_train_loss": history[-1]["total"] if history else None,
        "final_val_loss": trainer.best_val_loss if trainer.best_val_loss < float("inf") else None,
        "best_val_loss": trainer.best_val_loss,
        "steps": config.pretrain.max_steps,
        "checkpoint_dir": config.checkpoint_dir,
        "val_curve": val_curve,
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
    parser.add_argument("--fold-id", type=int, default=0, choices=[0, 1, 2],
                        help="Fold ID for 3-fold cross-validation (0=default, 1-2 for CV)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (500 steps, 1 seed)")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e1_benchmark.json")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing results and start fresh")
    parser.add_argument("--include-mamba", action="store_true",
                        help="Include Mamba backbone (requires mamba_ssm; SM8.6+ only)")
    args = parser.parse_args()

    if args.include_mamba:
        SKIP_BACKBONES.discard("mamba")

    if args.quick:
        args.max_steps = 500
        args.seeds = 1
        args.batch_size = 8

    backbones = BACKBONES if args.backbone == "all" else [args.backbone]

    # Resume: load existing results
    if args.no_resume:
        all_results = []
    else:
        all_results = load_existing_results(args.output)
        if all_results:
            completed = [(r["backbone"], r["seed"], r.get("steps", 0))
                         for r in all_results if r.get("error") is None]
            log.info(f"Resuming: found {len(completed)} successful runs: {completed}")

    # Seeds outer, backbones inner — get 1 full comparison table per seed/fold
    seeds = [42 + i for i in range(args.seeds)]
    for seed in seeds:
        for backbone in backbones:
            # Permanently skip hardware-incompatible architectures
            if backbone in SKIP_BACKBONES:
                log.info(f"SKIP {backbone} seed={seed} fold={args.fold_id} (hardware-incompatible)")
                # Ensure result is recorded if not already
                if not any(r["backbone"] == backbone and r["seed"] == seed and r.get("fold_id") == args.fold_id for r in all_results):
                    all_results.append({
                        "backbone": backbone, "seed": seed, "fold_id": args.fold_id,
                        "error": "SKIP: hardware incompatible",
                        "backbone_params": None, "total_params": None,
                        "train_time_sec": 0, "final_train_loss": None,
                        "final_val_loss": None, "best_val_loss": None,
                        "steps": 0,
                    })
                continue

            # Skip if already completed with matching step count and fold_id
            if is_completed(all_results, backbone, seed, args.fold_id, args.max_steps):
                log.info(f"SKIP {backbone} seed={seed} fold={args.fold_id} (already completed at {args.max_steps} steps)")
                continue

            try:
                result = run_single(
                    backbone=backbone,
                    seed=seed,
                    h5_path=args.h5_path,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    quick=args.quick,
                    fold_id=args.fold_id,
                )
                all_results.append(result)
            except torch.cuda.OutOfMemoryError:
                log.error(f"OOM on {backbone} seed={seed} fold={args.fold_id} -- skipping")
                torch.cuda.empty_cache()
                all_results.append({
                    "backbone": backbone, "seed": seed, "fold_id": args.fold_id,
                    "error": "CUDA OOM",
                    "backbone_params": None, "total_params": None,
                    "train_time_sec": 0, "final_train_loss": None,
                    "final_val_loss": None, "best_val_loss": None,
                    "steps": 0,
                })
            except Exception as e:
                log.error(f"FAILED {backbone} seed={seed} fold={args.fold_id}: {e}")
                log.error(traceback.format_exc())
                torch.cuda.empty_cache()
                all_results.append({
                    "backbone": backbone, "seed": seed, "fold_id": args.fold_id,
                    "error": str(e),
                    "backbone_params": None, "total_params": None,
                    "train_time_sec": 0, "final_train_loss": None,
                    "final_val_loss": None, "best_val_loss": None,
                    "steps": 0,
                })

            # Incremental save after each run
            save_results(all_results, backbones, args, args.output)

            # Free GPU memory between runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print summary table
    print(f"\n{'='*70}")
    print("  E1 BENCHMARK RESULTS")
    print(f"{'='*70}")
    hdr = f"  {'Backbone':<15} {'Params':>10} {'Val Loss':>15} {'Time(s)':>10}"
    print(hdr)
    print(f"  {'-'*55}")
    for bb in backbones:
        runs = [r for r in all_results
                if r["backbone"] == bb and r.get("best_val_loss") is not None
                and r.get("error") is None]
        if not runs:
            print(f"  {bb:<15} {'N/A':>10} {'FAILED':>15} {'N/A':>10}")
            continue
        val_losses = [r["best_val_loss"] for r in runs]
        if len(val_losses) > 1:
            vl = f"{np.mean(val_losses):.4f}+/-{np.std(val_losses):.4f}"
        else:
            vl = f"{val_losses[0]:.4f}"
        avg_time = np.mean([r["train_time_sec"] for r in runs])
        print(f"  {bb:<15} {runs[0]['backbone_params']:>10,} {vl:>15} "
              f"{avg_time:>10.0f}")
    print()

    log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
