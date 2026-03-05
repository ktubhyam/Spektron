#!/usr/bin/env python3
"""
E1-Scaling: D-LinOSS inductive bias scaling analysis.

Runs D-LinOSS at four parameter scales (0.5M, 2M, 5M, 10M) to test whether
the oscillatory inductive bias scales favourably — i.e., does the gap over
CNN/Transformer widen or narrow as capacity grows?

The 2M baseline is already completed in e1_benchmark.json; this script covers
0.5M, 5M, and 10M (plus re-running 2M if not present).

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 experiments/e1_scaling.py \\
        --h5-path data/raw/qm9s/qm9s_processed.h5 \\
        --output experiments/results/e1_scaling.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gc
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

# (label, d_model, d_state) — n_layers fixed at 4 (8 bidirectional blocks)
SCALE_CONFIGS: list[tuple[str, int, int]] = [
    ("dlinoss_0.5M", 128, 64),
    ("dlinoss_2M",   256, 128),
    ("dlinoss_5M",   384, 192),
    ("dlinoss_10M",  512, 256),
]


def count_backbone_params(model: Spektron) -> int:
    return sum(p.numel() for p in model.backbone.parameters())


def load_existing(path: str) -> list:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text()).get("runs", [])
        except (json.JSONDecodeError, KeyError):
            return []
    return []


def is_completed(existing: list, label: str, seed: int, target_steps: int) -> bool:
    for r in existing:
        if (r.get("label") == label
                and r.get("seed") == seed
                and not r.get("error")
                and r.get("steps", 0) >= target_steps):
            return True
    return False


def save_results(runs: list, args, output: str) -> None:
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "experiment": "E1-Scaling: D-LinOSS inductive bias scaling",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"max_steps": args.max_steps, "seeds": args.seeds},
        "runs": runs,
    }, indent=2))


def run_single(label: str, d_model: int, d_state: int,
               seed: int, h5_path: str, max_steps: int,
               batch_size: int, num_workers: int) -> dict:
    log.info(f"\n{'='*60}")
    log.info(f"  {label} | d_model={d_model} d_state={d_state} | seed={seed}")
    log.info(f"{'='*60}")

    config = get_benchmark_config(backbone="dlinoss", seed=seed)
    config.dlinoss.d_model = d_model
    config.dlinoss.d_state = d_state
    config.dlinoss.n_layers = 4
    config.transformer.d_model = d_model  # post-backbone transformer must match
    config.pretrain.max_steps = max_steps
    config.pretrain.batch_size = batch_size
    config.checkpoint_dir = f"checkpoints/scaling_{label}_s{seed}"

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, _ = build_qm9s_loaders(
        h5_path, batch_size=batch_size, num_workers=num_workers,
    )

    model = Spektron(config)
    pretrain_model = SpektronForPretraining(model, config)

    backbone_params = count_backbone_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Backbone params: {backbone_params:,}")
    log.info(f"  Total params:    {total_params:,}")

    run_name = f"scaling_{label}_s{seed}_{int(time.time())}"
    trainer = PretrainTrainer(
        pretrain_model, config, train_loader,
        val_loader=val_loader,
        use_wandb=False,
        run_name=run_name,
        force_single_gpu=False,
    )

    t0 = time.time()
    history = trainer.train(
        max_steps=max_steps,
        log_every=100,
        val_every=max(500, max_steps // 10),
        save_every=max_steps,
    )
    elapsed = time.time() - t0

    result: dict = {
        "label": label,
        "d_model": d_model,
        "d_state": d_state,
        "seed": seed,
        "backbone_params": backbone_params,
        "total_params": total_params,
        "steps": max_steps,
        "best_val_loss": trainer.best_val_loss if trainer.best_val_loss < float("inf") else None,
        "train_time_sec": elapsed,
    }
    if torch.cuda.is_available():
        result["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()

    log.info(f"  Done: val={result['best_val_loss']:.4f}, time={elapsed:.0f}s, "
             f"backbone_params={backbone_params:,}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E1-Scaling: D-LinOSS scaling analysis")
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="experiments/results/e1_scaling.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--scales", default="all",
                        help="Comma-separated labels or 'all'")
    args = parser.parse_args()

    if args.no_resume:
        all_results: list = []
    else:
        all_results = load_existing(args.output)

    scales = SCALE_CONFIGS
    if args.scales != "all":
        wanted = set(args.scales.split(","))
        scales = [(l, dm, ds) for l, dm, ds in SCALE_CONFIGS if l in wanted]

    seeds = [42 + i for i in range(args.seeds)]

    for label, d_model, d_state in scales:
        for seed in seeds:
            if is_completed(all_results, label, seed, args.max_steps):
                log.info(f"SKIP {label} s{seed} (already completed)")
                continue
            try:
                result = run_single(
                    label=label, d_model=d_model, d_state=d_state,
                    seed=seed, h5_path=args.h5_path,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                all_results.append(result)
            except torch.cuda.OutOfMemoryError:
                log.error(f"OOM {label} s{seed} — try smaller batch-size")
                torch.cuda.empty_cache()
                all_results.append({
                    "label": label, "d_model": d_model, "d_state": d_state,
                    "seed": seed, "error": "CUDA OOM",
                    "backbone_params": None, "total_params": None,
                    "steps": 0, "best_val_loss": None, "train_time_sec": 0,
                })
            except Exception as exc:
                log.error(f"FAILED {label} s{seed}: {exc}")
                torch.cuda.empty_cache()
                all_results.append({
                    "label": label, "d_model": d_model, "d_state": d_state,
                    "seed": seed, "error": str(exc),
                    "backbone_params": None, "total_params": None,
                    "steps": 0, "best_val_loss": None, "train_time_sec": 0,
                })

            save_results(all_results, args, args.output)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  E1-SCALING RESULTS")
    print(f"{'='*60}")
    for label, _, _ in scales:
        runs = [r for r in all_results
                if r.get("label") == label and not r.get("error")]
        if not runs:
            print(f"  {label:<20} FAILED/MISSING")
            continue
        vals = [r["best_val_loss"] for r in runs if r.get("best_val_loss")]
        bp = runs[0]["backbone_params"]
        if len(vals) > 1:
            vs = f"{np.mean(vals):.4f}±{np.std(vals):.4f}"
        else:
            vs = f"{vals[0]:.4f}" if vals else "N/A"
        print(f"  {label:<20} {bp:>10,}  val={vs}")
    print()


if __name__ == "__main__":
    main()
