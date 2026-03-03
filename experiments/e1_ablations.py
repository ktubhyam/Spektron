#!/usr/bin/env python3
"""
E1 Ablations: D-LinOSS architecture ablation studies.

Implements:
- A1: No damping (G=0, underdamped oscillators)
- A6: Shallow (2 layers instead of 4)

Compares against:
- Baseline: Full D-LinOSS (4 layers, damped)

Usage:
    python experiments/e1_ablations.py --h5-path data/raw/qm9s/qm9s_processed.h5
"""
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

ABLATIONS = {
    "baseline": {"ablation_no_damping": False, "ablation_shallow": False},
    "A1_no_damping": {"ablation_no_damping": True, "ablation_shallow": False},
    "A6_shallow": {"ablation_no_damping": False, "ablation_shallow": True},
}


def count_backbone_params(model: Spektron) -> int:
    """Count only backbone parameters."""
    return sum(p.numel() for p in model.backbone.parameters())


def run_experiment(
    h5_path: str,
    ablation_name: str,
    ablation_params: dict,
    seed: int,
    max_steps: int = 50000,
    batch_size: int = 16,
    num_workers: int = 4,
    _force_single: bool = False,
) -> dict:
    """Run single ablation experiment."""
    log.info(f"\n{'='*70}")
    log.info(f"Ablation: {ablation_name} (seed={seed})")
    log.info(f"{'='*70}")

    # Config
    config = get_benchmark_config(backbone="dlinoss", seed=seed)
    config.pretrain.max_steps = max_steps
    config.pretrain.batch_size = batch_size

    # Apply ablation parameters
    config.dlinoss.ablation_no_damping = ablation_params["ablation_no_damping"]
    config.dlinoss.ablation_shallow = ablation_params["ablation_shallow"]

    log.info(f"  ablation_no_damping: {config.dlinoss.ablation_no_damping}")
    log.info(f"  ablation_shallow: {config.dlinoss.ablation_shallow}")

    # Data
    train_loader, val_loader = build_qm9s_loaders(
        h5_path=h5_path,
        batch_size=config.pretrain.batch_size,
        num_workers=num_workers,
        split_train=0.85,
        seed=seed,
    )

    # Model
    backbone_kwargs = {
        "d_model": config.dlinoss.d_model,
        "n_layers": config.dlinoss.n_layers,
        "d_state": config.dlinoss.d_state,
        "dropout": config.dlinoss.dropout,
        "bidirectional": True,
    }
    if config.dlinoss.ablation_no_damping:
        backbone_kwargs["layer_name"] = "IM"  # Use undamped layer
    else:
        backbone_kwargs["layer_name"] = "Damped"

    spektron = Spektron(
        backbone="dlinoss",
        backbone_kwargs=backbone_kwargs,
        d_model=config.d_model,
        config=config,
    )

    pretrain_model = SpektronForPretraining(spektron, config)
    pretrain_model = pretrain_model.to(config.device)

    backbone_params = count_backbone_params(spektron)
    total_params = sum(p.numel() for p in pretrain_model.parameters())
    log.info(f"  Backbone params: {backbone_params:,}")
    log.info(f"  Total params: {total_params:,}")

    # Train
    run_name = f"e1_ablation_{ablation_name}_s{seed}_{int(time.time())}"
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
        save_every=config.pretrain.max_steps,
    )
    elapsed = time.time() - t0

    # Extract metrics
    result = {
        "ablation": ablation_name,
        "seed": seed,
        "ablation_params": ablation_params,
        "backbone_params": backbone_params,
        "total_params": total_params,
        "train_time_sec": elapsed,
        "best_val_loss": trainer.best_val_loss,
        "steps": config.pretrain.max_steps,
    }

    if torch.cuda.is_available():
        result["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()

    log.info(f"  Done: val_loss={result['best_val_loss']:.4f}, time={elapsed:.0f}s")

    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="E1 Ablations: D-LinOSS architecture studies")
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--ablation", type=str, default="all",
                        help="Single ablation or 'all'")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (500 steps, 1 seed)")
    parser.add_argument("--output", type=str,
                        default="experiments/results/e1_ablations.json")
    args = parser.parse_args()

    if args.quick:
        args.max_steps = 500
        args.seeds = 1

    ablations = ABLATIONS if args.ablation == "all" else {args.ablation: ABLATIONS[args.ablation]}

    log.info(f"\n{'='*70}")
    log.info(f"E1 ABLATIONS: D-LinOSS Architecture Studies")
    log.info(f"{'='*70}")
    log.info(f"Ablations: {list(ablations.keys())}")
    log.info(f"Seeds: {args.seeds}")
    log.info(f"Max steps: {args.max_steps}")

    all_results = []

    for ablation_name, ablation_params in ablations.items():
        for seed in range(42, 42 + args.seeds):
            try:
                result = run_experiment(
                    h5_path=args.h5_path,
                    ablation_name=ablation_name,
                    ablation_params=ablation_params,
                    seed=seed,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                all_results.append(result)
            except Exception as e:
                log.error(f"  Error: {e}")
                all_results.append({
                    "ablation": ablation_name,
                    "seed": seed,
                    "error": str(e),
                })

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "experiment": "E1 Ablations",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "max_steps": args.max_steps,
                "seeds": args.seeds,
                "batch_size": args.batch_size,
            },
            "runs": all_results,
        }, f, indent=2)

    log.info(f"\n✓ Results saved to {args.output}")

    # Summary
    by_ablation = {}
    for ablation_name in ablations.keys():
        runs = [r for r in all_results
                if r.get("ablation") == ablation_name and r.get("error") is None]
        if runs:
            val_losses = [r["best_val_loss"] for r in runs]
            log.info(f"{ablation_name}: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")


if __name__ == "__main__":
    main()
