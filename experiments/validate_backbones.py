#!/usr/bin/env python3
"""
Backbone Validation Smoke Test

Runs 1 forward + backward pass for each backbone at batch=1, L=64.
Catches OOM, shape errors, and NaN/Inf gradients before a full training run.
No data required — synthetic input.

Usage:
    python experiments/validate_backbones.py
    python experiments/validate_backbones.py --backbone dlinoss
    python experiments/validate_backbones.py --device cpu
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

import torch

from src.config import get_benchmark_config, get_light_dlinoss_config
from src.models.spektron import Spektron, SpektronForPretraining

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

BACKBONES = ["dlinoss", "mamba", "transformer", "cnn1d", "s4d"]


def validate_backbone(backbone: str, device: str = "cpu") -> dict:
    """Run 1 forward + backward pass. Returns result dict."""
    log.info(f"  Validating {backbone}...")

    try:
        config = get_benchmark_config(backbone=backbone, seed=42)
        # Use short sequence for speed
        config.n_channels = 64
        config.pretrain.batch_size = 1

        model = Spektron(config)
        pretrain_model = SpektronForPretraining(model, config)
        pretrain_model.to(device)
        pretrain_model.train()

        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        total_params = sum(p.numel() for p in model.parameters())

        # Synthetic batch
        B, L = 1, config.n_channels
        spectrum = torch.randn(B, L, device=device)
        mask = torch.zeros(B, config.seq_len, device=device)
        mask[0, :config.seq_len // 5] = 1.0  # mask 20%

        # Forward
        output = pretrain_model(spectrum)
        recon = output["reconstruction"]

        # Backward
        target = model._create_reconstruction_target(spectrum)
        loss = torch.nn.functional.mse_loss(
            recon * output["mask"].unsqueeze(-1),
            target * output["mask"].unsqueeze(-1)
        )
        loss.backward()

        # Check for NaN/Inf in gradients
        nan_grads = [
            n for n, p in pretrain_model.named_parameters()
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
        ]

        if nan_grads:
            return {
                "backbone": backbone,
                "status": "FAIL",
                "error": f"NaN/Inf gradients in: {nan_grads[:3]}",
                "backbone_params": backbone_params,
                "total_params": total_params,
            }

        log.info(f"    PASS: loss={loss.item():.4f}, "
                 f"backbone={backbone_params:,}, total={total_params:,}")
        return {
            "backbone": backbone,
            "status": "PASS",
            "loss": loss.item(),
            "backbone_params": backbone_params,
            "total_params": total_params,
        }

    except Exception as e:
        log.error(f"    FAIL: {e}")
        return {
            "backbone": backbone,
            "status": "FAIL",
            "error": str(e),
            "backbone_params": None,
            "total_params": None,
        }


def main():
    parser = argparse.ArgumentParser(description="Backbone validation smoke test")
    parser.add_argument("--backbone", type=str, default="all")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    backbones = BACKBONES if args.backbone == "all" else [args.backbone]
    results = []

    log.info(f"Running backbone validation on {args.device}...")
    log.info(f"Backbones: {backbones}")

    for backbone in backbones:
        result = validate_backbone(backbone, args.device)
        results.append(result)

    print(f"\n{'='*60}")
    print("  BACKBONE VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  {'Backbone':<15} {'Status':>8} {'Backbone Params':>16} {'Total Params':>14}")
    print(f"  {'-'*55}")
    all_pass = True
    for r in results:
        status = r["status"]
        bp = f"{r['backbone_params']:,}" if r["backbone_params"] else "N/A"
        tp = f"{r['total_params']:,}" if r["total_params"] else "N/A"
        print(f"  {r['backbone']:<15} {status:>8} {bp:>16} {tp:>14}")
        if status != "PASS":
            all_pass = False
            print(f"    Error: {r.get('error', 'unknown')}")
    print()

    if all_pass:
        log.info("All backbones passed validation.")
    else:
        log.error("Some backbones FAILED. Fix errors before running full training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
