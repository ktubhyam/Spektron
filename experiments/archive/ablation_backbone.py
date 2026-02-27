#!/usr/bin/env python3
"""
Quick Ablation: D-LinOSS vs Mamba backbone on existing corpus.

Runs a short pretraining (500 steps) on corn+tablet data with both
backbone configurations, comparing:
  - Training loss curve
  - Forward/backward speed
  - Memory usage
  - Reconstruction quality

Usage:
    python experiments/ablation_backbone.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json

from src.config import SpectralFMConfig, get_light_config, get_light_dlinoss_config
from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
from src.data.datasets import build_pretrain_loader
from src.losses.losses import SpectralFMPretrainLoss


def run_ablation(config: SpectralFMConfig, name: str, max_steps: int = 500,
                 log_every: int = 50) -> dict:
    """Run a short pretraining and collect metrics."""
    print(f"\n{'='*60}")
    print(f"  Ablation: {name} ({config.backbone.upper()} backbone)")
    print(f"{'='*60}")

    device = config.device
    torch.manual_seed(config.seed)

    # Build model
    model = SpectralFM(config)
    pretrain_model = SpectralFMForPretraining(model, config)
    pretrain_model.to(device)

    criterion = SpectralFMPretrainLoss(config)
    criterion.to(device)

    # Build data
    train_loader = build_pretrain_loader(
        config.data_dir,
        batch_size=config.pretrain.batch_size,
        target_length=config.n_channels,
    )
    print(f"  Data: {len(train_loader.dataset)} samples")
    print(f"  Seq len: {config.seq_len} ({'raw' if config.use_raw_embedding else 'patched'})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        pretrain_model.parameters(),
        lr=config.pretrain.lr,
        weight_decay=config.pretrain.weight_decay,
    )

    # Training metrics
    history = []
    total_forward_time = 0
    total_backward_time = 0
    step = 0

    pretrain_model.train()
    optimizer.zero_grad()

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            spectrum = batch["spectrum"].to(device)
            instrument_id = batch.get("instrument_id")
            if instrument_id is not None:
                instrument_id = instrument_id.to(device)
            domain = batch.get("domain", "NIR")

            # Forward
            t0 = time.perf_counter()
            output = pretrain_model(spectrum, domain, instrument_id)
            losses = criterion(output, instrument_id)
            t_forward = time.perf_counter() - t0
            total_forward_time += t_forward

            # Backward
            t0 = time.perf_counter()
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), config.pretrain.grad_clip)
            optimizer.step()
            t_backward = time.perf_counter() - t0
            total_backward_time += t_backward

            step += 1

            if step % log_every == 0 or step == 1:
                entry = {
                    "step": step,
                    "loss": losses["total"].item(),
                    "msrp": losses.get("msrp", torch.tensor(0.0)).item(),
                    "physics": losses.get("physics", torch.tensor(0.0)).item(),
                    "vib": losses.get("vib", torch.tensor(0.0)).item(),
                    "forward_ms": t_forward * 1000,
                    "backward_ms": t_backward * 1000,
                }
                history.append(entry)
                print(f"  Step {step:4d}/{max_steps} | "
                      f"Loss: {entry['loss']:.4f} | "
                      f"MSRP: {entry['msrp']:.4f} | "
                      f"Fwd: {entry['forward_ms']:.0f}ms | "
                      f"Bwd: {entry['backward_ms']:.0f}ms")

    # Summary
    n_params = sum(p.numel() for p in model.parameters())
    avg_fwd = total_forward_time / max_steps * 1000
    avg_bwd = total_backward_time / max_steps * 1000
    final_loss = history[-1]["loss"] if history else float("inf")

    summary = {
        "name": name,
        "backbone": config.backbone,
        "n_params": n_params,
        "seq_len": config.seq_len,
        "d_model": config.d_model,
        "avg_forward_ms": round(avg_fwd, 1),
        "avg_backward_ms": round(avg_bwd, 1),
        "avg_step_ms": round(avg_fwd + avg_bwd, 1),
        "final_loss": round(final_loss, 6),
        "initial_loss": round(history[0]["loss"], 6) if history else None,
        "loss_reduction": round(history[0]["loss"] - final_loss, 6) if len(history) > 1 else 0,
        "history": history,
    }

    print(f"\n  Summary:")
    print(f"    Params: {n_params:,}")
    print(f"    Avg forward:  {avg_fwd:.1f} ms")
    print(f"    Avg backward: {avg_bwd:.1f} ms")
    print(f"    Avg step:     {avg_fwd + avg_bwd:.1f} ms")
    print(f"    Loss: {history[0]['loss']:.4f} → {final_loss:.4f} "
          f"(Δ={history[0]['loss'] - final_loss:.4f})")

    return summary


def main():
    max_steps = 500
    batch_size = 16  # Small to fit CPU

    # ---- Mamba config ----
    mamba_cfg = get_light_config()
    mamba_cfg.pretrain.batch_size = batch_size
    mamba_cfg.pretrain.lr = 3e-4
    mamba_cfg.pretrain.mask_type = "contiguous"

    # ---- D-LinOSS config ----
    dlinoss_cfg = get_light_dlinoss_config()
    dlinoss_cfg.pretrain.batch_size = batch_size
    dlinoss_cfg.pretrain.lr = 3e-4
    dlinoss_cfg.pretrain.mask_type = "contiguous"

    # Run both
    mamba_results = run_ablation(mamba_cfg, "Mamba (patched)", max_steps=max_steps)
    dlinoss_results = run_ablation(dlinoss_cfg, "D-LinOSS (raw)", max_steps=max_steps)

    # Comparison
    print("\n" + "=" * 60)
    print("  COMPARISON: Mamba vs D-LinOSS")
    print("=" * 60)

    headers = ["Metric", "Mamba", "D-LinOSS"]
    rows = [
        ["Backbone", mamba_results["backbone"], dlinoss_results["backbone"]],
        ["Seq length", str(mamba_results["seq_len"]), str(dlinoss_results["seq_len"])],
        ["Parameters", f"{mamba_results['n_params']:,}", f"{dlinoss_results['n_params']:,}"],
        ["Avg forward (ms)", f"{mamba_results['avg_forward_ms']:.1f}", f"{dlinoss_results['avg_forward_ms']:.1f}"],
        ["Avg backward (ms)", f"{mamba_results['avg_backward_ms']:.1f}", f"{dlinoss_results['avg_backward_ms']:.1f}"],
        ["Avg step (ms)", f"{mamba_results['avg_step_ms']:.1f}", f"{dlinoss_results['avg_step_ms']:.1f}"],
        ["Initial loss", f"{mamba_results['initial_loss']:.4f}", f"{dlinoss_results['initial_loss']:.4f}"],
        ["Final loss", f"{mamba_results['final_loss']:.4f}", f"{dlinoss_results['final_loss']:.4f}"],
        ["Loss reduction", f"{mamba_results['loss_reduction']:.4f}", f"{dlinoss_results['loss_reduction']:.4f}"],
    ]

    # Print table
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(3)]
    fmt = "  {:<{}} | {:>{}} | {:>{}}"
    print(fmt.format(headers[0], col_widths[0], headers[1], col_widths[1], headers[2], col_widths[2]))
    print("  " + "-" * (sum(col_widths) + 6))
    for row in rows:
        print(fmt.format(row[0], col_widths[0], row[1], col_widths[1], row[2], col_widths[2]))

    # Save results
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip history for the comparison file
    for r in [mamba_results, dlinoss_results]:
        r.pop("history", None)

    with open(out_dir / "ablation_backbone.json", "w") as f:
        json.dump({"mamba": mamba_results, "dlinoss": dlinoss_results}, f, indent=2)
    print(f"\nSaved to {out_dir / 'ablation_backbone.json'}")


if __name__ == "__main__":
    main()
