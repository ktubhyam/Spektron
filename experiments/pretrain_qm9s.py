#!/usr/bin/env python3
"""
QM9S Pretraining: D-LinOSS backbone on 104K molecular spectra.

Trains SpectralFM with D-LinOSS backbone on QM9S IR+Raman spectra
using masked spectral reconstruction + physics losses + VIB.

Prerequisites:
    python -c "from src.data.qm9s import preprocess_qm9s_to_hdf5; preprocess_qm9s_to_hdf5('data/raw/qm9s')"

Usage:
    # Full training (4x RTX 5090)
    python experiments/pretrain_qm9s.py

    # Quick smoke test (CPU)
    python experiments/pretrain_qm9s.py --light --max-steps 100 --batch-size 8

    # Resume from checkpoint
    python experiments/pretrain_qm9s.py --resume checkpoints/pretrain_step_5000.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import logging

import torch
import torch.nn as nn
import numpy as np

from src.config import SpectralFMConfig, get_dlinoss_config, get_light_dlinoss_config
from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
from src.data.qm9s import build_qm9s_loaders
from src.losses.losses import SpectralFMPretrainLoss, OTAlignmentLoss
from src.training.trainer import PretrainTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
log = logging.getLogger(__name__)


def build_qm9s_config(args) -> SpectralFMConfig:
    """Build configuration for QM9S pretraining."""
    if args.light:
        config = get_light_dlinoss_config()
        # Override n_channels back to 2048 for real QM9S data
        config.n_channels = 2048
        config.pretrain.max_steps = args.max_steps or 500
        config.pretrain.batch_size = args.batch_size or 16
    else:
        config = get_dlinoss_config()
        config.pretrain.max_steps = args.max_steps or 50000
        config.pretrain.batch_size = args.batch_size or 64

    # Training hyperparameters
    config.pretrain.lr = args.lr or 3e-4
    config.pretrain.warmup_steps = min(1000, config.pretrain.max_steps // 10)
    config.pretrain.mask_type = "contiguous"
    config.pretrain.mask_ratio = 0.20
    config.pretrain.grad_clip = 1.0

    # Gradient accumulation for large sequences
    if not args.light and torch.cuda.is_available():
        # D-LinOSS with 2048 tokens at d_model=256: ~4GB per sample
        n_gpus = torch.cuda.device_count()
        if n_gpus <= 1:
            # Single GPU: small batches + high accumulation
            config.pretrain.grad_accumulation_steps = 4
            config.pretrain.batch_size = 16
        elif n_gpus <= 2:
            # 2x GPUs (e.g. 2x RTX 5060 Ti 16GB): 16/GPU, accum=2 → effective 64
            config.pretrain.batch_size = 32
            config.pretrain.grad_accumulation_steps = 2
        else:
            # 4+ GPUs (e.g. 4x RTX 5090): 16/GPU, no accum needed
            config.pretrain.batch_size = 64
            config.pretrain.grad_accumulation_steps = 1

    # Paths
    config.checkpoint_dir = "checkpoints/qm9s_dlinoss"
    config.log_dir = "logs/qm9s_dlinoss"

    return config


def main():
    parser = argparse.ArgumentParser(description="QM9S Pretraining with D-LinOSS")
    parser.add_argument("--h5-path", type=str,
                        default="data/raw/qm9s/qm9s_processed.h5",
                        help="Path to preprocessed HDF5 file")
    parser.add_argument("--light", action="store_true",
                        help="Use lightweight config for testing")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (for debugging)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    args = parser.parse_args()

    # Verify HDF5 exists
    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        log.error(f"HDF5 file not found: {h5_path}")
        log.error("Run preprocessing first:")
        log.error('  python -c "from src.data.qm9s import preprocess_qm9s_to_hdf5; '
                   'preprocess_qm9s_to_hdf5(\'data/raw/qm9s\')"')
        sys.exit(1)

    # Build config
    config = build_qm9s_config(args)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    log.info(f"Device: {config.device}")
    log.info(f"Backbone: {config.backbone} (D-LinOSS)")
    log.info(f"d_model: {config.d_model}, d_state: {config.dlinoss.d_state}")
    log.info(f"n_layers: {config.dlinoss.n_layers}")
    log.info(f"Seq len: {config.seq_len} ({'raw' if config.use_raw_embedding else 'patched'})")
    log.info(f"Batch size: {config.pretrain.batch_size}")
    log.info(f"Grad accum: {config.pretrain.grad_accumulation_steps}")
    log.info(f"Max steps: {config.pretrain.max_steps}")

    # Build data loaders
    log.info(f"Loading QM9S from {h5_path}...")
    train_loader, val_loader, test_loader = build_qm9s_loaders(
        str(h5_path),
        batch_size=config.pretrain.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    log.info(f"Train: {len(train_loader.dataset)} samples, "
             f"Val: {len(val_loader.dataset)}, "
             f"Test: {len(test_loader.dataset)}")

    # Build model
    model = SpectralFM(config)
    pretrain_model = SpectralFMForPretraining(model, config)

    # Create trainer
    run_name = f"qm9s_dlinoss_{config.dlinoss.d_state}osc_{int(time.time())}"
    if args.light:
        run_name = f"qm9s_light_{int(time.time())}"

    trainer = PretrainTrainer(
        pretrain_model, config, train_loader,
        val_loader=val_loader,
        use_wandb=not args.no_wandb,
        run_name=run_name,
    )

    # Resume if specified
    if args.resume:
        log.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    log.info("Starting QM9S pretraining...")
    history = trainer.train(
        max_steps=config.pretrain.max_steps,
        log_every=100,
        val_every=1000,
        save_every=5000,
    )

    # Save training history
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "pretrain_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Post-training: extract and save learned frequencies
    log.info("Extracting learned frequencies...")
    from experiments.analyze_frequencies import extract_frequencies, analyze_frequency_distribution

    extracted = extract_frequencies(model)
    analysis = analyze_frequency_distribution(extracted)

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "learned_frequencies.json", "w") as f:
        json.dump(analysis, f, indent=2)
    log.info(f"Frequency analysis saved to {results_dir / 'learned_frequencies.json'}")

    log.info("QM9S pretraining complete!")


if __name__ == "__main__":
    main()
