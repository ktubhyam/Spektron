"""
Checkpoint resume utilities for fault-tolerant training.

When training on Vast.ai instances that can be preempted, this module
provides:
1. Resume state from disk (model, optimizer, scheduler)
2. Checkpoint saving at regular intervals
3. Detection of crashes/incomplete runs
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Any
import json

import torch

log = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: directory containing checkpoint files

    Returns:
        Path to latest checkpoint, or None if none found
    """
    p = Path(checkpoint_dir)
    if not p.exists():
        return None

    checkpoints = sorted(p.glob("checkpoint_step_*.pt"),
                        key=lambda x: int(x.stem.split("_")[-1]))
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(checkpoint_path: Path,
                   model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Any) -> Tuple[int, float]:
    """Load model, optimizer, scheduler state from checkpoint.

    Args:
        checkpoint_path: path to checkpoint file
        model: model to load state into
        optimizer: optimizer to load state into
        scheduler: learning rate scheduler to load state into

    Returns:
        (step, best_val_loss): current training step and best validation loss so far
    """
    if not checkpoint_path.exists():
        log.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        if 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'], strict=False)
            log.info(f"Loaded model state from {checkpoint_path.name}")

        # Load optimizer state
        if 'optimizer_state' in ckpt and optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            log.info(f"Loaded optimizer state")

        # Load scheduler state
        if 'scheduler_state' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state'])
            log.info(f"Loaded scheduler state")

        step = ckpt.get('step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

        log.info(f"Resumed from step {step}, best_val_loss={best_val_loss:.4f}")
        return step, best_val_loss

    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}")
        return 0, float('inf')


def save_checkpoint(checkpoint_dir: str, step: int,
                   model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Any,
                   best_val_loss: float,
                   keep_last_n: int = 3) -> None:
    """Save training checkpoint.

    Args:
        checkpoint_dir: directory to save checkpoint
        step: current training step
        model: model to save
        optimizer: optimizer to save
        scheduler: scheduler to save
        best_val_loss: best validation loss achieved
        keep_last_n: number of recent checkpoints to keep
    """
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)

    checkpoint_path = p / f"checkpoint_step_{step:07d}.pt"

    try:
        torch.save({
            'step': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer else None,
            'scheduler_state': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
        log.info(f"Saved checkpoint: {checkpoint_path.name}")
    except Exception as e:
        log.error(f"Failed to save checkpoint: {e}")
        return

    # Clean up old checkpoints (keep only last N)
    all_checkpoints = sorted(
        p.glob("checkpoint_step_*.pt"),
        key=lambda x: int(x.stem.split("_")[-1])
    )

    for old_checkpoint in all_checkpoints[:-keep_last_n]:
        try:
            old_checkpoint.unlink()
            log.debug(f"Cleaned up old checkpoint: {old_checkpoint.name}")
        except Exception as e:
            log.warning(f"Failed to delete old checkpoint: {e}")


def has_resumable_checkpoint(checkpoint_dir: str, target_steps: int) -> bool:
    """Check if there's a checkpoint to resume from.

    A checkpoint is considered valid for resume if:
    1. It exists
    2. It has achieved at least some progress (step > 0)

    Args:
        checkpoint_dir: directory containing checkpoints
        target_steps: total target steps for training

    Returns:
        True if there's a valid checkpoint to resume from
    """
    latest = find_latest_checkpoint(checkpoint_dir)
    if latest is None:
        return False

    try:
        ckpt = torch.load(latest, map_location='cpu')
        step = ckpt.get('step', 0)
        return step > 0 and step < target_steps
    except Exception:
        return False


def save_resume_metadata(checkpoint_dir: str, metadata: dict) -> None:
    """Save experiment metadata for debugging incomplete runs.

    Args:
        checkpoint_dir: directory for checkpoint
        metadata: dict with run info (backbone, seed, start_time, etc.)
    """
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)

    metadata_path = p / "resume_metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    except Exception as e:
        log.warning(f"Failed to save resume metadata: {e}")


def load_resume_metadata(checkpoint_dir: str) -> dict:
    """Load experiment metadata from checkpoint directory.

    Args:
        checkpoint_dir: directory containing checkpoint

    Returns:
        metadata dict, or empty dict if not found
    """
    metadata_path = Path(checkpoint_dir) / "resume_metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load resume metadata: {e}")
        return {}
