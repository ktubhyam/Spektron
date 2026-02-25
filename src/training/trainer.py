"""
SpectralFM v2: Training Loops

1. Pretraining: MSRP + Contrastive + Physics + OT + VIB
2. Fine-tuning: LoRA-based calibration transfer
3. Test-Time Training: Zero-shot adaptation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.spectral_fm import SpectralFM, SpectralFMForPretraining
from losses.losses import SpectralFMPretrainLoss, OTAlignmentLoss, CalibrationTransferLoss
from config import SpectralFMConfig
from utils.logging import ExperimentLogger


def _config_to_dict(config) -> dict:
    """Convert dataclass config to flat dict for logging."""
    from dataclasses import fields, is_dataclass
    result = {}
    for f in fields(config):
        val = getattr(config, f.name)
        if is_dataclass(val):
            for sf in fields(val):
                result[f"{f.name}.{sf.name}"] = getattr(val, sf.name)
        else:
            result[f.name] = val
    return {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool))}


class PretrainTrainer:
    """Pretraining loop for SpectralFM."""

    def __init__(self, model: SpectralFMForPretraining,
                 config: SpectralFMConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 use_wandb: bool = True,
                 run_name: Optional[str] = None,
                 wandb_entity: Optional[str] = None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        # Loss
        self.criterion = SpectralFMPretrainLoss(config)
        self.ot_loss = OTAlignmentLoss(config.ot.reg, config.ot.n_iter)

        # Optimizer — exclude LayerNorm, bias, and embeddings from weight decay
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't decay: norms, biases, embeddings, 1-d params (scales, etc.)
            if ('norm' in name or 'ln' in name or 'bias' in name
                    or 'embedding' in name or 'cls_token' in name
                    or 'mask_token' in name or 'domain' in name
                    or param.dim() <= 1):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        self.optimizer = AdamW([
            {'params': decay_params, 'weight_decay': config.pretrain.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=config.pretrain.lr)

        # Scheduler: linear warmup + cosine decay
        warmup = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=config.pretrain.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=config.pretrain.max_steps - config.pretrain.warmup_steps,
            eta_min=config.pretrain.lr * 0.01,  # Don't decay to zero
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine],
            milestones=[config.pretrain.warmup_steps],
        )

        # Mixed precision — use bfloat16 (range 3.39e38) instead of float16
        # (range 65504) to prevent overflow from D-LinOSS SSM large activations.
        # bfloat16 does not need GradScaler.
        self.use_amp = torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)  # not needed for bf16

        # Gradient accumulation
        self.grad_accum_steps = config.pretrain.grad_accumulation_steps
        self._accum_count = 0
        self._accum_losses = {}  # accumulate losses over grad_accum window

        # Move to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # Multi-GPU via DataParallel
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.n_gpus > 1:
            self.model = nn.DataParallel(self.model)
            print(f"Using DataParallel on {self.n_gpus} GPUs")

        # Logging
        self.step = 0
        self.best_val_loss = float('inf')
        self.history = []

        # Experiment logger (W&B + JSON dual backend)
        self.exp_logger = ExperimentLogger(
            project="SpectralFM",
            run_name=run_name or f"pretrain_{int(time.time())}",
            config=_config_to_dict(config),
            log_dir=config.log_dir,
            use_wandb=use_wandb,
            wandb_entity=wandb_entity,
            tags=["pretraining"],
        )

    def _reduce_dp_scalars(self, output: Dict) -> Dict:
        """Reduce scalar tensors gathered by DataParallel.

        DataParallel concatenates 0-dim scalars from each GPU into 1D tensors.
        This reduces them back to scalars via mean.
        """
        if self.n_gpus <= 1:
            return output
        if output["moe_loss"].dim() > 0:
            output["moe_loss"] = output["moe_loss"].mean()
        if "vib" in output:
            for k in ("kl_chem", "kl_inst", "kl_loss"):
                if k in output["vib"] and output["vib"][k].dim() > 0:
                    output["vib"][k] = output["vib"][k].mean()
        return output

    def train_step(self, batch: Dict):
        """Single training step with mixed precision and gradient accumulation.

        Returns:
            (losses_dict, did_step): losses averaged over the accumulation window,
            and whether an optimizer step was taken.
        """
        self.model.train()

        spectrum = batch["spectrum"].to(self.device)
        instrument_id = batch.get("instrument_id")
        if instrument_id is not None:
            instrument_id = instrument_id.to(self.device)
        # Convert domain list[str] to tensor so DataParallel can scatter it.
        # Without this, both GPUs get the full list and domain[:batch_size]
        # gives GPU1 the WRONG domains (first N instead of second N).
        domain = batch.get("domain", "NIR")
        if isinstance(domain, list):
            _dmap = {"NIR": 0, "IR": 1, "RAMAN": 2, "UNKNOWN": 3}
            domain = torch.tensor(
                [_dmap.get(d, 3) for d in domain],
                dtype=torch.long, device=self.device,
            )

        # Forward with AMP
        with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            output = self.model(spectrum, domain, instrument_id)
            output = self._reduce_dp_scalars(output)
            losses = self.criterion(output, instrument_id)

        # OT alignment (outside AMP — Sinkhorn needs fp32 for numerical stability)
        if instrument_id is not None and len(instrument_id.unique()) > 1:
            z_chem_f32 = output["z_chem"].float()
            z_norm = F.normalize(z_chem_f32, dim=-1)
            z_by_inst = {}
            for iid in instrument_id.unique():
                z_by_inst[iid.item()] = z_norm[instrument_id == iid]
            ot_loss = self.config.pretrain.ot_weight * self.ot_loss(z_by_inst)
            if not (torch.isnan(ot_loss) or torch.isinf(ot_loss)):
                losses["ot"] = ot_loss
                losses["total"] = losses["total"] + ot_loss

        # NaN guard: skip this batch if loss is NaN/Inf
        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            self.optimizer.zero_grad()
            self._accum_count = 0
            self._accum_losses = {}
            print(f"  [WARN] NaN/Inf loss at step {self.step}, skipping batch", flush=True)
            return None, False

        # Backward with gradient accumulation
        (losses["total"] / self.grad_accum_steps).backward()

        # Accumulate loss values for averaging
        for k, v in losses.items():
            self._accum_losses[k] = self._accum_losses.get(k, 0.0) + v.item()
        self._accum_count += 1

        if self._accum_count >= self.grad_accum_steps:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.pretrain.grad_clip
            )

            # Check for NaN in gradients before stepping
            has_nan_grad = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in self.model.parameters()
            )
            if has_nan_grad:
                self.optimizer.zero_grad()
                self._accum_losses = {}
                self._accum_count = 0
                print(f"  [WARN] NaN gradients at step {self.step}, skipping optimizer step", flush=True)
                return None, False

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            # Average losses over the accumulation window
            avg_losses = {k: v / self.grad_accum_steps
                          for k, v in self._accum_losses.items()}
            self._accum_losses = {}
            self._accum_count = 0
            self.step += 1
            return avg_losses, True

        return None, False

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation pass."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_losses = {}
        n_batches = 0

        for batch in self.val_loader:
            spectrum = batch["spectrum"].to(self.device)
            instrument_id = batch.get("instrument_id")
            if instrument_id is not None:
                instrument_id = instrument_id.to(self.device)
            domain = batch.get("domain", "NIR")
            if isinstance(domain, list):
                _dmap = {"NIR": 0, "IR": 1, "RAMAN": 2, "UNKNOWN": 3}
                domain = torch.tensor(
                    [_dmap.get(d, 3) for d in domain],
                    dtype=torch.long, device=self.device,
                )

            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(spectrum, domain, instrument_id)
                output = self._reduce_dp_scalars(output)
                losses = self.criterion(output, instrument_id)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    def train(self, max_steps: int = None, log_every: int = 100,
              val_every: int = 500, save_every: int = 1000):
        """Full pretraining loop."""
        max_steps = max_steps or self.config.pretrain.max_steps
        save_dir = Path(self.config.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting pretraining for {max_steps} steps...", flush=True)
        start_time = time.time()
        self.optimizer.zero_grad()

        while self.step < max_steps:
            for batch in self.train_loader:
                if self.step >= max_steps:
                    break

                losses, did_step = self.train_step(batch)

                # Only log/validate/save after an actual optimizer step
                if not did_step:
                    continue

                # Log (skip step 0 — it's before any real training)
                if self.step % log_every == 0 and self.step > 0:
                    elapsed = time.time() - start_time
                    lr = self.scheduler.get_last_lr()[0]
                    effective_batch = self.config.pretrain.batch_size * self.grad_accum_steps
                    samples_per_sec = self.step * effective_batch / max(elapsed, 1)
                    steps_per_sec = self.step / max(elapsed, 1)
                    eta_sec = (max_steps - self.step) / max(steps_per_sec, 1e-8)
                    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                    log_msg = (
                        f"Step {self.step}/{max_steps} | "
                        f"Loss: {losses['total']:.4f} | "
                        f"MSRP: {losses.get('msrp', 0):.4f} | "
                        f"Physics: {losses.get('physics', 0):.4f} | "
                        f"VIB: {losses.get('vib', 0):.4f} | "
                        f"LR: {lr:.2e} | "
                        f"{samples_per_sec:.0f} samp/s | "
                        f"ETA: {eta_sec/60:.1f}m | "
                        f"GPU: {gpu_mem_gb:.1f}GB"
                    )
                    print(log_msg, flush=True)
                    self.history.append({"step": self.step, **losses})

                    # Log to W&B + JSON
                    self.exp_logger.log({
                        "train/loss": losses["total"],
                        "train/msrp": losses.get("msrp", 0),
                        "train/physics": losses.get("physics", 0),
                        "train/vib": losses.get("vib", 0),
                        "train/moe_balance": losses.get("moe_balance", 0),
                        "train/ot": losses.get("ot", 0),
                        "train/lr": lr,
                        "train/elapsed_sec": elapsed,
                        "perf/samples_per_sec": samples_per_sec,
                        "perf/steps_per_sec": steps_per_sec,
                        "perf/eta_minutes": eta_sec / 60,
                        "perf/gpu_mem_gb": gpu_mem_gb,
                    }, step=self.step)

                # Validate
                if self.step % val_every == 0 and self.step > 0 and self.val_loader:
                    val_losses = self.validate()
                    val_msg = f"  Val Loss: {val_losses.get('total', 0):.4f}"
                    print(val_msg, flush=True)

                    # Log validation to W&B + JSON
                    self.exp_logger.log({
                        "val/loss": val_losses.get("total", 0),
                        "val/msrp": val_losses.get("msrp", 0),
                    }, step=self.step)

                    if val_losses.get('total', float('inf')) < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.save_checkpoint(save_dir / "best_pretrain.pt")
                        self.exp_logger.log_summary({"best_val_loss": self.best_val_loss, "best_step": self.step})
                        print("  -> New best model saved!", flush=True)

                # Save periodic checkpoint
                if self.step % save_every == 0 and self.step > 0:
                    self.save_checkpoint(save_dir / f"pretrain_step_{self.step}.pt")

        # Final save
        self.save_checkpoint(save_dir / "pretrain_final.pt")
        self.exp_logger.finish()
        print(f"Pretraining complete in {time.time() - start_time:.0f}s")

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        model_to_save = self.model.module if self.n_gpus > 1 else self.model
        torch.save({
            "step": self.step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model_to_load = self.model.module if self.n_gpus > 1 else self.model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float('inf'))
        print(f"Loaded checkpoint from step {self.step}", flush=True)


class FinetuneTrainer:
    """Fine-tuning for calibration transfer with LoRA."""

    def __init__(self, model: SpectralFM, config: SpectralFMConfig,
                 use_wandb: bool = True, run_name: Optional[str] = None,
                 wandb_entity: Optional[str] = None):
        self.model = model
        self.config = config
        self.device = config.device
        self.criterion = CalibrationTransferLoss()

        # Experiment logger
        self.exp_logger = ExperimentLogger(
            project="SpectralFM",
            run_name=run_name or f"finetune_{int(time.time())}",
            config=_config_to_dict(config),
            log_dir=config.log_dir,
            use_wandb=use_wandb,
            wandb_entity=wandb_entity,
            tags=["finetuning"],
        )

    def finetune(self, train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 n_epochs: int = 100, lr: float = 1e-4,
                 patience: int = 15,
                 freeze_backbone: bool = True) -> Dict:
        """Fine-tune for calibration transfer.

        Args:
            train_loader: paired (source, target) spectra + labels
            val_loader: validation data
            n_epochs: max epochs
            lr: learning rate
            patience: early stopping patience
            freeze_backbone: whether to freeze encoder
        """
        self.model.to(self.device)

        if freeze_backbone:
            self.model.freeze_backbone()

        # Only optimize unfrozen parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_loss = float('inf')
        patience_counter = 0
        history = []

        for epoch in range(n_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                source = batch["source_spectrum"].to(self.device)
                target_val = batch["target_value"].to(self.device)

                output = self.model.transfer_forward(source)
                loss = self.criterion(output["prediction"], target_val)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            scheduler.step()

            # Validate
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    n_val = 0
                    for batch in val_loader:
                        source = batch["source_spectrum"].to(self.device)
                        target_val = batch["target_value"].to(self.device)
                        output = self.model.transfer_forward(source)
                        val_loss += self.criterion(
                            output["prediction"], target_val
                        ).item()
                        n_val += 1
                    val_loss /= max(n_val, 1)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in
                                  self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        self.model.load_state_dict(best_state)
                        break

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

            # Log to W&B + JSON
            self.exp_logger.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
            }, step=epoch)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{n_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        self.exp_logger.log_summary({"best_val_loss": best_val_loss})
        self.exp_logger.finish()
        self.model.unfreeze_all()
        return {"history": history, "best_val_loss": best_val_loss}


class TTTEvaluator:
    """Test-Time Training evaluation.

    Tests zero-shot calibration transfer using only self-supervised
    adaptation on unlabeled target instrument spectra.
    """

    def __init__(self, model: SpectralFM, config: SpectralFMConfig):
        self.model = model
        self.config = config
        self.device = config.device

    def evaluate_zero_shot(self, target_spectra_unlabeled: torch.Tensor,
                           target_spectra_eval: torch.Tensor,
                           target_values: torch.Tensor,
                           n_ttt_steps: List[int] = [0, 5, 10, 20, 50],
                           ) -> Dict:
        """Evaluate zero-shot TTT at different adaptation budgets.

        Args:
            target_spectra_unlabeled: (N_unlabeled, L) for TTT adaptation
            target_spectra_eval: (N_eval, L) for evaluation
            target_values: (N_eval,) ground truth
            n_ttt_steps: list of TTT step counts to evaluate

        Returns:
            dict mapping n_steps → metrics
        """
        import copy
        results = {}

        for n_steps in n_ttt_steps:
            # Copy model to avoid contamination between evaluations
            model_copy = copy.deepcopy(self.model)
            model_copy.to(self.device)

            if n_steps > 0:
                # Apply TTT
                model_copy.test_time_train(
                    target_spectra_unlabeled.to(self.device),
                    n_steps=n_steps,
                    lr=self.config.ttt.lr,
                    mask_ratio=self.config.ttt.mask_ratio,
                    adapt_layers=self.config.ttt.adapt_layers,
                )

            # Evaluate
            model_copy.eval()
            with torch.no_grad():
                output = model_copy.predict(
                    target_spectra_eval.to(self.device),
                    mc_samples=10,
                )

            preds = output["prediction"].cpu().numpy()
            targets = target_values.numpy()
            uncertainty = output.get("uncertainty", torch.zeros_like(
                output["prediction"])).cpu().numpy()

            # Metrics
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(targets, preds.flatten())
            rmse = np.sqrt(mean_squared_error(targets, preds.flatten()))

            results[n_steps] = {
                "r2": r2,
                "rmse": rmse,
                "predictions": preds,
                "uncertainty": uncertainty,
            }

            print(f"TTT steps={n_steps}: R²={r2:.4f}, RMSE={rmse:.4f}")

            del model_copy

        return results
