#!/usr/bin/env python3
"""
Spektron: Main Run Script
================================
Bridges State Space Models and Optimal Transport for
Zero-to-Few-Shot Spectral Calibration Transfer

Usage:
    python run.py --mode pretrain
    python run.py --mode finetune --checkpoint path/to/pretrain.pt
    python run.py --mode ttt --checkpoint path/to/pretrain.pt
    python run.py --mode baselines
    python run.py --mode all
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.config import SpectralFMConfig
from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
from src.data.datasets import (
    SpectralPreprocessor, SpectralAugmentor,
    CornDataset, TabletDataset, CalibrationTransferDataset,
    build_pretrain_loader,
)
from src.losses.losses import SpectralFMPretrainLoss
from src.training.trainer import PretrainTrainer, FinetuneTrainer, TTTEvaluator
from src.evaluation.metrics import (
    compute_metrics, run_baseline_comparison, PLSBaseline,
    PiecewiseDirectStandardization,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_data(config: SpectralFMConfig):
    """Verify all data files exist."""
    data_dir = Path(config.data_dir)
    required = [
        "processed/corn/m5_spectra.npy",
        "processed/corn/mp5_spectra.npy",
        "processed/corn/mp6_spectra.npy",
        "processed/corn/properties.npy",
        "processed/corn/wavelengths.npy",
        "processed/tablet/calibrate_1.npy",
        "processed/tablet/calibrate_2.npy",
        "processed/tablet/test_1.npy",
        "processed/tablet/test_2.npy",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        print(f"ERROR: Missing data files: {missing}")
        return False
    print(f"✓ All data files verified ({len(required)} files)")
    return True


def build_model(config: SpectralFMConfig) -> SpectralFM:
    """Build Spektron model."""
    model = SpectralFM(config)
    return model


def run_pretrain(config: SpectralFMConfig):
    """Run pretraining."""
    print("=" * 60)
    print("PHASE 1: PRETRAINING")
    print("=" * 60)

    # Build model
    model = build_model(config)
    pretrain_model = SpectralFMForPretraining(model, config)

    # Build dataloaders
    train_loader = build_pretrain_loader(
        config.data_dir,
        batch_size=config.pretrain.batch_size,
        target_length=config.n_channels,
    )

    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Batch size: {config.pretrain.batch_size}")
    print(f"Max steps: {config.pretrain.max_steps}")

    # Train
    trainer = PretrainTrainer(pretrain_model, config, train_loader)
    history = trainer.train(
        max_steps=config.pretrain.max_steps,
        log_every=50,
        val_every=500,
        save_every=1000,
    )

    # Save history
    with open(Path(config.log_dir) / "pretrain_history.json", "w") as f:
        json.dump(history, f)

    return model


def run_finetune(config: SpectralFMConfig, model: SpectralFM = None,
                 checkpoint: str = None):
    """Run fine-tuning for calibration transfer."""
    print("=" * 60)
    print("PHASE 2: FINE-TUNING (Calibration Transfer)")
    print("=" * 60)

    # Load model
    if model is None:
        model = build_model(config)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=config.device)
            model_state = {k.replace("model.", ""): v
                          for k, v in ckpt["model_state_dict"].items()
                          if k.startswith("model.")}
            if model_state:
                model.load_state_dict(model_state, strict=False)
            else:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Loaded pretrained model from {checkpoint}")

    preprocessor = SpectralPreprocessor(target_length=config.n_channels)

    # ========== CORN EXPERIMENTS ==========
    print("\n--- Corn Dataset ---")
    data_dir = Path(config.data_dir)
    corn_spectra = {
        inst: np.load(data_dir / f"processed/corn/{inst}_spectra.npy")
        for inst in ["m5", "mp5", "mp6"]
    }
    corn_props = np.load(data_dir / "processed/corn/properties.npy")
    corn_wavelengths = np.load(data_dir / "processed/corn/wavelengths.npy")

    results = {}
    instrument_pairs = [("m5", "mp5"), ("m5", "mp6"), ("mp5", "mp6")]

    for source_inst, target_inst in instrument_pairs:
        pair_key = f"{source_inst}→{target_inst}"
        print(f"\n  Transfer: {pair_key}")
        results[pair_key] = {}

        for n_transfer in config.finetune.n_transfer_samples:
            if n_transfer > len(corn_props):
                continue

            # Create transfer dataset
            transfer_ds = CalibrationTransferDataset(
                corn_spectra[source_inst],
                corn_spectra[target_inst],
                corn_props[:, 0],  # moisture
                n_transfer=n_transfer,
                preprocessor=preprocessor,
            )

            transfer_loader = torch.utils.data.DataLoader(
                transfer_ds, batch_size=min(n_transfer, 16), shuffle=True
            )

            # Fine-tune
            import copy
            model_copy = copy.deepcopy(model)
            trainer = FinetuneTrainer(model_copy, config)
            ft_result = trainer.finetune(
                transfer_loader,
                n_epochs=config.finetune.epochs,
                lr=config.finetune.lr,
                patience=config.finetune.patience,
                freeze_backbone=True,
            )

            # Evaluate on full dataset
            model_copy.eval()
            with torch.no_grad():
                # Preprocess all target spectra
                all_target = []
                for spec in corn_spectra[target_inst]:
                    processed = preprocessor.process(spec, corn_wavelengths)
                    all_target.append(processed["normalized"])
                all_target = torch.tensor(np.array(all_target), dtype=torch.float32)

                output = model_copy.predict(
                    all_target.to(config.device), mc_samples=10
                )

            preds = output["prediction"].cpu().numpy()
            metrics = compute_metrics(corn_props[:, 0], preds,
                                      output.get("uncertainty", None))

            results[pair_key][n_transfer] = metrics
            print(f"    N={n_transfer}: R²={metrics['r2']:.4f}, "
                  f"RMSE={metrics['rmse']:.4f}")

            del model_copy

    # Save results
    save_dir = Path(config.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy values to python types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(save_dir / "finetune_results.json", "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    return results


def run_ttt(config: SpectralFMConfig, checkpoint: str = None):
    """Run Test-Time Training evaluation."""
    print("=" * 60)
    print("PHASE 3: TEST-TIME TRAINING (Zero-Shot)")
    print("=" * 60)

    model = build_model(config)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=config.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    preprocessor = SpectralPreprocessor(target_length=config.n_channels)
    evaluator = TTTEvaluator(model, config)

    data_dir = Path(config.data_dir)
    corn_spectra = {
        inst: np.load(data_dir / f"processed/corn/{inst}_spectra.npy")
        for inst in ["m5", "mp5", "mp6"]
    }
    corn_props = np.load(data_dir / "processed/corn/properties.npy")
    corn_wavelengths = np.load(data_dir / "processed/corn/wavelengths.npy")

    # Preprocess target instrument spectra
    for target_inst in ["mp5", "mp6"]:
        print(f"\n  TTT on target: {target_inst}")

        target_processed = []
        for spec in corn_spectra[target_inst]:
            processed = preprocessor.process(spec, corn_wavelengths)
            target_processed.append(processed["normalized"])
        target_tensor = torch.tensor(np.array(target_processed), dtype=torch.float32)
        target_values = torch.tensor(corn_props[:, 0], dtype=torch.float32)

        results = evaluator.evaluate_zero_shot(
            target_spectra_unlabeled=target_tensor,
            target_spectra_eval=target_tensor,
            target_values=target_values,
            n_ttt_steps=[0, 5, 10, 20, 50],
        )

    return results


def run_baselines(config: SpectralFMConfig):
    """Run classical baseline methods."""
    print("=" * 60)
    print("BASELINES: Classical Methods")
    print("=" * 60)

    data_dir = Path(config.data_dir)
    corn_spectra = {
        inst: np.load(data_dir / f"processed/corn/{inst}_spectra.npy")
        for inst in ["m5", "mp5", "mp6"]
    }
    corn_props = np.load(data_dir / "processed/corn/properties.npy")

    # Train/test split
    n = len(corn_props)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_train = int(n * 0.75)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    results = {}
    for source_inst, target_inst in [("m5", "mp5"), ("m5", "mp6")]:
        pair_key = f"{source_inst}→{target_inst}"
        print(f"\n  {pair_key}")

        baseline_results = run_baseline_comparison(
            corn_spectra[source_inst][train_idx],
            corn_spectra[target_inst][train_idx],
            corn_spectra[source_inst][test_idx],
            corn_spectra[target_inst][test_idx],
            corn_props[train_idx, 0],
            corn_props[test_idx, 0],
        )

        for method, metrics in baseline_results.items():
            print(f"    {method}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")

        results[pair_key] = baseline_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Spektron")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["pretrain", "finetune", "ttt", "baselines", "all"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    # Config
    config = SpectralFMConfig()
    if args.device:
        config.device = args.device
    if args.max_steps:
        config.pretrain.max_steps = args.max_steps
    if args.batch_size:
        config.pretrain.batch_size = args.batch_size

    set_seed(config.seed)
    print(f"Device: {config.device}")
    print(f"Config: {config.name}")

    # Verify data
    if not verify_data(config):
        return

    if args.mode == "baselines" or args.mode == "all":
        run_baselines(config)

    model = None
    if args.mode == "pretrain" or args.mode == "all":
        model = run_pretrain(config)

    checkpoint = args.checkpoint
    if model is None and checkpoint is None:
        ckpt_dir = Path(config.checkpoint_dir)
        candidates = list(ckpt_dir.glob("pretrain_final.pt")) + \
                     list(ckpt_dir.glob("best_pretrain.pt"))
        if candidates:
            checkpoint = str(candidates[0])

    if args.mode == "finetune" or args.mode == "all":
        run_finetune(config, model, checkpoint)

    if args.mode == "ttt" or args.mode == "all":
        run_ttt(config, checkpoint)

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
