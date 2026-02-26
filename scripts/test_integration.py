#!/usr/bin/env python3
"""
Pre-GPU Integration Test

Validates the FULL pipeline end-to-end on CPU with real (tiny) data.
Run this BEFORE provisioning GPU to catch integration bugs.

Tests:
1. Model build + LoRA injection
2. Data loading (corn + tablet)  
3. Preprocessing pipeline
4. Forward pass (pretrain + transfer + predict)
5. TTT loop (1 step)
6. Fine-tuning loop (2 epochs)
7. Baseline runner
8. Visualization (generates test figure)
9. Experiment JSON save/load
10. Metric computation

Usage:
    python scripts/test_integration.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
import tempfile
from pathlib import Path

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✓ {name}")
        PASS += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        FAIL += 1
        import traceback
        traceback.print_exc()


def main():
    global PASS, FAIL
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"

    print("\n" + "="*60)
    print("Spektron: Pre-GPU Integration Test")
    print("="*60)

    # ── 1. Model Build ──
    print("\n[1] Model Build")

    def test_model_build():
        from src.config import SpectralFMConfig
        from src.models.spectral_fm import SpectralFM
        config = SpectralFMConfig()
        model = SpectralFM(config)
        assert sum(p.numel() for p in model.parameters()) > 1_000_000
    test("Build Spektron", test_model_build)

    def test_lora_inject():
        from src.config import SpectralFMConfig
        from src.models.spectral_fm import SpectralFM
        from src.models.lora import inject_lora
        model = SpectralFM(SpectralFMConfig())
        before = sum(p.numel() for p in model.parameters())
        inject_lora(model, ["q_proj", "k_proj", "v_proj"], rank=8, alpha=16)
        after = sum(p.numel() for p in model.parameters())
        assert after > before, "LoRA should add parameters"
    test("LoRA injection", test_lora_inject)

    # ── 2. Data Loading ──
    print("\n[2] Data Loading")

    def test_corn_data():
        from scripts.run_finetune import load_corn_data
        d = load_corn_data(data_dir, "m5", "mp6", 0, n_transfer=10, seed=42)
        assert d["X_source_train"].shape[0] == 10
        assert d["X_target_test"].shape[0] > 0
    test("Corn data loading", test_corn_data)

    def test_tablet_data():
        from scripts.run_finetune import load_tablet_data
        d = load_tablet_data(data_dir, 0, n_transfer=20, seed=42)
        assert d["X_source_train"].shape[0] == 20
    test("Tablet data loading", test_tablet_data)

    # ── 3. Preprocessing ──
    print("\n[3] Preprocessing")

    def test_preprocess():
        from scripts.run_finetune import preprocess_spectra
        spectra = np.random.randn(5, 700)
        wl = np.linspace(1100, 2500, 700)
        X = preprocess_spectra(spectra, wl)
        assert X.shape == (5, 2048), f"Expected (5, 2048), got {X.shape}"
    test("Preprocessing to 2048", test_preprocess)

    # ── 4. Forward Passes ──
    print("\n[4] Forward Passes")

    from src.config import SpectralFMConfig
    from src.models.spectral_fm import SpectralFM
    config = SpectralFMConfig()
    model = SpectralFM(config)

    def test_encode():
        x = torch.randn(2, 2048)
        enc = model.encode(x, domain="NIR")
        assert "z_chem" in enc
        assert "z_inst" in enc
        assert enc["z_chem"].shape[0] == 2
    test("Encode forward", test_encode)

    def test_predict():
        x = torch.randn(2, 2048)
        out = model.predict(x, domain="NIR", mc_samples=1)
        assert "prediction" in out
    test("Predict forward", test_predict)

    def test_mc_dropout():
        x = torch.randn(3, 2048)
        out = model.predict(x, domain="NIR", mc_samples=5)
        assert "uncertainty" in out
        assert out["uncertainty"].shape[0] == 3
    test("MC Dropout", test_mc_dropout)

    def test_pretrain_forward():
        x = torch.randn(2, 2048)
        mask = torch.zeros(2, config.n_patches)
        mask[:, :5] = 1  # mask first 5 patches
        out = model.pretrain_forward(x, mask, domain="RAMAN")
        assert "reconstruction" in out
    test("Pretrain forward", test_pretrain_forward)

    # ── 5. TTT Loop ──
    print("\n[5] TTT Loop")

    def test_ttt():
        x = torch.randn(10, 2048)
        model.test_time_train(x, n_steps=1, lr=1e-4)
    test("TTT (1 step)", test_ttt)

    # ── 6. Fine-Tuning ──
    print("\n[6] Fine-Tuning")

    def test_finetune():
        from src.models.lora import inject_lora
        from scripts.run_finetune import finetune_spectral_fm
        ft_model = SpectralFM(config)
        inject_lora(ft_model, ["q_proj", "k_proj", "v_proj"], rank=4, alpha=8)
        X = torch.randn(8, 2048)
        y = torch.randn(8, 1)
        trained, history = finetune_spectral_fm(
            ft_model, X, y, device="cpu", n_epochs=2, lr=1e-3, batch_size=4)
        assert len(history) > 0
    test("Fine-tune (2 epochs)", test_finetune)

    # ── 7. Baselines ──
    print("\n[7] Baselines")

    def test_baselines():
        from src.evaluation.baselines import run_baseline_comparison
        X_src = np.random.randn(30, 700)
        X_tgt = np.random.randn(30, 700)
        X_src_test = np.random.randn(10, 700)
        X_tgt_test = np.random.randn(10, 700)
        y_train = np.random.randn(30)
        y_test = np.random.randn(10)
        results = run_baseline_comparison(X_src, X_tgt, X_src_test, X_tgt_test, y_train, y_test)
        assert "PLS" in results or "DS" in results
    test("Baseline comparison", test_baselines)

    # ── 8. Visualization ──
    print("\n[8] Visualization")

    def test_viz():
        from src.evaluation.visualization import plot_sample_efficiency
        spectral = {1: {"r2_mean": 0.1, "r2_std": 0.05},
                    5: {"r2_mean": 0.4, "r2_std": 0.03},
                    10: {"r2_mean": 0.6, "r2_std": 0.02}}
        baselines = {"DS": {"r2": 0.5}, "PDS": {"r2": 0.3}}
        with tempfile.TemporaryDirectory() as td:
            plot_sample_efficiency(spectral, baselines, figures_dir=td)
            assert (Path(td) / "sample_efficiency.pdf").exists()
    test("Sample efficiency plot", test_viz)

    def test_latex():
        from src.evaluation.visualization import generate_latex_table
        results = {
            "m5→mp6 moisture": {
                "Spektron": {"r2_mean": 0.8, "r2_std": 0.02},
                "DS": {"r2_mean": 0.65, "r2_std": 0.05},
            }
        }
        latex = generate_latex_table(results, ["Spektron", "DS"])
        assert "\\begin{table}" in latex
        assert "\\textbf" in latex
    test("LaTeX table generation", test_latex)

    # ── 9. JSON Persistence ──
    print("\n[9] JSON Persistence")

    def test_json():
        data = {
            "r2": np.float64(0.85),
            "array": np.array([1, 2, 3]),
            "int": np.int64(42),
        }
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        s = json.dumps(data, default=convert)
        loaded = json.loads(s)
        assert loaded["r2"] == 0.85
    test("NumPy JSON serialization", test_json)

    # ── 10. Metrics ──
    print("\n[10] Metrics")

    def test_metrics():
        from src.evaluation.baselines import compute_metrics
        y_true = np.array([1, 2, 3, 4, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.8, 5.2])
        m = compute_metrics(y_true, y_pred)
        assert m["r2"] > 0.9
        assert m["rmsep"] < 0.5
    test("Metric computation", test_metrics)

    # ── Summary ──
    print("\n" + "="*60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"✅ ALL {total} TESTS PASSED — Ready for GPU!")
    else:
        print(f"⚠️  {PASS}/{total} passed, {FAIL} FAILED — Fix before GPU!")
    print("="*60 + "\n")

    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
