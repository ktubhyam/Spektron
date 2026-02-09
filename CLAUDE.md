# SpectralFM — Claude Code Instructions

## What This Project Is

SpectralFM is a **research paper implementation** for a novel ML architecture targeting publication in Analytical Chemistry (ACS, IF 7.4). It's the **first self-supervised foundation model for vibrational spectroscopy** that achieves few-shot calibration transfer across instruments and modalities.

**Paper Title:** "Bridging State Space Models and Optimal Transport for Zero-to-Few-Shot Spectral Calibration Transfer"

**Author:** Tubhyam Karthikeyan (ICT Mumbai / InvyrAI)

## Architecture Overview

```
Spectrum (B, 2048) 
  → WaveletEmbedding (DWT multi-scale + Conv1d patching + wavenumber PE + [CLS] + [DOMAIN])
  → MambaBackbone (4 selective SSM blocks, O(n) complexity)
  → MixtureOfExperts (4 experts, top-2 gating, optional KAN activations)
  → TransformerEncoder (2 blocks, 8 heads, global reasoning)
  → VIBHead (disentangle z_chem ∈ R^128 + z_inst ∈ R^64)
  → Heads: ReconstructionHead (pretraining) | RegressionHead (prediction) | FNOTransferHead (transfer)
```

## Project Structure

```
SpectralFM/
├── CLAUDE.md              ← YOU ARE HERE
├── PROJECT_STATUS.md      ← Current state, bugs, TODOs
├── IMPLEMENTATION_PLAN.md ← Detailed task list with priorities
├── requirements.txt
├── run.py                 ← Entry point (pretrain/finetune/evaluate)
├── src/
│   ├── config.py          ← All hyperparameters (dataclass-based)
│   ├── data/
│   │   └── datasets.py    ← Data loading, augmentation, wavelet preprocessing
│   ├── models/
│   │   ├── embedding.py   ← WaveletEmbedding + WavenumberPE
│   │   ├── mamba.py       ← Pure PyTorch Mamba (selective SSM)
│   │   ├── moe.py         ← MixtureOfExperts + KAN layers
│   │   ├── transformer.py ← Lightweight TransformerEncoder
│   │   ├── heads.py       ← VIB, Reconstruction, Regression, FNO heads
│   │   └── spectral_fm.py ← Full model assembly + TTT method
│   ├── losses/
│   │   └── losses.py      ← MSRP, contrastive, physics, OT, VIB, MoE losses
│   ├── training/
│   │   └── trainer.py     ← Pretrain + finetune + TTT training loops
│   └── evaluation/
│       └── metrics.py     ← R², RMSEP, RPD, bias, conformal prediction
├── data/
│   ├── raw/
│   │   ├── corn/corn.mat
│   │   └── tablet/nir_shootout_2002.mat
│   └── processed/
│       ├── corn/          ← 80 samples × 3 instruments × 700 channels
│       │   ├── m5_spectra.npy (80, 700)
│       │   ├── mp5_spectra.npy (80, 700)
│       │   ├── mp6_spectra.npy (80, 700)
│       │   ├── properties.npy (80, 4) [moisture, oil, protein, starch]
│       │   └── wavelengths.npy (700,)
│       └── tablet/        ← 655 samples × 2 instruments × 650 channels
│           ├── calibrate_1.npy (155, 650)  ← instrument 1 calibration
│           ├── calibrate_2.npy (155, 650)  ← instrument 2 calibration
│           ├── calibrate_Y.npy (155, 3)    ← [active, weight, hardness]
│           ├── test_1.npy (460, 650)
│           ├── test_2.npy (460, 650)
│           ├── test_Y.npy (460, 3)
│           ├── validate_1.npy (40, 650)
│           ├── validate_2.npy (40, 650)
│           └── validate_Y.npy (40, 3)
├── paper/
│   ├── BRAINSTORM_V2.md
│   └── RESEARCH_FULL_REFERENCE.md
├── checkpoints/
├── logs/
├── figures/
├── experiments/
└── notebooks/
```

## Key Technical Decisions

1. **Hybrid Mamba-Transformer** — Mamba (O(n)) handles long-range spectral dependencies, Transformer (O(n²)) adds global expressiveness. All 3 (pure Mamba, pure Transformer, hybrid) should be ablatable.
2. **Wavelet decomposition for embedding** — DWT separates sharp peaks (detail coeffs) from baselines (approx coeffs). Currently uses Haar-like approximation for differentiability; should use `pywt` for actual DWT in data preprocessing then pass coefficients to the model.
3. **Optimal Transport alignment** — Sinkhorn-based Wasserstein distance between latent distributions of different instruments. Use `POT` library.
4. **Physics-informed losses** — Beer-Lambert (linearity), non-negativity, smoothness, peak shape constraints. These are soft regularizers, not hard constraints.
5. **VIB disentanglement** — Split latent into z_chem (chemistry, transferable) and z_inst (instrument, discardable). Reparameterization trick + KL regularization.
6. **FNO transfer head** — Fourier Neural Operator for resolution-independent spectral mapping.
7. **Test-Time Training** — At inference on a new instrument, run K steps of MSRP self-supervision on unlabeled spectra to adapt. Enables zero-shot calibration transfer.

## Coding Standards

- **Python 3.10+**, PyTorch 2.x
- Type hints everywhere
- Docstrings on all public methods
- Config-driven (no magic numbers in code — everything from `SpectralFMConfig`)
- Each module independently testable
- Use `einops` for tensor reshaping where it improves readability
- Logging via Python `logging` module (not print statements)
- Reproducibility: all random seeds set via config.seed

## Important Constraints

- **Compute:** Colab Pro+ with A100 (40GB VRAM, ~24h max continuous runtime)
- **Data:** Corn (80 × 3 instruments × 700ch) and Tablet (655 × 2 instruments × 650ch) are SMALL datasets. Pretraining data (400K+ spectra from ChEMBL, USPTO, NIST, RRUFF) still needs to be downloaded and preprocessed.
- **All spectra must be resampled to 2048 points** for uniform input (pad/interpolate shorter spectra)
- **Key benchmark target:** Must beat LoRA-CT (R² = 0.952 on corn moisture) with ≤10 transfer samples

## What Needs Doing (See IMPLEMENTATION_PLAN.md for Details)

**PHASE 1 — Make It Run (HIGH PRIORITY):**
- [ ] Install dependencies and verify all imports work
- [ ] Fix any import/shape/dimension bugs in existing code
- [ ] Write a minimal smoke test: random data → forward pass → loss backward
- [ ] Verify data loading pipeline end-to-end

**PHASE 2 — Pretraining Data Pipeline:**
- [ ] Download + preprocess pretraining datasets (ChEMBL IR-Raman 220K, USPTO 177K, NIST 5.2K, RRUFF 8.6K)
- [ ] Unified HDF5 format with metadata
- [ ] Resample all spectra to 2048 points
- [ ] Data augmentation pipeline

**PHASE 3 — Training:**
- [ ] Pretraining loop (MSRP + multi-loss)
- [ ] Fine-tuning loop (LoRA-based transfer)
- [ ] TTT implementation
- [ ] Baseline implementations (PDS, SBC, DS, CCA, di-PLS)

**PHASE 4 — Experiments:**
- [ ] E1-E12 (see IMPLEMENTATION_PLAN.md)
- [ ] Ablation studies
- [ ] Figure generation

**PHASE 5 — Paper:**
- [ ] Results tables
- [ ] Architecture diagram
- [ ] Writing
