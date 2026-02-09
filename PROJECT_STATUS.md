# SpectralFM — Project Status

**Last Updated:** Feb 10, 2026
**Total Lines of Code:** ~2,800 (across 12 Python files)
**Status:** Code written but NOT YET TESTED. Needs debugging, smoke testing, and validation.

---

## FILE-BY-FILE STATUS

### ✅ Data (READY)
| File | Status | Notes |
|------|--------|-------|
| `data/processed/corn/*.npy` | ✅ Ready | 80 samples, 3 instruments (m5, mp5, mp6), 700 channels, 4 properties |
| `data/processed/tablet/*.npy` | ✅ Ready | 655 samples, 2 instruments, 650 channels, 3 properties |
| `data/raw/corn/corn.mat` | ✅ Ready | Original MATLAB file |
| `data/raw/tablet/nir_shootout_2002.mat` | ✅ Ready | Original MATLAB file |

**Missing data:** Pretraining corpus (ChEMBL 220K, USPTO 177K, NIST 5.2K, RRUFF 8.6K) NOT YET DOWNLOADED.

### 🟡 Config (223 lines) — LIKELY OK
`src/config.py`
- All hyperparameters defined as nested dataclasses
- Good defaults based on research
- **Potential issue:** `n_patches = 127` is hardcoded as `(2048 - 32) / 16 + 1` — verify this matches what WaveletEmbedding actually produces
- **Potential issue:** Paths are hardcoded to `/home/claude/SpectralFM/` — should be relative or configurable

### 🟡 Datasets (328 lines) — NEEDS TESTING
`src/data/datasets.py`
- Corn and Tablet dataset classes
- Augmentation pipeline (noise, baseline drift, wavelength shift, intensity scale)
- Resampling to 2048 points
- **Potential issue:** Resampling from 700 → 2048 uses `scipy.interpolate` — verify quality
- **Potential issue:** Augmentation params may need tuning
- **Missing:** Pretraining dataset class for large-scale HDF5 data
- **Missing:** DataLoader creation with proper train/val/test splits

### 🟡 Wavelet Embedding (214 lines) — NEEDS SHAPE VERIFICATION
`src/models/embedding.py`
- WavenumberPositionalEncoding: sinusoidal + learnable wavenumber projection
- WaveletEmbedding: Haar-like DWT approximation, Conv1d patching, fusion
- CLS + domain token prepending
- **Critical issue:** Haar approximation (average/difference of pairs) is simplistic. Consider using `pywt` for proper DWT preprocessing, then passing coefficients as channels
- **Potential issue:** Fusion layer input dimension calculation may be wrong — `d_model + d_model//(levels+1) * (levels+1)` may not simplify cleanly
- **Potential issue:** Output shape (B, N+2, d_model) — verify N matches config.n_patches

### 🟡 Mamba Backbone (201 lines) — NEEDS TESTING
`src/models/mamba.py`
- Pure PyTorch implementation of selective SSM (no CUDA kernels)
- SelectiveSSM: Δ projection, A/B/C/D parameters, discretization, selective scan
- MambaBlock: norm → SSM → residual
- MambaBackbone: stack of MambaBlock layers
- **Known limitation:** Pure PyTorch scan is O(n) but slow constant factor; `mamba-ssm` CUDA kernels are 5-10× faster
- **Potential issue:** Parallel scan (`pscan`) is mentioned in config but may not be implemented — verify the forward pass uses sequential scan, which works but is slow
- **Potential issue:** Discretization of continuous SSM params (A, B) via ZOH — verify math

### 🟡 MoE + KAN (240 lines) — NEEDS TESTING
`src/models/moe.py`
- KANLinear: B-spline based activation functions on edges
- Expert: 2-layer FFN with optional KAN activations
- TopKGating: noisy gating with top-k selection
- MixtureOfExperts: routing + expert combination + load balancing loss
- **Potential issue:** KAN grid initialization and B-spline basis computation — verify numerical stability
- **Potential issue:** Load balancing loss (importance loss + load loss) — verify it prevents expert collapse

### 🟡 Transformer (103 lines) — LIKELY OK
`src/models/transformer.py`
- Standard TransformerEncoderLayer: multi-head attention + FFN + LayerNorm
- TransformerEncoder: stack of layers
- **This is the simplest module — probably works**

### 🟡 Heads (316 lines) — NEEDS SHAPE VERIFICATION
`src/models/heads.py`
- VIBHead: reparameterization trick, z_chem + z_inst split, KL loss
- ReconstructionHead: MLP decoder for MSRP patch reconstruction
- RegressionHead: simple MLP for property prediction
- FNOTransferHead: Fourier Neural Operator with spectral convolution layers
- **Critical issue:** FNOTransferHead takes z_chem (128d vector) and must output a full spectrum — verify the reshape/interpolation logic
- **Potential issue:** VIB KL divergence computation — verify it's correct for multivariate Gaussian

### 🟡 Losses (429 lines) — NEEDS TESTING
`src/losses/losses.py`
- MSRPLoss: masked spectrum reconstruction loss
- ContrastiveLoss: BYOL-style instrument-invariance
- DenoisingLoss: denoising autoencoder loss
- PhysicsLoss: Beer-Lambert + smoothness + non-negativity + peak shape + derivative
- OTAlignmentLoss: Sinkhorn-based Wasserstein distance
- VIBLoss: KL divergence for information bottleneck
- MoEBalanceLoss: expert load balancing
- SpectralFMLoss: combined multi-objective loss with configurable weights
- **Potential issue:** OT loss requires POT library — may need fallback if not installed
- **Potential issue:** Physics loss terms may have wrong scaling relative to MSRP loss

### 🟡 Trainer (403 lines) — NEEDS TESTING
`src/training/trainer.py`
- PretrainTrainer: full pretraining loop with all losses
- FinetuneTrainer: LoRA fine-tuning for calibration transfer
- Checkpointing, logging, LR scheduling
- **Missing:** WandB integration (mentioned in config but not implemented)
- **Missing:** Proper validation loop
- **Potential issue:** LoRA injection into transformer layers — verify it actually adds LoRA params

### 🟡 Metrics (229 lines) — LIKELY OK
`src/evaluation/metrics.py`
- Standard regression metrics: R², RMSEP, RPD, bias, slope
- ConformalPredictor: wrapper for distribution-free prediction intervals
- TransferEvaluator: comprehensive evaluation for calibration transfer
- **Potential issue:** Conformal prediction requires `mapie` library — verify fallback

### 🟡 Main Entry (366 lines) — NEEDS TESTING
`run.py`
- CLI with argparse: `--mode pretrain|finetune|evaluate|ttt|smoke_test`
- Experiment runners for each mode
- **Good:** includes `smoke_test` mode for quick validation
- **Potential issue:** Import paths may be wrong depending on how it's invoked
- **Missing:** Proper experiment tracking / results saving

---

## KNOWN BUGS & ISSUES (Must Fix)

### Critical (Blocks Execution)
1. **Dimension mismatches**: WaveletEmbedding output N_tokens likely doesn't match config.n_patches (127). The Conv1d output depends on actual input length, padding, etc.
2. **FNO input shape**: FNOTransferHead receives z_chem (B, 128) but FNO expects (B, C, L). Need reshape/broadcast.
3. **Masking in pretrain**: `pretrain_forward` receives mask but WaveletEmbedding doesn't apply it. The mask should zero out certain patch tokens AFTER embedding.
4. **Data resampling**: Corn is 700ch, Tablet is 650ch, config expects 2048. The datasets.py resampling must work correctly.

### High Priority (Affects Results)
5. **LoRA not injected**: The model builds but LoRA adapters are never actually added to transformer layers. Need to implement `inject_lora()` method.
6. **OT loss fallback**: If POT library not installed, OTAlignmentLoss will crash.
7. **Wavelet quality**: Haar approximation loses information. Better to use `pywt.wavedec()` in preprocessing, then pass coefficients as multi-channel input.
8. **Contrastive loss**: Needs paired views of same spectrum (augmented versions). Verify the augmentation creates proper positive pairs.

### Medium Priority
9. **Paths**: Hardcoded to /home/claude — make relative
10. **Logging**: Mix of print and logging — standardize
11. **Seeding**: Config has seed=42 but it's not set everywhere
12. **Device handling**: Some tensors may end up on wrong device

---

## WHAT WORKS (Verified)

- ✅ Config loads correctly
- ✅ Corn data loads: shape (80, 700), wavelengths (700,), properties (80, 4)
- ✅ Tablet data loads: calibrate (155, 650), test (460, 650), validate (40, 650)
- ✅ Metadata JSON files are correct and informative
- ✅ Project structure is clean and modular

## WHAT'S UNTESTED

- ❓ Forward pass through full model
- ❓ Backward pass (gradients)
- ❓ Training loop
- ❓ Data augmentation pipeline
- ❓ Wavelet decomposition correctness
- ❓ Mamba SSM numerical stability
- ❓ MoE gating + load balancing
- ❓ FNO spectral convolution
- ❓ VIB reparameterization
- ❓ All loss functions
- ❓ Conformal prediction

---

## DEPENDENCIES NOT YET DOWNLOADED

### Pretraining Data
1. **ChEMBL IR-Raman** (~220K spectra): https://figshare.com — search for DreaMS/ChEMBL vibrational spectra
2. **USPTO-Spectra** (~177K): Zenodo repository
3. **NIST IR** (~5.2K): JCAMP-DX format from NIST WebBook
4. **RRUFF Raman** (~8.6K): https://rruff.info/zipped_data_files/raman/

These must be downloaded, parsed into numpy arrays, resampled to 2048 points, and stored in HDF5 format.
