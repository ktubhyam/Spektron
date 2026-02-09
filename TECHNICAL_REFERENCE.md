# SpectralFM — Technical Reference Card

## Quick Reference: What Each Component Does & Why

### PROBLEM
Calibration transfer in spectroscopy: a model trained on Instrument A doesn't work on Instrument B because instruments produce different spectra for the same sample (baseline shifts, resolution differences, noise patterns). Traditional fix: measure 30-100 samples on BOTH instruments. Our fix: pretrain a foundation model, then transfer with ≤10 samples (or zero via TTT).

### DATA
- **Corn:** 80 corn samples, each measured on 3 NIR instruments (m5, mp5, mp6), 700 wavelengths. Properties: moisture, oil, protein, starch.
- **Tablet:** 655 pharmaceutical tablets, 2 NIR instruments, 650 wavelengths. Properties: active ingredient, weight, hardness.
- All spectra must be resampled to 2048 points (interpolation).

### ARCHITECTURE COMPONENTS

**1. WaveletEmbedding** — Converts raw spectrum into multi-scale tokens
- DWT decomposes spectrum into approx (baselines) + detail (peaks) at 4 levels
- Conv1d creates patch tokens at full resolution
- Multi-scale features are fused via linear projection
- Wavenumber positional encoding (physics-aware — encodes actual wavelength, not just position)
- Prepends [CLS] token (for global representation) and [DOMAIN] token (NIR/IR/RAMAN)

**2. MambaBackbone** — Efficient sequence modeling via selective state space
- 4 layers of selective SSM blocks
- O(n) complexity vs O(n²) for attention
- Selective mechanism: dynamically chooses what to remember/forget
- Good at: long-range baseline dependencies, filtering instrument noise
- Math: x → Δ,B,C (input-dependent) → discretize continuous SSM → selective scan

**3. MixtureOfExperts** — Conditional computation per instrument type
- 4 expert FFN networks, sparse top-2 gating
- Each expert can specialize (e.g., NIR patterns vs Raman patterns)
- Load balancing loss prevents expert collapse
- Optional: KAN (learnable spline activations) instead of ReLU in experts

**4. TransformerEncoder** — Global reasoning
- 2 standard transformer layers (attention + FFN + LayerNorm)
- Handles global dependencies that Mamba might miss
- This is where LoRA adapters are injected for fine-tuning

**5. VIBHead** — Disentangle chemistry from instrument
- Splits representation into z_chem (128d, chemistry-invariant) and z_inst (64d, instrument-specific)
- Variational: samples z from learned Gaussian (reparameterization trick)
- KL regularization prevents information leakage
- For transfer: use z_chem (chemistry) and discard z_inst (instrument artifacts)

**6. FNOTransferHead** — Resolution-independent spectral transfer
- Fourier Neural Operator operates in frequency domain
- Input: z_chem (chemistry representation)
- Output: transferred spectrum at any resolution
- Key: SpectralConv1d learns filters in Fourier space → naturally resolution-independent

**7. Test-Time Training** — Zero-shot adaptation
- At inference on new instrument: run K steps of MSRP (masked reconstruction) on unlabeled spectra
- Only updates normalization layer parameters (safest, prevents catastrophic forgetting)
- No labels needed → enables zero-shot transfer

### LOSS FUNCTIONS

| Loss | Purpose | Weight | Formula |
|------|---------|--------|---------|
| L_MSRP | Reconstruct masked patches | 1.0 | MSE(pred[masked], target[masked]) |
| L_contrastive | Instrument-invariant features | 0.3 | BYOL: -cos(z1, z2) for augmented views |
| L_denoise | Noise robustness | 0.2 | MSE(denoise(noisy), clean) |
| L_OT | Align instrument distributions | 0.1 | Sinkhorn(P_inst1, P_inst2) |
| L_physics | Chemical plausibility | 0.1 | Smoothness + non-negativity + peak shape |
| L_VIB | Disentanglement | 0.05 | KL(q(z|x) || p(z)) |
| L_MoE | Expert balance | 0.01 | CV(expert_loads)² |

### KEY METRICS
- **R²** (coefficient of determination): must be > 0.95 for corn, > 0.96 for tablet
- **RMSEP** (root mean square error of prediction): lower is better
- **RPD** (ratio of performance to deviation): > 3 is excellent
- **Bias**: systematic prediction offset
- **Target:** SpectralFM with 10 samples must beat LoRA-CT with 50 samples (R² > 0.952)

### BASELINES TO BEAT
| Method | Type | Description |
|--------|------|-------------|
| PDS | Classical | Piecewise Direct Standardization — local transfer matrix |
| SBC | Classical | Slope/Bias Correction — linear correction |
| DS | Classical | Direct Standardization — full transfer matrix |
| CCA | Classical | Canonical Correlation Analysis |
| di-PLS | Classical | Domain-invariant PLS — SOTA classical (2020) |
| CNN | DL | 1D CNN with transfer fine-tuning |
| Transformer | DL | Standard transformer, no pretraining |
| LoRA-CT | DL | Current SOTA DL (2025) — LoRA fine-tuning, R²=0.952 |

### IMPORTANT IMPLEMENTATION NOTES

1. **Resampling:** Use `scipy.interpolate.interp1d(kind='cubic')` to resample spectra to 2048 points. Never truncate.

2. **Normalization:** Apply SNV (Standard Normal Variate) normalization per spectrum: `x = (x - x.mean()) / x.std()`

3. **Augmentation:** During pretraining, apply randomly:
   - Gaussian noise (σ = 0.01)
   - Baseline drift (low-freq polynomial addition)
   - Wavelength shift (±3 channels)
   - Intensity scaling (0.95-1.05×)

4. **Masking:** Contiguous block masking (mask 3-5 adjacent patches together), not random individual patches. This forces the model to learn spectral structure, not just interpolation.

5. **LoRA injection:** Only on transformer Q/K/V matrices (not Mamba). Rank 4-8, alpha=16.

6. **Transfer evaluation protocol:**
   - Source: calibrate model on Instrument 1 data
   - Transfer: provide N paired samples from both instruments
   - Test: predict properties for Instrument 2 test samples
   - Repeat 10 times with random N-sample selections, report mean ± std

7. **OT implementation:** Use `ot.sinkhorn2()` from POT library. If POT not available, implement simple Sinkhorn:
   ```python
   def sinkhorn(a, b, M, reg=0.05, n_iter=100):
       K = torch.exp(-M / reg)
       u = torch.ones_like(a)
       for _ in range(n_iter):
           v = b / (K.T @ u)
           u = a / (K @ v)
       return (u.unsqueeze(1) * K * v.unsqueeze(0) * M).sum()
   ```

8. **GPU memory:** Full model is ~10M params. Should fit on A100 with batch_size=64. If OOM, reduce batch size or use gradient accumulation.
