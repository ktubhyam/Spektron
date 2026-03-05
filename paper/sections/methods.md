# 2. Background and Methods

## 2.1 D-LinOSS: Damped Linear Oscillatory SSMs

The D-LinOSS model [CITE] is a state-space model whose hidden dynamics implement
second-order damped harmonic oscillators. In continuous time, each hidden channel
evolves as:

    z̈(t) + g · ż(t) + a · z(t) = b · u(t)

where u(t) is the input signal, z(t) is the hidden state, and (a, b, g) are
learnable per-channel parameters constrained to a > 0, g > 0 (ensuring stability).
The transfer function in the z-domain is:

    H(z) = dt² · b · z / [(1 + dt·g)·z² − (2 + dt·g − dt²·a)·z + 1]

The model exhibits oscillatory responses when underdamped (g² < 4a) and monotone
exponential decay when overdamped. The poles of H(z) lie on the unit circle when
g → 0 (pure oscillation) and inside the unit circle for g > 0 (decaying).

Bidirectional processing uses 4 forward + 4 backward LinOSS layers (8 total blocks),
each followed by a Gated Linear Unit (GLU) projection. All LinOSS computations are
forced to float32 under AMP (bfloat16 overflow in GLU at large magnitudes).

**Physics alignment.** The function-space prior of D-LinOSS matches the physics of
vibrational spectra: the DFT-computed IR and Raman signals are sums of Lorentzian
(in the frequency domain) or exponentially-damped sinusoidal (in time domain)
contributions from each normal mode. D-LinOSS has the representational capacity to
model these exactly; CNN and Transformer do not have this inductive bias. We
emphasise this is a *function-space argument* — the learned oscillator parameters
are signal-processing frequencies, not cm⁻¹ wavenumbers.


## 2.2 Competing Architectures

All backbones share the interface `(B, L, d_model) → (B, L, d_model)` and are
parameter-matched at approximately 2M backbone parameters for fair comparison.

| Architecture | Key property | Backbone params |
|---|---|---|
| D-LinOSS | 2nd-order oscillatory SSM, O(L) scan | ~2.0M |
| Transformer | Self-attention, O(L²), SDPA/Flash-3 | ~2.0M |
| 1D CNN | Dilated causal convolutions, FIR | ~2.0M |
| S4D (ablation) | 1st-order diagonal SSM, no oscillation | ~271K |
| Mamba (excluded) | Selective SSM; deadlocks on SM120 | N/A |

**Transformer.** Pre-norm transformer with rotary positional encoding and
`F.scaled_dot_product_attention`. n_layers=6, d_model=256, n_heads=8, d_ff=1024.
When used as sole backbone, the post-backbone transformer is set to `nn.Identity`
to avoid double-transformer stacking.

**1D CNN.** Stack of dilated 1D convolutions with exponentially increasing dilation
rates (1, 2, 4, 8, ...) and residual connections. n_layers=8, d_model=256, kernel=3.

**S4D.** Diagonal S4 with real-valued diagonal state matrix. n_layers=4, d_state=64,
d_model=256, ~271K backbone parameters. Included as structural ablation: first-order
diagonal SSM with no oscillatory structure, contrasting D-LinOSS's second-order design.
Not parameter-matched; results are reported separately.

**Mamba.** Excluded. The `mamba_ssm` custom CUDA kernels deadlock on RTX 5060 Ti
(Blackwell SM120, CUDA Compute 12.0) regardless of DataParallel configuration.
This is a hardware-incompatibility issue; all Mamba results are marked SKIP.


## 2.3 Dataset: QM9S

QM9S [CITE] provides DFT vibrational spectra for 129,817 small organic molecules
(up to 9 heavy atoms: C, N, O, F) optimised at B3LYP/def2-TZVP with D3(BJ)
dispersion. Raw spectra span 500–4000 cm⁻¹ at 3501 grid points; we resample to
2048 evenly-spaced points (power-of-two for FFT-based operations). Of 129,817
molecules, 103,992 possess both IR and Raman spectra.

**Normalisation.** Each spectrum is L2-normalised per-sample prior to training.

**Splits.** 85.5 / 4.5 / 10% train / val / test by stratified sampling on molecular
formula, ensuring no formula spans multiple splits.

**Symmetry.** 99.93% of QM9S molecules are non-centrosymmetric; the mutual exclusion
rule does not apply. IR and Raman are correlated but not identical.


## 2.4 Spectral Embedding

Each input spectrum x ∈ ℝ^{2048} is embedded via a 1D Conv1d (kernel=3, stride=1,
padding=1) projecting each wavenumber point to d_model dimensions, giving
(B, 2048, d_model). A learnable wavenumber positional encoding is added, along with a
[CLS] token and a domain token ([IR] or [RAMAN]), giving sequence length 2050.
No patching; every wavenumber is an independent token.


## 2.5 Pre-training Objective (E1)

**Masked Spectral Reconstruction (MSR).** Mask fraction p=0.30 of tokens with a
learned [MASK] token; train to reconstruct original values at masked positions.

    L = L_MSRP + λ_phys · L_physics + λ_VIB · L_VIB

- `L_MSRP`: peak-weighted MSE at masked positions (weights spectral peaks 3× over
  background by `find_peaks` detection)
- `L_physics`: soft positivity `mean(ReLU(-x̂))` — spectra are non-negative
- `L_VIB`: variational information bottleneck on [CLS] token (β-VAE style)

**Training.** AdamW, lr=3×10⁻⁴, cosine annealing (warmup 2000 steps),
batch_size=16, grad_accum=4 (effective batch 64), max_steps=50,000, AMP bfloat16.
All runs use 3 seeds (42, 43, 44); mean ± std reported.


## 2.6 Cross-Spectral Prediction (E2)

A CrossSpectralHead (3-layer MLP, hidden=d_model, tanh activation) is attached to
the pooled encoder output and trained end-to-end from scratch (no E1 initialisation).
Loss: per-element MSE. Data augmentation: Gaussian noise σ=0.01 on source spectra.

Training: AdamW lr=3×10⁻⁴, cosine annealing, batch_size=16, max_steps=30,000.
All architectures use the same batch size; D-LinOSS memory usage at the 2M-param scale
constrains batch_size to 16 on 22 GB GPUs.
Best validation checkpoint restored before test evaluation.

**Baselines.**
- *Identity*: predict source spectrum as target (measures correlation baseline)
- *PLS2*: PLSRegression n_components=30, trained on 10,000 samples

**Metrics.** Per-element MSE; cosine similarity (shape invariant to global scale);
peak recall (fraction of target peaks within ±5 cm⁻¹ in prediction). Peaks detected
by `find_peaks` with height=0.05·max, prominence=0.02·max.


## 2.7 Transfer Function Analysis (E3)

For trained D-LinOSS, the z-domain transfer function H(z) per oscillator per layer:

    H(z) = dt² · b · z / [(1 + dt·g)·z² − (2 + dt·g − dt²·a)·z + 1]

evaluated at 512 uniformly-spaced frequencies. Three statistics:

- **BC coupling CV**: coefficient of variation of |b_k · c_k| over oscillators in
  each layer. High CV = few dominant oscillators = selective coupling.
- **Peak wavenumber**: frequency at max |H(e^{jω})| converted to cm⁻¹ equivalent.
- **Layer specialisation**: inter-layer std of mean peak wavenumber.

Significance testing: Welch's t-test and Cohen's d vs. random-weight controls.


## 2.8 Calibration Transfer (E4)

Dataset: corn (80 samples × 3 instruments × 700 NIR channels, 1100–2500 nm);
target: protein content (%). The backbone is frozen; a linear head is attached to
the pooled [CLS] token and fit by ridge regression with cross-validation.
Metrics: R², RMSEP.

Classical baselines (CCA, DS, SBC, di-PLS) evaluated by `src/evaluation/baselines.py`.
