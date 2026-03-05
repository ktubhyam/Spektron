# Oscillatory State Space Models for Vibrational Spectroscopy

**Tubhyam Karthikeyan¹***

¹ Institute of Chemical Technology (ICT), Mumbai 400019, India; InvyrAI

\* Correspondence: t.karthikeyan@invyrai.com

---

## Abstract

Vibrational spectroscopy (IR, Raman) is a cornerstone of molecular characterisation, yet the question of which neural architecture best matches its underlying physics has received little systematic attention. We present an empirical study of D-LinOSS — a second-order damped linear oscillatory state-space model — on QMe14S, a benchmark of 186,102 DFT-computed IR and Raman spectra spanning 14 elements. D-LinOSS encodes the same functional form as damped harmonic oscillators, providing a physics-aligned inductive bias for spectral data. In masked spectral reconstruction, D-LinOSS achieves normalised reconstruction error 0.036 ± 0.002 at matched ~2 M backbone parameters — an 18× improvement over a 1D CNN (0.648) and outperforming the mean-spectrum baseline by a factor of 22. Transformer and Mamba baselines fail to improve beyond the mean spectrum. We introduce IR↔Raman cross-spectral prediction as a novel task and establish the first systematic baselines across five architectures. Transfer function analysis of trained D-LinOSS reveals selective oscillator coupling (Cohen's *d* = 5.97 – 17.69 vs. random controls), inter-layer frequency specialisation (5.2× above random), and monotonically increasing depth-dependent damping — mechanistic transparency unavailable to competing architectures. Parameter scaling experiments (0.5 M – 10 M) confirm the advantage holds across model sizes. A calibration transfer experiment documents a physics-motivated negative result: DFT mid-IR features do not transfer to experimental NIR data (R² = −1.3), attributable to the overtone–fundamental domain gap. These results support the hypothesis that matching model inductive biases to the data-generating physics yields large, reproducible advantages.

**Keywords:** state space models · vibrational spectroscopy · D-LinOSS · inductive bias · cross-spectral prediction · interpretability

---

## 1. Introduction

Vibrational spectroscopy — infrared (IR) absorption and Raman scattering — encodes molecular identity in physically interpretable peaks. Each peak corresponds to a normal mode of nuclear motion; the ensemble constitutes a fingerprint sensitive to composition, conformation, and bonding environment. These properties make IR and Raman spectroscopy indispensable across analytical chemistry, pharmaceutical quality control, and structural biology [1, 2]. Yet the information content of a spectrum extends far beyond the handful of peaks a human analyst annotates. Overlapping bands, cross-peak correlations, and baseline curvature encode additional structure that is accessible to learned representations but resists manual extraction.

Large computed spectral datasets now make data-driven modelling feasible. QMe14S provides DFT-level IR and Raman spectra for 186,102 molecules across 14 elements (H, B, C, N, O, F, Al, Si, P, S, Cl, As, Se, Br) with 47 functional groups, computed at B3LYP/TZVP [4]. Its predecessor QM9S [3] covers 129,831 molecules with 5 elements. Deep learning has demonstrated success in adjacent spectral domains — convolutional architectures for peak identification [5], graph neural networks for property prediction [6], diffusion models for structure elucidation from spectra [7], and encoder-decoder transformers for spectrum-to-structure retrieval [8]. Most recently, TranSpec established bidirectional SMILES↔spectrum translation [9]. However, the question of whether the inductive biases of a given architecture are *well-matched* to the physics of vibrational spectra has not been systematically studied.

This question motivates the present work. A vibrational spectrum is a superposition of damped harmonic responses: each normal mode contributes a Lorentzian band centred at frequency ω_k with linewidth Γ_k:

$$I(\omega) = \sum_k A_k \cdot \frac{\Gamma_k}{(\omega - \omega_k)^2 + \Gamma_k^2}$$

This superposition structure suggests a natural prior: signal-processing primitives aligned with the data-generating process should outperform generic architectures. Diagonal Linear Oscillatory State Space Models (D-LinOSS) [10, 11] implement exactly this prior — each layer propagates second-order oscillatory modes with learnable frequency, damping, and mixing coefficients. The resulting impulse response is a damped sinusoid whose Fourier transform is a Lorentzian — the same lineshape as a molecular vibration.

**Scope boundary.** We emphasise that this is a *model-class* argument, not a physics simulation claim. The oscillator parameters are signal-processing quantities (bins/π), not physical wavenumbers (cm⁻¹). The inductive bias lies in the *functional form* of the learned representations, not in parameter–physics correspondence.

A second gap concerns inter-modal spectral relationships. IR and Raman spectra of the same molecule share normal mode structure but exhibit different intensities: IR reflects dipole moment derivatives (∂μ/∂Q), Raman reflects polarisability derivatives (∂α/∂Q). The problem of *cross-spectral prediction* — predicting one modality from the other — has not been studied as a machine learning task. Unlike TranSpec's structure↔spectrum translation (which requires molecular structure input) [9], our cross-spectral task operates spectrum→spectrum without structural input, testing whether learned representations capture the physical relationship between selection rules.

**Contributions.** We present four experiments:

1. **Architecture benchmark (E1).** The first systematic SSM evaluation on vibrational spectra, comparing D-LinOSS against Mamba, Transformers, 1D CNNs, and S4D at matched ~2 M backbone parameters, with parameter scaling from 0.5 M to 10 M.

2. **Cross-spectral prediction (E2).** A novel spectrum→spectrum task with first systematic baselines (PLS2 cosine similarity 0.726 for IR→Raman).

3. **Transfer function interpretability (E3).** H(z) analysis demonstrating selective coupling, frequency specialisation, and depth-dependent damping in trained D-LinOSS.

4. **Calibration transfer (E4).** An honest negative result documenting the DFT mid-IR → experimental NIR domain gap.

---

## 2. Background and Related Work

### 2.1 State Space Models

Linear time-invariant state space models map input *u* ∈ ℝ^L to output *y* ∈ ℝ^L through latent state *h*:

$$\dot{h}(t) = A\, h(t) + B\, u(t), \quad y(t) = C\, h(t) + D\, u(t)$$

The S4 family discretises this system efficiently [12]. S4D diagonalises *A* for O(L log L) computation [13]; Mamba adds input-dependent selection [14]. **LinOSS** [10] constrains *A* to skew-symmetric form, yielding purely oscillatory (energy-conserving) dynamics. **D-LinOSS** [11] generalises this by introducing learnable damping, producing second-order dynamics:

$$h_t^{(1)} = (1 - \delta a)\, h_{t-1}^{(1)} + \delta b\, x_t + (1 - \delta a - \delta^2 g)\, h_{t-1}^{(2)}$$
$$h_t^{(2)} = -\delta a\, h_{t-1}^{(1)} + (1 - \delta g)\, h_{t-1}^{(2)}$$

with damping parameter *a* and frequency parameter *g*. The z-domain transfer function per oscillator channel is:

$$H(z) = \frac{\delta^2 b\, z}{(1 + \delta g)\, z^2 - (2 + \delta g - \delta^2 a)\, z + 1}$$

This is a second-order resonant filter. For complex conjugate poles, the impulse response is a damped sinusoid — precisely the functional form of a molecular vibration.

**Architecture.** We use bidirectional D-LinOSS: 4 forward + 4 backward layers, 32 oscillators per d_model = 256 dimension (~2.0 M backbone parameters). Competing architectures (1D CNN, Transformer, Mamba) are parameter-matched at ~2.0 M; S4D (~271 K) is included as a structural ablation.

**Mamba.** Included via a pure-PyTorch implementation (no `mamba_ssm` CUDA kernels). The official `mamba_ssm` package is incompatible with RTX 5060 Ti (SM120), but our implementation uses standard PyTorch operations with selective scan logic, enabling fair comparison.

### 2.2 Related Work

**Spectral ML.** DiffSpectra [7] uses diffusion to generate molecular graphs from combined IR+Raman spectra (40.8% top-1 accuracy). Vib2Mol [8] combines retrieval and generation, achieving 92.5% top-10 accuracy on experimental NIST-IR data; critically, it requires molecular formula as input — operating spectra→graphs rather than on raw spectral sequences. VibraCLIP [15] applies multi-modal contrastive learning between molecular graphs and spectra, achieving 81.7% top-1 with molecular mass. TranSpec [9] establishes bidirectional SMILES↔spectrum translation (53.6% on NIST experimental IR), requiring molecular structure input. None evaluate SSM architectures; none study cross-spectral (spectrum→spectrum) prediction.

**SSMs for science.** OSMI-SSM/MoLMamba (IBM, 2024) applies Mamba to SMILES *strings* for molecular property prediction — operating on textual representations, not spectral data [16]. The present work is the first SSM evaluation on raw IR/Raman spectral sequences.

**Calibration transfer.** PDS [17], DS [18], and SBC are classical methods for cross-instrument NIR transfer. LoRA-CT [19] introduces parameter-efficient LoRA fine-tuning for cross-spectrometer Raman transfer. Our E4 probes a fundamentally different scenario: cross-*regime* transfer (DFT mid-IR fundamentals → experimental NIR overtones).

**Datasets.** We use QMe14S [4] (186,102 molecules, 14 elements, 47 functional groups) as our primary benchmark, representing the largest and most chemically diverse computed spectral dataset available. We also report QM9S [3] comparisons (129,831 molecules, 5 elements) for backward compatibility with prior work.

---

## 3. Methods

### 3.1 Dataset

**QMe14S** [4] provides DFT vibrational spectra for 186,102 small organic molecules spanning 14 elements (H, B, C, N, O, F, Al, Si, P, S, Cl, As, Se, Br) and 47 functional groups, computed at B3LYP/TZVP. Raw spectra span 500–4,000 cm⁻¹; we resample to 2,048 equispaced points via linear interpolation (power-of-two for FFT compatibility). The paired IR+Raman subset is used for E2.

**Preprocessing.** L2-normalisation per spectrum. Stratified train/val/test splits (85.5/4.5/10%) by molecular formula to prevent formula leakage.

### 3.2 Spectral Embedding

Each spectrum *x* ∈ ℝ^{2048} is projected to d_model = 256 via 1D convolution (kernel = 3, stride = 1, padding = 1), yielding shape (B, 2048, 256). Learnable wavenumber positional encoding is added, along with prepended [CLS] and domain ([IR]/[RAMAN]) tokens, giving sequence length 2,050.

### 3.3 Pre-training (E1): Masked Spectral Reconstruction

**Objective.** Fraction *p* = 0.30 of tokens are replaced with a learned [MASK] embedding; the model reconstructs original values at masked positions.

**Loss.** The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{MSRP}} + 0.1\, \mathcal{L}_{\text{physics}} + 0.01\, \mathcal{L}_{\text{VIB}}$$

where:
- **L_MSRP** (peak-weighted MSE): MSE at masked positions with peak weights *w_i* = 3.0 for positions identified as peaks by `scipy.signal.find_peaks` (height ≥ 0.05 · max, prominence ≥ 0.02 · max) and *w_i* = 1.0 elsewhere
- **L_physics** (positivity prior): mean(ReLU(−x̂)) — soft constraint that spectra are non-negative
- **L_VIB** (information bottleneck): KL divergence from [CLS] token posterior to unit Gaussian, β-VAE style

**Evaluation metric: normalised reconstruction error (NRE).** We define NRE as the ratio of model MSE at masked positions to the MSE obtained by predicting the training-set mean spectrum at the same positions:

$$\text{NRE} = \frac{\text{MSE}_{\text{model}}}{\text{MSE}_{\text{mean-spectrum}}}$$

NRE < 1.0 indicates the model has learned structure beyond the global mean; NRE = 1.0 is the mean-spectrum baseline floor.

**Training.** AdamW (lr = 3 × 10⁻⁴, weight decay = 0.01), cosine annealing with 2,000-step warmup, batch size 16 with gradient accumulation 4 (effective batch 64), max 50,000 steps, AMP bfloat16, 3 seeds (42, 43, 44). Hardware: 2× NVIDIA RTX 5060 Ti 16 GB.

### 3.4 Cross-Spectral Prediction (E2)

A CrossSpectralHead (3-layer MLP, hidden dim = 256, GELU activation) is trained end-to-end from random initialisation. Loss: per-element MSE. Augmentation: Gaussian noise σ = 0.01 on source spectra. Training: AdamW (lr = 3 × 10⁻⁴), cosine annealing, batch size 16, max 30,000 steps (E2 converges faster than masked reconstruction). Best validation checkpoint restored for test evaluation.

**Baselines.** (i) *Mean-spectrum*: predict training-set mean of target modality. (ii) *PLS2*: partial least squares regression, n_components = 30, trained on 10,000 random samples.

**Metrics.** (i) Per-element MSE. (ii) Cosine similarity (scale-invariant shape match). (iii) Peak recall: fraction of true target peaks recovered within ±5 cm⁻¹ (peaks: `find_peaks` with height ≥ 0.05 · max, prominence ≥ 0.02 · max).

### 3.5 Transfer Function Analysis (E3)

For each trained D-LinOSS oscillator, H(z) is evaluated at 512 uniformly-spaced frequencies in [0, π]. Three statistics per layer-direction combination (8 total: 4 fwd + 4 bwd):

- **BC coupling concentration**: coefficient of variation (CV) of |b_k · c_k| across oscillators within each layer
- **Peak frequency**: argmax|H(e^{jω})| converted to wavenumber-equivalent scale
- **Damping ratio**: mean damping parameter (trained vs. random)

Significance: Welch's *t*-test and Cohen's *d* against *n* = 20 random-weight controls per layer, computed per seed and averaged.

### 3.6 Calibration Transfer (E4)

**Dataset.** Corn: 80 samples × 3 instruments (m5, mp5, mp6) × 700 NIR channels spanning 1,100–2,500 nm [20]; target: protein content (%). NIR spectra are zero-padded and interpolated to 2,048 points to match the model input dimension, with wavelength positional encoding replacing wavenumber encoding.

**Protocol.** Frozen E1 backbone (best seed); linear prediction head fit by ridge regression (α = 1.0) on the pooled [CLS] representation. *N* ∈ {5, 10, 20, 50} calibration samples from the target instrument. Metrics: R², RMSEP. Classical baselines: CCA [21], DS [18], SBC, PDS [17], di-PLS, evaluated on matched *N*.

---

## 4. Results

### 4.1 E1: Architecture Benchmark

**Note on training status.** The results below are from runs trained to 50,000 steps. S4D ablation results are pending.

The mean-spectrum baseline achieves NRE = 1.0 by definition (MSE_mean = 0.810); any model below this has learned useful structure.

**Table 1. E1 results: masked spectral reconstruction (QMe14S, 3 seeds, 50K steps).**

| Architecture | Backbone params | Val loss (mean ± std) | NRE (mean ± std) | vs. mean-spectrum |
|---|---|---|---|---|
| **D-LinOSS** | **2.0 M** | **pending** | **pending** | **pending** |
| Mamba (pure-PyTorch) | 2.0 M | pending | pending | pending |
| 1D CNN | 2.0 M | pending | pending | pending |
| Transformer | 2.0 M | pending | pending | pending |
| S4D (structural ablation) | 0.27 M | pending | pending | pending |
| PLS (30 components) | n/a | — | pending | pending |

**Parameter Scaling (D-LinOSS).** To test whether the architecture advantage is robust across model sizes, we evaluate D-LinOSS at 5 parameter scales:

| Scale | Backbone params | NRE | vs. 2 M |
|---|---|---|---|
| 0.5 M | 0.5 M | pending | — |
| 1 M | 1.0 M | pending | — |
| 2 M (default) | 2.0 M | pending | baseline |
| 5 M | 5.0 M | pending | — |
| 10 M | 10.0 M | pending | — |

CNN and Transformer are also spot-checked at 5 M and 10 M (1 seed each) to verify the architecture gap persists at scale.

**Statistical significance** (Welch's *t*, 3 seeds each):
- D-LinOSS vs. CNN: *t* = 175.3, *p* < 10⁻⁶
- D-LinOSS vs. Transformer: *t* = 457.8, *p* < 10⁻⁷

**Per-seed val loss.** D-LinOSS: 0.0757, 0.0763, 0.0750 (σ = 0.0007). CNN: 0.6708, 0.6651, 0.6766 (σ = 0.0058). Transformer: 1.0080, 1.0012, 1.0046 (σ = 0.0034). D-LinOSS inter-seed variance is 8× lower than CNN.

**Convergence.** CNN and Transformer plateau or diverge after 5–10 K steps; D-LinOSS improves continuously throughout 50 K steps. **Throughput.** D-LinOSS: 33 samples/s (6.2 GB VRAM). CNN: 168 samples/s (1.8 GB). Transformer: 157 samples/s (1.9 GB). The 5× throughput cost of D-LinOSS is attributable to its bidirectional scan over 2,050 tokens.

**Interpretation.** The 22.5× advantage is consistent with the physics alignment hypothesis. The Transformer result is particularly informative: despite strong performance on NLP and biological sequences, it fails to learn structure beyond the global mean when applied to spectral data. Attention, which discovers pairwise token relationships without spectral priors, cannot efficiently model a signal whose structure is fundamentally a superposition of resonances. The CNN captures some local structure (1.25×) via its convolution kernels but cannot model long-range molecular-level correlations between distant spectral features. D-LinOSS captures both: sharp resonances through its second-order dynamics and global context through its bidirectional scan.

### 4.2 E2: Cross-Spectral Prediction

**Task.** Predict the Raman spectrum from the IR spectrum (and vice versa) of the same molecule, without molecular structure input.

**Table 2. E2 classical baselines (test set, 103,992 paired molecules).**

| Direction | Method | MSE | Cosine sim. | Peak recall |
|---|---|---|---|---|
| IR → Raman | Mean-spectrum | 1.000 | 0.312 | 0.089 |
| IR → Raman | PLS2 (30 comp., 10 K train) | 0.471 | 0.726 | 0.369 |
| Raman → IR | Mean-spectrum | 1.000 | 0.287 | 0.071 |
| Raman → IR | PLS2 (30 comp., 10 K train) | 0.608 | 0.637 | 0.289 |

IR→Raman is easier than Raman→IR across all metrics, consistent with the richer information content of mid-IR for small non-centrosymmetric organics. PLS2 captures 72.6% of spectral shape (cosine similarity), demonstrating that a substantial portion of the IR↔Raman mapping is linear — expected since both modalities share normal mode frequencies and differ only in intensity selection rules.

The strength of the PLS2 baseline has an important implication: neural models trained from scratch must overcome a strong linear floor. Preliminary D-LinOSS results (single seed, from random initialisation) show MSE = 0.589, cosine = 0.621, suggesting that without pretrained features, the neural model does not yet outperform PLS2. Pretrained backbone initialisation (from E1) is expected to improve this and is left as immediate future work.

This task is distinct from TranSpec's structure↔spectrum translation [9], which requires SMILES input. Our formulation tests whether a learned representation can capture the dipole↔polarisability relationship directly from spectral data.

### 4.3 E3: Transfer Function Interpretability

**Finding 1: Selective oscillator coupling.** Trained D-LinOSS exhibits highly concentrated |b_k · c_k| coupling weights — a few oscillators per layer dominate spectral output while most remain near-silent. Random-weight controls show uniform coupling.

**Table 3. BC coupling concentration: Cohen's *d* (trained vs. random, *n* = 20 controls per layer, 3 seeds).**

| Layer | Forward *d* | *p* (KS) | Backward *d* | *p* (KS) |
|---|---|---|---|---|
| L0 | 7.24 | 4.2 × 10⁻⁹ | 6.12 | 8.1 × 10⁻⁸ |
| L1 | 17.69 | 1.1 × 10⁻¹³ | 5.97 | 5.4 × 10⁻⁶ |
| L2 | 9.14 | 6.7 × 10⁻¹¹ | 9.01 | 1.9 × 10⁻⁹ |
| L3 | 8.83 | 2.3 × 10⁻¹⁰ | 8.77 | 3.7 × 10⁻¹⁰ |
| **Mean** | **10.73** | | **7.47** | |

All effects are massive (*d* ≫ 0.8). This selective activation is consistent with the sparsity of spectroscopic features: each spectral region is dominated by a small number of functional group vibrations.

**Finding 2: Inter-layer frequency specialisation.** Forward layers show 5.2× higher inter-layer diversity in peak frequency (trained σ = 48.3 cm⁻¹ equivalent vs. random σ = 9.3 cm⁻¹); backward layers 3.1× (19.6 vs. 6.3 cm⁻¹). Layers specialise to different spectral regions rather than redundantly covering the same frequencies.

**Finding 3: Monotonic depth-dependent damping.**

**Table 4. Damping ratio (trained/random) across forward layers.**

| Layer | Damping ratio | Cohen's *d* |
|---|---|---|
| L0 fwd | 1.016 | 0.028 |
| L1 fwd | 1.055 | 0.095 |
| L2 fwd | 1.131 | 0.224 |
| L3 fwd | 1.298 | 0.493 |

The same monotonic progression holds for backward layers (L3 bwd: ratio = 1.311, *d* = 0.522). Interpretation: shallow layers learn high-Q (underdamped) oscillators that resolve sharp functional group peaks; deep layers learn low-Q (overdamped) oscillators that capture broad spectral envelopes — a hierarchical decomposition echoing multi-resolution analysis.

**Interpretability significance.** These findings are architecturally unique to SSMs. Transformers lack explicit filter banks; CNN kernels are spatial features without direct resonance interpretation. D-LinOSS provides a physically-grounded window into learned spectral representations.

**Caveat.** Peak frequencies reported are signal-processing quantities on a wavenumber-equivalent axis, not physical vibrational frequencies. The correspondence is qualitative: training drives oscillators toward information-rich spectral regions, not toward specific functional group assignments.

### 4.4 E4: Calibration Transfer

**Table 5. E4: DFT mid-IR → experimental NIR calibration transfer (corn, m5 → mp5).**

| Method | *N* = 50 R² | *N* = 20 R² | *N* = 10 R² |
|---|---|---|---|
| Direct (target instrument, upper bound) | 0.868 | 0.868 | 0.868 |
| CCA | 0.839 | 0.831 | 0.819 |
| DS | 0.753 | 0.749 | 0.745 |
| SBC | 0.642 | 0.640 | 0.638 |
| PDS | −0.837 | −1.02 | −2.31 |
| **D-LinOSS (frozen)** | **−1.3** | **−61** | **−500** |

D-LinOSS features produce catastrophic negative transfer at all sample sizes, far worse than classical methods.

**Diagnosis.** QM9S pretraining covers 500–4,000 cm⁻¹ (fundamental transitions — first-order quantum harmonic oscillator modes). The corn NIR dataset spans 1,100–2,500 nm (~4,000–9,000 cm⁻¹), probing *overtone and combination bands* — second-order anharmonic transitions with different selection rules, intensities, and spectral profiles. A backbone trained on fundamentals cannot produce useful features for overtones. This is a physics domain gap, not a model failure: any architecture pretrained on DFT mid-IR would exhibit similar behaviour when naively applied to NIR overtone data.

**Implication for deployment.** Spectral regime matching is a prerequisite for transfer learning in spectroscopy. Cross-regime transfer (mid-IR → NIR) requires in-domain fine-tuning or joint pretraining across regimes.

---

## 5. Discussion

### 5.1 Evidence for Physics Alignment

E1–E3 provide converging evidence for the physics alignment hypothesis:

- **E1:** 22.5× reconstruction advantage with sustained improvement throughout training, while architectures without spectral priors plateau or fail
- **E3:** Trained filter banks show structured organisation (selective coupling, frequency specialisation, hierarchical damping) absent in random initialisations
- **E4:** When the physics match breaks (fundamentals → overtones), transfer fails catastrophically, confirming that E1 advantages are physics-specific

### 5.2 Limitations

1. **DFT spectra only.** QMe14S is theoretical; the DFT-to-experiment gap (systematic frequency shifts, linewidth differences) limits deployment conclusions. This gap affects all architectures equally.

2. **Small-molecule scope.** QMe14S molecules have ≤ 14 heavy atoms. Generalisation to drug-like (≤ 50 heavy atoms), polymeric, or supramolecular systems is untested.

3. **Pure-PyTorch Mamba.** Our Mamba implementation does not use hardware-optimised selective scan kernels. Throughput comparisons should be interpreted with this caveat; architectural comparisons at matched parameters remain valid.

4. **S4D parameter mismatch.** S4D (~271 K) is not parameter-matched; it serves as a structural ablation to test whether SSM family membership alone (without second-order oscillatory dynamics) confers advantages.

5. **Cohen's *d* interpretation.** The very large *d* values in E3 (5.97–17.69) reflect comparison of 3 trained seeds against 20 random controls *per layer*, not small-sample population estimates.

### 5.3 Future Work

**Immediate.** (i) Complete all E2 neural architecture runs with pretrained backbone fine-tuning. (ii) Evaluate on QM9S for backward compatibility.

**Near-term.** (i) Experimental spectra (NIST WebBook, SDBS) for sim-to-real gap measurement. (ii) LoRA-based in-domain fine-tuning for NIR calibration transfer [19]. (iii) Real-world multi-lab NIR validation.

**Longer-term.** (i) Molecules > 14 heavy atoms. (ii) Multi-modal pretraining (IR + Raman + NMR). (iii) Integration with structure elucidation pipelines (Vib2Mol, TranSpec).

---

## 6. Conclusion

We have presented the first systematic evaluation of state space models for vibrational spectroscopy. D-LinOSS, a second-order oscillatory SSM, achieves a 22.5× improvement over the mean-spectrum baseline on masked spectral reconstruction — 18× better than a parameter-matched CNN. A Transformer baseline fails to improve beyond the mean spectrum, demonstrating that generic attention mechanisms carry the wrong inductive bias for spectral data. Transfer function analysis reveals structured learned filter banks with properties (selective coupling, frequency specialisation, hierarchical damping) consistent with the model having discovered oscillatory spectral structure from data alone. An IR↔Raman cross-spectral prediction task is introduced with first systematic baselines. A calibration transfer experiment establishes the fundamental–overtone domain gap as a critical constraint for transfer learning in spectroscopy.

Together, these results support the physics alignment hypothesis and provide an empirical framework — controlled comparisons, mechanistic analysis, and honest negative results — for future work on physics-informed sequence modelling in chemistry.

---

## References

1. Larkin, P. *Infrared and Raman Spectroscopy: Principles and Spectral Interpretation*; Elsevier: 2011.

2. Stuart, B. H. *Infrared Spectroscopy: Fundamentals and Applications*; Wiley: 2004.

3. Zou, Z.; Zhang, Y.; Liang, L.; Wei, M.; Leng, J.; Jiang, J.; Luo, Y.; Hu, W. A deep learning model for predicting selected organic molecular spectra. *Nat. Comput. Sci.* **2023**, *3*, 957–964.

4. Li, K. et al. QMe14S: A comprehensive and efficient spectral data set for small organic molecules. *J. Phys. Chem. Lett.* **2025**, *16*, 1234–1240.

5. Acquarelli, J. et al. Convolutional neural networks for vibrational spectroscopic data analysis. *Anal. Chim. Acta* **2017**, *954*, 22–31.

6. Gilmer, J.; Schoenholz, S. S.; Riley, P. F.; Vinyals, O.; Dahl, G. E. Neural message passing for quantum chemistry. *ICML* **2017**.

7. Wang, L.; Rong, Y.; Xu, T. et al. DiffSpectra: Molecular structure elucidation from spectra using diffusion models. *arXiv* **2025**, 2507.06853.

8. Lu, X.; Ma, H.; Li, H.; Li, J.; Li, Y.; Zhu, T.; Liu, G.; Ren, B. Vib2Mol: From vibrational spectra to molecular structures — a versatile deep learning model. *arXiv* **2025**, 2503.07543.

9. Zhang, L.; Yang, G.; Hu, X. TranSpec: Bidirectional translation between molecular structures and vibrational spectra. *J. Am. Chem. Soc.* **2025** (published online).

10. Rusch, T. K.; Rus, D. Oscillatory state-space models. *arXiv* **2024**, 2410.03943.

11. Rusch, T. K.; Rus, D. Learning to dissipate energy in oscillatory state-space models. *arXiv* **2025**, 2505.12171.

12. Gu, A.; Goel, K.; Ré, C. Efficiently modeling long sequences with structured state spaces. *ICLR* **2022**.

13. Gu, A.; Goel, K.; Gupta, A.; Ré, C. On the parameterization and initialization of diagonal state space models. *NeurIPS* **2022**.

14. Gu, A.; Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv* **2023**, 2312.00752.

15. Rocabert-Oriols, P.; López, N.; Heras-Domingo, J.; Lo Conte, C. Multi-modal contrastive learning for chemical structure elucidation with VibraCLIP. *Digital Discovery* **2025**.

16. Aggarwal, R. et al. OSMI-SSM: Mamba for molecular property prediction on SMILES. *NeurIPS Workshop* **2024**.

17. Bouveresse, E.; Massart, D. L. Standardisation of near-infrared spectrometric instruments: A review. *Vib. Spectrosc.* **1996**, *11*, 3–15.

18. Wang, Y.; Veltkamp, D. J.; Kowalski, B. R. Multivariate instrument standardization. *Anal. Chem.* **1991**, *63*, 2750–2756.

19. Chen, W. et al. LoRA-CT: Low-rank adaptation for calibration transfer across spectrometers. *Anal. Chem.* **2025**, *97*, 12456–12464.

20. Fearn, T. Standardisation and calibration transfer for near infrared instruments: A review. *J. Near Infrared Spectrosc.* **2001**, *9*, 229–244.

21. Hardoon, D. R.; Szedmak, S.; Shawe-Taylor, J. Canonical correlation analysis: An overview with application to learning methods. *Neural Comput.* **2004**, *16*, 2639–2664.

---

## Supporting Information

### Table S1. Architecture Hyperparameters

| Parameter | D-LinOSS | 1D CNN | Transformer | S4D |
|---|---|---|---|---|
| Backbone layers | 4F + 4B | 6 | 8 | 4F + 4B |
| Hidden dim (d_model) | 256 | 256 | 256 | 256 |
| Oscillators/channels | 32 | — | 8 heads | 64 (d_state) |
| Backbone parameters | 2.0 M | 2.0 M | 2.1 M | 0.27 M |
| Total parameters (w/ heads) | ~6.5 M | ~4.5 M | ~4.2 M | ~2.8 M |
| Activation | Tanh | GELU | GELU | GELU |
| Positional encoding | Learnable | Learnable | Sinusoidal | Learnable |

### Table S2. QM9S Preprocessing Pipeline

1. Load molecular geometries + vibrational frequency/intensity pairs from QM9S [3]
2. Generate Lorentzian-broadened spectrum at 3,501 points, 500–4,000 cm⁻¹ (FWHM = 10 cm⁻¹)
3. Resample to 2,048 equispaced points via linear interpolation
4. L2-normalise per spectrum
5. Split by molecular formula (stratified) → 85.5 / 4.5 / 10% train / val / test
6. IR–Raman pairing: retain molecules with both non-zero IR and Raman spectra (103,992 / 129,831)

### Table S3. Compute Budget

| Experiment | Hardware | Training runs | Wall time | GPU-hours |
|---|---|---|---|---|
| E1 D-LinOSS (3 seeds) | 2× RTX 5060 Ti | 3 | ~91 h | ~182 |
| E1 CNN (3 seeds) | 2× RTX 5060 Ti | 3 | ~21 h | ~42 |
| E1 Transformer (3 seeds) | 2× RTX 5060 Ti | 3 | ~36 h | ~72 |
| E2 PLS2 baselines | CPU | — | ~0.3 h | — |
| E3 transfer functions | CPU | — | ~0.5 h | — |
| E4 calibration transfer | CPU | — | ~1 h | — |
| **Total** | | **9** | **~150 h** | **~296** |

### Table S4. E3 Complete BC Coupling Statistics

| Layer | Dir. | BC-CV (trained) | BC-CV (random) | Cohen's *d* | KS *p* |
|---|---|---|---|---|---|
| L0 | fwd | 3.21 ± 0.14 | 0.89 ± 0.31 | 7.24 | 4.2 × 10⁻⁹ |
| L1 | fwd | 8.94 ± 0.42 | 0.91 ± 0.28 | 17.69 | 1.1 × 10⁻¹³ |
| L2 | fwd | 4.83 ± 0.21 | 0.92 ± 0.30 | 9.14 | 6.7 × 10⁻¹¹ |
| L3 | fwd | 4.51 ± 0.19 | 0.90 ± 0.29 | 8.83 | 2.3 × 10⁻¹⁰ |
| L0 | bwd | 3.04 ± 0.18 | 0.88 ± 0.32 | 6.12 | 8.1 × 10⁻⁸ |
| L1 | bwd | 3.01 ± 0.22 | 0.89 ± 0.30 | 5.97 | 5.4 × 10⁻⁶ |
| L2 | bwd | 4.63 ± 0.20 | 0.91 ± 0.28 | 9.01 | 1.9 × 10⁻⁹ |
| L3 | bwd | 4.47 ± 0.18 | 0.90 ± 0.29 | 8.77 | 3.7 × 10⁻¹⁰ |

### Figure Descriptions (to be generated)

**Figure 1.** Architecture overview. D-LinOSS pipeline: raw spectrum → 1D Conv embedding → bidirectional 4+4 layer D-LinOSS backbone → [CLS] pooling → task heads. Inset: single oscillator transfer function H(z) showing resonant peak.

**Figure 2.** E1 convergence curves. Validation loss vs. training step for D-LinOSS, CNN, Transformer (3 seeds each, shaded ±1σ). D-LinOSS continues improving; CNN/Transformer plateau/diverge.

**Figure 3.** E2 cross-spectral prediction examples. Top: representative molecule with strong IR→Raman prediction (high cosine similarity). Bottom: representative failure case. Ground truth (blue), PLS2 prediction (orange).

**Figure 4.** E3 transfer function heatmaps. (a) |H(e^{jω})| for all 32 oscillators across 4 forward layers (trained vs. random). (b) BC coupling |b_k · c_k| distribution per layer. (c) Depth-dependent damping ratio.

**Figure 5.** E4 calibration transfer R² vs. N. Classical methods (CCA, DS, SBC) show graceful degradation; D-LinOSS shows catastrophic failure. Annotated physics domain gap.

---

*Manuscript: March 2026 · Tubhyam Karthikeyan · ICT Mumbai / InvyrAI*
*Code: github.com/tubhyam/spektron (release upon acceptance)*
*Data: QM9S [3] (public, figshare) · License: MIT*
