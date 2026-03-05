# Theoretical Analysis: Why Oscillatory SSMs Are Natural for Vibrational Spectroscopy

## The Physics of Vibrational Spectra

Vibrational spectra fundamentally represent damped harmonic oscillations. Each molecular vibrational mode follows the equation of motion:

$$\ddot{x} + 2\zeta\omega_0\dot{x} + \omega_0^2 x = 0$$

where:
- $\zeta$ = damping ratio (energy dissipation due to anharmonicity and coupling)
- $\omega_0$ = natural frequency (determined by bond strength and atomic masses)

This second-order linear ODE has the solution:

$$x(t) = Ae^{-\zeta\omega_0 t}\cos(\omega_d t + \phi), \quad \omega_d = \omega_0\sqrt{1-\zeta^2}$$

A spectrum is not a random signal—it is an **ordered sequence of resonances**, one per vibrational mode.

## From Physics to Spectroscopy: The Lorentzian Resonance

When IR light probes these vibrations, the transmitted or reflected intensity exhibits a characteristic Lorentzian lineshape at each resonance:

$$I(\omega) = \frac{C}{(\omega_0^2 - \omega^2)^2 + (2\zeta\omega_0\omega)^2}$$

This is the **frequency-domain signature of a damped oscillator**. A complete vibrational spectrum is a superposition of hundreds to thousands of these resonances:

$$I_{\text{total}}(\omega) = \sum_j^{N_{\text{modes}}} I_j(\omega) + \text{baseline}$$

**Key insight:** Vibrational spectra are linearly composed of damped oscillatory basis functions. This is not an approximation—it is the physics.

## The Information-Theoretic Argument

Consider reconstructing a spectrum from a compressed representation (the central problem in spectroscopic machine learning). Two strategies:

1. **Generic basis:** Use filters with no relationship to oscillatory dynamics (e.g., convolutional filters, attention patterns). The model must learn, from data alone, that spectra are made of resonances. This requires:
   - Many parameters to approximate oscillatory behavior
   - High sample complexity (many spectra to learn the pattern)
   - Poor generalization across different spectral domains (IR vs. Raman—different frequency ranges, same physics)

2. **Oscillatory basis:** Build the model from damped oscillatory primitives. The physics is encoded in the architecture. The model only needs to learn:
   - Which frequencies are active (parameter values)
   - Which intensities are active (parameter magnitudes)
   - Domain-specific adjustments (calibration offsets)

The oscillatory basis dramatically reduces model complexity and sample requirements—it is the **correct inductive bias**.

## D-LinOSS: A Discrete Damped Oscillator SSM

D-LinOSS (Damped Linear Oscillator State Space Model) implements this physics in a learnable, discrete-time recurrence. Each layer maintains a hidden state $h_t = [h_t^{(1)}, h_t^{(2)}]^T$ and updates it via:

$$h_t^{(1)} = (1 - \delta_t a)h_{t-1}^{(1)} + \delta_t b x_t + (1 - \delta_t a - \delta_t^2 g)h_{t-1}^{(2)}$$

$$h_t^{(2)} = -\delta_t a h_{t-1}^{(1)} + (1 - \delta_t g)h_{t-1}^{(2)}$$

where:
- $\delta_t$ = integration step (time discretization)
- $a$ = damping parameter (maps to $\zeta\omega_0$ in continuous time)
- $g$ = frequency parameter (maps to $\omega_0^2$ in continuous time)
- $b$ = input coupling strength

### Transfer Function Interpretation

The z-domain transfer function is:

$$H(z) = \frac{\delta_t^2 b z}{(1+\delta_t g)z^2 - (2+\delta_t g - \delta_t^2 a)z + 1}$$

This is a **second-order resonant filter**. The poles are:

$$p_{1,2} = \frac{(2+\delta_t g - \delta_t^2 a) \pm \sqrt{\Delta}}{2(1+\delta_t g)}$$

where $\Delta = (2+\delta_t g - \delta_t^2 a)^2 - 4(1+\delta_t g)$.

For complex conjugate poles (oscillatory behavior), these encode the learned resonant frequencies and damping. This is the **signature of a damped oscillator**, not a generic filter.

## Why Other Architectures Miss This Structure

### Transformers (Self-Attention)
- **Mechanism:** Learn pairwise dependencies between all spectral points
- **Spectroscopy match:** None. Self-attention is fundamentally agnostic to oscillatory structure.
- **Cost:** The model must infer that spectra are composed of resonances. This requires:
  - High-dimensional representations (~200-400 hidden dims) to capture 100+ overlapping resonances
  - Quadratic attention complexity O(n²), prohibitive for long spectra
  - No transfer of learned resonance patterns across different spectral modalities

### 1D Convolutional Neural Networks
- **Mechanism:** Learn local FIR filters via weight sharing
- **Spectroscopy match:** Weak. While convolutions can approximate oscillatory filters, they:
  - Have no explicit damping structure
  - Require deep stacking to capture multiple resonant modes (each layer learns one low-order approximation)
  - Cannot produce interpretable filter banks—a trained CNN cannot tell you the frequencies it has learned
- **Cost:** High parameter count relative to D-LinOSS for the same spectral reconstruction accuracy

### S4/S4D (Diagonal SSM Baselines)
- **Mechanism:** First-order SSM with learnable dynamics but no oscillatory constraint
- **Spectroscopy match:** Partial. Diagonal SSMs are more parameter-efficient than CNNs but lack the second-order oscillatory structure.
- **Cost:** The model can learn to approximate oscillatory behavior, but it is not the natural representation. This shows up as:
  - Larger approximation error (higher reconstruction loss)
  - Less interpretable learned dynamics (no resonant poles)

### Mamba (Selective SSM)
- **Mechanism:** First-order SSM with input-dependent dynamics
- **Spectroscopy match:** Moderate. Selectivity helps the model learn which spectral regions are important, but:
  - Fundamentally a first-order system (cannot represent true damped oscillation without approximation)
  - Parameters select *which* regions, not *why* they matter (no oscillatory semantics)

## D-LinOSS: Matched to Chemistry

D-LinOSS aligns with the physics in three ways:

1. **Structural match:** The second-order recurrence directly instantiates damped harmonic dynamics. No approximation needed.

2. **Parameter efficiency:** Each layer learns two parameters ($a$ and $g$) that directly encode damping and frequency. A single D-LinOSS layer with learned parameters is equivalent to a hand-crafted resonant filter. A stack of layers is a learned filter bank.

3. **Interpretability:** The transfer function $H(z)$ can be extracted post-training and analyzed. The poles reveal learned frequencies:

$$\omega_{\text{learned}} = \arg(p) = \arccos\left(\frac{2+\delta_t g - \delta_t^2 a}{2\sqrt{1+\delta_t g}}\right)$$

These frequencies should cluster around known functional group absorption bands:
- C-H stretching: 2850–3000 cm⁻¹
- C=O stretching: 1650–1750 cm⁻¹
- O-H stretching: 3200–3600 cm⁻¹
- etc.

This interpretability is **unique to oscillatory SSMs**. A trained Transformer or CNN cannot produce frequency-domain filter characteristics.

## Summary: The Function-Space Argument

The key claim is **not** that D-LinOSS parameters correspond to true molecular vibrational frequencies (they do not—signal processing frequencies $\neq$ spectroscopic cm⁻¹).

The claim is that **damped sinusoids are the natural basis for vibrational spectra**. D-LinOSS is a learnable parameterization of this basis. It should therefore:
1. Achieve lower reconstruction error (fewer parameters, less data needed)
2. Transfer better across spectral domains (the physics is the same in IR and Raman)
3. Produce interpretable dynamics (learned poles match chemical intuition)

The empirical validation of these three claims is the subject of Experiments 1, 3, and 2 respectively.
