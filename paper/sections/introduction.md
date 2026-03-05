# 1. Introduction

Vibrational spectroscopy — infrared absorption and Raman scattering — encodes the
chemical identity of a molecule in a compact, physically interpretable form. Every
peak corresponds to a normal mode of nuclear motion; the ensemble of peaks constitutes
a molecular fingerprint that is sensitive to composition, conformation, and bonding
environment. These properties have made IR and Raman spectroscopy indispensable tools
across analytical chemistry, materials characterisation, pharmaceutical quality control,
and structural biology. Yet the information content of a spectrum is rarely exhausted
by the handful of peaks that a human analyst annotates. Buried in the continuum of
overlapping bands, cross-peak correlations, and baseline curvature lies structure that
resists manual extraction but may be accessible to learned representations.

The recent proliferation of large, high-quality computed spectral datasets creates a
new opportunity. The QM9S dataset, for example, provides DFT-level IR and Raman
spectra for 129,817 small organic molecules at the B3LYP/def2-TZVP level of theory,
each resolved to 2048 points across 500–4000 cm⁻¹. At this scale, data-driven models
can in principle learn the statistical regularities that connect molecular structure
to spectral shape. Deep learning has already demonstrated this principle in related
domains: convolutional architectures for peak identification, graph neural networks for
property prediction from structure, and transformer-based models for molecular
generation. The application of these ideas directly to raw spectral sequences, however,
remains comparatively underdeveloped. In particular, the question of whether the
inductive biases built into a given architecture are *well-matched* to the physics of
vibrational spectra has not been rigorously studied.

This question motivates the present work. A vibrational spectrum is, at its mechanistic
origin, a superposition of damped harmonic responses. Each normal mode contributes a
Lorentzian band centred at the mode frequency, with a width determined by the relaxation
time of that mode. The time-domain picture is equally clear: the free induction decay
following an impulsive excitation is a sum of exponentially decaying sinusoids. This
structure suggests a natural prior for any model that operates on spectra: the
signal-processing primitives most aligned with the data-generating process are damped
oscillators, not convolutions with fixed kernels, not scaled dot-product attention, and
not basis expansions that treat the spectrum as an arbitrary real-valued vector. Diagonal
Linear Oscillatory State Space Models (D-LinOSS) [CITE] implement exactly this prior:
each layer propagates a bank of second-order oscillatory modes with learnable frequency,
damping, and mixing coefficients. The function-space hypothesis of this paper is that
D-LinOSS is a more natural basis for representing vibrational spectral features than
architectures whose inductive biases carry no such physical alignment. We emphasise
that this is a *model-class argument*: the oscillator frequencies in the state-space
representation are signal-processing parameters, not wavenumbers, and we make no
claim of a one-to-one correspondence between learned modes and physical vibrations.

Despite the growing literature on deep learning for spectroscopy, no prior work has
evaluated state space models on vibrational spectral data. The SSM family — including
S4 [CITE], S4D [CITE], Mamba [CITE], and their variants — has achieved strong
performance on long-range sequence tasks across multiple domains, but its application
to spectroscopy has not been reported. This gap is notable given the physics alignment
argument above, and closing it requires a controlled empirical study with matched
parameter budgets across architectures.

A second gap concerns the relationship between IR and Raman spectra of the same
molecule. Both probe nuclear motion, but through complementary selection rules: IR
intensity reflects the change in dipole moment during a vibration, while Raman
intensity reflects the change in polarisability. For a given molecule, the two spectra
are not independent — they share the same underlying normal mode structure — yet they
are not simply related by a pointwise transformation. The problem of *cross-spectral
prediction*: predicting one spectral modality from the other, has not been studied as
a machine learning task. It is a natural testbed for learned inter-modal representations
and has practical relevance: Raman instrumentation is more costly and less ubiquitous
than FTIR, so a model that reliably predicts Raman spectra from IR measurements would
have direct analytical utility.

We present a four-experiment empirical study addressing both gaps. The specific
contributions are:

1. **Architecture benchmark (E1).** We provide the first systematic comparison of
   D-LinOSS against Transformers, 1D CNNs, S4D, and PLS on masked spectral
   reconstruction, finding that D-LinOSS achieves val loss 0.076 ± 0.001 — a 22.6×
   improvement over the mean-spectrum baseline, compared to 1.3× for CNN and 0.8×
   (below baseline) for Transformer — at matched ~2M backbone parameters.

2. **Cross-spectral prediction (E2).** We introduce IR↔Raman cross-spectral
   prediction as a novel ML task and evaluate all architectures against a PLS2
   classical baseline, providing the first quantitative characterisation of how well
   neural spectral representations support cross-modal transfer.

3. **Transfer function interpretability (E3).** We develop an H(z) analysis framework
   for trained D-LinOSS models, demonstrating massive BC coupling concentration
   (Cohen's d = 6.8–8.5 vs. random weights) and strong inter-layer specialisation
   (5.2× higher diversity in forward layers vs. random). This provides mechanistic
   insight unavailable for Transformers or CNNs.

4. **Calibration transfer (E4, honest negative).** We evaluate whether features
   pretrained on DFT mid-IR spectra transfer to experimental NIR calibration (corn
   dataset), finding catastrophic negative transfer (R² ≈ −1.3 at N = 50 samples)
   attributable to the fundamental physics gap between overtone NIR bands and
   DFT-computed fundamentals.

The remainder of the paper is organised as follows. Section 2 provides background on
D-LinOSS and competing architectures. Section 3 describes datasets and evaluation
metrics. Sections 4–7 present experiments E1–E4 in sequence. Section 8 discusses the
physics alignment hypothesis in light of the results, limitations, and future directions.
