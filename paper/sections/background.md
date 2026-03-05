# 2. Background

## 2.1 Vibrational Spectroscopy

A molecule with N atoms has 3N − 6 vibrational normal modes (3N − 5 for linear
molecules). Each normal mode is characterised by a frequency ω_k (cm⁻¹) and a
spatial displacement pattern. The vibrational spectrum is a superposition of
Lorentzian bands centred at ω_k:

    I(ω) = Σ_k A_k · Γ_k / [(ω − ω_k)² + Γ_k²]

where A_k is the mode intensity and Γ_k the linewidth. **IR intensity** A_k^IR is
proportional to the squared derivative of the molecular dipole moment along mode k:
A_k^IR ∝ |∂μ/∂Q_k|². **Raman intensity** A_k^R is proportional to the squared
derivative of the molecular polarisability tensor: A_k^R ∝ |∂α/∂Q_k|².

Both spectral types are computed in QM9S by density functional theory (DFT) at the
B3LYP/def2-TZVP level, providing access to both the frequency positions and the
intensity patterns.

**Selection rules.** Whether a mode is active in IR, Raman, or both depends on the
molecular symmetry group. For non-centrosymmetric molecules (lacking an inversion
centre), modes can be simultaneously IR- and Raman-active. For centrosymmetric
molecules, the mutual exclusion rule applies: modes that are IR-active are
Raman-inactive, and vice versa. 99.93% of QM9S molecules are non-centrosymmetric.


## 2.2 State Space Models

Linear time-invariant (LTI) state space models map an input sequence u ∈ ℝ^L to
an output y ∈ ℝ^L through a hidden state h:

    ḣ(t) = A h(t) + B u(t)
    y(t) = C h(t) + D u(t)

where A, B, C, D are learnable matrices. Efficient computation for long sequences
is achieved by diagonalising A (S4D [CITE]) or by making A, B input-dependent
(Mamba [CITE]).

**D-LinOSS** [CITE] is a second-order SSM where each channel follows damped
harmonic oscillator dynamics. In contrast to S4D (which uses first-order, real
diagonal A), D-LinOSS pairs of channels share complex conjugate poles, enabling
oscillatory responses. See Section 2.1 for the full formulation.

**Transfer function.** For a discrete-time LTI system with z-transform transfer
function H(z), the frequency response |H(e^{jω})| quantifies how the system
amplifies or attenuates each frequency component. This provides a direct window
into what the model "looks for" in the input — an interpretability advantage
unique to SSMs.


## 2.3 Related Work

**Spectral ML.** DiffSpectra [CITE] uses diffusion to generate molecular graphs
from combined IR+Raman spectra (40.8% top-1 accuracy). Vib2Mol [CITE] combines
retrieval with property prediction (87% top-10 accuracy). VibraCLIP [CITE] applies
contrastive learning (81.7% top-1 when molecular mass is provided). None evaluate
SSM architectures or study cross-spectral prediction.

**SSMs for science.** S4 [CITE] and Mamba [CITE] achieve state-of-the-art
performance on long-range sequence benchmarks. Applications in scientific domains
include genomics [CITE] and audio [CITE], but not vibrational spectroscopy.
The present work is the first systematic evaluation of SSMs on IR/Raman data.

**Calibration transfer.** Piecewise direct standardisation (PDS) [CITE], direct
standardisation (DS) [CITE], and spectral basis correction (SBC) [CITE] are
classical methods for instrument-to-instrument transfer in NIR spectroscopy.
Deep learning approaches to calibration transfer have been studied primarily on
NIR data [CITE]; none evaluate mid-IR pretrained models on NIR tasks.

**Cross-spectral prediction.** No prior work studies the prediction of Raman spectra
from IR or vice versa as a machine learning task. The closest related work is
spectral translation between related spectral types in remote sensing [CITE], which
operates on fundamentally different physics (reflectance vs. molecular vibrations).
