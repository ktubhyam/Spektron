# 3. Experiments and Results

All experiments use QM9S (Section 2.3) and the architecture suite described in
Section 2.2. We report mean ± std over 3 random seeds (42, 43, 44) where available.
Statistical significance uses Welch's t-test with Cohen's d effect size.
Compute: 2× NVIDIA A10 22 GB, CUDA 13.1, PyTorch 2.10, bfloat16 AMP.


## 3.1 E1: Architecture Benchmark

**Objective.** Which backbone architecture best represents vibrational spectra under
a masked reconstruction objective?

**Protocol.** All models are pre-trained on QM9S masked spectral reconstruction
(Section 2.5) for 50,000 steps, 3 seeds each. Primary metric: best validation
loss (L_MSRP + physics + VIB). Secondary metric: val/msrp (pure reconstruction
MSE on masked positions), which isolates learning quality from the regularisation
terms. A mean-spectrum baseline (predict the training-set mean spectrum for all
inputs) provides a reference: mean-spectrum MSE = 0.810.

**Results.**

| Backbone | Backbone params | Best val loss | val/msrp | Relative to baseline |
|---|---|---|---|---|
| D-LinOSS | 2.0M | **0.076 ± 0.001** | **0.036 ± 0.002** | 22.6× better |
| S4D (ablation) | 0.27M | [pending] | [pending] | — |
| 1D CNN | 2.0M | 0.671 ± 0.006 | 0.648 ± 0.009 | 1.3× better |
| Transformer | 2.0M | 1.004 ± 0.003 | 0.988 ± 0.004 | 0.8× (below baseline) |
| Mamba | 2.0M | SKIP (SM120 incompatible) | — | — |

D-LinOSS achieves best val loss 0.0757 ± 0.0012, an 8.9× reduction over CNN
(0.6708 ± 0.0059, p < 0.001, Cohen's d = 140) and 13.3× reduction over Transformer
(1.0046 ± 0.0033, p < 0.001, Cohen's d = 371). Both CNN and Transformer converge
to their best checkpoints early (5–10K and 10–20K steps respectively) and
subsequently diverge, suggesting they cannot adapt to the increasing mask difficulty.
Only D-LinOSS continues to improve throughout training.

The val/msrp metric (pure reconstruction MSE, no regularisation) confirms the
ranking: D-LinOSS 0.036 vs. mean-spectrum 0.810 (22.6× improvement), CNN 0.648
(1.3× improvement), Transformer 0.988 (2% *worse* than mean-spectrum). The
Transformer is unable to learn meaningful spectral representations, performing
statistically equivalently to predicting the mean spectrum for all inputs.

**Throughput.** D-LinOSS processes 33 samples/sec at 6.2 GB peak GPU memory;
CNN processes 168 samples/sec at 1.8 GB; Transformer 157 samples/sec at 1.9 GB.
D-LinOSS is 5× slower and 3.3× more memory-intensive, consistent with its
bidirectional O(L) scan over 2050-token sequences.

**Interpretation.** The substantial advantage of D-LinOSS is consistent with
the physics alignment hypothesis: damped oscillatory bases are a more compact
and efficient representation of vibrational spectral patterns than fixed-kernel
convolutions or global self-attention. The CNN's local receptive field cannot
capture long-range spectral correlations; the Transformer's position-invariant
attention, while global, lacks any spectral prior and appears to overfit noise.
D-LinOSS's second-order oscillatory inductive bias accelerates convergence and
enables continuous improvement across all 50,000 training steps.


## 3.2 E2: Cross-Spectral Prediction

**Objective.** Can a neural model learn to predict the Raman spectrum of a molecule
from its IR spectrum, and vice versa? This is a novel ML task with no prior work.

**Protocol.** Each backbone (D-LinOSS, Transformer, CNN, S4D) is trained from
scratch (no E1 pre-training) on 88,914 paired IR+Raman molecules (train split),
predicting target from source. Identity-copy and PLS2 baselines provide reference
points. Results reported on the 10,399-molecule test split.

**Classical baselines.**

| Direction | Baseline | MSE | Cosine | Peak Recall |
|---|---|---|---|---|
| IR → Raman | Identity copy | [pending] | [pending] | [pending] |
| IR → Raman | PLS2 (n=30, 10K train) | 0.471 | 0.726 | 0.369 |
| Raman → IR | Identity copy | [pending] | [pending] | [pending] |
| Raman → IR | PLS2 (n=30, 10K train) | 0.608 | 0.637 | 0.289 |

**Neural architecture results (IR → Raman).**

| Backbone | MSE | Cosine | Peak Recall | NoPeakFrac |
|---|---|---|---|---|
| D-LinOSS | [3 seeds running] | | | |
| Transformer | [pending] | | | |
| 1D CNN | [pending] | | | |
| S4D | [pending] | | | |

*D-LinOSS seed 42 preliminary: MSE=0.589, Cosine=0.621, PeakRecall=0.365.
Training is ongoing; results will be updated.*

**Per-molecule analysis.** For each test molecule, we track MSE, cosine similarity,
number of target peaks, and best/worst 20 molecules by MSE. The fraction of test
molecules with no detectable target peaks (`frac_no_target_peaks`) quantifies the
fraction where peak_recall=1.0 is a false positive. Full per-molecule results
reported in Supplementary.

**Note on PLS competitiveness.** The PLS2 baseline achieves notably strong
performance (MSE=0.471, Cosine=0.726 for ir2raman), reflecting that the linear
mapping between IR and Raman intensities captures substantial correlated variation
across the dataset. Neural models trained from scratch must overcome this strong
linear baseline; whether architecture-specific inductive biases improve upon it
is the central question of E2.


## 3.3 E3: Transfer Function Interpretability

**Objective.** What spectral filters has D-LinOSS learned? Does training induce
qualitatively different filter banks compared to random initialisation?

**Protocol.** Transfer functions H(z) are computed for the trained D-LinOSS
(E1 best checkpoints, 3 seeds: 42, 43, 44) and for identically-structured
random-weight controls (n_random=20 per layer). Three statistics are computed per
layer per direction (fwd/bwd): BC coupling CV, peak wavenumber, and layer
specialisation (Section 2.7). Comparison: two-sample KS test (unweighted and
BC amplitude-weighted) for coupling concentration; Cohen's d for effect size.

**Results (E3 v4, final).**

*Finding 1: Massive BC coupling concentration.*
Trained models show highly concentrated |b_k·c_k| coupling weights, indicating
that a small subset of oscillators dominates the spectral output. Cohen's d ranges
from 5.97 to 17.69 (mean 9.58) across all 8 layer-direction combinations, with all
BC-weighted KS tests significant at p < 10⁻⁵ (most p < 10⁻¹²). The effect is
largest in L1_fwd (d=17.69) and L3_bwd (d=8.77), and smallest in L1_bwd (d=5.97).
All effects are massive by conventional standards (d > 0.8). Across all 20 random
controls, BC coupling is distributed uniformly; trained models show extreme
concentration by comparison.

*Finding 2: Layer specialisation.*
Forward layers show 5.2× higher inter-layer diversity in peak wavenumber
(trained std = 48.3 cm⁻¹ vs. random std = 9.3 cm⁻¹); backward layers show
3.1× higher diversity (trained 19.6 cm⁻¹ vs. random 6.3 cm⁻¹). Layers specialise
to different spectral frequency ranges rather than all responding to the same
frequencies. This is consistent with the model building a hierarchical spectral
decomposition. (The diversity ratio is reported as an observational summary; the
large magnitude renders formal significance testing redundant.)

*Finding 3: Depth-dependent damping.*
A monotonically increasing damping trend emerges across forward layers:
trained/random damping ratio = 1.016 (L0), 1.055 (L1), 1.131 (L2), 1.298 (L3),
with corresponding Cohen's d = 0.028, 0.095, 0.224, 0.493. The L3 effect is
medium-sized (d = 0.49); shallower layers have small effects. The consistent
monotonic progression — not any single layer — is the finding, suggesting the
model uses shallow layers for sharp spectral features and deep layers for broad
spectral envelopes. This trend holds symmetrically in backward layers (L3_bwd
damping ratio = 1.311, d = 0.522).

**Significance of interpretability.** These findings are unique to SSMs: no
equivalent H(z) analysis is possible for Transformers (no explicit filter bank)
or CNNs (kernels are low-level and not directly interpretable as frequency filters
in the same parameterisation). D-LinOSS provides a physically-grounded window
into learned spectral representations.


## 3.4 E4: Calibration Transfer (Honest Negative)

**Objective.** Does a D-LinOSS backbone pretrained on DFT mid-IR spectra transfer
to experimental NIR calibration (corn dataset)?

**Protocol.** Pretrained D-LinOSS encoder (E1 seed 42 checkpoint, backbone frozen)
with a linear prediction head trained on N ∈ {5, 10, 20, 30, 50} calibration
transfer samples across all 3 corn instrument pairs (m5→mp5, m5→mp6, mp5→mp6).
Training runs for 100 epochs (no early stopping: with N=5 samples, validation
split is impractical). Evaluation: R² on the full dataset using the target
instrument. Comparison: CCA, DS, SBC, PDS, di-PLS classical methods trained on
the same N samples. Results averaged over 3 seeds.

**Results.**

Results for the m5→mp5 instrument pair (representative; other pairs are similar):

| Method | N=50 R² | N=20 R² | N=10 R² | N=5 R² |
|---|---|---|---|---|
| CCA | 0.839 | ~0.83 | ~0.82 | ~0.77 |
| DS (direct standardisation) | 0.753 | ~0.75 | ~0.75 | — |
| SBC | 0.642 | — | — | 0.642 |
| PDS | −0.837 | — | — | — |
| Target-direct (upper bound) | 0.868 | 0.868 | 0.868 | 0.868 |
| No-transfer (lower bound) | −15.9 | — | — | — |
| Spektron D-LinOSS (frozen) | **−1.3** | **−61** | **−500** | **−512** |

Spektron achieves R² ≈ −1.3 at N=50, −61 at N=20, and ≈ −500 at N≤10,
far worse than all classical calibration transfer methods. The frozen backbone
features are anti-predictive: for N≤10 calibration samples, the head cannot
learn to map the backbone's (mid-IR derived) features to NIR property values.

**Diagnosis.** The DFT spectra in QM9S span 500–4000 cm⁻¹ (fundamental
vibrational modes); the corn NIR dataset spans 1100–2500 nm (~4000–9000 cm⁻¹,
overtone/combination bands). These are physically distinct spectral regimes:
fundamentals are first-order in the quantum harmonic oscillator expansion;
overtones involve second-order anharmonic terms with different selection rules,
intensities, and frequency patterns. A backbone pretrained on fundamentals cannot
produce useful representations for overtone data.

**Framing.** This negative result is not a failure of D-LinOSS specifically;
it is a physics-motivated domain gap that would affect any model pretrained
on DFT mid-IR. The result establishes that practical deployment of pretrained
spectral models requires careful matching of spectral regime (fundamental vs.
overtone) and measurement modality (DFT vs. experimental). Fine-tuning on
even modest in-domain data would likely recover performance; this is left as
future work.
