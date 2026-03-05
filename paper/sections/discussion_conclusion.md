# 4. Discussion

## 4.1 Physics Alignment Hypothesis

The E1 results are striking: D-LinOSS outperforms the mean-spectrum baseline by
22.6× on val/msrp, while CNN achieves only 1.3× and Transformer 0.8× (below
baseline). The magnitude of this gap — nearly an order of magnitude — is difficult
to attribute to parameter count alone, since all backbones are matched at ~2M
parameters. We interpret this as evidence for the physics alignment hypothesis:
D-LinOSS's second-order oscillatory inductive bias provides a qualitatively better
function-space prior for vibrational spectra.

A useful analogy: a Fourier basis is an efficient representation for periodic
signals not because individual Fourier components correspond to individual signal
components, but because the function class is well-matched. Similarly, D-LinOSS's
damped oscillatory basis is well-matched to the superposition of damped Lorentzians
that constitutes a vibrational spectrum, regardless of whether any single learned
oscillator corresponds to any single physical mode.

The CNN's failure is mechanistic: local receptive fields (even with dilations)
cannot model the global spectral envelope or long-range correlations between peaks
arising from related molecular motions. The Transformer's failure is more surprising:
despite global attention, it falls below the mean-spectrum baseline, suggesting it
is fitting noise rather than spectral structure. This may reflect that
position-agnostic attention (without strong positional biases) is poorly suited
to 1D sequences where *position* (wavenumber) is the fundamental physical variable.

## 4.2 Cross-Spectral Prediction and the Role of Architecture

E2 addresses the question of whether neural spectral representations support
cross-modal transfer. The strong PLS2 baseline (ir2raman MSE=0.471, Cosine=0.726)
demonstrates that substantial correlated variation between IR and Raman is captured
by a linear mapping from 10,000 training samples. This is not surprising from
a physical standpoint: both spectra derive from the same normal modes, so
systematic patterns in the IR intensity landscape will correlate with patterns
in the Raman landscape.

Whether neural models can improve beyond PLS2 — and whether D-LinOSS specifically
provides an advantage — is the key empirical question of E2. Results are pending
for most backbone/seed combinations. The preliminary D-LinOSS seed 42 result
(MSE=0.589 after 30,000 steps) is below PLS2, consistent with a model that has
not fully converged. With 3 seeds and full training, the picture will clarify.

The cross-spectral task also provides insight into what the learned representations
encode. A model that learns good spectral representations should in principle
encode the shared molecular structure that drives both IR and Raman patterns.
Whether this is actually true — whether pretrained E1 features help E2 performance
over training from scratch — is an important question for future work.

## 4.3 Transfer Function Interpretability

E3's findings (BC coupling concentration Cohen's d = 5.97–17.69, layer specialisation)
demonstrate that trained D-LinOSS models develop qualitatively structured filter
banks. The massive effect sizes are not due to the three findings being subtle —
they indicate that training radically reorganises the coupling structure from the
random initialisation.

The layer specialisation finding (shallow layers for sharp features, deep layers
for broad envelopes) is consistent with standard deep learning intuition about
hierarchical feature extraction, here manifested as depth-dependent spectral
bandwidth. This provides mechanistic transparency that is uniquely available
for SSMs: we can assign an interpretable spectral function to each layer.

## 4.4 Limitations

1. **S4D incomplete.** The S4D ablation (first-order diagonal SSM vs. D-LinOSS
   second-order) is still running. Without these results, we cannot confirm that
   the *oscillatory structure specifically* (rather than the SSM family broadly)
   drives E1 performance.

2. **E2 incomplete.** Cross-spectral results are available for only one architecture
   and one seed. The relative ranking of neural architectures vs. PLS2 remains
   to be established.

3. **Training from scratch in E2.** Our E2 setup trains end-to-end from random
   initialisation rather than fine-tuning pretrained E1 weights. Pretrained
   initialisation might improve performance and better demonstrate the value of
   pretraining; this is left for future work.

4. **QM9S is DFT-computed.** All spectra are theoretical, not experimental.
   The sim-to-real gap (DFT vs. experimental linewidths, peak positions, intensities)
   means results may not transfer directly to instrument-measured spectra.

5. **Small molecule scope.** QM9S contains molecules with ≤9 heavy atoms.
   Generalization to larger drug-like or natural product molecules is not validated.


# 5. Conclusion

We have presented the first systematic evaluation of state space models for
vibrational spectroscopy, using the QM9S dataset (129,817 DFT-computed IR and
Raman spectra) as a benchmark. D-LinOSS, a second-order oscillatory SSM, achieves
masked spectral reconstruction loss 0.076 ± 0.001 — a 22.6× improvement over the
mean-spectrum baseline and an 8.8× improvement over the best competing architecture
(CNN, 0.671 ± 0.006). The Transformer baseline falls below the mean-spectrum baseline,
demonstrating that strong general-purpose architectures do not automatically succeed
on spectral data without appropriate inductive biases.

We introduce cross-spectral prediction as a novel ML task and establish PLS2 as a
strong classical baseline. Transfer function analysis of trained D-LinOSS reveals
highly structured filter banks (BC coupling concentration Cohen's d = 5.97–17.69,
inter-layer specialisation 5.2× above random) that provide mechanistic transparency
not available for Transformers or CNNs. A calibration transfer experiment documents
a physics-motivated negative result: DFT mid-IR pretraining does not transfer to
experimental NIR overtone data.

Together, these results support the physics alignment hypothesis: when model
function-space priors match the data-generating process, the resulting inductive
bias provides large empirical advantages. For vibrational spectroscopy, damped
harmonic oscillators are a natural prior; D-LinOSS instantiates this prior and
benefits substantially from it.
