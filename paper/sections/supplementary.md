# Supplementary Information

## S1. Training Details and Hyperparameters

**E1 (Architecture Benchmark)**

| Hyperparameter | Value |
|---|---|
| max_steps | 50,000 |
| batch_size | 16 (effective 64 with grad_accum=4) |
| optimizer | AdamW, lr=3×10⁻⁴, weight_decay=0.01 |
| lr_schedule | Cosine annealing, warmup 2000 steps |
| AMP | bfloat16 (LinOSS blocks forced float32) |
| Mask fraction | 0.30 |
| Seeds | 42, 43, 44 |
| Devices | 2× NVIDIA A10 22GB (DataParallel, except S4D/Mamba: single GPU) |

**D-LinOSS architecture (E1)**

| Parameter | Value |
|---|---|
| n_layers | 4 (bidirectional: 4 fwd + 4 bwd = 8 LinOSS blocks) |
| d_model | 256 |
| d_state | 64 |
| Backbone params | ~2.0M |
| Total params | ~6.6M |

**E2 (Cross-Spectral Prediction)**

| Hyperparameter | Value |
|---|---|
| max_steps | 30,000 |
| batch_size | 64 |
| optimizer | AdamW, lr=3×10⁻⁴, weight_decay=0.01 |
| lr_schedule | Cosine annealing, warmup 1000 steps |
| Source augmentation | Gaussian noise σ=0.01 |
| CrossSpectralHead | 3-layer MLP, hidden=d_model, tanh |


## S2. E1 Per-Run Results

| Backbone | Seed | Best Val Loss | Val/MSRP | Peak GPU (GB) | Time (h) |
|---|---|---|---|---|---|
| D-LinOSS | 42 | 0.0747 | 0.0358 | 6.2 | ~6.7 |
| D-LinOSS | 43 | 0.0774 | 0.0360 | 6.2 | ~6.7 |
| D-LinOSS | 44 | 0.0748 | 0.0360 | 6.2 | ~6.7 |
| 1D CNN | 42 | 0.6636 | 0.6477 | 1.8 | ~1.3 |
| 1D CNN | 43 | 0.6708 | ~0.65 | 1.8 | ~1.3 |
| 1D CNN | 44 | 0.6781 | ~0.65 | 1.8 | ~1.3 |
| Transformer | 42 | 1.0089 | 0.9877 | 1.9 | ~1.4 |
| Transformer | 43 | 1.0009 | ~0.99 | 1.9 | ~1.4 |
| Transformer | 44 | 1.0038 | ~0.99 | 1.9 | ~1.4 |
| S4D | 42 | [pending] | | | |
| S4D | 43 | [pending] | | | |
| S4D | 44 | [pending] | | | |
| Mamba | all | SKIP (SM120 hardware) | | | |

**Mean-spectrum baseline MSE**: 0.810 (predict training mean for all queries)
**Zero-prediction baseline MSE**: 1.000 (predict all zeros)


## S3. E2 Per-Run Results (IR → Raman)

| Backbone | Seed | MSE | Cosine | PeakRecall | NoPeakFrac |
|---|---|---|---|---|---|
| D-LinOSS | 42 | 0.589 | 0.621 | 0.365 | [pending] |
| D-LinOSS | 43 | [running] | | | |
| D-LinOSS | 44 | [pending] | | | |
| Transformer | all | [pending] | | | |
| 1D CNN | all | [pending] | | | |
| S4D | all | [pending] | | | |
| PLS2 (baseline) | — | 0.471 | 0.726 | 0.369 | — |

## S3b. E2 Per-Run Results (Raman → IR)

| Backbone | Seed | MSE | Cosine | PeakRecall | NoPeakFrac |
|---|---|---|---|---|---|
| D-LinOSS | all | [pending] | | | |
| Transformer | all | [pending] | | | |
| 1D CNN | all | [pending] | | | |
| S4D | all | [pending] | | | |
| PLS2 (baseline) | — | 0.608 | 0.637 | 0.289 | — |


## S4. E3 Transfer Function Statistics (Full Table)

All values: trained D-LinOSS (mean over 3 seeds s42, s43, s44) vs. random controls (n=20).

| Layer | Direction | Cohen's d (BC coupling) | KS p-value (weighted) | Damping ratio | Cohen's d (damping) |
|---|---|---|---|---|---|
| L0 | fwd | 8.48 | 1.4e-15 | 1.016 | 0.028 |
| L1 | fwd | 17.69 | 2.2e-13 | 1.055 | 0.095 |
| L2 | fwd | 7.00 | 1.3e-09 | 1.131 | 0.224 |
| L3 | fwd | 6.78 | 7.8e-37 | 1.298 | 0.493 |
| L0 | bwd | 7.42 | 3.9e-06 | 1.010 | 0.017 |
| L1 | bwd | 5.97 | 7.2e-13 | 1.045 | 0.078 |
| L2 | bwd | 14.48 | 3.2e-20 | 1.176 | 0.300 |
| L3 | bwd | 8.77 | 4.7e-23 | 1.311 | 0.522 |

KS p-values: two-sample KS test on BC amplitude-weighted frequency distributions.
All Cohen's d for BC coupling are massive (d > 0.8); damping Cohen's d is small-to-medium.


## S5. E4 Calibration Transfer — Full Results

Corn dataset. Three instrument pairs. Baselines use N transfer samples from source→target.
Spektron: D-LinOSS backbone (seed 42, frozen), linear head, 100 epochs, 3 seeds.

**m5→mp5 pair:**

| Method | N=5 | N=10 | N=20 | N=30 | N=50 |
|---|---|---|---|---|---|
| CCA | ~0.77 | ~0.80 | ~0.82 | ~0.83 | 0.839 |
| DS | — | ~0.75 | ~0.75 | ~0.75 | 0.753 |
| SBC | 0.642 | — | — | — | 0.642 |
| PDS | — | — | — | — | −0.837 |
| No-transfer | — | — | — | — | −15.9 |
| Target-direct (upper bound) | — | — | — | — | 0.868 |
| Spektron D-LinOSS (frozen) | −512 ± 0.7 | −500 ± 0.3 | −60 ± 1.4 | −46 ± 0.8 | −1.34 ± 0.15 |

**m5→mp6 pair:** CCA=0.820/DS=0.763/SBC=0.533/Spektron N=50: −1.25 ± 0.37

**mp5→mp6 pair:** CCA=0.798/DS=0.763/SBC=0.791/Spektron N=50: −1.26 ± 0.10

All Spektron values: R² mean ± std over 3 seeds. Full results: `experiments/results/e4_calibration_transfer.json`


## S6. E2 Per-Molecule Analysis (D-LinOSS, Best Run)

*To be completed when E2 training finishes.*

Best 20 molecules (lowest MSE) and worst 20 molecules (highest MSE) by SMILES,
with MSE, cosine similarity, and number of target Raman peaks per molecule.
Per-molecule data stored in `experiments/results/e2_cross_spectral.json` under
`runs[*].per_molecule`.
