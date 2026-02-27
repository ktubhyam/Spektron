# Spektron — Claude Code Instructions

## What This Project Is

Spektron is a **research paper + implementation** targeting publication at **Digital Discovery** (RSC).

**Paper Title:** "Oscillatory State Space Models for Vibrational Spectroscopy: Benchmarking, Cross-Spectral Prediction, and Interpretability"

**Author:** Tubhyam Karthikeyan (ICT Mumbai / InvyrAI)

## Paper Direction (2026-02-27)

An **honest empirical study** evaluating whether oscillatory SSMs (D-LinOSS) bring unique advantages to vibrational spectroscopy. NOT a "foundation model" paper — a careful benchmark + novel task + interpretability analysis.

**Key Claims (defensible):**
1. **First SSM evaluation for vibrational spectroscopy** — zero prior work applies SSMs to IR/Raman
2. **Novel cross-spectral prediction task** — predicting Raman from IR (and vice versa). No prior work exists (confirmed across 6 major surveys)
3. **Function-space argument** — D-LinOSS dynamics (damped sinusoids) are a natural basis for vibrational spectra. This is a model-class argument, NOT a parameter correspondence claim
4. **Transfer function interpretability** — H(z) analysis reveals what spectral filters each D-LinOSS layer has learned. Unique to SSMs (Transformers can't produce this)

**What NOT to claim:**
- D-LinOSS oscillator frequencies correspond to physical vibrational frequencies (they don't — signal-processing frequencies ≠ cm⁻¹)
- "Foundation model" (130K is too small; DreaMS used 700M)
- Pretrained mid-IR transfers to experimental NIR (different physics — overtones vs fundamentals)
- Peak-aware masking is novel (PRISM already did this)

## 4 Experiments

**E1: Architecture Benchmark** (`experiments/e1_benchmark.py`)
- D-LinOSS vs Mamba vs Transformer vs 1D CNN vs PLS + S4D ablation
- All param-matched at ~1M backbone params
- Masked spectral reconstruction on QM9S (130K, 2048 pts)
- 3 seeds, mean±std, FLOPs, wall-clock, memory

**E2: Cross-Spectral Prediction** (`experiments/e2_cross_spectral.py`)
- THE novel contribution — predict Raman from IR and vice versa
- 99.93% of QM9 molecules are non-centrosymmetric (mutual exclusion barely applies)
- Metrics: MSE + peak-position recall + intensity correlation on peaks
- Must include identity-copy baseline and molecular-level error analysis
- Characterize best/worst 100 molecules by symmetry and functional groups

**E3: Transfer Function Analysis** (`experiments/e3_transfer_function.py`)
- Compute H(z) = dt²·b·z / [(1+dt·g)·z² - (2+dt·g-dt²·a)·z + 1] for trained D-LinOSS
- Per-layer filter bank visualization (heatmap)
- Pole-zero plots overlaid with known functional group absorption bands
- Statistical comparison: learned frequencies vs random (control)

**E4: Calibration Transfer** (`experiments/e4_calibration_transfer.py`)
- Fine-tune pretrained architectures on corn (80×3×700) + tablet (655×2×650)
- Compare against PDS, SBC, DS, CCA, di-PLS baselines
- Honest about DFT mid-IR → experimental NIR domain gap
- Negative result here is fine if framed properly

## Project Structure

```
Spektron/
├── CLAUDE.md              ← YOU ARE HERE
├── requirements.txt
├── src/
│   ├── config.py          ← All hyperparameters (dataclass-based)
│   ├── data/
│   │   ├── qm9s.py           ← QM9S HDF5 loading + preprocessing
│   │   ├── cross_spectral.py ← IR↔Raman paired dataset
│   │   └── datasets.py       ← Corn/Tablet datasets + augmentation
│   ├── models/
│   │   ├── backbones.py   ← CNN1D backbone (new)
│   │   ├── dlinoss.py     ← D-LinOSS bidirectional backbone
│   │   ├── mamba.py       ← Mamba backbone
│   │   ├── transformer.py ← Transformer backbone
│   │   ├── embedding.py   ← WaveletEmbedding + WavenumberPE
│   │   ├── heads.py       ← VIB, Reconstruction, CrossSpectral heads
│   │   ├── spektron.py    ← Full model assembly (Spektron class)
│   │   └── linoss/        ← Vendored D-LinOSS core (DO NOT MODIFY)
│   │       ├── layers.py  ← DampedLayer, LinOSSBlock, GLU
│   │       └── scan.py    ← Parallel associative scan
│   ├── losses/
│   │   └── losses.py      ← MSRP, contrastive, physics, OT, VIB losses
│   ├── training/
│   │   └── trainer.py     ← Training loops
│   ├── evaluation/
│   │   ├── baselines.py              ← PDS, SBC, DS, CCA, di-PLS, PLS
│   │   ├── metrics.py                ← R², RMSEP, RPD, peak metrics
│   │   └── cross_spectral_metrics.py ← MSE, cosine, peak recall, SID
│   └── analysis/
│       └── transfer_function.py ← H(z) computation + visualization
├── experiments/
│   ├── e1_benchmark.py        ← Architecture benchmark
│   ├── e2_cross_spectral.py   ← Cross-spectral prediction
│   ├── e3_transfer_function.py← Transfer function analysis
│   ├── e4_calibration_transfer.py ← Calibration transfer
│   ├── pretrain_qm9s.py      ← Pretraining entry point
│   ├── run_all.py             ← Master experiment runner
│   ├── results/               ← JSON results
│   └── archive/               ← Old experiment scripts
├── data/
│   ├── raw/qm9s/             ← QM9S HDF5 (130K molecules)
│   ├── processed/corn/       ← 80 × 3 instruments × 700ch
│   └── processed/tablet/     ← 655 × 2 instruments × 650ch
├── figures/
│   ├── e1_benchmark/
│   ├── e2_cross_spectral/
│   ├── e3_transfer_function/
│   └── e4_calibration/
├── checkpoints/
├── logs/
├── paper/
│   ├── sections/             ← LaTeX/Markdown sections
│   └── archive/              ← Old paper planning docs
└── tests/
```

## Architecture (for benchmark)

All backbones share the same interface: `(B, L, d_model) → (B, L, d_model)`

| Architecture | Key Properties | Param Budget |
|-------------|---------------|-------------|
| D-LinOSS | 2nd-order oscillatory, O(n) scan, damped sinusoid basis | ~1M |
| Mamba | 1st-order selective SSM, O(n) scan, input-dependent | ~1M |
| Transformer | Self-attention, O(n²), positional encoding | ~1M |
| 1D CNN | Local convolutions, FIR filters, translation equivariant | ~1M |
| S4D (ablation) | 1st-order diagonal SSM, no oscillatory structure | ~1M |
| PLS | Classical partial least squares (no backbone) | N/A |

## Coding Standards

- **Python 3.10+**, PyTorch 2.x
- Type hints everywhere
- Config-driven (no magic numbers — everything from dataclass configs)
- Logging via Python `logging` module (not print statements)
- Reproducibility: all random seeds set via config
- D-LinOSS forced to float32 under AMP (prevents NaN from GLU overflow)

## Compute

- **Remote:** 2× RTX 5060 Ti 16GB on Vast.ai (`ssh -p 38170 root@62.107.25.198`)
- batch_size=16 (8/GPU), grad_accum=4, effective batch 64
- AMP bfloat16, LinOSSBlock forced float32
- ~39 samples/sec, ~23 hours for 50K steps
- Total compute budget: ~106 GPU-hours (~10-12 days wall-clock)

## Key Data

- **QM9S:** 129,817 molecules, IR (103,991) + Raman (129,817), DFT B3LYP/def2-TZVP
  - 3501 pts at 500-4000 cm⁻¹, resampled to 2048
  - 85.5/4.5/10% train/val/test split
  - 99.93% non-centrosymmetric (mutual exclusion barely applies)
- **Corn:** 80 samples × 3 instruments (m5, mp5, mp6) × 700 NIR channels
- **Tablet:** 655 samples × 2 instruments × 650 NIR channels
