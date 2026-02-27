# Spektron

**Oscillatory State Space Models for Vibrational Spectroscopy: Benchmarking, Cross-Spectral Prediction, and Interpretability**

An honest empirical study evaluating whether oscillatory SSMs (D-LinOSS) bring unique advantages to vibrational spectroscopy. Includes the first SSM evaluation for IR/Raman spectroscopy, a novel cross-spectral prediction task (IR to Raman and vice versa), and transfer function interpretability analysis unique to SSM architectures.

> **Author:** Tubhyam Karthikeyan (ICT Mumbai / InvyrAI)
>
> **Target Journal:** Digital Discovery (RSC)

**[Project Page](https://tubhyam.dev/projects/spektron)** | **[Blog](https://tubhyam.dev/blog/spectral-inverse-problem)**

---

## Key Claims

1. **First SSM evaluation for vibrational spectroscopy** — zero prior work applies SSMs to IR/Raman
2. **Novel cross-spectral prediction task** — predicting Raman from IR (and vice versa). No prior work exists
3. **Function-space argument** — D-LinOSS dynamics (damped sinusoids) are a natural basis for vibrational spectra
4. **Transfer function interpretability** — H(z) analysis reveals what spectral filters each D-LinOSS layer has learned. Unique to SSMs

## Experiments

| Experiment | Script | Description |
|-----------|--------|-------------|
| **E1: Architecture Benchmark** | `experiments/e1_benchmark.py` | D-LinOSS vs Mamba vs Transformer vs CNN vs S4D, param-matched at ~2M backbone params, masked spectral reconstruction on QM9S |
| **E2: Cross-Spectral Prediction** | `experiments/e2_cross_spectral.py` | Predict Raman from IR and vice versa. Includes identity-copy baseline. 5 metrics: MSE, cosine similarity, peak recall, intensity correlation, SID |
| **E3: Transfer Function Analysis** | `experiments/e3_transfer_function.py` | H(z) filter bank visualization, pole-zero plots, per-layer frequency coverage, trained vs random comparison |
| **E4: Calibration Transfer** | `experiments/e4_calibration_transfer.py` | Fine-tune on Corn (80x3x700) + Tablet (655x2x650), compare against PDS, SBC, DS, CCA, di-PLS baselines |

Run all experiments:
```bash
python experiments/run_all.py --h5-path data/raw/qm9s/qm9s_processed.h5
```

## Architecture

```
Spectrum (B, 2048)
  -> RawSpectralEmbedding    Conv1d + wavenumber PE + [CLS] + [DOMAIN]
  -> Backbone                D-LinOSS (bidirectional) | Mamba | Transformer | CNN1D | S4D
  -> MixtureOfExperts        4 experts, top-2 gating
  -> TransformerEncoder      2 blocks (skipped when backbone=Transformer)
  -> VIBHead                 disentangle z_chem (128d) + z_inst (64d)
  -> Heads                   Reconstruction | CrossSpectral | Regression | FNO Transfer
```

### Param-Matched Backbones (E1 Benchmark)

| Architecture | Backbone Params | Key Properties |
|-------------|:-:|---|
| D-LinOSS | 2.1M | 2nd-order oscillatory SSM, O(n), bidirectional (4+4 layers) |
| Mamba | 1.8M | 1st-order selective SSM, O(n), input-dependent |
| Transformer | 2.1M | Self-attention, O(n^2), 4 layers |
| 1D CNN | 2.1M | Local convolutions, FIR filters, 2 layers |
| S4D (ablation) | 1.1M | 1st-order diagonal SSM, no oscillatory structure |

## Project Structure

```
Spektron/
├── CLAUDE.md                        # Claude Code instructions
├── README.md
├── requirements.txt
├── run.py                           # Entry point (pretrain/finetune/evaluate/ttt)
├── src/
│   ├── config.py                    # All hyperparameters (dataclass-based)
│   ├── models/
│   │   ├── spektron.py              # Full model assembly (Spektron + SpektronForPretraining)
│   │   ├── dlinoss.py               # D-LinOSS bidirectional backbone
│   │   ├── backbones.py             # CNN1D + S4D backbone implementations
│   │   ├── mamba.py                 # Mamba backbone (selective SSM)
│   │   ├── transformer.py           # Transformer backbone
│   │   ├── embedding.py             # WaveletEmbedding + RawSpectralEmbedding + WavenumberPE
│   │   ├── heads.py                 # VIB, Reconstruction, CrossSpectral, Regression, FNO heads
│   │   ├── moe.py                   # Mixture of Experts + KAN layers
│   │   ├── lora.py                  # LoRA injection for fine-tuning
│   │   └── linoss/                  # Vendored D-LinOSS core (DO NOT MODIFY)
│   │       ├── layers.py            # LinOSSBlock, DampedLayer, GLU
│   │       └── scan.py              # Parallel associative scan
│   ├── losses/
│   │   └── losses.py                # MSRP, contrastive, physics, OT, VIB losses
│   ├── training/
│   │   └── trainer.py               # Pretrain + finetune + TTT training loops
│   ├── evaluation/
│   │   ├── metrics.py               # R2, RMSEP, RPD, bias
│   │   ├── baselines.py             # PDS, SBC, DS, CCA, di-PLS, PLS
│   │   ├── cross_spectral_metrics.py # MSE, cosine, peak recall, SID
│   │   └── visualization.py         # Plotting utilities
│   ├── data/
│   │   ├── qm9s.py                  # QM9S HDF5 loading + preprocessing
│   │   ├── cross_spectral.py        # Paired IR/Raman dataset for E2
│   │   └── datasets.py              # Corn/Tablet datasets + augmentation
│   ├── analysis/
│   │   └── transfer_function.py     # H(z) computation + visualization for E3
│   └── utils/
│       └── logging.py               # Experiment logger
├── experiments/
│   ├── e1_benchmark.py              # E1: Architecture benchmark
│   ├── e2_cross_spectral.py         # E2: Cross-spectral prediction
│   ├── e3_transfer_function.py      # E3: Transfer function analysis
│   ├── e4_calibration_transfer.py   # E4: Calibration transfer
│   ├── pretrain_qm9s.py            # Pretraining entry point
│   ├── run_all.py                   # Master experiment runner
│   ├── results/                     # JSON results
│   └── archive/                     # Old experiment scripts
├── data/
│   ├── raw/qm9s/                    # QM9S HDF5 (130K molecules)
│   ├── processed/corn/              # 80 x 3 instruments x 700ch
│   └── processed/tablet/            # 655 x 2 instruments x 650ch
├── figures/                         # Generated plots (E1-E4)
├── checkpoints/                     # Saved model weights
├── logs/
├── paper/
│   ├── sections/                    # LaTeX/Markdown sections
│   └── archive/                     # Old planning docs
├── scripts/                         # Utility scripts
└── tests/
    └── smoke_test.py
```

## Dataset

**QM9S** — 129,817 molecules with DFT-computed spectra (B3LYP/def2-TZVP):

| Property | Value |
|----------|-------|
| Molecules | 129,817 |
| IR spectra | 103,991 |
| Raman spectra | 129,817 |
| Spectral range | 500-4000 cm^-1 (resampled to 2048 points) |
| Split | 85.5% train / 4.5% val / 10% test |
| Non-centrosymmetric | 99.93% (mutual exclusion barely applies) |

**Calibration transfer datasets:**
- **Corn:** 80 samples x 3 NIR instruments (m5, mp5, mp6) x 700 channels
- **Tablet:** 655 samples x 2 instruments x 650 channels

## Installation

```bash
git clone https://github.com/ktubhyam/Spektron.git
cd Spektron
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+. Training uses bfloat16 AMP on CUDA GPUs. ~7.5GB VRAM per GPU with batch_size=8.

## Usage

```bash
# Smoke test — verify all components
python run.py --mode smoke_test

# Pretrain on QM9S (D-LinOSS backbone)
python experiments/pretrain_qm9s.py --h5-path data/raw/qm9s/qm9s_processed.h5

# Run architecture benchmark (E1)
python experiments/e1_benchmark.py --h5-path data/raw/qm9s/qm9s_processed.h5

# Run cross-spectral prediction (E2)
python experiments/e2_cross_spectral.py --h5-path data/raw/qm9s/qm9s_processed.h5

# Run all experiments
python experiments/run_all.py --h5-path data/raw/qm9s/qm9s_processed.h5

# Quick test (CPU, fewer steps)
python experiments/run_all.py --h5-path ... --quick
```

## Compute

Training runs on 2x RTX 5060 Ti 16GB via Vast.ai:
- batch_size=16 (8/GPU), grad_accumulation=4, effective batch 64
- AMP bfloat16, D-LinOSS forced to float32 (prevents overflow in GLU)
- ~39 samples/sec, ~23 hours for 50K steps
- Total compute budget: ~106 GPU-hours

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{karthikeyan2026spektron,
  title={Oscillatory State Space Models for Vibrational Spectroscopy:
         Benchmarking, Cross-Spectral Prediction, and Interpretability},
  author={Karthikeyan, Tubhyam},
  journal={Digital Discovery},
  year={2026}
}
```
