# SpectralFM (Spektron)

**Toward Standard-Free Calibration Transfer in Vibrational Spectroscopy via Self-Supervised Learning**

A self-supervised foundation model for vibrational spectroscopy (NIR, IR, Raman) that learns to disentangle chemical content from instrument-specific artifacts. SpectralFM combines a **D-LinOSS backbone** (damped linear oscillatory state-space model) with Sinkhorn-based optimal transport domain adaptation, a variational information bottleneck, and physics-informed regularization to achieve calibration transfer across spectrometers using 10 or fewer labeled samples — where classical methods require 30-60.

> **Author:** Tubhyam Karthikeyan (ICT Mumbai / InvyrAI)
>
> **Target Journal:** Analytical Chemistry (ACS, IF 7.4)

**[Project Page](https://tubhyam.dev/projects/spektron)** | **[Research Paper](https://tubhyam.dev/research/hybrid-ssa-spectroscopy)** | **[Blog Post](https://tubhyam.dev/blog/spectral-inverse-problem)**

---

## The Problem

Calibration transfer in spectroscopy has relied on the same approaches for 30+ years: measure 10-60 transfer samples on both instruments, then apply PDS/DS/SBC correction. At $50-200 per reference analysis, this is expensive and must be repeated for every new instrument. After four decades, the field still lacks a scalable solution.

## Our Approach

SpectralFM proposes a **fifth strategy** for calibration transfer — after instrument matching, global modeling, model updating, and sensor selection (Workman & Mark, 2017): learn instrument-invariant chemical representations from a large pretraining corpus, then adapt with minimal transfer data.

- **Standard-free (TTT):** Test-time training on unlabeled spectra from the new instrument — no paired standards needed
- **Sample-efficient (LoRA):** Fine-tuning with as few as 5-10 transfer samples
- **Target:** 10 transfer samples outperforms classical methods using 50

## Architecture

```
Spectrum (B, 2048)
  -> RawSpectralEmbedding   Conv1d patching + wavenumber PE + [CLS] + [DOMAIN]
  -> DLinOSSBackbone         4 damped linear oscillatory SSM blocks, O(n) complexity
  -> MixtureOfExperts        4 experts, top-2 gating, optional KAN activations
  -> TransformerEncoder      2 blocks, 8 heads, global reasoning
  -> VIBHead                 disentangle z_chem (128d) + z_inst (64d)
  -> Heads                   Reconstruction | Regression | FNO Transfer
```

**Design rationale:**
- **D-LinOSS (O(n))** — damped linear oscillatory state-space model with 2nd-order dynamics matching molecular vibrations (damped harmonic oscillators), IMEX symplectic discretization, and proven universal approximation (Theorem 3.3, Rusch & Rus 2025)
- **Transformer (O(n^2))** — global self-attention for expressiveness where it matters
- **VIB disentanglement** — variational information bottleneck with gradient reversal splits the latent space into transferable chemistry (z_chem) and discardable instrument signature (z_inst)
- **Optimal transport** — Sinkhorn divergence aligns latent distributions across instruments
- **Physics-informed losses** — Beer-Lambert linearity, spectral smoothness, non-negativity, peak shape constraints

The model also supports a legacy **Mamba backbone** mode (selective SSM) via `config.backbone = "mamba"`, but D-LinOSS is the default and recommended backbone due to its physics-aligned oscillatory dynamics.

## Project Structure

```
Spektron/
├── run.py                              # Entry point (pretrain / finetune / evaluate / ttt)
├── src/
│   ├── config.py                       # All hyperparameters (dataclass-based)
│   ├── models/
│   │   ├── embedding.py                # WaveletEmbedding + RawSpectralEmbedding + WavenumberPE
│   │   ├── dlinoss.py                  # D-LinOSS backbone (damped oscillatory SSM)
│   │   ├── linoss/                     # LinOSS core: layers.py (LinOSSBlock, DampedLayer), scan.py
│   │   ├── mamba.py                    # Legacy Mamba backbone (selective SSM, pure PyTorch)
│   │   ├── moe.py                      # Mixture of Experts + KAN layers
│   │   ├── transformer.py              # Lightweight TransformerEncoder
│   │   ├── heads.py                    # VIB, Reconstruction, Regression, FNO heads + GradientReversal
│   │   ├── lora.py                     # LoRA injection for fine-tuning
│   │   └── spectral_fm.py              # Full model assembly (supports Mamba + D-LinOSS) + TTT
│   ├── losses/
│   │   └── losses.py                   # MSRP, contrastive, physics, OT, VIB, MoE losses
│   ├── training/
│   │   └── trainer.py                  # Pretrain + finetune + TTT training loops
│   ├── evaluation/
│   │   ├── metrics.py                  # R2, RMSEP, RPD, bias, conformal prediction
│   │   ├── baselines.py               # PDS, SBC, DS classical baselines
│   │   └── visualization.py           # Plotting utilities
│   ├── data/
│   │   ├── datasets.py                 # Data loading, augmentation, preprocessing
│   │   ├── qm9s.py                     # QM9S dataset pipeline
│   │   ├── build_pretrain_corpus.py     # Download + preprocess pretraining data
│   │   ├── corpus_downloader.py        # Multi-source corpus downloaders
│   │   └── pretraining_pipeline.py      # Pretraining dataset class
│   └── utils/
│       └── logging.py                  # Dual W&B + JSONL experiment logger
├── scripts/
│   ├── run_baselines.py                # Run classical baseline comparison
│   └── run_finetune_test.py            # Fine-tuning validation script
├── data/
│   ├── raw/                            # Original .mat files
│   └── processed/                      # Preprocessed .npy arrays
│       ├── corn/                       # 80 samples x 3 instruments x 700 channels
│       └── tablet/                     # 655 samples x 2 instruments x 650 channels
├── experiments/                        # Experiment results (JSON)
├── checkpoints/                        # Saved model weights
├── figures/                            # Generated plots
├── paper/                              # Research notes and theory documents
├── requirements.txt
├── PROJECT_STATUS.md                   # Current state and known issues
└── IMPLEMENTATION_PLAN.md              # Detailed task breakdown
```

## Datasets

### Evaluation (preprocessed, included)

| Dataset | Samples | Instruments | Channels | Properties |
|---------|---------|-------------|----------|------------|
| **Corn** | 80 | 3 (m5, mp5, mp6) | 700 | moisture, oil, protein, starch |
| **Tablet** | 655 | 2 | 650 | active ingredient, weight, hardness |

### Pretraining Corpus

| Source | Spectra | Modality |
|--------|---------|----------|
| QM9S (DFT-computed) | ~222K | IR, Raman |
| RRUFF | ~9.9K | Raman |
| OpenSpecy | ~5.4K | Raman, FTIR |
| ChEMBL IR-Raman | ~220K | IR, Raman (ready, not yet downloaded) |
| USPTO-Spectra | ~177K | Mixed (ready, not yet downloaded) |

## Installation

```bash
git clone https://github.com/ktubhyam/Spektron.git
cd Spektron
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+. Training uses bfloat16 AMP on CUDA GPUs. ~7.5GB VRAM per GPU with batch_size=8.

## Usage

```bash
# Smoke test -- verify forward/backward pass
python run.py --mode smoke_test

# Classical baselines on corn dataset
python scripts/run_baselines.py

# Pretrain on QM9S spectral corpus (D-LinOSS backbone)
python run.py --mode pretrain

# LoRA fine-tune for calibration transfer
python run.py --mode finetune --checkpoint checkpoints/pretrain_best.pt

# Standard-free transfer via test-time training
python run.py --mode ttt --checkpoint checkpoints/pretrain_best.pt
```

## Baselines

| Method | Type | Description |
|--------|------|-------------|
| PDS | Classical | Piecewise Direct Standardization (Wang, Veltkamp & Kowalski, 1991) |
| DS | Classical | Direct Standardization |
| SBC | Classical | Slope/Bias Correction |
| PLS | Classical | Partial Least Squares regression |
| SpectralFM (ours) | Foundation model | Self-supervised pretraining + LoRA transfer + TTT |

## Benchmark Targets

| Method | R2 (corn moisture) | Transfer Samples |
|--------|-------------------|-----------------|
| PDS | ~0.55 | 30 |
| DS | ~0.69 | 30 |
| LoRA-CT (literature) | 0.952 | 50 |
| **SpectralFM (target)** | **>0.96** | **10** |

## Pretraining Objectives

| Loss | Purpose |
|------|---------|
| **MSRP** | Masked Spectrum Reconstruction -- contiguous block masking, learn spectral structure |
| **Contrastive** | BYOL-style instrument-invariance between augmented views of same spectrum |
| **Denoising** | Reconstruct clean spectrum from synthetically corrupted input |
| **Physics** | Beer-Lambert linearity, smoothness, non-negativity, peak shape constraints |
| **OT Alignment** | Sinkhorn-based Wasserstein distance across instrument latent distributions |
| **VIB** | Variational Information Bottleneck -- disentangle z_chem from z_inst via gradient reversal |

## Transfer Methods

| Method | Transfer Samples | Description |
|--------|-----------------|-------------|
| **TTT** | 0 (unlabeled only) | Run K steps of MSRP self-supervision on unlabeled target spectra |
| **LoRA** | 5-10 (labeled) | Low-rank adaptation of transformer attention layers |
| **FNO** | N/A | Fourier Neural Operator head for resolution-independent spectral mapping |

## Current Status

Training is actively running on Vast.ai (2x RTX 5060 Ti 16GB) at step ~5300/50000. The D-LinOSS backbone is producing stable training with bfloat16 AMP, MSRP loss 0.08-0.13, and zero NaN/Inf errors. 20 critical bug fixes have been applied and committed. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for details.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{karthikeyan2026spectralfm,
  title={Toward Standard-Free Calibration Transfer in Vibrational Spectroscopy via Self-Supervised Learning},
  author={Karthikeyan, Tubhyam},
  journal={Analytical Chemistry},
  year={2026}
}
```

---

*Under active development. Targeting publication in Analytical Chemistry (ACS).*
