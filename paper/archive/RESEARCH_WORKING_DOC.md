# Spektron: Self-Supervised Spectral Foundation Model for Few-Shot Calibration Transfer
## Running Working Document — Last Updated: Feb 10, 2026 (Phase 3: Implementation)

---

## PROJECT OVERVIEW

**Paper Title:** "Self-Supervised Spectral Foundation Model for Few-Shot Calibration Transfer Across Instruments and Modalities"
**Author:** Tubhyam Karthikeyan (ICT Mumbai, InvyrAI)
**Target Journal:** Analytical Chemistry (ACS) — IF 7.4
**Timeline:** 4 weeks (full throttle)
**Compute:** Colab Pro+ A100

---

## GAP VERIFIED (Feb 10, 2026)

Spectroscopy Online Jan/Feb 2026 (published today) calls out "calibration transferability" as unsolved and "foundation-scale AI" as the future. No one has built a self-supervised foundation model for vibrational spectroscopy calibration transfer.

### Competitors
- DreaMS/PRISM/DSCF: Mass spec only
- LoRA-CT (AC 2025): No pretraining, R²=0.952 on methanol
- RamanMAE/SMAE: Classification only, no CT
- BDSER-InceptionNet: CNN, no self-supervised pretraining
- ACT/Vib2Mol/MACE4IR: Wrong downstream task
- Federated NIR: Different paradigm
- di-PLS/mdi-PLS: Classical, no deep learning

### Our Claims
1. FIRST self-supervised FM for vibrational spectroscopy
2. FIRST FM pretraining for few-shot calibration transfer
3. Physics-informed design (wavenumber PE, contiguous masking)
4. Cross-modality pretraining (IR→NIR transfer)
5. Hybrid: FM pretraining + LoRA fine-tuning

---

## ARCHITECTURE

### Pretraining: SpectralBERT
- Input: 1D spectrum → resample to 2048 points → patch (P=64, size=32, stride=16)
- Wavenumber positional encoding (physics-aware)
- 6 transformer layers, dim=256, 8 heads, FFN=1024, ~8-10M params
- MSRP: mask 15-25% contiguous patches, predict from context, MSE loss
- Augmentation: noise, baseline drift, wavelength shift, intensity scaling

### Fine-tuning: LoRA Transfer
- LoRA rank=4-8, on Q/K/V matrices, ~50K trainable params
- Transfer head: MLP for spectral correction
- 10-50 paired samples, AdamW lr=1e-4, cosine schedule

---

## DATASETS

### Pretraining (~400K spectra)
- ChEMBL IR-Raman: 220K (Figshare)
- USPTO-Spectra: 177K (Zenodo)
- NIST IR: 5.2K (JCAMP-DX)
- RRUFF Raman: 8.6K

### Benchmarks
- Corn: 80 samples, 3 NIR instruments, 4 components
- Tablet: 655 samples, 2 NIR instruments, 3 components
- Raman API: 3510 samples, multiple instruments

---

## EXPERIMENTS
- E1: Pretraining ablation
- E2: Masking strategy (contiguous vs random vs peak-aware)
- E3: Sample efficiency curve (N=5,10,20,30,50,100)
- E4: Full baseline comparison (must beat LoRA-CT R²>0.952)
- E5: Cross-modality (IR pretrain → NIR fine-tune)
- E6: Interpretability (attention, t-SNE, GradCAM)
- Ablations: corpus size, model size, mask ratio, PE type, LoRA rank

---

## BASELINES
Classical: PDS, SBC, DS, CCA, di-PLS, mdi-PLS
DL: CNN, Transformer (no pretrain), LoRA-CT, BDSER-InceptionNet, Full FT

---

## TOOLS
Core: PyTorch, Transformers, scikit-learn, scipy
Spectro: jcamp, rampy, lmfit, diPLSlib
Data: numpy, pandas, h5py, rdkit
Viz: matplotlib, seaborn, plotly
Track: wandb
Utils: einops, tqdm, torch-lr-finder

---

## TIMELINE
Week 1 (Feb 10-16): Data download + preprocessing + architecture implementation
Week 2 (Feb 17-23): Pretraining (24-48h) + baseline implementation
Week 3 (Feb 24-Mar 2): All experiments + ablations
Week 4 (Mar 3-9): Writing + figures + submission

---

## ENHANCED ARCHITECTURE (Feb 10 update)
- Dual pretraining: MSRP + Contrastive (BYOL-style instrument-invariance)
- Physics-informed loss: MSE + derivative matching + peak preservation + SAM
- Domain tokens: [IR], [RAMAN], [NIR] prepended to sequence
- Classical-neural hybrid: PDS residual correction (model learns what PDS misses)
- Uncertainty quantification: MC dropout at inference
- Test-time augmentation for error reduction
- "Fifth strategy" framing (beyond Workman & Mark 2017's four strategies)

---

## ADVANCED TECHNIQUES (Deep Research Phase)
See: /home/claude/ADVANCED_TECHNIQUES_SYNTHESIS.md

**10 techniques identified, 3 tiers:**
- Tier 1 (Must): Mamba backbone, OT alignment, Physics-informed loss, Wavelet embedding
- Tier 2 (Should): Test-Time Training, MoE, FNO transfer head
- Tier 3 (Stretch): KAN, VIB disentanglement, MAML, Conformal prediction

**Revised architecture: Hybrid Mamba-Transformer with wavelet input, MoE layer, disentangled latent space, FNO transfer head, TTT for zero-shot**

---

## KEY CITATIONS (50+ papers)
Must-cite: DreaMS, LoRA-CT, di-PLS, SMAE, RamanMAE, ChEMBL dataset, USPTO dataset, Spectroscopy Online reviews.
**NEW:** Workman & Mark, Spectroscopy 32(10), 2017 — comprehensive CT review (PDS/DS/OSC/FIR/MLPCA math)
**ADVANCED:** Mamba (Gu & Dao 2023), FNO (Li et al. 2021), KAN (Liu et al. 2024), PINNs (Raissi et al. 2019), TTT (Sun et al. 2020), DeepJDOT (Courty et al. 2017), PhysioWave (NeurIPS 2025), MambaTS (ICLR 2025), SpectraKAN (Feb 2025)
