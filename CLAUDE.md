# Spektron — Claude Code Instructions

## What This Project Is

Spektron is a **research paper + implementation** targeting publication at Nature Communications or JACS. The paper provides the **first formal identifiability theory for the spectral inverse problem** in vibrational spectroscopy, connecting group theory, information theory, and deep learning.

**Paper Title:** "Can One Hear the Shape of a Molecule? Group-Theoretic Identifiability and Modal Complementarity in Vibrational Spectroscopy"

**Author:** Tubhyam Karthikeyan (ICT Mumbai / InvyrAI)

## Paper Direction (CORRECTED 2026-02-11)

The paper has **two pillars**: rigorous theory + empirical ML validation.

**Pillar 1 — Theory (the novel contribution):**
- **Theorem 1 (Symmetry Quotient + Information Completeness):** Spectra are G-invariant; the inverse map is to M/G. The Information Completeness Ratio R(G,N) = (N_IR + N_Raman)/(3N-6) quantifies observable fraction. PROVABLE.
- **Theorem 2 (Modal Complementarity):** For centrosymmetric molecules, IR and Raman observe disjoint mode sets (mutual exclusion). Combined, they strictly increase observable DOF. PROVABLE. **DO NOT claim superadditivity** (violates MI submodularity). **DO NOT claim PID redundancy = 0** (disjoint features ≠ zero PID redundancy).
- **Conjecture 3 (Generic Identifiability):** Combined IR+Raman generically determines force constants up to symmetry. Supported by Jacobian rank analysis, parameter counting, and counterexample search. CONJECTURE, not theorem. **DO NOT claim Sard's theorem applies** (smoothness breaks at degeneracies).
- **Borg analogy:** Motivation ONLY, not a theorem. 1D SL ≠ d×d Hessian.

**Pillar 2 — ML Model:**
- Symmetry-aware foundation model (CNN-Transformer encoder + VIB + retrieval decoder)
- Pretrained on QM9S (130K) + ChEMBL (220K)
- Symmetry-stratified experiments validating theoretical predictions
- Calibration transfer on corn/tablet datasets

**Key reference document:** `paper/CORRECTED_PAPER_BLUEPRINT.md` (supersedes UNIFIED_PAPER_BLUEPRINT.md)

## Architecture Overview

```
Spectrum (B, 2048) 
  → WaveletEmbedding (DWT multi-scale + Conv1d patching + wavenumber PE + [CLS] + [DOMAIN])
  → MambaBackbone (4 selective SSM blocks, O(n) complexity)
  → MixtureOfExperts (4 experts, top-2 gating, optional KAN activations)
  → TransformerEncoder (2 blocks, 8 heads, global reasoning)
  → VIBHead (disentangle z_chem ∈ R^128 + z_inst ∈ R^64)
  → Heads: ReconstructionHead (pretraining) | RegressionHead (prediction) | FNOTransferHead (transfer)
```

## Project Structure

```
Spektron/
├── CLAUDE.md              ← YOU ARE HERE
├── PROJECT_STATUS.md      ← Current state, bugs, TODOs
├── IMPLEMENTATION_PLAN.md ← Detailed task list with priorities
├── requirements.txt
├── run.py                 ← Entry point (pretrain/finetune/evaluate)
├── src/
│   ├── config.py          ← All hyperparameters (dataclass-based)
│   ├── data/
│   │   └── datasets.py    ← Data loading, augmentation, wavelet preprocessing
│   ├── models/
│   │   ├── embedding.py   ← WaveletEmbedding + WavenumberPE
│   │   ├── mamba.py       ← Pure PyTorch Mamba (selective SSM)
│   │   ├── moe.py         ← MixtureOfExperts + KAN layers
│   │   ├── transformer.py ← Lightweight TransformerEncoder
│   │   ├── heads.py       ← VIB, Reconstruction, Regression, FNO heads
│   │   └── spectral_fm.py ← Full model assembly + TTT method
│   ├── losses/
│   │   └── losses.py      ← MSRP, contrastive, physics, OT, VIB, MoE losses
│   ├── training/
│   │   └── trainer.py     ← Pretrain + finetune + TTT training loops
│   └── evaluation/
│       └── metrics.py     ← R², RMSEP, RPD, bias, conformal prediction
├── data/
│   ├── raw/
│   │   ├── corn/corn.mat
│   │   └── tablet/nir_shootout_2002.mat
│   └── processed/
│       ├── corn/          ← 80 samples × 3 instruments × 700 channels
│       │   ├── m5_spectra.npy (80, 700)
│       │   ├── mp5_spectra.npy (80, 700)
│       │   ├── mp6_spectra.npy (80, 700)
│       │   ├── properties.npy (80, 4) [moisture, oil, protein, starch]
│       │   └── wavelengths.npy (700,)
│       └── tablet/        ← 655 samples × 2 instruments × 650 channels
│           ├── calibrate_1.npy (155, 650)  ← instrument 1 calibration
│           ├── calibrate_2.npy (155, 650)  ← instrument 2 calibration
│           ├── calibrate_Y.npy (155, 3)    ← [active, weight, hardness]
│           ├── test_1.npy (460, 650)
│           ├── test_2.npy (460, 650)
│           ├── test_Y.npy (460, 3)
│           ├── validate_1.npy (40, 650)
│           ├── validate_2.npy (40, 650)
│           └── validate_Y.npy (40, 3)
├── paper/
│   ├── CORRECTED_PAPER_BLUEPRINT.md  ← THE AUTHORITATIVE PAPER PLAN (v2.0)
│   ├── UNIFIED_PAPER_BLUEPRINT.md    ← SUPERSEDED (has mathematical errors)
│   ├── SYMMETRY_IDENTIFIABILITY_THEORY.md ← Group theory reference
│   ├── PID_RESOLUTION_COMPREHENSIVE.md   ← PID analysis (use cautiously)
│   ├── FORWARD_MODEL_PHYSICS.md      ← Wilson GF method reference
│   ├── BRAINSTORM_V2.md
│   └── RESEARCH_FULL_REFERENCE.md
├── checkpoints/
├── logs/
├── figures/
├── experiments/
└── notebooks/
```

## Key Technical Decisions

1. **CNN-Transformer Encoder** — 1D CNN tokenizer (8-10% accuracy gain over raw transformer) + 4-layer Transformer with 8 heads. Multi-modal variant uses cross-attention for IR+Raman fusion.
2. **VIB disentanglement** — Split latent into z_chem (chemistry, transferable) and z_inst (instrument, discardable). Reparameterization trick + KL regularization + adversarial loss.
3. **Retrieval decoder** (primary) — z_chem → nearest neighbor in database. More defensible than generative for the paper. Conformal prediction for uncertainty.
4. **Optimal Transport alignment** — Sinkhorn-based Wasserstein distance for calibration transfer. Use `POT` library.
5. **Physics-informed positional encoding** — Wavenumber (cm⁻¹) as positional information.
6. **Test-Time Training** — At inference on a new instrument, run K steps of self-supervision to adapt.

## Mathematical Guardrails (CRITICAL)

When working on theory/paper content, NEVER:
- Claim I(X; Y₁, Y₂) > I(X; Y₁) + I(X; Y₂) — this violates MI submodularity
- Claim PID redundancy = 0 from disjoint features — shared Hessian creates coupling
- Use Sard's theorem on the forward map — smoothness breaks at eigenvalue degeneracies
- Cite Borg's theorem as proof of anything — it's a 1D result, our problem is d-dimensional
- Call Conjecture 3 a "theorem" — it's unproven
- Use I_loss = log₂|G| - Σ log₂|Fix(g)| — Fix(g) can be negative

## Coding Standards

- **Python 3.10+**, PyTorch 2.x
- Type hints everywhere
- Docstrings on all public methods
- Config-driven (no magic numbers in code — everything from `SpectralFMConfig`)
- Each module independently testable
- Use `einops` for tensor reshaping where it improves readability
- Logging via Python `logging` module (not print statements)
- Reproducibility: all random seeds set via config.seed

## Important Constraints

- **Compute:** 4× RTX 5090 GPUs available for training
- **Primary dataset:** QM9S (130K molecules, IR+Raman+UV, B3LYP/def2-TZVP, on Figshare)
- **Additional datasets:** QMe14S (186K), ChEMBL IR-Raman (220K), NIST (5K), SDBS (34K)
- **Transfer test sets:** Corn (80 × 3 instruments × 700ch), Tablet (655 × 2 instruments × 650ch)
- **All spectra must be resampled to 2048 points** for uniform input
- **Key benchmark target:** Must beat LoRA-CT (R² = 0.952 on corn moisture) with ≤10 transfer samples
- **Theory must be correct:** Every theorem must have a valid proof. Conjectures must be clearly labeled.

## What Needs Doing (See CORRECTED_PAPER_BLUEPRINT.md for Full Plan)

**PHASE 1 — Data Pipeline (2 weeks):**
- [ ] Download QM9S (130K, Figshare) + ChEMBL IR-Raman (220K)
- [ ] Preprocess: resample to 2048 pts, normalize, stratify by point group
- [ ] Compute R(G,N) for each molecule (Information Completeness Ratio)
- [ ] Smoke test: forward pass + loss backward

**PHASE 2 — Theory Implementation (2 weeks):**
- [ ] R(G,N) computation from character tables (Theorem 1)
- [ ] Jacobian rank analysis for ~1000 molecules (Conjecture 3 evidence)
- [ ] Confusable set construction (Proposition A)
- [ ] Fano bound computation

**PHASE 3 — Model Training (3 weeks):**
- [ ] CNN-Transformer encoder + VIB head
- [ ] Pretraining: masked reconstruction + contrastive + denoising
- [ ] Fine-tuning: LoRA-based transfer
- [ ] Retrieval decoder + conformal prediction

**PHASE 4 — Experiments (2 weeks):**
- [ ] E1: Symmetry stratification (R(G,N) vs. accuracy)
- [ ] E2: Modal complementarity (IR vs. Raman vs. IR+Raman)
- [ ] E3: Confusable set validation (empirical error vs. Fano bound)
- [ ] E4: Jacobian rank histogram (generic identifiability evidence)
- [ ] E5: Calibration transfer (corn, tablet)
- [ ] E6: Uncertainty quantification (conformal prediction)

**PHASE 5 — Paper Writing (3 weeks):**
- [ ] Sections 1-3: Intro, Background, Theoretical Framework
- [ ] Sections 4-5: Methods, Experiments & Results
- [ ] Sections 6-7: Discussion, Conclusion
- [ ] Figures (8 main) + Tables (4 main)
- [ ] Supplementary (15 pages)
