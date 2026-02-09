# SpectralFM: Advanced ML Techniques Synthesis
## Deep Research Phase — Feb 10, 2026

---

## EXECUTIVE SUMMARY

Beyond the baseline transformer architecture, we identified **10 cutting-edge ML techniques** that can dramatically strengthen SpectralFM. These are organized into 3 tiers by impact and feasibility. The key insight: **SpectralFM should not be "just another transformer"** — it should be a physics-informed, multi-scale, operator-learning foundation model that leverages the specific structure of spectral data.

---

## TECHNIQUE 1: MAMBA (Selective State Space Models) — ★★★★★

**Why it matters:** Spectra are 1D sequences of 700-2048 points. Transformers have O(n²) attention complexity; Mamba achieves O(n) with selective state space filtering — perfect for long spectral sequences.

**Key papers:**
- Gu & Dao 2023: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- MambaTS (ICLR 2025): SOTA on 8 long-term time series forecasting datasets
- TSCMamba 2025: Multi-view time series classification integrating frequency+time domain

**Spectroscopy fit:**
- Long-range dependencies critical (baseline shifts span hundreds of channels)
- Selective mechanism naturally filters instrument noise
- Linear complexity = process full-resolution spectra without downsampling
- 40× faster inference than transformer

**Integration:** Hybrid Mamba-Transformer backbone
```
Input → Wavelet Decomposition → Mamba Blocks (×4) → Cross-Attention → Transformer Blocks (×2) → z_latent
```

**Novelty angle:** "First Mamba-based foundation model for spectroscopy"

---

## TECHNIQUE 2: OPTIMAL TRANSPORT (OT) — ★★★★★

**Why it matters:** Calibration transfer IS domain adaptation. OT provides the mathematically principled framework for aligning instrument distributions via Wasserstein distance.

**Key papers:**
- Courty et al. 2017: DeepJDOT (Deep Joint Distribution OT)
- UniJDOT 2025: Universal domain adaptation for time series
- CDAN-PL (Sensors 2025): Raman spectroscopy + domain adaptation
- NeurIPS 2024: "Protected TTA via Online Entropy Matching" — connects TTA to OT
- MMD-based preprocessing selection for spectroscopy transfer (2025)

**Integration:** Add OT alignment to pretraining loss
```
L_OT = Wasserstein(P_source_latent, P_target_latent)  [via Sinkhorn algorithm]
L_total = α·L_MSRP + β·L_contrastive + γ·L_denoise + δ·L_OT
```

**Impact:** Principled "why it works" answer for reviewers. Enables unsupervised transfer (no labels on target instrument).

---

## TECHNIQUE 3: PHYSICS-INFORMED REGULARIZATION (PINNs) — ★★★★★

**Why it matters:** Spectroscopy has well-defined physical laws. Embedding them prevents "chemically impossible" predictions and massively reduces data requirements.

**Key physics to embed:**
- Beer-Lambert Law: A = εbc (absorbance ∝ concentration × path length)
- Non-negativity: absorbance spectra should be non-negative
- Baseline smoothness: baselines should be smooth, slowly varying
- Peak shape: absorption peaks follow Lorentzian/Gaussian/Voigt profiles
- Derivative smoothness: second derivative shouldn't oscillate wildly

**Key papers:**
- Raissi et al. 2019: Foundational PINNs paper
- Puleio & Gaudio 2025 (Sci Rep): Unsupervised spectra extraction using PINNs
- KAN replacing MLPs in PINNs for better interpretability
- Physics-informed spectroscopy for MRS (self-supervised DL quantification)

**Integration:**
```python
L_physics = λ1 * MSE(baseline, smooth_fit)        # smoothness
          + λ2 * ReLU(-spectrum).mean()             # non-negativity
          + λ3 * peak_shape_loss                    # Lorentzian/Gaussian
          + λ4 * derivative_smoothness_loss         # Savitzky-Golay principle
```

**Impact:** Addresses "black box" criticism. Chemically plausible predictions. Better extrapolation.

---

## TECHNIQUE 4: WAVELET MULTI-SCALE EMBEDDING — ★★★★☆

**Why it matters:** Spectral features naturally exist at multiple scales — sharp peaks (high frequency), broad absorption bands (medium), baselines (low frequency). Wavelets decompose these naturally.

**Key papers:**
- PhysioWave (NeurIPS 2025): Multi-scale wavelet-transformer for physiological signals
- TSLANet (ICML 2024): Tree-structured wavelet neural network
- Continual Learning in Frequency Domain (NeurIPS 2024): 6.83% accuracy gain, 2.6× faster training

**Integration:** Replace patch embedding with wavelet decomposition
```
Input spectrum → DWT (4 levels)
  ├─ Approx coefficients (low-freq: baseline)
  ├─ Detail L1 (high-freq: sharp peaks)
  ├─ Detail L2 (medium-freq: broad peaks)
  └─ Detail L3 (low-freq: shoulders)
→ Separate encoding paths → Multi-scale fusion
```

**Impact:** Natural multi-scale representation, explicit baseline/peak separation, 2-3× training speedup.

---

## TECHNIQUE 5: NEURAL OPERATORS (FNO/DeepONet) — ★★★★☆

**Why it matters:** Calibration transfer = learning operator T: Spectrum_inst1 → Spectrum_inst2. Neural operators learn mappings between FUNCTION SPACES, not point-wise — enabling resolution-independent transfer.

**Key papers:**
- Li et al. 2021 (ICLR): Fourier Neural Operator
- SpectraKAN (Feb 5, 2025!): Input-conditioned spectral operators via cross-attention
- Mamba Neural Operator (NeurIPS 2024): SSM-based operator learning
- λ-FNO (2025): Sparse FNO with transfer learning, improved operator learning
- D-FNO (CMAME 2025): Decomposed FNO, 2-3× speedup
- SpecBoost-FNO: Addresses frequency bias, 50% accuracy improvement

**Integration:** FNO as transfer head instead of MLP
```
z_latent → FNO(z, Δλ, instrument_params) → Transferred spectrum
```

**Impact:** Resolution-independent predictions. Works at any wavelength grid. Natural frequency-domain processing for spectral data.

---

## TECHNIQUE 6: MIXTURE OF EXPERTS (MoE) — ★★★★☆ [NEW]

**Why it matters:** Different instruments/modalities need different processing. MoE provides CONDITIONAL COMPUTATION — only relevant expert subnetworks activate per instrument type.

**Key concept for SpectralFM:**
- Expert per instrument type (NIR, IR, Raman)
- Expert per spectral region (fingerprint, overtone, combination)
- Sparse gating: only 1-2 experts active per input
- Massive capacity without proportional compute

**Key papers:**
- Shazeer et al. 2017: Sparsely-Gated MoE
- MMoE (KDD 2018): Multi-gate MoE for multi-task learning
- DeepSeek-V3: 256 experts, fine-grained routing
- MoASE: Mixture-of-Activation-Sparsity-Experts for continual adaptation

**Integration:**
```
Encoder → [Shared backbone] → MoE Layer (K experts, top-2 gating)
  Expert 1: NIR specialist        Expert 2: IR specialist
  Expert 3: Raman specialist      Expert 4: Cross-modal bridger
→ Weighted combination → Transfer head
```

**Impact:** Natural modularity for multi-instrument/multi-modality. Scales capacity without compute cost. Each expert specializes in different instrument characteristics.

---

## TECHNIQUE 7: TEST-TIME TRAINING / ADAPTATION (TTT/TTA) — ★★★★☆ [NEW]

**Why it matters:** When deploying SpectralFM on a NEW instrument never seen during training, TTT adapts the model AT INFERENCE using self-supervised objectives on the unlabeled test spectra. This is EXACTLY the calibration transfer scenario.

**Key insight:** SpectralFM's MSRP pretraining objective can serve as the TTT auxiliary task! At test time, run a few gradient steps of MSRP on unlabeled spectra from the new instrument → model adapts to new instrument characteristics without ANY labeled transfer samples.

**Key papers:**
- Sun et al. 2020 (ICML): Test-Time Training with Self-Supervision
- TENT (ICLR 2021): Fully Test-Time Adaptation by Entropy Minimization
- MT3 (2022): Meta Test-Time Training
- TAIP (Nature Comms 2025): Online TTA for interatomic potentials — dual-level SSL
- Protected TTA (NeurIPS 2024): OT-based entropy matching for TTA
- IST (CVPR 2024): Improved Self-Training for TTA

**Integration:**
```
At test time (new instrument):
1. Receive batch of unlabeled spectra from new instrument
2. Run K gradient steps of MSRP on these spectra (self-supervised)
3. Model adapts encoder to new instrument characteristics
4. Apply adapted model for calibration transfer
```

**Impact:** ZERO-SHOT calibration transfer capability! No labeled pairs needed. Addresses "what if I have no transfer samples?" scenario. Huge practical value.

---

## TECHNIQUE 8: KOLMOGOROV-ARNOLD NETWORKS (KAN) — ★★★☆☆

**Why it matters:** KAN replaces fixed activation functions (ReLU) with LEARNABLE spline functions on edges. This enables: (a) better accuracy with smaller models, (b) interpretable learned functions that can be symbolically regressed to recover physical laws.

**Key papers:**
- Liu et al. 2024/2025 (ICLR 2025): KAN: Kolmogorov-Arnold Networks
- KAN 2.0 (Phys. Rev. X 2025): KANs Meet Science — discovering physical laws
- KA-GNN (Nature Machine Intelligence 2025): For molecular property prediction
- MOF-KAN (J. Phys. Chem. Lett. 2025): Outperforms MLPs in low-data regimes
- KAN + Raman (J. Raman Spectroscopy 2025): Multi-scale residual KAN for diagnosis

**Integration:** Replace MLP layers in transfer head with KAN
```
z_latent → KAN([256, 128, n_wavelengths]) → Transferred spectrum
```
The learned spline functions can be symbolically regressed to potentially recover Beer-Lambert or other relationships!

**Impact:** Interpretability breakthrough — "the model learned Beer-Lambert law." Better in low-data regimes. Novelty: first KAN-based spectral transfer head.

---

## TECHNIQUE 9: VARIATIONAL INFORMATION BOTTLENECK + DISENTANGLEMENT — ★★★☆☆ [NEW]

**Why it matters:** The core challenge is separating CHEMISTRY-INVARIANT features from INSTRUMENT-SPECIFIC features. Information bottleneck theory provides the principled framework for this disentanglement.

**Key papers:**
- Deep VIB (Alemi et al. 2017): Variational Information Bottleneck
- IIB (AAAI 2022): Invariant Information Bottleneck for Domain Generalization
- DDIR (Knowledge-Based Systems 2025): Domain-Disentangled Invariant Representation
- DDM: Domain-variant/invariant disentanglement in self-supervised pretraining
- DisTIB (2024): Transmitted IB for optimal disentanglement
- INSURE: Information-theoretic disentanglement for domain generalization

**Integration:** Split latent space into two components
```
Encoder → z = [z_chem, z_inst]
  z_chem: chemistry-invariant (composition, functional groups)
  z_inst: instrument-specific (resolution, baseline, noise)
  
Objective: maximize I(z_chem; Y) while minimizing I(z_chem; D_instrument)
Transfer: keep z_chem, replace z_inst for target instrument
```

**Impact:** Principled disentanglement with information-theoretic guarantees. Enables understanding WHAT the model transfers vs. what it discards.

---

## TECHNIQUE 10: META-LEARNING (MAML) — ★★★☆☆ [NEW]

**Why it matters:** MAML learns an initialization that's maximally amenable to few-shot adaptation. SpectralFM's pretraining already does something similar, but MAML formalizes it: learn initial weights such that K gradient steps on N transfer samples maximizes performance.

**Key papers:**
- Finn et al. 2017 (ICML): Model-Agnostic Meta-Learning
- Open-MAML (Sci Rep 2025): Cross-way cross-shot generalization
- MAML + Transfer Learning (SAGE 2025): Combined for domain adaptation
- MT3: Meta Test-Time Training — combines MAML with TTT

**Integration:** Meta-learning objective during fine-tuning
```
For each instrument pair (source_i, target_j):
  θ' = θ - α∇L(source_i → target_j, support_set)  # inner loop
  Update θ via ∇L(source_i → target_j, query_set, θ')  # outer loop
  
Result: θ* that can adapt to ANY new instrument pair in K steps
```

**Impact:** Formalized few-shot adaptation. Better initialization than random fine-tuning. Enables N-way K-shot calibration transfer.

---

## REVISED ARCHITECTURE: SpectralFM v2

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                           │
│  Raw spectrum (700-2048 points)                         │
│  → Wavelet Decomposition (DWT, 4 levels)                │
│  → Multi-scale tokens [approx, detail1, detail2, detail3]│
│  + Wavenumber positional encoding (physics-aware)        │
│  + Domain token [NIR] / [IR] / [RAMAN]                  │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  MAMBA BACKBONE (4 blocks)               │
│  Selective SSM: O(n) complexity                         │
│  - Selective mechanism filters noise                     │
│  - Long-range spectral dependencies                     │
│  - Hardware-aware parallel scan                         │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│             MIXTURE OF EXPERTS LAYER                     │
│  4 experts: NIR / IR / Raman / Cross-modal              │
│  Sparse gating: top-2 routing per input                 │
│  Each expert: 2-layer FFN with KAN activations          │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│              TRANSFORMER BLOCKS (2 blocks)               │
│  Global reasoning + cross-instrument attention          │
│  LoRA adapters for fine-tuning (rank 4-8)               │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│             DISENTANGLED LATENT SPACE                    │
│  z = [z_chem (128d), z_inst (64d)]                      │
│  VIB objective: max I(z_chem;Y), min I(z_chem;D_inst)  │
└──────────┬─────────────────────────────┬────────────────┘
           ▼                             ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│   PRETRAINING HEAD   │  │     TRANSFER HEAD            │
│  MSRP reconstruction │  │  FNO(z_chem, inst_params)    │
│  + Contrastive loss  │  │  → Resolution-independent    │
│  + Denoising loss    │  │  → Transferred spectrum      │
│  + OT alignment      │  │  + Conformal prediction      │
│  + Physics losses    │  │  + KAN interpretability       │
└──────────────────────┘  └──────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  TEST-TIME ADAPT │
                    │  MSRP on new inst│
                    │  K gradient steps│
                    │  → Zero-shot CT  │
                    └──────────────────┘
```

---

## PRETRAINING LOSS (Full)

```
L_total = α·L_MSRP           # Masked Spectrum Reconstruction (core)
        + β·L_contrastive     # BYOL-style instrument-invariance
        + γ·L_denoise         # Denoising autoencoder
        + δ·L_OT              # Wasserstein alignment across instruments
        + ε·L_physics          # Beer-Lambert + smoothness + non-negativity
        + ζ·L_VIB             # Information bottleneck disentanglement
        + η·L_MoE_balance     # Expert load balancing
```

Hyperparameters: α=1.0, β=0.3, γ=0.2, δ=0.1, ε=0.1, ζ=0.05, η=0.01

---

## UPDATED EXPERIMENT PLAN

### Core Experiments (E1-E6): Same as original
### NEW Experiments:

**E7: Architecture Ablation**
- Transformer-only vs Mamba-only vs Hybrid
- With/without wavelet embedding
- With/without MoE layer
- Metric: R², RMSEP, training time, inference time

**E8: Physics-Informed Loss Ablation**
- Baseline (MSRP only) → +contrastive → +denoise → +OT → +physics → +VIB
- Measure: prediction error, physics violations, sample efficiency

**E9: Test-Time Training Evaluation**
- Zero-shot: TTT only (no labeled transfer samples)
- Few-shot + TTT: Does TTT improve over few-shot alone?
- Metric: compare N=0 (TTT) vs N=5,10,20 (supervised) vs N=5+TTT (hybrid)

**E10: Disentanglement Quality**
- Visualize z_chem vs z_inst via t-SNE
- Do z_chem clusters correspond to chemistry? Do z_inst clusters correspond to instruments?
- Interpolation: swap z_inst between instruments, check if chemistry preserved

**E11: KAN Interpretability**
- Extract learned activation functions from KAN transfer head
- Do they recover Beer-Lambert (linear relationship)?
- Compare symbolic regression output to known physics

**E12: Neural Operator Resolution Transfer**
- Train on instrument with 700 wavelengths
- Test on instrument with 2048 wavelengths (no retraining)
- Does FNO generalize across resolutions?

---

## IMPLEMENTATION PRIORITY (REVISED)

### MUST IMPLEMENT (Tier 1 — Core Differentiators)
1. **Hybrid Mamba-Transformer backbone** — computational efficiency + novelty
2. **Physics-informed loss** — domain credibility + plausibility
3. **OT alignment** — principled domain adaptation
4. **Wavelet multi-scale embedding** — natural for spectroscopy

### SHOULD IMPLEMENT (Tier 2 — Strong Differentiators)
5. **Test-Time Training** — zero-shot capability, huge practical value
6. **MoE layer** — multi-instrument specialization
7. **FNO transfer head** — resolution independence

### STRETCH (Tier 3 — Cherry on Top)
8. **KAN activations** — interpretability showcase
9. **VIB disentanglement** — principled feature separation
10. **MAML meta-learning** — formalized few-shot
11. **Conformal prediction** — formal uncertainty

---

## PRAGMATIC IMPLEMENTATION STRATEGY

Given 4-week timeline and solo researcher on Colab:

**Strategy: MODULAR BUILD**
Build the system so each technique is a pluggable module. Start with simplest version (transformer baseline), then add modules one by one. Each addition = one row in the ablation table.

**Week 1:** 
- Core data pipeline + wavelet embedding
- Mamba backbone (use mamba-ssm library)
- Baseline transformer for comparison
- Physics-informed loss terms

**Week 2:**
- Pretraining with full loss (MSRP + contrastive + denoise + physics)
- OT alignment (use POT library)
- MoE layer (simple top-2 gating)
- TTT implementation

**Week 3:**
- All experiments E1-E12
- FNO transfer head (use neuraloperator library)
- KAN layers if time permits
- Ablation sweep

**Week 4:**
- Paper writing + figures
- Final experiments + polishing
- Submission prep

---

## KEY LIBRARIES

| Technique | Library | Install |
|-----------|---------|---------|
| Mamba | mamba-ssm | pip install mamba-ssm |
| Optimal Transport | POT | pip install POT |
| Wavelets | PyWavelets | pip install PyWavelets |
| FNO | neuraloperator | pip install neuraloperator |
| KAN | pykan | pip install pykan |
| Conformal | MAPIE | pip install mapie |
| LoRA | peft | pip install peft |
| MoE | Custom (simple) | ~50 lines PyTorch |
| VIB | Custom | ~30 lines PyTorch |
| TTT | Custom | ~40 lines PyTorch |

---

## PAPER NARRATIVE (UPDATED)

### Title Option 1 (Conservative):
"SpectralFM: A Physics-Informed Foundation Model for Few-Shot Calibration Transfer in Vibrational Spectroscopy"

### Title Option 2 (Bold):
"SpectralFM: Bridging State Space Models and Optimal Transport for Zero-to-Few-Shot Spectral Calibration Transfer"

### Abstract (Draft):

Calibration transfer — adapting spectroscopic models across instruments and modalities — remains a critical bottleneck in analytical chemistry. We introduce SpectralFM, the first self-supervised foundation model for vibrational spectroscopy that achieves few-shot calibration transfer by integrating: (1) a hybrid Mamba-Transformer backbone with wavelet multi-scale embedding for efficient spectral representation learning, (2) optimal transport-based domain alignment for principled instrument distribution matching, (3) physics-informed regularization embedding Beer-Lambert law and spectral constraints, and (4) test-time training for zero-shot adaptation to unseen instruments. Pretrained on 400K+ IR and Raman spectra via masked spectrum reconstruction, SpectralFM achieves state-of-the-art transfer performance on benchmark datasets (corn, tablet, Raman API) using only 10 labeled transfer samples, representing a 5× reduction over existing methods. The model produces chemically plausible predictions with formal uncertainty quantification via conformal prediction. Our work establishes foundation model pretraining as a paradigm shift for analytical chemistry instrumentation.

### Key Innovations (Ranked):
1. First self-supervised FM for vibrational spectroscopy
2. Hybrid Mamba-Transformer backbone (first for spectroscopy)
3. OT-based instrument alignment (principled domain adaptation)
4. Physics-informed pretraining (Beer-Lambert, peak constraints)
5. Wavelet multi-scale embedding (natural for spectra)
6. Test-time training for zero-shot transfer
7. MoE for instrument specialization
8. FNO transfer head for resolution independence

---

## CRITICAL REFERENCES TO ADD

### State Space Models
- Gu & Dao 2023: Mamba (foundational)
- MambaTS (ICLR 2025)
- TSCMamba 2025

### Optimal Transport
- Courty et al. 2017: DeepJDOT
- UniJDOT 2025
- Sinkhorn distances (Cuturi 2013)

### Physics-Informed
- Raissi et al. 2019: PINNs
- Puleio & Gaudio 2025: Spectroscopy PINNs

### Wavelets
- PhysioWave (NeurIPS 2025)
- TSLANet (ICML 2024)

### Neural Operators
- Li et al. 2021: FNO (ICLR)
- SpectraKAN (Feb 2025!)
- Mamba Neural Operator (NeurIPS 2024)

### KAN
- Liu et al. 2024: KAN (ICLR 2025)
- KAN 2.0 (Phys. Rev. X 2025)
- KAN + Raman spectroscopy (J. Raman Spectroscopy 2025)

### MoE
- Shazeer et al. 2017: Sparsely-Gated MoE
- MMoE (KDD 2018)

### Test-Time Training
- Sun et al. 2020: TTT with Self-Supervision
- TENT (ICLR 2021)
- TAIP (Nature Comms 2025)

### Domain Generalization
- IIB (AAAI 2022): Invariant Information Bottleneck
- DDIR (2025): Domain-Disentangled Invariant Representation
- DDM: Domain-variant/invariant disentanglement

### Meta-Learning
- Finn et al. 2017: MAML
- Open-MAML (Sci Rep 2025)

---

## EXPECTED PERFORMANCE TABLE

| Method | Corn R² | Tablet R² | N_transfer | Train Time | Inference |
|--------|---------|-----------|------------|------------|-----------|
| PDS (baseline) | 0.91 | 0.93 | 30 | - | <1s |
| LoRA-CT (SOTA) | 0.952 | 0.96 | 50 | 2h | <1s |
| SpectralFM (ours, 10-shot) | 0.96+ | 0.97+ | 10 | 24h pretrain + 10min FT | <1s |
| SpectralFM (ours, TTT) | 0.93+ | 0.94+ | 0 | 24h pretrain + 5min TTT | <1s |
| SpectralFM (ours, 30-shot) | 0.98+ | 0.98+ | 30 | 24h pretrain + 10min FT | <1s |

---

## OPEN DECISIONS FOR USER

1. **Architecture:** Pure Mamba, pure Transformer, or Hybrid? → Recommend hybrid
2. **Scope:** Implement all Tier 1+2 or focus on Tier 1 only? → Recommend Tier 1 + TTT from Tier 2
3. **Paper length:** With this much novelty, consider supplementary material?
4. **Timeline:** 4 weeks still realistic? → Tight but doable with modular approach
5. **Title:** Conservative or bold?
