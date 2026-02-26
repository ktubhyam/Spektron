# SpectralFM -- Project Status

**Last Updated:** Feb 26, 2026
**Status:** P3 IN PROGRESS. Training running on Vast.ai (2x RTX 5060 Ti). Step ~5300/50000. 20 critical bug fixes applied. Stable training with zero NaN/Inf.

---

## ACTIVE TRAINING (v2)

**Infrastructure:** Vast.ai instance ID 32021974, 2x RTX 5060 Ti 16GB
**Configuration:**
- Backbone: D-LinOSS (damped linear oscillatory SSM)
- d_model=256, d_state=128 oscillators, 4 D-LinOSS layers
- batch_size=16 (8 per GPU via DataParallel), grad_accum=4, effective batch 64
- bfloat16 AMP (LinOSSBlock forced to float32 to prevent overflow)
- Optimizer: AdamW with weight decay groups (only weight matrices), cosine schedule with eta_min=lr*0.01
- Gradient clipping: 5.0
- Dataset: QM9S 222K training samples (IR + Raman, 2048 points each)
- ~7.5GB GPU memory per GPU, ~39 samples/sec

**Current Metrics (step ~5300):**
- MSRP loss: 0.08-0.13
- Validation loss: 0.1070
- NaN/Inf count: 0 (entire run)
- ETA: ~23 hours total for 50K steps

---

## COMPLETED MILESTONES

### Phase 1: Make It Run (COMPLETE)
- All 16 original smoke tests passing
- Fixed VIBHead missing `kl_loss` key
- Fixed `PhysicsLoss` class alias
- Fixed missing `List` import in trainer.py
- Model forward/backward passes verified

### Phase 2A: Classical Baselines (COMPLETE)
- Implemented PDS, SBC, DS, PLS in `src/evaluation/baselines.py`
- Fixed critical transfer direction bug (target -> source space)
- Baseline results on corn m5 -> mp6 (30 transfer samples):
  - DS: R2=0.69, RMSEP=0.22
  - SBC: R2=0.38, RMSEP=0.31
  - PDS: R2=-5.50 (poor due to limited samples)
  - No Transfer: R2=-21.46 (expected failure)
  - Target Direct: R2=0.69 (upper bound)

### Phase 2B: Pretraining Data Pipeline (COMPLETE)
- Built pretraining corpus v2: 61,420 spectra (15,355 real from 2 sources + 3x augmentation)
- Sources: RRUFF (9,941 Raman) + OpenSpecy (4,778 Raman + 636 FTIR)
- Created multi-source corpus downloaders (RRUFF, OpenSpecy, ChEMBL, USPTO)
- SNV normalization and 2048-point resampling pipeline

### Phase 2C: Architecture Fixes + W&B Integration (COMPLETE)
- Real Daubechies-4 DWT via PyWavelets in embedding.py
- LoRA injection module for fine-tuning (~0.4% of backbone params)
- Dual W&B + JSONL experiment logger
- All 19/19 smoke tests passing

### Phase 2D: D-LinOSS Backbone + QM9S Pipeline (COMPLETE)
- Implemented D-LinOSS backbone (`src/models/dlinoss.py`, `src/models/linoss/`)
- Damped linear oscillatory SSM with 2nd-order dynamics matching molecular vibrations
- IMEX symplectic discretization, O(n) parallel associative scan
- QM9S dataset pipeline (`src/data/qm9s.py`) for 222K DFT-computed spectra
- RawSpectralEmbedding for full 2048-point processing without patching

### Phase 3: GPU Training (IN PROGRESS)
- v2 training launched and running stably on Vast.ai
- 20 critical bug fixes applied across 5 rounds (see below)
- Zero NaN/Inf errors in entire training run
- MSRP loss converging: started ~1.0, now at 0.08-0.13

---

## 20 CRITICAL BUG FIXES APPLIED

### Round 1: NaN Fixes
1. **D-LinOSS + AMP NaN** -- LinOSSBlock SSM produces values +/-200K which overflow float16 in GLU. Fix: Force entire LinOSSBlock to run in float32 via `torch.amp.autocast('cuda', enabled=False)`
2. **DataParallel domain list** -- Domain (list of strings) not split by DataParallel. Fix: slice `domain[:batch_size]` in `_resolve_domain_indices`
3. **OT loss + AMP** -- Sinkhorn kernel `exp(-C/reg)` underflows in float16. Fix: Move OT computation outside AMP autocast block in trainer
4. **OT reg too low** -- Changed from 0.05 to 1.0 for 128-dim embeddings

### Round 2: Fundamental Training Fixes
5. **CRITICAL: Mask not applied to encoder input** -- pretrain_forward() passed full unmasked spectrum to encode(). Model learned near-identity (MSRP 0.82 -> 0.003 in 700 steps). Fix: Add learnable `mask_token` param, replace masked positions in embedding space BEFORE backbone
6. **CRITICAL: Grad accum + logging/val/save fired 4x** -- With grad_accum=4, logging/validation/checkpoint happened every mini-batch sub-step. Fix: train_step returns `(losses, did_step)` flag, only log/validate/save when `did_step=True`
7. **Step 0 triggers all periodic actions** -- 0%N==0 always true. Fix: Add `self.step > 0` to all periodic checks
8. **VIB adversarial loss wrong** -- Was training classifier to output uniform (KL-to-uniform). Fix: Added GradientReversal layer in heads.py, changed loss to cross_entropy with gradient reversal
9. **Physics loss dead (zero gradient)** -- Was applied to `target_patches` (ground truth). Fix: Apply to `reconstruction` output
10. **Loss averaging** -- Now averages losses over grad_accum window instead of logging single sub-step
11. **Throughput metric** -- Now accounts for grad_accum in samples_per_sec calculation

### Round 3: CFL Stability
12. **CRITICAL: D-LinOSS CFL instability** -- DampedLayer `M_22 = 1 - step^2*A/S` goes below -1 when A_diag grows during training, causing eigenvalues to exit unit circle, exponential divergence in 2048-step scan, NaN at step ~1000-1400. Fix: Clamp CFL ratio `alpha = step^2*A/S <= 1.99`. Also switched to bfloat16 AMP and added NaN guard in trainer

### Round 4: v3 Refinements
13. **CFL soft clamp** -- Hard `torch.clamp(alpha, max=1.99)` replaced with soft `1.99 * torch.tanh(alpha / 1.99)` for differentiability
14. **GradientReversal DataParallel safety** -- `x.view_as(x)` replaced with `x.clone()` in GradientReversal.forward()
15. **MoE gating float32** -- `F.softmax(top_k_logits.float(), dim=-1)` for AMP safety
16. **VIB weight increase** -- 0.05 -> 0.15 for better disentanglement

### Round 5: Deep Evaluation Fixes
17. **Weight decay on ALL params** -- AdamW applied weight_decay to LayerNorm, bias, embeddings. Fix: Parameter group differentiation -- only decay weight matrices
18. **DataParallel domain scatter (proper fix)** -- Previous fix #2 only worked for GPU0. GPU1 still got wrong domains because list[str] is replicated, not scattered. Fix: Convert domain list to integer tensor in trainer BEFORE model.forward(), so DataParallel scatters correctly
19. **Cosine schedule decays to LR=0** -- CosineAnnealingLR had no eta_min. Fix: Added `eta_min=lr*0.01` to maintain minimum learning rate
20. **Grad clip too tight** -- Was hardcoded as 1.0. Fix: Removed hardcode, increased default to 5.0 in config.py

---

## ARCHITECTURE (Current)

```
Spectrum (B, 2048)
  -> RawSpectralEmbedding     Conv1d patching + wavenumber PE + [CLS] + [DOMAIN]
  -> DLinOSSBackbone           4 damped linear oscillatory SSM blocks, O(n)
     - Each block: BatchNorm -> DampedSSM -> GELU -> Dropout -> GLU -> Dropout -> Residual
     - DampedSSM: 2nd-order oscillatory dynamics, 128 oscillators per layer
     - IMEX symplectic discretization with CFL-stable soft clamping
     - Forced float32 computation to prevent AMP overflow
  -> MixtureOfExperts          4 experts, top-2 gating (float32 softmax)
  -> TransformerEncoder        2 blocks, 8 heads, global reasoning
  -> VIBHead                   z_chem (128d) + z_inst (64d) with GradientReversal
  -> Heads                     Reconstruction | Regression | FNO Transfer
```

**Key physics insight:** D-LinOSS dynamics are mathematically identical to damped harmonic oscillators, which is exactly the physics of molecular vibrations. The learned natural frequencies and damping coefficients can be compared against physical vibrational modes.

---

## FILE-BY-FILE STATUS

### Data (READY)
| File | Status | Notes |
|------|--------|-------|
| `data/processed/corn/*.npy` | Ready | 80 samples, 3 instruments (m5, mp5, mp6), 700 channels, 4 properties |
| `data/processed/tablet/*.npy` | Ready | 655 samples, 2 instruments, 650 channels, 3 properties |
| `data/pretrain/spectral_corpus_v2.h5` | Ready | 61,420 spectra (15,355 real), 2048 channels, 0.47 GB |
| QM9S corpus (remote) | Active | 222K IR+Raman spectra, DFT-computed, on Vast.ai training instance |

### Models
| File | Status | Notes |
|------|--------|-------|
| `src/models/dlinoss.py` | Active | D-LinOSS backbone -- current default backbone |
| `src/models/linoss/layers.py` | Active | LinOSSBlock + DampedLayer with CFL soft clamp, float32 forced |
| `src/models/linoss/scan.py` | Active | Parallel associative scan |
| `src/models/embedding.py` | Active | WaveletEmbedding + RawSpectralEmbedding, domain tensor support |
| `src/models/spectral_fm.py` | Active | Full model assembly, supports both Mamba and D-LinOSS |
| `src/models/heads.py` | Active | VIB + GradientReversal, Reconstruction, Regression, FNO |
| `src/models/moe.py` | Active | MoE with float32 gating for AMP safety |
| `src/models/transformer.py` | Active | 2-block TransformerEncoder |
| `src/models/mamba.py` | Legacy | Pure PyTorch Mamba (kept for ablation, not used in training) |
| `src/models/lora.py` | Ready | LoRA injection for fine-tuning phase |

### Training + Data Pipeline
| File | Status | Notes |
|------|--------|-------|
| `src/training/trainer.py` | Active | Grad accum fix, loss averaging, NaN guard, bfloat16 AMP |
| `src/config.py` | Active | Weight decay groups, grad clip 5.0, cosine eta_min |
| `src/data/qm9s.py` | Active | QM9S 222K dataset pipeline |
| `src/data/datasets.py` | Ready | Corn/Tablet + PretrainHDF5Dataset |
| `src/data/corpus_downloader.py` | Ready | RRUFF, OpenSpecy, ChEMBL, USPTO downloaders |
| `src/losses/losses.py` | Active | All losses computing correctly, OT outside AMP |
| `src/utils/logging.py` | Active | Dual W&B + JSONL logging |

---

## NEXT STEPS

### Immediate (P3 completion)
1. **Complete 50K step pretraining** -- currently at ~5300, ETA ~18 hours remaining
2. **Sync v3/v4/v5 fixes to remote** -- fixes 13-20 applied locally, not yet on training instance
3. **Validate training curves** -- confirm continued loss convergence through 50K steps

### Phase 4: Fine-tuning + Evaluation
4. **LoRA fine-tune on corn** -- load pretrained checkpoint, 10/20/30 transfer samples
5. **Compare to baselines** -- target: R2 > 0.95 with 10 samples (vs DS R2=0.69 with 30)
6. **TTT evaluation** -- zero-shot transfer via test-time self-supervision
7. **Ablation studies** -- D-LinOSS vs Mamba, VIB weight, MoE experts, etc.

### Phase 5: Paper
8. **Full experiment suite** (E1-E12)
9. **Symmetry-stratified analysis** (leveraging theory from CORRECTED_PAPER_BLUEPRINT.md)
10. **Paper writing** -- targeting Analytical Chemistry (ACS)

---

## KEY LESSONS LEARNED

1. **Mask before encoder, not after:** BERT-style masked pretraining requires replacing masked positions with a learnable mask token BEFORE the encoder, not just using the mask for loss selection. Without this, the model degenerates to a trivial copy task.

2. **CFL stability in SSMs:** Learned parameters in oscillatory SSMs can grow during training, causing the discretization matrix eigenvalues to exit the unit circle. A soft CFL clamp is essential for long sequences (2048 steps).

3. **DataParallel + non-tensor inputs:** Python lists and strings are replicated (not scattered) by DataParallel. Convert all model inputs to tensors before DataParallel boundary.

4. **AMP precision hierarchy:** Some operations (SSM scan, Sinkhorn OT, softmax gating) must be forced to float32 even under AMP to prevent overflow/underflow.

5. **Gradient accumulation logging:** Only log metrics, validate, and save checkpoints on actual optimizer steps, not on every micro-batch.
