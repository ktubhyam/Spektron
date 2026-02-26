# Spektron — Implementation Plan

## Task Breakdown (Ordered by Priority)

---

## PHASE 1: MAKE IT RUN (Day 1-2) 🔴 CRITICAL

### Task 1.1: Environment Setup
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn pandas matplotlib seaborn
pip install pywt pot einops tqdm h5py
pip install peft  # For LoRA
# Optional (install if available):
pip install mapie  # Conformal prediction
pip install mamba-ssm  # CUDA Mamba (only on Linux with CUDA)
pip install neuraloperator  # FNO reference
pip install wandb  # Experiment tracking
```

### Task 1.2: Fix Import Paths
- Make all imports relative within `src/`
- Ensure `run.py` can find `src.*` modules
- Add proper `__init__.py` exports

### Task 1.3: Fix Dimension Bugs (CRITICAL)
Run a shape trace through the full model:
```python
# This MUST work before anything else:
cfg = SpectralFMConfig()
model = SpectralFM(cfg)
x = torch.randn(4, 2048)  # Batch of 4, 2048-point spectra
output = model.encode(x, domain="NIR")
print({k: v.shape for k, v in output.items() if isinstance(v, torch.Tensor)})
```

Known dimension issues to fix:
1. `WaveletEmbedding`: output token count N must be deterministic and match across all downstream consumers
2. `config.n_patches`: must equal actual `WaveletEmbedding` output - 2 (minus CLS + domain)
3. `ReconstructionHead`: output shape must match `_patchify()` output
4. `FNOTransferHead`: input is z_chem (B, 128 vector), needs reshaping for FNO which expects (B, C, L)
5. `VIBHead`: input is cls_token (B, d_model=256), outputs z_chem (B, 128) + z_inst (B, 64)

### Task 1.4: Smoke Test
Run the existing `smoke_test` mode in `run.py`:
```bash
python run.py --mode smoke_test
```
Fix until it passes. The smoke test should:
- Create model → print param count
- Generate random data → forward pass → get loss
- Backward pass → verify gradients flow
- Save + load checkpoint

### Task 1.5: Data Pipeline Validation
```python
from src.data.datasets import CornDataset, TabletDataset
ds = CornDataset("data/processed/corn", source="m5", target="mp6", n_channels=2048)
x_source, x_target, y = ds[0]
assert x_source.shape == (2048,), f"Expected (2048,), got {x_source.shape}"
```

---

## PHASE 2: PRETRAINING DATA (Day 2-4)

### Task 2.1: Download Pretraining Datasets
Create `scripts/download_pretrain_data.py`:

**ChEMBL IR-Raman (~220K spectra):**
- Source: Figshare (from DreaMS paper)
- Format: JCAMP-DX or CSV
- Parse, extract wavenumber + absorbance arrays

**USPTO-Spectra (~177K):**
- Source: Zenodo
- Format: Various spectral formats
- Parse into numpy arrays

**NIST WebBook IR (~5.2K):**
- Source: NIST Chemistry WebBook
- Format: JCAMP-DX
- Parse with `jcamp` library

**RRUFF Raman (~8.6K):**
- Source: https://rruff.info/zipped_data_files/raman/
- Format: .rruff files (text)
- Parse into arrays

### Task 2.2: Unified HDF5 Storage
Create `scripts/build_pretrain_hdf5.py`:
```python
# Target structure:
# pretrain_data.h5
#   /spectra         (N, 2048) float32  — resampled to 2048
#   /modality        (N,) int8          — 0=IR, 1=Raman, 2=NIR
#   /source          (N,) int16         — dataset source ID
#   /wavenumber_orig (N, var) float32   — original wavenumber grids (variable length)
#   /metadata        attrs: counts per source, statistics
```

### Task 2.3: Pretraining Dataset Class
Implement `PretrainDataset(Dataset)` in `src/data/datasets.py`:
- Loads from HDF5
- On-the-fly augmentation
- Returns (spectrum, modality, source_id) tuples
- Proper shuffling across sources

---

## PHASE 3: TRAINING (Day 4-10)

### Task 3.1: Pretraining Loop
Fix and validate `PretrainTrainer` in `src/training/trainer.py`:
- Proper masking strategy (contiguous blocks)
- Multi-loss computation with correct weighting
- Learning rate warmup + cosine decay
- Gradient clipping
- Checkpoint saving every N steps
- Validation loss tracking
- GPU memory optimization (gradient accumulation if needed)

**Target:** Pretrain for 50K steps on A100, ~24-48 hours

### Task 3.2: Implement LoRA Injection
The current code has LoRA config but doesn't inject it. Implement:
```python
def inject_lora(model, config):
    """Replace Q/K/V projections in transformer with LoRA versions."""
    from peft import LoraConfig, get_peft_model
    # Or manual implementation:
    # For each attention layer, wrap Linear with LoRA
```

### Task 3.3: Fine-tuning Loop
Fix and validate `FinetuneTrainer`:
- Freeze backbone, unfreeze LoRA + heads
- Small learning rate (1e-4)
- Early stopping with patience
- Sample efficiency curves: N = [5, 10, 20, 30, 50, 100]

### Task 3.4: Test-Time Training
Already implemented in `SpectralFM.test_time_train()` — validate it works:
- Run MSRP on unlabeled test spectra
- Only update normalization layers (safest)
- Verify performance improves on new instrument

### Task 3.5: Classical Baselines
Implement in `src/evaluation/baselines.py`:
```python
class PDS:
    """Piecewise Direct Standardization."""
    
class SBC:
    """Slope/Bias Correction."""
    
class DS:
    """Direct Standardization."""

class CCA:
    """Canonical Correlation Analysis."""

class DiPLS:
    """Domain-Invariant PLS (use diPLSlib if available)."""
```

These are critical — the paper must show Spektron beats all classical methods.

---

## PHASE 4: EXPERIMENTS (Day 10-18)

### Core Experiments

**E1: Pretraining Ablation**
- No pretrain vs pretrain (MSRP only) vs pretrain (full multi-loss)
- Metric: downstream R² on corn transfer

**E2: Masking Strategy**
- Random vs contiguous vs peak-aware masking
- Metric: pretraining loss convergence + downstream transfer

**E3: Sample Efficiency Curve (KEY RESULT)**
- N = [0 (TTT only), 5, 10, 20, 30, 50, 100] transfer samples
- Compare: Spektron vs PDS vs SBC vs LoRA-CT vs Full FT
- Plot: R² vs N for each method
- **Must show Spektron@10 > LoRA-CT@50**

**E4: Full Baseline Comparison (KEY TABLE)**
| Method | Corn (moisture) | Corn (oil) | Corn (protein) | Tablet (API) | N_transfer |
|--------|----------------|------------|----------------|--------------|------------|
| PDS    | ?              | ?          | ?              | ?            | 30         |
| SBC    | ?              | ?          | ?              | ?            | 30         |
| DS     | ?              | ?          | ?              | ?            | 30         |
| CCA    | ?              | ?          | ?              | ?            | 30         |
| di-PLS | ?              | ?          | ?              | ?            | 30         |
| CNN    | ?              | ?          | ?              | ?            | 30         |
| Transformer | ?         | ?          | ?              | ?            | 30         |
| LoRA-CT| ?              | ?          | ?              | ?            | 50         |
| Spektron (ours, 10-shot) | **?** | **?** | **?** | **?** | **10** |
| Spektron (ours, TTT)     | ?     | ?     | ?     | ?     | **0**  |

**E5: Cross-Modality Transfer**
- Pretrain on IR → fine-tune for NIR task
- Does cross-modality pretraining help?

**E6: Interpretability**
- t-SNE of z_chem colored by chemistry vs colored by instrument
- Attention heatmaps on spectra
- GradCAM on spectral features

### Advanced Experiments

**E7: Architecture Ablation**
- Transformer only vs Mamba only vs Hybrid
- With/without wavelet embedding
- With/without MoE
- Training time comparison

**E8: Loss Ablation**
- Progressive: MSRP → +contrastive → +denoise → +OT → +physics → +VIB
- Physics violation rate at each stage

**E9: TTT Evaluation**
- Zero-shot (TTT only) vs Few-shot vs Few-shot + TTT
- Number of TTT steps vs performance

**E10: Disentanglement Quality**
- t-SNE of z_chem: should cluster by chemistry
- t-SNE of z_inst: should cluster by instrument
- Swap z_inst between instruments → check chemistry preserved

**E11: KAN Interpretability** (stretch)
- Extract learned spline functions
- Do they approximate Beer-Lambert?

**E12: Resolution Transfer** (stretch)
- Train on 700ch instrument, test on 2048ch
- Does FNO generalize?

---

## PHASE 5: PAPER (Day 18-24)

### Task 5.1: Results Tables
- Main comparison table (E4)
- Ablation tables (E7, E8)
- Sample efficiency curves (E3)

### Task 5.2: Figures
1. Architecture diagram (TikZ or draw.io)
2. Sample efficiency curves
3. t-SNE visualizations
4. Attention heatmaps
5. Loss convergence plots
6. Wavelet decomposition example

### Task 5.3: Writing
- Abstract, Introduction, Related Work, Methods, Experiments, Discussion, Conclusion
- Target: ~8 pages + supplementary
- Analytical Chemistry format

---

## SUGGESTED CLAUDE CODE PROMPTING SEQUENCE

Here's the recommended order of prompts to give Claude Code:

### Prompt 1: "Fix all imports and run the smoke test"
```
Read CLAUDE.md and PROJECT_STATUS.md. Fix all import errors and dimension 
mismatches in the codebase. Then run `python run.py --mode smoke_test` 
and fix until it passes cleanly. Show me the output.
```

### Prompt 2: "Fix the wavelet embedding"
```
The WaveletEmbedding in src/models/embedding.py uses a Haar approximation.
Replace it with proper pywt integration where the DWT is done in data 
preprocessing and coefficients are passed as input. Fix dimension issues.
```

### Prompt 3: "Implement LoRA injection and validate fine-tuning"
```
LoRA is configured but never injected. Implement inject_lora() that adds
LoRA adapters to the transformer Q/K/V layers. Then run a minimal 
fine-tuning test on the corn dataset.
```

### Prompt 4: "Implement classical baselines"
```
Create src/evaluation/baselines.py with PDS, SBC, DS, CCA implementations.
Run them on corn (m5→mp6) and tablet datasets. Save results to 
experiments/baselines.json.
```

### Prompt 5: "Build the pretraining data pipeline"
```
Create scripts to download and preprocess the pretraining corpus:
- RRUFF Raman (8.6K): https://rruff.info/zipped_data_files/raman/
- NIST IR (5.2K): JCAMP-DX format
Build into HDF5 with resampling to 2048 points.
```

### Prompt 6: "Run pretraining"
```
Pretrain Spektron on the corpus for 50K steps. Use all losses with
configured weights. Save checkpoints every 5K steps. Log losses to 
experiments/pretrain_log.json.
```

### Prompt 7: "Run all experiments E1-E6"
```
See IMPLEMENTATION_PLAN.md for experiment descriptions. Run each one,
save results to experiments/, and generate figures to figures/.
```

---

## TIMELINE

| Days | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Make it run | Smoke test passes, data pipeline works |
| 2-4 | Pretrain data | HDF5 corpus ready (start with RRUFF + NIST) |
| 4-7 | Pretraining | Model pretrained (50K steps, ~24-48h on A100) |
| 7-10 | Fine-tuning | LoRA transfer working, baselines implemented |
| 10-14 | Core experiments | E1-E6 complete, main results table |
| 14-18 | Advanced experiments | E7-E12, ablations, TTT evaluation |
| 18-24 | Paper | Full draft, figures, submission prep |
| 24-30 | Polish | Reviews, rewrites, supplementary, submit |

**Total: ~30 days (relaxed from 4 weeks to 5-6 weeks given full tier scope)**
