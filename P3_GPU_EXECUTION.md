# P3: GPU Execution — SpectralFM v2
# Instance: 4× RTX 5090 · AMD EPYC 9654 · 515GB RAM · 209GB Disk
# IP: 211.72.13.201 · $0.117/hr
# Model: 6.5M params · Corpus: 444MB (61,420 spectra)
# Estimated wall time: 2–3 hours · Estimated cost: ~$0.35

# ═══════════════════════════════════════════════════════════════
# PHASE 1: INSTANCE SETUP (10 min)
# ═══════════════════════════════════════════════════════════════

## 1.1 SSH In

```bash
ssh ubuntu@211.72.13.201
```

If key-based auth isn't set up yet, Lambda provides a web terminal.
Once in, verify GPU access immediately:

```bash
nvidia-smi
# Expected: 4× NVIDIA GeForce RTX 5090, ~32GB each
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"
```

## 1.2 Clone Repository

```bash
cd ~
git clone https://github.com/ktubhyam/VS3L.git
cd ~/VS3L
```

## 1.3 Transfer Data Files (from LOCAL machine, new terminal)

The corpus and processed data are gitignored. Transfer them via scp.
Run these commands from your LOCAL Mac terminal:

```bash
# From your LOCAL machine:
REMOTE=ubuntu@211.72.13.201

# Create directories on remote first
ssh $REMOTE "mkdir -p ~/VS3L/data/pretrain ~/VS3L/data/processed/corn ~/VS3L/data/processed/tablet ~/VS3L/checkpoints ~/VS3L/experiments ~/VS3L/figures ~/VS3L/logs"

# Transfer corpus (444MB — ~4 sec at 903 Mbps)
scp /Users/admin/Documents/GitHub/VS3L/data/pretrain/spectral_corpus_v2.h5 $REMOTE:~/VS3L/data/pretrain/

# Transfer processed corn data (7 files, ~4.5MB total)
scp /Users/admin/Documents/GitHub/VS3L/data/processed/corn/*.npy $REMOTE:~/VS3L/data/processed/corn/

# Transfer processed tablet data (10 files, ~3.4MB total)
scp /Users/admin/Documents/GitHub/VS3L/data/processed/tablet/*.npy $REMOTE:~/VS3L/data/processed/tablet/
scp /Users/admin/Documents/GitHub/VS3L/data/processed/tablet/metadata.json $REMOTE:~/VS3L/data/processed/tablet/
```

## 1.4 Install Dependencies (on the remote instance)

```bash
cd ~/VS3L

# Core dependencies (torch should already be on Lambda, but verify)
pip install h5py PyWavelets POT einops wandb scipy scikit-learn matplotlib seaborn tqdm pyyaml peft pandas jcamp

# Verify critical imports
python3 -c "
import torch, h5py, pywt, ot, scipy, sklearn, matplotlib, wandb, einops
print(f'PyTorch {torch.__version__}')
print(f'CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
print(f'GPU 0: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('All imports OK')
"
```

## 1.5 Verify Data Integrity

```bash
cd ~/VS3L
python3 -c "
import h5py, numpy as np
from pathlib import Path

# Corpus
with h5py.File('data/pretrain/spectral_corpus_v2.h5', 'r') as f:
    keys = list(f.keys())
    print(f'Corpus keys: {keys}')
    print(f'Spectra shape: {f[\"spectra\"].shape}')
    print(f'Corpus size: {f[\"spectra\"].shape[0]:,} spectra')

# Corn
corn = Path('data/processed/corn')
for name in ['m5_spectra', 'mp5_spectra', 'mp6_spectra', 'properties', 'wavelengths']:
    a = np.load(corn / f'{name}.npy')
    print(f'  corn/{name}: {a.shape} {a.dtype}')

# Tablet
tablet = Path('data/processed/tablet')
for name in ['calibrate_1', 'calibrate_2', 'calibrate_Y', 'test_1', 'test_2', 'test_Y']:
    a = np.load(tablet / f'{name}.npy')
    print(f'  tablet/{name}: {a.shape}')

print()
print('DATA INTEGRITY: ALL OK')
"
```

Expected output:
```
Corpus size: 61,420 spectra
corn/m5_spectra: (80, 700)
corn/mp5_spectra: (80, 700)
corn/mp6_spectra: (80, 700)
corn/properties: (80, 4)
corn/wavelengths: (700,)
tablet/calibrate_1: (155, 650)
tablet/calibrate_2: (155, 650)
tablet/calibrate_Y: (155, 3)
tablet/test_1: (460, 650)
tablet/test_2: (460, 650)
tablet/test_Y: (460, 3)
```

## 1.6 Verify Model Builds on GPU

```bash
cd ~/VS3L
python3 -c "
import torch
from src.config import SpectralFMConfig
from src.models.spectral_fm import SpectralFM, SpectralFMForPretraining
from src.models.lora import inject_lora

config = SpectralFMConfig()
model = SpectralFM(config).cuda()
x = torch.randn(4, 2048).cuda()
enc = model.encode(x, domain='NIR')
print(f'z_chem: {enc[\"z_chem\"].shape}')
print(f'z_inst: {enc[\"z_inst\"].shape}')
pred = model.predict(x, mc_samples=3)
print(f'prediction: {pred[\"prediction\"].shape}')
print(f'uncertainty: {pred[\"uncertainty\"].shape}')
print(f'GPU mem used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
print()
print('MODEL ON GPU: OK')
del model; torch.cuda.empty_cache()
"
```

Expected: GPU mem < 1 GB. This model is tiny (6.5M params), leaving >30GB headroom per GPU.

## 1.7 Run Smoke Tests

```bash
cd ~/VS3L
python3 tests/smoke_test.py
```

Expected: 19/19 pass. If anything fails, stop and debug before proceeding.

## 1.8 Run Integration Tests

```bash
python3 scripts/test_integration.py
```

Expected: 16/16 pass.

## 1.9 W&B Login (recommended)

```bash
wandb login
# Paste your API key when prompted
# Or: export WANDB_API_KEY=<your-key>
```

# ═══════════════════════════════════════════════════════════════
# CHECKPOINT: Phase 1 Complete
# At this point:
#   ✓ GPU verified (4× RTX 5090)
#   ✓ Code cloned and all imports work
#   ✓ Data transferred and shapes verified
#   ✓ Model builds and runs on GPU
#   ✓ 19/19 smoke + 16/16 integration tests pass
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# PHASE 2: DIAGNOSTIC EXPERIMENT (5–10 min)
# ═══════════════════════════════════════════════════════════════
#
# PURPOSE: Validate the pretrain→TTT→predict pipeline works AT ALL
# before spending 1–2 hours on full pretraining.
#
# What it does:
#   1. Pretrains ONLY on corn m5 + mp5 (160 spectra, same modality)
#   2. Applies zero-shot TTT to mp6 (unseen instrument)
#   3. Measures R² with ZERO labeled transfer samples
#
# This bypasses the modality gap (corpus is 96% Raman, corn is NIR)
# and directly tests: does the architecture learn anything useful?

```bash
cd ~/VS3L
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_diagnostic.py \
    --device cuda \
    --max-steps 2000 \
    --batch-size 32
```

## What to watch for:

**During pretraining** (20 log lines):
```
Step  100/2000 | Loss: 0.XXXX | MSRP: 0.XXXX | ...
Step  200/2000 | Loss: 0.XXXX | ...
```
- Loss should decrease from ~1.0 → ~0.2–0.5
- If loss stays >0.8 after 500 steps: learning rate may be wrong
- If loss goes NaN: reduce lr to 1e-4

**TTT evaluation** (5 lines):
```
TTT steps=  0: R²=X.XXXX, RMSE=X.XXXX
TTT steps=  5: R²=X.XXXX, RMSE=X.XXXX
TTT steps= 10: R²=X.XXXX, RMSE=X.XXXX
TTT steps= 20: R²=X.XXXX, RMSE=X.XXXX
TTT steps= 50: R²=X.XXXX, RMSE=X.XXXX
```

**Classical baselines** (for comparison):
```
Method               R²      RMSE
DS                   0.69XX  X.XXXX
PDS                  0.XX    X.XXXX
```

## Decision Gate:

| Diagnostic Result | Meaning | Action |
|---|---|---|
| ✅ Best TTT R² > 0.0 | Architecture learns useful representations | → Proceed to Phase 3 |
| 🎉 Best TTT R² > 0.3 | Extraordinary — already paper-worthy | → Proceed with high confidence |
| ⚠️ Best TTT R² ≤ 0.0 | Pipeline not working yet | → Run with `--max-steps 5000`, if still ≤ 0 → debug (see Troubleshooting) |

**Results saved to:** `experiments/diagnostic_results.json`

**Checkpoint saved to:** `checkpoints/diagnostic_pretrain.pt`

**Read the results:**
```bash
python3 -c "
import json
with open('experiments/diagnostic_results.json') as f:
    d = json.load(f)
print(f'Pretrain loss: {d[\"pretrain_final_loss\"]:.4f}')
print(f'Pretrain time: {d[\"pretrain_time_sec\"]:.0f}s')
for steps, r in sorted(d['ttt_results'].items(), key=lambda x: int(x[0])):
    print(f'  TTT steps={steps:>3s}: R²={r[\"r2\"]:.4f}')
print(f'Diagnosis: {\"PASS\" if d[\"diagnosis\"][\"pass\"] else \"FAIL\"} (best R²={d[\"diagnosis\"][\"best_ttt_r2\"]:.4f})')
"
```


# ═══════════════════════════════════════════════════════════════
# PHASE 3: FULL PRETRAINING (30–90 min on 4× RTX 5090)
# ═══════════════════════════════════════════════════════════════
#
# Only proceed if diagnostic PASSED (TTT R² > 0.0)
#
# The full corpus (61,420 spectra) will be used to pretrain for
# 50,000 steps. With batch_size=128 on RTX 5090, expect:
#   - ~500–2000 samp/s throughput
#   - ~30–90 minutes total
#   - GPU memory: ~2–4 GB (model is tiny)
#
# Using a single GPU is sufficient. The other 3 GPUs will be used
# later for parallel evaluation experiments.

## 3.1 Sanity Check (2 min)

Quick 100-step run to verify throughput, loss decreasing, checkpointing:

```bash
cd ~/VS3L
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_pretrain.py \
    --max-steps 100 \
    --batch-size 128 \
    --lr 3e-4 \
    --num-workers 8 \
    --log-every 10 \
    --save-every 50 \
    --val-every 50 \
    --no-wandb \
    --run-name "sanity_100"
```

**Check these things:**
1. ✓ Loss decreasing (step 0: ~1.0, step 100: < 0.8)
2. ✓ Throughput > 300 samp/s (should be much higher on 5090)
3. ✓ Validation runs at step 50 (shows "Val Loss: X.XXXX")
4. ✓ Checkpoint saved at step 50 and 100
5. ✓ No OOM (GPU mem < 10 GB at batch 128)
6. ✓ ETA is reasonable (< 90 min for 50K steps)

**If OOM at batch 128:** use batch 64 (should not happen on 32GB 5090)
**If throughput < 100 samp/s:** increase `--num-workers` to 16

**Verify checkpoint was saved:**
```bash
ls -la checkpoints/
# Should see: pretrain_step_50.pt, pretrain_step_100.pt, pretrain_final.pt
```

**Cleanup sanity artifacts:**
```bash
rm -f checkpoints/pretrain_step_50.pt checkpoints/pretrain_step_100.pt checkpoints/pretrain_final.pt
```

## 3.2 Full Pretraining Run

```bash
cd ~/VS3L
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_pretrain.py \
    --max-steps 50000 \
    --batch-size 128 \
    --lr 3e-4 \
    --num-workers 8 \
    --log-every 100 \
    --save-every 5000 \
    --val-every 1000 \
    --wandb \
    --run-name "pretrain_v2_50k"
```

**Expected training trajectory:**

| Step | Loss | MSRP | Physics | VIB | ETA |
|------|------|------|---------|-----|-----|
| 0 | ~1.0 | ~0.8 | ~0.15 | ~0.05 | full |
| 1,000 | ~0.5 | ~0.3 | ~0.10 | ~0.04 | ~80% left |
| 5,000 | ~0.3 | ~0.15 | ~0.07 | ~0.03 | ~60% left |
| 10,000 | ~0.2 | ~0.10 | ~0.05 | ~0.02 | ~40% left |
| 25,000 | ~0.15 | ~0.07 | ~0.04 | ~0.02 | ~20% left |
| 50,000 | ~0.1 | ~0.05 | ~0.03 | ~0.01 | done |

**Monitor from a second SSH terminal:**
```bash
# Live loss monitoring
tail -f ~/VS3L/logs/pretrain_v2_50k/metrics.jsonl 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        if 'val/loss' in d:
            print(f'Step {d.get(\"step\",\"?\"):>6} | Val: {d[\"val/loss\"]:.4f}')
        elif 'train/loss' in d:
            print(f'Step {d.get(\"step\",\"?\"):>6} | Loss: {d[\"train/loss\"]:.4f} | {d.get(\"perf/samples_per_sec\",0):.0f} samp/s | ETA: {d.get(\"perf/eta_minutes\",0):.0f}m | GPU: {d.get(\"perf/gpu_mem_gb\",0):.1f}GB')
    except: pass
"

# Or just watch GPU utilization
watch -n 5 nvidia-smi
```

**If training is interrupted** (SSH drops, need to restart):
```bash
# Resume from latest checkpoint
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_pretrain.py \
    --max-steps 50000 \
    --batch-size 128 \
    --lr 3e-4 \
    --num-workers 8 \
    --log-every 100 \
    --save-every 5000 \
    --val-every 1000 \
    --wandb \
    --run-name "pretrain_v2_50k_resumed" \
    --resume checkpoints/pretrain_step_XXXXX.pt
```

## 3.3 Verify Pretrained Checkpoint

```bash
python3 -c "
import torch
ckpt = torch.load('checkpoints/best_pretrain.pt', map_location='cpu', weights_only=False)
print('Checkpoint keys:', list(ckpt.keys()))
print(f'Training step: {ckpt[\"step\"]}')
print(f'Best val loss: {ckpt[\"best_val_loss\"]:.4f}')
print(f'Model keys: {len(ckpt[\"model_state_dict\"])} tensors')
total_params = sum(v.numel() for v in ckpt['model_state_dict'].values())
print(f'Total parameters: {total_params:,}')
print()
print('PRETRAINED CHECKPOINT: OK')
"
```

**Also verify final checkpoint exists:**
```bash
ls -lh checkpoints/*.pt
# Expected:
#   best_pretrain.pt          (best validation loss)
#   pretrain_final.pt         (step 50000)
#   pretrain_step_5000.pt     (intermediate)
#   pretrain_step_10000.pt    (intermediate)
#   ...
```

# ═══════════════════════════════════════════════════════════════
# CHECKPOINT: Phase 3 Complete
# At this point:
#   ✓ Diagnostic passed
#   ✓ 50K-step pretraining complete
#   ✓ best_pretrain.pt saved with best val loss
#   ✓ W&B dashboard shows convergence
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# PHASE 4: EVALUATION SUITE (60–90 min)
# ═══════════════════════════════════════════════════════════════
#
# Now we run all experiments against the pretrained checkpoint.
# With 4 GPUs available, we can parallelize.
#
# Strategy:
#   GPU 0: E3 (sample efficiency) + E4 (TTT) — THE KEY RESULTS
#   GPU 1: E1 (pretrain ablation) + E2 (LoRA vs full FT)
#   GPU 2: E5 (cross-instrument) + E10 (disentanglement) + E11 (calibration)
#   GPU 3: E12 (tablet)
#   CPU:   Baselines (no GPU needed)

## 4.0 Complete Baselines (CPU, ~15 min) — START FIRST

Run this in its own terminal (no GPU needed):

```bash
cd ~/VS3L
python3 scripts/run_baselines_complete.py --n-seeds 5 \
    > logs/baselines_stdout.txt 2>&1 &
echo "Baselines PID: $!"
```

This runs all 6 methods × 4 corn properties × 6 instrument pairs × 5 seeds.
Output: `experiments/baselines_complete.json`

## 4.1 E3 + E4: THE KEY RESULTS (GPU 0, ~45 min)

**This is the most important experiment.** Run it on GPU 0:

```bash
cd ~/VS3L

# E3: Sample efficiency sweep (R² vs N transfer samples)
# This generates THE KEY FIGURE for the paper
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_finetune.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn \
    --all-properties \
    --sweep \
    --n-seeds 5 \
    > logs/E3_stdout.txt 2>&1 &
E3_PID=$!
echo "E3 PID: $E3_PID"
```

**While E3 runs**, check zero-shot TTT immediately in a quick separate call:

```bash
# Quick zero-shot check (just moisture, m5→mp6, ~2 min)
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn \
    --source m5 --target mp6 \
    --property moisture
```

**READ THIS RESULT IMMEDIATELY:**
```bash
python3 -c "
import json, glob
files = sorted(glob.glob('experiments/ttt_zeroshot_*.json'))
if files:
    with open(files[-1]) as f: d = json.load(f)
    r = d.get('results', d)
    # Look for zero-shot (0 steps with varying TTT steps)
    for k, v in sorted(r.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        r2 = v['r2'] if isinstance(v, dict) else v
        print(f'  TTT steps={k}: R²={r2:.4f}' if isinstance(v, dict) else f'  {k}: {v}')
    print()
    best = max((v['r2'] if isinstance(v, dict) else -999) for v in r.values() if isinstance(v, dict))
    if best > 0.3:
        print('🎉 R² > 0.3 — NEW PARADIGM RESULT!')
    elif best > 0.1:
        print('✅ R² > 0.1 — viable paper result')
    elif best > 0.0:
        print('⚠️  R² > 0.0 — marginal, focus on few-shot story')
    else:
        print('❌ R² ≤ 0.0 — TTT not working, pivot to LoRA-only paper')
else:
    print('No TTT results found yet')
"
```

**Then run full TTT sweep (after quick check completes):**

```bash
# E4: Full TTT sweep (all properties, TTT+LoRA, ablation)
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn \
    --sweep \
    --n-seeds 5 \
    > logs/E4_stdout.txt 2>&1 &
E4_PID=$!
echo "E4 PID: $E4_PID"
```

## 4.2 E1 + E2: Ablation Studies (GPU 2, ~30 min)

```bash
cd ~/VS3L

# E1: Pretraining ablation (pretrained vs random init)
CUDA_VISIBLE_DEVICES=2 python3 scripts/run_experiments.py \
    --experiment E1 \
    --checkpoint checkpoints/best_pretrain.pt \
    --n-seeds 3 \
    > logs/E1_stdout.txt 2>&1

# E2: LoRA rank sweep vs full fine-tuning
CUDA_VISIBLE_DEVICES=2 python3 scripts/run_experiments.py \
    --experiment E2 \
    --checkpoint checkpoints/best_pretrain.pt \
    --n-seeds 3 \
    > logs/E2_stdout.txt 2>&1
```

## 4.3 E5 + E10 + E11: Supporting Experiments (GPU 3, ~25 min)

```bash
cd ~/VS3L

# E5: Cross-instrument generalization (all 6 pairs)
CUDA_VISIBLE_DEVICES=3 python3 scripts/run_experiments.py \
    --experiment E5 \
    --checkpoint checkpoints/best_pretrain.pt \
    > logs/E5_stdout.txt 2>&1

# E10: VIB disentanglement (t-SNE visualization)
CUDA_VISIBLE_DEVICES=3 python3 scripts/run_experiments.py \
    --experiment E10 \
    --checkpoint checkpoints/best_pretrain.pt \
    > logs/E10_stdout.txt 2>&1

# E11: Uncertainty calibration (MC Dropout)
CUDA_VISIBLE_DEVICES=3 python3 scripts/run_experiments.py \
    --experiment E11 \
    --checkpoint checkpoints/best_pretrain.pt \
    > logs/E11_stdout.txt 2>&1
```

## 4.4 E12: Tablet Validation (GPU 3, ~15 min, after E5+E10+E11)

```bash
CUDA_VISIBLE_DEVICES=3 python3 scripts/run_experiments.py \
    --experiment E12 \
    --checkpoint checkpoints/best_pretrain.pt \
    > logs/E12_stdout.txt 2>&1
```

## 4.5 TTT Ablation (GPU 1, after E4 finishes, ~20 min)

```bash
# Wait for E4 to finish
wait $E4_PID

CUDA_VISIBLE_DEVICES=1 python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn \
    --ablation \
    > logs/ttt_ablation_stdout.txt 2>&1
```

## 4.6 Monitor All Jobs

```bash
# Check which jobs are still running
jobs -l

# Check GPU utilization across all 4 GPUs
nvidia-smi

# Check experiment outputs as they complete
ls -lt experiments/*.json | head -20

# Wait for all background jobs
wait
echo "ALL EXPERIMENTS COMPLETE"
```


# ═══════════════════════════════════════════════════════════════
# PHASE 5: RESULTS ANALYSIS & FIGURE GENERATION (5 min)
# ═══════════════════════════════════════════════════════════════

## 5.1 Generate All Figures

```bash
cd ~/VS3L
python3 -c "
from src.evaluation.visualization import generate_all_figures_from_experiments
generate_all_figures_from_experiments('experiments', 'figures')
print('Figures generated:')
import os
for f in sorted(os.listdir('figures')):
    print(f'  {f}')
"
```

## 5.2 Print Paper-Ready Results Table

```bash
python3 -c "
import json, glob

# Load baselines
with open('experiments/baselines_complete.json') as f:
    baselines = json.load(f)

print('='*80)
print('TABLE 1: Calibration Transfer Results (Corn Dataset)')
print('='*80)

# Print aggregated results
if 'aggregated' in baselines:
    for exp_key, methods in baselines['aggregated'].items():
        print(f'\n{exp_key}:')
        for method, metrics in methods.items():
            r2 = metrics.get('r2_mean', metrics.get('r2', float('nan')))
            std = metrics.get('r2_std', 0)
            print(f'  {method:<12s}: R² = {r2:.4f} ± {std:.4f}')
"
```

## 5.3 Print Key Result Summary

```bash
python3 << 'PYTHON'
import json, glob, os
from pathlib import Path

print("=" * 70)
print("SPECTRALFM v2 — KEY RESULTS SUMMARY")
print("=" * 70)

# 1. Zero-shot TTT
ttt_files = sorted(glob.glob("experiments/ttt_zeroshot_*.json"))
if ttt_files:
    with open(ttt_files[-1]) as f:
        d = json.load(f)
    results = d.get("results", d)
    best_r2 = -999
    best_steps = 0
    for k, v in results.items():
        if isinstance(v, dict) and "r2" in v:
            if v["r2"] > best_r2:
                best_r2 = v["r2"]
                best_steps = k
    zero_r2 = results.get("0", results.get(0, {}))
    if isinstance(zero_r2, dict):
        zero_r2 = zero_r2.get("r2", "N/A")
    print(f"\n1. ZERO-SHOT TTT (0 labeled samples)")
    print(f"   R² at 0 steps:    {zero_r2}")
    print(f"   Best R²:          {best_r2:.4f} (at {best_steps} TTT steps)")

# 2. Sample efficiency
ft_files = sorted(glob.glob("experiments/finetune_*.json"))
if ft_files:
    with open(ft_files[-1]) as f:
        d = json.load(f)
    print(f"\n2. SAMPLE EFFICIENCY (LoRA fine-tuning)")
    for prop, res in d.get("results", {}).items():
        sweep = res.get("sweep", {})
        if sweep:
            for n in sorted(sweep.keys(), key=int):
                r2 = sweep[n].get("r2_mean", "?")
                print(f"   {prop} @ n={n}: R²={r2}")

# 3. DS baseline comparison
bl_path = "experiments/baselines_complete.json"
if os.path.exists(bl_path):
    with open(bl_path) as f:
        bl = json.load(f)
    print(f"\n3. CLASSICAL BASELINES (30 transfer samples)")
    # Just show m5→mp6 moisture
    for exp in bl.get("results", []):
        if "m5" in str(exp.get("source","")) and "mp6" in str(exp.get("target","")) and "moisture" in str(exp.get("property","")):
            for method, metrics in exp.get("methods", {}).items():
                r2 = metrics.get("r2_mean", metrics.get("r2", "?"))
                print(f"   {method}: R²={r2}")
            break

# 4. Paper viability
print(f"\n4. PAPER VIABILITY ASSESSMENT")
if ttt_files:
    if best_r2 > 0.3:
        print("   🎉 Anal. Chem. STRONG: Zero-shot R² > 0.3 — new paradigm")
        print("   Consider Nature MI if scaling experiment (E7) also works")
    elif best_r2 > 0.1:
        print("   ✅ Anal. Chem. VIABLE: Zero-shot R² > 0.1")
        print("   Lead with few-shot LoRA, TTT as key novelty")
    elif best_r2 > 0.0:
        print("   ⚠️  Anal. Chem. POSSIBLE: TTT shows signal but weak")
        print("   Focus on LoRA efficiency + physics pretraining story")
    else:
        print("   ❌ PIVOT: TTT not working at zero-shot")
        print("   Paper becomes: physics-informed pretraining + LoRA transfer")

print("\n" + "=" * 70)
PYTHON
```


# ═══════════════════════════════════════════════════════════════
# PHASE 6: CONDITIONAL EXPERIMENTS (only if R² > 0.3)
# ═══════════════════════════════════════════════════════════════
#
# These experiments target Nature Machine Intelligence.
# Only run if zero-shot TTT R² > 0.3.

## 6.1 E7: Scaling Law (requires retraining at different corpus sizes)

This tests whether downstream performance scales with pretraining corpus size.
Runs 3 additional pretraining jobs at subsampled corpus sizes.

```bash
cd ~/VS3L

# Run 3 scaled pretraining jobs in parallel (one per GPU)
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_pretrain.py \
    --max-steps 20000 --batch-size 128 --lr 3e-4 \
    --max-samples 5000 --num-workers 4 \
    --checkpoint-dir checkpoints/scaling_5k \
    --no-wandb --run-name "scaling_5k" \
    > logs/scaling_5k.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python3 scripts/run_pretrain.py \
    --max-steps 20000 --batch-size 128 --lr 3e-4 \
    --max-samples 15000 --num-workers 4 \
    --checkpoint-dir checkpoints/scaling_15k \
    --no-wandb --run-name "scaling_15k" \
    > logs/scaling_15k.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 scripts/run_pretrain.py \
    --max-steps 20000 --batch-size 128 --lr 3e-4 \
    --max-samples 30000 --num-workers 4 \
    --checkpoint-dir checkpoints/scaling_30k \
    --no-wandb --run-name "scaling_30k" \
    > logs/scaling_30k.txt 2>&1 &

wait
echo "All scaling runs complete"

# Evaluate each scaled checkpoint
for size in 5000 15000 30000; do
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_ttt.py \
        --checkpoint checkpoints/scaling_${size}k/best_pretrain.pt \
        --dataset corn --source m5 --target mp6 --property moisture \
        > logs/scaling_eval_${size}k.txt 2>&1
done

# Full model (61K) was already evaluated in E4
# Plot scaling curve
python3 -c "
from src.evaluation.visualization import plot_scaling_law
import json, glob

sizes = []
r2s = []
for size in [5000, 15000, 30000, 61420]:
    # Find corresponding TTT result
    files = glob.glob(f'experiments/ttt_*_{size}*.json') or glob.glob('experiments/ttt_zeroshot_*.json')
    if files:
        with open(files[-1]) as f:
            d = json.load(f)
        # Get best TTT R²
        best = max(v['r2'] for v in d.get('results',{}).values() if isinstance(v,dict) and 'r2' in v)
        sizes.append(size)
        r2s.append(best)
        print(f'  Corpus {size:>6,}: R²={best:.4f}')

if len(sizes) >= 2:
    plot_scaling_law(sizes, r2s, figures_dir='figures')
    print('Scaling law figure saved')
"
```


# ═══════════════════════════════════════════════════════════════
# PHASE 7: DOWNLOAD RESULTS (5 min, from LOCAL machine)
# ═══════════════════════════════════════════════════════════════

Run these from your LOCAL Mac terminal:

```bash
REMOTE=ubuntu@211.72.13.201
LOCAL_DIR=/Users/admin/Documents/GitHub/VS3L

# Download all experiment JSONs
scp -r $REMOTE:~/VS3L/experiments/ $LOCAL_DIR/experiments_gpu/

# Download all figures
scp -r $REMOTE:~/VS3L/figures/ $LOCAL_DIR/figures_gpu/

# Download best checkpoint (most important file!)
mkdir -p $LOCAL_DIR/checkpoints
scp $REMOTE:~/VS3L/checkpoints/best_pretrain.pt $LOCAL_DIR/checkpoints/

# Download final checkpoint
scp $REMOTE:~/VS3L/checkpoints/pretrain_final.pt $LOCAL_DIR/checkpoints/

# Download all logs
scp -r $REMOTE:~/VS3L/logs/ $LOCAL_DIR/logs_gpu/

# Download scaling checkpoints (if E7 was run)
scp -r $REMOTE:~/VS3L/checkpoints/scaling_*/ $LOCAL_DIR/checkpoints/ 2>/dev/null

echo "ALL RESULTS DOWNLOADED"
echo "Total cost: check Lambda dashboard"
echo "TERMINATE THE INSTANCE NOW to stop billing"
```

**⚠️ TERMINATE THE INSTANCE AFTER DOWNLOAD.**
At $0.117/hr, forgetting to terminate costs $2.81/day.


# ═══════════════════════════════════════════════════════════════
# TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════

## OOM (Out of Memory)
```bash
# Check current memory
nvidia-smi
# Reduce batch size
--batch-size 64  # or 32
# Should NOT happen — model is 6.5M params, 5090 has 32GB
```

## Loss NaN
```bash
# Lower learning rate
--lr 1e-4
# Check for bad data
python3 -c "
import h5py, numpy as np
with h5py.File('data/pretrain/spectral_corpus_v2.h5','r') as f:
    s = f['spectra'][:]
    print(f'NaN count: {np.isnan(s).sum()}')
    print(f'Inf count: {np.isinf(s).sum()}')
    print(f'Min: {s.min():.4f}, Max: {s.max():.4f}')
"
```

## Loss not decreasing
1. Verify data is loading: add `--log-every 1` for first 10 steps
2. Try different LR: `--lr 1e-4` or `--lr 5e-4`
3. Check gradient norms (add print to trainer.py temporarily)

## TTT R² always negative
1. Try more TTT steps: modify run_ttt.py to test [0, 10, 50, 100, 200, 500]
2. Try different adapt layers: change from "norm" to "all"
3. Try lower TTT LR: 1e-5 instead of 1e-4
4. Check reconstruction loss is actually decreasing during TTT:
```bash
# Add print to SpectralFM.test_time_train() temporarily:
# print(f"  TTT step {step}: recon_loss={recon_loss.item():.6f}")
```

## SSH dropped / training interrupted
```bash
# Use tmux to prevent this!
tmux new -s train
# Run training commands inside tmux
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t train
```

## Import errors
```bash
pip install --upgrade torch h5py PyWavelets POT einops scipy scikit-learn
# If mamba-related error: the code uses pure PyTorch Mamba, no mamba-ssm needed
```

## Disk space
```bash
df -h /home
# Corpus is 444MB, checkpoints ~100MB each, total ~5GB needed
# Instance has 209GB, plenty of space
```


# ═══════════════════════════════════════════════════════════════
# EXECUTION CHECKLIST
# ═══════════════════════════════════════════════════════════════
#
# Copy this checklist and check off items as you go:
#
# PHASE 1: Setup
# [ ] SSH connected
# [ ] nvidia-smi shows 4× RTX 5090
# [ ] Git repo cloned
# [ ] Data transferred (corpus + corn + tablet)
# [ ] Dependencies installed
# [ ] Data integrity verified (shapes match expected)
# [ ] Model builds on GPU (< 1 GB mem)
# [ ] 19/19 smoke tests pass
# [ ] 16/16 integration tests pass
# [ ] W&B logged in
#
# PHASE 2: Diagnostic
# [ ] Diagnostic complete in < 10 min
# [ ] Pretrain loss decreased to < 0.5
# [ ] TTT R² > 0.0 → PASS
# [ ] Results in experiments/diagnostic_results.json
#
# PHASE 3: Full Pretraining
# [ ] Sanity 100-step: loss decreasing, throughput good
# [ ] Full 50K-step training launched
# [ ] W&B dashboard shows convergence
# [ ] best_pretrain.pt saved
# [ ] Final val loss < 0.2
#
# PHASE 4: Evaluation
# [ ] Baselines complete (baselines_complete.json)
# [ ] E3 sample efficiency complete
# [ ] E4 TTT sweep complete
# [ ] Zero-shot R² value recorded: _________
# [ ] E1 pretrain ablation complete
# [ ] E2 LoRA vs full FT complete
# [ ] E5 cross-instrument complete
# [ ] E10 disentanglement complete
# [ ] E11 calibration complete
# [ ] E12 tablet complete
# [ ] TTT ablation complete
#
# PHASE 5: Figures
# [ ] All figures generated in figures/
# [ ] Results summary printed and reviewed
#
# PHASE 6: Conditional (if R² > 0.3)
# [ ] E7 scaling runs complete
# [ ] Scaling law figure generated
#
# PHASE 7: Download
# [ ] experiments/ downloaded
# [ ] figures/ downloaded
# [ ] checkpoints/best_pretrain.pt downloaded
# [ ] logs/ downloaded
# [ ] INSTANCE TERMINATED
#
# ═══════════════════════════════════════════════════════════════
