# Spektron Execution Guide

**Based on:** `/Users/admin/Brain/02-Software/Spektron/Training/Training-Plan.md`
**Updated:** 2026-03-04
**Status:** Ready to Execute

---

## Summary of Previous Attempt

Previous Phase 1 run (March 3-4, 2026):
- **Planned:** 12 trainings (4 backbones × 3 seeds)
- **Actual:** 3 trainings completed (~25%)
- **Root cause:** Broken orchestration script on remote (background job syntax)
- **Solution:** Use documented Training Plan phases with proper orchestration

---

## Correction from Earlier Misunderstandings

**What was incorrect:**
- I initially assumed 3-fold CV with 5 seeds = 75 trainings per experiment
- This was based on a misunderstanding of scope, not the documented plan
- The Training Plan clearly shows: 4 backbones × 3 seeds = **12 trainings** for Phase 1

**What is correct (per Training Plan):**
- Phase T: 8 test gates (T0-T8) before committing to ~180 GPU-hours
- Phase 1: E1 Benchmark = 4 backbones × 3 seeds × 50K steps
- Phase 2-5: Sequential experiments (E2, E3, E4, E5)
- Mamba is **permanently disabled** due to Blackwell SM120 hardware incompatibility
- Fold-id support exists in code but is optional (not used in main Phase 1)

---

## Code Fixes Applied

1. **E1 Benchmark** (`experiments/e1_benchmark.py`)
   - ✅ Fixed: Mamba now in `SKIP_BACKBONES = {"mamba"}` per Training Plan
   - ✅ Fold-id support remains (optional for advanced CV experiments)
   - ✅ Code is correct; just wasn't documented in main orchestration

2. **E2 Cross-Spectral** (`experiments/e2_cross_spectral.py`)
   - ✅ Fixed: Mamba now in `SKIP_BACKBONES = {"mamba"}`

3. **Orchestration** (New)
   - ✅ Created: `PHASE_EXECUTOR.sh` following Training Plan structure

---

## How to Execute (Remote GPU Instance)

### Prerequisites

Provision GPU instance per Training Plan T1:
- Hardware: 2× RTX 5060 Ti 16GB (Blackwell SM120)
- Connectivity: SSH access documented in `~/.claude/projects/-Users-admin/memory/spektron.md`
- Storage: 100GB+ available

### Step 1: Test Gates (Phase T) — Local CPU Only

```bash
cd /Users/admin/Work/Code/GitHub/Spektron

# T0: CPU backbone validation
PYTHONUNBUFFERED=1 python3 experiments/validate_backbones.py --device cpu
# Expected: All 4 backbones PASS (dlinoss, transformer, cnn1d, s4d)
```

**Gate T0 must pass before provisioning GPU instance.**

### Step 2: Provision and Sync to Remote

Follow Training Plan section T1-T2:

```bash
# Set SSH connection (from Vast.ai dashboard)
export SSH_PORT=<port>
export SSH_HOST=<host>

# Sync code to remote (excludes large data/checkpoints)
rsync -avz \
    --exclude='data/' --exclude='checkpoints/' --exclude='logs/' \
    --exclude='experiments/results/' --exclude='__pycache__/' \
    -e "ssh -p $SSH_PORT" \
    /Users/admin/Work/Code/GitHub/Spektron/ \
    root@$SSH_HOST:/root/Spektron/

# SSH to remote
ssh -p $SSH_PORT root@$SSH_HOST
cd /root/Spektron
```

### Step 3: Remote GPU Validation (Phase T, Gates T3-T8)

```bash
# T3: Data integrity
python3 -c "
import h5py
with h5py.File('data/raw/qm9s/qm9s_processed.h5', 'r') as f:
    print('n_molecules:', f.attrs['n_molecules'])
    print('has_ir:', f['has_ir'][:].sum())
    print('ir shape:', f['ir'].shape)
"
# Expected: n_molecules: 103991, has_ir: 103991, ir shape: (103991, 2048)

# T4: GPU backbone validation
PYTHONUNBUFFERED=1 python3 experiments/validate_backbones.py --device cuda
# Expected: All 4 backbones PASS

# T5: Mini run (D-LinOSS, 500 steps)
PYTHONUNBUFFERED=1 python3 experiments/e1_benchmark.py \
    --h5-path data/raw/qm9s/qm9s_processed.h5 \
    --backbone dlinoss \
    --seeds 1 \
    --max-steps 500 \
    --batch-size 16 \
    --num-workers 4 \
    --output experiments/results/e1_test_dlinoss.json \
    2>&1 | tee logs/test_dlinoss_$(date +%Y%m%d_%H%M%S).log

# Expected: Loss < 0.5 at step 500, no NaN, checkpoint created

# T6: All backbones (200 steps each)
PYTHONUNBUFFERED=1 python3 experiments/e1_benchmark.py \
    --h5-path data/raw/qm9s/qm9s_processed.h5 \
    --backbone all \
    --seeds 1 \
    --max-steps 200 \
    --batch-size 16 \
    --num-workers 4 \
    --output experiments/results/e1_test_all.json

# Expected: dlinoss, cnn1d, transformer, s4d PASS; mamba SKIP

# T7: Resume logic test
# Run 100 steps, then re-run same command with 100 steps again
# Expected: Second run skips (already completed)

# T8: Go/No-Go decision
# All gates must pass before proceeding to Phase 1
```

### Step 4: Phase 1 — E1 Architecture Benchmark

```bash
# Create tmux session (survives SSH disconnects)
tmux new-session -s e1_benchmark -d

# Attach and run (takes ~7.5 days)
tmux attach -t e1_benchmark
# Inside tmux:
cd /root/Spektron
PYTHONUNBUFFERED=1 python3 experiments/e1_benchmark.py \
    --h5-path data/raw/qm9s/qm9s_processed.h5 \
    --backbone all \
    --seeds 3 \
    --max-steps 50000 \
    --batch-size 16 \
    --num-workers 4 \
    --output experiments/results/e1_benchmark.json \
    2>&1 | tee logs/e1_benchmark_$(date +%Y%m%d_%H%M%S).log

# Detach: Ctrl+B, D
# Re-attach from another terminal: tmux attach -t e1_benchmark
```

**Timeline for Phase 1:**
| Backbone | Seeds | Steps | Per Run | Total | Wall Clock |
|----------|-------|-------|---------|-------|-----------|
| D-LinOSS | 3 | 50K | ~30.5h | ~91.5h | Days 1–4 |
| 1D CNN | 3 | 50K | ~7h | ~21h | Day 4–5 |
| Transformer | 3 | 50K | ~12h | ~36h | Days 5–7 |
| S4D | 3 | 50K | ~10h | ~30h | Days 7–8 |
| **Total** | | | | **~178.5h** | **~7.5 days** |

**Monitoring during Phase 1:**
```bash
# In a separate terminal
tail -f logs/e1_benchmark_*.log

# Check progress
python3 -c "
import json
with open('experiments/results/e1_benchmark.json') as f:
    runs = json.load(f).get('runs', [])
print(f'Completed: {len([r for r in runs if not r.get(\"error\")])} / 12')
for r in sorted(runs, key=lambda x: (x['backbone'], x['seed'])):
    status = '✓' if not r.get('error') else '✗'
    print(f\"{status} {r['backbone']:12} s{r['seed']} loss={r.get('best_val_loss', '—'):.4f}\")
"

# GPU usage
watch -n 5 nvidia-smi

# Disk space
df -h / && du -sh checkpoints/
```

### Step 5: Phase 2-5 (After Phase 1 Complete)

```bash
# Phase 2: Cross-Spectral Prediction
PYTHONUNBUFFERED=1 python3 experiments/e2_cross_spectral.py \
    --h5-path data/raw/qm9s/qm9s_processed.h5 \
    --seeds 3 \
    --output experiments/results/e2_cross_spectral.json

# Phase 3: Transfer Function Analysis (requires Phase 1 checkpoint)
checkpoint="checkpoints/e1_dlinoss_s42/best_pretrain.pt"
PYTHONUNBUFFERED=1 python3 experiments/e3_transfer_function.py \
    --h5-path data/raw/qm9s/qm9s_processed.h5 \
    --checkpoints "$checkpoint" \
    --output experiments/results/e3_transfer_function.json

# Phase 4: Calibration Transfer
PYTHONUNBUFFERED=1 python3 experiments/e4_calibration_transfer.py \
    --data-dir data \
    --seeds 3 \
    --checkpoint "$checkpoint"

# Phase 5: Ablation Study
PYTHONUNBUFFERED=1 python3 experiments/e1_ablations.py \
    --h5-path data/raw/qm9s/qm9s_processed.h5 \
    --seeds 3 \
    --max-steps 50000
```

---

## Recovery Procedures

### Training Interrupted

Resume logic automatically skips completed runs:

```bash
# Just re-run the same command
python3 experiments/e1_benchmark.py \
    --h5-path data/raw/qm9s/qm9s_processed.h5 \
    --backbone all \
    --seeds 3 \
    --max-steps 50000 \
    --output experiments/results/e1_benchmark.json

# Script checks existing results and skips (backbone, seed) pairs already completed
```

### CUDA OOM

The trainer automatically skips problematic runs and records error:

```bash
# Reduce batch size if needed
python3 experiments/e1_benchmark.py ... --batch-size 8
```

### Instance Preempted

Vast.ai instances may be preempted. Resume by:
1. Note new SSH endpoint
2. Re-sync code: `rsync ... /root/Spektron/`
3. Re-run same command
4. Resume logic handles everything

---

## Expected Results

**Phase 1 results file:** `experiments/results/e1_benchmark.json`

```json
{
  "runs": [
    {
      "backbone": "dlinoss",
      "seed": 42,
      "fold_id": 0,
      "best_val_loss": 0.0241,
      "final_train_loss": 0.0108,
      "steps": 50000,
      "train_time_sec": 29060,
      "backbone_params": 2106880
    },
    ...
  ]
}
```

**Expected performance (D-LinOSS):**
- Best val_loss: 0.05–0.15 (vs PLS baseline 0.5634 → ~4–10× better)
- Faster convergence than Transformer
- Stable across all 3 seeds

---

## Key Documentation Links

- **Full Training Plan:** `/Users/admin/Brain/02-Software/Spektron/Training/Training-Plan.md`
- **Experiment Details:**
  - E1: `/Users/admin/Brain/02-Software/Spektron/Experiments/E1-Architecture-Benchmark.md`
  - E2: `/Users/admin/Brain/02-Software/Spektron/Experiments/E2-Cross-Spectral.md`
  - E3: `/Users/admin/Brain/02-Software/Spektron/Experiments/E3-Transfer-Functions.md`
  - E4: `/Users/admin/Brain/02-Software/Spektron/Experiments/E4-Calibration-Transfer.md`
  - E5: `/Users/admin/Brain/02-Software/Spektron/Experiments/E5-Ablation-Study.md`
- **Quick Orchestration:** `PHASE_EXECUTOR.sh`

---

## Next Steps

1. ✅ Code fixed (Mamba disabled in E1 and E2)
2. ✅ Orchestration scripts created (`PHASE_EXECUTOR.sh`)
3. ⏳ **TODO:** Run local T0 test gates
4. ⏳ **TODO:** Provision GPU instance
5. ⏳ **TODO:** Run remote T1-T8 gates
6. ⏳ **TODO:** Launch Phase 1 (E1 benchmark)

**Status:** Ready to execute per Training Plan.
