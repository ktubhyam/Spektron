# P4: GPU Execution Playbook
# Spektron — Lambda Cloud A10 (24GB)
# Estimated total GPU time: 6-10 hours

## PHASE 1: Instance Setup (15 min)

### 1.1 SSH into instance
```bash
ssh -i ~/.ssh/lambda_key ubuntu@<INSTANCE_IP>
```

### 1.2 Clone repo + transfer data
```bash
git clone https://github.com/<your-repo>/VS3L.git ~/VS3L
cd ~/VS3L

# Transfer corpus from local (run from LOCAL machine):
# scp -i ~/.ssh/lambda_key data/pretrain/spectral_corpus_v2.h5 ubuntu@<IP>:~/VS3L/data/pretrain/
# scp -i ~/.ssh/lambda_key -r data/processed/ ubuntu@<IP>:~/VS3L/data/processed/
```

### 1.3 Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install h5py wandb scipy scikit-learn matplotlib mamba-ssm causal-conv1d pywt
pip install einops  # if needed by any module

# Verify GPU
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_mem / 1e9, 'GB')"
```

### 1.4 W&B login (optional but recommended)
```bash
wandb login <YOUR_API_KEY>
```

### 1.5 Verify data
```bash
python3 -c "
import h5py, numpy as np
with h5py.File('data/pretrain/spectral_corpus_v2.h5','r') as f:
    print('Corpus:', {k: f[k].shape for k in f.keys()})
for inst in ['m5','mp5','mp6']:
    a = np.load(f'data/processed/corn/{inst}_spectra.npy')
    print(f'Corn {inst}: {a.shape}')
print('Data OK')
"
```

---

## PHASE 2: Diagnostic (20 min) — RUN FIRST!

**Purpose**: Validate architecture works before investing 2-6 hours in full pretraining.

```bash
cd ~/VS3L
python3 scripts/run_diagnostic.py --device cuda --max-steps 2000
```

**Expected output**: 
- Pretrain loss drops from ~1.0 to ~0.3-0.5
- Zero-shot TTT R² > 0.0 at some step count

**Decision gate**:
| Result | Action |
|--------|--------|
| ✅ TTT R² > 0.0 | Proceed to Phase 3 |
| 🎉 TTT R² > 0.3 | Excellent — proceed with confidence |
| ⚠️ TTT R² ≤ 0.0 | Debug (see Troubleshooting below) |

---

## PHASE 3: Full Pretraining (2-6 hours)

### 3.1 Sanity check (5 min)
```bash
python3 scripts/run_pretrain.py \
    --max-steps 100 --batch-size 64 --num-workers 4 \
    --log-every 10 --save-every 50 --val-every 50 \
    --no-wandb --run-name "sanity_100"
```

Check:
- Loss decreasing
- Throughput > 100 samp/s on A10
- No OOM (if OOM, reduce batch to 32)
- Val loss computed at step 50

### 3.2 Full training
```bash
python3 scripts/run_pretrain.py \
    --max-steps 50000 --batch-size 64 --lr 3e-4 \
    --num-workers 4 --log-every 100 --save-every 5000 --val-every 1000 \
    --wandb --run-name "pretrain_v2_50k"
```

**Monitor via W&B** or:
```bash
# In another terminal
tail -f logs/pretrain_v2_50k/metrics.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if 'val_loss' in d:
        print(f\"Step {d['step']:>6d} | Train {d.get('train_loss',0):.4f} | Val {d['val_loss']:.4f} | {d.get('throughput',0):.0f} samp/s\")
"
```

**Expected**:
- Step 0: loss ~1.0
- Step 5K: loss ~0.3-0.5
- Step 50K: loss ~0.1-0.2 (converged)
- Time: ~2-4 hours on A10

### 3.3 Verify checkpoint
```bash
python3 -c "
import torch
ckpt = torch.load('checkpoints/best_pretrain.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Model keys:', len(ckpt['model_state_dict']))
print('Step:', ckpt.get('step'))
print('Val loss:', ckpt.get('val_loss'))
"
```

---

## PHASE 4: Evaluation Suite (2-4 hours)

### 4.0 Baselines (CPU, ~15 min — can run in parallel)
```bash
python3 scripts/run_baselines_complete.py --n-seeds 5
```
Output: `experiments/baselines_complete.json`

### 4.1 E3: Sample Efficiency (THE KEY FIGURE) — ~45 min
```bash
python3 scripts/run_finetune.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --all-properties --sweep --n-seeds 5
```

### 4.2 E4: Zero-Shot TTT (THE KEY RESULT) — ~30 min
```bash
python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --sweep --n-seeds 5
```

**CRITICAL**: Check zero-shot R² immediately:
```bash
python3 -c "
import json
with open('experiments/ttt_zeroshot_corn_*.json') as f:
    d = json.load(f)
r2 = d['results'].get('0', d['results'].get(0, {}))
print(f'ZERO-SHOT TTT R² = {r2}')
" 2>/dev/null || echo "Check experiments/ for TTT results"
```

| Zero-Shot R² | Paper Tier | Action |
|-------------|------------|--------|
| R² > 0.5 | Nature MI | Run E7 scaling + full suite |
| R² > 0.3 | Anal. Chem. (strong) | Run E1-E5, E10-E12 |
| R² > 0.1 | Anal. Chem. (viable) | Focus on few-shot story |
| R² < 0.1 | Pivot | Lead with LoRA transfer, TTT as future work |

### 4.3 E1: Pretraining Ablation — ~20 min
```bash
python3 scripts/run_experiments.py --experiment E1 --checkpoint checkpoints/best_pretrain.pt
```

### 4.4 E2: LoRA vs Full FT — ~20 min
```bash
python3 scripts/run_experiments.py --experiment E2 --checkpoint checkpoints/best_pretrain.pt
```

### 4.5 E5: Cross-Instrument — ~15 min
```bash
python3 scripts/run_experiments.py --experiment E5 --checkpoint checkpoints/best_pretrain.pt
```

### 4.6 E10: Disentanglement Viz — ~5 min
```bash
python3 scripts/run_experiments.py --experiment E10 --checkpoint checkpoints/best_pretrain.pt
```

### 4.7 E11: Uncertainty Calibration — ~10 min
```bash
python3 scripts/run_experiments.py --experiment E11 --checkpoint checkpoints/best_pretrain.pt
```

### 4.8 E12: Tablet Validation — ~15 min
```bash
python3 scripts/run_experiments.py --experiment E12 --checkpoint checkpoints/best_pretrain.pt
```

### 4.9 TTT Ablation — ~30 min
```bash
python3 scripts/run_ttt.py \
    --checkpoint checkpoints/best_pretrain.pt \
    --dataset corn --ablation
```

---

## PHASE 5: Figure Generation (5 min, CPU)

```bash
python3 -c "
from src.evaluation.visualization import generate_all_figures_from_experiments
generate_all_figures_from_experiments('experiments', 'figures')
"
```

Check `figures/` for:
- `sample_efficiency.pdf` — Fig 1 (KEY)
- `ttt_steps.pdf` — Fig 2
- `tsne_disentanglement.pdf` — Fig 3
- `calibration.pdf` — Fig 4

---

## PHASE 6: Download Results (5 min)

**From LOCAL machine**:
```bash
INSTANCE=ubuntu@<INSTANCE_IP>
KEY=~/.ssh/lambda_key

# All experiment JSONs
scp -i $KEY -r $INSTANCE:~/VS3L/experiments/ ./experiments_gpu/

# All figures
scp -i $KEY -r $INSTANCE:~/VS3L/figures/ ./figures_gpu/

# Best checkpoint
scp -i $KEY $INSTANCE:~/VS3L/checkpoints/best_pretrain.pt ./checkpoints/

# Logs
scp -i $KEY -r $INSTANCE:~/VS3L/logs/ ./logs_gpu/

# W&B should already be synced
```

**Then terminate the instance to stop billing.**

---

## PHASE 7: Conditional Experiments (if zero-shot R² > 0.3)

These are Nature MI-tier experiments, only worth running if core results are strong.

### E7: Scaling Law (requires multiple pretraining runs)
```bash
# Subsample corpus to different sizes and retrain
for size in 15000 30000 61420; do
    python3 scripts/run_pretrain.py \
        --max-steps 20000 --batch-size 64 --lr 3e-4 \
        --max-samples $size \
        --run-name "scaling_${size}" --no-wandb
    
    python3 scripts/run_ttt.py \
        --checkpoint checkpoints/best_scaling_${size}.pt \
        --dataset corn --source m5 --target mp6 --property moisture
done
```

### E8: Physics Loss Ablation (requires modified pretraining)
For each loss component (Beer-Lambert, smoothness, OT, contrastive), retrain without it:
```bash
for ablation in no_beer_lambert no_smoothness no_ot no_contrastive; do
    python3 scripts/run_pretrain.py \
        --max-steps 20000 --batch-size 64 \
        --ablation $ablation \
        --run-name "ablation_${ablation}" --no-wandb
done
```
**Note**: Requires adding `--ablation` flag to run_pretrain.py and trainer.

### E9: Architecture Ablation
Similar to E8 but modifies config: no_mamba, no_moe, no_transformer, no_vib.

---

## Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size
--batch-size 32  # or 16 for A10 24GB

# Check memory usage
nvidia-smi
```

### Loss not decreasing
1. Check learning rate: try 1e-4 instead of 3e-4
2. Check data loading: `python3 -c "from src.data.datasets import PretrainHDF5Dataset; d=PretrainHDF5Dataset('data/pretrain/spectral_corpus_v2.h5'); print(d[0][0].shape)"`
3. Check masking: ensure mask ratio is 0.15-0.30

### TTT R² negative
1. Try more steps: `--ttt-steps 100 200 500`
2. Try different adapt layers: `--adapt-layers all`
3. Try lower LR: `--ttt-lr 1e-5`
4. Check if reconstruction loss decreases during TTT

### Import errors on Lambda
```bash
pip install mamba-ssm causal-conv1d  # Mamba needs CUDA
pip install pywt  # PyWavelets for DWT embedding
```

---

## Time Budget Summary

| Phase | Time | GPU? |
|-------|------|------|
| Setup | 15 min | No |
| Diagnostic | 20 min | Yes |
| Full pretrain | 2-4 hours | Yes |
| Eval suite | 2-3 hours | Yes |
| Figures | 5 min | No |
| Download | 5 min | No |
| **Total** | **~5-8 hours** | |

**Cost estimate**: A10 at ~$0.75/hr × 8 hours = ~$6
