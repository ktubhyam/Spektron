#!/bin/bash
# ============================================================
# Spekron: vast.ai GPU Training Setup
# Run this on the vast.ai instance after SSH-ing in
# ============================================================
set -e

echo "=========================================="
echo "  Spekron GPU Training Setup"
echo "=========================================="

# 1. Install dependencies
echo "[1/5] Installing Python dependencies..."
pip install --quiet torch numpy scipy scikit-learn h5py pandas \
    PyWavelets POT einops tqdm pyyaml matplotlib seaborn

# Optional: W&B (uncomment if you want experiment tracking)
# pip install wandb && wandb login

# 2. Verify GPU
echo "[2/5] Checking GPU..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  GPU {i}: {name} ({mem:.1f} GB)')
else:
    echo 'ERROR: No CUDA GPU detected!'
    exit 1
"

# 3. Verify data
echo "[3/5] Checking data..."
python3 -c "
import h5py
f = h5py.File('data/raw/qm9s/qm9s_processed.h5', 'r')
n = f.attrs['n_molecules']
has_rgn = f.attrs.get('has_rgn', False)
print(f'QM9S HDF5: {n:,} molecules, R(G,N)={has_rgn}')
print(f'IR: {f[\"ir\"].shape}, Raman: {f[\"raman\"].shape}')
f.close()
"

# 4. Quick smoke test (5 steps, ~10 seconds)
echo "[4/5] Smoke test (5 steps)..."
python3 experiments/pretrain_qm9s.py \
    --max-steps 5 \
    --batch-size 8 \
    --max-samples 64 \
    --no-wandb \
    --num-workers 0

echo "[5/5] Smoke test passed!"
echo ""
echo "=========================================="
echo "  Ready to train! Run:"
echo "  python3 experiments/pretrain_qm9s.py --no-wandb"
echo "=========================================="
