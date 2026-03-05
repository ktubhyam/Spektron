#!/bin/bash
################################################################################
# SPEKTRON PHASE EXECUTOR
#
# Orchestrates complete training pipeline following documented Training Plan:
# - Phase T: Test gates (T0-T8) before committing to large compute
# - Phase 1: E1 Architecture Benchmark (4 backbones × 3 seeds = 12 trainings)
# - Phase 2: E2 Cross-Spectral Prediction
# - Phase 3: E3 Transfer Function Analysis
# - Phase 4: E4 Calibration Transfer
# - Phase 5: E5 Ablation Study
#
# Usage:
#   bash PHASE_EXECUTOR.sh --phase t   # Run test gates T0-T8
#   bash PHASE_EXECUTOR.sh --phase 1   # Run Phase 1 (E1 benchmark)
#   bash PHASE_EXECUTOR.sh --all       # Run all phases sequentially
#
# Source: /Users/admin/Brain/02-Software/Spektron/Training/Training-Plan.md
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PYTHON="${PYTHON:-python3}"
H5_PATH="${H5_PATH:-data/raw/qm9s/qm9s_processed.h5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-50000}"

# Logging
LOG_DIR="logs/phase_executor"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

# ==============================================================================
# PHASE T: Test Gates
# ==============================================================================

phase_t() {
    log_info "========== PHASE T: TEST GATES =========="
    log_info "Purpose: Validate full stack before committing to ~180 GPU-hours"

    # T0: Local CPU smoke test
    log_info "\n[T0] Local CPU backbone validation..."
    if [ -f "experiments/validate_backbones.py" ]; then
        PYTHONUNBUFFERED=1 "$PYTHON" experiments/validate_backbones.py --device cpu 2>&1 | tee -a "$LOG_FILE"
        log_success "T0: CPU validation passed"
    else
        log_warn "T0: validate_backbones.py not found, skipping"
    fi

    # Note: T1-T8 require remote GPU instance
    log_warn "T1-T8 require GPU instance on Vast.ai. See Training-Plan.md for instructions."
    log_info "After provisioning, run:"
    log_info "  ssh -p <PORT> root@<HOST> 'cd /root/Spektron && bash PHASE_EXECUTOR.sh --phase 1'"
}

# ==============================================================================
# PHASE 1: E1 Architecture Benchmark
# ==============================================================================

phase_1() {
    log_info "========== PHASE 1: E1 ARCHITECTURE BENCHMARK =========="
    log_info "Goal: 4 backbones × 3 seeds × 50K steps"
    log_info "Compute: ~178 GPU-hours (~7.5 days on 2× RTX 5060 Ti)"
    log_info "Backbones: D-LinOSS → CNN → Transformer → S4D (Mamba skipped)"

    # Use tmux if available
    if command -v tmux &> /dev/null; then
        tmux_session="e1_benchmark_$(date +%s)"
        log_info "Starting in tmux session: $tmux_session"
        tmux new-session -d -s "$tmux_session" -c "$SCRIPT_DIR" \
            "PYTHONUNBUFFERED=1 $PYTHON experiments/e1_benchmark.py \
                --h5-path '$H5_PATH' \
                --backbone all \
                --seeds 3 \
                --max-steps $MAX_STEPS \
                --batch-size $BATCH_SIZE \
                --num-workers $NUM_WORKERS \
                --output experiments/results/e1_benchmark.json \
                2>&1 | tee logs/e1_benchmark_\$(date +%Y%m%d_%H%M%S).log"

        log_success "Phase 1 launched in tmux session: $tmux_session"
        log_info "Attach with: tmux attach -t $tmux_session"
        log_info "Detach with: Ctrl+B, D"
    else
        # Run directly
        log_info "Running Phase 1 directly (no tmux)..."
        PYTHONUNBUFFERED=1 "$PYTHON" experiments/e1_benchmark.py \
            --h5-path "$H5_PATH" \
            --backbone all \
            --seeds 3 \
            --max-steps $MAX_STEPS \
            --batch-size $BATCH_SIZE \
            --num-workers $NUM_WORKERS \
            --output experiments/results/e1_benchmark.json \
            2>&1 | tee "logs/e1_benchmark_$(date +%Y%m%d_%H%M%S).log"

        log_success "Phase 1 completed"
    fi
}

# ==============================================================================
# PHASE 2: E2 Cross-Spectral Prediction
# ==============================================================================

phase_2() {
    log_info "========== PHASE 2: E2 CROSS-SPECTRAL PREDICTION =========="
    log_info "Goal: Predict Raman from IR and vice versa"
    log_info "Architectures: 5 backbones × 2 directions × 3 seeds"

    PYTHONUNBUFFERED=1 "$PYTHON" experiments/e2_cross_spectral.py \
        --h5-path "$H5_PATH" \
        --seeds 3 \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --output experiments/results/e2_cross_spectral.json \
        2>&1 | tee "logs/e2_cross_spectral_$(date +%Y%m%d_%H%M%S).log"

    log_success "Phase 2 completed"
}

# ==============================================================================
# PHASE 3: E3 Transfer Function Analysis
# ==============================================================================

phase_3() {
    log_info "========== PHASE 3: E3 TRANSFER FUNCTION ANALYSIS =========="
    log_info "Goal: Extract and visualize H(z) transfer functions"
    log_info "Requires: E1 checkpoints (especially e1_dlinoss_s42/best_pretrain.pt)"

    # Find best checkpoint
    checkpoint=$(find checkpoints -name "best_pretrain.pt" -path "*dlinoss*" | head -1)
    if [ -z "$checkpoint" ]; then
        log_error "No D-LinOSS checkpoint found for E3"
        return 1
    fi

    log_info "Using checkpoint: $checkpoint"

    PYTHONUNBUFFERED=1 "$PYTHON" experiments/e3_transfer_function.py \
        --h5-path "$H5_PATH" \
        --checkpoints "$checkpoint" \
        --compare-random \
        --output experiments/results/e3_transfer_function.json \
        2>&1 | tee "logs/e3_transfer_function_$(date +%Y%m%d_%H%M%S).log"

    log_success "Phase 3 completed"
}

# ==============================================================================
# PHASE 4: E4 Calibration Transfer
# ==============================================================================

phase_4() {
    log_info "========== PHASE 4: E4 CALIBRATION TRANSFER =========="
    log_info "Goal: Fine-tune on experimental Corn/Tablet datasets"
    log_info "Requires: E1 checkpoints"

    # Check if checkpoint exists
    checkpoint=$(find checkpoints -name "best_pretrain.pt" | head -1)
    if [ -z "$checkpoint" ]; then
        log_warn "No checkpoint found; E4 will train from scratch"
    fi

    PYTHONUNBUFFERED=1 "$PYTHON" experiments/e4_calibration_transfer.py \
        --data-dir data \
        --seeds 3 \
        --batch-size $BATCH_SIZE \
        $([ -n "$checkpoint" ] && echo "--checkpoint $checkpoint") \
        --output experiments/results/e4_calibration_transfer.json \
        2>&1 | tee "logs/e4_calibration_transfer_$(date +%Y%m%d_%H%M%S).log"

    log_success "Phase 4 completed"
}

# ==============================================================================
# PHASE 5: E5 Ablation Study
# ==============================================================================

phase_5() {
    log_info "========== PHASE 5: E5 ABLATION STUDY =========="
    log_info "Goal: D-LinOSS ablation variations (damping, depth)"

    PYTHONUNBUFFERED=1 "$PYTHON" experiments/e1_ablations.py \
        --h5-path "$H5_PATH" \
        --seeds 3 \
        --max-steps $MAX_STEPS \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --output experiments/results/e5_ablations.json \
        2>&1 | tee "logs/e5_ablations_$(date +%Y%m%d_%H%M%S).log"

    log_success "Phase 5 completed"
}

# ==============================================================================
# PHASE 6 & 7: Post-experiment (not in this script)
# ==============================================================================

# ==============================================================================
# Main
# ==============================================================================

main() {
    phase="${1:-t}"

    case "$phase" in
        t|test)
            phase_t
            ;;
        1|e1)
            phase_1
            ;;
        2|e2)
            phase_2
            ;;
        3|e3)
            phase_3
            ;;
        4|e4)
            phase_4
            ;;
        5|e5)
            phase_5
            ;;
        all)
            log_info "Running all phases sequentially..."
            phase_t
            phase_1
            phase_2
            phase_3
            phase_4
            phase_5
            log_success "All phases completed!"
            ;;
        *)
            log_error "Unknown phase: $phase"
            echo "Usage: $0 [t|1|2|3|4|5|all]"
            exit 1
            ;;
    esac
}

main "$@"
