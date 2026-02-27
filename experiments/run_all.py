#!/usr/bin/env python3
"""
Master Experiment Runner

Runs all 4 experiments sequentially:
  E1: Architecture Benchmark
  E2: Cross-Spectral Prediction
  E3: Transfer Function Analysis
  E4: Calibration Transfer

Usage:
    # Full run (GPU required)
    python experiments/run_all.py --h5-path data/raw/qm9s/qm9s_processed.h5

    # Quick smoke test (CPU ok)
    python experiments/run_all.py --h5-path ... --quick

    # Run specific experiments
    python experiments/run_all.py --h5-path ... --experiments e1 e2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import subprocess
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def run_experiment(name: str, cmd: list, dry_run: bool = False):
    """Run a single experiment as a subprocess."""
    log.info(f"\n{'='*70}")
    log.info(f"  STARTING: {name}")
    log.info(f"  Command: {' '.join(cmd)}")
    log.info(f"{'='*70}\n")

    if dry_run:
        log.info("  [DRY RUN] Skipping execution")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode == 0:
        log.info(f"\n  {name} completed in {elapsed:.0f}s")
        return True
    else:
        log.error(f"\n  {name} FAILED (return code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run All Experiments")
    parser.add_argument("--h5-path", type=str, required=True,
                        help="Path to QM9S HDF5 file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pretrained checkpoint for E3/E4")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory for E4")
    parser.add_argument("--experiments", nargs="+",
                        default=["e1", "e2", "e3", "e4"],
                        help="Which experiments to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer steps/seeds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    python = sys.executable
    quick_flag = ["--quick"] if args.quick else []

    results = {}
    t_total = time.time()

    # E1: Architecture Benchmark
    if "e1" in args.experiments:
        cmd = [
            python, "experiments/e1_benchmark.py",
            "--h5-path", args.h5_path,
            "--seeds", str(args.seeds),
            "--num-workers", str(args.num_workers),
        ] + quick_flag
        results["e1"] = run_experiment("E1: Architecture Benchmark", cmd,
                                       args.dry_run)

    # E2: Cross-Spectral Prediction
    if "e2" in args.experiments:
        cmd = [
            python, "experiments/e2_cross_spectral.py",
            "--h5-path", args.h5_path,
            "--seeds", str(args.seeds),
            "--num-workers", str(args.num_workers),
        ] + quick_flag
        results["e2"] = run_experiment("E2: Cross-Spectral Prediction", cmd,
                                       args.dry_run)

    # E3: Transfer Function Analysis (requires checkpoint)
    if "e3" in args.experiments:
        checkpoint = args.checkpoint
        if checkpoint is None:
            # Try to find best checkpoint
            candidates = list(Path("checkpoints").glob("**/best_pretrain.pt"))
            if candidates:
                checkpoint = str(candidates[0])
            else:
                log.warning("No checkpoint found for E3 — skipping")
                results["e3"] = False

        if checkpoint:
            cmd = [
                python, "experiments/e3_transfer_function.py",
                "--checkpoint", checkpoint,
                "--compare-random",
            ]
            results["e3"] = run_experiment("E3: Transfer Function Analysis",
                                           cmd, args.dry_run)

    # E4: Calibration Transfer (requires checkpoint + corn/tablet data)
    if "e4" in args.experiments:
        checkpoint = args.checkpoint
        if checkpoint is None:
            candidates = list(Path("checkpoints").glob("**/best_pretrain.pt"))
            if candidates:
                checkpoint = str(candidates[0])

        cmd = [
            python, "experiments/e4_calibration_transfer.py",
            "--data-dir", args.data_dir,
            "--seeds", str(args.seeds),
        ] + quick_flag
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        results["e4"] = run_experiment("E4: Calibration Transfer", cmd,
                                       args.dry_run)

    # Summary
    elapsed_total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)")
    print(f"{'='*70}")
    for exp, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {exp.upper()}: {status}")
    print()

    n_failed = sum(1 for v in results.values() if not v)
    if n_failed > 0:
        log.error(f"{n_failed} experiment(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
