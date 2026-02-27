#!/usr/bin/env python3
"""
Spektron: Master Experiment Runner

Runs all experiments (E1-E6) from a single pretrained checkpoint and
aggregates results for the paper.

Usage:
    # Run everything
    python experiments/run_all_experiments.py --checkpoint checkpoints/qm9s_dlinoss/best_pretrain.pt

    # Run specific experiments
    python experiments/run_all_experiments.py --checkpoint ... --experiments e1,e5,e6

    # Quick test (small subset)
    python experiments/run_all_experiments.py --checkpoint ... --quick

Experiment Map:
    E1: Symmetry-stratified evaluation (Theorem 1 validation)
    E2: Modal complementarity (Theorem 2 validation)
    E3: Confusable set analysis (standalone, no checkpoint needed)
    E4: Jacobian rank analysis (standalone, no checkpoint needed)
    E5: Calibration transfer with sample efficiency curves
    E6: Uncertainty quantification via conformal prediction
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import subprocess
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def run_experiment(name, cmd, results_dir):
    """Run a single experiment as a subprocess."""
    log.info(f"\n{'='*70}")
    log.info(f"  STARTING: {name}")
    log.info(f"  Command: {' '.join(cmd)}")
    log.info(f"{'='*70}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,  # 2h timeout
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            log.info(f"  {name}: COMPLETED in {elapsed:.0f}s")
        else:
            log.error(f"  {name}: FAILED (exit code {result.returncode})")
            log.error(f"  stderr: {result.stderr[-500:]}")

        return {
            "name": name,
            "status": "success" if result.returncode == 0 else "failed",
            "elapsed_sec": elapsed,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1000:] if result.stdout else "",
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        log.error(f"  {name}: TIMEOUT (>2h)")
        return {"name": name, "status": "timeout", "elapsed_sec": 7200}
    except Exception as e:
        log.error(f"  {name}: ERROR - {e}")
        return {"name": name, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run all Spektron experiments")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--h5-path", type=str,
                        default="data/raw/qm9s/qm9s_processed.h5")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--rgn-path", type=str,
                        default="experiments/results/qm9_rgn_data.npz")
    parser.add_argument("--pg-path", type=str,
                        default="experiments/results/qm9_point_groups.json")
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--results-dir", type=str, default="experiments/results")
    parser.add_argument("--experiments", type=str, default="all",
                        help="Comma-separated list: e1,e2,e3,e4,e5,e6 or 'all'")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (fewer seeds, samples)")
    parser.add_argument("--use-ttt", action="store_true",
                        help="Include TTT experiments in E5")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Parse experiments
    if args.experiments == "all":
        experiments = ["e1", "e2", "e3", "e4", "e5", "e6"]
    else:
        experiments = [e.strip().lower() for e in args.experiments.split(",")]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    device_args = ["--device", args.device] if args.device else []
    quick_args = []
    if args.quick:
        quick_args = ["--n-seeds", "2", "--n-mask-trials", "2"]

    run_results = []
    t_start = time.time()

    # ================================================================
    # E1: Symmetry Stratification
    # ================================================================
    if "e1" in experiments:
        cmd = [
            python, "experiments/e1_symmetry_stratification.py",
            "--checkpoint", args.checkpoint,
            "--h5-path", args.h5_path,
            "--rgn-path", args.rgn_path,
            "--pg-path", args.pg_path,
            "--figures-dir", str(figures_dir),
            "--output", str(results_dir / "e1_symmetry_stratification.json"),
        ] + device_args
        if args.quick:
            cmd += ["--n-mask-trials", "2", "--batch-size", "128"]
        run_results.append(run_experiment("E1: Symmetry Stratification", cmd, results_dir))

    # ================================================================
    # E2: Modal Complementarity
    # ================================================================
    if "e2" in experiments:
        cmd = [
            python, "experiments/e2_modal_complementarity.py",
            "--checkpoint", args.checkpoint,
            "--h5-path", args.h5_path,
            "--rgn-path", args.rgn_path,
            "--figures-dir", str(figures_dir),
            "--output", str(results_dir / "e2_modal_complementarity.json"),
        ] + device_args
        if args.quick:
            cmd += ["--n-mask-trials", "2", "--batch-size", "128"]
        run_results.append(run_experiment("E2: Modal Complementarity", cmd, results_dir))

    # ================================================================
    # E3: Confusable Pairs (standalone — no checkpoint needed)
    # ================================================================
    if "e3" in experiments:
        cmd = [python, "experiments/symmetry_confusability.py"]
        run_results.append(run_experiment("E3: Confusable Pairs", cmd, results_dir))

    # ================================================================
    # E4: Jacobian Rank (standalone — no checkpoint needed)
    # ================================================================
    if "e4" in experiments:
        cmd = [python, "experiments/jacobian_rank.py"]
        run_results.append(run_experiment("E4: Jacobian Rank", cmd, results_dir))

    # ================================================================
    # E5: Calibration Transfer
    # ================================================================
    if "e5" in experiments:
        n_seeds = "2" if args.quick else "5"
        n_transfer = "5,10,20" if args.quick else "5,10,20,30,50"
        cmd = [
            python, "experiments/e5_calibration_transfer.py",
            "--checkpoint", args.checkpoint,
            "--data-dir", args.data_dir,
            "--n-seeds", n_seeds,
            "--n-transfer", n_transfer,
            "--dataset", "both",
            "--figures-dir", str(figures_dir),
            "--output", str(results_dir / "e5_calibration_transfer.json"),
        ] + device_args
        if args.use_ttt:
            cmd.append("--use-ttt")
        run_results.append(run_experiment("E5: Calibration Transfer", cmd, results_dir))

    # ================================================================
    # E6: Uncertainty Quantification
    # ================================================================
    if "e6" in experiments:
        mc = "10" if args.quick else "30"
        cmd = [
            python, "experiments/e6_uncertainty_quantification.py",
            "--checkpoint", args.checkpoint,
            "--data-dir", args.data_dir,
            "--mc-samples", mc,
            "--dataset", "corn",
            "--figures-dir", str(figures_dir),
            "--output", str(results_dir / "e6_uncertainty_quantification.json"),
        ] + device_args
        run_results.append(run_experiment("E6: Uncertainty Quantification", cmd, results_dir))

    # ================================================================
    # Summary
    # ================================================================
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print()

    for r in run_results:
        status_icon = {
            "success": "[OK]",
            "failed": "[FAIL]",
            "timeout": "[TIMEOUT]",
            "error": "[ERROR]",
        }.get(r["status"], "[?]")
        elapsed = r.get("elapsed_sec", 0)
        print(f"  {status_icon} {r['name']:<40} {elapsed:>6.0f}s")

    print()

    # Collect all result files
    collected = {}
    for name in ["e1_symmetry_stratification", "e2_modal_complementarity",
                  "e5_calibration_transfer", "e6_uncertainty_quantification"]:
        path = results_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                collected[name] = json.load(f)

    # Save aggregated results
    agg_path = results_dir / "all_experiments.json"
    with open(agg_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checkpoint": args.checkpoint,
            "total_time_sec": total_time,
            "run_summary": run_results,
            "results": collected,
        }, f, indent=2)
    log.info(f"Aggregated results saved to {agg_path}")

    # List generated figures
    fig_files = sorted(figures_dir.glob("e*.pdf"))
    if fig_files:
        print(f"\nGenerated {len(fig_files)} figures:")
        for f in fig_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
