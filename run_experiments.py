"""
Master experiment runner — runs everything end-to-end.

Steps:
  1. LP vs Grid comparison for all registered models
  2. Cross-model summary plots
  3. Markdown report

Usage:
    python run_experiments.py                        # all models
    python run_experiments.py --only qwen2-7b        # single model
    python run_experiments.py --report-only          # skip runs, just regenerate plots + report
    python run_experiments.py --verbose
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}]  {elapsed:.1f}s")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all FlexGen LP vs Grid experiments.")
    parser.add_argument("--only",        nargs="+", default=None,
                        help="Only run these model slugs")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip model runs; regenerate plots and report from existing results")
    parser.add_argument("--verbose",     action="store_true",
                        help="Pass --verbose to each model run")
    args = parser.parse_args()

    steps: list[tuple[list[str], str]] = []

    # Step 1 — model comparisons
    if not args.report_only:
        model_cmd = [sys.executable, "experiments/run_all_models.py"]
        if args.only:
            model_cmd += ["--only"] + args.only
        if args.verbose:
            model_cmd.append("--verbose")
        steps.append((model_cmd, "Step 1/3 — LP vs Grid for all models"))

    # Step 2 — cross-model summary plots
    steps.append(([sys.executable, "analysis/plot_model_summary.py"],
                  "Step 2/3 — Cross-model summary plots"))

    # Step 3 — markdown report
    steps.append(([sys.executable, "analysis/generate_report.py"],
                  "Step 3/3 — Generate report/cross_model_results.md"))

    failed = []
    wall_start = time.perf_counter()
    for cmd, label in steps:
        ok = _run(cmd, label)
        if not ok:
            failed.append(label)

    total = time.perf_counter() - wall_start
    print(f"\n{'='*70}")
    if failed:
        print(f"  FINISHED WITH ERRORS ({total:.0f}s)")
        for f in failed:
            print(f"    FAILED: {f}")
        sys.exit(1)
    else:
        print(f"  ALL DONE  ({total:.0f}s)")
        print()
        print("  Results:  experiments/results/model_tracking.json")
        print("  Report:   report/cross_model_results.md")
        print("  Plots:    analysis/plots/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
