"""
Run LP vs Grid comparison for every model and update model_tracking.json.

Usage:
    python experiments/run_all_models.py            # all models
    python experiments/run_all_models.py --only qwen2-7b mistral-7b-v0-1
    python experiments/run_all_models.py --verbose
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------
MODELS = [
    {
        "slug":        "smollm2-135m-instruct",
        "model":       "models/smollm2-135m-instruct",
        "sim_gpu_gb":  None,    # 0.067 GB int4 -- fits in any GPU
        "sim_ram_gb":  None,
        "description": "SmoLLM2-135M  (no offloading needed)",
    },
    {
        "slug":        "tinyllama-1.1b-chat",
        "model":       "models/tinyllama-1.1b-chat",
        "sim_gpu_gb":  0.3,     # 0.55 GB int4 -- constrain to show offloading
        "sim_ram_gb":  8.0,
        "description": "TinyLLaMA-1.1B  (sim 0.3 GB GPU)",
    },
    {
        "slug":        "qwen2-7b",
        "model":       "Qwen/Qwen2-7B",
        "sim_gpu_gb":  2.0,     # 3.04 GB int4 -- needs offloading
        "sim_ram_gb":  16.0,
        "description": "Qwen2-7B  (sim 2.0 GB GPU)",
    },
    {
        "slug":        "mistral-7b-v0-1",
        "model":       "mistralai/Mistral-7B-v0.1",
        "sim_gpu_gb":  2.0,     # 3.25 GB int4 -- needs offloading
        "sim_ram_gb":  16.0,
        "description": "Mistral-7B-v0.1  (sim 2.0 GB GPU)",
    },
]


def run_model(cfg: dict, verbose: bool) -> bool:
    print(f"\n{'='*68}")
    print(f"  {cfg['description']}")
    print(f"{'='*68}")

    cmd = [
        sys.executable, "experiments/compare_lp_vs_grid.py",
        "--run-both",
        "--model",      cfg["model"],
        "--model-slug", cfg["slug"],
    ]
    if cfg.get("sim_gpu_gb") is not None:
        cmd += ["--sim-gpu-gb", str(cfg["sim_gpu_gb"])]
    if cfg.get("sim_ram_gb") is not None:
        cmd += ["--sim-ram-gb", str(cfg["sim_ram_gb"])]
    if verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  FAILED for {cfg['slug']} (exit {result.returncode})")
        return False
    return True


def print_summary() -> None:
    tracking_path = ROOT / "experiments" / "results" / "model_tracking.json"
    if not tracking_path.exists():
        return
    import json
    data = json.loads(tracking_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 80)
    print("CROSS-MODEL SUMMARY")
    print("=" * 80)
    print(f"{'Model':<30} {'LP (ms)':<12} {'Grid (ms)':<12} {'LP better':<12} {'Grid time(s)'}")
    print("-" * 80)
    for slug, m in data.items():
        gpu_note = f" [sim {m['sim_gpu_gb']}GB]" if m.get("sim_gpu_gb") else ""
        name = slug + gpu_note
        print(f"{name:<30} {m['lp_latency_ms']:<12.2f} {m['grid_latency_ms']:<12.2f} "
              f"{m['lp_better_by_pct']:+.1f}%      {m['grid_search_time_s']}")
    print("=" * 80)
    print(f"Full tracking: {tracking_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LP vs Grid for all registered models.")
    parser.add_argument("--only",    nargs="+", default=None,
                        help="Run only these slugs (e.g. --only qwen2-7b mistral-7b-v0-1)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    targets = MODELS
    if args.only:
        targets = [m for m in MODELS if m["slug"] in args.only]
        missing = set(args.only) - {m["slug"] for m in targets}
        if missing:
            print(f"Unknown slugs: {missing}")
            print(f"Available: {[m['slug'] for m in MODELS]}")
            sys.exit(1)

    failed = []
    for cfg in targets:
        ok = run_model(cfg, args.verbose)
        if not ok:
            failed.append(cfg["slug"])

    print_summary()

    if failed:
        print(f"\nFailed models: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
