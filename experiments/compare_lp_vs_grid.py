"""
Compare FlexGen LP policy search vs Grid search baseline.

Results and plots are saved per-model under:
  experiments/results/<model-slug>/
  analysis/plots/<model-slug>/

A cross-model tracking file is kept at:
  experiments/results/model_tracking.json

Usage (run both searches and compare in one command):
    python experiments/compare_lp_vs_grid.py --run-both --model Qwen/Qwen2-7B --sim-gpu-gb 2.0

Usage (compare existing result JSONs):
    python experiments/compare_lp_vs_grid.py --lp <path> --grid <path> --model Qwen/Qwen2-7B
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = ROOT / "experiments" / "results"
PLOTS_ROOT   = ROOT / "analysis" / "plots"
TRACKING_FILE = RESULTS_ROOT / "model_tracking.json"


# -----------------------------------------------------------------------------
# Model slug
# -----------------------------------------------------------------------------

def model_slug(model_id: str) -> str:
    """Convert any model ID or local path to a filesystem-safe slug."""
    name = Path(model_id).name if Path(model_id).exists() else model_id.split("/")[-1]
    return re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latency_ms(p: dict) -> float:
    return p["objective"]["per_token_latency_ms"]


def _throughput(p: dict) -> float:
    return p["objective"]["throughput_tok_s"]


def _topk_latencies(p: dict) -> list[float]:
    return [c["per_token_latency_ms"] for c in p.get("top_k_candidates", [])]


def _search_time(p: dict) -> float | None:
    return p.get("search_stats", {}).get("elapsed_s")


# -----------------------------------------------------------------------------
# Comparison dict
# -----------------------------------------------------------------------------

def build_comparison(lp: dict, grid: dict, model_id: str,
                     sim_gpu_gb: float | None, sim_ram_gb: float | None) -> dict:
    lp_lat  = _latency_ms(lp)
    grd_lat = _latency_ms(grid)
    lp_thr  = _throughput(lp)
    grd_thr = _throughput(grid)
    gap_lat = (grd_lat - lp_lat) / lp_lat * 100.0
    gap_thr = (lp_thr - grd_thr) / grd_thr * 100.0

    return {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "model_id":    model_id,
        "sim_gpu_gb":  sim_gpu_gb,
        "sim_ram_gb":  sim_ram_gb,
        "lp_result":   {"timestamp": lp.get("timestamp"),  "method": "lp_search"},
        "grid_result": {"timestamp": grid.get("timestamp"), "method": "grid_search"},
        "latency_ms": {
            "lp":   lp_lat,
            "grid": grd_lat,
            "grid_worse_by_pct": round(gap_lat, 2),
        },
        "throughput_tok_s": {
            "lp":   lp_thr,
            "grid": grd_thr,
            "lp_better_by_pct": round(gap_thr, 2),
        },
        "search_time_s": {
            "lp":   lp.get("search_stats", {}).get("elapsed_s", "N/A"),
            "grid": _search_time(grid),
        },
        "best_policy": {
            "lp":   lp["best_policy"],
            "grid": grid["best_policy"],
        },
        "input": lp.get("input", {}),
    }


def print_comparison_table(cmp: dict) -> None:
    lp_lat  = cmp["latency_ms"]["lp"]
    grd_lat = cmp["latency_ms"]["grid"]
    lp_thr  = cmp["throughput_tok_s"]["lp"]
    grd_thr = cmp["throughput_tok_s"]["grid"]
    gap_lat = cmp["latency_ms"]["grid_worse_by_pct"]
    gap_thr = cmp["throughput_tok_s"]["lp_better_by_pct"]
    lp_t    = cmp["search_time_s"]["lp"]
    grd_t   = cmp["search_time_s"]["grid"]
    lp_bp   = cmp["best_policy"]["lp"]
    grd_bp  = cmp["best_policy"]["grid"]

    model_line = f"Model: {cmp.get('model_id', '?')}"
    if cmp.get("sim_gpu_gb"):
        model_line += f"  (sim GPU={cmp['sim_gpu_gb']} GB)"

    print("\n" + "=" * 68)
    print("FlexGen LP  vs  Grid Search Baseline -- Performance Comparison")
    print(model_line)
    print("=" * 68)
    print(f"{'Metric':<32} {'LP (inner LP)':<18} {'Grid (step=0.25)':<18}")
    print("-" * 68)
    print(f"{'Best latency (ms/token)':<32} {lp_lat:<18.4f} {grd_lat:<18.4f}")
    print(f"{'Throughput (tok/s)':<32} {lp_thr:<18.4f} {grd_thr:<18.4f}")
    print(f"{'Search time (s)':<32} {str(lp_t):<18} {str(grd_t):<18}")
    print("-" * 68)
    print(f"  Grid latency worse than LP by : {gap_lat:+.2f}%")
    print(f"  LP throughput better than Grid: {gap_thr:+.2f}%")
    print("=" * 68)

    print("\n-- Best-policy parameters (all 14) --------------------------------")
    params = ["gpu_batch_size", "num_gpu_batches", "block_size",
              "compression", "cpu_compute_delegate", "overlap_io_compute"]
    for k in params:
        lv = lp_bp.get(k, "-"); gv = grd_bp.get(k, "-")
        m = "  <-- differ" if str(lv) != str(gv) else ""
        print(f"  {k:<28} LP={str(lv):<10} Grid={str(gv):<10}{m}")

    print("\n-- Placement fractions (9 params) ----------------------------------")
    for tensor, short in [("weights","w"), ("kv_cache","c"), ("activations","h")]:
        for tier in ["gpu", "cpu", "disk"]:
            lv = lp_bp[tensor].get(tier, 0.0)
            gv = grd_bp[tensor].get(tier, 0.0)
            m = "  <-- differ" if abs(lv - gv) > 1e-3 else ""
            print(f"  {short}_{tier[0]:<25}    LP={lv:.4f}  Grid={gv:.4f}{m}")
    print()


# -----------------------------------------------------------------------------
# Plots (per-model subdirectory)
# -----------------------------------------------------------------------------

def plot_latency_throughput(cmp: dict, out_dir: Path) -> str:
    labels = ["LP (inner LP)", "Grid (step=0.25)"]
    lats   = [cmp["latency_ms"]["lp"],        cmp["latency_ms"]["grid"]]
    thrs   = [cmp["throughput_tok_s"]["lp"],  cmp["throughput_tok_s"]["grid"]]
    colors = ["#2196F3", "#FF9800"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, vals, ylabel, title in [
        (ax1, lats, "Per-token latency (ms)", "Best Latency: LP vs Grid"),
        (ax2, thrs, "Throughput (tokens/s)",  "Best Throughput: LP vs Grid"),
    ]:
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", width=0.5)
        ax.set_ylabel(ylabel); ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    gap = cmp["latency_ms"]["grid_worse_by_pct"]
    model = cmp.get("model_id", "")
    fig.suptitle(f"{model}  |  Grid latency {gap:+.2f}% vs LP", fontsize=10)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / "lp_vs_grid_latency.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}"); return out


def plot_placement(cmp: dict, out_dir: Path) -> str:
    tensors     = ["weights", "kv_cache", "activations"]
    tiers       = ["gpu", "cpu", "disk"]
    tier_colors = {"gpu": "#4CAF50", "cpu": "#2196F3", "disk": "#FF5722"}
    lp_bp  = cmp["best_policy"]["lp"]
    grd_bp = cmp["best_policy"]["grid"]
    x = np.arange(len(tensors))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, bp, title in zip(axes, [lp_bp, grd_bp], ["LP (inner LP)", "Grid (step=0.25)"]):
        bottoms = np.zeros(len(tensors))
        for tier in tiers:
            vals = np.array([bp[t].get(tier, 0.0) for t in tensors])
            ax.bar(x, vals, width=0.6, bottom=bottoms,
                   label=tier, color=tier_colors[tier], edgecolor="white")
            for i, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 0.05:
                    ax.text(i, b + v/2, f"{v:.2f}", ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
            bottoms += vals
        ax.set_xticks(x); ax.set_xticklabels(tensors)
        ax.set_ylim(0, 1.1); ax.set_ylabel("Fraction")
        ax.set_title(title); ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Placement fractions: LP vs Grid  |  {cmp.get('model_id','')}", fontsize=11)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / "lp_vs_grid_placement.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}"); return out


def plot_topk(lp: dict, grid: dict, cmp: dict, out_dir: Path) -> str:
    lp_lats  = sorted(_topk_latencies(lp))
    grd_lats = sorted(_topk_latencies(grid))
    fig, ax = plt.subplots(figsize=(9, 5))
    if lp_lats:
        ax.plot(range(1, len(lp_lats)+1),  lp_lats,  "o-",  color="#2196F3",
                label="LP top-k", linewidth=2, markersize=5)
    if grd_lats:
        ax.plot(range(1, len(grd_lats)+1), grd_lats, "s--", color="#FF9800",
                label="Grid top-k", linewidth=2, markersize=5)
    ax.set_xlabel("Candidate rank (1 = best)")
    ax.set_ylabel("Per-token latency (ms)")
    ax.set_title(f"Top-k candidates  |  {cmp.get('model_id','')}")
    ax.legend(); plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / "lp_vs_grid_topk.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}"); return out


# -----------------------------------------------------------------------------
# Model tracking
# -----------------------------------------------------------------------------

def update_tracking(cmp: dict, slug: str, plots_dir: Path, cmp_path: Path) -> None:
    tracking = {}
    if TRACKING_FILE.exists():
        tracking = json.loads(TRACKING_FILE.read_text(encoding="utf-8"))

    inp = cmp.get("input", {})
    model_info = inp.get("model", {})

    tracking[slug] = {
        "model_id":             cmp.get("model_id", ""),
        "sim_gpu_gb":           cmp.get("sim_gpu_gb"),
        "sim_ram_gb":           cmp.get("sim_ram_gb"),
        "num_layers":           model_info.get("num_layers"),
        "hidden_dim":           model_info.get("hidden_dim"),
        "lp_latency_ms":        round(cmp["latency_ms"]["lp"], 4),
        "grid_latency_ms":      round(cmp["latency_ms"]["grid"], 4),
        "lp_throughput_tok_s":  round(cmp["throughput_tok_s"]["lp"], 4),
        "grid_throughput_tok_s":round(cmp["throughput_tok_s"]["grid"], 4),
        "lp_better_by_pct":     cmp["latency_ms"]["grid_worse_by_pct"],
        "grid_search_time_s":   cmp["search_time_s"]["grid"],
        "last_updated":         cmp["timestamp"],
        "comparison_path":      str(cmp_path.relative_to(ROOT)),
        "plots_dir":            str(plots_dir.relative_to(ROOT)),
    }

    TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACKING_FILE.write_text(json.dumps(tracking, indent=2), encoding="utf-8")
    print(f"Tracking updated: {TRACKING_FILE}")


# -----------------------------------------------------------------------------
# Run-both helper
# -----------------------------------------------------------------------------

def _run_both(args: argparse.Namespace, out_dir: Path) -> tuple[str, str]:
    from experiments.run_flexgen import run as run_lp
    from experiments.run_grid_baseline import run as run_grid

    common = dict(
        model_id=args.model,
        workload_path=args.workload,
        recalibrate=args.recalibrate,
        output_dir=str(out_dir),
        log_dir=args.log_dir,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        sim_gpu_gb=getattr(args, "sim_gpu_gb", None),
        sim_ram_gb=getattr(args, "sim_ram_gb", None),
    )
    print("-- Running FlexGen LP search ---------------------------------------")
    lp_path = run_lp(**common)
    print("-- Running Grid search baseline ------------------------------------")
    grid_path = run_grid(**common)
    return lp_path, grid_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FlexGen LP vs Grid search with per-model result tracking."
    )
    parser.add_argument("--lp",          default=None)
    parser.add_argument("--grid",        default=None)
    parser.add_argument("--run-both",    action="store_true")
    parser.add_argument("--model",       default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--model-slug",  default=None,
                        help="Override auto-derived model slug for directory names")
    parser.add_argument("--workload",    default=str(ROOT / "configs" / "workload.yaml"))
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--log-dir",     default=str(ROOT / "experiments" / "logs"))
    parser.add_argument("--cache-dir",   default=str(ROOT / "configs" / "system_calibration"))
    parser.add_argument("--no-plots",    action="store_true")
    parser.add_argument("--sim-gpu-gb",  type=float, default=None)
    parser.add_argument("--sim-ram-gb",  type=float, default=None)
    args = parser.parse_args()

    slug     = args.model_slug or model_slug(args.model)
    out_dir  = RESULTS_ROOT / slug
    plot_dir = PLOTS_ROOT   / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_both:
        lp_path, grid_path = _run_both(args, out_dir)
    elif args.lp and args.grid:
        lp_path, grid_path = args.lp, args.grid
    else:
        parser.error("Provide --lp and --grid, or use --run-both.")

    lp   = _load(lp_path)
    grid = _load(grid_path)

    cmp = build_comparison(lp, grid,
                           model_id=args.model,
                           sim_gpu_gb=getattr(args, "sim_gpu_gb", None),
                           sim_ram_gb=getattr(args, "sim_ram_gb", None))
    print_comparison_table(cmp)

    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cmp_path = out_dir / f"comparison_{ts}.json"
    cmp_path.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    print(f"Comparison JSON: {cmp_path}")

    if not args.no_plots:
        print("\nGenerating plots...")
        plot_latency_throughput(cmp, plot_dir)
        plot_placement(cmp, plot_dir)
        plot_topk(lp, grid, cmp, plot_dir)

    update_tracking(cmp, slug, plot_dir, cmp_path)


if __name__ == "__main__":
    main()
