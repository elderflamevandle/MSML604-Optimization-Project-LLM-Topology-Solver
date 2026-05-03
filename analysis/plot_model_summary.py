"""
Cross-model comparison plot from model_tracking.json.

Usage:
    python analysis/plot_model_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRACKING = ROOT / "experiments" / "results" / "model_tracking.json"
OUT_DIR  = ROOT / "analysis" / "plots"


def load_tracking() -> dict:
    if not TRACKING.exists():
        raise FileNotFoundError(f"No tracking file found at {TRACKING}. "
                                "Run experiments/run_all_models.py first.")
    return json.loads(TRACKING.read_text(encoding="utf-8"))


def _label(slug: str, m: dict) -> str:
    gpu = f"\n(sim {m['sim_gpu_gb']} GB GPU)" if m.get("sim_gpu_gb") else "\n(no sim)"
    return slug + gpu


def plot_latency_comparison(data: dict) -> str:
    slugs  = list(data.keys())
    labels = [_label(s, data[s]) for s in slugs]
    lp_lat  = [data[s]["lp_latency_ms"]   for s in slugs]
    grd_lat = [data[s]["grid_latency_ms"]  for s in slugs]
    gaps    = [data[s]["lp_better_by_pct"] for s in slugs]

    x     = np.arange(len(slugs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(slugs)*2.5), 6))
    bars1 = ax.bar(x - width/2, lp_lat,  width, label="LP (inner LP)",     color="#2196F3", edgecolor="black")
    bars2 = ax.bar(x + width/2, grd_lat, width, label="Grid (step=0.25)", color="#FF9800", edgecolor="black")

    for bar, v in zip(bars1, lp_lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2196F3")
    for bar, v in zip(bars2, grd_lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#FF9800")

    for i, (xi, g) in enumerate(zip(x, gaps)):
        y_top = max(lp_lat[i], grd_lat[i])
        color = "#4CAF50" if g > 0 else "#9E9E9E"
        ax.text(xi, y_top * 1.06, f"LP {g:+.1f}%", ha="center", fontsize=9,
                color=color, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Per-token latency (ms)")
    ax.set_title("LP vs Grid Search: Per-token Latency Across All Models")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(lp_lat + grd_lat) * 1.20)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = str(OUT_DIR / "model_summary_latency.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}"); return out


def plot_throughput_comparison(data: dict) -> str:
    slugs  = list(data.keys())
    labels = [_label(s, data[s]) for s in slugs]
    lp_thr  = [data[s]["lp_throughput_tok_s"]   for s in slugs]
    grd_thr = [data[s]["grid_throughput_tok_s"]  for s in slugs]

    x     = np.arange(len(slugs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(slugs)*2.5), 6))
    bars1 = ax.bar(x - width/2, lp_thr,  width, label="LP (inner LP)",     color="#2196F3", edgecolor="black")
    bars2 = ax.bar(x + width/2, grd_thr, width, label="Grid (step=0.25)", color="#FF9800", edgecolor="black")

    for bar, v in zip(bars1, lp_thr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2196F3")
    for bar, v in zip(bars2, grd_thr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#FF9800")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("LP vs Grid Search: Throughput Across All Models")
    ax.legend(fontsize=10)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = str(OUT_DIR / "model_summary_throughput.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}"); return out


def plot_search_time(data: dict) -> str:
    slugs  = list(data.keys())
    labels = [_label(s, data[s]) for s in slugs]
    times  = [data[s]["grid_search_time_s"] or 0 for s in slugs]

    fig, ax = plt.subplots(figsize=(max(8, len(slugs)*2), 5))
    bars = ax.bar(range(len(slugs)), times, color="#9C27B0", edgecolor="black")
    for bar, v in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(slugs))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Grid search elapsed (s)")
    ax.set_title("Grid Search Time per Model (LP takes ~10s for all)")
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = str(OUT_DIR / "model_summary_search_time.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}"); return out


def print_table(data: dict) -> None:
    print("\n" + "=" * 82)
    print("CROSS-MODEL TRACKING SUMMARY")
    print("=" * 82)
    print(f"{'Model':<28} {'Params':<8} {'LP (ms)':<10} {'Grid (ms)':<10} "
          f"{'LP better':<12} {'Grid time'}")
    print("-" * 82)
    for slug, m in data.items():
        # Estimate params from layers × hidden (rough)
        layers = m.get("num_layers") or 0
        hidden = m.get("hidden_dim") or 0
        params_rough = layers * hidden * hidden * 8 / 1e9 if layers and hidden else 0
        gpu_note = f" [{m['sim_gpu_gb']}GB]" if m.get("sim_gpu_gb") else ""
        name = slug + gpu_note
        print(f"{name:<28} {'~'+str(round(params_rough,1))+'B':<8} "
              f"{m['lp_latency_ms']:<10.2f} {m['grid_latency_ms']:<10.2f} "
              f"{m['lp_better_by_pct']:+.1f}%       {m['grid_search_time_s']}")
    print("=" * 82)


def main() -> None:
    data = load_tracking()
    if not data:
        print("No models in tracking file yet.")
        return
    print_table(data)
    print("\nGenerating summary plots...")
    plot_latency_comparison(data)
    plot_throughput_comparison(data)
    plot_search_time(data)


if __name__ == "__main__":
    main()
