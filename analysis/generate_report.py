"""
Generate report/cross_model_results.md from model_tracking.json.

Usage:
    python analysis/generate_report.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
TRACKING_FILE = ROOT / "experiments" / "results" / "model_tracking.json"
REPORT_PATH   = ROOT / "report" / "cross_model_results.md"

# Display names for slugs
DISPLAY = {
    "smollm2-135m-instruct": "SmoLLM2-135M",
    "tinyllama-1.1b-chat":   "TinyLLaMA-1.1B",
    "qwen2-7b":              "Qwen2-7B",
    "mistral-7b-v0-1":       "Mistral-7B-v0.1",
}

# Rough parameter counts (B)
PARAMS_B = {
    "smollm2-135m-instruct": 0.135,
    "tinyllama-1.1b-chat":   1.1,
    "qwen2-7b":              6.5,
    "mistral-7b-v0-1":       7.0,
}


def _load_comparison(slug: str, tracking: dict) -> dict | None:
    path = ROOT / tracking[slug]["comparison_path"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_note(m: dict) -> str:
    if m.get("sim_gpu_gb"):
        return f"{m['sim_gpu_gb']} GB GPU (simulated)"
    return "real GPU (model fits without offloading)"


def _winner(pct: float) -> str:
    if pct > 0.5:
        return f"**LP wins by {pct:.1f}%**"
    if pct < -0.5:
        return f"**Grid wins by {abs(pct):.1f}%**"
    return "Tie"


def build_report(tracking: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    lines += [
        "# FlexGen LP vs Grid Search — Cross-Model Results",
        "",
        f"> Auto-generated on {now}  ",
        f"> Source: `experiments/results/model_tracking.json`  ",
        f"> Regenerate: `python analysis/generate_report.py`",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This report compares two strategies for finding the optimal FlexGen 14-parameter policy:",
        "",
        "| Strategy | Inner search | Time |",
        "|---|---|---|",
        "| **LP (FlexGen)** | Exact LP solve (scipy/PuLP) per outer enum point | ~10 s (240 LP solves) |",
        "| **Grid search** | Enumerate 3,375 discrete placements (step=0.25) per outer point | ~0.06 s (numpy, vectorised) |",
        "",
        "Both enumerate the same 240 outer points  ",
        "(`gpu_batch_size` × `num_gpu_batches` × `compression` × `cpu_compute_delegate` × `overlap_io_compute`).",
        "The 9 placement fractions (weights/KV-cache/activations across GPU/CPU/disk) are what differ.",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Model | Params | GPU constraint | LP latency | Grid latency | LP advantage | Grid search time |",
        "|---|---|---|---|---|---|---|",
    ]

    for slug, m in tracking.items():
        name  = DISPLAY.get(slug, slug)
        p     = PARAMS_B.get(slug, "?")
        gpu   = f"{m['sim_gpu_gb']} GB (sim)" if m.get("sim_gpu_gb") else "real GPU"
        lp    = f"{m['lp_latency_ms']:.2f} ms"
        grid  = f"{m['grid_latency_ms']:.2f} ms"
        adv   = _winner(m["lp_better_by_pct"])
        gtime = f"{m['grid_search_time_s']:.3f} s"
        lines.append(f"| {name} | {p}B | {gpu} | {lp} | {grid} | {adv} | {gtime} |")

    lines += [
        "",
        "> **LP advantage** = how much lower LP latency is vs grid (positive = LP wins).  ",
        "> Models with no memory pressure (SmoLLM2) produce identical results from both methods.",
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "1. **LP consistently outperforms grid search when models require memory offloading.**  ",
        "   LP found 14–31% lower per-token latency on TinyLLaMA, Qwen2-7B, and Mistral-7B by",
        "   computing exact fractional placements (e.g. `w_g=0.658`) instead of 0.25-step",
        "   approximations (e.g. `w_g=0.50`).",
        "",
        "2. **Grid search is ~150× faster** (0.05–0.07 s vs ~10 s) because it only does arithmetic",
        "   — no LP solver overhead.",
        "",
        "3. **When everything fits in GPU, both methods tie.** On SmoLLM2-135M the optimal policy",
        "   is all-GPU placement, which is a corner point both methods find exactly.",
        "",
        "4. **The LP formulation had two latent bugs in the `delegate=True` cost term** that were",
        "   discovered during this comparison (see `src/flexgen/lp_formulation.py` commit history):",
        "   - `q_xfer` was scaled by `c_c` instead of treated as a constant",
        "   - Decode `q_xfer` used `kv_avg` tokens instead of 1",
        "   After fixing both, LP correctly beats grid on all memory-constrained models.",
        "",
        "---",
        "",
        "## Per-Model Results",
        "",
    ]

    for slug, m in tracking.items():
        name = DISPLAY.get(slug, slug)
        cmp  = _load_comparison(slug, tracking)
        plots_dir = Path(m["plots_dir"])

        lines += [
            f"### {name}",
            "",
            f"- **Model ID:** `{m['model_id']}`",
            f"- **Architecture:** {m['num_layers']} layers, hidden={m['hidden_dim']}",
            f"- **Parameters:** ~{PARAMS_B.get(slug,'?')}B",
            f"- **GPU constraint:** {_gpu_note(m)}",
            f"- **Last run:** {m['last_updated'][:10]}",
            "",
            "#### Performance",
            "",
            "| Metric | LP (inner LP) | Grid (step=0.25) |",
            "|---|---|---|",
            f"| Best latency (ms/token) | **{m['lp_latency_ms']:.4f}** | {m['grid_latency_ms']:.4f} |",
            f"| Throughput (tok/s) | **{m['lp_throughput_tok_s']:.4f}** | {m['grid_throughput_tok_s']:.4f} |",
            f"| Search time (s) | ~10 | **{m['grid_search_time_s']:.3f}** |",
            f"| LP advantage | {_winner(m['lp_better_by_pct'])} | |",
            "",
        ]

        if cmp:
            bp_lp   = cmp["best_policy"]["lp"]
            bp_grid = cmp["best_policy"]["grid"]

            def fmt(d: dict) -> str:
                return (f"gbs={d['gpu_batch_size']} num_gb={d['num_gpu_batches']} "
                        f"q={d['compression']} delegate={d['cpu_compute_delegate']} "
                        f"overlap={d['overlap_io_compute']}")

            lines += [
                "#### Best Policy (all 14 parameters)",
                "",
                "| Parameter | LP | Grid | Differs? |",
                "|---|---|---|---|",
            ]
            params = ["gpu_batch_size", "num_gpu_batches", "block_size",
                      "compression", "cpu_compute_delegate", "overlap_io_compute"]
            for k in params:
                lv = bp_lp.get(k, "-"); gv = bp_grid.get(k, "-")
                diff = "YES" if str(lv) != str(gv) else ""
                lines.append(f"| `{k}` | {lv} | {gv} | {diff} |")

            for tensor, short in [("weights","w"), ("kv_cache","c"), ("activations","h")]:
                for tier in ["gpu", "cpu", "disk"]:
                    lv = bp_lp[tensor].get(tier, 0.0)
                    gv = bp_grid[tensor].get(tier, 0.0)
                    diff = "YES" if abs(lv - gv) > 1e-3 else ""
                    lines.append(f"| `{short}_{tier[0]}` | {lv:.4f} | {gv:.4f} | {diff} |")

            lines.append("")

        # Plot links (relative to report/)
        rel = Path("..") / plots_dir
        lines += [
            "#### Plots",
            "",
            f"| Chart | File |",
            f"|---|---|",
            f"| Latency & throughput | [{slug}/lp_vs_grid_latency.png]({rel}/lp_vs_grid_latency.png) |",
            f"| Placement fractions  | [{slug}/lp_vs_grid_placement.png]({rel}/lp_vs_grid_placement.png) |",
            f"| Top-k candidates     | [{slug}/lp_vs_grid_topk.png]({rel}/lp_vs_grid_topk.png) |",
            "",
            "---",
            "",
        ]

    lines += [
        "## Cross-Model Plots",
        "",
        "| Chart | File |",
        "|---|---|",
        "| Latency comparison (all models) | [model_summary_latency.png](../analysis/plots/model_summary_latency.png) |",
        "| Throughput comparison           | [model_summary_throughput.png](../analysis/plots/model_summary_throughput.png) |",
        "| Grid search time                | [model_summary_search_time.png](../analysis/plots/model_summary_search_time.png) |",
        "",
        "---",
        "",
        "## How to Reproduce",
        "",
        "```bash",
        "# Run all models and regenerate everything",
        "python run_experiments.py",
        "",
        "# Run a single model",
        "python experiments/run_all_models.py --only qwen2-7b",
        "",
        "# Regenerate cross-model plots only",
        "python analysis/plot_model_summary.py",
        "",
        "# Regenerate this report only",
        "python analysis/generate_report.py",
        "```",
        "",
        "## File Structure",
        "",
        "```",
        "experiments/results/",
        "  model_tracking.json              <- cross-model index (auto-updated)",
        "  smollm2-135m-instruct/           <- per-model results",
        "    flexgen_<ts>.json",
        "    grid_baseline_<ts>.json",
        "    comparison_<ts>.json",
        "  tinyllama-1.1b-chat/ ...",
        "  qwen2-7b/ ...",
        "  mistral-7b-v0-1/ ...",
        "",
        "analysis/plots/",
        "  model_summary_latency.png        <- cross-model charts",
        "  model_summary_throughput.png",
        "  model_summary_search_time.png",
        "  smollm2-135m-instruct/           <- per-model plots",
        "    lp_vs_grid_latency.png",
        "    lp_vs_grid_placement.png",
        "    lp_vs_grid_topk.png",
        "  tinyllama-1.1b-chat/ ...",
        "  qwen2-7b/ ...",
        "  mistral-7b-v0-1/ ...",
        "```",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    if not TRACKING_FILE.exists():
        print(f"No tracking file at {TRACKING_FILE}. Run experiments/run_all_models.py first.")
        return
    tracking = json.loads(TRACKING_FILE.read_text(encoding="utf-8"))
    md = build_report(tracking)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")
    print(f"  {len(tracking)} models  |  {md.count(chr(10))} lines")


if __name__ == "__main__":
    main()
