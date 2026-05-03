import os
import json as _json
import numpy as _np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def plot_flexgen_pareto(payload_path: str, out_dir: str) -> str:
    payload = _json.loads(open(payload_path).read())
    cands = payload["top_k_candidates"]
    best = payload["best_policy"]

    xs = [c["block_size"] for c in cands]
    ys = [c["per_token_latency_ms"] for c in cands]
    colors = ["#1f77b4" if c["compression"] == "fp16" else "#d62728" for c in cands]

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, c=colors, alpha=0.7, s=60)
    plt.scatter([best["block_size"]], [best["per_token_latency_ms"]],
                marker="*", s=300, c="gold", edgecolor="black", label="best")
    plt.xscale("log", base=2)
    plt.xlabel("Effective batch size (gbs * num_gb)")
    plt.ylabel("Per-token latency (ms)")
    plt.title("FlexGen policy search: latency vs effective batch")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "flexgen_pareto.png")
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_flexgen_placement_heatmap(payload_path: str, out_dir: str) -> str:
    payload = _json.loads(open(payload_path).read())
    cands = payload["top_k_candidates"]

    rows = []
    labels = []
    for c in cands:
        rows.append([c["weights"]["gpu"], c["weights"]["cpu"], c["weights"]["disk"],
                     c["kv_cache"]["gpu"], c["kv_cache"]["cpu"], c["kv_cache"]["disk"],
                     c["activations"]["gpu"], c["activations"]["cpu"], c["activations"]["disk"]])
        labels.append(f"B={c['block_size']} {c['compression']}")
    arr = _np.array(rows)

    plt.figure(figsize=(9, max(3, 0.3 * len(rows))))
    plt.imshow(arr, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="fraction")
    plt.xticks(range(9), ["w_g", "w_c", "w_d", "c_g", "c_c", "c_d", "h_g", "h_c", "h_d"])
    plt.yticks(range(len(labels)), labels)
    plt.title("FlexGen top-k placement fractions")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "flexgen_placement_heatmap.png")
    plt.savefig(out, dpi=120)
    plt.close()
    return out


if __name__ == "__main__":
    import glob
    results_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    payloads = sorted(glob.glob(os.path.join(results_dir, "flexgen_*.json")))
    payloads = [p for p in payloads if "pipeline" not in p]
    if payloads:
        latest = payloads[-1]
        print(f"Plotting from {latest}")
        print(plot_flexgen_pareto(latest, _PLOTS_DIR))
        print(plot_flexgen_placement_heatmap(latest, _PLOTS_DIR))
    else:
        print("No flexgen result JSON found. Run experiments/run_flexgen.py first.")
