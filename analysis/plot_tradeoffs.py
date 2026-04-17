import os
import pandas as pd
import matplotlib.pyplot as plt

_PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def plot_qps_vs_latency(grid_csv: str, out_dir: str = _PLOTS_DIR) -> None:
    df = pd.read_csv(grid_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, q in enumerate(["fp16", "int8", "int4"]):
        sub = df[df["config_q"] == q]
        sc = axes[i].scatter(
            sub["delay_p99_ms"], sub["qps"],
            c=sub["config_b"], cmap="viridis", alpha=0.7, s=80,
        )
        axes[i].axvline(x=10000, color="r", linestyle="--", label="SLO limit")
        axes[i].set_xlabel("Delay P99 (ms)")
        axes[i].set_ylabel("QPS")
        axes[i].set_title(f"Quantization: {q}")
        axes[i].legend()
        plt.colorbar(sc, ax=axes[i], label="Batch size")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qps_vs_latency.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_memory_vs_throughput(grid_csv: str, out_dir: str = _PLOTS_DIR) -> None:
    df = pd.read_csv(grid_csv)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["memory_gb"], df["qps"], c=df["config_b"], cmap="plasma", alpha=0.7, s=80)
    plt.colorbar(sc, label="Batch size")
    plt.xlabel("Memory (GB)")
    plt.ylabel("QPS")
    plt.title("Memory vs Throughput Trade-off")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "memory_vs_throughput.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_optimizer_comparison(grid_csv: str, bayes_csv: str, out_dir: str = _PLOTS_DIR) -> None:
    grid_df = pd.read_csv(grid_csv)
    bayes_df = pd.read_csv(bayes_csv)
    grid_df["efficiency"] = grid_df["qps"] / grid_df["cost_proxy"].replace(0, float("inf"))
    bayes_df["efficiency"] = bayes_df["qps"] / bayes_df["cost_proxy"].replace(0, float("inf"))

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(len(grid_df)),
        grid_df["efficiency"].sort_values(ascending=False).values,
        label="Grid Search", color="steelblue",
    )
    plt.plot(
        range(len(bayes_df)),
        bayes_df["efficiency"].sort_values(ascending=False).values,
        label="Bayesian (Optuna)", color="darkorange", linestyle="--",
    )
    plt.xlabel("Config rank")
    plt.ylabel("Efficiency (QPS/Cost)")
    plt.title("Grid Search vs Bayesian Optimization")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "optimizer_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    plot_qps_vs_latency(os.path.join(results_dir, "vidur_grid.csv"))
    plot_memory_vs_throughput(os.path.join(results_dir, "vidur_grid.csv"))
    plot_optimizer_comparison(
        os.path.join(results_dir, "vidur_grid.csv"),
        os.path.join(results_dir, "vidur_bayes.csv"),
    )
