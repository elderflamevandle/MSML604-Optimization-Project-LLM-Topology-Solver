import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load Vidur
    vidur_path = os.path.join(results_dir, "vidur_grid.csv")
    if os.path.exists(vidur_path):
        vidur_df = pd.read_csv(vidur_path)
        # Find best QPS
        best_vidur_qps = vidur_df["qps"].max()
    else:
        best_vidur_qps = 0
        
    # Load Helix
    helix_path = os.path.join(results_dir, "helix_milp.json")
    if os.path.exists(helix_path):
        with open(helix_path, "r") as f:
            helix_data = json.load(f)
        helix_qps = helix_data.get("total_flow", 0)
    else:
        helix_qps = 0
        
    # Plotting
    labels = ["Vidur (Best Single-Node Config)", "Helix (Multi-GPU Pipeline)"]
    values = [best_vidur_qps, helix_qps]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("System Throughput (QPS / Total Flow)")
    plt.title("Comparison of System Throughput Optimizations")
    
    # Adding data labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontweight='bold')
        
    out_path = os.path.join(plots_dir, "overall_throughput_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
