import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

SLO = {"ttft_p99_ms": 2000.0, "tbt_p99_ms": 200.0, "delay_p99_ms": 10000.0}


def generate_recommendation(grid_csv: str, flexgen_json: str, helix_json: str) -> dict:
    df = pd.read_csv(grid_csv)
    feasible = df[
        (df["ttft_p99_ms"] <= SLO["ttft_p99_ms"])
        & (df["tbt_p99_ms"] <= SLO["tbt_p99_ms"])
        & (df["delay_p99_ms"] <= SLO["delay_p99_ms"])
    ].copy()
    if feasible.empty:
        vidur_rec: dict = {"error": "No feasible Vidur config within SLO"}
    else:
        feasible["efficiency"] = feasible["qps"] / feasible["cost_proxy"].replace(0, float("inf"))
        best = feasible.sort_values("efficiency", ascending=False).iloc[0]
        vidur_rec = {
            "q": best["config_q"],
            "b": int(best["config_b"]),
            "p": int(best["config_p"]),
            "qps": round(float(best["qps"]), 2),
            "efficiency": round(float(best["efficiency"]), 4),
        }

    with open(flexgen_json) as f:
        fg = json.load(f)
    fg_best = min(
        (r for r in fg if r["status"] == "Optimal"),
        key=lambda r: r["objective"],
        default=fg[0],
    )

    with open(helix_json) as f:
        helix = json.load(f)

    recommendation = {
        "vidur": vidur_rec,
        "flexgen": {
            "best_scenario": fg_best["scenario"],
            "w_gpu": round(fg_best["w_gpu"], 3),
            "objective": round(fg_best["objective"], 4),
        },
        "helix": {
            "total_flow": round(helix["total_flow"], 4),
            "status": helix["status"],
        },
    }

    print("\n=== DEPLOYMENT RECOMMENDATION ===")
    if "error" not in vidur_rec:
        print(f"[Vidur] Best config: q={vidur_rec['q']}, b={vidur_rec['b']}, "
              f"p={vidur_rec['p']} | QPS={vidur_rec['qps']}, efficiency={vidur_rec['efficiency']}")
    else:
        print(f"[Vidur] {vidur_rec['error']}")
    print(f"[FlexGen] Best scenario: {recommendation['flexgen']['best_scenario']} "
          f"| GPU fraction={recommendation['flexgen']['w_gpu']:.2f}")
    print(f"[Helix] Max flow={recommendation['helix']['total_flow']:.4f} | {recommendation['helix']['status']}")
    return recommendation


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
    generate_recommendation(
        os.path.join(results_dir, "vidur_grid.csv"),
        os.path.join(results_dir, "flexgen_lp.json"),
        os.path.join(results_dir, "helix_milp.json"),
    )
