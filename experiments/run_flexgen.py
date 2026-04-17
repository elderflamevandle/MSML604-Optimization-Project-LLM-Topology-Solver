import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.flexgen.lp_formulation import MemoryCapacity, ModelMemoryRequirement, solve_memory_placement

LLAMA3_8B = ModelMemoryRequirement(weights_gb=16.0, kv_cache_gb=4.0, activations_gb=2.0)

SCENARIOS = [
    {"name": "gpu_80gb_only",         "cap": MemoryCapacity(gpu_gb=80,  cpu_gb=0,  disk_gb=0)},
    {"name": "gpu_24gb_cpu_64gb",      "cap": MemoryCapacity(gpu_gb=24,  cpu_gb=64, disk_gb=0)},
    {"name": "gpu_8gb_cpu_32gb_disk",  "cap": MemoryCapacity(gpu_gb=8,   cpu_gb=32, disk_gb=500)},
]


def main() -> None:
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output = []
    for s in SCENARIOS:
        result = solve_memory_placement(s["cap"], LLAMA3_8B)
        print(f"{s['name']}: status={result.status}, objective={result.objective:.4f}")
        print(f"  weights GPU={result.w_gpu:.2f} CPU={result.w_cpu:.2f} disk={result.w_disk:.2f}")
        output.append({
            "scenario": s["name"],
            "status": result.status,
            "objective": result.objective,
            "w_gpu": result.w_gpu, "w_cpu": result.w_cpu, "w_disk": result.w_disk,
            "c_gpu": result.c_gpu, "c_cpu": result.c_cpu, "c_disk": result.c_disk,
            "h_gpu": result.h_gpu, "h_cpu": result.h_cpu, "h_disk": result.h_disk,
        })
    out_path = os.path.join(results_dir, "flexgen_lp.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
