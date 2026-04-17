import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.helix.milp_formulation import GPUNode, NetworkLink, solve_max_flow_placement

CLUSTER = {
    "nodes": [
        GPUNode(node_id="gpu0", throughput_gbs=100.0, memory_gb=80.0),
        GPUNode(node_id="gpu1", throughput_gbs=60.0,  memory_gb=40.0),
    ],
    "links": [
        NetworkLink(src="source", dst="gpu0", bandwidth_gbs=50.0),
        NetworkLink(src="source", dst="gpu1", bandwidth_gbs=50.0),
        NetworkLink(src="gpu0",   dst="sink",  bandwidth_gbs=50.0),
        NetworkLink(src="gpu1",   dst="sink",  bandwidth_gbs=50.0),
        NetworkLink(src="gpu0",   dst="gpu1",  bandwidth_gbs=20.0),
        NetworkLink(src="gpu1",   dst="gpu0",  bandwidth_gbs=20.0),
    ],
}


def main() -> None:
    os.makedirs("experiments/results", exist_ok=True)
    print("Solving Helix MILP (32 layers, 2-GPU cluster)...")
    result = solve_max_flow_placement(CLUSTER["nodes"], CLUSTER["links"], num_layers=32)
    print(f"Status: {result.status} | Total flow: {result.total_flow:.4f}")
    print(f"Layer sample: {dict(list(result.layer_placement.items())[:4])}")

    output = {
        "status": result.status,
        "total_flow": result.total_flow,
        "flow_assignments": {str(k): v for k, v in result.flow_assignments.items()},
        "layer_placement": result.layer_placement,
    }
    with open("experiments/results/helix_milp.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved experiments/results/helix_milp.json")


if __name__ == "__main__":
    main()
