from dataclasses import dataclass
from typing import Dict, Tuple

import pulp


@dataclass(frozen=True)
class GPUNode:
    node_id: str
    throughput_gbs: float
    memory_gb: float


@dataclass(frozen=True)
class NetworkLink:
    src: str
    dst: str
    bandwidth_gbs: float


@dataclass(frozen=True)
class FlowResult:
    total_flow: float
    flow_assignments: dict
    layer_placement: dict
    status: str


def solve_max_flow_placement(
    nodes: list[GPUNode],
    links: list[NetworkLink],
    num_layers: int = 32,
) -> FlowResult:
    """
    Solve max-flow multi-GPU layer placement using MILP.

    Args:
        nodes: List of GPUNode objects with throughput and memory
        links: List of NetworkLink objects with bandwidth constraints
        num_layers: Number of model layers to place

    Returns:
        FlowResult with total flow, flow assignments, layer placement, and status
    """
    prob = pulp.LpProblem("helix_flow", pulp.LpMaximize)
    node_ids = [n.node_id for n in nodes]
    node_map = {n.node_id: n for n in nodes}

    # Flow variables for each link
    flow_vars = {
        (lk.src, lk.dst): pulp.LpVariable(f"f_{lk.src}_{lk.dst}", lowBound=0)
        for lk in links
    }

    # Placement variables: place_vars[(layer, node)] = 1 if layer is on node
    place_vars = {
        (l, n): pulp.LpVariable(f"place_{l}_{n}", cat="Binary")
        for l in range(num_layers)
        for n in node_ids
    }

    # Objective: Maximize total source outflow
    prob += pulp.lpSum(v for (s, d), v in flow_vars.items() if s == "source")

    # Flow conservation at each GPU node
    for nid in node_ids:
        in_flow = pulp.lpSum(v for (s, d), v in flow_vars.items() if d == nid)
        out_flow = pulp.lpSum(v for (s, d), v in flow_vars.items() if s == nid)
        prob += in_flow == out_flow

    # Each layer placed on exactly one node
    for l in range(num_layers):
        prob += pulp.lpSum(place_vars[(l, n)] for n in node_ids) == 1

    # Node capacity: inflow <= (layers placed * throughput / total_layers)
    for nid in node_ids:
        in_flow = pulp.lpSum(v for (s, d), v in flow_vars.items() if d == nid)
        layers_on = pulp.lpSum(place_vars[(l, nid)] for l in range(num_layers))
        prob += in_flow <= layers_on * node_map[nid].throughput_gbs / num_layers

    # Link bandwidth constraints
    for lk in links:
        if (lk.src, lk.dst) in flow_vars:
            prob += flow_vars[(lk.src, lk.dst)] <= lk.bandwidth_gbs

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract layer placement solution
    layer_placement = {}
    for l in range(num_layers):
        for n in node_ids:
            val = pulp.value(place_vars[(l, n)])
            if val is not None and val > 0.5:
                layer_placement[f"layer_{l}"] = n

    return FlowResult(
        total_flow=pulp.value(prob.objective) or 0.0,
        flow_assignments={(k): (pulp.value(v) or 0.0) for k, v in flow_vars.items()},
        layer_placement=layer_placement,
        status=pulp.LpStatus[prob.status],
    )
