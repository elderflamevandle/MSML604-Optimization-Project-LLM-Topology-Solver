"""Helix MILP formulation for multi-GPU LLM serving.

Faithful to the paper (arXiv:2406.01566):
- Each compute node i holds a contiguous layer range [s_i, e_i) where s_i, k_i = e_i - s_i
  are integer decision variables.
- A directed link (i, j) between compute nodes is "valid" only if e_i = s_j, meaning the
  pipeline aligns. Validity is a binary d_{i,j}; we linearize the equality with big-M.
- Source -> i is valid iff s_i = 0; i -> sink is valid iff e_i = L.
- Objective: maximize total flow from source = serving throughput (tokens/sec).
- Compute capacity: in_flow <= T_i * k_i / L  (fraction of layers held).
- Memory capacity: k_i * mem_per_layer <= memory_gb_i.
"""

from dataclasses import dataclass

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
    node_ranges: dict        # node_id -> (start, end) layer range
    status: str


def _pv(var) -> float:
    v = pulp.value(var)
    return max(0.0, v) if v is not None else 0.0


def solve_max_flow_placement(
    nodes: list[GPUNode],
    links: list[NetworkLink],
    num_layers: int = 32,
    mem_per_layer_gb: float = 0.5,
    time_limit_sec: int = 60,
) -> FlowResult:
    L = num_layers
    prob = pulp.LpProblem("helix_milp", pulp.LpMaximize)
    node_ids = [n.node_id for n in nodes]
    node_map = {n.node_id: n for n in nodes}

    # Layer placement: s_i (start layer), k_i (num layers held)
    s = {n: pulp.LpVariable(f"s_{n}", lowBound=0, upBound=L, cat="Integer") for n in node_ids}
    k = {n: pulp.LpVariable(f"k_{n}", lowBound=0, upBound=L, cat="Integer") for n in node_ids}
    e = {n: s[n] + k[n] for n in node_ids}

    # Flow variables (continuous, non-negative)
    flow = {
        (lk.src, lk.dst): pulp.LpVariable(f"f_{lk.src}_{lk.dst}", lowBound=0)
        for lk in links
    }

    # Connection validity (binary) for each link
    d = {
        (lk.src, lk.dst): pulp.LpVariable(f"d_{lk.src}_{lk.dst}", cat="Binary")
        for lk in links
    }

    link_map = {(lk.src, lk.dst): lk for lk in links}

    # Objective: maximize total flow leaving the source
    prob += pulp.lpSum(v for (src, _), v in flow.items() if src == "source")

    # Contiguous range fits within [0, L]
    for nid in node_ids:
        prob += e[nid] <= L

    # Connection-validity constraints (big-M linearization)
    for (src, dst), lk in link_map.items():
        if src == "source" and dst in node_ids:
            # Valid iff s_dst = 0
            prob += s[dst] <= L * (1 - d[(src, dst)])
        elif dst == "sink" and src in node_ids:
            # Valid iff e_src = L
            prob += L - e[src] <= L * (1 - d[(src, dst)])
        elif src in node_ids and dst in node_ids:
            # Valid iff e_src = s_dst (pipeline alignment)
            prob += s[dst] - e[src] <= L * (1 - d[(src, dst)])
            prob += e[src] - s[dst] <= L * (1 - d[(src, dst)])

        # Bandwidth gated by validity
        prob += flow[(src, dst)] <= lk.bandwidth_gbs * d[(src, dst)]

    # Flow conservation at every compute node
    for nid in node_ids:
        in_flow = pulp.lpSum(v for (src, dst), v in flow.items() if dst == nid)
        out_flow = pulp.lpSum(v for (src, dst), v in flow.items() if src == nid)
        prob += in_flow == out_flow

    # Compute capacity: in_flow * L <= T_i * k_i   (i.e., in_flow <= T_i * k_i / L)
    for nid in node_ids:
        in_flow = pulp.lpSum(v for (src, dst), v in flow.items() if dst == nid)
        prob += in_flow * L <= node_map[nid].throughput_gbs * k[nid]

    # Memory: k_i * mem_per_layer <= memory_gb
    for nid in node_ids:
        prob += k[nid] * mem_per_layer_gb <= node_map[nid].memory_gb

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_sec))

    # Extract node ranges and derive layer_placement
    node_ranges = {}
    layer_placement = {}
    for nid in node_ids:
        s_val = int(round(pulp.value(s[nid]) or 0))
        k_val = int(round(pulp.value(k[nid]) or 0))
        node_ranges[nid] = (s_val, s_val + k_val)
        for layer_idx in range(s_val, s_val + k_val):
            layer_placement[f"layer_{layer_idx}"] = nid

    return FlowResult(
        total_flow=max(0.0, pulp.value(prob.objective) or 0.0),
        flow_assignments={k: _pv(v) for k, v in flow.items()},
        layer_placement=layer_placement,
        node_ranges=node_ranges,
        status=pulp.LpStatus[prob.status],
    )
