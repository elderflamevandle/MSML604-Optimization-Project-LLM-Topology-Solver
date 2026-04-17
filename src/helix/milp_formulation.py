from dataclasses import dataclass

import pulp


@dataclass(frozen=True)
class GPUNode:
    node_id: str
    throughput_gbs: float
    # memory_gb: not used in flow constraints (throughput-only capacity model)
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


def _pv(var) -> float:
    """Read a PuLP variable value, clamping numerical residuals to zero."""
    v = pulp.value(var)
    return max(0.0, v) if v is not None else 0.0


def solve_max_flow_placement(
    nodes: list[GPUNode],
    links: list[NetworkLink],
    num_layers: int = 32,
) -> FlowResult:
    prob = pulp.LpProblem("helix_flow", pulp.LpMaximize)
    node_ids = [n.node_id for n in nodes]
    node_map = {n.node_id: n for n in nodes}

    flow_vars = {
        (lk.src, lk.dst): pulp.LpVariable(f"f_{lk.src}_{lk.dst}", lowBound=0)
        for lk in links
    }

    place_vars = {
        (l, n): pulp.LpVariable(f"place_{l}_{n}", cat="Binary")
        for l in range(num_layers)
        for n in node_ids
    }

    prob += pulp.lpSum(v for (s, d), v in flow_vars.items() if s == "source")

    for nid in node_ids:
        in_flow = pulp.lpSum(v for (s, d), v in flow_vars.items() if d == nid)
        out_flow = pulp.lpSum(v for (s, d), v in flow_vars.items() if s == nid)
        prob += in_flow == out_flow

    for l in range(num_layers):
        prob += pulp.lpSum(place_vars[(l, n)] for n in node_ids) == 1

    for nid in node_ids:
        in_flow = pulp.lpSum(v for (s, d), v in flow_vars.items() if d == nid)
        layers_on = pulp.lpSum(place_vars[(l, nid)] for l in range(num_layers))
        prob += in_flow <= layers_on * node_map[nid].throughput_gbs / num_layers

    for lk in links:
        if (lk.src, lk.dst) in flow_vars:
            prob += flow_vars[(lk.src, lk.dst)] <= lk.bandwidth_gbs

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    layer_placement = {}
    for l in range(num_layers):
        for n in node_ids:
            val = pulp.value(place_vars[(l, n)])
            if val is not None and val > 0.5:
                layer_placement[f"layer_{l}"] = n

    return FlowResult(
        total_flow=max(0.0, pulp.value(prob.objective) or 0.0),
        flow_assignments={k: _pv(v) for k, v in flow_vars.items()},
        layer_placement=layer_placement,
        status=pulp.LpStatus[prob.status],
    )
