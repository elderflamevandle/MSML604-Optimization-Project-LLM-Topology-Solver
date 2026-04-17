from dataclasses import dataclass

import pulp


@dataclass(frozen=True)
class MemoryCapacity:
    gpu_gb: float
    cpu_gb: float
    disk_gb: float


@dataclass(frozen=True)
class ModelMemoryRequirement:
    weights_gb: float
    kv_cache_gb: float
    activations_gb: float


@dataclass(frozen=True)
class PlacementResult:
    w_gpu: float
    w_cpu: float
    w_disk: float
    c_gpu: float
    c_cpu: float
    c_disk: float
    h_gpu: float
    h_cpu: float
    h_disk: float
    objective: float
    status: str


def _pv(var) -> float:
    """Read a PuLP variable value, clamping numerical residuals to zero."""
    v = pulp.value(var)
    return max(0.0, v) if v is not None else 0.0


def solve_memory_placement(capacity: MemoryCapacity, req: ModelMemoryRequirement) -> PlacementResult:
    # Fractional placement: LP relaxation of FlexGen's block-level offloading policy
    prob = pulp.LpProblem("flexgen", pulp.LpMinimize)

    w_g = pulp.LpVariable("w_g", 0, 1)
    w_c = pulp.LpVariable("w_c", 0, 1)
    w_d = pulp.LpVariable("w_d", 0, 1)
    c_g = pulp.LpVariable("c_g", 0, 1)
    c_c = pulp.LpVariable("c_c", 0, 1)
    c_d = pulp.LpVariable("c_d", 0, 1)
    h_g = pulp.LpVariable("h_g", 0, 1)
    h_c = pulp.LpVariable("h_c", 0, 1)
    h_d = pulp.LpVariable("h_d", 0, 1)

    # Minimize off-GPU placement: CPU costs 1x, disk costs 10x latency penalty
    prob += (
        req.weights_gb * (w_c * 1.0 + w_d * 10.0)
        + req.kv_cache_gb * (c_c * 1.0 + c_d * 10.0)
        + req.activations_gb * (h_c * 1.0 + h_d * 10.0)
    )

    prob += w_g + w_c + w_d == 1
    prob += c_g + c_c + c_d == 1
    prob += h_g + h_c + h_d == 1

    prob += req.weights_gb * w_g + req.kv_cache_gb * c_g + req.activations_gb * h_g <= capacity.gpu_gb
    prob += req.weights_gb * w_c + req.kv_cache_gb * c_c + req.activations_gb * h_c <= capacity.cpu_gb
    prob += req.weights_gb * w_d + req.kv_cache_gb * c_d + req.activations_gb * h_d <= capacity.disk_gb

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    return PlacementResult(
        w_gpu=_pv(w_g), w_cpu=_pv(w_c), w_disk=_pv(w_d),
        c_gpu=_pv(c_g), c_cpu=_pv(c_c), c_disk=_pv(c_d),
        h_gpu=_pv(h_g), h_cpu=_pv(h_c), h_disk=_pv(h_d),
        objective=max(0.0, pulp.value(prob.objective) or 0.0),
        status=pulp.LpStatus[prob.status],
    )
