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


def _pv(var, eps: float = 1e-9) -> float:
    """Read a PuLP variable value, clamping numerical residuals to zero."""
    v = pulp.value(var)
    if v is None:
        return 0.0
    return 0.0 if abs(v) < eps else max(0.0, v)


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


from src.flexgen.cost_model import (
    EnumPoint as _EnumPoint, PlacementFractions as _PlacementFractions,
    prefill_layer_terms as _prefill_layer_terms,
    decode_layer_terms as _decode_layer_terms,
)
from src.flexgen.system_probe import LiveCapacity as _LiveCapacity
from src.flexgen.calibration import SystemCoefficients as _SystemCoefficients
from src.flexgen.model_introspect import (
    ModelSpec as _ModelSpec,
    weights_per_layer_bytes as _weights_per_layer_bytes,
    kv_per_token_bytes as _kv_per_token_bytes,
)
from src.flexgen.workload import WorkloadSpec as _WorkloadSpec


@dataclass(frozen=True)
class InnerLPResult:
    placement: _PlacementFractions
    t_per_token_s: float
    t_block_s: float
    status: str


def solve_inner_lp(
    enum: _EnumPoint, cap: _LiveCapacity, spec: _ModelSpec,
    wl: _WorkloadSpec, coef: _SystemCoefficients,
) -> InnerLPResult:
    """For a fixed enumeration point, find continuous placement fractions minimizing
    per-token latency. With overlap=True, use epigraph variables τ_pre, τ_dec ≥ each term."""
    prob = pulp.LpProblem("flexgen_inner", pulp.LpMinimize)

    w_g = pulp.LpVariable("w_g", 0, 1); w_c = pulp.LpVariable("w_c", 0, 1); w_d = pulp.LpVariable("w_d", 0, 1)
    c_g = pulp.LpVariable("c_g", 0, 1); c_c = pulp.LpVariable("c_c", 0, 1); c_d = pulp.LpVariable("c_d", 0, 1)
    h_g = pulp.LpVariable("h_g", 0, 1); h_c = pulp.LpVariable("h_c", 0, 1); h_d = pulp.LpVariable("h_d", 0, 1)

    prob += w_g + w_c + w_d == 1
    prob += c_g + c_c + c_d == 1
    prob += h_g + h_c + h_d == 1

    B = enum.block_size
    s, d = wl.prompt_len, wl.decode_len
    L = spec.num_layers

    w_bytes_total = _weights_per_layer_bytes(spec, enum.q) * L
    kv_bytes_total = _kv_per_token_bytes(spec, enum.q) * B * (s + d)
    act_bytes_total = B * s * spec.hidden_dim * 2 * L

    GB = 1024**3
    prob += (w_bytes_total / GB) * w_g + (kv_bytes_total / GB) * c_g + (act_bytes_total / GB) * h_g <= cap.gpu_vram_gb
    prob += (w_bytes_total / GB) * w_c + (kv_bytes_total / GB) * c_c + (act_bytes_total / GB) * h_c <= cap.ram_gb
    prob += (w_bytes_total / GB) * w_d + (kv_bytes_total / GB) * c_d + (act_bytes_total / GB) * h_d <= cap.disk_gb

    pcie = coef.pcie_bw_gbs
    disk_eff = 1.0 / (1.0 / coef.disk_bw_gbs + 1.0 / coef.pcie_bw_gbs)

    def _w_load_term():
        wlpl_gb = _weights_per_layer_bytes(spec, enum.q) / GB
        return wlpl_gb * (w_c / pcie + w_d / disk_eff)

    def _kv_term(seq_len: int, q_tokens: int | None = None):
        kv_pl_gb = _kv_per_token_bytes(spec, enum.q) * B * seq_len / L / GB
        if enum.delegate:
            # Cost model charges FULL q_xfer whenever c_c > 0 (not proportional to c_c).
            # q_tokens is the number of query tokens: seq_len for prefill, 1 for decode.
            n_q = q_tokens if q_tokens is not None else seq_len
            q_xfer_gb = (B * n_q * spec.hidden_dim * 2) / GB
            return q_xfer_gb / pcie + kv_pl_gb * (c_d / disk_eff)
        return kv_pl_gb * (c_c / pcie + c_d / disk_eff)

    def _act_term(seq_len: int):
        a_gb = (B * seq_len * spec.hidden_dim * 2) / GB
        return a_gb * (h_c / pcie + h_d / disk_eff)

    on_gpu = _PlacementFractions(1, 0, 0, 1, 0, 0, 1, 0, 0)
    pre_const = _prefill_layer_terms(enum, on_gpu, spec, wl, coef)
    t_compute_pre = pre_const.t_compute
    kv_avg = s + (d - 1) / 2.0 if d > 1 else s
    dec_const = _decode_layer_terms(enum, on_gpu, spec, wl, coef, kv_len=int(kv_avg))
    t_compute_dec = dec_const.t_compute

    if enum.overlap:
        tau_pre = pulp.LpVariable("tau_pre", 0)
        tau_dec = pulp.LpVariable("tau_dec", 0)
        prob += tau_pre >= t_compute_pre
        prob += tau_pre >= _w_load_term()
        prob += tau_pre >= _kv_term(s, q_tokens=s)
        prob += tau_pre >= _act_term(s)
        prob += tau_dec >= t_compute_dec
        prob += tau_dec >= _w_load_term()
        prob += tau_dec >= _kv_term(int(kv_avg), q_tokens=1)
        prob += tau_dec >= _act_term(1)
        t_block_expr = L * (tau_pre + d * tau_dec)
    else:
        t_pre = t_compute_pre + _w_load_term() + _kv_term(s, q_tokens=s) + _act_term(s)
        t_dec = t_compute_dec + _w_load_term() + _kv_term(int(kv_avg), q_tokens=1) + _act_term(1)
        t_block_expr = L * (t_pre + d * t_dec)

    prob += t_block_expr / B

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return InnerLPResult(
            placement=_PlacementFractions(1, 0, 0, 1, 0, 0, 1, 0, 0),
            t_per_token_s=float("inf"), t_block_s=float("inf"), status=status,
        )

    placement = _PlacementFractions(
        w_g=_pv(w_g), w_c=_pv(w_c), w_d=_pv(w_d),
        c_g=_pv(c_g), c_c=_pv(c_c), c_d=_pv(c_d),
        h_g=_pv(h_g), h_c=_pv(h_c), h_d=_pv(h_d),
    )
    t_block_s = float(pulp.value(t_block_expr))
    return InnerLPResult(
        placement=placement,
        t_per_token_s=t_block_s / B,
        t_block_s=t_block_s,
        status="Optimal",
    )
