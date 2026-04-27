from dataclasses import dataclass
from src.flexgen.model_introspect import (
    ModelSpec, params_per_layer, weights_per_layer_bytes, kv_per_token_bytes,
)
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.workload import WorkloadSpec


@dataclass(frozen=True)
class EnumPoint:
    gbs: int
    num_gb: int
    q: str
    delegate: bool
    overlap: bool

    @property
    def block_size(self) -> int:
        return self.gbs * self.num_gb


@dataclass(frozen=True)
class PlacementFractions:
    w_g: float; w_c: float; w_d: float
    c_g: float; c_c: float; c_d: float
    h_g: float; h_c: float; h_d: float


def prefill_flops_per_layer(spec: ModelSpec, batch: int, seq_len: int) -> float:
    matmul = 2 * batch * seq_len * params_per_layer(spec)
    attn = 4 * batch * seq_len * seq_len * spec.num_kv_heads * spec.head_dim
    return float(matmul + attn)


def decode_flops_per_layer(spec: ModelSpec, batch: int, kv_len: int) -> float:
    matmul = 2 * batch * 1 * params_per_layer(spec)
    attn = 4 * batch * 1 * kv_len * spec.num_kv_heads * spec.head_dim
    return float(matmul + attn)


@dataclass(frozen=True)
class LayerTerms:
    t_compute: float
    t_load_w: float
    t_io_kv: float
    t_io_act: float


def _tflops_for(coef: SystemCoefficients, q: str) -> float:
    return {"fp16": coef.tflops_fp16, "int8": coef.tflops_int8, "int4": coef.tflops_int4}[q]


def _disk_effective_gbs(coef: SystemCoefficients) -> float:
    return 1.0 / (1.0 / coef.disk_bw_gbs + 1.0 / coef.pcie_bw_gbs)


def _bytes_to_gb(b: float) -> float:
    return b / 1024**3


def prefill_layer_terms(
    enum: EnumPoint, p: PlacementFractions,
    spec: ModelSpec, wl: WorkloadSpec, coef: SystemCoefficients,
) -> LayerTerms:
    B = enum.block_size
    s = wl.prompt_len
    h = spec.hidden_dim

    flops = prefill_flops_per_layer(spec, batch=B, seq_len=s)
    t_compute = flops / (_tflops_for(coef, enum.q) * 1e12)

    w_bytes = weights_per_layer_bytes(spec, enum.q)
    t_load_w = (
        _bytes_to_gb(w_bytes) * (p.w_c / coef.pcie_bw_gbs + p.w_d / _disk_effective_gbs(coef))
    )

    kv_bytes_total = kv_per_token_bytes(spec, enum.q) * B * s / spec.num_layers
    if enum.delegate and p.c_c > 0:
        q_xfer_bytes = B * s * h * 2
        t_io_kv = (
            _bytes_to_gb(q_xfer_bytes) / coef.pcie_bw_gbs
            + _bytes_to_gb(kv_bytes_total) * (p.c_d / _disk_effective_gbs(coef))
        )
    else:
        t_io_kv = (
            _bytes_to_gb(kv_bytes_total)
            * (p.c_c / coef.pcie_bw_gbs + p.c_d / _disk_effective_gbs(coef))
        )

    act_bytes = B * s * h * 2
    t_io_act = (
        _bytes_to_gb(act_bytes) * (p.h_c / coef.pcie_bw_gbs + p.h_d / _disk_effective_gbs(coef))
    )

    return LayerTerms(t_compute=t_compute, t_load_w=t_load_w, t_io_kv=t_io_kv, t_io_act=t_io_act)


def _combine(terms: LayerTerms, overlap: bool) -> float:
    if overlap:
        return max(terms.t_compute, terms.t_load_w, terms.t_io_kv, terms.t_io_act)
    return terms.t_compute + terms.t_load_w + terms.t_io_kv + terms.t_io_act


def decode_layer_terms(
    enum: EnumPoint, p: PlacementFractions,
    spec: ModelSpec, wl: WorkloadSpec, coef: SystemCoefficients,
    kv_len: int,
) -> LayerTerms:
    B = enum.block_size
    h = spec.hidden_dim

    flops = decode_flops_per_layer(spec, batch=B, kv_len=kv_len)
    t_compute = flops / (_tflops_for(coef, enum.q) * 1e12)

    w_bytes = weights_per_layer_bytes(spec, enum.q)
    t_load_w = (
        _bytes_to_gb(w_bytes) * (p.w_c / coef.pcie_bw_gbs + p.w_d / _disk_effective_gbs(coef))
    )

    kv_bytes_total = kv_per_token_bytes(spec, enum.q) * B * kv_len / spec.num_layers
    if enum.delegate and p.c_c > 0:
        q_xfer_bytes = B * 1 * h * 2
        t_io_kv = (
            _bytes_to_gb(q_xfer_bytes) / coef.pcie_bw_gbs
            + _bytes_to_gb(kv_bytes_total) * (p.c_d / _disk_effective_gbs(coef))
        )
    else:
        t_io_kv = (
            _bytes_to_gb(kv_bytes_total)
            * (p.c_c / coef.pcie_bw_gbs + p.c_d / _disk_effective_gbs(coef))
        )

    act_bytes = B * 1 * h * 2
    t_io_act = (
        _bytes_to_gb(act_bytes) * (p.h_c / coef.pcie_bw_gbs + p.h_d / _disk_effective_gbs(coef))
    )

    return LayerTerms(t_compute=t_compute, t_load_w=t_load_w, t_io_kv=t_io_kv, t_io_act=t_io_act)


def t_block_seconds(
    enum: EnumPoint, p: PlacementFractions,
    spec: ModelSpec, wl: WorkloadSpec, coef: SystemCoefficients,
) -> float:
    pre = prefill_layer_terms(enum, p, spec, wl, coef)
    t_pre_layer = _combine(pre, enum.overlap)

    s = wl.prompt_len
    d = wl.decode_len
    kv_avg = s + (d - 1) / 2.0 if d > 1 else s
    dec = decode_layer_terms(enum, p, spec, wl, coef, kv_len=int(kv_avg))
    t_dec_layer = _combine(dec, enum.overlap)

    t_layer = t_pre_layer + d * t_dec_layer
    return spec.num_layers * t_layer


def t_per_token_seconds(
    enum: EnumPoint, p: PlacementFractions,
    spec: ModelSpec, wl: WorkloadSpec, coef: SystemCoefficients,
) -> float:
    return t_block_seconds(enum, p, spec, wl, coef) / enum.block_size
