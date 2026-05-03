"""
Grid search baseline over all 14 FlexGen decision parameters.

Outer loop (identical to LP search):
    enumerate (gbs, num_gb, compression, cpu_delegate, overlap)
    6 × 5 × 2 × 2 × 2 = 240 outer points

Inner search (GRID instead of LP):
    enumerate discretized placement fractions for weights / kv_cache / activations
    using step=0.25  =>  15 valid triples per tensor  =>  3 375 inner combos/outer point
    Total evaluations: 240 × 3 375 = 810 000  (pure arithmetic via numpy, ~0.1–1 s)

Memory feasibility uses the same constraint as solve_inner_lp:
    w_total_gb * w_g + kv_total_gb * c_g + act_total_gb * h_g  <=  cap.gpu_vram_gb
    (same for cpu / disk)
"""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass

import numpy as np

from src.flexgen.cost_model import (
    EnumPoint, PlacementFractions,
    prefill_flops_per_layer, decode_flops_per_layer,
)
from src.flexgen.model_introspect import ModelSpec, weights_per_layer_bytes, kv_per_token_bytes
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.workload import WorkloadSpec
from src.flexgen.policy_search import GBS_GRID, NUM_GB_GRID, QUANT_GRID, Candidate, PolicyResult

logger = logging.getLogger(__name__)

_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)
_GB = 1024 ** 3


def _build_simplex_grid() -> np.ndarray:
    """Return (N, 3) array of all (a, b, c) triples from _STEPS summing to 1.0."""
    triples = []
    for a, b in itertools.product(_STEPS, _STEPS):
        c = round(1.0 - a - b, 10)
        if any(abs(c - v) < 1e-9 for v in _STEPS) and c >= -1e-9:
            triples.append((a, b, max(0.0, c)))
    return np.array(sorted(set(triples)), dtype=np.float64)


_SIMPLEX: np.ndarray = _build_simplex_grid()   # shape (15, 3)

# Pre-compute cartesian product of all three tensor placements: (3375, 9)
_w_idx, _c_idx, _h_idx = map(np.array, zip(*itertools.product(range(len(_SIMPLEX)), repeat=3)))
_ALL_INNER: np.ndarray = np.concatenate(
    [_SIMPLEX[_w_idx], _SIMPLEX[_c_idx], _SIMPLEX[_h_idx]], axis=1
)  # (3375, 9): [w_g,w_c,w_d, c_g,c_c,c_d, h_g,h_c,h_d]


@dataclass(frozen=True)
class GridSearchResult:
    policy_result: PolicyResult
    n_total: int
    n_feasible: int
    n_infeasible: int
    elapsed_s: float
    inner_grid_size: int
    step: float


def _tflops(coef: SystemCoefficients, q: str) -> float:
    return {"fp16": coef.tflops_fp16, "int4": coef.tflops_int4}.get(q, coef.tflops_fp16)


def _vectorised_eval(
    enum: EnumPoint,
    spec: ModelSpec,
    wl: WorkloadSpec,
    coef: SystemCoefficients,
    cap: LiveCapacity,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate all 3 375 inner placements for one outer enum point.

    Returns
    -------
    feasible : bool array  (3375,)
    t_per_token : float array  (3375,)   — inf for infeasible rows
    """
    p = _ALL_INNER          # (3375, 9)
    B = enum.block_size
    s, d = wl.prompt_len, wl.decode_len
    L = spec.num_layers

    w_total_gb = weights_per_layer_bytes(spec, enum.q) * L / _GB
    kv_total_gb = kv_per_token_bytes(spec, enum.q) * B * (s + d) / _GB
    act_total_gb = B * s * spec.hidden_dim * 2 * L / _GB

    gpu_use = w_total_gb * p[:, 0] + kv_total_gb * p[:, 3] + act_total_gb * p[:, 6]
    cpu_use = w_total_gb * p[:, 1] + kv_total_gb * p[:, 4] + act_total_gb * p[:, 7]
    disk_use = w_total_gb * p[:, 2] + kv_total_gb * p[:, 5] + act_total_gb * p[:, 8]
    feasible = (
        (gpu_use <= cap.gpu_vram_gb + 1e-9)
        & (cpu_use <= cap.ram_gb + 1e-9)
        & (disk_use <= cap.disk_gb + 1e-9)
    )

    pcie = coef.pcie_bw_gbs
    disk_eff = 1.0 / (1.0 / coef.disk_bw_gbs + 1.0 / coef.pcie_bw_gbs)
    tflops = _tflops(coef, enum.q)

    # Weight-load I/O term (same for prefill & decode)
    w_gb_pl = weights_per_layer_bytes(spec, enum.q) / _GB
    t_load_w = w_gb_pl * (p[:, 1] / pcie + p[:, 2] / disk_eff)

    # ── Prefill ──────────────────────────────────────────────────────────────
    t_compute_pre = prefill_flops_per_layer(spec, B, s) / (tflops * 1e12)

    kv_pl_pre_gb = kv_per_token_bytes(spec, enum.q) * B * s / L / _GB
    if enum.delegate:
        q_gb = B * s * spec.hidden_dim * 2 / _GB
        t_io_kv_pre = q_gb / pcie * p[:, 4] + kv_pl_pre_gb * (p[:, 5] / disk_eff)
    else:
        t_io_kv_pre = kv_pl_pre_gb * (p[:, 4] / pcie + p[:, 5] / disk_eff)

    act_gb_pre = B * s * spec.hidden_dim * 2 / _GB
    t_io_act_pre = act_gb_pre * (p[:, 7] / pcie + p[:, 8] / disk_eff)

    # ── Decode ───────────────────────────────────────────────────────────────
    kv_avg = int(s + (d - 1) / 2.0) if d > 1 else s
    t_compute_dec = decode_flops_per_layer(spec, B, kv_avg) / (tflops * 1e12)

    kv_pl_dec_gb = kv_per_token_bytes(spec, enum.q) * B * kv_avg / L / _GB
    if enum.delegate:
        q_gb_dec = B * 1 * spec.hidden_dim * 2 / _GB
        t_io_kv_dec = q_gb_dec / pcie * p[:, 4] + kv_pl_dec_gb * (p[:, 5] / disk_eff)
    else:
        t_io_kv_dec = kv_pl_dec_gb * (p[:, 4] / pcie + p[:, 5] / disk_eff)

    act_gb_dec = B * 1 * spec.hidden_dim * 2 / _GB
    t_io_act_dec = act_gb_dec * (p[:, 7] / pcie + p[:, 8] / disk_eff)

    # ── Combine ───────────────────────────────────────────────────────────────
    if enum.overlap:
        t_pre_layer = np.maximum(
            t_compute_pre, np.maximum(t_load_w, np.maximum(t_io_kv_pre, t_io_act_pre))
        )
        t_dec_layer = np.maximum(
            t_compute_dec, np.maximum(t_load_w, np.maximum(t_io_kv_dec, t_io_act_dec))
        )
    else:
        t_pre_layer = t_compute_pre + t_load_w + t_io_kv_pre + t_io_act_pre
        t_dec_layer = t_compute_dec + t_load_w + t_io_kv_dec + t_io_act_dec

    t_block = L * (t_pre_layer + d * t_dec_layer)
    t_per_token = np.where(feasible, t_block / B, np.inf)

    return feasible, t_per_token


def run_grid_search(
    cap: LiveCapacity,
    spec: ModelSpec,
    wl: WorkloadSpec,
    coef: SystemCoefficients,
    top_k: int = 20,
) -> GridSearchResult:
    """
    Grid search baseline.  Returns a GridSearchResult wrapping a PolicyResult
    with the same Candidate / PolicyResult types as the LP search, so all
    downstream formatters (build_output_payload, etc.) work unchanged.
    """
    t0 = time.perf_counter()
    candidates: list[Candidate] = []
    n_total = n_feasible = n_infeasible = 0

    for gbs in GBS_GRID:
        for num_gb in NUM_GB_GRID:
            for q in QUANT_GRID:
                for delegate in (False, True):
                    for overlap in (False, True):
                        enum = EnumPoint(gbs=gbs, num_gb=num_gb, q=q,
                                         delegate=delegate, overlap=overlap)
                        feasible_mask, t_pt_arr = _vectorised_eval(enum, spec, wl, coef, cap)

                        n_total += len(t_pt_arr)
                        n_feasible += int(feasible_mask.sum())
                        n_infeasible += int((~feasible_mask).sum())

                        feas_idx = np.where(feasible_mask)[0]
                        if feas_idx.size == 0:
                            continue

                        best_local = feas_idx[np.argmin(t_pt_arr[feas_idx])]
                        row = _ALL_INNER[best_local]
                        placement = PlacementFractions(
                            w_g=float(row[0]), w_c=float(row[1]), w_d=float(row[2]),
                            c_g=float(row[3]), c_c=float(row[4]), c_d=float(row[5]),
                            h_g=float(row[6]), h_c=float(row[7]), h_d=float(row[8]),
                        )
                        t_pt = float(t_pt_arr[best_local])
                        candidates.append(Candidate(
                            enum=enum,
                            placement=placement,
                            t_per_token_s=t_pt,
                            t_block_s=t_pt * enum.block_size,
                        ))

    elapsed = time.perf_counter() - t0

    if not candidates:
        raise RuntimeError(
            f"Grid search found no feasible config (n_total={n_total}). "
            "Check memory capacities — GPU/CPU/disk may be too small for all grid points."
        )

    candidates.sort(
        key=lambda c: (c.t_per_token_s, -(c.placement.w_g + c.placement.c_g + c.placement.h_g))
    )
    logger.info(
        "grid search: total=%d feasible=%d infeasible=%d best_t=%.4fs elapsed=%.1fs",
        n_total, n_feasible, n_infeasible, candidates[0].t_per_token_s, elapsed,
    )
    return GridSearchResult(
        policy_result=PolicyResult(best=candidates[0], top_k=candidates[:top_k]),
        n_total=n_total,
        n_feasible=n_feasible,
        n_infeasible=n_infeasible,
        elapsed_s=elapsed,
        inner_grid_size=len(_ALL_INNER),
        step=0.25,
    )
