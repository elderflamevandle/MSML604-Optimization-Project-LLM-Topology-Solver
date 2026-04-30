from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.flexgen.calibration import SystemCoefficients
from src.flexgen.cost_model import EnumPoint, PlacementFractions, t_block_seconds
from src.flexgen.lp_formulation import solve_inner_lp
from src.flexgen.model_introspect import (
    ModelSpec,
    kv_per_token_bytes,
    weights_per_layer_bytes,
)
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.workload import WorkloadSpec


def _placement_to_json(p: PlacementFractions) -> dict[str, dict[str, float]]:
    return {
        "weights": {"gpu": round(p.w_g, 4), "cpu": round(p.w_c, 4), "disk": round(p.w_d, 4)},
        "kv_cache": {"gpu": round(p.c_g, 4), "cpu": round(p.c_c, 4), "disk": round(p.c_d, 4)},
        "activations": {"gpu": round(p.h_g, 4), "cpu": round(p.h_c, 4), "disk": round(p.h_d, 4)},
    }


def _policy_to_json(enum: EnumPoint, p: PlacementFractions) -> dict[str, Any]:
    return {
        "gpu_batch_size": enum.gbs,
        "num_gpu_batches": enum.num_gb,
        "block_size": enum.block_size,
        "compression": enum.q,
        "cpu_compute_delegate": enum.delegate,
        "overlap_io_compute": enum.overlap,
        **_placement_to_json(p),
    }


def _objective(t_block_s: float, block_size: int) -> dict[str, float]:
    t_per_token_s = t_block_s / block_size
    return {
        "per_token_latency_ms": round(t_per_token_s * 1000, 4),
        "throughput_tok_s": round(1.0 / t_per_token_s, 4),
        "t_block_ms": round(t_block_s * 1000, 4),
    }


def _types_from_payload(payload: dict[str, Any]) -> tuple[
    ModelSpec, LiveCapacity, SystemCoefficients, WorkloadSpec
]:
    model = payload["input"]["model"]
    system = payload["input"]["system"]
    workload = payload["input"]["workload"]
    spec = ModelSpec(**model)
    cap = LiveCapacity(
        gpu_vram_gb=system["gpu_vram_gb"],
        ram_gb=system["ram_gb"],
        disk_gb=system["disk_gb"],
    )
    coef = SystemCoefficients(
        pcie_bw_gbs=system["pcie_bw_gbs"],
        disk_bw_gbs=system["disk_bw_gbs"],
        tflops_fp16=system["tflops_fp16"],
        tflops_int8=system["tflops_int8"],
        tflops_int4=system["tflops_int4"],
    )
    wl = WorkloadSpec(**workload)
    return spec, cap, coef, wl


def _with_improvement(entry: dict[str, Any], optimized_latency_ms: float) -> dict[str, Any]:
    if entry["status"] != "Optimal":
        return entry
    latency = entry["objective"]["per_token_latency_ms"]
    entry["vs_optimized"] = {
        "latency_ratio_baseline_over_optimized": round(latency / optimized_latency_ms, 4),
        "optimized_latency_reduction_pct": round((latency - optimized_latency_ms) / latency * 100, 2)
        if latency > 0 else 0.0,
    }
    return entry


def _fixed_lp_baseline(
    name: str,
    description: str,
    enum: EnumPoint,
    spec: ModelSpec,
    cap: LiveCapacity,
    coef: SystemCoefficients,
    wl: WorkloadSpec,
    optimized_latency_ms: float,
) -> dict[str, Any]:
    result = solve_inner_lp(enum, cap, spec, wl, coef)
    if result.status != "Optimal":
        return {
            "name": name,
            "description": description,
            "type": "fixed_discrete_lp_placement",
            "status": result.status,
            "policy": {
                "gpu_batch_size": enum.gbs,
                "num_gpu_batches": enum.num_gb,
                "block_size": enum.block_size,
                "compression": enum.q,
                "cpu_compute_delegate": enum.delegate,
                "overlap_io_compute": enum.overlap,
            },
        }

    entry = {
        "name": name,
        "description": description,
        "type": "fixed_discrete_lp_placement",
        "status": "Optimal",
        "policy": _policy_to_json(enum, result.placement),
        "objective": _objective(result.t_block_s, enum.block_size),
    }
    return _with_improvement(entry, optimized_latency_ms)


def _manual_all_gpu_feasible(
    enum: EnumPoint,
    spec: ModelSpec,
    cap: LiveCapacity,
    wl: WorkloadSpec,
) -> bool:
    gb = 1024**3
    b = enum.block_size
    weights_gb = weights_per_layer_bytes(spec, enum.q) * spec.num_layers / gb
    kv_gb = kv_per_token_bytes(spec, enum.q) * b * (wl.prompt_len + wl.decode_len) / gb
    act_gb = b * wl.prompt_len * spec.hidden_dim * 2 * spec.num_layers / gb
    return weights_gb + kv_gb + act_gb <= cap.gpu_vram_gb


def _manual_all_gpu_baseline(
    spec: ModelSpec,
    cap: LiveCapacity,
    coef: SystemCoefficients,
    wl: WorkloadSpec,
    optimized_latency_ms: float,
) -> dict[str, Any]:
    enum = EnumPoint(gbs=1, num_gb=1, q="fp16", delegate=False, overlap=False)
    placement = PlacementFractions(1, 0, 0, 1, 0, 0, 1, 0, 0)
    if not _manual_all_gpu_feasible(enum, spec, cap, wl):
        return {
            "name": "manual_all_gpu_fp16_b1_no_overlap",
            "description": "Naive manual baseline: batch 1, fp16, no overlap, everything on GPU.",
            "type": "manual_fixed_placement",
            "status": "Infeasible",
            "policy": _policy_to_json(enum, placement),
        }

    t_block_s = t_block_seconds(enum, placement, spec, wl, coef)
    entry = {
        "name": "manual_all_gpu_fp16_b1_no_overlap",
        "description": "Naive manual baseline: batch 1, fp16, no overlap, everything on GPU.",
        "type": "manual_fixed_placement",
        "status": "Optimal",
        "policy": _policy_to_json(enum, placement),
        "objective": _objective(t_block_s, enum.block_size),
    }
    return _with_improvement(entry, optimized_latency_ms)


def build_baseline_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    spec, cap, coef, wl = _types_from_payload(payload)
    optimized_latency_ms = payload["objective"]["per_token_latency_ms"]
    optimized_policy = payload["best_policy"]

    baseline_specs = [
        (
            "lp_fixed_fp16_b1_no_overlap",
            "Only optimize placement; keep batch 1, fp16, no CPU delegation, no I/O-compute overlap.",
            EnumPoint(gbs=1, num_gb=1, q="fp16", delegate=False, overlap=False),
        ),
        (
            "lp_fixed_fp16_b1_overlap",
            "Only optimize placement; keep batch 1 and fp16, but allow overlap.",
            EnumPoint(gbs=1, num_gb=1, q="fp16", delegate=False, overlap=True),
        ),
        (
            "lp_fixed_int4_b1_overlap",
            "Only optimize placement; allow int4 and overlap, but keep batch 1.",
            EnumPoint(gbs=1, num_gb=1, q="int4", delegate=False, overlap=True),
        ),
        (
            "lp_fixed_int4_b8_overlap",
            "Only optimize placement; allow int4 and overlap with a manually chosen batch 8.",
            EnumPoint(gbs=8, num_gb=1, q="int4", delegate=False, overlap=True),
        ),
    ]

    baselines = [_manual_all_gpu_baseline(spec, cap, coef, wl, optimized_latency_ms)]
    baselines.extend(
        _fixed_lp_baseline(name, desc, enum, spec, cap, coef, wl, optimized_latency_ms)
        for name, desc, enum in baseline_specs
    )

    feasible = [b for b in baselines if b["status"] == "Optimal"]
    best_baseline = min(
        feasible,
        key=lambda b: b["objective"]["per_token_latency_ms"],
    ) if feasible else None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": spec.hf_id,
        "comparison_method": (
            "Baselines use the same cost model. Fixed-discrete baselines still solve "
            "the inner placement LP, but they do not run the full outer policy search."
        ),
        "optimized": {
            "policy": optimized_policy,
            "objective": payload["objective"],
        },
        "best_baseline_name": best_baseline["name"] if best_baseline else None,
        "optimized_vs_best_baseline": {
            "latency_ratio_best_baseline_over_optimized": round(
                best_baseline["objective"]["per_token_latency_ms"] / optimized_latency_ms, 4
            ) if best_baseline else None,
            "optimized_latency_reduction_pct": round(
                (
                    best_baseline["objective"]["per_token_latency_ms"] - optimized_latency_ms
                ) / best_baseline["objective"]["per_token_latency_ms"] * 100,
                2,
            ) if best_baseline else None,
        },
        "baselines": baselines,
    }


def _model_slug(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
    return slug[-80:] if len(slug) > 80 else slug


def write_baseline_comparison(comparison: dict[str, Any], output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"baseline_comparison_{_model_slug(comparison['model_id'])}_{ts}.json"
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return str(path)


def print_baseline_comparison(comparison: dict[str, Any]) -> None:
    print("\n=== Baseline vs Optimized Comparison ===")
    print(f"model_id: {comparison['model_id']}")
    print(f"best_baseline_name: {comparison['best_baseline_name']}")
    for name, value in comparison["optimized_vs_best_baseline"].items():
        print(f"{name}: {value}")

    print("\nname | status | latency_ms | throughput_tok_s | baseline/optimized")
    print("-" * 76)
    optimized_latency = comparison["optimized"]["objective"]["per_token_latency_ms"]
    optimized_throughput = comparison["optimized"]["objective"]["throughput_tok_s"]
    print(
        f"optimized_policy | Optimal | {optimized_latency} | "
        f"{optimized_throughput} | 1.0"
    )
    for baseline in comparison["baselines"]:
        if baseline["status"] != "Optimal":
            print(f"{baseline['name']} | {baseline['status']} | n/a | n/a | n/a")
            continue
        obj = baseline["objective"]
        ratio = baseline["vs_optimized"]["latency_ratio_baseline_over_optimized"]
        print(
            f"{baseline['name']} | Optimal | {obj['per_token_latency_ms']} | "
            f"{obj['throughput_tok_s']} | {ratio}"
        )

