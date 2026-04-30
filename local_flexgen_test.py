from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.policy_search import run_policy_search
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.workload import WorkloadSpec


ROOT = Path(__file__).resolve().parent


def decision_variables_14(best_policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "gpu_batch_size": best_policy["gpu_batch_size"],
        "num_gpu_batches": best_policy["num_gpu_batches"],
        "compression": best_policy["compression"],
        "cpu_compute_delegate": best_policy["cpu_compute_delegate"],
        "overlap_io_compute": best_policy["overlap_io_compute"],
        "weights_gpu": best_policy["weights"]["gpu"],
        "weights_cpu": best_policy["weights"]["cpu"],
        "weights_disk": best_policy["weights"]["disk"],
        "kv_cache_gpu": best_policy["kv_cache"]["gpu"],
        "kv_cache_cpu": best_policy["kv_cache"]["cpu"],
        "kv_cache_disk": best_policy["kv_cache"]["disk"],
        "activations_gpu": best_policy["activations"]["gpu"],
        "activations_cpu": best_policy["activations"]["cpu"],
        "activations_disk": best_policy["activations"]["disk"],
    }


def candidate_to_json(candidate: Any) -> dict[str, Any]:
    return {
        "gpu_batch_size": candidate.enum.gbs,
        "num_gpu_batches": candidate.enum.num_gb,
        "block_size": candidate.enum.block_size,
        "compression": candidate.enum.q,
        "cpu_compute_delegate": candidate.enum.delegate,
        "overlap_io_compute": candidate.enum.overlap,
        "weights": {
            "gpu": round(candidate.placement.w_g, 4),
            "cpu": round(candidate.placement.w_c, 4),
            "disk": round(candidate.placement.w_d, 4),
        },
        "kv_cache": {
            "gpu": round(candidate.placement.c_g, 4),
            "cpu": round(candidate.placement.c_c, 4),
            "disk": round(candidate.placement.c_d, 4),
        },
        "activations": {
            "gpu": round(candidate.placement.h_g, 4),
            "cpu": round(candidate.placement.h_c, 4),
            "disk": round(candidate.placement.h_d, 4),
        },
        "per_token_latency_ms": round(candidate.t_per_token_s * 1000, 4),
    }


def build_payload(
    model: ModelSpec,
    capacity: LiveCapacity,
    coefficients: SystemCoefficients,
    workload: WorkloadSpec,
    top_k: int,
) -> dict[str, Any]:
    result = run_policy_search(capacity, model, workload, coefficients, top_k=top_k)
    best_policy = candidate_to_json(result.best)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "local_synthetic_no_model_load",
        "input": {
            "model": asdict(model),
            "system": {**asdict(capacity), **asdict(coefficients)},
            "workload": asdict(workload),
        },
        "best_policy": best_policy,
        "decision_variables_14": decision_variables_14(best_policy),
        "objective": {
            "per_token_latency_ms": round(result.best.t_per_token_s * 1000, 4),
            "throughput_tok_s": round(1.0 / result.best.t_per_token_s, 4),
            "t_block_ms": round(result.best.t_block_s * 1000, 4),
        },
        "top_k_candidates": [candidate_to_json(c) for c in result.top_k],
    }


def print_summary(payload: dict[str, Any]) -> None:
    print("\n=== Local FlexGen synthetic test ===")
    print("No Qwen weights loaded. No HuggingFace call. No GPU required.")

    print("\n=== 14 FlexGen policy parameters ===")
    for idx, (name, value) in enumerate(payload["decision_variables_14"].items(), start=1):
        print(f"{idx:02d}. {name}: {value}")

    print("\n=== Derived ===")
    print(f"block_size: {payload['best_policy']['block_size']}")

    print("\n=== Objective ===")
    for name, value in payload["objective"].items():
        print(f"{name}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local FlexGen test with synthetic Qwen/system parameters; does not load Qwen."
    )
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "results"))
    parser.add_argument("--top-k", type=int, default=20)

    parser.add_argument("--model-name", default="synthetic-qwen2-1.5b")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=8960)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--dtype-bytes", type=int, default=2)

    parser.add_argument("--gpu-vram-gb", type=float, default=24.0)
    parser.add_argument("--ram-gb", type=float, default=128.0)
    parser.add_argument("--disk-gb", type=float, default=1000.0)
    parser.add_argument("--pcie-bw-gbs", type=float, default=24.0)
    parser.add_argument("--disk-bw-gbs", type=float, default=3.0)
    parser.add_argument("--tflops-fp16", type=float, default=80.0)
    parser.add_argument("--tflops-int8", type=float, default=160.0)
    parser.add_argument("--tflops-int4", type=float, default=240.0)

    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=128)
    args = parser.parse_args()

    model = ModelSpec(
        hf_id=args.model_name,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        dtype_bytes=args.dtype_bytes,
    )
    capacity = LiveCapacity(
        gpu_vram_gb=args.gpu_vram_gb,
        ram_gb=args.ram_gb,
        disk_gb=args.disk_gb,
    )
    coefficients = SystemCoefficients(
        pcie_bw_gbs=args.pcie_bw_gbs,
        disk_bw_gbs=args.disk_bw_gbs,
        tflops_fp16=args.tflops_fp16,
        tflops_int8=args.tflops_int8,
        tflops_int4=args.tflops_int4,
    )
    workload = WorkloadSpec(prompt_len=args.prompt_len, decode_len=args.decode_len)

    payload = build_payload(
        model=model,
        capacity=capacity,
        coefficients=coefficients,
        workload=workload,
        top_k=args.top_k,
    )
    print_summary(payload)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"local_flexgen_test_{ts}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nresult_json: {out_path}")


if __name__ == "__main__":
    main()

