from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.flexgen.config_file import (
    load_flexgen_config,
    override,
    require_section,
    require_value,
    resolve_repo_path,
)


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
    model: Any,
    capacity: Any,
    coefficients: Any,
    workload: Any,
    top_k: int,
) -> dict[str, Any]:
    from src.flexgen.policy_search import run_policy_search

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
    parser.add_argument("--config", default="config_flexgen.yml",
                        help="YAML config file containing synthetic local test values.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=None)

    parser.add_argument("--model-name", default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--dtype-bytes", type=int, default=None)

    parser.add_argument("--gpu-vram-gb", type=float, default=None)
    parser.add_argument("--ram-gb", type=float, default=None)
    parser.add_argument("--disk-gb", type=float, default=None)
    parser.add_argument("--pcie-bw-gbs", type=float, default=None)
    parser.add_argument("--disk-bw-gbs", type=float, default=None)
    parser.add_argument("--tflops-fp16", type=float, default=None)
    parser.add_argument("--tflops-int8", type=float, default=None)
    parser.add_argument("--tflops-int4", type=float, default=None)

    parser.add_argument("--prompt-len", type=int, default=None)
    parser.add_argument("--decode-len", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, ROOT)
    config = load_flexgen_config(config_path)
    paths_cfg = require_section(config, "paths")
    local_cfg = require_section(config, "local_synthetic")
    model_cfg = require_section(local_cfg, "model")
    system_cfg = require_section(local_cfg, "system")
    workload_cfg = require_section(local_cfg, "workload")

    args.output_dir = resolve_repo_path(
        override(require_value(paths_cfg, "output_dir", "paths"), args.output_dir),
        ROOT,
    )
    args.top_k = int(override(require_value(local_cfg, "top_k", "local_synthetic"), args.top_k))

    from src.flexgen.calibration import SystemCoefficients
    from src.flexgen.model_introspect import ModelSpec
    from src.flexgen.system_probe import LiveCapacity
    from src.flexgen.workload import WorkloadSpec

    model = ModelSpec(
        hf_id=override(require_value(model_cfg, "name", "local_synthetic.model"), args.model_name),
        num_layers=int(override(require_value(model_cfg, "num_layers", "local_synthetic.model"), args.num_layers)),
        hidden_dim=int(override(require_value(model_cfg, "hidden_dim", "local_synthetic.model"), args.hidden_dim)),
        num_heads=int(override(require_value(model_cfg, "num_heads", "local_synthetic.model"), args.num_heads)),
        num_kv_heads=int(override(require_value(model_cfg, "num_kv_heads", "local_synthetic.model"), args.num_kv_heads)),
        intermediate_size=int(override(
            require_value(model_cfg, "intermediate_size", "local_synthetic.model"),
            args.intermediate_size,
        )),
        vocab_size=int(override(require_value(model_cfg, "vocab_size", "local_synthetic.model"), args.vocab_size)),
        dtype_bytes=int(override(require_value(model_cfg, "dtype_bytes", "local_synthetic.model"), args.dtype_bytes)),
    )
    capacity = LiveCapacity(
        gpu_vram_gb=float(override(
            require_value(system_cfg, "gpu_vram_gb", "local_synthetic.system"),
            args.gpu_vram_gb,
        )),
        ram_gb=float(override(require_value(system_cfg, "ram_gb", "local_synthetic.system"), args.ram_gb)),
        disk_gb=float(override(require_value(system_cfg, "disk_gb", "local_synthetic.system"), args.disk_gb)),
    )
    coefficients = SystemCoefficients(
        pcie_bw_gbs=float(override(
            require_value(system_cfg, "pcie_bw_gbs", "local_synthetic.system"),
            args.pcie_bw_gbs,
        )),
        disk_bw_gbs=float(override(
            require_value(system_cfg, "disk_bw_gbs", "local_synthetic.system"),
            args.disk_bw_gbs,
        )),
        tflops_fp16=float(override(
            require_value(system_cfg, "tflops_fp16", "local_synthetic.system"),
            args.tflops_fp16,
        )),
        tflops_int8=float(override(
            require_value(system_cfg, "tflops_int8", "local_synthetic.system"),
            args.tflops_int8,
        )),
        tflops_int4=float(override(
            require_value(system_cfg, "tflops_int4", "local_synthetic.system"),
            args.tflops_int4,
        )),
    )
    workload = WorkloadSpec(
        prompt_len=int(override(
            require_value(workload_cfg, "prompt_len", "local_synthetic.workload"),
            args.prompt_len,
        )),
        decode_len=int(override(
            require_value(workload_cfg, "decode_len", "local_synthetic.workload"),
            args.decode_len,
        )),
    )

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
