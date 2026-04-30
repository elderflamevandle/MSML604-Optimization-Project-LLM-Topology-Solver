import json
from pathlib import Path

from src.flexgen.baseline_compare import (
    build_baseline_comparison,
    write_baseline_comparison,
)


def _payload():
    return {
        "timestamp": "2026-04-30T00:00:00Z",
        "machine_id": "test_machine",
        "input": {
            "system": {
                "gpu_vram_gb": 8.0,
                "ram_gb": 32.0,
                "disk_gb": 500.0,
                "pcie_bw_gbs": 16.0,
                "disk_bw_gbs": 2.0,
                "tflops_fp16": 20.0,
                "tflops_int8": 40.0,
                "tflops_int4": 80.0,
            },
            "model": {
                "hf_id": "synthetic/test-model",
                "num_layers": 4,
                "hidden_dim": 256,
                "num_heads": 8,
                "num_kv_heads": 2,
                "intermediate_size": 1024,
                "vocab_size": 32000,
                "dtype_bytes": 2,
            },
            "workload": {
                "prompt_len": 64,
                "decode_len": 16,
            },
        },
        "best_policy": {
            "gpu_batch_size": 8,
            "num_gpu_batches": 1,
            "block_size": 8,
            "compression": "int4",
            "cpu_compute_delegate": False,
            "overlap_io_compute": True,
            "weights": {"gpu": 1.0, "cpu": 0.0, "disk": 0.0},
            "kv_cache": {"gpu": 1.0, "cpu": 0.0, "disk": 0.0},
            "activations": {"gpu": 1.0, "cpu": 0.0, "disk": 0.0},
            "per_token_latency_ms": 0.1,
        },
        "objective": {
            "per_token_latency_ms": 0.1,
            "throughput_tok_s": 10000.0,
            "t_block_ms": 0.8,
        },
        "top_k_candidates": [],
    }


def test_build_baseline_comparison_includes_optimized_and_baselines():
    comparison = build_baseline_comparison(_payload())

    assert comparison["model_id"] == "synthetic/test-model"
    assert comparison["optimized"]["objective"]["per_token_latency_ms"] == 0.1
    assert len(comparison["baselines"]) >= 4
    assert "optimized_vs_best_baseline" in comparison


def test_write_baseline_comparison_uses_dedicated_output_dir(tmp_path):
    comparison = build_baseline_comparison(_payload())

    path = write_baseline_comparison(comparison, str(tmp_path))

    assert "baseline_comparison_" in path
    data = json.loads(Path(path).read_text())
    assert data["model_id"] == "synthetic/test-model"
