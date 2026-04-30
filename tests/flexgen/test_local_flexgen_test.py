from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.workload import WorkloadSpec
from local_flexgen_test import build_payload, decision_variables_14


def test_build_payload_returns_14_decision_variables():
    payload = build_payload(
        model=ModelSpec(
            hf_id="synthetic",
            num_layers=4,
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            intermediate_size=1024,
            vocab_size=32000,
            dtype_bytes=2,
        ),
        capacity=LiveCapacity(gpu_vram_gb=8, ram_gb=32, disk_gb=500),
        coefficients=SystemCoefficients(
            pcie_bw_gbs=16,
            disk_bw_gbs=2,
            tflops_fp16=20,
            tflops_int8=40,
            tflops_int4=80,
        ),
        workload=WorkloadSpec(prompt_len=64, decode_len=16),
        top_k=5,
    )

    assert payload["mode"] == "local_synthetic_no_model_load"
    assert len(payload["decision_variables_14"]) == 14
    assert payload["best_policy"]["block_size"] > 0
    assert payload["objective"]["per_token_latency_ms"] > 0
    assert 1 <= len(payload["top_k_candidates"]) <= 5


def test_decision_variables_14_excludes_derived_block_size():
    best = {
        "gpu_batch_size": 8,
        "num_gpu_batches": 4,
        "block_size": 32,
        "compression": "int4",
        "cpu_compute_delegate": True,
        "overlap_io_compute": True,
        "weights": {"gpu": 1, "cpu": 0, "disk": 0},
        "kv_cache": {"gpu": 1, "cpu": 0, "disk": 0},
        "activations": {"gpu": 1, "cpu": 0, "disk": 0},
    }

    decisions = decision_variables_14(best)
    assert len(decisions) == 14
    assert "block_size" not in decisions

