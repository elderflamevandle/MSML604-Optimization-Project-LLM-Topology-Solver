import json
from pathlib import Path
from unittest.mock import patch
from experiments.run_flexgen import run
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec


LLAMA3_8B_SPEC = ModelSpec(
    hf_id="meta-llama/Meta-Llama-3-8B", num_layers=32, hidden_dim=4096,
    num_heads=32, num_kv_heads=8, intermediate_size=14336, vocab_size=128256,
    dtype_bytes=2,
)


def test_end_to_end_llama3_8b_with_synthetic_system(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "workload.yaml").write_text("prompt_len: 512\ndecode_len: 128\n")
    (tmp_path / "experiments" / "results").mkdir(parents=True)
    (tmp_path / "experiments" / "logs").mkdir(parents=True)
    (tmp_path / "configs" / "system_calibration").mkdir()

    cap = LiveCapacity(gpu_vram_gb=38.0, ram_gb=250.0, disk_gb=1500.0)
    coef = SystemCoefficients(pcie_bw_gbs=24.6, disk_bw_gbs=3.1,
                              tflops_fp16=142.0, tflops_int8=280.0, tflops_int4=480.0)

    with patch("experiments.run_flexgen.probe_live_capacity", return_value=cap), \
         patch("experiments.run_flexgen.ensure_calibration", return_value=coef), \
         patch("experiments.run_flexgen.load_model_spec", return_value=LLAMA3_8B_SPEC):
        out_path = run(
            model_id="meta-llama/Meta-Llama-3-8B",
            workload_path=str(tmp_path / "configs" / "workload.yaml"),
            recalibrate=False,
            output_dir=str(tmp_path / "experiments" / "results"),
            log_dir=str(tmp_path / "experiments" / "logs"),
            cache_dir=str(tmp_path / "configs" / "system_calibration"),
            verbose=False,
        )

    payload = json.loads(Path(out_path).read_text())
    bp = payload["best_policy"]

    for field in ["gpu_batch_size", "num_gpu_batches", "block_size", "compression",
                  "cpu_compute_delegate", "overlap_io_compute"]:
        assert field in bp
    for tier in ("weights", "kv_cache", "activations"):
        for level in ("gpu", "cpu", "disk"):
            assert level in bp[tier]
            assert 0.0 <= bp[tier][level] <= 1.0

    for tier in ("weights", "kv_cache", "activations"):
        s = sum(bp[tier].values())
        assert abs(s - 1.0) < 0.01

    assert payload["objective"]["per_token_latency_ms"] > 0
    assert payload["objective"]["per_token_latency_ms"] < 10_000

    assert 1 <= len(payload["top_k_candidates"]) <= 20
