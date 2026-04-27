import json
from pathlib import Path
from unittest.mock import patch
from experiments.run_flexgen import build_output_payload, run
from src.flexgen.cost_model import EnumPoint, PlacementFractions
from src.flexgen.policy_search import Candidate, PolicyResult
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.workload import WorkloadSpec


def _make_result() -> PolicyResult:
    enum = EnumPoint(gbs=8, num_gb=4, q="int4", delegate=True, overlap=True)
    placement = PlacementFractions(w_g=0.45, w_c=0.55, w_d=0.0,
                                   c_g=0.20, c_c=0.80, c_d=0.0,
                                   h_g=1.0, h_c=0.0, h_d=0.0)
    cand = Candidate(enum=enum, placement=placement, t_per_token_s=0.0843, t_block_s=2.6976)
    return PolicyResult(best=cand, top_k=[cand])


def test_build_output_payload_contains_all_14_outputs():
    cap = LiveCapacity(gpu_vram_gb=24.0, ram_gb=64.0, disk_gb=800.0)
    coef = SystemCoefficients(pcie_bw_gbs=14.0, disk_bw_gbs=3.0,
                              tflops_fp16=10.0, tflops_int8=20.0, tflops_int4=40.0)
    spec = ModelSpec(hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32,
                     num_kv_heads=8, intermediate_size=14336, vocab_size=1, dtype_bytes=2)
    wl = WorkloadSpec(prompt_len=512, decode_len=128)
    payload = build_output_payload(_make_result(), cap, coef, spec, wl, machine_id="m1")

    bp = payload["best_policy"]
    assert bp["gpu_batch_size"] == 8
    assert bp["num_gpu_batches"] == 4
    assert bp["block_size"] == 32
    assert bp["compression"] == "int4"
    assert bp["cpu_compute_delegate"] is True
    assert bp["overlap_io_compute"] is True
    assert bp["weights"] == {"gpu": 0.45, "cpu": 0.55, "disk": 0.0}
    assert bp["kv_cache"] == {"gpu": 0.20, "cpu": 0.80, "disk": 0.0}
    assert bp["activations"] == {"gpu": 1.0, "cpu": 0.0, "disk": 0.0}
    assert payload["objective"]["per_token_latency_ms"] > 0


def test_run_writes_json_and_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "workload.yaml").write_text("prompt_len: 512\ndecode_len: 128\n")
    (tmp_path / "experiments" / "results").mkdir(parents=True)
    (tmp_path / "experiments" / "logs").mkdir(parents=True)
    (tmp_path / "configs" / "system_calibration").mkdir()

    with patch("experiments.run_flexgen.probe_live_capacity",
               return_value=LiveCapacity(24, 64, 800)), \
         patch("experiments.run_flexgen.ensure_calibration",
               return_value=SystemCoefficients(
                   pcie_bw_gbs=14, disk_bw_gbs=3,
                   tflops_fp16=10, tflops_int8=20, tflops_int4=40)), \
         patch("experiments.run_flexgen.load_model_spec",
               return_value=ModelSpec(hf_id="x", num_layers=32, hidden_dim=4096,
                                       num_heads=32, num_kv_heads=8,
                                       intermediate_size=14336, vocab_size=1,
                                       dtype_bytes=2)), \
         patch("experiments.run_flexgen.run_policy_search", return_value=_make_result()):
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
    assert payload["best_policy"]["gpu_batch_size"] == 8
    log_files = list((tmp_path / "experiments" / "logs").glob("*.log"))
    assert len(log_files) == 1
    assert len(log_files[0].read_text()) > 0
