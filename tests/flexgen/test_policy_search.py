from src.flexgen.policy_search import (
    GBS_GRID, NUM_GB_GRID, run_policy_search, PolicyResult,
)
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.workload import WorkloadSpec

SPEC3 = ModelSpec(
    hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=8,
    intermediate_size=14336, vocab_size=128256, dtype_bytes=2,
)
COEF3 = SystemCoefficients(
    pcie_bw_gbs=14.0, disk_bw_gbs=3.0,
    tflops_fp16=10.0, tflops_int8=20.0, tflops_int4=40.0,
)
WL3 = WorkloadSpec(prompt_len=512, decode_len=128)


def test_search_grid_constants():
    assert tuple(GBS_GRID) == (1, 2, 4, 8, 16, 32)
    assert tuple(NUM_GB_GRID) == (1, 2, 4, 8, 16)


def test_run_policy_search_returns_best_and_top_k():
    cap = LiveCapacity(gpu_vram_gb=24.0, ram_gb=64.0, disk_gb=800.0)
    res = run_policy_search(cap, SPEC3, WL3, COEF3, top_k=10)
    assert isinstance(res, PolicyResult)
    assert res.best.t_per_token_s > 0
    assert len(res.top_k) <= 10
    for i in range(1, len(res.top_k)):
        assert res.top_k[i].t_per_token_s >= res.top_k[i-1].t_per_token_s
    assert res.best.t_per_token_s == res.top_k[0].t_per_token_s


def test_run_policy_search_with_huge_gpu_picks_no_offload():
    cap = LiveCapacity(gpu_vram_gb=10000.0, ram_gb=10000.0, disk_gb=10000.0)
    res = run_policy_search(cap, SPEC3, WL3, COEF3, top_k=5)
    assert res.best.placement.w_g > 0.99
    assert res.best.placement.c_g > 0.99
