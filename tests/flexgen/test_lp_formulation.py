from src.flexgen.lp_formulation import (
    MemoryCapacity, ModelMemoryRequirement, solve_memory_placement
)

LLAMA3_8B = ModelMemoryRequirement(weights_gb=16.0, kv_cache_gb=4.0, activations_gb=2.0)

def test_solve_with_large_gpu_puts_all_on_gpu():
    result = solve_memory_placement(MemoryCapacity(gpu_gb=80, cpu_gb=0, disk_gb=0), LLAMA3_8B)
    assert result.status == "Optimal"
    assert abs(result.w_gpu - 1.0) < 0.01

def test_solve_with_tight_gpu_spills_weights():
    result = solve_memory_placement(MemoryCapacity(gpu_gb=8, cpu_gb=32, disk_gb=500), LLAMA3_8B)
    assert result.status == "Optimal"
    assert result.w_gpu < 1.0

def test_placement_fractions_sum_to_one():
    result = solve_memory_placement(MemoryCapacity(gpu_gb=24, cpu_gb=64, disk_gb=0), LLAMA3_8B)
    assert abs(result.w_gpu + result.w_cpu + result.w_disk - 1.0) < 1e-4
    assert abs(result.c_gpu + result.c_cpu + result.c_disk - 1.0) < 1e-4
    assert abs(result.h_gpu + result.h_cpu + result.h_disk - 1.0) < 1e-4

def test_objective_is_nonnegative():
    result = solve_memory_placement(MemoryCapacity(gpu_gb=24, cpu_gb=64, disk_gb=0), LLAMA3_8B)
    assert result.objective >= 0


from src.flexgen.lp_formulation import solve_inner_lp, InnerLPResult
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.workload import WorkloadSpec
from src.flexgen.cost_model import EnumPoint

SPEC2 = ModelSpec(
    hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=8,
    intermediate_size=14336, vocab_size=128256, dtype_bytes=2,
)
COEF2 = SystemCoefficients(
    pcie_bw_gbs=14.0, disk_bw_gbs=3.0,
    tflops_fp16=10.0, tflops_int8=20.0, tflops_int4=40.0,
)
WL2 = WorkloadSpec(prompt_len=512, decode_len=128)


def test_inner_lp_with_huge_gpu_keeps_everything_on_gpu():
    cap = LiveCapacity(gpu_vram_gb=200.0, ram_gb=200.0, disk_gb=2000.0)
    enum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=True)
    res = solve_inner_lp(enum, cap, SPEC2, WL2, COEF2)
    assert res.status == "Optimal"
    assert res.placement.w_g > 0.99
    assert res.placement.c_g > 0.99
    assert res.placement.h_g > 0.99


def test_inner_lp_with_tight_gpu_spills_some_weights():
    cap = LiveCapacity(gpu_vram_gb=4.0, ram_gb=64.0, disk_gb=800.0)
    enum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=True)
    res = solve_inner_lp(enum, cap, SPEC2, WL2, COEF2)
    assert res.status == "Optimal"
    assert res.placement.w_g < 1.0


def test_inner_lp_returns_per_token_latency_in_seconds():
    cap = LiveCapacity(gpu_vram_gb=200.0, ram_gb=200.0, disk_gb=2000.0)
    enum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=True)
    res = solve_inner_lp(enum, cap, SPEC2, WL2, COEF2)
    assert 0 < res.t_per_token_s < 100.0
