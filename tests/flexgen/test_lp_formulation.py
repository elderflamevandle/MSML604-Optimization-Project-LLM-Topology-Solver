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
