"""Tests for grid_search_baseline.py"""
import pytest
from src.flexgen.grid_search_baseline import (
    _build_simplex_grid, _ALL_INNER, _SIMPLEX, run_grid_search,
)
from src.flexgen.cost_model import EnumPoint, PlacementFractions
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.workload import WorkloadSpec


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_spec() -> ModelSpec:
    return ModelSpec(
        hf_id="test-model",
        num_layers=4,
        hidden_dim=64,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        vocab_size=1000,
        dtype_bytes=2,
    )


@pytest.fixture
def generous_cap() -> LiveCapacity:
    return LiveCapacity(gpu_vram_gb=24.0, ram_gb=64.0, disk_gb=500.0)


@pytest.fixture
def tight_cap() -> LiveCapacity:
    # Only 0.5 GB GPU — forces heavy offloading
    return LiveCapacity(gpu_vram_gb=0.5, ram_gb=16.0, disk_gb=200.0)


@pytest.fixture
def coef() -> SystemCoefficients:
    return SystemCoefficients(
        pcie_bw_gbs=12.0,
        disk_bw_gbs=2.0,
        tflops_fp16=20.0,
        tflops_int8=40.0,
        tflops_int4=80.0,
    )


@pytest.fixture
def wl() -> WorkloadSpec:
    return WorkloadSpec(prompt_len=128, decode_len=32)


# ── Unit tests ─────────────────────────────────────────────────────────────────

class TestSimplexGrid:
    def test_all_sum_to_one(self):
        grid = _build_simplex_grid()
        for row in grid:
            assert abs(row.sum() - 1.0) < 1e-9, f"row {row} does not sum to 1"

    def test_all_non_negative(self):
        grid = _build_simplex_grid()
        assert (grid >= -1e-9).all()

    def test_15_triples(self):
        # With step=0.25 there are exactly 15 valid triples
        assert len(_build_simplex_grid()) == 15

    def test_inner_grid_shape(self):
        # Cartesian product of 3 tensors × 15 triples = 3375 rows, 9 columns
        assert _ALL_INNER.shape == (3375, 9)

    def test_inner_grid_fractions_sum_to_one(self):
        # Each tensor (3 consecutive columns) must sum to 1
        for start in (0, 3, 6):
            sums = _ALL_INNER[:, start:start + 3].sum(axis=1)
            assert (abs(sums - 1.0) < 1e-9).all()


class TestRunGridSearch:
    def test_returns_policy_result(self, tiny_spec, generous_cap, coef, wl):
        gs = run_grid_search(generous_cap, tiny_spec, wl, coef, top_k=5)
        assert gs.policy_result.best is not None
        assert len(gs.policy_result.top_k) <= 5

    def test_best_latency_positive(self, tiny_spec, generous_cap, coef, wl):
        gs = run_grid_search(generous_cap, tiny_spec, wl, coef)
        assert gs.policy_result.best.t_per_token_s > 0

    def test_topk_sorted_ascending(self, tiny_spec, generous_cap, coef, wl):
        gs = run_grid_search(generous_cap, tiny_spec, wl, coef, top_k=10)
        lats = [c.t_per_token_s for c in gs.policy_result.top_k]
        assert lats == sorted(lats), "top-k should be sorted by ascending latency"

    def test_placement_fractions_sum_to_one(self, tiny_spec, generous_cap, coef, wl):
        gs = run_grid_search(generous_cap, tiny_spec, wl, coef)
        p = gs.policy_result.best.placement
        assert abs(p.w_g + p.w_c + p.w_d - 1.0) < 1e-6
        assert abs(p.c_g + p.c_c + p.c_d - 1.0) < 1e-6
        assert abs(p.h_g + p.h_c + p.h_d - 1.0) < 1e-6

    def test_memory_feasibility_respected(self, tiny_spec, generous_cap, coef, wl):
        gs = run_grid_search(generous_cap, tiny_spec, wl, coef)
        best = gs.policy_result.best
        p = best.placement
        from src.flexgen.model_introspect import weights_per_layer_bytes, kv_per_token_bytes
        GB = 1024 ** 3
        L = tiny_spec.num_layers
        B = best.enum.block_size
        s, d = wl.prompt_len, wl.decode_len
        w_gb  = weights_per_layer_bytes(tiny_spec, best.enum.q) * L / GB
        kv_gb = kv_per_token_bytes(tiny_spec, best.enum.q) * B * (s + d) / GB
        act_gb = B * s * tiny_spec.hidden_dim * 2 * L / GB
        gpu_use = w_gb * p.w_g + kv_gb * p.c_g + act_gb * p.h_g
        assert gpu_use <= generous_cap.gpu_vram_gb + 1e-6

    def test_tight_cap_still_finds_policy(self, tiny_spec, tight_cap, coef, wl):
        gs = run_grid_search(tight_cap, tiny_spec, wl, coef)
        assert gs.policy_result.best is not None
        # Tiny model is very small — feasibility check should pass for many placements
        assert gs.n_feasible > 0

    def test_stats_n_total_correct(self, tiny_spec, generous_cap, coef, wl):
        gs = run_grid_search(generous_cap, tiny_spec, wl, coef)
        # Outer grid: 6 gbs × 5 num_gb × 2 quant × 2 delegate × 2 overlap = 240
        assert gs.n_total == 240 * 3375
        assert gs.n_feasible + gs.n_infeasible == gs.n_total

    def test_elapsed_is_positive(self, tiny_spec, generous_cap, coef, wl):
        gs = run_grid_search(generous_cap, tiny_spec, wl, coef)
        assert gs.elapsed_s > 0

    def test_grid_result_worse_or_equal_than_lp(self, tiny_spec, generous_cap, coef, wl):
        """
        The LP inner solver finds the exact optimum within the continuous relaxation.
        Grid search is restricted to step=0.25 — so LP should always find latency
        <= grid search best.
        """
        from src.flexgen.policy_search import run_policy_search
        lp_result  = run_policy_search(generous_cap, tiny_spec, wl, coef)
        gs         = run_grid_search(generous_cap, tiny_spec, wl, coef)
        lp_lat  = lp_result.best.t_per_token_s
        grd_lat = gs.policy_result.best.t_per_token_s
        assert lp_lat <= grd_lat + 1e-6, (
            f"LP ({lp_lat:.6f}s) should be <= grid ({grd_lat:.6f}s)"
        )
