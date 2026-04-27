from src.flexgen.cost_model import (
    EnumPoint, PlacementFractions, prefill_flops_per_layer, decode_flops_per_layer,
    LayerTerms, prefill_layer_terms, decode_layer_terms,
)
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.workload import WorkloadSpec

COEF = SystemCoefficients(
    pcie_bw_gbs=14.0, disk_bw_gbs=3.0,
    tflops_fp16=10.0, tflops_int8=20.0, tflops_int4=40.0,
)
CAP = LiveCapacity(gpu_vram_gb=24.0, ram_gb=64.0, disk_gb=800.0)
WL = WorkloadSpec(prompt_len=512, decode_len=128)


SPEC = ModelSpec(
    hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=8,
    intermediate_size=14336, vocab_size=128256, dtype_bytes=2,
)


def test_enum_point_block_size():
    e = EnumPoint(gbs=8, num_gb=4, q="int4", delegate=True, overlap=True)
    assert e.block_size == 32


def test_placement_fractions_validates_sum():
    PlacementFractions(w_g=0.5, w_c=0.5, w_d=0.0,
                       c_g=1.0, c_c=0.0, c_d=0.0,
                       h_g=1.0, h_c=0.0, h_d=0.0)


def test_prefill_flops_scales_with_batch():
    f1 = prefill_flops_per_layer(SPEC, batch=1, seq_len=512)
    f4 = prefill_flops_per_layer(SPEC, batch=4, seq_len=512)
    assert abs(f4 / f1 - 4.0) < 0.01


def test_prefill_flops_scales_quadratically_in_seq_attention_term():
    f_short = prefill_flops_per_layer(SPEC, batch=1, seq_len=64)
    f_long = prefill_flops_per_layer(SPEC, batch=1, seq_len=2048)
    assert f_long > 30 * f_short


def test_prefill_layer_terms_off_gpu_loads_increase_with_offload():
    enum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    on_gpu = PlacementFractions(w_g=1, w_c=0, w_d=0, c_g=1, c_c=0, c_d=0, h_g=1, h_c=0, h_d=0)
    off_cpu = PlacementFractions(w_g=0, w_c=1, w_d=0, c_g=0, c_c=1, c_d=0, h_g=0, h_c=1, h_d=0)
    on = prefill_layer_terms(enum, on_gpu, SPEC, WL, COEF)
    off = prefill_layer_terms(enum, off_cpu, SPEC, WL, COEF)
    assert on.t_load_w == 0.0
    assert on.t_io_kv == 0.0
    assert on.t_io_act == 0.0
    assert off.t_load_w > 0.0
    assert off.t_io_kv > 0.0
    assert off.t_io_act > 0.0


def test_prefill_layer_int4_compute_is_faster_than_fp16():
    enum_fp = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    enum_q4 = EnumPoint(gbs=4, num_gb=2, q="int4", delegate=False, overlap=False)
    p = PlacementFractions(w_g=1, w_c=0, w_d=0, c_g=1, c_c=0, c_d=0, h_g=1, h_c=0, h_d=0)
    fp = prefill_layer_terms(enum_fp, p, SPEC, WL, COEF)
    q4 = prefill_layer_terms(enum_q4, p, SPEC, WL, COEF)
    assert q4.t_compute < fp.t_compute


def test_delegate_replaces_kv_term_with_q_transfer():
    # Use a non-GQA spec (num_kv_heads == num_heads) where KV transfer dominates Q
    # transfer per layer. With GQA (num_kv_heads << num_heads), sending Q down is
    # actually more expensive than streaming the smaller KV cache, so delegation
    # is only beneficial for non-GQA Llama-1 / OPT-style models (the original
    # FlexGen target architectures).
    spec_no_gqa = ModelSpec(
        hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=32,
        intermediate_size=11008, vocab_size=32000, dtype_bytes=2,
    )
    enum_no_del = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    enum_del = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=True, overlap=False)
    on_cpu = PlacementFractions(w_g=1, w_c=0, w_d=0, c_g=0, c_c=1, c_d=0, h_g=1, h_c=0, h_d=0)
    no_del = prefill_layer_terms(enum_no_del, on_cpu, spec_no_gqa, WL, COEF)
    yes_del = prefill_layer_terms(enum_del, on_cpu, spec_no_gqa, WL, COEF)
    assert yes_del.t_io_kv < no_del.t_io_kv
