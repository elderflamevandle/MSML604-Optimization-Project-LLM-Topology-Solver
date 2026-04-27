from src.flexgen.cost_model import (
    EnumPoint, PlacementFractions, prefill_flops_per_layer, decode_flops_per_layer,
    LayerTerms, prefill_layer_terms, decode_layer_terms,
    t_block_seconds, t_per_token_seconds,
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


def test_overlap_no_worse_than_sum():
    p = PlacementFractions(w_g=0, w_c=1, w_d=0, c_g=0, c_c=1, c_d=0, h_g=0, h_c=1, h_d=0)
    enum_sum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    enum_max = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=True)
    t_sum = t_block_seconds(enum_sum, p, SPEC, WL, COEF)
    t_max = t_block_seconds(enum_max, p, SPEC, WL, COEF)
    assert t_max <= t_sum + 1e-9


def test_t_per_token_divides_block_by_effective_batch():
    # Weights on CPU: per-block weight-load is fixed cost amortized over B*(s+d) tokens,
    # so a bigger block lowers per-token latency. (With everything on GPU, compute is the
    # only cost and it scales linearly with B, so per-token is invariant — different test.)
    p = PlacementFractions(w_g=0, w_c=1, w_d=0, c_g=1, c_c=0, c_d=0, h_g=1, h_c=0, h_d=0)
    enum1 = EnumPoint(gbs=1, num_gb=1, q="fp16", delegate=False, overlap=False)
    enum8 = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    tt1 = t_per_token_seconds(enum1, p, SPEC, WL, COEF)
    tt8 = t_per_token_seconds(enum8, p, SPEC, WL, COEF)
    assert tt8 < tt1
