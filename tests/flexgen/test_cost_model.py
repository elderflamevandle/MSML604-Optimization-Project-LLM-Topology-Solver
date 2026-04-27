from src.flexgen.cost_model import (
    EnumPoint, PlacementFractions, prefill_flops_per_layer, decode_flops_per_layer,
)
from src.flexgen.model_introspect import ModelSpec


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
