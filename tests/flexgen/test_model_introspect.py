import json
from unittest.mock import patch
from src.flexgen.model_introspect import (
    ModelSpec, load_model_spec, weights_per_layer_bytes,
    kv_per_token_bytes, params_per_layer,
)


LLAMA3_8B_CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "torch_dtype": "bfloat16",
}


def test_load_model_spec_parses_llama3_config(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(LLAMA3_8B_CONFIG))
    with patch("src.flexgen.model_introspect.snapshot_download", return_value=str(tmp_path)):
        spec = load_model_spec("meta-llama/Meta-Llama-3-8B")
    assert spec.num_layers == 32
    assert spec.hidden_dim == 4096
    assert spec.num_heads == 32
    assert spec.num_kv_heads == 8
    assert spec.intermediate_size == 14336
    assert spec.dtype_bytes == 2


def test_load_model_spec_accepts_local_model_folder(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(LLAMA3_8B_CONFIG))

    with patch("src.flexgen.model_introspect.snapshot_download") as mock_download:
        spec = load_model_spec(str(tmp_path))

    mock_download.assert_not_called()
    assert spec.hf_id == str(tmp_path)
    assert spec.num_layers == 32
    assert spec.hidden_dim == 4096


def test_load_model_spec_falls_back_to_num_heads_when_no_gqa(tmp_path):
    cfg = dict(LLAMA3_8B_CONFIG)
    del cfg["num_key_value_heads"]
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    with patch("src.flexgen.model_introspect.snapshot_download", return_value=str(tmp_path)):
        spec = load_model_spec("any/model")
    assert spec.num_kv_heads == spec.num_heads


def test_params_per_layer_matches_llama3_8b():
    spec = ModelSpec(
        hf_id="meta-llama/Meta-Llama-3-8B", num_layers=32, hidden_dim=4096,
        num_heads=32, num_kv_heads=8, intermediate_size=14336,
        vocab_size=128256, dtype_bytes=2,
    )
    p = params_per_layer(spec)
    assert 200_000_000 < p < 280_000_000


def test_weights_per_layer_bytes_int4_is_quarter_of_fp16():
    spec = ModelSpec(
        hf_id="x", num_layers=1, hidden_dim=4096, num_heads=32, num_kv_heads=8,
        intermediate_size=14336, vocab_size=1, dtype_bytes=2,
    )
    fp16 = weights_per_layer_bytes(spec, "fp16")
    int4 = weights_per_layer_bytes(spec, "int4")
    assert abs(int4 / fp16 - 0.25) < 0.01


def test_kv_per_token_bytes_uses_num_kv_heads():
    spec = ModelSpec(
        hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=8,
        intermediate_size=14336, vocab_size=1, dtype_bytes=2,
    )
    expected_fp16 = 2 * 8 * 128 * 32 * 2
    assert kv_per_token_bytes(spec, "fp16") == expected_fp16
