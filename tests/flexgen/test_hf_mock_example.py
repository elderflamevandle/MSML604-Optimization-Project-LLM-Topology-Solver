"""Examples of how to mock HuggingFace Hub in tests.

When testing code that touches `huggingface_hub.snapshot_download`, you do NOT
want real network calls — they're slow, flaky, gated for some models, and
rate-limited. The pattern below intercepts the download and returns a local
fixture directory containing only what the code under test reads (usually
just `config.json`).

Run these examples offline:

    pytest tests/flexgen/test_hf_mock_example.py -v

There are two scenarios covered:
  1. Mocking `load_model_spec` — the optimizer's HF entry point that reads
     `config.json` only.
  2. Mocking `download_model` — the full-weights utility, including its
     gated-repo and missing-repo error branches.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock
import pytest
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError


def _gated_error() -> GatedRepoError:
    return GatedRepoError("locked", response=MagicMock(status_code=403))


def _missing_error() -> RepositoryNotFoundError:
    return RepositoryNotFoundError("nope", response=MagicMock(status_code=404))

from src.flexgen.model_introspect import load_model_spec
from scripts.download_model import download_model


_LLAMA3_8B_CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "torch_dtype": "bfloat16",
}

_MISTRAL_7B_CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "torch_dtype": "bfloat16",
}


def test_mock_load_model_spec_with_llama_config(tmp_path):
    """Pattern: write a fake config.json, patch snapshot_download to return
    the directory containing it. The code under test reads the local file
    as if it were freshly downloaded — no network, no auth."""
    (tmp_path / "config.json").write_text(json.dumps(_LLAMA3_8B_CONFIG))

    with patch("src.flexgen.model_introspect.snapshot_download",
               return_value=str(tmp_path)):
        spec = load_model_spec("meta-llama/Meta-Llama-3-8B")

    assert spec.num_layers == 32
    assert spec.hidden_dim == 4096
    assert spec.num_kv_heads == 8


def test_mock_load_model_spec_swap_architectures(tmp_path):
    """Same pattern, different fixture — proves cross-architecture handling."""
    (tmp_path / "config.json").write_text(json.dumps(_MISTRAL_7B_CONFIG))

    with patch("src.flexgen.model_introspect.snapshot_download",
               return_value=str(tmp_path)):
        spec = load_model_spec("mistralai/Mistral-7B-v0.1")

    assert spec.vocab_size == 32000


def test_mock_download_model_returns_local_path(tmp_path):
    """Mock the snapshot_download import inside scripts.download_model."""
    fake_dir = str(tmp_path / "downloaded_model")
    with patch("scripts.download_model.snapshot_download",
               return_value=fake_dir) as mock_dl:
        result = download_model("mistralai/Mistral-7B-v0.1")

    assert result == fake_dir
    mock_dl.assert_called_once()
    # patterns argument was the default (full weights)
    assert "*.safetensors" in mock_dl.call_args.kwargs["allow_patterns"]


def test_mock_download_model_gated_repo_exits_with_help(capsys):
    """Mock GatedRepoError to verify the friendly help message + exit code 2."""
    with patch("scripts.download_model.snapshot_download",
               side_effect=_gated_error()):
        with pytest.raises(SystemExit) as exc:
            download_model("meta-llama/Meta-Llama-3-8B")

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "gated model" in captured.out
    assert "huggingface-cli login" in captured.out


def test_mock_download_model_missing_repo_exits(capsys):
    """Mock RepositoryNotFoundError to verify the typo guidance + exit code 3."""
    with patch("scripts.download_model.snapshot_download",
               side_effect=_missing_error()):
        with pytest.raises(SystemExit) as exc:
            download_model("definitely/not-a-real-model")

    assert exc.value.code == 3
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_mock_download_model_config_only_uses_narrow_patterns(tmp_path):
    """When the CLI passes --config-only, patterns must exclude weight files."""
    fake_dir = str(tmp_path / "cfg_only")
    with patch("scripts.download_model.snapshot_download",
               return_value=fake_dir) as mock_dl:
        download_model("any/model", patterns=["config.json", "tokenizer*", "*.model"])

    patterns = mock_dl.call_args.kwargs["allow_patterns"]
    assert "*.safetensors" not in patterns
    assert "config.json" in patterns
