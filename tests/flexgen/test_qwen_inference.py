from unittest.mock import MagicMock, patch

import pytest
import torch

from src.flexgen.qwen_inference import (
    InferenceConfig,
    _resolve_dtype,
    _select_device,
    run_qwen_inference,
    write_inference_result,
)


class _FakeTokenizer:
    eos_token_id = 0
    chat_template = None

    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=True):
        return "Prompt plus generated answer"


class _FakeModel:
    device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 5
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_select_device_rejects_cuda_when_unavailable():
    with patch("src.flexgen.qwen_inference.torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="cuda"):
            _select_device("cuda")


def test_resolve_dtype_auto_uses_fp16_on_cuda():
    assert _resolve_dtype("auto", "cuda") is torch.float16
    assert _resolve_dtype("auto", "cpu") is torch.float32


def test_run_qwen_inference_with_mocked_transformers():
    fake_tokenizer_cls = MagicMock()
    fake_model_cls = MagicMock()
    fake_tokenizer_cls.from_pretrained.return_value = _FakeTokenizer()
    fake_model_cls.from_pretrained.return_value = _FakeModel()

    with patch.dict("sys.modules", {
        "transformers": MagicMock(
            AutoTokenizer=fake_tokenizer_cls,
            AutoModelForCausalLM=fake_model_cls,
        )
    }):
        result = run_qwen_inference(InferenceConfig(
            model="/models/qwen",
            prompt="hello",
            max_new_tokens=5,
            device="cpu",
        ))

    assert result.model == "/models/qwen"
    assert result.prompt_tokens == 3
    assert result.generated_tokens == 2
    assert result.tokens_per_s > 0
    assert "generated answer" in result.generated_text


def test_write_inference_result(tmp_path):
    result = run_qwen_inference_with_static_result()
    path = write_inference_result(result, str(tmp_path))
    assert path.endswith(".json")
    assert "qwen_inference_" in path


def run_qwen_inference_with_static_result():
    from src.flexgen.qwen_inference import InferenceResult

    return InferenceResult(
        model="Qwen/Qwen2-1.5B",
        device="cpu",
        dtype="float32",
        prompt="hello",
        generated_text="hello world",
        prompt_tokens=1,
        generated_tokens=1,
        latency_s=0.1,
        tokens_per_s=10.0,
    )
