from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class InferenceConfig:
    model: str
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 0.95
    dtype: str = "auto"
    device: str = "auto"
    device_map: str | None = None
    trust_remote_code: bool = True
    use_chat_template: bool = True


@dataclass(frozen=True)
class InferenceResult:
    model: str
    device: str
    dtype: str
    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    latency_s: float
    tokens_per_s: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested, but torch.cuda.is_available() is False")
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    return device


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError("dtype must be one of: auto, float16, bfloat16, float32") from exc


def _move_inputs(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    return batch


def run_qwen_inference(config: InferenceConfig) -> InferenceResult:
    """Load a Qwen-compatible causal LM and run one generation request.

    `config.model` may be a HuggingFace repo id such as `Qwen/Qwen2-1.5B` or a
    local directory containing `config.json`, tokenizer files, and model weights.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _select_device(config.device)
    torch_dtype = _resolve_dtype(config.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        trust_remote_code=config.trust_remote_code,
    )

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.device_map:
        model_kwargs["device_map"] = config.device_map

    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    if not config.device_map:
        model = model.to(device)
    model.eval()

    prompt_text = config.prompt
    if config.use_chat_template and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": config.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_device = getattr(model, "device", torch.device(device))
    inputs = _move_inputs(inputs, str(input_device))
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if config.temperature > 0:
        generate_kwargs.update({
            "do_sample": True,
            "temperature": config.temperature,
            "top_p": config.top_p,
        })
    else:
        generate_kwargs["do_sample"] = False

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - t0

    total_tokens = int(output_ids.shape[-1])
    generated_tokens = max(0, total_tokens - prompt_tokens)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    tok_s = generated_tokens / latency_s if latency_s > 0 else 0.0

    return InferenceResult(
        model=config.model,
        device=device if not config.device_map else f"device_map={config.device_map}",
        dtype=str(torch_dtype).replace("torch.", ""),
        prompt=config.prompt,
        generated_text=text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        latency_s=latency_s,
        tokens_per_s=tok_s,
    )


def write_inference_result(result: InferenceResult, output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    path = Path(output_dir) / f"qwen_inference_{ts}.json"
    path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return str(path)
