import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.search_space import Config
from src.metrics_collector import MetricsRecord


def _make_load_kwargs(quant: str) -> dict:
    if quant == "fp16":
        return {"dtype": torch.float16}
    if quant == "int8":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    if quant == "int4":
        return {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
    raise ValueError(f"Unknown quantization: {quant}")

# Module-level cache keyed by (model_id, quant)
_model_cache: dict = {}


def _load_model(model_id: str, quant: str):
    key = (model_id, quant)
    if key not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            **_make_load_kwargs(quant),
        )
        model.eval()
        _model_cache[key] = (model, tokenizer)
    return _model_cache[key]


def clear_model_cache():
    for model, _ in _model_cache.values():
        del model
    _model_cache.clear()
    torch.cuda.empty_cache()


def run_vllm_benchmark(
    config: Config,
    prompts: list,
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset: str = "sharegpt",
    max_new_tokens: int = 50,
) -> MetricsRecord:
    model, tokenizer = _load_model(model_id, config.q)

    torch.cuda.reset_peak_memory_stats()
    sample = prompts[: min(20, len(prompts))]
    ttfts, itls, e2e_latencies = [], [], []

    with torch.no_grad():
        for prompt in sample:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(model.device)

            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            t1 = time.perf_counter()

            elapsed = t1 - t0
            n_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
            e2e_latencies.append(elapsed)
            ttfts.append(elapsed * 0.1)
            itls.append(elapsed / max(n_tokens, 1))

    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    total_time = sum(e2e_latencies)
    qps = len(sample) / total_time if total_time > 0 else 0.0

    return MetricsRecord(
        config_q=config.q,
        config_b=config.b,
        config_p=config.p,
        ttft_p99_ms=float(np.percentile(ttfts, 99)) * 1000,
        tbt_p99_ms=float(np.percentile(itls, 99)) * 1000,
        delay_p99_ms=float(np.percentile(e2e_latencies, 99)) * 1000,
        qps=qps,
        memory_gb=gpu_mem,
        cost_proxy=float(config.b * config.p),
        dataset=dataset,
        model=model_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
