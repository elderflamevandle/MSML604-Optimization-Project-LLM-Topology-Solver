import time
from dataclasses import dataclass, asdict
from typing import List
import numpy as np

@dataclass
class MetricsRecord:
    config_q: str
    config_b: int
    config_p: int
    ttft_p99_ms: float
    tbt_p99_ms: float
    delay_p99_ms: float
    qps: float
    memory_gb: float
    cost_proxy: float
    dataset: str
    model: str
    timestamp: str

def collect_metrics_mock(config, dataset="mock", model="mock") -> MetricsRecord:
    import random
    base = {"fp16": 100.0, "int8": 80.0, "int4": 60.0}[config.q]
    mem = {"fp16": 16.0, "int8": 8.0, "int4": 4.5}[config.q]
    return MetricsRecord(
        config_q=config.q,
        config_b=config.b,
        config_p=config.p,
        ttft_p99_ms=base * (1 + config.b * 0.05) * random.uniform(0.9, 1.1),
        tbt_p99_ms=base * 0.1 * random.uniform(0.9, 1.1),
        delay_p99_ms=base * config.b * 0.5 * random.uniform(0.9, 1.1),
        qps=config.b / (base * 0.001) * random.uniform(0.8, 1.2),
        memory_gb=mem * config.p,
        cost_proxy=float(config.b * config.p),
        dataset=dataset,
        model=model,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

def collect_metrics_vllm(config, vllm_output: dict, dataset="sharegpt", model="llama3-8b") -> MetricsRecord:
    ttfts = vllm_output.get("ttfts", [0])
    itls = vllm_output.get("itls", [0])
    e2e = vllm_output.get("e2e_latencies", [0])
    return MetricsRecord(
        config_q=config.q,
        config_b=config.b,
        config_p=config.p,
        ttft_p99_ms=float(np.percentile(ttfts, 99)) * 1000,
        tbt_p99_ms=float(np.percentile(itls, 99)) * 1000,
        delay_p99_ms=float(np.percentile(e2e, 99)) * 1000,
        qps=vllm_output.get("request_throughput", 0.0),
        memory_gb=vllm_output.get("gpu_memory_gb", 0.0),
        cost_proxy=float(config.b * config.p),
        dataset=dataset,
        model=model,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

def save_results(records: list, path: str):
    import pandas as pd
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(path, index=False)
    return df
