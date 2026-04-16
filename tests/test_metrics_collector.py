from src.metrics_collector import collect_metrics_mock, save_results, MetricsRecord
from src.search_space import Config
import os
import pandas as pd

def test_mock_returns_metrics_record():
    config = Config(q="fp16", b=8, p=1)
    record = collect_metrics_mock(config)
    assert isinstance(record, MetricsRecord)
    assert record.config_q == "fp16"
    assert record.config_b == 8
    assert record.ttft_p99_ms > 0
    assert record.qps > 0
    assert record.memory_gb > 0

def test_fp16_uses_more_memory_than_int4():
    fp16 = collect_metrics_mock(Config(q="fp16", b=8, p=1))
    int4 = collect_metrics_mock(Config(q="int4", b=8, p=1))
    assert fp16.memory_gb > int4.memory_gb

def test_save_results_writes_csv(tmp_path):
    records = [collect_metrics_mock(Config(q="fp16", b=8, p=1))]
    path = str(tmp_path / "out.csv")
    save_results(records, path)
    assert os.path.exists(path)
    df = pd.read_csv(path)
    assert len(df) == 1
    assert "ttft_p99_ms" in df.columns
    assert "qps" in df.columns
