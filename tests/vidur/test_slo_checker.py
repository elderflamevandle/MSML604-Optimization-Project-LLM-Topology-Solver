import time
from src.metrics_collector import MetricsRecord
from src.vidur.slo_checker import check_slo, efficiency_score


def _record(ttft=500, tbt=50, delay=2000, qps=10.0, cost=2.0):
    return MetricsRecord(
        config_q="fp16", config_b=8, config_p=1,
        ttft_p99_ms=ttft, tbt_p99_ms=tbt, delay_p99_ms=delay,
        qps=qps, memory_gb=16.0, cost_proxy=cost,
        dataset="test", model="test",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def test_passes_within_slo():
    assert check_slo(_record(ttft=500, tbt=50, delay=2000)) is True


def test_fails_on_ttft_breach():
    assert check_slo(_record(ttft=3000)) is False


def test_fails_on_tbt_breach():
    assert check_slo(_record(tbt=500)) is False


def test_fails_on_delay_breach():
    assert check_slo(_record(delay=15000)) is False


def test_efficiency_score_is_qps_over_cost():
    assert efficiency_score(_record(qps=10.0, cost=2.0)) == 5.0


def test_efficiency_score_zero_cost_returns_zero():
    assert efficiency_score(_record(qps=10.0, cost=0.0)) == 0.0
