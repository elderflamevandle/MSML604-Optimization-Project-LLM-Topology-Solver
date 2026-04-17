from types import MappingProxyType

from src.metrics_collector import MetricsRecord

SLO_DEFAULTS: MappingProxyType = MappingProxyType({
    "ttft_p99_ms": 2000.0,
    "tbt_p99_ms": 200.0,
    "delay_p99_ms": 10000.0,
})

_REQUIRED_SLO_KEYS = frozenset({"ttft_p99_ms", "tbt_p99_ms", "delay_p99_ms"})


def check_slo(record: MetricsRecord, slo: dict | None = None) -> bool:
    if slo is None:
        slo = SLO_DEFAULTS
    if not _REQUIRED_SLO_KEYS.issubset(slo):
        raise ValueError(f"slo dict missing required keys: {_REQUIRED_SLO_KEYS - slo.keys()}")
    return (
        record.ttft_p99_ms <= slo["ttft_p99_ms"]
        and record.tbt_p99_ms <= slo["tbt_p99_ms"]
        and record.delay_p99_ms <= slo["delay_p99_ms"]
    )


def efficiency_score(record: MetricsRecord) -> float:
    if record.cost_proxy <= 0.0:
        return 0.0
    return record.qps / record.cost_proxy
