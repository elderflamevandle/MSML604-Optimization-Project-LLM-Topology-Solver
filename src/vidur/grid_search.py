from src.search_space import get_search_space, Config
from src.metrics_collector import MetricsRecord, collect_metrics_mock
from src.vidur.slo_checker import check_slo, efficiency_score
from typing import Callable


def run_grid_search(
    measure_fn: Callable[[Config], MetricsRecord],
    slo: dict | None = None,
) -> tuple[Config | None, MetricsRecord | None, list[dict]]:
    configs = get_search_space()
    best_config, best_record, best_score = None, None, -1.0
    results = []

    for config in configs:
        record = measure_fn(config)
        feasible = check_slo(record, slo)
        score = efficiency_score(record) if feasible else 0.0
        results.append({
            "config": {"q": config.q, "b": config.b, "p": config.p},
            "score": score,
            "feasible": feasible,
            "record": record
        })
        if score > best_score:
            best_score, best_config, best_record = score, config, record

    return best_config, best_record, results
