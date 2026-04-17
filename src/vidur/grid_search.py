from typing import Callable

from src.metrics_collector import MetricsRecord
from src.search_space import Config, get_search_space
from src.vidur.slo_checker import check_slo, efficiency_score


def run_grid_search(
    measure_fn: Callable[[Config], MetricsRecord],
    slo: dict | None = None,
) -> tuple[Config | None, MetricsRecord | None, list[dict]]:
    configs = get_search_space()
    best_config: Config | None = None
    best_record: MetricsRecord | None = None
    best_score = -1.0
    results = []

    for config in configs:
        record = measure_fn(config)
        feasible = check_slo(record, slo)
        score = efficiency_score(record) if feasible else 0.0
        results.append({
            "config": {"q": config.q, "b": config.b, "p": config.p},
            "score": score,
            "feasible": feasible,
            "record": record,
        })
        if feasible and score > best_score:
            best_score = score
            best_config = config
            best_record = record

    return best_config, best_record, results
