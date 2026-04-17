import optuna
from typing import Callable

from src.metrics_collector import MetricsRecord
from src.search_space import Config, QUANTIZATIONS, BATCH_SIZES, PARALLELISMS
from src.vidur.slo_checker import check_slo, efficiency_score


def run_bayesian_search(
    measure_fn: Callable[[Config], MetricsRecord],
    n_trials: int = 30,
    slo: dict | None = None,
) -> tuple[Config | None, MetricsRecord | None, list[dict]]:
    results = []

    def objective(trial: optuna.Trial) -> float:
        q = trial.suggest_categorical("q", QUANTIZATIONS)
        b = trial.suggest_categorical("b", BATCH_SIZES)
        p = trial.suggest_categorical("p", PARALLELISMS)
        config = Config(q=q, b=b, p=p)
        record = measure_fn(config)
        feasible = check_slo(record, slo)
        score = efficiency_score(record) if feasible else 0.0
        results.append({
            "config": {"q": q, "b": b, "p": p},
            "score": score,
            "feasible": feasible,
            "record": record,
        })
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_config = Config(q=best.params["q"], b=best.params["b"], p=best.params["p"])
    best_record = measure_fn(best_config)
    return best_config, best_record, results
