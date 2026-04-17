from src.vidur.bayesian_search import run_bayesian_search
from src.metrics_collector import collect_metrics_mock
from src.search_space import QUANTIZATIONS, BATCH_SIZES, PARALLELISMS


def test_bayesian_returns_valid_config():
    best_config, best_record, results = run_bayesian_search(collect_metrics_mock, n_trials=10)
    assert best_config is not None
    assert best_config.q in QUANTIZATIONS
    assert best_config.b in BATCH_SIZES
    assert best_config.p in PARALLELISMS


def test_bayesian_runs_correct_number_of_trials():
    _, _, results = run_bayesian_search(collect_metrics_mock, n_trials=10)
    assert len(results) == 10


def test_bayesian_results_have_required_keys():
    _, _, results = run_bayesian_search(collect_metrics_mock, n_trials=5)
    for r in results:
        assert "config" in r
        assert "score" in r
        assert "feasible" in r
