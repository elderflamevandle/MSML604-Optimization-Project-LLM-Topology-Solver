from src.vidur.grid_search import run_grid_search
from src.metrics_collector import collect_metrics_mock
from src.search_space import get_search_space


def test_grid_search_returns_best_config():
    best_config, best_record, results = run_grid_search(collect_metrics_mock)
    assert best_config is not None
    assert best_record is not None


def test_grid_search_covers_full_space():
    _, _, results = run_grid_search(collect_metrics_mock)
    assert len(results) == len(get_search_space())


def test_grid_results_have_required_keys():
    _, _, results = run_grid_search(collect_metrics_mock)
    for r in results:
        assert "config" in r
        assert "score" in r
        assert "feasible" in r
