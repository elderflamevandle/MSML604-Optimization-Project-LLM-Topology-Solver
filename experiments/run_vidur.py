import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.search_space import Config
from src.metrics_collector import collect_metrics_mock, save_results
from src.vidur.grid_search import run_grid_search
from src.vidur.bayesian_search import run_bayesian_search


def main(use_gpu: bool = False, model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> None:
    if use_gpu:
        from src.vllm_runner import run_vllm_benchmark
        prompts = [
            "Explain the concept of machine learning in simple terms.",
            "What are the main differences between Python and Java?",
            "How does a neural network learn from data?",
            "Describe the process of natural language processing.",
            "What is the significance of the transformer architecture in AI?",
        ] * 4  # 20 prompts total for a quick local test
        measure_fn = lambda config: run_vllm_benchmark(config, prompts, model_id=model_id)
    else:
        measure_fn = collect_metrics_mock

    print("=== Grid Search ===")
    grid_best, grid_record, grid_results = run_grid_search(measure_fn)
    if grid_best is not None:
        print(f"Best: q={grid_best.q}, b={grid_best.b}, p={grid_best.p} "
              f"| QPS={grid_record.qps:.2f}, delay_p99={grid_record.delay_p99_ms:.1f}ms")
    else:
        print("No feasible config found within SLO constraints")

    print("\n=== Bayesian Search (30 trials) ===")
    bayes_best, bayes_record, bayes_results = run_bayesian_search(measure_fn, n_trials=30)
    if bayes_best is not None:
        print(f"Best: q={bayes_best.q}, b={bayes_best.b}, p={bayes_best.p} "
              f"| QPS={bayes_record.qps:.2f}, delay_p99={bayes_record.delay_p99_ms:.1f}ms")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_results([r["record"] for r in grid_results], os.path.join(results_dir, "vidur_grid.csv"))
    save_results([r["record"] for r in bayes_results], os.path.join(results_dir, "vidur_bayes.csv"))
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()
    main(use_gpu=args.gpu, model_id=args.model)
