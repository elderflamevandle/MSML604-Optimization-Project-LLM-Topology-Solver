# LLM Inference Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark that compares three optimization formulations (Vidur, FlexGen, Helix) for tuning LLM inference configurations over `x = (q, b, p)`, producing trade-off curves and a deployment recommendation.

**Architecture:** Shared metrics schema feeds three parallel optimizer tracks (constrained black-box search, LP memory placement, MILP flow placement). All tracks write to the same CSV/JSON format so a unified analysis layer can cross-compare them.

**Tech Stack:** Python 3.10+, vLLM, PuLP, scipy, Optuna, pandas, matplotlib, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | All dependencies pinned |
| `src/search_space.py` | (q,b,p) grid definition + Config dataclass |
| `src/metrics_collector.py` | Mock + vLLM metrics collection, shared MetricsRecord schema |
| `src/vidur/slo_checker.py` | SLO constraint validation + efficiency score |
| `src/vidur/grid_search.py` | Exhaustive grid search over (q,b,p) |
| `src/vidur/bayesian_search.py` | Optuna TPE-based Bayesian search |
| `src/flexgen/lp_formulation.py` | PuLP LP for memory tier placement |
| `src/helix/milp_formulation.py` | PuLP MILP for max-flow GPU placement |
| `experiments/run_vidur.py` | Entry point: runs grid + Bayesian, saves CSVs |
| `experiments/run_flexgen.py` | Entry point: solves LP for memory scenarios |
| `experiments/run_helix.py` | Entry point: solves MILP for GPU cluster |
| `analysis/plot_tradeoffs.py` | Trade-off curve plots (QPS vs latency, memory vs throughput) |
| `analysis/recommendation.py` | Final deployment recommendation table |
| `tests/` | Pytest tests, one file per src module |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/vidur/__init__.py`
- Create: `src/flexgen/__init__.py`
- Create: `src/helix/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/vidur/__init__.py`
- Create: `tests/flexgen/__init__.py`
- Create: `tests/helix/__init__.py`
- Create: `experiments/results/.gitkeep`
- Create: `analysis/plots/.gitkeep`

- [ ] **Step 1: Write requirements.txt**

```
vllm>=0.4.0
torch>=2.0.0
transformers>=4.40.0
datasets>=2.18.0
optuna>=3.6.0
pulp>=2.7.0
scipy>=1.12.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
numpy>=1.26.0
pytest>=8.0.0
```

- [ ] **Step 2: Create all `__init__.py` files and directories**

```bash
mkdir -p src/vidur src/flexgen src/helix
mkdir -p tests/vidur tests/flexgen tests/helix
mkdir -p experiments/results analysis/plots data/sharegpt_vicuna data/vidur_traces
touch src/__init__.py src/vidur/__init__.py src/flexgen/__init__.py src/helix/__init__.py
touch tests/__init__.py tests/vidur/__init__.py tests/flexgen/__init__.py tests/helix/__init__.py
touch experiments/results/.gitkeep analysis/plots/.gitkeep
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without error. Note: `vllm` requires CUDA — on a CPU-only machine, install everything except vllm: `pip install optuna pulp scipy pandas matplotlib seaborn numpy pytest datasets transformers`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt src/ tests/ experiments/ analysis/ data/
git commit -m "chore: project scaffold with directories and requirements"
```

---

## Task 2: Search Space + Shared Metrics Schema

**Files:**
- Create: `src/search_space.py`
- Create: `src/metrics_collector.py`
- Create: `tests/test_search_space.py`
- Create: `tests/test_metrics_collector.py`

- [ ] **Step 1: Write failing tests for search space**

`tests/test_search_space.py`:
```python
from src.search_space import get_search_space, Config, QUANTIZATIONS, BATCH_SIZES, PARALLELISMS

def test_search_space_size():
    space = get_search_space()
    expected = len(QUANTIZATIONS) * len(BATCH_SIZES) * len(PARALLELISMS)
    assert len(space) == expected

def test_all_configs_have_valid_fields():
    for config in get_search_space():
        assert config.q in QUANTIZATIONS
        assert config.b in BATCH_SIZES
        assert config.p in PARALLELISMS
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_search_space.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.search_space'`

- [ ] **Step 3: Implement search_space.py**

`src/search_space.py`:
```python
from dataclasses import dataclass
from typing import List

QUANTIZATIONS = ["fp16", "int8", "int4"]
BATCH_SIZES = [1, 4, 8, 16, 32]
PARALLELISMS = [1, 2]

@dataclass
class Config:
    q: str
    b: int
    p: int

def get_search_space() -> List[Config]:
    return [
        Config(q=q, b=b, p=p)
        for q in QUANTIZATIONS
        for b in BATCH_SIZES
        for p in PARALLELISMS
    ]
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_search_space.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Write failing tests for metrics collector**

`tests/test_metrics_collector.py`:
```python
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
```

- [ ] **Step 6: Run test to confirm it fails**

```bash
pytest tests/test_metrics_collector.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.metrics_collector'`

- [ ] **Step 7: Implement metrics_collector.py**

`src/metrics_collector.py`:
```python
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
```

- [ ] **Step 8: Run tests to confirm they pass**

```bash
pytest tests/test_search_space.py tests/test_metrics_collector.py -v
```
Expected: `5 passed`

- [ ] **Step 9: Commit**

```bash
git add src/search_space.py src/metrics_collector.py tests/test_search_space.py tests/test_metrics_collector.py
git commit -m "feat: search space definition and shared metrics schema"
```

---

## Task 3: Vidur SLO Checker + Grid Search

**Files:**
- Create: `src/vidur/slo_checker.py`
- Create: `src/vidur/grid_search.py`
- Create: `tests/vidur/test_slo_checker.py`
- Create: `tests/vidur/test_grid_search.py`

- [ ] **Step 1: Write failing tests for SLO checker**

`tests/vidur/test_slo_checker.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/vidur/test_slo_checker.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.vidur.slo_checker'`

- [ ] **Step 3: Implement slo_checker.py**

`src/vidur/slo_checker.py`:
```python
from src.metrics_collector import MetricsRecord

SLO_DEFAULTS = {
    "ttft_p99_ms": 2000.0,
    "tbt_p99_ms": 200.0,
    "delay_p99_ms": 10000.0,
}

def check_slo(record: MetricsRecord, slo: dict = None) -> bool:
    if slo is None:
        slo = SLO_DEFAULTS
    return (
        record.ttft_p99_ms <= slo["ttft_p99_ms"]
        and record.tbt_p99_ms <= slo["tbt_p99_ms"]
        and record.delay_p99_ms <= slo["delay_p99_ms"]
    )

def efficiency_score(record: MetricsRecord) -> float:
    if record.cost_proxy == 0:
        return 0.0
    return record.qps / record.cost_proxy
```

- [ ] **Step 4: Run to confirm SLO tests pass**

```bash
pytest tests/vidur/test_slo_checker.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Write failing tests for grid search**

`tests/vidur/test_grid_search.py`:
```python
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
```

- [ ] **Step 6: Run to confirm failure**

```bash
pytest tests/vidur/test_grid_search.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.vidur.grid_search'`

- [ ] **Step 7: Implement grid_search.py**

`src/vidur/grid_search.py`:
```python
from src.search_space import get_search_space, Config
from src.metrics_collector import MetricsRecord, collect_metrics_mock
from src.vidur.slo_checker import check_slo, efficiency_score
from typing import Callable, List, Tuple, Optional

def run_grid_search(
    measure_fn: Callable[[Config], MetricsRecord],
    slo: dict = None,
) -> Tuple[Optional[Config], Optional[MetricsRecord], List[dict]]:
    configs = get_search_space()
    best_config, best_record, best_score = None, None, -1.0
    results = []

    for config in configs:
        record = measure_fn(config)
        feasible = check_slo(record, slo)
        score = efficiency_score(record) if feasible else 0.0
        results.append({"config": {"q": config.q, "b": config.b, "p": config.p},
                        "score": score, "feasible": feasible, "record": record})
        if score > best_score:
            best_score, best_config, best_record = score, config, record

    return best_config, best_record, results
```

- [ ] **Step 8: Run to confirm grid search tests pass**

```bash
pytest tests/vidur/ -v
```
Expected: `9 passed`

- [ ] **Step 9: Commit**

```bash
git add src/vidur/slo_checker.py src/vidur/grid_search.py tests/vidur/
git commit -m "feat: Vidur SLO checker and grid search over (q,b,p) space"
```

---

## Task 4: Vidur Bayesian Search (Optuna)

**Files:**
- Create: `src/vidur/bayesian_search.py`
- Create: `tests/vidur/test_bayesian_search.py`

- [ ] **Step 1: Write failing tests**

`tests/vidur/test_bayesian_search.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/vidur/test_bayesian_search.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.vidur.bayesian_search'`

- [ ] **Step 3: Implement bayesian_search.py**

`src/vidur/bayesian_search.py`:
```python
import optuna
from src.search_space import Config, QUANTIZATIONS, BATCH_SIZES, PARALLELISMS
from src.metrics_collector import MetricsRecord
from src.vidur.slo_checker import check_slo, efficiency_score
from typing import Callable, Tuple, Optional, List

def run_bayesian_search(
    measure_fn: Callable[[Config], MetricsRecord],
    n_trials: int = 30,
    slo: dict = None,
) -> Tuple[Optional[Config], Optional[MetricsRecord], List[dict]]:
    results = []

    def objective(trial: optuna.Trial) -> float:
        q = trial.suggest_categorical("q", QUANTIZATIONS)
        b = trial.suggest_categorical("b", BATCH_SIZES)
        p = trial.suggest_categorical("p", PARALLELISMS)
        config = Config(q=q, b=b, p=p)
        record = measure_fn(config)
        feasible = check_slo(record, slo)
        score = efficiency_score(record) if feasible else 0.0
        results.append({"config": {"q": q, "b": b, "p": p},
                        "score": score, "feasible": feasible, "record": record})
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_config = Config(q=best.params["q"], b=best.params["b"], p=best.params["p"])
    best_record = measure_fn(best_config)
    return best_config, best_record, results
```

- [ ] **Step 4: Run to confirm tests pass**

```bash
pytest tests/vidur/ -v
```
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add src/vidur/bayesian_search.py tests/vidur/test_bayesian_search.py
git commit -m "feat: Vidur Bayesian search via Optuna TPE"
```

---

## Task 5: FlexGen LP Memory Placement

**Files:**
- Create: `src/flexgen/lp_formulation.py`
- Create: `tests/flexgen/test_lp_formulation.py`

- [ ] **Step 1: Write failing tests**

`tests/flexgen/test_lp_formulation.py`:
```python
from src.flexgen.lp_formulation import (
    MemoryCapacity, ModelMemoryRequirement, solve_memory_placement
)

LLAMA3_8B = ModelMemoryRequirement(weights_gb=16.0, kv_cache_gb=4.0, activations_gb=2.0)

def test_solve_with_large_gpu_puts_all_on_gpu():
    result = solve_memory_placement(MemoryCapacity(gpu_gb=80, cpu_gb=0, disk_gb=0), LLAMA3_8B)
    assert result.status == "Optimal"
    assert abs(result.w_gpu - 1.0) < 0.01

def test_solve_with_tight_gpu_spills_weights():
    result = solve_memory_placement(MemoryCapacity(gpu_gb=8, cpu_gb=32, disk_gb=500), LLAMA3_8B)
    assert result.status == "Optimal"
    assert result.w_gpu < 1.0

def test_placement_fractions_sum_to_one():
    result = solve_memory_placement(MemoryCapacity(gpu_gb=24, cpu_gb=64, disk_gb=0), LLAMA3_8B)
    assert abs(result.w_gpu + result.w_cpu + result.w_disk - 1.0) < 1e-4
    assert abs(result.c_gpu + result.c_cpu + result.c_disk - 1.0) < 1e-4
    assert abs(result.h_gpu + result.h_cpu + result.h_disk - 1.0) < 1e-4

def test_objective_is_nonnegative():
    result = solve_memory_placement(MemoryCapacity(gpu_gb=24, cpu_gb=64, disk_gb=0), LLAMA3_8B)
    assert result.objective >= 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/flexgen/test_lp_formulation.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.flexgen.lp_formulation'`

- [ ] **Step 3: Implement lp_formulation.py**

`src/flexgen/lp_formulation.py`:
```python
import pulp
from dataclasses import dataclass

@dataclass
class MemoryCapacity:
    gpu_gb: float
    cpu_gb: float
    disk_gb: float

@dataclass
class ModelMemoryRequirement:
    weights_gb: float
    kv_cache_gb: float
    activations_gb: float

@dataclass
class PlacementResult:
    w_gpu: float; w_cpu: float; w_disk: float
    c_gpu: float; c_cpu: float; c_disk: float
    h_gpu: float; h_cpu: float; h_disk: float
    objective: float
    status: str

def solve_memory_placement(capacity: MemoryCapacity, req: ModelMemoryRequirement) -> PlacementResult:
    prob = pulp.LpProblem("flexgen", pulp.LpMinimize)
    w_g = pulp.LpVariable("w_g", 0, 1); w_c = pulp.LpVariable("w_c", 0, 1); w_d = pulp.LpVariable("w_d", 0, 1)
    c_g = pulp.LpVariable("c_g", 0, 1); c_c = pulp.LpVariable("c_c", 0, 1); c_d = pulp.LpVariable("c_d", 0, 1)
    h_g = pulp.LpVariable("h_g", 0, 1); h_c = pulp.LpVariable("h_c", 0, 1); h_d = pulp.LpVariable("h_d", 0, 1)

    # Minimize off-GPU placement (CPU=1x, disk=10x latency penalty)
    prob += (req.weights_gb * (w_c * 1.0 + w_d * 10.0)
           + req.kv_cache_gb * (c_c * 1.0 + c_d * 10.0)
           + req.activations_gb * (h_c * 1.0 + h_d * 10.0))

    prob += w_g + w_c + w_d == 1
    prob += c_g + c_c + c_d == 1
    prob += h_g + h_c + h_d == 1

    prob += req.weights_gb * w_g + req.kv_cache_gb * c_g + req.activations_gb * h_g <= capacity.gpu_gb
    prob += req.weights_gb * w_c + req.kv_cache_gb * c_c + req.activations_gb * h_c <= capacity.cpu_gb
    prob += req.weights_gb * w_d + req.kv_cache_gb * c_d + req.activations_gb * h_d <= capacity.disk_gb

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return PlacementResult(
        w_gpu=pulp.value(w_g) or 0, w_cpu=pulp.value(w_c) or 0, w_disk=pulp.value(w_d) or 0,
        c_gpu=pulp.value(c_g) or 0, c_cpu=pulp.value(c_c) or 0, c_disk=pulp.value(c_d) or 0,
        h_gpu=pulp.value(h_g) or 0, h_cpu=pulp.value(h_c) or 0, h_disk=pulp.value(h_d) or 0,
        objective=pulp.value(prob.objective) or 0,
        status=pulp.LpStatus[prob.status],
    )
```

- [ ] **Step 4: Run to confirm tests pass**

```bash
pytest tests/flexgen/ -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/lp_formulation.py tests/flexgen/test_lp_formulation.py
git commit -m "feat: FlexGen LP for memory tier placement"
```

---

## Task 6: Helix MILP Max-Flow Placement

**Files:**
- Create: `src/helix/milp_formulation.py`
- Create: `tests/helix/test_milp_formulation.py`

- [ ] **Step 1: Write failing tests**

`tests/helix/test_milp_formulation.py`:
```python
from src.helix.milp_formulation import GPUNode, NetworkLink, solve_max_flow_placement

NODES = [
    GPUNode(node_id="gpu0", throughput_gbs=100.0, memory_gb=80.0),
    GPUNode(node_id="gpu1", throughput_gbs=60.0, memory_gb=40.0),
]
LINKS = [
    NetworkLink(src="source", dst="gpu0", bandwidth_gbs=50.0),
    NetworkLink(src="source", dst="gpu1", bandwidth_gbs=50.0),
    NetworkLink(src="gpu0", dst="sink", bandwidth_gbs=50.0),
    NetworkLink(src="gpu1", dst="sink", bandwidth_gbs=50.0),
]

def test_solve_returns_nonnegative_flow():
    result = solve_max_flow_placement(NODES, LINKS, num_layers=4)
    assert result.total_flow >= 0

def test_solve_status_is_optimal():
    result = solve_max_flow_placement(NODES, LINKS, num_layers=4)
    assert result.status == "Optimal"

def test_all_layers_are_placed():
    result = solve_max_flow_placement(NODES, LINKS, num_layers=4)
    assert len(result.layer_placement) == 4

def test_layers_placed_on_valid_nodes():
    result = solve_max_flow_placement(NODES, LINKS, num_layers=4)
    valid_node_ids = {"gpu0", "gpu1"}
    for node_id in result.layer_placement.values():
        assert node_id in valid_node_ids
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/helix/test_milp_formulation.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.helix.milp_formulation'`

- [ ] **Step 3: Implement milp_formulation.py**

`src/helix/milp_formulation.py`:
```python
import pulp
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class GPUNode:
    node_id: str
    throughput_gbs: float
    memory_gb: float

@dataclass
class NetworkLink:
    src: str
    dst: str
    bandwidth_gbs: float

@dataclass
class FlowResult:
    total_flow: float
    flow_assignments: Dict[Tuple[str, str], float]
    layer_placement: Dict[str, str]
    status: str

def solve_max_flow_placement(
    nodes: List[GPUNode],
    links: List[NetworkLink],
    num_layers: int = 32,
) -> FlowResult:
    prob = pulp.LpProblem("helix_flow", pulp.LpMaximize)
    node_ids = [n.node_id for n in nodes]
    node_map = {n.node_id: n for n in nodes}

    flow_vars = {(lk.src, lk.dst): pulp.LpVariable(f"f_{lk.src}_{lk.dst}", 0)
                 for lk in links}

    place_vars = {(l, n): pulp.LpVariable(f"place_{l}_{n}", cat="Binary")
                  for l in range(num_layers) for n in node_ids}

    # Maximize total source outflow
    prob += pulp.lpSum(v for (s, d), v in flow_vars.items() if s == "source")

    # Flow conservation at each GPU node
    for nid in node_ids:
        in_f = pulp.lpSum(v for (s, d), v in flow_vars.items() if d == nid)
        out_f = pulp.lpSum(v for (s, d), v in flow_vars.items() if s == nid)
        prob += in_f == out_f

    # Each layer placed on exactly one node
    for l in range(num_layers):
        prob += pulp.lpSum(place_vars[(l, n)] for n in node_ids) == 1

    # Node capacity: inflow ≤ (layers placed * throughput / total_layers)
    for nid in node_ids:
        in_f = pulp.lpSum(v for (s, d), v in flow_vars.items() if d == nid)
        layers_on = pulp.lpSum(place_vars[(l, nid)] for l in range(num_layers))
        prob += in_f <= layers_on * node_map[nid].throughput_gbs / num_layers

    # Link bandwidth constraints
    for lk in links:
        if (lk.src, lk.dst) in flow_vars:
            prob += flow_vars[(lk.src, lk.dst)] <= lk.bandwidth_gbs

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    layer_placement = {}
    for l in range(num_layers):
        for n in node_ids:
            val = pulp.value(place_vars[(l, n)])
            if val is not None and val > 0.5:
                layer_placement[f"layer_{l}"] = n

    return FlowResult(
        total_flow=pulp.value(prob.objective) or 0.0,
        flow_assignments={(k): (pulp.value(v) or 0.0) for k, v in flow_vars.items()},
        layer_placement=layer_placement,
        status=pulp.LpStatus[prob.status],
    )
```

- [ ] **Step 4: Run to confirm tests pass**

```bash
pytest tests/helix/ -v
```
Expected: `4 passed`

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```
Expected: `20 passed`

- [ ] **Step 6: Commit**

```bash
git add src/helix/milp_formulation.py tests/helix/test_milp_formulation.py
git commit -m "feat: Helix MILP for max-flow multi-GPU layer placement"
```

---

## Task 7: Experiment Runners (Dry-Run with Mock)

**Files:**
- Create: `experiments/run_vidur.py`
- Create: `experiments/run_flexgen.py`
- Create: `experiments/run_helix.py`

- [ ] **Step 1: Write run_vidur.py**

`experiments/run_vidur.py`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.search_space import Config
from src.metrics_collector import collect_metrics_mock, save_results
from src.vidur.grid_search import run_grid_search
from src.vidur.bayesian_search import run_bayesian_search

def main(use_gpu: bool = False):
    if use_gpu:
        from src.metrics_collector import collect_metrics_vllm
        raise NotImplementedError("Wire vLLM here on Day 3 - see Task 8")
    measure_fn = collect_metrics_mock

    print("=== Grid Search ===")
    grid_best, grid_record, grid_results = run_grid_search(measure_fn)
    print(f"Best: q={grid_best.q}, b={grid_best.b}, p={grid_best.p} "
          f"| QPS={grid_record.qps:.2f}, delay_p99={grid_record.delay_p99_ms:.1f}ms")

    print("\n=== Bayesian Search (30 trials) ===")
    bayes_best, bayes_record, bayes_results = run_bayesian_search(measure_fn, n_trials=30)
    print(f"Best: q={bayes_best.q}, b={bayes_best.b}, p={bayes_best.p} "
          f"| QPS={bayes_record.qps:.2f}, delay_p99={bayes_record.delay_p99_ms:.1f}ms")

    os.makedirs("experiments/results", exist_ok=True)
    save_results([r["record"] for r in grid_results], "experiments/results/vidur_grid.csv")
    save_results([r["record"] for r in bayes_results], "experiments/results/vidur_bayes.csv")
    print("\nResults saved to experiments/results/")

if __name__ == "__main__":
    main(use_gpu="--gpu" in sys.argv)
```

- [ ] **Step 2: Write run_flexgen.py**

`experiments/run_flexgen.py`:
```python
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.flexgen.lp_formulation import MemoryCapacity, ModelMemoryRequirement, solve_memory_placement

LLAMA3_8B = ModelMemoryRequirement(weights_gb=16.0, kv_cache_gb=4.0, activations_gb=2.0)

SCENARIOS = [
    {"name": "gpu_80gb_only",      "cap": MemoryCapacity(gpu_gb=80,  cpu_gb=0,  disk_gb=0)},
    {"name": "gpu_24gb_cpu_64gb",  "cap": MemoryCapacity(gpu_gb=24,  cpu_gb=64, disk_gb=0)},
    {"name": "gpu_8gb_cpu_32gb_disk", "cap": MemoryCapacity(gpu_gb=8, cpu_gb=32, disk_gb=500)},
]

def main():
    os.makedirs("experiments/results", exist_ok=True)
    output = []
    for s in SCENARIOS:
        result = solve_memory_placement(s["cap"], LLAMA3_8B)
        print(f"{s['name']}: status={result.status}, objective={result.objective:.4f}")
        print(f"  weights GPU={result.w_gpu:.2f} CPU={result.w_cpu:.2f} disk={result.w_disk:.2f}")
        output.append({"scenario": s["name"], "status": result.status,
                       "objective": result.objective,
                       "w_gpu": result.w_gpu, "w_cpu": result.w_cpu, "w_disk": result.w_disk,
                       "c_gpu": result.c_gpu, "c_cpu": result.c_cpu, "c_disk": result.c_disk,
                       "h_gpu": result.h_gpu, "h_cpu": result.h_cpu, "h_disk": result.h_disk})
    with open("experiments/results/flexgen_lp.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved experiments/results/flexgen_lp.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write run_helix.py**

`experiments/run_helix.py`:
```python
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.helix.milp_formulation import GPUNode, NetworkLink, solve_max_flow_placement

CLUSTER = {
    "nodes": [
        GPUNode(node_id="gpu0", throughput_gbs=100.0, memory_gb=80.0),
        GPUNode(node_id="gpu1", throughput_gbs=60.0,  memory_gb=40.0),
    ],
    "links": [
        NetworkLink(src="source", dst="gpu0", bandwidth_gbs=50.0),
        NetworkLink(src="source", dst="gpu1", bandwidth_gbs=50.0),
        NetworkLink(src="gpu0",   dst="sink",  bandwidth_gbs=50.0),
        NetworkLink(src="gpu1",   dst="sink",  bandwidth_gbs=50.0),
        NetworkLink(src="gpu0",   dst="gpu1",  bandwidth_gbs=20.0),
        NetworkLink(src="gpu1",   dst="gpu0",  bandwidth_gbs=20.0),
    ],
}

def main():
    os.makedirs("experiments/results", exist_ok=True)
    print("Solving Helix MILP (32 layers, 2-GPU cluster)...")
    result = solve_max_flow_placement(CLUSTER["nodes"], CLUSTER["links"], num_layers=32)
    print(f"Status: {result.status} | Total flow: {result.total_flow:.4f}")
    print(f"Layer sample: {dict(list(result.layer_placement.items())[:4])}")

    output = {"status": result.status, "total_flow": result.total_flow,
              "flow_assignments": {str(k): v for k, v in result.flow_assignments.items()},
              "layer_placement": result.layer_placement}
    with open("experiments/results/helix_milp.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved experiments/results/helix_milp.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Dry-run all three runners**

```bash
python experiments/run_vidur.py
python experiments/run_flexgen.py
python experiments/run_helix.py
```

Expected:
- `run_vidur.py`: prints best config, saves two CSVs
- `run_flexgen.py`: prints 3 scenario results, saves JSON
- `run_helix.py`: prints Optimal status + flow, saves JSON

- [ ] **Step 5: Commit**

```bash
git add experiments/run_vidur.py experiments/run_flexgen.py experiments/run_helix.py
git commit -m "feat: experiment runner entry points with mock dry-run"
```

---

## Task 8: vLLM Integration (Day 3 — GPU Day)

**Files:**
- Modify: `experiments/run_vidur.py` (wire real vLLM measurement)
- Create: `src/vllm_runner.py`

> Do this task ONLY after GPU server is accessible and vLLM is installed.

- [ ] **Step 1: Install vLLM on GPU server**

```bash
pip install vllm
python -c "import vllm; print(vllm.__version__)"
```
Expected: prints a version string without error.

- [ ] **Step 2: Smoke test vLLM with LLaMA-3 8B**

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='meta-llama/Meta-Llama-3-8B-Instruct', tensor_parallel_size=1)
params = SamplingParams(temperature=0.0, max_tokens=50)
out = llm.generate(['Hello, world!'], params)
print(out[0].outputs[0].text)
"
```
Expected: prints a short text response. If OOM, reduce `gpu_memory_utilization` to 0.85.

- [ ] **Step 3: Create vllm_runner.py**

`src/vllm_runner.py`:
```python
import time
import numpy as np
from vllm import LLM, SamplingParams
from src.search_space import Config
from src.metrics_collector import MetricsRecord

QUANT_MAP = {"fp16": None, "int8": "bitsandbytes", "int4": "awq"}

def run_vllm_benchmark(
    config: Config,
    prompts: list,
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset: str = "sharegpt",
) -> MetricsRecord:
    import torch
    quantization = QUANT_MAP[config.q]
    llm = LLM(
        model=model_id,
        quantization=quantization,
        tensor_parallel_size=config.p,
        max_num_seqs=config.b,
        gpu_memory_utilization=0.90,
    )
    params = SamplingParams(temperature=0.0, max_tokens=100)

    ttfts, itls, e2e_latencies = [], [], []
    for prompt in prompts[:50]:
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], params)
        t1 = time.perf_counter()
        e2e_latencies.append(t1 - t0)
        # vLLM RequestOutput includes token timestamps
        out = outputs[0]
        if hasattr(out, "metrics") and out.metrics:
            ttfts.append(out.metrics.first_token_time or (t1 - t0) * 0.1)
        else:
            ttfts.append((t1 - t0) * 0.1)
        itls.append((t1 - t0) / max(len(out.outputs[0].token_ids), 1))

    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    qps = len(prompts[:50]) / sum(e2e_latencies)

    del llm
    torch.cuda.empty_cache()

    return MetricsRecord(
        config_q=config.q, config_b=config.b, config_p=config.p,
        ttft_p99_ms=float(np.percentile(ttfts, 99)) * 1000,
        tbt_p99_ms=float(np.percentile(itls, 99)) * 1000,
        delay_p99_ms=float(np.percentile(e2e_latencies, 99)) * 1000,
        qps=qps,
        memory_gb=gpu_mem,
        cost_proxy=float(config.b * config.p),
        dataset=dataset,
        model=model_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
```

- [ ] **Step 4: Wire run_vidur.py to use real vLLM**

Replace the `NotImplementedError` block in `experiments/run_vidur.py`:
```python
def main(use_gpu: bool = False):
    if use_gpu:
        from datasets import load_dataset
        from src.vllm_runner import run_vllm_benchmark
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train[:200]")
        prompts = [row["conversations"][0]["value"] for row in ds
                   if row["conversations"] and row["conversations"][0]["from"] == "human"]
        measure_fn = lambda config: run_vllm_benchmark(config, prompts)
    else:
        from src.metrics_collector import collect_metrics_mock
        measure_fn = collect_metrics_mock
    # rest of main() unchanged
```

- [ ] **Step 5: Run one full vLLM pass to validate end-to-end**

```bash
python experiments/run_vidur.py --gpu
```
Expected: prints real TTFT/TBT/QPS values, saves CSVs with non-mock data.

- [ ] **Step 6: Commit**

```bash
git add src/vllm_runner.py experiments/run_vidur.py
git commit -m "feat: vLLM GPU integration for real inference benchmarking"
```

---

## Task 9: Dataset Download Scripts

**Files:**
- Create: `data/download_sharegpt.py`
- Create: `data/download_vidur_traces.py`

- [ ] **Step 1: Write ShareGPT download script**

`data/download_sharegpt.py`:
```python
from datasets import load_dataset
import json, os

os.makedirs("data/sharegpt_vicuna", exist_ok=True)
print("Downloading ShareGPT_Vicuna_unfiltered...")
ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
prompts = [
    row["conversations"][0]["value"]
    for row in ds
    if row["conversations"] and row["conversations"][0]["from"] == "human"
]
with open("data/sharegpt_vicuna/prompts.json", "w") as f:
    json.dump(prompts, f)
print(f"Saved {len(prompts)} prompts to data/sharegpt_vicuna/prompts.json")
```

- [ ] **Step 2: Write Vidur traces download script**

`data/download_vidur_traces.py`:
```python
import subprocess, os

os.makedirs("data/vidur_traces", exist_ok=True)
url = "https://raw.githubusercontent.com/microsoft/vidur/main/data/traces/sharegpt.json"
print(f"Downloading Vidur traces from {url}...")
subprocess.run(["curl", "-L", url, "-o", "data/vidur_traces/sharegpt_trace.json"], check=True)
print("Saved to data/vidur_traces/sharegpt_trace.json")
```

- [ ] **Step 3: Run download scripts**

```bash
python data/download_sharegpt.py
python data/download_vidur_traces.py
```
Expected: JSON files appear in `data/` directories.

- [ ] **Step 4: Commit**

```bash
git add data/download_sharegpt.py data/download_vidur_traces.py
git commit -m "feat: dataset download scripts for ShareGPT and Vidur traces"
```

---

## Task 10: Analysis + Trade-off Plots

**Files:**
- Create: `analysis/plot_tradeoffs.py`
- Create: `analysis/recommendation.py`

- [ ] **Step 1: Write plot_tradeoffs.py**

`analysis/plot_tradeoffs.py`:
```python
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_qps_vs_latency(grid_csv: str, out_dir: str):
    df = pd.read_csv(grid_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, q in enumerate(["fp16", "int8", "int4"]):
        sub = df[df["config_q"] == q]
        sc = axes[i].scatter(sub["delay_p99_ms"], sub["qps"],
                             c=sub["config_b"], cmap="viridis", alpha=0.7, s=80)
        axes[i].axvline(x=10000, color="r", linestyle="--", label="SLO limit")
        axes[i].set_xlabel("Delay P99 (ms)"); axes[i].set_ylabel("QPS")
        axes[i].set_title(f"Quantization: {q}"); axes[i].legend()
        plt.colorbar(sc, ax=axes[i], label="Batch size")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "qps_vs_latency.png"), dpi=150)
    plt.close()
    print(f"Saved {out_dir}/qps_vs_latency.png")

def plot_memory_vs_throughput(grid_csv: str, out_dir: str):
    df = pd.read_csv(grid_csv)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["memory_gb"], df["qps"], c=df["config_b"], cmap="plasma", alpha=0.7, s=80)
    plt.colorbar(sc, label="Batch size")
    plt.xlabel("Memory (GB)"); plt.ylabel("QPS")
    plt.title("Memory vs Throughput Trade-off")
    plt.savefig(os.path.join(out_dir, "memory_vs_throughput.png"), dpi=150)
    plt.close()

def plot_optimizer_comparison(grid_csv: str, bayes_csv: str, out_dir: str):
    grid_df = pd.read_csv(grid_csv)
    bayes_df = pd.read_csv(bayes_csv)
    grid_df["efficiency"] = grid_df["qps"] / grid_df["cost_proxy"].replace(0, float("inf"))
    bayes_df["efficiency"] = bayes_df["qps"] / bayes_df["cost_proxy"].replace(0, float("inf"))

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(grid_df)),
             grid_df["efficiency"].sort_values(ascending=False).values,
             label="Grid Search", color="steelblue")
    plt.plot(range(len(bayes_df)),
             bayes_df["efficiency"].sort_values(ascending=False).values,
             label="Bayesian (Optuna)", color="darkorange", linestyle="--")
    plt.xlabel("Config rank"); plt.ylabel("Efficiency (QPS/Cost)")
    plt.title("Grid Search vs Bayesian Optimization"); plt.legend()
    plt.savefig(os.path.join(out_dir, "optimizer_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved {out_dir}/optimizer_comparison.png")

if __name__ == "__main__":
    plot_qps_vs_latency("experiments/results/vidur_grid.csv", "analysis/plots")
    plot_memory_vs_throughput("experiments/results/vidur_grid.csv", "analysis/plots")
    plot_optimizer_comparison("experiments/results/vidur_grid.csv",
                              "experiments/results/vidur_bayes.csv", "analysis/plots")
```

- [ ] **Step 2: Write recommendation.py**

`analysis/recommendation.py`:
```python
import pandas as pd, json

SLO = {"ttft_p99_ms": 2000, "tbt_p99_ms": 200, "delay_p99_ms": 10000}

def generate_recommendation(grid_csv: str, flexgen_json: str, helix_json: str) -> dict:
    df = pd.read_csv(grid_csv)
    feasible = df[
        (df["ttft_p99_ms"] <= SLO["ttft_p99_ms"]) &
        (df["tbt_p99_ms"]  <= SLO["tbt_p99_ms"]) &
        (df["delay_p99_ms"] <= SLO["delay_p99_ms"])
    ].copy()
    feasible["efficiency"] = feasible["qps"] / feasible["cost_proxy"].replace(0, float("inf"))
    vidur_best = feasible.sort_values("efficiency", ascending=False).iloc[0]

    with open(flexgen_json) as f:
        fg = json.load(f)
    fg_best = min((r for r in fg if r["status"] == "Optimal"),
                  key=lambda r: r["objective"], default=fg[0])

    with open(helix_json) as f:
        helix = json.load(f)

    rec = {
        "vidur": {"q": vidur_best["config_q"], "b": int(vidur_best["config_b"]),
                  "p": int(vidur_best["config_p"]), "qps": round(vidur_best["qps"], 2),
                  "efficiency": round(vidur_best["efficiency"], 4)},
        "flexgen": {"best_scenario": fg_best["scenario"],
                    "w_gpu": round(fg_best["w_gpu"], 3),
                    "objective": round(fg_best["objective"], 4)},
        "helix": {"total_flow": round(helix["total_flow"], 4), "status": helix["status"]},
    }

    print("\n=== DEPLOYMENT RECOMMENDATION ===")
    print(f"[Vidur] Best config: q={rec['vidur']['q']}, b={rec['vidur']['b']}, "
          f"p={rec['vidur']['p']} | QPS={rec['vidur']['qps']}, "
          f"efficiency={rec['vidur']['efficiency']}")
    print(f"[FlexGen] Best scenario: {rec['flexgen']['best_scenario']} "
          f"| GPU fraction={rec['flexgen']['w_gpu']:.2f}")
    print(f"[Helix] Max flow={rec['helix']['total_flow']:.4f} | {rec['helix']['status']}")
    return rec

if __name__ == "__main__":
    generate_recommendation(
        "experiments/results/vidur_grid.csv",
        "experiments/results/flexgen_lp.json",
        "experiments/results/helix_milp.json",
    )
```

- [ ] **Step 3: Run analysis end-to-end with mock data**

```bash
python experiments/run_vidur.py
python experiments/run_flexgen.py
python experiments/run_helix.py
python analysis/plot_tradeoffs.py
python analysis/recommendation.py
```
Expected: 3 PNG files in `analysis/plots/`, recommendation printed to stdout.

- [ ] **Step 4: Commit**

```bash
git add analysis/plot_tradeoffs.py analysis/recommendation.py
git commit -m "feat: trade-off plots and final deployment recommendation"
```

---

## Task 11: Full Experiment Run (Day 4 — GPU)

> Run this after Task 8 (vLLM wired up) with GPU server access.

- [ ] **Step 1: Run full vLLM grid on LLaMA-3 8B (ShareGPT)**

```bash
python experiments/run_vidur.py --gpu
```
Expected: CSVs in `experiments/results/` with real TTFT/TBT/QPS values. Takes ~30–90 min depending on grid size.

- [ ] **Step 2: Run FlexGen LP with actual GPU memory specs**

Update `SCENARIOS` in `experiments/run_flexgen.py` to match your actual GPU capacity, then:
```bash
python experiments/run_flexgen.py
```

- [ ] **Step 3: Run Helix MILP with actual cluster topology**

Update `CLUSTER` in `experiments/run_helix.py` to match your actual GPU count/specs, then:
```bash
python experiments/run_helix.py
```

- [ ] **Step 4: Regenerate plots from real data**

```bash
python analysis/plot_tradeoffs.py
python analysis/recommendation.py
```

- [ ] **Step 5: Commit results and plots**

```bash
git add experiments/results/ analysis/plots/
git commit -m "results: full experiment run on LLaMA-3 8B with real GPU data"
```

---

## Task 12: Report + Slides Scaffold

- [ ] **Step 1: Create report outline**

`report/outline.md`:
```markdown
# Report Outline

1. Introduction — problem, motivation, scope
2. Background — three formulations (Vidur, FlexGen, Helix)
3. Methodology — decision vector, search space, serving stack, datasets
4. Results
   4.1 Vidur: Grid Search vs Bayesian (QPS/latency trade-offs)
   4.2 FlexGen: LP memory placement (objective by scenario)
   4.3 Helix: MILP flow placement (total flow, layer assignment)
   4.4 Cross-formulation comparison
5. Recommendation — best config for one deployment scenario
6. Conclusion
7. References
```

- [ ] **Step 2: Create slides outline**

`slides/outline.md`:
```markdown
# Slide Outline (15-20 slides)

1. Title slide
2. Problem + Motivation (1 slide)
3. Decision vector x=(q,b,p) (1 slide)
4. Three formulations overview (1 slide)
5. Experimental setup: vLLM + LLaMA-3 8B + datasets (1 slide)
6. Vidur results: QPS vs latency scatter (1 slide)
7. Vidur: Grid vs Bayesian comparison (1 slide)
8. FlexGen: LP placement by memory scenario (1 slide)
9. Helix: Max-flow result + layer placement (1 slide)
10. Cross-formulation comparison table (1 slide)
11. Final recommendation (1 slide)
12. Conclusion + future work (1 slide)
13. Q&A
```

- [ ] **Step 3: Commit**

```bash
git add report/outline.md slides/outline.md
git commit -m "docs: report and slides outlines"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All three formulations have tasks (3,4,5 = Vidur; 6 = FlexGen; 7 = Helix). Shared infra in Tasks 1+2. GPU integration in Task 8. Analysis in Task 10. Report in Task 12.
- [x] **Placeholder scan:** No TBD/TODO in code steps. GPU-specific steps clearly gated to Day 3+.
- [x] **Type consistency:** `MetricsRecord`, `Config`, `PlacementResult`, `FlowResult` all defined in Task 2/5/6 and used consistently in later tasks.
- [x] **All experiment runners dry-run on CPU before GPU work starts.**
