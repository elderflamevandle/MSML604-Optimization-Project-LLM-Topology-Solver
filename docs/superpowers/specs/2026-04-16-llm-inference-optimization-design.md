# Design Spec: Optimization-Driven Tuning of LLM Inference Configurations

**Date:** 2026-04-16  
**Deadline:** 2026-04-23 (1 week)  
**Team:** 3 members  
**Deliverables:** Codebase + Experimental Results + Full Report + Slides

---

## Problem Statement

Find the optimal LLM inference configuration over a compact decision vector `x = (q, b, p)`:
- `q` — quantization/precision: `{fp16, int8, int4}`
- `b` — batch size / max-tokens: `{1, 4, 8, 16, 32}`
- `p` — parallelism (tensor-parallel degree): `{1, 2}`

No single configuration dominates across all objectives. The project benchmarks three optimization formulations and compares their recommended configurations via trade-off curves.

---

## System Architecture

```
[ Datasets ]          ShareGPT_Vicuna_unfiltered + Vidur Synthetic Traces
      ↓
[ Serving Layer ]     vLLM on GPU server — LLaMA-3 8B (primary)
      ↓
[ Metrics Collector ] TTFT(p99), TBT(p99), Delay(p99), QPS, memory_gb
      ↓
[ Optimizer Layer ]
    ├── Track 1 (Vidur):   Grid Search + Bayesian Optimization (Optuna)
    ├── Track 2 (FlexGen): LP solver — scipy/PuLP for memory placement
    └── Track 3 (Helix):   MILP solver — PuLP for max-flow GPU placement
      ↓
[ Analysis ]          Trade-off curves, solver comparison, deployment recommendation
```

All three tracks output results in the same shared metrics schema.

---

## Optimization Formulations

### Track 1 — Vidur Constrained Search (Black-Box)
**Owner:** User

Maximize QPS/Cost ratio subject to SLO constraints:
```
max  QPSmax(x) / Cost(x)
s.t. TTFT_p99(x) ≤ τ1
     TBT_p99(x)  ≤ τ2
     Delay_p99(x) ≤ τ3
```
**Solvers:** Grid search (baseline) + Optuna Bayesian optimization  
**SLO thresholds:** τ1=2s, τ2=200ms, τ3=10s (adjustable after smoke tests)

### Track 2 — FlexGen Memory Placement LP
**Owner:** Person 2

Minimize inference latency by optimally splitting weights/KV-cache/activations across GPU, CPU, and disk:
```
min  T / (b·l·s)
s.t. GPU_peak_mem ≤ GPU_cap
     CPU_peak_mem ≤ CPU_cap
     Disk_peak_mem ≤ Disk_cap
     w_g + w_c + w_d = 1
     c_g + c_c + c_d = 1
     h_g + h_c + h_d = 1
```
**Solver:** scipy.optimize.linprog or PuLP

### Track 3 — Helix Max-Flow MILP
**Owner:** Person 3

Maximize feasible request flow through a heterogeneous GPU cluster:
```
max  Σ f_source,i
s.t. Σ f_u,i = Σ f_i,v   (flow conservation)
     Σ f_u,i ≤ Σ b_i^j * T_j
     f_i,j ≤ d_i,j * S_i,j
```
**Solver:** PuLP with CBC (or Gurobi if available)

---

## Datasets

| Dataset | Source | Use |
|---------|--------|-----|
| ShareGPT_Vicuna_unfiltered | HuggingFace | Real conversation traces for latency benchmarking |
| Vidur Synthetic Workload Traces | Microsoft GitHub | Controlled load patterns for throughput benchmarking |

---

## Shared Metrics Schema

Every experiment records:
```python
{
  "config": {"q": str, "b": int, "p": int},
  "ttft_p99_ms": float,
  "tbt_p99_ms": float,
  "delay_p99_ms": float,
  "qps": float,
  "memory_gb": float,
  "cost_proxy": float,   # b * p as proxy until real hardware cost confirmed
  "dataset": str,
  "model": str,
  "timestamp": str
}
```

---

## Codebase Structure

```
optimization_project/
├── data/
│   ├── sharegpt_vicuna/
│   └── vidur_traces/
├── src/
│   ├── metrics_collector.py
│   ├── search_space.py
│   ├── vidur/
│   │   ├── grid_search.py
│   │   ├── bayesian_search.py
│   │   └── slo_checker.py
│   ├── flexgen/
│   │   ├── lp_formulation.py
│   │   └── memory_solver.py
│   └── helix/
│       ├── milp_formulation.py
│       └── flow_solver.py
├── experiments/
│   ├── run_vidur.py
│   ├── run_flexgen.py
│   ├── run_helix.py
│   └── results/
├── analysis/
│   ├── plot_tradeoffs.py
│   ├── compare_solvers.py
│   └── recommendation.py
├── report/
├── slides/
└── requirements.txt
```

---

## Day-by-Day Plan

### Day 1 — Apr 16 | Shared Foundation (No GPU)
- Set up Python environment: vLLM, PuLP, scipy, Optuna, pandas, matplotlib
- Download + preprocess both datasets
- Define and document the shared metrics schema
- Write `metrics_collector.py` with mock mode for CPU testing
- Define full (q, b, p) search space in `search_space.py`

### Day 2 — Apr 17 | Parallel Implementation (No GPU)
Each member implements their formulation:

| Track 1 (You) | Track 2 | Track 3 |
|---|---|---|
| Grid search over (q,b,p) | LP in PuLP/scipy | MILP in PuLP |
| Optuna Bayesian optimizer | Memory tier constraints | Max-flow + placement |
| SLO constraint checker | Solve + extract placement | Solve + extract flow |
| Dry-run with mock metrics | Unit test with synthetic caps | Unit test with synthetic graph |

**Exit criterion:** All three `run_*.py` scripts execute end-to-end with mocked metrics.

### Day 3 — Apr 18 | GPU Arrives + Integration
- Connect to GPU server, install vLLM, load LLaMA-3 8B
- Smoke test: 10 requests → confirm metrics flow into collector
- Run one full pass per track, fix integration bugs

### Day 4 — Apr 19 | Full Experiments
- Run complete (q, b, p) grid on LLaMA-3 8B, both datasets
- Save all raw results to `experiments/results/`
- If time allows: subset run on larger model (LLaMA-3 70B or Mistral 7B quantized)

### Day 5 — Apr 20 | Analysis + Visualizations
- Trade-off curves: QPS vs latency, memory vs throughput, cost vs quality
- Cross-formulation comparison table
- Final deployment recommendation for one scenario

### Day 6 — Apr 21 | Report Writing
- Sections: Introduction, Problem Formulation, Methodology, Results, Discussion, Recommendation, References
- One subsection per formulation in Methods
- All plots + tables embedded

### Day 7 — Apr 22–23 | Slides + Buffer
- 15–20 slide deck from report
- Rehearse results narrative
- Buffer: re-run failed experiments, polish plots, final edits

---

## Hardware + Stack

| Component | Choice |
|-----------|--------|
| Primary model | LLaMA-3 8B |
| Serving framework | vLLM |
| GPU | Server GPU (TBD, arrives Day 3) |
| LP solver | scipy.optimize.linprog + PuLP/CBC |
| MILP solver | PuLP/CBC (Gurobi if available) |
| Black-box optimizer | Optuna (TPE sampler) |
| Language | Python 3.10+ |

---

## Risks + Mitigations

| Risk | Mitigation |
|------|-----------|
| GPU arrives late (past Day 3) | All code mock-testable on CPU; experiments compress into Days 4–5 |
| Experiment runtime too long | Reduce (q,b,p) grid; use Optuna's early pruning |
| MILP solver too slow for Helix | Reduce cluster graph size; use CBC with time limit |
| Team misalignment on metrics format | Shared schema defined Day 1, validated before GPU work starts |
