# FlexGen Optimizer Comparison — Design Spec

**Date:** 2026-05-02
**Status:** Draft (awaiting user review)
**Course:** MSML604 — Convex Optimization
**Branch:** `chaitanya`

## 1. Goal

Add a measurable novel contribution on top of the existing faithful-FlexGen
implementation by:

1. Generalizing the inner placement LP from **uniform-across-layers** to
   **per-layer placement** (Layer 1 — math novelty).
2. Solving the same per-layer FlexGen problem with **five distinct optimization
   techniques** and reporting wall-time vs solution-quality (Layer 2 — optimizer
   comparison, the headline novelty).
3. Validating the predicted per-token latency against **real TinyLlama-1.1B-Chat
   inference** on the local RTX 4050 (Layer 3 — empirical validation).

The deliverable is a class-presentation-ready report and figure showing that
- per-layer placement reduces predicted latency vs the FlexGen-paper uniform LP, and
- different convex / global optimization techniques produce comparable solution
  quality but with very different wall-time and code-elegance trade-offs.

## 2. Non-goals

- Modifying model weights, training, or fine-tuning.
- Stochastic / robust extension (parked, see brainstorming log).
- Multi-objective / Pareto front (parked).
- Replacing the existing FlexGen entry points (`experiments/run_flexgen.py`,
  `pipeline.py`); they continue to work unchanged.

## 3. Background — what already exists

| Path | Role |
|---|---|
| [src/flexgen/cost_model.py](../../../src/flexgen/cost_model.py) | Per-layer + block latency formulas (compute, weight load, KV I/O, activation I/O, with overlap). |
| [src/flexgen/lp_formulation.py](../../../src/flexgen/lp_formulation.py) | Inner LP with **uniform-across-layers** 9-fraction placement, PuLP solver. |
| [src/flexgen/policy_search.py](../../../src/flexgen/policy_search.py) | Outer enumeration over 480 (gbs, num_gb, q, delegate, overlap) points. |
| [src/flexgen/baseline_compare.py](../../../src/flexgen/baseline_compare.py) | Compares optimized policy vs 5 hand-picked baseline policies (same solver, same cost model). |
| [experiments/run_flexgen.py](../../../experiments/run_flexgen.py) | Main CLI. |
| [src/flexgen/qwen_inference.py](../../../src/flexgen/qwen_inference.py) | GPU text-generation runner used for real-inference experiments. |

**The repo does not currently compare different optimizers on the FlexGen
problem itself.** Existing comparisons are policy-vs-baseline, not
optimizer-vs-optimizer.

## 4. Layer 1 — Per-layer placement LP

### 4.1 Variables

For each layer i ∈ {1, …, L} and category c ∈ {weights, kv_cache, activations}
and tier t ∈ {gpu, cpu, disk}:

```
x_{i,c,t} ∈ [0, 1]
```

with the simplex constraint per (i, c):

```
x_{i,c,gpu} + x_{i,c,cpu} + x_{i,c,disk} = 1   ∀ i ∈ [L], c ∈ {w, kv, h}
```

Total continuous variables: **9 · L** (e.g., 32 layers × 9 = 288 for Llama-3-8B).

### 4.2 Capacity constraints

Let `Wi`, `Ki(b, S)`, `Hi(b, S)` denote per-layer byte costs for weights, KV
cache, and activations respectively (functions of compression `q`, batch
`b = gbs · num_gb`, and sequence length `S = prompt_len + decode_len`).

```
Σ_i Wi · x_{i,w,gpu}  +  Σ_i Ki · x_{i,kv,gpu}  +  Σ_i Hi · x_{i,h,gpu}  ≤  GPU_VRAM_bytes
Σ_i Wi · x_{i,w,cpu}  +  Σ_i Ki · x_{i,kv,cpu}  +  Σ_i Hi · x_{i,h,cpu}  ≤  RAM_bytes
Σ_i Wi · x_{i,w,disk} +  Σ_i Ki · x_{i,kv,disk} +  Σ_i Hi · x_{i,h,disk} ≤  DISK_bytes
```

### 4.3 Per-layer cost terms

For fixed enumerated point E = (gbs, num_gb, q, delegate, overlap), define for
each layer i:

```
compute_i      = f_compute(E, x_{i,*,*}, coef)        # depends on delegate flag
weight_load_i  = f_w_io  (E, x_{i,w,*}, coef)
kv_io_i        = f_kv_io (E, x_{i,kv,*}, coef)
act_io_i       = f_act_io(E, x_{i,h,*}, coef)
```

These are linear in x given E (the enumerated point fixes the discrete dims).

### 4.4 Objective

```
overlap = True:
    minimize  T_block = Σ_i τ_i
    subject to τ_i ≥ compute_i, τ_i ≥ weight_load_i, τ_i ≥ kv_io_i, τ_i ≥ act_io_i
    (epigraph reformulation of max — keeps it LP)

overlap = False:
    minimize  T_block = Σ_i (compute_i + weight_load_i + kv_io_i + act_io_i)
```

Per-token latency: `T_block / (gbs · num_gb)`.

### 4.5 New module

`src/flexgen/lp_per_layer.py` — exposes `solve_inner_lp_per_layer(enum, cap,
spec, wl, coef) -> PerLayerResult`. Existing
[lp_formulation.py](../../../src/flexgen/lp_formulation.py) is **kept
unchanged**; per-layer is an additional formulation, not a replacement.

The result type carries 9·L fractions plus a uniform-projection helper that
collapses per-layer fractions to the closest 9-vector (for back-compat
visualization).

## 5. Layer 2 — Five optimizers, one problem

### 5.1 Common interface

```python
# src/flexgen/optimizers/base.py
@dataclass(frozen=True)
class OptimizerResult:
    name: str                          # "enum_lp", "lp_relax", "milp", "cvxpy", "bo_optuna"
    enum: EnumPoint                    # discrete vars chosen
    placement: PerLayerPlacement       # 9·L fractions
    t_block_s: float
    per_token_latency_ms: float
    wall_time_s: float
    iterations: int                    # solver iterations / trials / branch nodes
    feasible: bool
    notes: dict[str, Any]              # solver-specific diagnostics

class Optimizer(Protocol):
    name: str
    def solve(self, spec: ModelSpec, cap: LiveCapacity, coef: SystemCoefficients,
              wl: WorkloadSpec) -> OptimizerResult: ...
```

All five optimizers solve the **same Layer 1 per-layer LP problem**.

### 5.2 O1 — Enumeration × LP (PuLP)

`src/flexgen/optimizers/enum_lp.py`. Wraps current behavior but invokes the
per-layer LP. Loops over all 480 enumerated points, solves the per-layer LP for
each, returns the minimum.

Expected: highest solution quality (exhaustive over outer); wall time ≈ N · (LP
solve time per point).

### 5.3 O2 — LP relaxation + rounding

`src/flexgen/optimizers/lp_relax.py`. Strategy:

1. **Split on compression.** The byte coefficients (Wi, Ki) depend on `q ∈
   {fp16, int4}` in a way that cannot be linearly relaxed without losing
   tightness. So solve two relaxed LPs — one with q fixed to `fp16`, one with
   q fixed to `int4`.
2. **Within each compression-fixed LP**, relax `gbs ∈ [1, 32]`, `num_gb ∈ [1,
   16]` to continuous, and relax `delegate, overlap ∈ [0, 1]`. The cost
   becomes nonlinear in `gbs · num_gb` (batch dimension multiplies KV/act
   bytes) — linearize via a McCormick envelope on `gbs · num_gb` (or
   equivalently a 6 × 5 piecewise-linear grid).
3. **Solve** the relaxed LP. Read out (gbs*, num_gb*, delegate*, overlap*).
4. **Round** to the nearest enumerated values: `gbs* → nearest in {1,2,4,8,16,32}`,
   `num_gb* → nearest in {1,2,4,8,16}`, `delegate*, overlap* → nearest in {0,
   1}`. Re-solve the **per-layer LP** at the rounded enum point to get a
   feasible placement.
5. Compare the two compression candidates and return the better one.

Total LP solves: 2 relaxed + 2 re-solves at rounded points = **4 LPs** (vs O1's 480).

Expected: faster than O1, but with a relaxation gap. Pedagogical value:
demonstrates *how much* the LP relaxation loses vs exhaustive enumeration —
the gap is the headline number from this optimizer.

### 5.4 O3 — MILP one-shot (Pyomo + CBC)

`src/flexgen/optimizers/milp.py`. Encode all 480 enumerated points as binary
indicators z_p ∈ {0, 1} with `Σ_p z_p = 1`. For each enumerated point p,
attach its per-layer placement variables and capacity / cost constraints
**conditionally on z_p** using big-M:

```
For each constraint  A_p · x_p ≤ b_p  in enum point p:
    A_p · x_p ≤ b_p + M · (1 - z_p)
```

Objective: `Σ_p z_p · t_block_p` where `t_block_p` is itself a continuous
linear expression in `x_p`. Solver: Pyomo + CBC.

Expected: same global optimum as O1 (it's the same problem), but slower —
B&B is overkill when the outer space is small and exhaustive enumeration is
cheap. Pedagogical value: shows decomposition beats monolithic for
small-outer-space problems.

### 5.5 O4 — Direct convex program in CVXPY

`src/flexgen/optimizers/cvxpy_direct.py`. Reformulate the inner per-layer LP
in CVXPY's DCP grammar. The overlap case uses `cp.maximum(...)` natively (no
explicit epigraph), and CVXPY converts it to ECOS / SCS internally:

```python
tau = cp.Variable(L)
constraints += [tau >= compute, tau >= weight_load, tau >= kv_io, tau >= act_io]
objective = cp.Minimize(cp.sum(tau))
```

Wraps the outer enumeration in a Python loop. Same global optimum as O1.

Pedagogical value: course-style demonstration that DCP makes the formulation
trivial; PuLP requires explicit epigraph reformulation by hand.

### 5.6 O5 — Bayesian Optimization (Optuna)

`src/flexgen/optimizers/bo_optuna.py`. TPE sampler over the discrete outer
space (480 points). Each trial:

1. Sampler suggests `(gbs, num_gb, q, delegate, overlap)`.
2. Inner per-layer LP is solved (PuLP, same as O1).
3. Trial value = `per_token_latency_ms`; infeasible LP returns `+∞`.

Budget: **50 trials** (~10× fewer than 480). Expected: near-optimal (within
~2% of O1) at ~10× speedup. Pedagogical value: surrogate-model search over a
small categorical space.

### 5.7 Optimizer comparison harness

`experiments/run_optimizer_comparison.py`:

- CLI flags mirror `run_flexgen.py`: `--model`, `--workload`, `--output-dir`.
- Runs all five optimizers on the same `(spec, cap, coef, wl)` tuple.
- Writes `experiments/results/optimizer_comparison_<model_slug>_<ts>.json`:

```json
{
  "timestamp": "...",
  "machine_id": "...",
  "input": { "system": {...}, "model": {...}, "workload": {...} },
  "optimizers": [
    {
      "name": "enum_lp",
      "enum": {...},
      "placement_summary": { "weights_avg": [g, c, d], "kv_avg": [...], "act_avg": [...] },
      "objective": { "per_token_latency_ms": ..., "throughput_tok_s": ... },
      "wall_time_s": ...,
      "iterations": ...,
      "feasible": true,
      "gap_to_best_pct": 0.0,
      "notes": {...}
    },
    ...
  ],
  "best_optimizer_name": "enum_lp"
}
```

- Logs to `experiments/logs/optimizer_comparison_<ts>.log`.

### 5.8 Plot

`analysis/plot_optimizer_comparison.py`:

- Scatter: x = `wall_time_s` (log scale), y = `per_token_latency_ms`. One
  point per optimizer. Annotated.
- Bar chart: `gap_to_best_pct` per optimizer.
- Output: `analysis/plots/optimizer_comparison_<model_slug>.png`.

## 6. Layer 3 — Empirical validation on TinyLlama

### 6.1 Setup

- Model: `models/tinyllama-1.1b-chat` (already local).
- Hardware: RTX 4050 Laptop, CUDA 12.1, `venv312` (per project memory).
- Workload: `prompt_len=512`, `decode_len=128` (default `configs/workload.yaml`).

### 6.2 Procedure

`experiments/run_validation.py`:

1. Pick the **best policy from O1** and the **`manual_all_gpu_fp16_b1_no_overlap`
   baseline** from `baseline_compare.py`.
2. For each, call `qwen_inference.py` (extended to accept a policy dict) to
   generate text on **10 fixed prompts** of length ~512 tokens each, decoding
   128 tokens.
3. Measure wall-clock ms/token (averaged across the 10 prompts, excluding the
   first as warm-up).
4. Compare measured ms/token to predicted ms/token from each optimizer.

### 6.3 Output

`experiments/results/validation_<ts>.json`:

```json
{
  "policies": [
    { "name": "enum_lp_best",
      "predicted_ms_per_token": 84.3,
      "measured_ms_per_token": 91.7,
      "error_pct": 8.07,
      "n_prompts": 10 },
    { "name": "manual_naive",
      "predicted_ms_per_token": ...,
      "measured_ms_per_token": ...,
      "error_pct": ... }
  ]
}
```

This is the credibility section of the report: *the LP doesn't just produce
numbers, it produces numbers that match reality.*

## 7. File structure

```
src/flexgen/
    lp_per_layer.py                       # NEW — per-layer LP
    optimizers/
        __init__.py                       # NEW
        base.py                           # NEW — Optimizer protocol + result type
        enum_lp.py                        # NEW — O1
        lp_relax.py                       # NEW — O2
        milp.py                           # NEW — O3
        cvxpy_direct.py                   # NEW — O4
        bo_optuna.py                      # NEW — O5
experiments/
    run_optimizer_comparison.py           # NEW
    run_validation.py                     # NEW
analysis/
    plot_optimizer_comparison.py          # NEW
tests/flexgen/
    test_lp_per_layer.py                  # NEW
    test_optimizers/
        test_base.py                      # NEW
        test_enum_lp.py                   # NEW
        test_lp_relax.py                  # NEW
        test_milp.py                      # NEW
        test_cvxpy_direct.py              # NEW
        test_bo_optuna.py                 # NEW
        test_consistency.py               # NEW — cross-optimizer property tests
    test_run_optimizer_comparison.py      # NEW
    test_run_validation.py                # NEW
docs/superpowers/specs/
    2026-05-02-flexgen-optimizer-comparison-design.md  # this file
docs/superpowers/plans/
    2026-05-02-flexgen-optimizer-comparison.md         # implementation plan (next step)
```

Existing files **unchanged**: `lp_formulation.py`, `policy_search.py`,
`run_flexgen.py`, `baseline_compare.py`, `pipeline.py`, etc.

## 8. New dependencies

| Package | Reason |
|---|---|
| `cvxpy` | O4 — DCP grammar |
| `pyomo` | O3 — MILP modeling |
| `coincbc` (or system CBC) | O3 — MILP solver |
| `optuna` | O5 — Bayesian Optimization (already in `requirements.txt` — no new add) |

To be added to `requirements.txt`:

```
cvxpy>=1.5.0
pyomo>=6.7.0
```

CBC: already available because `pulp` (used by current FlexGen LP) bundles
the CBC binary — Pyomo can be pointed at the same binary via
`SolverFactory("cbc", executable=pulp.PULP_CBC_CMD().path)`. No new system
install required on Windows or Linux.

## 9. Testing strategy (rigid TDD per project conventions)

### 9.1 Layer 1 unit tests

- `test_lp_per_layer.py`:
  - **Reduces to uniform**: when L = 1 OR all layers have identical
    coefficients, per-layer LP must equal uniform LP within 1e-6.
  - **Capacity respected**: for any random feasible workload, all 9·L
    fractions in [0, 1], category sums = 1, capacity sums ≤ limits.
  - **Monotonicity**: increasing GPU VRAM cannot worsen objective.
  - **Improvement**: on a heterogeneous-layer fixture (fake spec with varying
    `Wi`), per-layer LP must beat uniform LP by ≥ 0.1%.

### 9.2 Layer 2 cross-optimizer consistency tests

- `test_consistency.py`:
  - On a small fixture (Qwen2-1.5B-style spec, fixed system, fixed workload):
    - O1, O3, O4 must agree on best per-token latency within 1% (all three
      are exact methods on the same convex problem).
    - O5 (BO) must be within 5% of O1.
    - O2 (relaxation+rounding) is allowed to be worse than O1 (relaxation
      gap), but must be feasible and within 50% of O1.
  - All optimizers must report the same `feasible=False` when capacity is
    set artificially tiny.

### 9.3 Per-optimizer unit tests

Each optimizer module gets at least:
- "happy path on small fixture returns OptimizerResult with feasible=True"
- "infeasible workload returns feasible=False, no exception"
- "wall_time_s > 0, iterations > 0"

### 9.4 End-to-end tests

- `test_run_optimizer_comparison.py`: run the harness on a mocked HF spec,
  assert the output JSON has all 5 optimizer entries with the expected schema,
  assert `gap_to_best_pct = 0` for the best optimizer.
- `test_run_validation.py`: mocks `qwen_inference.py` to return a deterministic
  ms/token value; asserts validation JSON shape and that `error_pct` is
  computed correctly.

### 9.5 Coverage target

Maintain or improve current `tests/flexgen/` pass rate (47+ tests, ~40 s).
After this work, expect ~80 tests, ≤ 90 s total.

## 10. Output artifacts (what the professor sees)

For each model in {TinyLlama-1.1B, Qwen2-1.5B, Mistral-7B}:

1. **Per-layer vs uniform comparison plot** —
   `analysis/plots/per_layer_vs_uniform_<model>.png`. Bar chart of
   per-token latency.
2. **Optimizer comparison plot** —
   `analysis/plots/optimizer_comparison_<model>.png`. Scatter of wall-time vs
   solution quality.
3. **Validation table** (TinyLlama only, hardware-bound) — predicted vs
   measured ms/token for `enum_lp_best` and `manual_naive`.
4. **One-page write-up** at `report/optimizer_comparison_summary.md` (sibling
   to the existing `report/outline.md`) summarizing findings, suitable for
   inclusion in the final project report.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| CBC fails to install on Windows | Use `pulp.PULP_CBC_CMD()` (already used in O1 — bundled). Document fallback to `glpk` for Pyomo. |
| CVXPY DCP rejects formulation | Pre-validate with a 2-layer toy spec in a unit test before full integration. |
| Optuna finds infeasible-only points first | Set initial trials to seed with the corner (gbs=1, num_gb=1) to guarantee at least one feasible trial; document a `--bo-warmup` flag. |
| MILP big-M makes the problem numerically ill-conditioned | Bound M tightly to per-constraint scale (max possible RHS); warn in logs if any constraint's effective M > 1e6. |
| Per-layer LP doesn't actually beat uniform on Llama-style models | Acceptable result — report it honestly. The novelty is the *formulation and comparison*, not a guaranteed win. |
| Real TinyLlama inference shows the cost model is wildly off | Investigate root cause (likely calibration); adjust calibration constants. The validation experiment exists *to find* this. |

## 12. Out of scope (parked for future work)

- Stochastic / robust LP (workload-distribution version of Layer 1).
- Multi-objective / Pareto front (latency × energy × memory).
- Mixed-precision per-tensor quantization.
- Genetic Algorithm / Simulated Annealing optimizers (would be O6/O7 if added).
- Trace-driven workload sampling.

These are all viable extensions but would each warrant their own design spec.

## 13. Implementation order (preview, full plan in next doc)

1. Layer 1 — `lp_per_layer.py` and tests (gives us the new problem).
2. `optimizers/base.py` — interface contract.
3. O1 — wraps existing search to produce `OptimizerResult`.
4. O4 — easiest sanity check (CVXPY directly mirrors O1).
5. O5 — BO (low risk, mostly Optuna boilerplate).
6. O3 — MILP (highest risk, do it after O1/O4 are proven).
7. O2 — LP relaxation + rounding.
8. Cross-optimizer consistency tests.
9. `run_optimizer_comparison.py` + plotting.
10. Layer 3 — `run_validation.py` + TinyLlama runs on RTX 4050.
11. Final write-up.

Detailed task breakdown will live at
`docs/superpowers/plans/2026-05-02-flexgen-optimizer-comparison.md` (next
deliverable).
