# FlexGen Project — Novelty & Comparison Ideas

**Date:** 2026-05-02
**Context:** MSML604 (Convex Optimization) class project. Repo already implements
faithful FlexGen LP-based policy search (480-point outer enumeration + 9-variable
inner LP) for `meta-llama/Meta-Llama-3-8B`, `mistralai/Mistral-7B-v0.1`,
`Qwen/Qwen2-1.5B`, plus local TinyLlama smoke tests. Question is what novelty
to add for the final deliverable / what alternate optimizer to compare against.

## Q1 (open) — What kind of novelty does the professor want: math/theory, or experimental engineering?

### Options surfaced

**Math / theory novelty (strong fits for a convex-optimization course):**

- **A. Per-layer (non-uniform) placement** — FlexGen forces one set of placement
  fractions across all transformer layers. Replace with per-layer variables
  `(w_g_i, w_c_i, w_d_i)` and either:
    - Solve as an extended LP (3·L extra variables, still convex, still tractable), or
    - Solve via dynamic programming over layers with bandwidth/capacity state.
  Genuine math novelty over the FlexGen paper. Cleanly framed in convex-opt language.

- **D. Stochastic / robust LP for workload uncertainty** — FlexGen optimizes for a
  single fixed `(prompt_len, decode_len)`. Real workloads are distributions.
  Two formulations:
    - **Stochastic**: minimize expected per-token latency over a workload
      distribution (sample average approximation with N scenarios → still LP).
    - **Robust / DRO**: min-max over a Wasserstein ball around the empirical
      distribution → still convex (SOCP / LP dual).
  This is the kind of contribution professors love because the math is rigorous
  and the motivation is real.

- **B. Multi-objective Pareto: latency × energy × memory-headroom** — convert the
  scalar objective into a vector and trace the Pareto front via NSGA-II (`pymoo`)
  or weighted-sum sweep. FlexGen paper does single-objective. Visually striking
  deliverable (Pareto plot per model).

- **G. Mixed-precision per-tensor / per-layer quantization** — current code has a
  binary `fp16` vs `int4` switch shared across all components. Extend to allow
  different components (weights / KV / activations) or different layers to pick
  different precisions. Still LP-shaped, but more variables and more interesting.

**Experimental / comparison novelty:**

- **C. Bayesian Optimization (Optuna) over outer + LP inner — vs. full enumeration**
  — direct apples-to-apples speedup measurement. Easy lift, clean comparison story.

- **E. MINLP (Pyomo + Bonmin/Couenne) one-shot, all 14 variables jointly** —
  monolithic vs. decomposed. Tells the professor "we tried solving the whole thing
  together; here's why decomposition wins." Risk: MINLP solvers are flaky.

- **F. Empirical validation: predicted vs. measured latency** — actually run
  inference under the LP-recommended policy using the existing `qwen_inference.py`
  and `models/tinyllama-1.1b-chat`, compare predicted ms/token to measured ms/token.
  Turns the project from a math exercise into measured science. Low effort, high
  presentation value.

- **H. Neural surrogate** — train a small MLP on LP outputs over many synthetic
  (system, model, workload) inputs to replace the solver. ML-replaces-optimizer
  angle. Cute but probably weak as a *convex-optimization* class deliverable.

### Recommendation (initial)

For a **convex-optimization course**, the best storyline is **one math contribution
+ one empirical validation**:

1. **A (per-layer LP) or D (stochastic/robust LP)** as the math novelty — both are
   genuine convex-opt extensions of FlexGen, and both are paper-grade.
2. **F (predicted-vs-measured)** as the empirical validation — leverages
   `qwen_inference.py` and the local TinyLlama already wired up.
3. **C (BO vs enumeration)** as a cheap sanity-check comparison if time permits.

Of (A) and (D), my pick is **(D) stochastic/robust** because (i) it answers a real
weakness in the FlexGen paper, (ii) it gives you a clean convex-opt formulation
(SAA → LP, or DRO → SOCP), and (iii) it produces a striking result: "we picked a
policy that's robust to ±50% workload variation, FlexGen's optimum is X% slower
under shifted workloads."

## Q2 — Are we already comparing different optimizers in the repo?

**Finding:** No. Two existing comparisons, neither is "different optimizers on the same
FlexGen problem":

1. **`src/flexgen/baseline_compare.py`** — compares the LP-optimized policy against
   5 hand-picked baseline policies (`manual_all_gpu_fp16_b1_no_overlap`,
   `lp_fixed_*`). Same PuLP solver, same cost model, only the fixed enumeration
   varies. This is "we beat the heuristic", not optimizer comparison.
2. **`run_all.py`** — runs the FlexGen policy search. Not a multi-optimizer comparison.

So multi-optimizer comparison on the FlexGen problem is a real gap.

## Decision

User picked **option (ii) — clean convex formulation + tangible result + real novelty**.
Plan has three layers:

- **Layer 1 — Math novelty:** Per-layer (non-uniform) placement LP. Strictly
  generalizes FlexGen. 9·L vars instead of 9, still convex, still seconds to solve.
- **Layer 2 — Optimizer comparison (headline novelty):** Solve the same
  per-layer FlexGen problem with five techniques and compare wall time + solution
  quality:
    - **O1**: Outer enumeration × inner LP (current FlexGen, PuLP). Baseline.
    - **O2**: Pure LP relaxation + rounding. Demonstrates relaxation gap.
    - **O3**: MILP one-shot via Pyomo + CBC, big-M joins outer choice with
      placement LP. Demonstrates why decomposition wins.
    - **O4**: Direct convex program in CVXPY (DCP grammar with native `max()` for
      overlap). Course-style elegance demo.
    - **O5**: Bayesian Optimization (Optuna, TPE) over outer + LP inner. Demonstrates surrogate-model search.
- **Layer 3 — Empirical validation:** Use existing `src/flexgen/qwen_inference.py`
  to run real TinyLlama inference under the LP-recommended policy and a naive
  policy on RTX 4050; measure ms/token; compare to predicted.

Spec to be written at `docs/superpowers/specs/2026-05-02-flexgen-optimizer-comparison-design.md`.
