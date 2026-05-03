# FlexGen Faithful Policy-Search — Design Spec

**Date:** 2026-04-26
**Owner:** Chaitanya
**Status:** Draft (awaiting user review)
**Brainstorming log:** [`docs/superpowers/brainstorming/2026-04-26-flexgen-faithful.md`](../brainstorming/2026-04-26-flexgen-faithful.md)

---

## 1. Goal

Replace the current toy LP at [`src/flexgen/lp_formulation.py`](../../../src/flexgen/lp_formulation.py)
with a faithful FlexGen policy search. The new optimizer takes any HuggingFace causal-LM
identifier and any GPU server, auto-detects the relevant system + model parameters, and
solves for the **14 placement-and-batching parameters** the FlexGen paper treats as
decision variables — minimizing per-token serving latency.

Out of scope for this iteration: actually executing inference with the chosen policy
(weight-streaming runtime, KV management, CPU-delegated attention kernel). That is a
separate multi-week phase listed under Future Work.

## 2. Decision variables (the 14 outputs)

| # | Variable | Symbol | Type | Search range |
|---|---|---|---|---|
| 1 | GPU batch size | `gbs` | integer | `{1, 2, 4, 8, 16, 32}` |
| 2 | # GPU batches per block | `num_gb` | integer | `{1, 2, 4, 8, 16}` |
| 3 | Block size (effective batch) | `B = gbs · num_gb` | derived | product |
| 4 | 4-bit compression on/off | `q` | binary | `{fp16, int4}` |
| 5 | CPU compute delegation | `δ` | binary | `{0, 1}` |
| 6 | I/O–compute overlap | `o` | binary | `{0, 1}` |
| 7–9 | Weights placement fractions | `w_g, w_c, w_d` | continuous | `[0,1]`, sum = 1 |
| 10–12 | KV-cache placement fractions | `c_g, c_c, c_d` | continuous | `[0,1]`, sum = 1 |
| 13–15 | Activations placement fractions | `h_g, h_c, h_d` | continuous | `[0,1]`, sum = 1 |

(Counting as 14 decisions: rows 1, 2, 4, 5, 6 + the 9 fractions in rows 7–15. `B` is
derived, not a free variable.)

## 3. Inputs

### 3.1 System (live each run + cached per machine)

**Live each run** (volatile — depends on what else is running):

| Field | Source |
|---|---|
| `gpu_vram_gb` | `torch.cuda.mem_get_info()` |
| `ram_gb` | `psutil.virtual_memory().available` |
| `disk_gb` | `psutil.disk_usage(project_root).free` |

**Cached per machine** (hardware constants — calibrated on first run, reused after):

| Field | Calibration method |
|---|---|
| `pcie_bw_gbs` | Time `torch.empty(N).pin_memory().to('cuda')` for several N |
| `disk_bw_gbs` | Time write+read of a 200 MB temp file under project root |
| `tflops_fp16` | Time `torch.matmul(A, B)` at fp16 for several shapes |
| `tflops_int8` | Same, with `bitsandbytes` int8 matmul |
| `tflops_int4` | Same, with `bitsandbytes` int4 matmul |

- **Cache key:** `f"{socket.gethostname()}_{torch.cuda.get_device_name(0).replace(' ', '_')}"`
- **Cache path:** `configs/system_calibration/{key}.json` (gitignored — each server keeps its own)
- **CLI flag:** `--recalibrate` to force re-running after hardware upgrades.
- **First-run cost:** ~30 s. Subsequent runs: cache hit, no extra latency.

### 3.2 Model (HuggingFace config-driven)

- **CLI:** `--model <hf_id>`, e.g. `meta-llama/Meta-Llama-3-8B`. Default = `meta-llama/Meta-Llama-3-8B`.
- **Fetch:** `huggingface_hub.snapshot_download(repo_id=hf_id, allow_patterns=["config.json"])`. ~4 KB.
- **Cache:** standard HF cache dir (no extra plumbing).
- **Parsed fields:** `num_hidden_layers` (L), `hidden_size` (d), `num_attention_heads` (h),
  `num_key_value_heads` (h_kv, falls back to h for non-GQA models), `intermediate_size` (d_ff),
  `vocab_size` (V), `torch_dtype` (default dtype-bytes).
- **Derived:**
  - `weights_per_layer_bytes(q)` — attention QKVO + 2-FFN + norms, scaled by quantization
    (fp16 → 2 bytes, int4 → 0.5 bytes; norms always fp16).
  - `kv_per_token_bytes(q)` — `2 · h_kv · (d / h) · L · dtype_bytes(q)`.
  - `activation_per_seq_bytes` — `seq_len · d · L · 2` (fp16 always for activations).
  - `flops_prefill_per_layer(B, s)`, `flops_decode_per_layer(B, s_total)` — standard
    transformer FLOP counts.
- **Out of scope:** weight tensors themselves are NOT downloaded. The optimizer is
  purely mathematical.

### 3.3 Workload (YAML)

- **CLI:** `--workload configs/workload.yaml` (default).
- **Default file shipped with project:**
  ```yaml
  prompt_len: 512   # tokens per request (prefill phase)
  decode_len: 128   # tokens generated per request (decode phase)
  ```
- **Sequence semantics:** all `B = gbs · num_gb` sequences in a block share the same
  `prompt_len` and `decode_len`. This matches the FlexGen paper's offline-batched model
  (uniform sequence length within a batch). Heterogeneous lengths are future work
  (the trace-driven option below).
- **Future (not this iteration):** `--workload sharegpt` to sample distributions from
  `data/sharegpt_vicuna/`. Code paths reserved but not implemented.

## 4. Cost model

For a given enumerated tuple `(gbs, num_gb, q, δ, o)` and continuous fractions
`(w_*, c_*, h_*)`, the per-token latency is:

```
T_per_token = T_block / (gbs · num_gb)

T_block     = L · (T_prefill_layer + decode_len · T_decode_layer)
```

Each per-layer term decomposes into **compute, weight-load, KV I/O, activation I/O**. The
overlap binary `o` switches the combinator:

```
T_layer(o = 1) = max(T_compute, T_load_w, T_io_kv, T_io_act)
T_layer(o = 0) =     T_compute + T_load_w + T_io_kv + T_io_act
```

### 4.1 Term definitions (prefill; decode is analogous)

Let `B = gbs · num_gb`, `s = prompt_len`. All bandwidths are bytes/s.

```
T_compute_pre  = flops_prefill_per_layer(B, s) / tflops(q)

T_load_w       = weights_per_layer_bytes(q) · (w_c / pcie_bw + w_d / disk_bw_effective)
                 -- w_g costs 0; off-GPU portions stream in each block

T_io_kv_pre    = kv_per_token_bytes(q) · B · s
               · (c_c / pcie_bw + c_d / disk_bw_effective)
                 -- store fresh KV out to off-GPU tiers
               IF δ = 1 AND c_c > 0:
                 -- CPU-delegated attention: skip KV upload to GPU; instead send Q down,
                 -- result up. Replace the c_c-weighted term with:
                 (B · s · d · 2) / pcie_bw    (Q + result, hidden-dim-sized)

T_io_act_pre   = (B · s · d · 2) · (h_c / pcie_bw + h_d / disk_bw_effective)
```

`disk_bw_effective` accounts for the disk→CPU→GPU two-hop:
`1 / disk_bw_effective = 1 / disk_bw + 1 / pcie_bw`.

Decode terms have the same structure, but FLOPs use `flops_decode_per_layer(B, s + t)`
where `t` is the decode position, and the KV term *loads existing cache* rather than
storing new entries — magnitude scales with `s + t` instead of `s`. Code in
`src/flexgen/cost_model.py` will integrate over `t ∈ [0, decode_len)`.

### 4.2 Constraints (the inner LP)

- **Conservation:** `w_g + w_c + w_d = 1`, same for `c_*` and `h_*`.
- **GPU memory cap:** `weights · w_g · L + kv_per_token · B · (s + decode_len) · c_g + activations · h_g ≤ gpu_vram_gb`.
- **CPU memory cap:** same shape, with `_c` fractions, against `ram_gb`.
- **Disk cap:** `_d` fractions against `disk_gb`.
- **Bandwidth feasibility:** none as hard constraints — bandwidths show up only in the
  objective via the load/IO terms. (Capacity is the binding constraint; bandwidth slows
  rather than forbids.)

For fixed `(gbs, num_gb, q, δ, o)`, every term in `T_block` becomes **linear** in
`(w_*, c_*, h_*)`. The objective `T_block` (or `max(...)` under overlap) is linear or
piecewise-linear, so the inner problem is a small LP solvable by CBC/PuLP in milliseconds.

The `max(...)` in the overlap case is handled by introducing a single variable
`τ ≥ T_compute, T_load_w, T_io_kv, T_io_act` and minimizing `τ` — standard LP epigraph
trick.

## 5. Optimizer structure

```
Outer enumeration (faithful to FlexGen paper's policy-search):
  for gbs in {1, 2, 4, 8, 16, 32}:
    for num_gb in {1, 2, 4, 8, 16}:
      for q in {fp16, int4}:
        for δ in {0, 1}:
          for o in {0, 1}:
            inner_lp_solve(gbs, num_gb, q, δ, o) → (placement, T_per_token, status)
            if feasible: track in candidate_list

  global_best = candidate_list.min(key = T_per_token)
  top_k = candidate_list.sorted()[:20]
```

Search-space size: `6 · 5 · 2 · 2 · 2 = 480` outer points. Each inner LP has 9 variables
+ ~6 constraints → solves in <10 ms. Total optimizer time: under 10 seconds.

**Tie-breaking:** primary key = `T_per_token`, secondary key = `w_g + c_g + h_g`
(prefer policies that keep more on GPU when latencies tie, since they're more robust to
calibration noise).

## 6. Outputs

### 6.1 Per-run results JSON

`experiments/results/flexgen_{YYYYMMDD_HHMMSS}.json`:

```json
{
  "timestamp": "2026-04-26T14:31:08Z",
  "machine_id": "server-04_NVIDIA_A100-SXM4-40GB",
  "input": {
    "system": { "gpu_vram_gb": 38.4, "ram_gb": 250.1, "disk_gb": 1450.0,
                "pcie_bw_gbs": 24.6, "disk_bw_gbs": 3.1,
                "tflops_fp16": 142.0, "tflops_int4": 480.0 },
    "model":  { "hf_id": "meta-llama/Meta-Llama-3-8B",
                "num_layers": 32, "hidden_dim": 4096, "num_heads": 32,
                "num_kv_heads": 8, "intermediate_size": 14336, "dtype_bytes": 2 },
    "workload": { "prompt_len": 512, "decode_len": 128 }
  },
  "best_policy": {
    "gpu_batch_size":         8,
    "num_gpu_batches":        4,
    "block_size":             32,
    "compression":            "int4",
    "cpu_compute_delegate":   true,
    "overlap_io_compute":     true,
    "weights":     { "gpu": 0.45, "cpu": 0.55, "disk": 0.00 },
    "kv_cache":    { "gpu": 0.20, "cpu": 0.80, "disk": 0.00 },
    "activations": { "gpu": 1.00, "cpu": 0.00, "disk": 0.00 }
  },
  "objective": {
    "per_token_latency_ms": 84.3,
    "throughput_tok_s":     11.86,
    "t_block_ms":           2697.6
  },
  "top_k_candidates": [ /* 19 next-best policies */ ]
}
```

### 6.2 Per-run log file

`experiments/logs/flexgen_{YYYYMMDD_HHMMSS}.log`:

- **Format:** `%(asctime)s | %(levelname)s | %(name)s | %(message)s` via stdlib `logging`.
- **Levels:**
  - `INFO`: progress milestones (calibration start/end, model fetched, search start/end, best policy).
  - `DEBUG`: per-config inner-LP status, objective, placement.
  - `WARNING`: cache miss, infeasible enum points.
  - `ERROR`: HF download failure, calibration crash.
- Always written to file at DEBUG level; console mirrors INFO+ unless `--verbose`.

## 7. Code structure

```
src/flexgen/
  __init__.py
  system_probe.py        # live_system_capacity()  → MemoryCapacity
  calibration.py         # ensure_calibration(machine_id) → SystemCoefficients
  model_introspect.py    # load_model_spec(hf_id) → ModelSpec
  workload.py            # load_workload(yaml_path) → WorkloadSpec
  cost_model.py          # t_block(enum, fractions, system, model, workload) → float
                         # also exposes per-term breakdown for debugging
  lp_formulation.py      # solve_inner_lp(enum, system, model, workload) → InnerResult
                         # (replaces the existing toy LP — old API kept as a thin
                         #  wrapper for the existing tests until refactored)
  policy_search.py       # run_policy_search(...) → PolicyResult (best + top_k)
configs/
  workload.yaml                          # default workload
  system_calibration/.gitkeep            # cache dir; .gitignore added
.gitignore                               # add configs/system_calibration/*.json
experiments/
  run_flexgen.py         # CLI (argparse) + logging setup + orchestration + JSON write
tests/flexgen/
  test_system_probe.py        # mock psutil/torch.cuda
  test_calibration.py         # mock the timing primitives; verify cache I/O
  test_model_introspect.py    # mock huggingface_hub; verify config parsing for
                              # Llama, Mistral, Qwen2 (covers GQA + non-GQA)
  test_workload.py            # YAML round-trip + defaults
  test_cost_model.py          # numerical tests: known closed-form cases,
                              # overlap on/off agreement at boundary, monotonicity
  test_lp_formulation.py      # extend existing tests for the inner LP
  test_policy_search.py       # property tests (bigger batch → bigger memory pressure;
                              # tighter VRAM → more spilling)
```

## 8. Test strategy

- **Unit tests** for each new module, isolating I/O via mocks.
- **Cost-model property tests:** a no-overlap solution must have `T_block(o=0) ≥ T_block(o=1)`
  (overlap can only help). int4 must use ≤ fp16 weight bytes. `B → 2B` should at least
  double the KV term.
- **Integration test:** end-to-end `run_flexgen.py --model meta-llama/Meta-Llama-3-8B` on a
  synthetic system spec (no real calibration), verifying all 14 outputs are populated and
  sane.
- **Determinism:** for fixed inputs, the optimizer output is exactly reproducible
  (no randomness in the LP path).

## 9. CLI surface

```bash
python experiments/run_flexgen.py \
    --model meta-llama/Meta-Llama-3-8B \         # any HF causal-LM id
    --workload configs/workload.yaml \           # default if omitted
    [--recalibrate]                              # force calibration refresh
    [--verbose]                                  # console gets DEBUG too
    [--output-dir experiments/results]           # default
```

`run_all.py` continues to invoke `experiments/run_flexgen.py` with default args.

## 10. 5-day implementation timeline

| Day | Deliverable | Demo-ready? |
|---|---|---|
| **Mon (today)** | Brainstorm → spec (this doc) → implementation plan via writing-plans skill. Spike: `system_probe.py` + `calibration.py` skeleton. | — |
| **Tue (day-1 demo)** | `system_probe`, `calibration`, `model_introspect`, `workload` complete. `cost_model.py` + `lp_formulation.py` working for a single enum point. CLI runs end-to-end and prints all 14 outputs for one fixed `(gbs, num_gb, q, δ, o)`. | ✅ "plug any HF model, see all 14 params for a real auto-detected system" |
| Wed | Full outer enumeration + top-k tracking. JSON output schema. Logging integration. | |
| Thu | Test suite (unit + integration). Plot extensions in `analysis/plot_tradeoffs.py` (Pareto: per-token latency vs effective batch; heatmap of placement fractions across enum points). | |
| Fri | End-to-end validation on Llama-3-8B, Mistral-7B-v0.1, Qwen2-1.5B. Polish, docstrings, demo prep. | ✅ full pipeline |

## 11. Future work (not this iteration)

- **C — Full FlexGen runtime.** Actually execute the policy: download weights, implement
  weight streaming, KV management, CPU-delegated attention kernel. Validate the
  optimizer's predicted latency against measured. Multi-week.
- **C — Trace-driven workload.** `--workload sharegpt` reads real prompt/decode
  distributions from `data/sharegpt_vicuna/` instead of fixed YAML. Optimizer changes
  from a single objective to expected-value-over-distribution.
- **Multi-GPU extension.** Pipe the FlexGen single-node policy as the per-stage
  configuration for a multi-GPU pipeline plan.

## 12. Risks / open questions

- **CPU delegation latency model.** The "send Q, receive result" approximation in §4.1
  ignores the FFN cost of running attention on CPU. We assume CPU TFLOPS for the
  delegated portion is small relative to GPU compute and treat the term as I/O-bound. If
  results look wrong, we'll add a `T_compute_cpu_attn` term in a follow-up.
- **Calibration reliability.** Disk benchmarks are noisy (page cache, concurrent I/O).
  We'll average over 3 runs and warn if variance exceeds 20%.
- **bitsandbytes dependency.** int8/int4 matmul timing requires `bitsandbytes`. If not
  available on a server, calibration falls back to estimating from fp16 with a fixed
  scaling factor (`int8 = 2× fp16 throughput`, `int4 = 4× fp16`) and logs a WARNING.
