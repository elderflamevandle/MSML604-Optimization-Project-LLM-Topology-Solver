# FlexGen LP vs Grid Search — Cross-Model Results

> Auto-generated on 2026-05-03 20:09 UTC  
> Source: `experiments/results/model_tracking.json`  
> Regenerate: `python analysis/generate_report.py`

---

## Overview

This report compares two strategies for finding the optimal FlexGen 14-parameter policy:

| Strategy | Inner search | Time |
|---|---|---|
| **LP (FlexGen)** | Exact LP solve (scipy/PuLP) per outer enum point | ~10 s (240 LP solves) |
| **Grid search** | Enumerate 3,375 discrete placements (step=0.25) per outer point | ~0.06 s (numpy, vectorised) |

Both enumerate the same 240 outer points  
(`gpu_batch_size` × `num_gpu_batches` × `compression` × `cpu_compute_delegate` × `overlap_io_compute`).
The 9 placement fractions (weights/KV-cache/activations across GPU/CPU/disk) are what differ.

---

## Summary Table

| Model | Params | GPU constraint | LP latency | Grid latency | LP advantage | Grid search time |
|---|---|---|---|---|---|---|
| SmoLLM2-135M | 0.135B | real GPU | 1.45 ms | 1.45 ms | Tie | 0.065 s |
| TinyLLaMA-1.1B | 1.1B | 0.3 GB (sim) | 33.04 ms | 42.50 ms | **LP wins by 28.6%** | 0.059 s |
| Qwen2-7B | 6.5B | 2.0 GB (sim) | 120.63 ms | 158.19 ms | **LP wins by 31.1%** | 0.054 s |
| Mistral-7B-v0.1 | 7.0B | 2.0 GB (sim) | 166.56 ms | 190.69 ms | **LP wins by 14.5%** | 0.073 s |

> **LP advantage** = how much lower LP latency is vs grid (positive = LP wins).  
> Models with no memory pressure (SmoLLM2) produce identical results from both methods.

---

## Key Findings

1. **LP consistently outperforms grid search when models require memory offloading.**  
   LP found 14–31% lower per-token latency on TinyLLaMA, Qwen2-7B, and Mistral-7B by
   computing exact fractional placements (e.g. `w_g=0.658`) instead of 0.25-step
   approximations (e.g. `w_g=0.50`).

2. **Grid search is ~150× faster** (0.05–0.07 s vs ~10 s) because it only does arithmetic
   — no LP solver overhead.

3. **When everything fits in GPU, both methods tie.** On SmoLLM2-135M the optimal policy
   is all-GPU placement, which is a corner point both methods find exactly.

4. **The LP formulation had two latent bugs in the `delegate=True` cost term** that were
   discovered during this comparison (see `src/flexgen/lp_formulation.py` commit history):
   - `q_xfer` was scaled by `c_c` instead of treated as a constant
   - Decode `q_xfer` used `kv_avg` tokens instead of 1
   After fixing both, LP correctly beats grid on all memory-constrained models.

---

## Per-Model Results

### SmoLLM2-135M

- **Model ID:** `models/smollm2-135m-instruct`
- **Architecture:** 30 layers, hidden=576
- **Parameters:** ~0.135B
- **GPU constraint:** real GPU (model fits without offloading)
- **Last run:** 2026-05-03

#### Performance

| Metric | LP (inner LP) | Grid (step=0.25) |
|---|---|---|
| Best latency (ms/token) | **1.4532** | 1.4532 |
| Throughput (tok/s) | **688.1233** | 688.1233 |
| Search time (s) | ~10 | **0.065** |
| LP advantage | Tie | |

#### Best Policy (all 14 parameters)

| Parameter | LP | Grid | Differs? |
|---|---|---|---|
| `gpu_batch_size` | 1 | 1 |  |
| `num_gpu_batches` | 2 | 1 | YES |
| `block_size` | 2 | 1 | YES |
| `compression` | int4 | int4 |  |
| `cpu_compute_delegate` | False | False |  |
| `overlap_io_compute` | True | False | YES |
| `w_g` | 1.0000 | 1.0000 |  |
| `w_c` | 0.0000 | 0.0000 |  |
| `w_d` | 0.0000 | 0.0000 |  |
| `c_g` | 1.0000 | 1.0000 |  |
| `c_c` | 0.0000 | 0.0000 |  |
| `c_d` | 0.0000 | 0.0000 |  |
| `h_g` | 1.0000 | 1.0000 |  |
| `h_c` | 0.0000 | 0.0000 |  |
| `h_d` | 0.0000 | 0.0000 |  |

#### Plots

| Chart | File |
|---|---|
| Latency & throughput | [smollm2-135m-instruct/lp_vs_grid_latency.png](..\analysis\plots\smollm2-135m-instruct/lp_vs_grid_latency.png) |
| Placement fractions  | [smollm2-135m-instruct/lp_vs_grid_placement.png](..\analysis\plots\smollm2-135m-instruct/lp_vs_grid_placement.png) |
| Top-k candidates     | [smollm2-135m-instruct/lp_vs_grid_topk.png](..\analysis\plots\smollm2-135m-instruct/lp_vs_grid_topk.png) |

---

### TinyLLaMA-1.1B

- **Model ID:** `models/tinyllama-1.1b-chat`
- **Architecture:** 22 layers, hidden=2048
- **Parameters:** ~1.1B
- **GPU constraint:** 0.3 GB GPU (simulated)
- **Last run:** 2026-05-03

#### Performance

| Metric | LP (inner LP) | Grid (step=0.25) |
|---|---|---|
| Best latency (ms/token) | **33.0359** | 42.4978 |
| Throughput (tok/s) | **30.2701** | 23.5306 |
| Search time (s) | ~10 | **0.059** |
| LP advantage | **LP wins by 28.6%** | |

#### Best Policy (all 14 parameters)

| Parameter | LP | Grid | Differs? |
|---|---|---|---|
| `gpu_batch_size` | 16 | 32 | YES |
| `num_gpu_batches` | 16 | 16 |  |
| `block_size` | 256 | 512 | YES |
| `compression` | int4 | int4 |  |
| `cpu_compute_delegate` | True | True |  |
| `overlap_io_compute` | True | True |  |
| `w_g` | 0.6649 | 0.5000 | YES |
| `w_c` | 0.3351 | 0.5000 | YES |
| `w_d` | 0.0000 | 0.0000 |  |
| `c_g` | 0.0000 | 0.0000 |  |
| `c_c` | 0.9686 | 1.0000 | YES |
| `c_d` | 0.0314 | 0.0000 | YES |
| `h_g` | 0.0000 | 0.0000 |  |
| `h_c` | 0.6379 | 0.2500 | YES |
| `h_d` | 0.3621 | 0.7500 | YES |

#### Plots

| Chart | File |
|---|---|
| Latency & throughput | [tinyllama-1.1b-chat/lp_vs_grid_latency.png](..\analysis\plots\tinyllama-1.1b-chat/lp_vs_grid_latency.png) |
| Placement fractions  | [tinyllama-1.1b-chat/lp_vs_grid_placement.png](..\analysis\plots\tinyllama-1.1b-chat/lp_vs_grid_placement.png) |
| Top-k candidates     | [tinyllama-1.1b-chat/lp_vs_grid_topk.png](..\analysis\plots\tinyllama-1.1b-chat/lp_vs_grid_topk.png) |

---

### Qwen2-7B

- **Model ID:** `Qwen/Qwen2-7B`
- **Architecture:** 28 layers, hidden=3584
- **Parameters:** ~6.5B
- **GPU constraint:** 2.0 GB GPU (simulated)
- **Last run:** 2026-05-03

#### Performance

| Metric | LP (inner LP) | Grid (step=0.25) |
|---|---|---|
| Best latency (ms/token) | **120.6318** | 158.1947 |
| Throughput (tok/s) | **8.2897** | 6.3213 |
| Search time (s) | ~10 | **0.054** |
| LP advantage | **LP wins by 31.1%** | |

#### Best Policy (all 14 parameters)

| Parameter | LP | Grid | Differs? |
|---|---|---|---|
| `gpu_batch_size` | 32 | 32 |  |
| `num_gpu_batches` | 16 | 16 |  |
| `block_size` | 512 | 512 |  |
| `compression` | int4 | int4 |  |
| `cpu_compute_delegate` | True | True |  |
| `overlap_io_compute` | True | True |  |
| `w_g` | 0.6582 | 0.5000 | YES |
| `w_c` | 0.3418 | 0.5000 | YES |
| `w_d` | 0.0000 | 0.0000 |  |
| `c_g` | 0.0000 | 0.0000 |  |
| `c_c` | 0.9552 | 1.0000 | YES |
| `c_d` | 0.0448 | 0.0000 | YES |
| `h_g` | 0.0000 | 0.0000 |  |
| `h_c` | 0.2200 | 0.0000 | YES |
| `h_d` | 0.7800 | 1.0000 | YES |

#### Plots

| Chart | File |
|---|---|
| Latency & throughput | [qwen2-7b/lp_vs_grid_latency.png](..\analysis\plots\qwen2-7b/lp_vs_grid_latency.png) |
| Placement fractions  | [qwen2-7b/lp_vs_grid_placement.png](..\analysis\plots\qwen2-7b/lp_vs_grid_placement.png) |
| Top-k candidates     | [qwen2-7b/lp_vs_grid_topk.png](..\analysis\plots\qwen2-7b/lp_vs_grid_topk.png) |

---

### Mistral-7B-v0.1

- **Model ID:** `mistralai/Mistral-7B-v0.1`
- **Architecture:** 32 layers, hidden=4096
- **Parameters:** ~7.0B
- **GPU constraint:** 2.0 GB GPU (simulated)
- **Last run:** 2026-05-03

#### Performance

| Metric | LP (inner LP) | Grid (step=0.25) |
|---|---|---|
| Best latency (ms/token) | **166.5605** | 190.6935 |
| Throughput (tok/s) | **6.0038** | 5.2440 |
| Search time (s) | ~10 | **0.073** |
| LP advantage | **LP wins by 14.5%** | |

#### Best Policy (all 14 parameters)

| Parameter | LP | Grid | Differs? |
|---|---|---|---|
| `gpu_batch_size` | 32 | 32 |  |
| `num_gpu_batches` | 16 | 16 |  |
| `block_size` | 512 | 512 |  |
| `compression` | int4 | int4 |  |
| `cpu_compute_delegate` | True | True |  |
| `overlap_io_compute` | True | True |  |
| `w_g` | 0.6154 | 0.5000 | YES |
| `w_c` | 0.3846 | 0.5000 | YES |
| `w_d` | 0.0000 | 0.0000 |  |
| `c_g` | 0.0000 | 0.0000 |  |
| `c_c` | 0.9766 | 1.0000 | YES |
| `c_d` | 0.0234 | 0.0000 | YES |
| `h_g` | 0.0000 | 0.0000 |  |
| `h_c` | 0.0779 | 0.0000 | YES |
| `h_d` | 0.9221 | 1.0000 | YES |

#### Plots

| Chart | File |
|---|---|
| Latency & throughput | [mistral-7b-v0-1/lp_vs_grid_latency.png](..\analysis\plots\mistral-7b-v0-1/lp_vs_grid_latency.png) |
| Placement fractions  | [mistral-7b-v0-1/lp_vs_grid_placement.png](..\analysis\plots\mistral-7b-v0-1/lp_vs_grid_placement.png) |
| Top-k candidates     | [mistral-7b-v0-1/lp_vs_grid_topk.png](..\analysis\plots\mistral-7b-v0-1/lp_vs_grid_topk.png) |

---

## Cross-Model Plots

| Chart | File |
|---|---|
| Latency comparison (all models) | [model_summary_latency.png](../analysis/plots/model_summary_latency.png) |
| Throughput comparison           | [model_summary_throughput.png](../analysis/plots/model_summary_throughput.png) |
| Grid search time                | [model_summary_search_time.png](../analysis/plots/model_summary_search_time.png) |

---

## How to Reproduce

```bash
# Run all models and regenerate everything
python run_experiments.py

# Run a single model
python experiments/run_all_models.py --only qwen2-7b

# Regenerate cross-model plots only
python analysis/plot_model_summary.py

# Regenerate this report only
python analysis/generate_report.py
```

## File Structure

```
experiments/results/
  model_tracking.json              <- cross-model index (auto-updated)
  smollm2-135m-instruct/           <- per-model results
    flexgen_<ts>.json
    grid_baseline_<ts>.json
    comparison_<ts>.json
  tinyllama-1.1b-chat/ ...
  qwen2-7b/ ...
  mistral-7b-v0-1/ ...

analysis/plots/
  model_summary_latency.png        <- cross-model charts
  model_summary_throughput.png
  model_summary_search_time.png
  smollm2-135m-instruct/           <- per-model plots
    lp_vs_grid_latency.png
    lp_vs_grid_placement.png
    lp_vs_grid_topk.png
  tinyllama-1.1b-chat/ ...
  qwen2-7b/ ...
  mistral-7b-v0-1/ ...
```
