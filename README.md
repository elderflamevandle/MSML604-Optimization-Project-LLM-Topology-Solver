# LLM-Topology-Solver 🚀

An end-to-end framework for solving the mathematical bottlenecks of Large Language Model (LLM) serving and deployment. 

Serving LLMs is notoriously resource-intensive, often constrained by GPU memory and network compute bandwidth. Rather than relying on simple heuristics, this project translates the LLM infrastructure bottleneck into rigorous mathematical equations. It utilizes **Linear Programming (LP)**, **Mixed-Integer Linear Programming (MILP)**, and **Bayesian Optimization** to calculate the absolute optimal hardware placement and system configurations for maximizing token throughput and minimizing latency.

## 🧠 Optimization Paradigms Implemented

This project implements and compares three distinct state-of-the-art optimization architectures:

### 1. FlexGen (Single-Node Memory Bottleneck)
- **Problem**: The LLM is too large to fit in a single GPU's VRAM.
- **Math**: Continuous Linear Programming (LP).
- **Solution**: Formulates fractional memory placement variables across a tiered memory system (GPU $\rightarrow$ CPU $\rightarrow$ Disk). The LP solver minimizes latency penalties to ensure critical weights/KV-cache components stay on the fastest available hardware.

### 2. Helix (Multi-GPU Cluster Routing)
- **Problem**: Partitioning a model across heterogeneous GPUs with varying network bandwidths and memory capacities to maximize token flow.
- **Math**: Mixed-Integer Linear Programming (MILP) & Network Flow Graphing.
- **Solution**: Models the cluster as a network flow graph. Defines integer boundaries for layer chunking and binary variables for valid pipeline connections. Finds the exact network routing that maximizes end-to-end flow without violating any single GPU's constraints.

### 3. Vidur (System Configuration & SLO Compliance)
- **Problem**: Finding the absolute best system configurations (quantization, batch sizes, pipeline parallelism) to maximize Queries Per Second (QPS) without violating strict tail-latency (P99) constraints.
- **Math**: Bayesian Optimization & Grid Search.
- **Solution**: Uses probabilistic search (`optuna`) to intelligently navigate the configuration search space, learning from previous states to predict the optimal setup significantly faster than exhaustive grid search.

---

## 💻 Installation

This project uses `uv` for blazing-fast dependency resolution and virtual environment management.

```bash
# 1. Create a Python 3.12 virtual environment using uv
uv venv --python 3.12

# 2. Activate the environment (Windows)
.venv\Scripts\activate
# (Mac/Linux: source .venv/bin/activate)

# 3. Install dependencies
uv pip install -r requirements.txt
```

---

## 🚀 Quick Start

The project includes an end-to-end orchestrator that runs all three mathematical optimizations, plots their trade-offs, and outputs a unified deployment recommendation.

```bash
python run_all.py
```

### What happens when you run `run_all.py`?
1. Runs the **FlexGen LP solver** to find optimal fractional placement.
2. Runs the **Helix MILP solver** to calculate maximum multi-GPU pipeline flow.
3. Runs the **Vidur configurations** via Bayesian Search to maximize QPS under latency constraints.
4. Generates **comparison graphs** in the `analysis/plots/` directory.
5. Prints a final **Deployment Recommendation** directly to your terminal.

---

## 📊 Visual Reports & Analytics

After running the pipeline, check the `analysis/plots/` folder for generated visual comparisons, including:
- `overall_throughput_comparison.png`: Main bar chart comparing Vidur's best QPS with Helix's total multi-GPU flow.
- `optimizer_comparison.png`: Shows the efficiency of Bayesian Search vs Grid Search.
- `qps_vs_latency.png`: Analyzes throughput tradeoffs across `int4`, `int8`, and `fp16` quantization.
- `memory_vs_throughput.png`: Visualizes how batch sizing impacts memory consumption.

---

## 🧪 Running Unit Tests

To validate the mathematical equations and logic constraints, you can run the comprehensive test suite using `pytest`:

```bash
pytest tests/
```

---

## 🛠️ FlexGen Faithful Policy Search (in development)

A faithful re-implementation of the FlexGen paper's policy search is being added. When complete, it will replace the current toy LP at `src/flexgen/lp_formulation.py` with an optimizer that:

- Auto-detects the host system (GPU VRAM, RAM, disk; PCIe / disk bandwidth; compute throughput) — works across servers without hardcoded values.
- Auto-introspects any HuggingFace causal-LM via its `config.json` (no weight download).
- Solves for **all 14 FlexGen decision variables**: GPU batch size, # GPU batches per block, 4-bit compression flag, CPU compute delegation flag, I/O–compute overlap flag, and the 9 placement fractions for weights / KV cache / activations across GPU / CPU / disk.

### Directories created by the toolchain

- `configs/system_calibration/` — per-machine calibration cache (gitignored). First run on a new server takes ~30 s; subsequent runs reuse the cache.
- `experiments/logs/` — per-run log files (gitignored). One file per CLI invocation.

### Optimizer structure (faithful FlexGen policy search)

The full search enumerates the 5 discrete decision variables and solves an inner LP for the 9 placement fractions at each enumerated point:

| Outer (enumerated, 480 points total) | Inner (LP, 9 fractions) |
|---|---|
| `gbs ∈ {1, 2, 4, 8, 16, 32}` | `w_g, w_c, w_d` (weights placement) |
| `num_gb ∈ {1, 2, 4, 8, 16}` | `c_g, c_c, c_d` (KV cache placement) |
| `compression ∈ {fp16, int4}` | `h_g, h_c, h_d` (activations placement) |
| `cpu_compute_delegate ∈ {False, True}` | |
| `overlap_io_compute ∈ {False, True}` | |

End-to-end search time on a typical box: ~30 seconds.

The objective is per-token latency `T_block / (gbs · num_gb)`, where `T_block` decomposes into compute, weight-load, KV I/O, and activation I/O terms per layer. With overlap=True the LP uses an epigraph variable `τ ≥ each term` (max), with overlap=False the terms sum.

### Plug any HuggingFace causal-LM

The optimizer takes a HuggingFace model id and pulls only `config.json` (~4 KB — no weight download needed for the math). Architecture fields like `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads` (GQA-aware), and `intermediate_size` are parsed and used to derive memory footprints analytically.

Tested architectures (Llama-style with SwiGLU FFN):
- `meta-llama/Meta-Llama-3-8B`
- `mistralai/Mistral-7B-v0.1`
- `Qwen/Qwen2-1.5B`

For gated repos (e.g. Llama), authenticate first: `huggingface-cli login`.

### Per-machine calibration

The first time the FlexGen optimizer runs on a new server, it micro-benchmarks the host (~30 s):

- **PCIe bandwidth** — timed pinned host→device tensor copies (or 16 GB/s fallback if no CUDA)
- **Disk bandwidth** — timed write+read of a 200 MB probe file under `configs/system_calibration/`
- **Compute throughput** — timed `torch.matmul` at fp16 (with int8/int4 scaled approximations)

Results are cached under [`configs/system_calibration/{hostname}_{gpu_model}.json`](configs/system_calibration/), keyed per machine. Subsequent runs on the same box reuse the cache and add zero startup latency. Force a recalibration after a hardware upgrade with `--recalibrate` (CLI lands in Task 12).

### Live system probe

Each run reads volatile capacities directly from the host — no hardcoded values:

| Field | Source |
|---|---|
| `gpu_vram_gb` (free) | `torch.cuda.mem_get_info()` |
| `ram_gb` (free) | `psutil.virtual_memory().available` |
| `disk_gb` (free) | `psutil.disk_usage(project_root).free` |

If CUDA is unavailable, `gpu_vram_gb` reports 0.0. Quick check that the probe works on your box:

```bash
python -c "from src.flexgen.system_probe import probe_live_capacity; \
           c = probe_live_capacity('.'); \
           print(f'GPU={c.gpu_vram_gb:.1f}GB RAM={c.ram_gb:.1f}GB DISK={c.disk_gb:.1f}GB')"
```

### Workload spec

The workload (sequence lengths) is read from a YAML file. Default at [`configs/workload.yaml`](configs/workload.yaml):

```yaml
prompt_len: 512   # tokens per request (prefill phase)
decode_len: 128   # tokens generated per request (decode phase)
```

All sequences in a block share the same prompt / decode lengths (matches the FlexGen paper's offline-batched model). Future work: trace-driven sampling from `data/sharegpt_vicuna/` or `data/vidur_traces/`.

### Additional dependencies

The standard `uv pip install -r requirements.txt` now also installs:

- `huggingface_hub` — fetches model `config.json`
- `pyyaml` — workload spec parsing
- `psutil` — live RAM / disk capacity reads

Full CLI documentation will land here as each task in [`docs/superpowers/plans/2026-04-26-flexgen-faithful.md`](docs/superpowers/plans/2026-04-26-flexgen-faithful.md) ships.
