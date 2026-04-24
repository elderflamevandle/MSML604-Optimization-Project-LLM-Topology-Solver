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
