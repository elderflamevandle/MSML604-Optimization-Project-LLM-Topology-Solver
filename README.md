# FlexGen Optimizer

A faithful implementation of the **FlexGen** policy search for LLM inference on memory-constrained hardware. The optimizer formulates fractional memory placement across a tiered memory hierarchy (GPU → CPU → Disk) using Linear Programming, then enumerates discrete decision variables to find the policy that minimizes per-token latency.

---

## How it works

FlexGen decomposes each transformer block's latency into four components: compute, weight-load I/O, KV-cache I/O, and activation I/O. The optimizer runs a two-level search:

- **Outer loop** (480 enumerated points): `gpu_batch_size`, `num_gpu_batches`, `compression`, `cpu_compute_delegate`, `overlap_io_compute`
- **Inner LP** (9 continuous fractions): `w_g/w_c/w_d`, `c_g/c_c/c_d`, `h_g/h_c/h_d` — placement of weights, KV cache, and activations across GPU/CPU/disk

With `overlap_io_compute=True`, the LP minimizes an epigraph variable `τ ≥ max(compute, I/O)` per layer. Without overlap, terms sum. The objective is `T_block / (gbs · num_gb)` — per-token latency.

---

## Installation

```bash
# Python 3.12 virtual environment via uv
uv venv --python 3.12
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

uv pip install -r requirements.txt
```

---

## Quick start

```bash
# Default: Llama-3-8B, auto-detected system
python experiments/run_flexgen.py

# Any HuggingFace causal-LM
python experiments/run_flexgen.py --model mistralai/Mistral-7B-v0.1

# Force hardware recalibration
python experiments/run_flexgen.py --recalibrate

# DEBUG-level console output
python experiments/run_flexgen.py --verbose
```

Run experiments + plot in one step:

```bash
python run_all.py
```

---

## pipeline.py — one-command full pipeline

`pipeline.py` is the recommended entry point for GPU runs. It chains four steps in one command:

1. Runs the full FlexGen test suite (`pytest tests/flexgen`)
2. Runs the policy search (`experiments/run_flexgen.py`)
3. Builds a baseline-vs-optimized comparison and prints it to the terminal
4. Optionally loads the model and runs actual inference

All settings come from a YAML config file. Edit the config — not `pipeline.py` — when paths or defaults change.

### Minimal run (reads `config_flexgen.yml`)

```bash
python pipeline.py
```

### Local GPU with a small downloaded model

Uses `config_flexgen_local_gpu.yml` which points to `models/smollm2-135m-instruct/` and enables inference on CUDA:

```bash
python pipeline.py --config config_flexgen_local_gpu.yml
```

### HPC / remote GPU server (Qwen model)

```bash
python pipeline.py --config config_flexgen.yml
```

The default `config_flexgen.yml` sets `paths.model` to the server Qwen path. Override it temporarily:

```bash
python pipeline.py --model /path/to/your/Qwen/
```

### Run with GPU inference

After the policy search, load the model and generate text on CUDA:

```bash
python pipeline.py \
  --run-inference \
  --device cuda \
  --prompt "Explain GPU memory offloading for LLM inference in two sentences." \
  --max-new-tokens 80
```

Multi-GPU (Transformers `device_map="auto"`):

```bash
python pipeline.py \
  --run-inference \
  --device cuda \
  --device-map auto \
  --prompt "Explain FlexGen in simple terms." \
  --max-new-tokens 80
```

### Run all tests + full report

```bash
python pipeline.py --detailed-report
```

Prints all 14 policy parameters, full model/system inputs, and the top-20 candidate table.

### Verbose optimizer logs

```bash
python pipeline.py --verbose
```

### Force hardware recalibration

```bash
python pipeline.py --recalibrate --verbose
```

### Skip tests (re-use previous test result)

```bash
python pipeline.py --skip-tests
```

### Run only the test suite standalone

```bash
pytest tests/flexgen/ -v
```

### pipeline.py CLI flags

| Flag | Default | Description |
|---|---|---|
| `--config <yml>` | `config_flexgen.yml` | YAML config file |
| `--model <path\|hf_id>` | from config | Local folder or HuggingFace id |
| `--workload <yaml>` | from config | Workload spec (prompt/decode lengths) |
| `--skip-tests` | off | Skip pytest and go straight to policy search |
| `--test-verbose` | off | Show per-test output instead of quiet summary |
| `--recalibrate` | off | Force re-benchmark hardware |
| `--verbose` | off | DEBUG-level optimizer output |
| `--detailed-report` | off | Print 14 parameters + top-k candidates |
| `--run-inference` | off | Load model and generate text after search |
| `--prompt <str>` | from config | Prompt for inference |
| `--max-new-tokens <int>` | from config | Tokens to generate |
| `--device <str>` | from config | `cuda`, `cpu`, or `auto` |
| `--device-map <str>` | from config | Transformers `device_map` (e.g. `auto`) |
| `--dtype <str>` | from config | `float16`, `bfloat16`, `float32`, or `auto` |
| `--output-dir <dir>` | from config | Result JSON directory |
| `--baseline-dir <dir>` | from config | Baseline comparison JSON directory |
| `--log-dir <dir>` | from config | Log file directory |
| `--cache-dir <dir>` | from config | Calibration cache directory |

### Output files written by pipeline.py

```
experiments/results/flexgen_<ts>.json               — best policy + top-20 candidates
experiments/results/flexgen_pipeline_<ts>.json      — combined pipeline summary
experiments/baseline_comparisons/baseline_<ts>.json — baseline vs optimised comparison
experiments/logs/flexgen_<ts>.log                   — full DEBUG log
experiments/results/qwen_inference_<ts>.json        — inference result (if --run-inference)
```

---

## CLI flags (run_flexgen.py standalone)

| Flag | Default | Description |
|---|---|---|
| `--model <hf_id>` | `meta-llama/Meta-Llama-3-8B` | Any HuggingFace causal-LM |
| `--workload <yaml>` | `configs/workload.yaml` | Workload YAML (prompt/decode lengths) |
| `--recalibrate` | off | Re-run hardware benchmarks even if cached |
| `--verbose` | off | DEBUG-level output on console |
| `--output-dir <dir>` | `experiments/results` | JSON output directory |
| `--log-dir <dir>` | `experiments/logs` | Log file directory |
| `--cache-dir <dir>` | `configs/system_calibration` | Calibration cache directory |

---

## Output

Each run writes two files (timestamped UTC):

- `experiments/results/flexgen_<ts>.json` — best policy (14 decision variables), full system/model/workload context, top-20 candidates for sensitivity analysis
- `experiments/logs/flexgen_<ts>.log` — DEBUG-level structured log of the full search

Sample best-policy block:

```json
{
  "gpu_batch_size":         8,
  "num_gpu_batches":        4,
  "block_size":             32,
  "compression":            "int4",
  "cpu_compute_delegate":   true,
  "overlap_io_compute":     true,
  "weights":     { "gpu": 0.45, "cpu": 0.55, "disk": 0.0 },
  "kv_cache":    { "gpu": 0.20, "cpu": 0.80, "disk": 0.0 },
  "activations": { "gpu": 1.00, "cpu": 0.00, "disk": 0.0 }
}
```

---

## System probe & calibration

On first run, the optimizer auto-benchmarks the host (~30 s) and caches results under `configs/system_calibration/{hostname}_{gpu}.json`. Subsequent runs reuse the cache (zero overhead). Use `--recalibrate` after a hardware change.

Live capacity is read every run (not hardcoded):

| Field | Source |
|---|---|
| `gpu_vram_gb` (free) | `torch.cuda.mem_get_info()` |
| `ram_gb` (free) | `psutil.virtual_memory().available` |
| `disk_gb` (free) | `psutil.disk_usage(project_root).free` |

Quick check:

```bash
python -c "from src.flexgen.system_probe import probe_live_capacity; \
           c = probe_live_capacity('.'); \
           print(f'GPU={c.gpu_vram_gb:.1f}GB RAM={c.ram_gb:.1f}GB DISK={c.disk_gb:.1f}GB')"
```

---

## Workload config

Default at `configs/workload.yaml`:

```yaml
prompt_len: 512   # tokens per request (prefill)
decode_len: 128   # tokens generated per request (decode)
```

---

## Supported architectures

Tested Llama-style models with SwiGLU FFN (no weight download needed — only `config.json` is fetched):

- `meta-llama/Meta-Llama-3-8B`
- `mistralai/Mistral-7B-v0.1`
- `Qwen/Qwen2-1.5B`

For gated repos: `huggingface-cli login` first.

Local small models (pre-downloaded, no auth required):

- `models/smollm2-135m-instruct/`
- `models/tinyllama-1.1b-chat/`

---

## Running tests

```bash
pytest tests/flexgen/ -v
```

Covers: system probe, calibration cache, HF model introspection, cost-model properties (overlap dominance, batch amortization, GQA-aware delegate cost), inner LP optimality, end-to-end orchestration, plot generation.

---

## Source layout

```
src/flexgen/
  system_probe.py      — live GPU/RAM/disk capacity + hardware benchmarks
  calibration.py       — calibration cache load/save
  model_introspect.py  — HuggingFace config.json parser (any causal-LM)
  cost_model.py        — per-layer latency cost model (paper-faithful)
  lp_formulation.py    — inner LP (9 placement fractions, scipy linprog)
  policy_search.py     — outer search loop (480 points × inner LP)
  workload.py          — workload YAML loader
  config_file.py       — YAML config loader/override helpers
  baseline_compare.py  — baseline policy comparison utilities
  qwen_inference.py    — Qwen model inference runner
  __init__.py

experiments/
  run_flexgen.py       — CLI entry point

analysis/
  plot_tradeoffs.py    — throughput/latency trade-off plots

configs/
  workload.yaml        — default workload spec
  system_calibration/  — per-machine calibration cache (gitignored)

tests/flexgen/         — full test suite (pytest)
guides/                — step-by-step runbooks
report/                — analysis report
```

---

## Guides

- [USER-GUIDE-FLEXGEN.md](USER-GUIDE-FLEXGEN.md) — full walkthrough: clone → CUDA → HF auth → run → inspect results → write tests
- [guides/LOCAL_GPU_SMALL_MODEL_TEST.md](guides/LOCAL_GPU_SMALL_MODEL_TEST.md) — local GPU test with SmoLLM2-135M
- [guides/LOCAL_GPU_TINYLLAMA_1B_TEST.md](guides/LOCAL_GPU_TINYLLAMA_1B_TEST.md) — local GPU test with TinyLLaMA-1.1B
- [guides/LOCAL_SYNTHETIC_FLEXGEN_TEST.md](guides/LOCAL_SYNTHETIC_FLEXGEN_TEST.md) — synthetic (no-model) local test
- [guides/PIPELINE_ONE_COMMAND.md](guides/PIPELINE_ONE_COMMAND.md) — single-command pipeline reference
- [guides/FLEXGEN_PLUS_CONVEX_OPTIMIZATION_COMPARISON.md](guides/FLEXGEN_PLUS_CONVEX_OPTIMIZATION_COMPARISON.md) — LP formulation deep-dive
