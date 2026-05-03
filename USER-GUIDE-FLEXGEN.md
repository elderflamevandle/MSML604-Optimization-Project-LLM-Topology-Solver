# FlexGen Faithful Policy Search — End-to-End User Guide

This guide walks you from **zero to running results**: clone the repo, set up Python, install PyTorch with CUDA, install dependencies, optionally download a HuggingFace model, run the optimizer, and inspect the output.

The example commands target **Linux**. For Windows / macOS, swap shell-specific lines (the Python/git/uv/pip commands are identical).

---

## Table of contents

1. [What you'll have at the end](#1-what-youll-have-at-the-end)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Get the code](#3-step-1--get-the-code)
4. [Step 2 — Create a Python 3.12 environment](#4-step-2--create-a-python-312-environment)
5. [Step 3 — Install PyTorch with CUDA support](#5-step-3--install-pytorch-with-cuda-support)
6. [Step 4 — Install project dependencies](#6-step-4--install-project-dependencies)
7. [Step 5 — Verify the install](#7-step-5--verify-the-install)
8. [Step 6 — HuggingFace authentication (gated models)](#8-step-6--huggingface-authentication-gated-models)
9. [Step 7 — (Optional) Download model weights](#9-step-7--optional-download-model-weights)
10. [Step 8 — Run the FlexGen optimizer](#10-step-8--run-the-flexgen-optimizer)
11. [Step 9 — Inspect the results](#11-step-9--inspect-the-results)
12. [Step 10 — Run the test suite](#12-step-10--run-the-test-suite)
13. [Mocking HuggingFace in your own tests](#13-mocking-huggingface-in-your-own-tests)
14. [Troubleshooting](#14-troubleshooting)
15. [Reference — what each script does](#15-reference--what-each-script-does)

---

## 1. What you'll have at the end

After this guide:

- The repo cloned and tests passing on your machine.
- PyTorch built against your GPU's CUDA version, so calibration uses the GPU.
- The FlexGen optimizer runs end-to-end on any HuggingFace causal-LM and emits all **14 FlexGen decision variables** (GPU batch size, # GPU batches, block size, compression flag, CPU-delegate flag, I/O–compute overlap flag, and 9 placement fractions across `{weights, KV cache, activations} × {GPU, CPU, disk}`) into a JSON file plus a structured log.
- Calibration cached per-machine so subsequent runs add zero startup cost.

---

## 2. Prerequisites

| Requirement | Why |
|---|---|
| **Linux x86_64** (Ubuntu 22.04+ tested) | NVIDIA driver + PyTorch CUDA wheels are best-supported here |
| **NVIDIA GPU** with recent driver | Calibration measures real PCIe + compute throughput |
| **CUDA Toolkit version known** | You'll match PyTorch's CUDA wheel to it |
| **Python 3.12** | What the project pins |
| **Git** | To clone the repo |
| **`uv`** (recommended) or `pip` | Fast dependency resolver |
| **HuggingFace account** (only for gated models) | Llama / Gemma / etc. require ToS acceptance |

Check your CUDA version:

```bash
nvidia-smi
# Look at "CUDA Version" in the top-right header (e.g. 12.1, 12.4)
```

If `nvidia-smi` is missing, install the NVIDIA driver first (e.g. `sudo apt install nvidia-driver-535` on Ubuntu).

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# then: source ~/.cargo/env  (or restart your shell)
```

---

## 3. Step 1 — Get the code

```bash
git clone <repo-url> Optimization_Project
cd Optimization_Project
git checkout chaitanya     # or whichever branch holds the FlexGen-faithful work
git pull
```

If you've already cloned, just:

```bash
cd Optimization_Project
git pull
```

---

## 4. Step 2 — Create a Python 3.12 environment

```bash
uv venv --python 3.12
source .venv/bin/activate
```

The `(. venv)` prefix should appear in your prompt. Confirm Python:

```bash
python --version
# Python 3.12.x
```

---

## 5. Step 3 — Install PyTorch with CUDA support

This is the step that **actually wires up your GPU**. Match the CUDA index URL to the CUDA version `nvidia-smi` reported.

| Your CUDA version | Install command |
|---|---|
| **12.1** | `uv pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| **12.4** | `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| **11.8** | `uv pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| Other / unsure | Use the official selector at https://pytorch.org/get-started/locally/ |

Verify CUDA is now visible:

```bash
python -c "import torch; print(f'cuda_available={torch.cuda.is_available()} | device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected:

```
cuda_available=True | device=NVIDIA GeForce RTX 4050 Laptop GPU   (or whatever GPU you have)
```

If `cuda_available=False`, the wheel didn't match your driver — re-run the install with a different CUDA tag, or check `nvidia-smi` again.

---

## 6. Step 4 — Install project dependencies

```bash
uv pip install -r requirements.txt
```

This pulls everything else: `pulp` (LP solver), `huggingface_hub`, `pyyaml`, `psutil`, `optuna`, `transformers`, `datasets`, `matplotlib`, `pandas`, `seaborn`, `scipy`, `numpy`, `pytest`.

---

## 7. Step 5 — Verify the install

```bash
# Live system probe — confirms torch + psutil work
python -c "from src.flexgen.system_probe import probe_live_capacity; \
           c = probe_live_capacity('.'); \
           print(f'GPU={c.gpu_vram_gb:.1f}GB RAM={c.ram_gb:.1f}GB DISK={c.disk_gb:.1f}GB')"

# Run the FlexGen test suite (offline, no HF or GPU needed)
pytest tests/flexgen/ -q
```

Expected: 47+ tests pass in ~40 s.

---

## 8. Step 6 — HuggingFace authentication (gated models)

**Skip this section** if you only want to use public models like `mistralai/Mistral-7B-v0.1` or `Qwen/Qwen2-1.5B`.

For gated models (Llama, Gemma, etc.), you need a HF token:

1. Visit the model page (e.g. https://huggingface.co/meta-llama/Meta-Llama-3-8B) and accept the license.
2. Create an access token at https://huggingface.co/settings/tokens (read-only is fine).
3. Either log in interactively:

   ```bash
   huggingface-cli login
   # paste the token when prompted
   ```

   Or set an environment variable (preferred for CI/servers):

   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

The optimizer and download script both pick the token up automatically.

---

## 9. Step 7 — (Optional) Download model weights

The FlexGen **optimizer itself does not need weights** — it pulls only `config.json` (~4 KB) for each model. Download full weights only if you plan to actually run inference using the chosen policy (future Phase-C work).

```bash
# Public model
python scripts/download_model.py --model mistralai/Mistral-7B-v0.1

# Gated model (after Step 6)
python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B

# Custom destination (default is HF cache, ~/.cache/huggingface/)
python scripts/download_model.py --model mistralai/Mistral-7B-v0.1 \
    --local-dir ~/models/mistral-7b

# Just config.json + tokenizer (skip the multi-GB safetensors)
python scripts/download_model.py --model mistralai/Mistral-7B-v0.1 --config-only
```

Sizes for reference:

| Model | Full weights | Config-only |
|---|---|---|
| `Qwen/Qwen2-1.5B` | ~3 GB | ~5 KB |
| `mistralai/Mistral-7B-v0.1` | ~14 GB | ~5 KB |
| `meta-llama/Meta-Llama-3-8B` | ~16 GB | ~5 KB |

---

## 10. Step 8 — Run the FlexGen optimizer

```bash
# Defaults: meta-llama/Meta-Llama-3-8B + configs/workload.yaml
python experiments/run_flexgen.py

# Pick any HuggingFace causal-LM
python experiments/run_flexgen.py --model mistralai/Mistral-7B-v0.1

# DEBUG-level console output
python experiments/run_flexgen.py --model Qwen/Qwen2-1.5B --verbose

# Force a calibration refresh (after a hardware upgrade, etc.)
python experiments/run_flexgen.py --recalibrate
```

### What happens on the first run on this machine

1. **Live system probe** (~50 ms) — reads free GPU VRAM, RAM, and disk via `torch.cuda` + `psutil`.
2. **Calibration** (~30 s, **first run only**) — micro-benchmarks PCIe bandwidth, disk bandwidth, and fp16 / int8 / int4 compute throughput. Result is cached at `configs/system_calibration/{hostname}_{gpu_name}.json`.
3. **HuggingFace fetch** (~1 s) — pulls just `config.json` for the requested model. Results cached in `~/.cache/huggingface/`.
4. **Workload load** (~10 ms) — reads `configs/workload.yaml`.
5. **Policy search** (~30 s) — solves 480 inner LPs (one per enumerated outer point), tracks the top 20 by per-token latency.
6. **Output write** — JSON to `experiments/results/`, log to `experiments/logs/`.

Subsequent runs on the same machine skip step 2 entirely (cache hit), so total wall time drops to ~30 s.

### Customizing the workload

Edit [`configs/workload.yaml`](configs/workload.yaml):

```yaml
prompt_len: 1024     # tokens per request (prefill)
decode_len: 256      # tokens generated per request (decode)
```

Or use `--workload <path>` to point at a different file.

---

## 11. Step 9 — Inspect the results

Each invocation writes two files (timestamped, UTC):

```
experiments/results/flexgen_20260426_143108.json
experiments/logs/flexgen_20260426_143108.log
```

### The JSON output

```json
{
  "timestamp": "2026-04-26T14:31:08Z",
  "machine_id": "server-04_NVIDIA_GeForce_RTX_4050_Laptop_GPU",
  "input": {
    "system": { "gpu_vram_gb": 6.0, "ram_gb": 32.0, "disk_gb": 388.9,
                "pcie_bw_gbs": 14.2, "disk_bw_gbs": 2.8,
                "tflops_fp16": 9.1, "tflops_int4": 36.4 },
    "model":  { "hf_id": "meta-llama/Meta-Llama-3-8B",
                "num_layers": 32, "hidden_dim": 4096, "num_heads": 32,
                "num_kv_heads": 8, "intermediate_size": 14336 },
    "workload": { "prompt_len": 512, "decode_len": 128 }
  },
  "best_policy": {
    "gpu_batch_size":         8,
    "num_gpu_batches":        4,
    "block_size":             32,
    "compression":            "int4",
    "cpu_compute_delegate":   true,
    "overlap_io_compute":     true,
    "weights":     { "gpu": 0.45, "cpu": 0.55, "disk": 0.0 },
    "kv_cache":    { "gpu": 0.20, "cpu": 0.80, "disk": 0.0 },
    "activations": { "gpu": 1.00, "cpu": 0.00, "disk": 0.0 }
  },
  "objective": {
    "per_token_latency_ms": 84.3,
    "throughput_tok_s":     11.86,
    "t_block_ms":           2697.6
  },
  "top_k_candidates": [ /* 19 next-best policies */ ]
}
```

All **14 FlexGen decision variables** appear under `best_policy`. The `top_k_candidates` list is sorted ascending by per-token latency — useful for sensitivity analysis and trade-off plots.

### The log file

```
2026-04-26 14:30:38 | INFO | run_flexgen | === FlexGen policy-search run ===
2026-04-26 14:30:38 | INFO | run_flexgen | machine_id=server-04_NVIDIA_RTX_4050_Laptop
2026-04-26 14:30:38 | INFO | run_flexgen | system: gpu_vram=6.00GB ram=32.00GB disk=388.90GB
2026-04-26 14:30:38 | INFO | src.flexgen.calibration | Loaded calibration from cache: ...
2026-04-26 14:30:38 | INFO | run_flexgen | calib: pcie=14.2GB/s disk=2.8GB/s fp16=9.1TFLOPS int4=36.4TFLOPS
2026-04-26 14:30:39 | INFO | run_flexgen | model: layers=32 hidden=4096 heads=32 kv_heads=8
2026-04-26 14:30:39 | INFO | run_flexgen | workload: prompt_len=512 decode_len=128
2026-04-26 14:30:39 | INFO | run_flexgen | Running policy search...
2026-04-26 14:31:08 | INFO | src.flexgen.policy_search | policy search: total=480 feasible=312 infeasible=168 best_t=0.0843s
2026-04-26 14:31:08 | INFO | run_flexgen | Best: gbs=8 num_gb=4 q=int4 delegate=True overlap=True -> 84.30 ms/token
2026-04-26 14:31:08 | INFO | run_flexgen | Wrote results: experiments/results/flexgen_20260426_143108.json
```

### Generating trade-off plots

```python
from analysis.plot_tradeoffs import plot_flexgen_pareto, plot_flexgen_placement_heatmap

# Use the most recent result file
import glob
latest = sorted(glob.glob("experiments/results/flexgen_*.json"))[-1]

plot_flexgen_pareto(latest, out_dir="analysis/plots")
plot_flexgen_placement_heatmap(latest, out_dir="analysis/plots")
```

Output: `analysis/plots/flexgen_pareto.png` and `flexgen_placement_heatmap.png`.

---

## 12. Step 10 — Run the test suite

```bash
# All FlexGen tests
pytest tests/flexgen/ -v

# Cost-model property tests only
pytest tests/flexgen/test_cost_model.py -v

# End-to-end integration (mocked HF + system)
pytest tests/flexgen/test_integration.py -v
```

Expected: **47+ tests pass in ~40 s**. None require GPU or network.

---

## 13. Mocking HuggingFace in your own tests

When you write new tests that touch model loading or the download utility, **don't hit the real HF Hub** — it's slow, can rate-limit you, and gated models will fail in CI.

A worked example is at [`tests/flexgen/test_hf_mock_example.py`](tests/flexgen/test_hf_mock_example.py). The two patterns:

### Pattern 1 — Mock `load_model_spec` (config-only)

```python
import json
from unittest.mock import patch
from src.flexgen.model_introspect import load_model_spec

def test_my_thing(tmp_path):
    fake_config = {
        "num_hidden_layers": 32, "hidden_size": 4096,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 14336, "vocab_size": 128256,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(fake_config))

    with patch("src.flexgen.model_introspect.snapshot_download",
               return_value=str(tmp_path)):
        spec = load_model_spec("meta-llama/Meta-Llama-3-8B")

    # ... rest of your assertions
```

### Pattern 2 — Mock `download_model` error paths

```python
from unittest.mock import patch, MagicMock
from huggingface_hub.utils import GatedRepoError
from scripts.download_model import download_model

def test_gated_model_path():
    err = GatedRepoError("locked", response=MagicMock(status_code=403))
    with patch("scripts.download_model.snapshot_download", side_effect=err):
        # download_model calls sys.exit(2) — assert that, not the exception
        with pytest.raises(SystemExit) as exc:
            download_model("some/gated-model")
        assert exc.value.code == 2
```

Run these examples to see them work:

```bash
pytest tests/flexgen/test_hf_mock_example.py -v
```

---

## 14. Troubleshooting

### "torch.cuda.is_available() returns False"

PyTorch was installed without CUDA, or the CUDA wheel doesn't match your driver. Re-run Step 3 with the correct CUDA tag from `nvidia-smi`.

### "GatedRepoError: 401 Unauthorized" when running `experiments/run_flexgen.py`

Your HF token is missing or doesn't have access to the model. Re-do Step 6:

```bash
huggingface-cli whoami        # check current login
huggingface-cli login          # re-paste token if needed
# OR
export HF_TOKEN=hf_xxxxxxxxxxxx
```

Also visit the model page in a browser and accept the license.

### "RuntimeError: Policy search found no feasible config"

Every enumerated `(gbs, num_gb, ...)` violates capacity constraints — usually because GPU VRAM is too small for even `gbs=1, num_gb=1`. Try:

- Drop the workload sizes in `configs/workload.yaml` (smaller `prompt_len` / `decode_len`).
- Free GPU memory (close other CUDA processes) and re-run.
- Use a smaller model: `--model Qwen/Qwen2-1.5B` is a good first test.

### Calibration is unreasonably slow or the numbers look wrong

```bash
python experiments/run_flexgen.py --recalibrate --verbose
```

Then check `configs/system_calibration/{your_machine_id}.json`. If `pcie_bw_gbs` is suspiciously low (< 1 GB/s) or `tflops_fp16` doesn't match your GPU's spec, run `nvidia-smi` to ensure no other process is saturating the GPU.

### "No module named 'pulp'" or similar

You forgot to activate the venv:

```bash
source .venv/bin/activate
```

### Tests fail with `tkinter.TclError`

The plotting code uses the non-interactive `Agg` backend; if you're seeing Tk errors, you're running an older copy of `analysis/plot_tradeoffs.py`. Pull the latest.

---

## 15. Reference — what each script does

| Path | Purpose |
|---|---|
| [`experiments/run_flexgen.py`](experiments/run_flexgen.py) | **Main entry point.** CLI orchestrator: probe → calibrate → introspect → search → JSON + log. |
| [`scripts/download_model.py`](scripts/download_model.py) | Optional utility to download full HF model weights (not needed by the optimizer itself). |
| [`src/flexgen/system_probe.py`](src/flexgen/system_probe.py) | Live read of free GPU VRAM / RAM / disk. |
| [`src/flexgen/calibration.py`](src/flexgen/calibration.py) | Micro-benchmarks for PCIe / disk / compute; cache I/O keyed by hostname + GPU. |
| [`src/flexgen/model_introspect.py`](src/flexgen/model_introspect.py) | Pulls `config.json` from HF Hub, derives memory footprints. |
| [`src/flexgen/workload.py`](src/flexgen/workload.py) | YAML → `WorkloadSpec`. |
| [`src/flexgen/cost_model.py`](src/flexgen/cost_model.py) | Per-layer + block latency formulas (compute, weight load, KV I/O, activation I/O, with overlap). |
| [`src/flexgen/lp_formulation.py`](src/flexgen/lp_formulation.py) | Inner LP solving the 9 placement fractions for a fixed enum point. |
| [`src/flexgen/policy_search.py`](src/flexgen/policy_search.py) | Outer enumeration over (`gbs`, `num_gb`, compression, delegate, overlap). |
| [`analysis/plot_tradeoffs.py`](analysis/plot_tradeoffs.py) | `plot_flexgen_pareto` + `plot_flexgen_placement_heatmap`. |
| [`tests/flexgen/`](tests/flexgen/) | Full pytest suite — unit, property, integration. |
| [`tests/flexgen/test_hf_mock_example.py`](tests/flexgen/test_hf_mock_example.py) | Worked examples of mocking HuggingFace calls in tests. |
| [`configs/workload.yaml`](configs/workload.yaml) | Default workload (sequence lengths). |
| [`configs/system_calibration/`](configs/system_calibration/) | Per-machine calibration cache (gitignored). |
| [`docs/superpowers/specs/2026-04-26-flexgen-faithful-design.md`](docs/superpowers/specs/2026-04-26-flexgen-faithful-design.md) | Design spec — formulas, search structure, output schema. |
| [`docs/superpowers/plans/2026-04-26-flexgen-faithful.md`](docs/superpowers/plans/2026-04-26-flexgen-faithful.md) | 14-task TDD implementation plan. |

---

## Quick recap of the full flow

```bash
git clone <repo-url> Optimization_Project && cd Optimization_Project
git checkout chaitanya && git pull

uv venv --python 3.12 && source .venv/bin/activate

uv pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
uv pip install -r requirements.txt

# (Optional) Llama needs auth
huggingface-cli login

# (Optional) Pre-download weights for later inference work
python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B

# Run the optimizer
python experiments/run_flexgen.py --model meta-llama/Meta-Llama-3-8B --verbose

# Inspect results
ls experiments/results/         # latest JSON
ls experiments/logs/            # latest log

# Run tests
pytest tests/flexgen/ -v
```

That's it. If anything in this guide is wrong or unclear, the source of truth is the design spec at [`docs/superpowers/specs/2026-04-26-flexgen-faithful-design.md`](docs/superpowers/specs/2026-04-26-flexgen-faithful-design.md).
