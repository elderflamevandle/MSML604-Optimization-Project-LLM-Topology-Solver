# FlexGen Qwen GPU Server Runbook

This guide is for running only the FlexGen part of the repository on a GPU server with a Qwen model.

The main file you will run is:

```bash
python experiments/run_flexgen.py --model Qwen/Qwen2-1.5B --verbose
```

Replace `Qwen/Qwen2-1.5B` with your exact Qwen HuggingFace repo id if you downloaded a different Qwen model.

## What This Code Does

`experiments/run_flexgen.py` runs the FlexGen faithful policy search:

1. Reads live server capacity with `src/flexgen/system_probe.py`.
2. Calibrates the GPU server with `src/flexgen/calibration.py`.
3. Reads Qwen model dimensions from HuggingFace `config.json` with `src/flexgen/model_introspect.py`.
4. Reads prompt/decode lengths from `configs/workload.yaml`.
5. Solves the FlexGen placement LP and outer policy search.
6. Writes a JSON result and a log file.

This optimizer does not run real Qwen inference. It computes the FlexGen policy: batch size, number of GPU batches, compression, CPU delegation, overlap, and placement fractions across GPU, CPU, and disk.

## Important Note About Downloaded Qwen Models

The optimizer currently expects a HuggingFace repo id in `--model`, for example:

```bash
--model Qwen/Qwen2-1.5B
```

It only needs `config.json`; it does not need full weights. If you already downloaded the Qwen model through HuggingFace cache, passing the repo id is still the right command.

If you downloaded Qwen into a plain local folder only, make sure the GPU server either:

- has internet or HuggingFace cache access for that repo id, or
- has the same model available in the HuggingFace cache.

The current code does not accept `--model /path/to/local/qwen-folder` as a local path.

## Server Setup

Run these commands on the GPU server from the repository root.

```bash
cd Optimization_Project
uv venv --python 3.12
source .venv/bin/activate
```

Install PyTorch with the CUDA wheel matching the server. Check CUDA first:

```bash
nvidia-smi
```

Examples:

```bash
# CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Then install the project dependencies:

```bash
uv pip install -r requirements.txt
```

## Verify GPU Access

Run:

```bash
python -c "import torch; print(f'cuda_available={torch.cuda.is_available()}'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected:

```text
cuda_available=True
<server GPU name>
```

Then verify the FlexGen live system probe:

```bash
python -c "from src.flexgen.system_probe import probe_live_capacity; c=probe_live_capacity('.'); print(f'GPU={c.gpu_vram_gb:.1f}GB RAM={c.ram_gb:.1f}GB DISK={c.disk_gb:.1f}GB')"
```

Expected: nonzero GPU memory.

## Workload File

Default workload:

```yaml
prompt_len: 512
decode_len: 128
```

File:

```bash
configs/workload.yaml
```

Edit this file on the server if you want a larger or smaller workload.

## First Real Run On Qwen

Use your exact Qwen model id. For Qwen2 1.5B:

```bash
python experiments/run_flexgen.py --model Qwen/Qwen2-1.5B --verbose
```

For another Qwen model:

```bash
python experiments/run_flexgen.py --model Qwen/<your-qwen-model-name> --verbose
```

The first run on a new GPU server performs calibration and may take longer. It writes calibration values to:

```bash
configs/system_calibration/<hostname>_<gpu_name>.json
```

Run fresh calibration manually when needed:

```bash
python experiments/run_flexgen.py --model Qwen/Qwen2-1.5B --recalibrate --verbose
```

## Output Files

After a successful run, check:

```bash
ls -lt experiments/results/flexgen_*.json | head
ls -lt experiments/logs/flexgen_*.log | head
```

Expected output files:

```text
experiments/results/flexgen_<timestamp>.json
experiments/logs/flexgen_<timestamp>.log
```

The JSON contains:

- `best_policy.gpu_batch_size`
- `best_policy.num_gpu_batches`
- `best_policy.block_size`
- `best_policy.compression`
- `best_policy.cpu_compute_delegate`
- `best_policy.overlap_io_compute`
- `best_policy.weights`
- `best_policy.kv_cache`
- `best_policy.activations`
- `objective.per_token_latency_ms`
- `top_k_candidates`

## Generate FlexGen Plots

After the optimizer writes a result JSON:

```bash
python - <<'PY'
import glob
from analysis.plot_tradeoffs import plot_flexgen_pareto, plot_flexgen_placement_heatmap

latest = sorted(glob.glob("experiments/results/flexgen_*.json"))[-1]
print("Using", latest)
print(plot_flexgen_pareto(latest, out_dir="analysis/plots"))
print(plot_flexgen_placement_heatmap(latest, out_dir="analysis/plots"))
PY
```

Expected plot files:

```text
analysis/plots/flexgen_pareto.png
analysis/plots/flexgen_placement_heatmap.png
```

## Focused Tests

Run only the FlexGen tests:

```bash
pytest tests/flexgen/ -q
```

These tests do not require a GPU or internet because they use mocks for the heavy pieces.

## Files You Need To Know

```text
experiments/run_flexgen.py          Main FlexGen CLI
experiments/run_qwen_inference.py   Actual Qwen text generation demo
scripts/download_model.py           Optional model download helper
configs/workload.yaml               Prompt/decode workload
src/flexgen/system_probe.py         Live GPU/RAM/disk capacity
src/flexgen/calibration.py          GPU server calibration and cache
src/flexgen/model_introspect.py     Qwen config parsing
src/flexgen/qwen_inference.py       Actual Qwen model loading and generation
src/flexgen/cost_model.py           FlexGen latency model
src/flexgen/lp_formulation.py       Inner LP placement solver
src/flexgen/policy_search.py        Outer policy search
analysis/plot_tradeoffs.py          Result plots
tests/flexgen/                      Focused FlexGen tests
```

## Actual Qwen Inference For The Demo

The FlexGen optimizer computes the placement policy. To also show real model execution, run:

```bash
python experiments/run_qwen_inference.py \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --prompt "Explain FlexGen in simple terms." \
  --max-new-tokens 80
```

If you downloaded Qwen into a local folder, put that folder path in `--model`:

```bash
python experiments/run_qwen_inference.py \
  --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ \
  --device cuda \
  --prompt "Explain FlexGen in simple terms." \
  --max-new-tokens 80
```

That is the path you replace. You do not edit code for the path.

Your local model folder should look roughly like:

```text
/home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors
```

Some models split weights across multiple `.safetensors` files; that is fine.

The inference result is saved to:

```text
experiments/results/qwen_inference_<timestamp>.json
```

For multi-GPU server loading, try Transformers device mapping:

```bash
python experiments/run_qwen_inference.py \
  --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ \
  --device cuda \
  --device-map auto \
  --prompt "Explain FlexGen in simple terms." \
  --max-new-tokens 80
```

If `--device-map auto` asks for `accelerate`, install it on the server:

```bash
uv pip install accelerate
```

The FlexGen optimizer can also read `config.json` from that exact local folder:

```bash
python experiments/run_flexgen.py \
  --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ \
  --verbose
```

For the one-command pipeline, the same path is stored in:

```text
config_flexgen.yml
```

Run:

```bash
python pipeline.py
```

## Troubleshooting

If CUDA is false:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Reinstall PyTorch with the CUDA wheel matching `nvidia-smi`.

If HuggingFace cannot find the Qwen model:

```bash
python scripts/download_model.py --model Qwen/Qwen2-1.5B --config-only
```

Then rerun:

```bash
python experiments/run_flexgen.py --model Qwen/Qwen2-1.5B --verbose
```

If policy search says no feasible config, try a smaller workload:

```yaml
prompt_len: 256
decode_len: 64
```

Then rerun the optimizer.
