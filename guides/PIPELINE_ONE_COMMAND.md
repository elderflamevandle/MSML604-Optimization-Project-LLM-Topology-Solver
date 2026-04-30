# One-Command FlexGen Pipeline

Run this from the repo root on the HPC GPU server:

```bash
python pipeline.py
```

By default, it reads settings from:

```text
config_flexgen.yml
```

That config currently points to your local Qwen folder:

```text
/home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/
```

Edit `config_flexgen.yml`, not `pipeline.py`, when the model path or run settings change.

## What It Does

`pipeline.py` performs these steps:

1. Runs all FlexGen tests:

   ```bash
   python -m pytest tests/flexgen -q
   ```

2. Runs FlexGen policy search:

   ```bash
   python experiments/run_flexgen.py \
     --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/
   ```

3. Prints the 14 FlexGen policy parameters:

   ```text
   gpu_batch_size
   num_gpu_batches
   compression
   cpu_compute_delegate
   overlap_io_compute
   weights_gpu
   weights_cpu
   weights_disk
   kv_cache_gpu
   kv_cache_cpu
   kv_cache_disk
   activations_gpu
   activations_cpu
   activations_disk
   ```

4. Prints the derived `block_size`.

5. Saves a combined summary JSON:

   ```text
   experiments/results/flexgen_pipeline_<timestamp>.json
   ```

## Basic Command

```bash
python pipeline.py
```

The normal terminal output is intentionally concise: tests run first, policy search runs next, files are saved, and the final terminal section is one baseline-vs-optimized comparison table.

For the full model/system/14-parameter/top-k dump:

```bash
python pipeline.py --detailed-report
```

## Verbose Optimizer Logs

```bash
python pipeline.py --verbose
```

## Force Fresh GPU Calibration

```bash
python pipeline.py --recalibrate --verbose
```

## Skip Tests

Use this when you already ran tests and only want a new policy result:

```bash
python pipeline.py --skip-tests --verbose
```

## Run Actual Qwen Inference Too

This loads Qwen and generates text after the FlexGen optimizer finishes:

```bash
python pipeline.py \
  --run-inference \
  --device cuda \
  --prompt "Explain FlexGen in simple terms." \
  --max-new-tokens 80
```

For multi-GPU Transformers placement:

```bash
python pipeline.py \
  --run-inference \
  --device cuda \
  --device-map auto \
  --prompt "Explain FlexGen in simple terms." \
  --max-new-tokens 80
```

## Override Model Path

Preferred: edit `paths.model` in `config_flexgen.yml`.

Temporary CLI override:

```bash
python pipeline.py --model /new/path/to/Qwen/
```

## Use Another Config File

```bash
python pipeline.py --config my_flexgen_config.yml
```

## Output Files

The pipeline writes:

```text
experiments/results/flexgen_<timestamp>.json
experiments/baseline_comparisons/baseline_comparison_<model>_<timestamp>.json
experiments/logs/flexgen_<timestamp>.log
experiments/results/flexgen_pipeline_<timestamp>.json
```

If `--run-inference` is used, it also writes:

```text
experiments/results/qwen_inference_<timestamp>.json
```
