# Local GPU Small-Model Test

This is the successful local GPU smoke test on the RTX 4050 Laptop GPU.

## Model Used

Downloaded model:

```text
models/smollm2-135m-instruct
```

HuggingFace repo:

```text
HuggingFaceTB/SmolLM2-135M-Instruct
```

I used this instead of TinyLlama for the local smoke test because it is much smaller and safer for a 6 GB laptop GPU. TinyLlama can be tested the same way later.

## Config File

Use:

```text
config_flexgen_local_gpu.yml
```

It points to:

```yaml
paths:
  model: models/smollm2-135m-instruct
```

## Run The Full Local GPU Pipeline

```bash
python pipeline.py --config config_flexgen_local_gpu.yml
```

This does:

1. Runs FlexGen tests.
2. Loads the local model config.
3. Probes the RTX 4050 GPU.
4. Calibrates GPU/PCIe/disk speed.
5. Solves the FlexGen LP policy search.
6. Loads the small model on CUDA and generates text.
7. Prints one final baseline-vs-optimized comparison table.

If you want the detailed 14-parameter and top-k report:

```bash
python pipeline.py --config config_flexgen_local_gpu.yml --detailed-report
```

## Run Only Actual Inference

```bash
python experiments/run_qwen_inference.py \
  --model models/smollm2-135m-instruct \
  --device cuda \
  --prompt "Explain GPU memory offloading for LLM inference in two sentences." \
  --max-new-tokens 60
```

## Observed Local Result

On this machine:

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
torch: 2.5.1+cu121
CUDA available: True
generated tokens: 60
latency: about 4.2s
throughput: about 14.2 tok/s
```

## Output Files

```text
experiments/results/flexgen_<timestamp>.json
experiments/baseline_comparisons/baseline_comparison_<model>_<timestamp>.json
experiments/results/flexgen_pipeline_<timestamp>.json
experiments/results/qwen_inference_<timestamp>.json
experiments/logs/flexgen_<timestamp>.log
```
