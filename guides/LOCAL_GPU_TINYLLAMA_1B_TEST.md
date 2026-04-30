# Local GPU TinyLlama 1.1B Test

This is the 1B-class local GPU test.

## Model

Downloaded to:

```text
models/tinyllama-1.1b-chat
```

HuggingFace repo:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Config

```text
config_flexgen_tinyllama_local_gpu.yml
```

## Run With Tests

```bash
python pipeline.py --config config_flexgen_tinyllama_local_gpu.yml
```

## Run Faster Without Tests

```bash
python pipeline.py --config config_flexgen_tinyllama_local_gpu.yml --skip-tests
```

## Latest Observed Result

Policy search:

```text
optimized_policy predicted latency: 12.6216 ms/token
optimized_policy predicted throughput: 79.2292 tok/s
manual fp16 baseline predicted latency: 49.0392 ms/token
improvement over fp16 baseline: about 3.89x lower latency
```

Actual GPU inference:

```text
generated_tokens: 50
latency_s: about 5.5
tokens_per_s: about 9.0
```

Output files:

```text
experiments/results/flexgen_<timestamp>.json
experiments/baseline_comparisons/baseline_comparison_models_tinyllama-1.1b-chat_<timestamp>.json
experiments/results/qwen_inference_<timestamp>.json
experiments/results/flexgen_pipeline_<timestamp>.json
```

