# Local Synthetic FlexGen Test

Use this when you are on a local machine without GPU and you do not want to load the actual Qwen model.

Script:

```bash
local_flexgen_test.py
```

It runs the same FlexGen policy search, but uses synthetic model/system parameters that you pass through the CLI.

The default values live in:

```text
config_flexgen.yml
```

Edit the `local_synthetic` section there instead of editing Python code.

## Basic Command

```bash
python local_flexgen_test.py
```

This prints all 14 FlexGen policy parameters:

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

It also prints the derived `block_size` and objective values.

## Output

The script writes:

```text
experiments/results/local_flexgen_test_<timestamp>.json
```

## No Model Load

This script does not:

- load Qwen weights
- call HuggingFace
- require CUDA
- require a GPU

It only uses numeric parameters.

## Common Local Test

Preferred: edit `local_synthetic.system` and `local_synthetic.workload` in `config_flexgen.yml`.

Temporary CLI override:

```bash
python local_flexgen_test.py \
  --gpu-vram-gb 16 \
  --ram-gb 64 \
  --disk-gb 500 \
  --prompt-len 256 \
  --decode-len 64
```

## Qwen-Like Model Parameters

The configured defaults are Qwen2-1.5B-like:

```text
num_layers: 28
hidden_dim: 1536
num_heads: 12
num_kv_heads: 2
intermediate_size: 8960
vocab_size: 151936
dtype_bytes: 2
```

Preferred: edit `local_synthetic.model` in `config_flexgen.yml`.

Temporary CLI override:

```bash
python local_flexgen_test.py \
  --num-layers 32 \
  --hidden-dim 4096 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --intermediate-size 14336
```

## System/Calibration Parameters

```bash
python local_flexgen_test.py \
  --gpu-vram-gb 24 \
  --ram-gb 128 \
  --disk-gb 1000 \
  --pcie-bw-gbs 24 \
  --disk-bw-gbs 3 \
  --tflops-fp16 80 \
  --tflops-int8 160 \
  --tflops-int4 240
```

## Use Another Config File

```bash
python local_flexgen_test.py --config my_flexgen_config.yml
```
