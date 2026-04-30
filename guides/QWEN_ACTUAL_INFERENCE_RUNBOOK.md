# Qwen Actual Inference Runbook

This guide is for the final-project demo where you want to actually load Qwen and generate text on a GPU.

The new script is:

```bash
experiments/run_qwen_inference.py
```

The code behind it is:

```bash
src/flexgen/qwen_inference.py
```

## Where To Put Your Downloaded Model Path

You put the path in the `--model` argument.

Example local folder:

```bash
python experiments/run_qwen_inference.py \
  --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ \
  --device cuda \
  --prompt "Explain FlexGen in simple terms." \
  --max-new-tokens 80
```

You do not replace a `.pth` path in code.

For HuggingFace cached/downloaded models, you can also use the repo id:

```bash
python experiments/run_qwen_inference.py \
  --model Qwen/Qwen2-1.5B \
  --device cuda \
  --prompt "Explain FlexGen in simple terms." \
  --max-new-tokens 80
```

## What The Local Qwen Folder Must Contain

The folder should contain HuggingFace-format files:

```text
/home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors
```

It may also contain split weights:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

That is okay. Transformers can load these.

## GPU Server Setup

```bash
cd Optimization_Project
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected:

```text
True
<GPU name>
```

## Run Actual Inference

Greedy decoding:

```bash
python experiments/run_qwen_inference.py \
  --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ \
  --device cuda \
  --prompt "Give a short explanation of GPU memory offloading for LLM inference." \
  --max-new-tokens 100
```

Sampling:

```bash
python experiments/run_qwen_inference.py \
  --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ \
  --device cuda \
  --prompt "Give a short explanation of GPU memory offloading for LLM inference." \
  --max-new-tokens 100 \
  --temperature 0.7 \
  --top-p 0.9
```

Multi-GPU or automatic placement:

```bash
python experiments/run_qwen_inference.py \
  --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ \
  --device cuda \
  --device-map auto \
  --prompt "Give a short explanation of GPU memory offloading for LLM inference." \
  --max-new-tokens 100
```

If the command asks for Accelerate:

```bash
uv pip install accelerate
```

## Output

The script prints:

- generated text
- prompt token count
- generated token count
- latency
- tokens per second

It also writes:

```text
experiments/results/qwen_inference_<timestamp>.json
```

## How To Present This In The Final Project

Run both commands:

```bash
python experiments/run_flexgen.py --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ --verbose
python experiments/run_qwen_inference.py --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ --device cuda --prompt "Explain FlexGen in simple terms." --max-new-tokens 80
```

Then show:

- FlexGen policy JSON from `experiments/results/flexgen_<timestamp>.json`
- actual Qwen inference JSON from `experiments/results/qwen_inference_<timestamp>.json`
- terminal output with real generated text and tokens/sec
