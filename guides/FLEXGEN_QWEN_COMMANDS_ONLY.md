# FlexGen Qwen GPU Server Commands

Run these on the GPU server from the repo root.

## 0. One-Command Pipeline

This runs FlexGen tests, runs the optimizer on your local Qwen folder, prints the 14 policy parameters, and saves a pipeline summary JSON:

```bash
python pipeline.py
```

The final terminal section is one baseline-vs-optimized comparison table. For the full 14-parameter/top-k report:

```bash
python pipeline.py --detailed-report
```

Settings come from:

```bash
config_flexgen.yml
```

With actual Qwen text generation too:

```bash
python pipeline.py --run-inference --device cuda
```

Baseline comparison JSON is saved under:

```text
experiments/baseline_comparisons/
```

## 0.1 Local Synthetic Test, No Qwen Load

This is for your laptop/local machine. It does not load Qwen, does not call HuggingFace, and does not require GPU:

```bash
python local_flexgen_test.py
```

Synthetic values come from `config_flexgen.yml`.

Change synthetic parameters:

```bash
python local_flexgen_test.py --gpu-vram-gb 16 --ram-gb 64 --prompt-len 256 --decode-len 64
```

## 1. Environment

```bash
cd Optimization_Project

uv venv --python 3.12
source .venv/bin/activate

# Pick the CUDA wheel that matches nvidia-smi.
# Example for CUDA 12.1:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

uv pip install -r requirements.txt
```

## 2. Verify GPU

```bash
nvidia-smi

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

python -c "from src.flexgen.system_probe import probe_live_capacity; c=probe_live_capacity('.'); print(f'GPU={c.gpu_vram_gb:.1f}GB RAM={c.ram_gb:.1f}GB DISK={c.disk_gb:.1f}GB')"
```

## 3. Run FlexGen on Qwen

Use your exact Qwen HuggingFace repo id. Common examples:

```bash
python experiments/run_flexgen.py --model Qwen/Qwen2-1.5B --verbose
```

If you are using a different downloaded Qwen model:

```bash
python experiments/run_flexgen.py --model Qwen/<your-qwen-model-name> --verbose
```

Force fresh GPU-server calibration:

```bash
python experiments/run_flexgen.py --model Qwen/Qwen2-1.5B --recalibrate --verbose
```

## 4. Check Outputs

```bash
ls -lt experiments/results/flexgen_*.json | head
ls -lt experiments/logs/flexgen_*.log | head
```

## 5. Generate Plots From Latest Result

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

## 6. Focused Test Suite

```bash
pytest tests/flexgen/ -q
```

## 7. Actual Qwen Inference Demo

Use a HuggingFace id:

```bash
python experiments/run_qwen_inference.py --model Qwen/Qwen2-1.5B --device cuda --prompt "Explain FlexGen in simple terms." --max-new-tokens 80
```

Use your downloaded local Qwen folder:

```bash
python experiments/run_qwen_inference.py --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ --device cuda --prompt "Explain FlexGen in simple terms." --max-new-tokens 80
```

The local folder must contain `config.json`, tokenizer files, and model weights such as `.safetensors` or `.bin`.

You can also run the FlexGen optimizer from that same local Qwen folder:

```bash
python experiments/run_flexgen.py --model /home/cbagul07/MSML604/MSML604-Optimization-Project-LLM-Topology-Solver/models/Qwen/ --verbose
```
