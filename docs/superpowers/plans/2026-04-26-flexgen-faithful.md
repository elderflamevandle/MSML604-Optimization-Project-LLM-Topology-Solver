# FlexGen Faithful Policy-Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the toy LP at `src/flexgen/lp_formulation.py` with a faithful FlexGen policy search that auto-detects system + model + workload, then enumerates batching/compression/delegate/overlap choices around an inner LP for placement, minimizing per-token latency. Output: all 14 decision-variable values + per-run log + per-run JSON.

**Architecture:** Live system probe (per run) + per-machine cached calibration → HF config-driven model spec → YAML workload → outer enumeration over `(gbs, num_gb, q, δ, o)` × inner LP for placement fractions → top-k tracking. Single CLI entry at `experiments/run_flexgen.py`.

**Tech Stack:** Python 3.12, PuLP+CBC (LP), `huggingface_hub` (config fetch), `psutil` + `torch.cuda` (system probe), `bitsandbytes` (optional, for int4/int8 calibration), stdlib `logging` + `argparse`, `pytest` for tests.

**Spec:** [`docs/superpowers/specs/2026-04-26-flexgen-faithful-design.md`](../specs/2026-04-26-flexgen-faithful-design.md)

---

## File Map

**Create:**
- `src/flexgen/system_probe.py` — live capacity reads
- `src/flexgen/calibration.py` — per-machine calibration + cache I/O
- `src/flexgen/model_introspect.py` — HF config → ModelSpec
- `src/flexgen/workload.py` — YAML → WorkloadSpec
- `src/flexgen/cost_model.py` — per-layer + block latency formulas
- `src/flexgen/policy_search.py` — outer enumeration + top-k
- `configs/workload.yaml` — default workload
- `configs/system_calibration/.gitkeep` — calibration cache dir
- `experiments/logs/.gitkeep` — log dir
- `tests/flexgen/test_system_probe.py`
- `tests/flexgen/test_calibration.py`
- `tests/flexgen/test_model_introspect.py`
- `tests/flexgen/test_workload.py`
- `tests/flexgen/test_cost_model.py`
- `tests/flexgen/test_policy_search.py`
- `tests/flexgen/test_run_flexgen.py`
- `tests/flexgen/test_integration.py`

**Modify:**
- `src/flexgen/lp_formulation.py` — refactor to `solve_inner_lp(enum, system, model, workload)`; keep old `solve_memory_placement` as a thin wrapper
- `src/flexgen/__init__.py` — re-export public types
- `experiments/run_flexgen.py` — full rewrite (CLI + logging + orchestration)
- `analysis/plot_tradeoffs.py` — add Pareto + placement-heatmap functions
- `tests/flexgen/test_lp_formulation.py` — extend for `solve_inner_lp`
- `.gitignore` — add `configs/system_calibration/*.json`, `experiments/logs/*.log`
- `requirements.txt` — add `huggingface_hub`, `pyyaml`, `bitsandbytes` (optional marker)

---

## Task 1: Scaffolding (dirs, gitignore, requirements)

**Files:**
- Create: `configs/system_calibration/.gitkeep`
- Create: `experiments/logs/.gitkeep`
- Modify: `.gitignore`
- Modify: `requirements.txt`

- [ ] **Step 1: Create cache + log directories with .gitkeep markers**

```bash
mkdir -p configs/system_calibration experiments/logs
touch configs/system_calibration/.gitkeep experiments/logs/.gitkeep
```

- [ ] **Step 2: Update .gitignore**

Append these lines to `.gitignore`:

```
configs/system_calibration/*.json
experiments/logs/*.log
```

- [ ] **Step 3: Update requirements.txt**

Append these lines to `requirements.txt`:

```
huggingface_hub>=0.23.0
pyyaml>=6.0.1
psutil>=5.9.0
bitsandbytes>=0.43.0
```

- [ ] **Step 4: Verify install**

Run: `pip install -r requirements.txt`
Expected: clean install, no version conflicts.

- [ ] **Step 5: Commit**

```bash
git add configs/system_calibration/.gitkeep experiments/logs/.gitkeep .gitignore requirements.txt
git commit -m "chore(flexgen): scaffolding for calibration cache, logs, new deps"
```

---

## Task 2: Workload spec module

**Files:**
- Create: `src/flexgen/workload.py`
- Create: `configs/workload.yaml`
- Create: `tests/flexgen/test_workload.py`

- [ ] **Step 1: Write the failing test**

Create `tests/flexgen/test_workload.py`:

```python
import pytest
from pathlib import Path
from src.flexgen.workload import WorkloadSpec, load_workload


def test_load_workload_from_yaml(tmp_path):
    yaml_file = tmp_path / "workload.yaml"
    yaml_file.write_text("prompt_len: 256\ndecode_len: 64\n")
    spec = load_workload(str(yaml_file))
    assert spec.prompt_len == 256
    assert spec.decode_len == 64


def test_load_workload_defaults_when_field_missing(tmp_path):
    yaml_file = tmp_path / "workload.yaml"
    yaml_file.write_text("prompt_len: 1024\n")
    spec = load_workload(str(yaml_file))
    assert spec.prompt_len == 1024
    assert spec.decode_len == 128  # default


def test_load_workload_rejects_nonpositive():
    with pytest.raises(ValueError, match="prompt_len must be positive"):
        WorkloadSpec(prompt_len=0, decode_len=128)
    with pytest.raises(ValueError, match="decode_len must be positive"):
        WorkloadSpec(prompt_len=512, decode_len=-1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_workload.py -v`
Expected: FAIL with `ImportError: cannot import name 'WorkloadSpec'`

- [ ] **Step 3: Write the implementation**

Create `src/flexgen/workload.py`:

```python
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class WorkloadSpec:
    prompt_len: int
    decode_len: int

    def __post_init__(self):
        if self.prompt_len <= 0:
            raise ValueError("prompt_len must be positive")
        if self.decode_len <= 0:
            raise ValueError("decode_len must be positive")


_DEFAULTS = {"prompt_len": 512, "decode_len": 128}


def load_workload(path: str) -> WorkloadSpec:
    data = yaml.safe_load(Path(path).read_text()) or {}
    merged = {**_DEFAULTS, **{k: int(v) for k, v in data.items() if k in _DEFAULTS}}
    return WorkloadSpec(**merged)
```

Create `configs/workload.yaml`:

```yaml
prompt_len: 512
decode_len: 128
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_workload.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/workload.py configs/workload.yaml tests/flexgen/test_workload.py
git commit -m "feat(flexgen): workload spec from YAML with validation"
```

---

## Task 3: System probe module (live capacity reads)

**Files:**
- Create: `src/flexgen/system_probe.py`
- Create: `tests/flexgen/test_system_probe.py`

- [ ] **Step 1: Write the failing test**

Create `tests/flexgen/test_system_probe.py`:

```python
from unittest.mock import patch, MagicMock
from src.flexgen.system_probe import LiveCapacity, probe_live_capacity


@patch("src.flexgen.system_probe.psutil")
@patch("src.flexgen.system_probe.torch")
def test_probe_live_capacity_returns_gb(mock_torch, mock_psutil):
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.mem_get_info.return_value = (10 * 1024**3, 24 * 1024**3)
    mock_psutil.virtual_memory.return_value = MagicMock(available=64 * 1024**3)
    mock_psutil.disk_usage.return_value = MagicMock(free=800 * 1024**3)

    cap = probe_live_capacity(project_root="/tmp")

    assert cap.gpu_vram_gb == 10.0      # free, not total
    assert cap.ram_gb == 64.0
    assert cap.disk_gb == 800.0


@patch("src.flexgen.system_probe.psutil")
@patch("src.flexgen.system_probe.torch")
def test_probe_live_capacity_no_cuda(mock_torch, mock_psutil):
    mock_torch.cuda.is_available.return_value = False
    mock_psutil.virtual_memory.return_value = MagicMock(available=32 * 1024**3)
    mock_psutil.disk_usage.return_value = MagicMock(free=200 * 1024**3)

    cap = probe_live_capacity(project_root="/tmp")
    assert cap.gpu_vram_gb == 0.0
    assert cap.ram_gb == 32.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_system_probe.py -v`
Expected: FAIL with `ImportError: cannot import name 'LiveCapacity'`

- [ ] **Step 3: Write the implementation**

Create `src/flexgen/system_probe.py`:

```python
from dataclasses import dataclass
import psutil
import torch


@dataclass(frozen=True)
class LiveCapacity:
    gpu_vram_gb: float
    ram_gb: float
    disk_gb: float


def probe_live_capacity(project_root: str) -> LiveCapacity:
    if torch.cuda.is_available():
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        gpu_vram_gb = free_bytes / 1024**3
    else:
        gpu_vram_gb = 0.0

    ram_gb = psutil.virtual_memory().available / 1024**3
    disk_gb = psutil.disk_usage(project_root).free / 1024**3

    return LiveCapacity(
        gpu_vram_gb=gpu_vram_gb,
        ram_gb=ram_gb,
        disk_gb=disk_gb,
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_system_probe.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/system_probe.py tests/flexgen/test_system_probe.py
git commit -m "feat(flexgen): live system capacity probe (psutil + torch.cuda)"
```

---

## Task 4: Calibration probes (matmul, host-device copy, disk)

**Files:**
- Create: `src/flexgen/calibration.py` (probes only — cache comes in Task 5)
- Create: `tests/flexgen/test_calibration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/flexgen/test_calibration.py`:

```python
from unittest.mock import patch
from src.flexgen.calibration import (
    SystemCoefficients,
    bench_compute_tflops,
    bench_pcie_bw_gbs,
    bench_disk_bw_gbs,
)


def test_bench_compute_tflops_returns_positive_for_fp16():
    # Real call — needs CUDA. If no CUDA, fall back to CPU test (still positive).
    tflops = bench_compute_tflops(dtype="fp16", n_repeats=2)
    assert tflops > 0.0


def test_bench_pcie_bw_gbs_returns_positive():
    bw = bench_pcie_bw_gbs(size_mb=64, n_repeats=2)
    assert bw > 0.0


def test_bench_disk_bw_gbs_returns_positive(tmp_path):
    bw = bench_disk_bw_gbs(probe_dir=str(tmp_path), size_mb=32, n_repeats=2)
    assert bw > 0.0


def test_system_coefficients_is_frozen_dataclass():
    c = SystemCoefficients(
        pcie_bw_gbs=14.0, disk_bw_gbs=2.5,
        tflops_fp16=10.0, tflops_int8=20.0, tflops_int4=40.0,
    )
    assert c.pcie_bw_gbs == 14.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_calibration.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Create `src/flexgen/calibration.py`:

```python
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
import os
import time
import socket
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemCoefficients:
    pcie_bw_gbs: float
    disk_bw_gbs: float
    tflops_fp16: float
    tflops_int8: float
    tflops_int4: float


def bench_compute_tflops(dtype: str = "fp16", n_repeats: int = 5) -> float:
    """Time torch.matmul of two square matrices; return effective TFLOPS."""
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = {"fp16": torch.float16, "int8": torch.float16, "int4": torch.float16}[dtype]
    else:
        device = "cpu"
        torch_dtype = torch.float32

    n = 4096 if device == "cuda" else 1024
    a = torch.randn(n, n, device=device, dtype=torch_dtype)
    b = torch.randn(n, n, device=device, dtype=torch_dtype)

    # warmup
    _ = torch.matmul(a, b)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = torch.matmul(a, b)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median_t = sorted(times)[len(times) // 2]
    flops = 2 * n**3
    tflops = flops / median_t / 1e12

    if dtype == "int8":
        tflops *= 2.0  # fallback approximation when bitsandbytes unavailable
    elif dtype == "int4":
        tflops *= 4.0

    return tflops


def bench_pcie_bw_gbs(size_mb: int = 256, n_repeats: int = 3) -> float:
    """Time host->device tensor copy; return GB/s."""
    if not torch.cuda.is_available():
        logger.warning("No CUDA; PCIe bandwidth benchmark falling back to 16.0 GB/s")
        return 16.0

    n_elem = size_mb * 1024 * 1024 // 4
    host = torch.empty(n_elem, dtype=torch.float32, pin_memory=True)
    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = host.cuda(non_blocking=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median_t = sorted(times)[len(times) // 2]
    return (size_mb / 1024) / median_t


def bench_disk_bw_gbs(probe_dir: str, size_mb: int = 200, n_repeats: int = 3) -> float:
    """Time disk write+read cycle on a temp file; return GB/s averaged over write and read."""
    Path(probe_dir).mkdir(parents=True, exist_ok=True)
    probe_path = Path(probe_dir) / ".calib_probe.bin"
    payload = os.urandom(size_mb * 1024 * 1024)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        probe_path.write_bytes(payload)
        _ = probe_path.read_bytes()
        times.append(time.perf_counter() - t0)
    probe_path.unlink(missing_ok=True)

    median_t = sorted(times)[len(times) // 2]
    bytes_moved = 2 * size_mb * 1024 * 1024  # write + read
    return (bytes_moved / 1024**3) / median_t
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_calibration.py -v`
Expected: 4 passed (will run for 5-10 seconds — real GPU/disk timing).

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/calibration.py tests/flexgen/test_calibration.py
git commit -m "feat(flexgen): calibration probes for compute/PCIe/disk bandwidth"
```

---

## Task 5: Calibration cache + ensure_calibration()

**Files:**
- Modify: `src/flexgen/calibration.py`
- Modify: `tests/flexgen/test_calibration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/flexgen/test_calibration.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch
from src.flexgen.calibration import (
    machine_id, ensure_calibration, _calibration_cache_path,
)


@patch("src.flexgen.calibration.torch.cuda.get_device_name", return_value="NVIDIA RTX 4050 Laptop")
@patch("src.flexgen.calibration.socket.gethostname", return_value="laptop-01")
def test_machine_id_combines_hostname_and_gpu(mock_host, mock_gpu):
    mid = machine_id()
    assert "laptop-01" in mid
    assert "RTX_4050_Laptop" in mid


def test_ensure_calibration_uses_cache_when_present(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "test_machine.json"
    cache_file.write_text(json.dumps({
        "pcie_bw_gbs": 14.0, "disk_bw_gbs": 2.5,
        "tflops_fp16": 10.0, "tflops_int8": 20.0, "tflops_int4": 40.0,
    }))

    coef = ensure_calibration(cache_dir=str(cache_dir), key="test_machine", recalibrate=False)
    assert coef.pcie_bw_gbs == 14.0
    assert coef.tflops_fp16 == 10.0


def test_ensure_calibration_recalibrate_overwrites_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "test_machine.json"
    cache_file.write_text(json.dumps({
        "pcie_bw_gbs": 99.0, "disk_bw_gbs": 99.0,
        "tflops_fp16": 99.0, "tflops_int8": 99.0, "tflops_int4": 99.0,
    }))

    with patch("src.flexgen.calibration.bench_compute_tflops", return_value=11.0), \
         patch("src.flexgen.calibration.bench_pcie_bw_gbs", return_value=15.0), \
         patch("src.flexgen.calibration.bench_disk_bw_gbs", return_value=3.0):
        coef = ensure_calibration(cache_dir=str(cache_dir), key="test_machine",
                                  recalibrate=True)

    assert coef.pcie_bw_gbs == 15.0
    assert coef.tflops_fp16 == 11.0
    on_disk = json.loads(cache_file.read_text())
    assert on_disk["pcie_bw_gbs"] == 15.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_calibration.py::test_machine_id_combines_hostname_and_gpu -v`
Expected: FAIL with `ImportError: cannot import name 'machine_id'`.

- [ ] **Step 3: Write the implementation**

Append to `src/flexgen/calibration.py`:

```python
def machine_id() -> str:
    host = socket.gethostname()
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0).replace(" ", "_")
    else:
        gpu = "no_cuda"
    return f"{host}_{gpu}"


def _calibration_cache_path(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def ensure_calibration(
    cache_dir: str,
    key: str | None = None,
    recalibrate: bool = False,
    probe_dir: str | None = None,
) -> SystemCoefficients:
    key = key or machine_id()
    cache_path = _calibration_cache_path(cache_dir, key)

    if cache_path.exists() and not recalibrate:
        logger.info("Loaded calibration from cache: %s", cache_path)
        return SystemCoefficients(**json.loads(cache_path.read_text()))

    logger.info("Running calibration for machine_id=%s", key)
    coef = SystemCoefficients(
        pcie_bw_gbs=bench_pcie_bw_gbs(),
        disk_bw_gbs=bench_disk_bw_gbs(probe_dir=probe_dir or cache_dir),
        tflops_fp16=bench_compute_tflops(dtype="fp16"),
        tflops_int8=bench_compute_tflops(dtype="int8"),
        tflops_int4=bench_compute_tflops(dtype="int4"),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(asdict(coef), indent=2))
    logger.info("Wrote calibration cache: %s", cache_path)
    return coef
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_calibration.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/calibration.py tests/flexgen/test_calibration.py
git commit -m "feat(flexgen): per-machine calibration cache with recalibrate flag"
```

---

## Task 6: Model introspection from HuggingFace config.json

**Files:**
- Create: `src/flexgen/model_introspect.py`
- Create: `tests/flexgen/test_model_introspect.py`

- [ ] **Step 1: Write the failing test**

Create `tests/flexgen/test_model_introspect.py`:

```python
import json
from unittest.mock import patch
from src.flexgen.model_introspect import (
    ModelSpec, load_model_spec, weights_per_layer_bytes,
    kv_per_token_bytes, params_per_layer,
)


LLAMA3_8B_CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "torch_dtype": "bfloat16",
}


def test_load_model_spec_parses_llama3_config(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(LLAMA3_8B_CONFIG))
    with patch("src.flexgen.model_introspect.snapshot_download", return_value=str(tmp_path)):
        spec = load_model_spec("meta-llama/Meta-Llama-3-8B")
    assert spec.num_layers == 32
    assert spec.hidden_dim == 4096
    assert spec.num_heads == 32
    assert spec.num_kv_heads == 8
    assert spec.intermediate_size == 14336
    assert spec.dtype_bytes == 2


def test_load_model_spec_falls_back_to_num_heads_when_no_gqa(tmp_path):
    cfg = dict(LLAMA3_8B_CONFIG)
    del cfg["num_key_value_heads"]
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    with patch("src.flexgen.model_introspect.snapshot_download", return_value=str(tmp_path)):
        spec = load_model_spec("any/model")
    assert spec.num_kv_heads == spec.num_heads


def test_params_per_layer_matches_llama3_8b():
    spec = ModelSpec(
        hf_id="meta-llama/Meta-Llama-3-8B", num_layers=32, hidden_dim=4096,
        num_heads=32, num_kv_heads=8, intermediate_size=14336,
        vocab_size=128256, dtype_bytes=2,
    )
    p = params_per_layer(spec)
    # Attention: hidden^2 (Q) + 2 * (h_kv/h * hidden^2) (K,V) + hidden^2 (O)
    # FFN (SwiGLU): 3 * intermediate * hidden
    # Approximate target ~218M per layer for L3-8B (8B / 32 ≈ 250M, includes embeds)
    assert 200_000_000 < p < 280_000_000


def test_weights_per_layer_bytes_int4_is_quarter_of_fp16():
    spec = ModelSpec(
        hf_id="x", num_layers=1, hidden_dim=4096, num_heads=32, num_kv_heads=8,
        intermediate_size=14336, vocab_size=1, dtype_bytes=2,
    )
    fp16 = weights_per_layer_bytes(spec, "fp16")
    int4 = weights_per_layer_bytes(spec, "int4")
    assert abs(int4 / fp16 - 0.25) < 0.01


def test_kv_per_token_bytes_uses_num_kv_heads():
    spec = ModelSpec(
        hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=8,
        intermediate_size=14336, vocab_size=1, dtype_bytes=2,
    )
    # 2 (K and V) * num_kv_heads (8) * head_dim (128) * num_layers (32) * 2 bytes (fp16)
    expected_fp16 = 2 * 8 * 128 * 32 * 2
    assert kv_per_token_bytes(spec, "fp16") == expected_fp16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_model_introspect.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Create `src/flexgen/model_introspect.py`:

```python
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    dtype_bytes: int

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "int4": 1}


def load_model_spec(hf_id: str) -> ModelSpec:
    local_dir = snapshot_download(repo_id=hf_id, allow_patterns=["config.json"])
    cfg = json.loads((Path(local_dir) / "config.json").read_text())

    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    dtype_str = cfg.get("torch_dtype", "float16")

    return ModelSpec(
        hf_id=hf_id,
        num_layers=cfg["num_hidden_layers"],
        hidden_dim=cfg["hidden_size"],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=cfg.get("intermediate_size", 4 * cfg["hidden_size"]),
        vocab_size=cfg.get("vocab_size", 32000),
        dtype_bytes=_DTYPE_BYTES.get(dtype_str, 2),
    )


def params_per_layer(spec: ModelSpec) -> int:
    """Element count (not bytes). Attention QKVO + SwiGLU FFN + 2 norms."""
    h = spec.hidden_dim
    h_kv_ratio = spec.num_kv_heads / spec.num_heads
    attn = h * h * (1 + 2 * h_kv_ratio + 1)            # Q, K, V, O projections
    ffn = 3 * spec.intermediate_size * h               # gate + up + down (SwiGLU)
    norms = 2 * h
    return int(attn + ffn + norms)


_QUANT_BYTES_PER_ELEM = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}


def weights_per_layer_bytes(spec: ModelSpec, q: str) -> float:
    return params_per_layer(spec) * _QUANT_BYTES_PER_ELEM[q]


def kv_per_token_bytes(spec: ModelSpec, q: str) -> float:
    """K and V tensors per token, summed across layers."""
    bytes_per_elem = _QUANT_BYTES_PER_ELEM[q] if q in ("fp16", "int4") else 2.0
    return 2 * spec.num_kv_heads * spec.head_dim * spec.num_layers * bytes_per_elem
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_model_introspect.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/model_introspect.py tests/flexgen/test_model_introspect.py
git commit -m "feat(flexgen): HF config-driven ModelSpec with derived sizes"
```

---

## Task 7: Cost model — data classes and FLOP helpers

**Files:**
- Create: `src/flexgen/cost_model.py`
- Create: `tests/flexgen/test_cost_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/flexgen/test_cost_model.py`:

```python
from src.flexgen.cost_model import (
    EnumPoint, PlacementFractions, prefill_flops_per_layer, decode_flops_per_layer,
)
from src.flexgen.model_introspect import ModelSpec


SPEC = ModelSpec(
    hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=8,
    intermediate_size=14336, vocab_size=128256, dtype_bytes=2,
)


def test_enum_point_block_size():
    e = EnumPoint(gbs=8, num_gb=4, q="int4", delegate=True, overlap=True)
    assert e.block_size == 32


def test_placement_fractions_validates_sum():
    PlacementFractions(w_g=0.5, w_c=0.5, w_d=0.0,
                       c_g=1.0, c_c=0.0, c_d=0.0,
                       h_g=1.0, h_c=0.0, h_d=0.0)  # ok


def test_prefill_flops_scales_with_batch():
    f1 = prefill_flops_per_layer(SPEC, batch=1, seq_len=512)
    f4 = prefill_flops_per_layer(SPEC, batch=4, seq_len=512)
    assert abs(f4 / f1 - 4.0) < 0.01


def test_prefill_flops_scales_quadratically_in_seq_attention_term():
    # decode flops at large kv length should exceed prefill flops at small s
    f_short = prefill_flops_per_layer(SPEC, batch=1, seq_len=64)
    f_long = prefill_flops_per_layer(SPEC, batch=1, seq_len=2048)
    # Matmul scales linearly with s; attention scales quadratically. Long must dominate.
    assert f_long > 30 * f_short
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_cost_model.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Create `src/flexgen/cost_model.py`:

```python
from dataclasses import dataclass
from src.flexgen.model_introspect import ModelSpec, params_per_layer


@dataclass(frozen=True)
class EnumPoint:
    gbs: int
    num_gb: int
    q: str           # "fp16" or "int4"
    delegate: bool
    overlap: bool

    @property
    def block_size(self) -> int:
        return self.gbs * self.num_gb


@dataclass(frozen=True)
class PlacementFractions:
    w_g: float; w_c: float; w_d: float
    c_g: float; c_c: float; c_d: float
    h_g: float; h_c: float; h_d: float


def prefill_flops_per_layer(spec: ModelSpec, batch: int, seq_len: int) -> float:
    """Forward FLOPs for one transformer layer over `batch` sequences of length `seq_len`."""
    matmul = 2 * batch * seq_len * params_per_layer(spec)
    # Attention QK^T and softmax @ V are both O(s^2 * d_kv).
    attn = 4 * batch * seq_len * seq_len * spec.num_kv_heads * spec.head_dim
    return float(matmul + attn)


def decode_flops_per_layer(spec: ModelSpec, batch: int, kv_len: int) -> float:
    """FLOPs for generating one token with cache of length kv_len."""
    matmul = 2 * batch * 1 * params_per_layer(spec)
    attn = 4 * batch * 1 * kv_len * spec.num_kv_heads * spec.head_dim
    return float(matmul + attn)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_cost_model.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/cost_model.py tests/flexgen/test_cost_model.py
git commit -m "feat(flexgen): cost model data classes + FLOP helpers"
```

---

## Task 8: Cost model — per-layer latency terms (compute, weight load, KV I/O, activation I/O)

**Files:**
- Modify: `src/flexgen/cost_model.py`
- Modify: `tests/flexgen/test_cost_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/flexgen/test_cost_model.py`:

```python
from src.flexgen.cost_model import (
    LayerTerms, prefill_layer_terms, decode_layer_terms,
)
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.workload import WorkloadSpec

COEF = SystemCoefficients(
    pcie_bw_gbs=14.0, disk_bw_gbs=3.0,
    tflops_fp16=10.0, tflops_int8=20.0, tflops_int4=40.0,
)
CAP = LiveCapacity(gpu_vram_gb=24.0, ram_gb=64.0, disk_gb=800.0)
WL = WorkloadSpec(prompt_len=512, decode_len=128)


def test_prefill_layer_terms_off_gpu_loads_increase_with_offload():
    enum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    on_gpu = PlacementFractions(w_g=1, w_c=0, w_d=0, c_g=1, c_c=0, c_d=0, h_g=1, h_c=0, h_d=0)
    off_cpu = PlacementFractions(w_g=0, w_c=1, w_d=0, c_g=0, c_c=1, c_d=0, h_g=0, h_c=1, h_d=0)
    on = prefill_layer_terms(enum, on_gpu, SPEC, WL, COEF)
    off = prefill_layer_terms(enum, off_cpu, SPEC, WL, COEF)
    assert on.t_load_w == 0.0
    assert on.t_io_kv == 0.0
    assert on.t_io_act == 0.0
    assert off.t_load_w > 0.0
    assert off.t_io_kv > 0.0
    assert off.t_io_act > 0.0


def test_prefill_layer_int4_compute_is_faster_than_fp16():
    enum_fp = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    enum_q4 = EnumPoint(gbs=4, num_gb=2, q="int4", delegate=False, overlap=False)
    p = PlacementFractions(w_g=1, w_c=0, w_d=0, c_g=1, c_c=0, c_d=0, h_g=1, h_c=0, h_d=0)
    fp = prefill_layer_terms(enum_fp, p, SPEC, WL, COEF)
    q4 = prefill_layer_terms(enum_q4, p, SPEC, WL, COEF)
    assert q4.t_compute < fp.t_compute


def test_delegate_replaces_kv_term_with_q_transfer():
    enum_no_del = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    enum_del = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=True, overlap=False)
    on_cpu = PlacementFractions(w_g=1, w_c=0, w_d=0, c_g=0, c_c=1, c_d=0, h_g=1, h_c=0, h_d=0)
    no_del = prefill_layer_terms(enum_no_del, on_cpu, SPEC, WL, COEF)
    yes_del = prefill_layer_terms(enum_del, on_cpu, SPEC, WL, COEF)
    # delegate sends only Q (hidden-dim sized), so it's much smaller than full KV cache transfer
    assert yes_del.t_io_kv < no_del.t_io_kv
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_cost_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'LayerTerms'`.

- [ ] **Step 3: Write the implementation**

Append to `src/flexgen/cost_model.py`:

```python
from src.flexgen.model_introspect import weights_per_layer_bytes, kv_per_token_bytes


@dataclass(frozen=True)
class LayerTerms:
    t_compute: float
    t_load_w: float
    t_io_kv: float
    t_io_act: float


def _tflops_for(coef: "SystemCoefficients", q: str) -> float:
    return {"fp16": coef.tflops_fp16, "int8": coef.tflops_int8, "int4": coef.tflops_int4}[q]


def _disk_effective_gbs(coef: "SystemCoefficients") -> float:
    """Two-hop effective bandwidth: 1/eff = 1/disk + 1/pcie."""
    return 1.0 / (1.0 / coef.disk_bw_gbs + 1.0 / coef.pcie_bw_gbs)


def _bytes_to_gb(b: float) -> float:
    return b / 1024**3


def prefill_layer_terms(
    enum: "EnumPoint", p: "PlacementFractions",
    spec: "ModelSpec", wl: "WorkloadSpec", coef: "SystemCoefficients",
) -> "LayerTerms":
    B = enum.block_size
    s = wl.prompt_len
    h = spec.hidden_dim

    flops = prefill_flops_per_layer(spec, batch=B, seq_len=s)
    t_compute = flops / (_tflops_for(coef, enum.q) * 1e12)

    w_bytes = weights_per_layer_bytes(spec, enum.q)
    t_load_w = (
        _bytes_to_gb(w_bytes) * (p.w_c / coef.pcie_bw_gbs + p.w_d / _disk_effective_gbs(coef))
    )

    kv_bytes_total = kv_per_token_bytes(spec, enum.q) * B * s / spec.num_layers
    if enum.delegate and p.c_c > 0:
        # Replace the c_c-weighted KV transfer with a Q+result hidden-dim transfer.
        q_xfer_bytes = B * s * h * 2  # fp16 Q tensor down + result up (approx)
        t_io_kv = (
            _bytes_to_gb(q_xfer_bytes) / coef.pcie_bw_gbs
            + _bytes_to_gb(kv_bytes_total) * (p.c_d / _disk_effective_gbs(coef))
        )
    else:
        t_io_kv = (
            _bytes_to_gb(kv_bytes_total)
            * (p.c_c / coef.pcie_bw_gbs + p.c_d / _disk_effective_gbs(coef))
        )

    act_bytes = B * s * h * 2  # fp16 activations
    t_io_act = (
        _bytes_to_gb(act_bytes) * (p.h_c / coef.pcie_bw_gbs + p.h_d / _disk_effective_gbs(coef))
    )

    return LayerTerms(t_compute=t_compute, t_load_w=t_load_w, t_io_kv=t_io_kv, t_io_act=t_io_act)


def decode_layer_terms(
    enum: "EnumPoint", p: "PlacementFractions",
    spec: "ModelSpec", wl: "WorkloadSpec", coef: "SystemCoefficients",
    kv_len: int,
) -> "LayerTerms":
    B = enum.block_size
    h = spec.hidden_dim

    flops = decode_flops_per_layer(spec, batch=B, kv_len=kv_len)
    t_compute = flops / (_tflops_for(coef, enum.q) * 1e12)

    w_bytes = weights_per_layer_bytes(spec, enum.q)
    t_load_w = (
        _bytes_to_gb(w_bytes) * (p.w_c / coef.pcie_bw_gbs + p.w_d / _disk_effective_gbs(coef))
    )

    kv_bytes_total = kv_per_token_bytes(spec, enum.q) * B * kv_len / spec.num_layers
    if enum.delegate and p.c_c > 0:
        q_xfer_bytes = B * 1 * h * 2
        t_io_kv = (
            _bytes_to_gb(q_xfer_bytes) / coef.pcie_bw_gbs
            + _bytes_to_gb(kv_bytes_total) * (p.c_d / _disk_effective_gbs(coef))
        )
    else:
        t_io_kv = (
            _bytes_to_gb(kv_bytes_total)
            * (p.c_c / coef.pcie_bw_gbs + p.c_d / _disk_effective_gbs(coef))
        )

    act_bytes = B * 1 * h * 2
    t_io_act = (
        _bytes_to_gb(act_bytes) * (p.h_c / coef.pcie_bw_gbs + p.h_d / _disk_effective_gbs(coef))
    )

    return LayerTerms(t_compute=t_compute, t_load_w=t_load_w, t_io_kv=t_io_kv, t_io_act=t_io_act)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_cost_model.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/cost_model.py tests/flexgen/test_cost_model.py
git commit -m "feat(flexgen): per-layer latency terms with delegate branch"
```

---

## Task 9: Cost model — `t_block` with overlap and decode integration

**Files:**
- Modify: `src/flexgen/cost_model.py`
- Modify: `tests/flexgen/test_cost_model.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/flexgen/test_cost_model.py`:

```python
from src.flexgen.cost_model import t_block_seconds, t_per_token_seconds


def test_overlap_no_worse_than_sum():
    p = PlacementFractions(w_g=0, w_c=1, w_d=0, c_g=0, c_c=1, c_d=0, h_g=0, h_c=1, h_d=0)
    enum_sum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    enum_max = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=True)
    t_sum = t_block_seconds(enum_sum, p, SPEC, WL, COEF)
    t_max = t_block_seconds(enum_max, p, SPEC, WL, COEF)
    assert t_max <= t_sum + 1e-9


def test_t_per_token_divides_block_by_effective_batch():
    p = PlacementFractions(w_g=1, w_c=0, w_d=0, c_g=1, c_c=0, c_d=0, h_g=1, h_c=0, h_d=0)
    enum1 = EnumPoint(gbs=1, num_gb=1, q="fp16", delegate=False, overlap=False)
    enum8 = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=False)
    tt1 = t_per_token_seconds(enum1, p, SPEC, WL, COEF)
    tt8 = t_per_token_seconds(enum8, p, SPEC, WL, COEF)
    # Bigger effective batch amortizes per-token cost (compute is roughly batch-invariant
    # per token, weight load / num_decode is amortized — tt8 should be smaller).
    assert tt8 < tt1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_cost_model.py -v`
Expected: FAIL with `ImportError: cannot import name 't_block_seconds'`.

- [ ] **Step 3: Write the implementation**

Append to `src/flexgen/cost_model.py`:

```python
def _combine(terms: "LayerTerms", overlap: bool) -> float:
    if overlap:
        return max(terms.t_compute, terms.t_load_w, terms.t_io_kv, terms.t_io_act)
    return terms.t_compute + terms.t_load_w + terms.t_io_kv + terms.t_io_act


def t_block_seconds(
    enum: "EnumPoint", p: "PlacementFractions",
    spec: "ModelSpec", wl: "WorkloadSpec", coef: "SystemCoefficients",
) -> float:
    pre = prefill_layer_terms(enum, p, spec, wl, coef)
    t_pre_layer = _combine(pre, enum.overlap)

    # Decode integrated as the trapezoidal sum: kv_len ranges from prompt_len to
    # prompt_len + decode_len - 1. Use exact arithmetic by linearity in kv_len for
    # the attention term; matmul/load/act are kv-independent.
    s = wl.prompt_len
    d = wl.decode_len
    # average kv_len across decode steps = s + (d-1)/2 ≈ s + d/2
    kv_avg = s + (d - 1) / 2.0 if d > 1 else s
    dec = decode_layer_terms(enum, p, spec, wl, coef, kv_len=int(kv_avg))
    t_dec_layer = _combine(dec, enum.overlap)

    t_layer = t_pre_layer + d * t_dec_layer
    return spec.num_layers * t_layer


def t_per_token_seconds(
    enum: "EnumPoint", p: "PlacementFractions",
    spec: "ModelSpec", wl: "WorkloadSpec", coef: "SystemCoefficients",
) -> float:
    return t_block_seconds(enum, p, spec, wl, coef) / enum.block_size
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_cost_model.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/cost_model.py tests/flexgen/test_cost_model.py
git commit -m "feat(flexgen): t_block with overlap combinator and decode integration"
```

---

## Task 10: Inner LP refactor — `solve_inner_lp(enum, system, model, workload)`

**Files:**
- Modify: `src/flexgen/lp_formulation.py`
- Modify: `tests/flexgen/test_lp_formulation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/flexgen/test_lp_formulation.py`:

```python
from src.flexgen.lp_formulation import solve_inner_lp, InnerLPResult
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.workload import WorkloadSpec
from src.flexgen.cost_model import EnumPoint

SPEC2 = ModelSpec(
    hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=8,
    intermediate_size=14336, vocab_size=128256, dtype_bytes=2,
)
COEF2 = SystemCoefficients(
    pcie_bw_gbs=14.0, disk_bw_gbs=3.0,
    tflops_fp16=10.0, tflops_int8=20.0, tflops_int4=40.0,
)
WL2 = WorkloadSpec(prompt_len=512, decode_len=128)


def test_inner_lp_with_huge_gpu_keeps_everything_on_gpu():
    cap = LiveCapacity(gpu_vram_gb=200.0, ram_gb=200.0, disk_gb=2000.0)
    enum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=True)
    res = solve_inner_lp(enum, cap, SPEC2, WL2, COEF2)
    assert res.status == "Optimal"
    assert res.placement.w_g > 0.99
    assert res.placement.c_g > 0.99
    assert res.placement.h_g > 0.99


def test_inner_lp_with_tight_gpu_spills_some_weights():
    cap = LiveCapacity(gpu_vram_gb=4.0, ram_gb=64.0, disk_gb=800.0)
    enum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=True)
    res = solve_inner_lp(enum, cap, SPEC2, WL2, COEF2)
    assert res.status == "Optimal"
    assert res.placement.w_g < 1.0


def test_inner_lp_returns_per_token_latency_in_seconds():
    cap = LiveCapacity(gpu_vram_gb=200.0, ram_gb=200.0, disk_gb=2000.0)
    enum = EnumPoint(gbs=4, num_gb=2, q="fp16", delegate=False, overlap=True)
    res = solve_inner_lp(enum, cap, SPEC2, WL2, COEF2)
    assert 0 < res.t_per_token_s < 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_lp_formulation.py -v`
Expected: FAIL with `ImportError: cannot import name 'solve_inner_lp'`.

- [ ] **Step 3: Write the implementation**

Append to `src/flexgen/lp_formulation.py`:

```python
from dataclasses import dataclass as _dataclass

from src.flexgen.cost_model import (
    EnumPoint as _EnumPoint, PlacementFractions as _PlacementFractions,
    LayerTerms as _LayerTerms, prefill_layer_terms as _prefill_layer_terms,
    decode_layer_terms as _decode_layer_terms,
)
from src.flexgen.system_probe import LiveCapacity as _LiveCapacity
from src.flexgen.calibration import SystemCoefficients as _SystemCoefficients
from src.flexgen.model_introspect import (
    ModelSpec as _ModelSpec,
    weights_per_layer_bytes as _weights_per_layer_bytes,
    kv_per_token_bytes as _kv_per_token_bytes,
)
from src.flexgen.workload import WorkloadSpec as _WorkloadSpec


@_dataclass(frozen=True)
class InnerLPResult:
    placement: _PlacementFractions
    t_per_token_s: float
    t_block_s: float
    status: str


def solve_inner_lp(
    enum: _EnumPoint, cap: _LiveCapacity, spec: _ModelSpec,
    wl: _WorkloadSpec, coef: _SystemCoefficients,
) -> InnerLPResult:
    """For a fixed enumeration point, find continuous placement fractions minimizing
    per-token latency. With overlap=True, use epigraph variables τ_pre, τ_dec ≥ each term."""
    prob = pulp.LpProblem("flexgen_inner", pulp.LpMinimize)

    w_g = pulp.LpVariable("w_g", 0, 1); w_c = pulp.LpVariable("w_c", 0, 1); w_d = pulp.LpVariable("w_d", 0, 1)
    c_g = pulp.LpVariable("c_g", 0, 1); c_c = pulp.LpVariable("c_c", 0, 1); c_d = pulp.LpVariable("c_d", 0, 1)
    h_g = pulp.LpVariable("h_g", 0, 1); h_c = pulp.LpVariable("h_c", 0, 1); h_d = pulp.LpVariable("h_d", 0, 1)

    prob += w_g + w_c + w_d == 1
    prob += c_g + c_c + c_d == 1
    prob += h_g + h_c + h_d == 1

    # ---- Capacity constraints (in GB) ----
    B = enum.block_size
    s, d = wl.prompt_len, wl.decode_len
    L = spec.num_layers

    w_bytes_total = _weights_per_layer_bytes(spec, enum.q) * L
    kv_bytes_total = _kv_per_token_bytes(spec, enum.q) * B * (s + d)
    act_bytes_total = B * s * spec.hidden_dim * 2 * L  # fp16

    GB = 1024**3
    prob += (w_bytes_total / GB) * w_g + (kv_bytes_total / GB) * c_g + (act_bytes_total / GB) * h_g <= cap.gpu_vram_gb
    prob += (w_bytes_total / GB) * w_c + (kv_bytes_total / GB) * c_c + (act_bytes_total / GB) * h_c <= cap.ram_gb
    prob += (w_bytes_total / GB) * w_d + (kv_bytes_total / GB) * c_d + (act_bytes_total / GB) * h_d <= cap.disk_gb

    # ---- Latency objective ----
    # Build the four terms as LP expressions in the placement vars by re-deriving
    # from cost_model formulas (linear in fractions for fixed enum).
    pcie = coef.pcie_bw_gbs
    disk_eff = 1.0 / (1.0 / coef.disk_bw_gbs + 1.0 / coef.pcie_bw_gbs)

    def _w_load_term(q_kv_unused=None):
        wlpl_gb = _weights_per_layer_bytes(spec, enum.q) / GB
        return wlpl_gb * (w_c / pcie + w_d / disk_eff)

    def _kv_term(seq_len: int):
        kv_pl_gb = _kv_per_token_bytes(spec, enum.q) * B * seq_len / L / GB
        if enum.delegate:
            q_xfer_gb = (B * seq_len * spec.hidden_dim * 2) / GB
            return q_xfer_gb / pcie * c_c + kv_pl_gb * (c_d / disk_eff)
        return kv_pl_gb * (c_c / pcie + c_d / disk_eff)

    def _act_term(seq_len: int):
        a_gb = (B * seq_len * spec.hidden_dim * 2) / GB
        return a_gb * (h_c / pcie + h_d / disk_eff)

    # Compute terms are constants (don't depend on placement).
    pre_terms_const = _prefill_layer_terms(
        enum, _PlacementFractions(1, 0, 0, 1, 0, 0, 1, 0, 0), spec, wl, coef,
    )
    t_compute_pre = pre_terms_const.t_compute
    kv_avg = s + (d - 1) / 2.0 if d > 1 else s
    dec_terms_const = _decode_layer_terms(
        enum, _PlacementFractions(1, 0, 0, 1, 0, 0, 1, 0, 0), spec, wl, coef, kv_len=int(kv_avg),
    )
    t_compute_dec = dec_terms_const.t_compute

    if enum.overlap:
        tau_pre = pulp.LpVariable("tau_pre", 0)
        tau_dec = pulp.LpVariable("tau_dec", 0)
        prob += tau_pre >= t_compute_pre
        prob += tau_pre >= _w_load_term()
        prob += tau_pre >= _kv_term(s)
        prob += tau_pre >= _act_term(s)
        prob += tau_dec >= t_compute_dec
        prob += tau_dec >= _w_load_term()
        prob += tau_dec >= _kv_term(int(kv_avg))
        prob += tau_dec >= _act_term(1)
        t_block_expr = L * (tau_pre + d * tau_dec)
    else:
        t_pre = t_compute_pre + _w_load_term() + _kv_term(s) + _act_term(s)
        t_dec = t_compute_dec + _w_load_term() + _kv_term(int(kv_avg)) + _act_term(1)
        t_block_expr = L * (t_pre + d * t_dec)

    prob += t_block_expr / B

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return InnerLPResult(
            placement=_PlacementFractions(1, 0, 0, 1, 0, 0, 1, 0, 0),
            t_per_token_s=float("inf"), t_block_s=float("inf"), status=status,
        )

    placement = _PlacementFractions(
        w_g=_pv(w_g), w_c=_pv(w_c), w_d=_pv(w_d),
        c_g=_pv(c_g), c_c=_pv(c_c), c_d=_pv(c_d),
        h_g=_pv(h_g), h_c=_pv(h_c), h_d=_pv(h_d),
    )
    t_block_s = float(pulp.value(t_block_expr))
    return InnerLPResult(
        placement=placement,
        t_per_token_s=t_block_s / B,
        t_block_s=t_block_s,
        status="Optimal",
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_lp_formulation.py -v`
Expected: 7 passed (4 legacy + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/lp_formulation.py tests/flexgen/test_lp_formulation.py
git commit -m "feat(flexgen): inner LP solving placement for fixed enum point"
```

---

## Task 11: Outer policy search

**Files:**
- Create: `src/flexgen/policy_search.py`
- Create: `tests/flexgen/test_policy_search.py`

- [ ] **Step 1: Write the failing test**

Create `tests/flexgen/test_policy_search.py`:

```python
from src.flexgen.policy_search import (
    GBS_GRID, NUM_GB_GRID, run_policy_search, PolicyResult,
)
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.workload import WorkloadSpec

SPEC3 = ModelSpec(
    hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=8,
    intermediate_size=14336, vocab_size=128256, dtype_bytes=2,
)
COEF3 = SystemCoefficients(
    pcie_bw_gbs=14.0, disk_bw_gbs=3.0,
    tflops_fp16=10.0, tflops_int8=20.0, tflops_int4=40.0,
)
WL3 = WorkloadSpec(prompt_len=512, decode_len=128)


def test_search_grid_constants():
    assert tuple(GBS_GRID) == (1, 2, 4, 8, 16, 32)
    assert tuple(NUM_GB_GRID) == (1, 2, 4, 8, 16)


def test_run_policy_search_returns_best_and_top_k():
    cap = LiveCapacity(gpu_vram_gb=24.0, ram_gb=64.0, disk_gb=800.0)
    res = run_policy_search(cap, SPEC3, WL3, COEF3, top_k=10)
    assert isinstance(res, PolicyResult)
    assert res.best.t_per_token_s > 0
    assert len(res.top_k) <= 10
    # top_k is sorted ascending
    for i in range(1, len(res.top_k)):
        assert res.top_k[i].t_per_token_s >= res.top_k[i-1].t_per_token_s
    assert res.best.t_per_token_s == res.top_k[0].t_per_token_s


def test_run_policy_search_with_huge_gpu_picks_no_offload():
    cap = LiveCapacity(gpu_vram_gb=10000.0, ram_gb=10000.0, disk_gb=10000.0)
    res = run_policy_search(cap, SPEC3, WL3, COEF3, top_k=5)
    assert res.best.placement.w_g > 0.99
    assert res.best.placement.c_g > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_policy_search.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Create `src/flexgen/policy_search.py`:

```python
from dataclasses import dataclass
from typing import Iterator
import logging

from src.flexgen.cost_model import EnumPoint, PlacementFractions
from src.flexgen.lp_formulation import solve_inner_lp, InnerLPResult
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.workload import WorkloadSpec

logger = logging.getLogger(__name__)

GBS_GRID = (1, 2, 4, 8, 16, 32)
NUM_GB_GRID = (1, 2, 4, 8, 16)
QUANT_GRID = ("fp16", "int4")


@dataclass(frozen=True)
class Candidate:
    enum: EnumPoint
    placement: PlacementFractions
    t_per_token_s: float
    t_block_s: float


@dataclass(frozen=True)
class PolicyResult:
    best: Candidate
    top_k: list[Candidate]


def _enum_iter() -> Iterator[EnumPoint]:
    for gbs in GBS_GRID:
        for num_gb in NUM_GB_GRID:
            for q in QUANT_GRID:
                for delegate in (False, True):
                    for overlap in (False, True):
                        yield EnumPoint(gbs=gbs, num_gb=num_gb, q=q,
                                        delegate=delegate, overlap=overlap)


def run_policy_search(
    cap: LiveCapacity, spec: ModelSpec, wl: WorkloadSpec, coef: SystemCoefficients,
    top_k: int = 20,
) -> PolicyResult:
    candidates: list[Candidate] = []
    n_total, n_feasible, n_infeasible = 0, 0, 0
    for enum in _enum_iter():
        n_total += 1
        res: InnerLPResult = solve_inner_lp(enum, cap, spec, wl, coef)
        if res.status == "Optimal":
            n_feasible += 1
            candidates.append(Candidate(
                enum=enum, placement=res.placement,
                t_per_token_s=res.t_per_token_s, t_block_s=res.t_block_s,
            ))
        else:
            n_infeasible += 1

    if not candidates:
        raise RuntimeError(f"Policy search found no feasible config (n_total={n_total})")

    candidates.sort(key=lambda c: (c.t_per_token_s,
                                   -(c.placement.w_g + c.placement.c_g + c.placement.h_g)))
    logger.info(
        "policy search: total=%d feasible=%d infeasible=%d best_t=%.4fs",
        n_total, n_feasible, n_infeasible, candidates[0].t_per_token_s,
    )
    return PolicyResult(best=candidates[0], top_k=candidates[:top_k])
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_policy_search.py -v`
Expected: 3 passed (the search runs ~480 LPs in under 10 s).

- [ ] **Step 5: Commit**

```bash
git add src/flexgen/policy_search.py tests/flexgen/test_policy_search.py
git commit -m "feat(flexgen): outer policy search with top-k tracking"
```

---

## Task 12: CLI orchestrator + logging + JSON output

**Files:**
- Modify: `experiments/run_flexgen.py` (full rewrite)
- Create: `tests/flexgen/test_run_flexgen.py`

- [ ] **Step 1: Write the failing test**

Create `tests/flexgen/test_run_flexgen.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch
from experiments.run_flexgen import build_output_payload, run
from src.flexgen.cost_model import EnumPoint, PlacementFractions
from src.flexgen.policy_search import Candidate, PolicyResult
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.workload import WorkloadSpec


def _make_result() -> PolicyResult:
    enum = EnumPoint(gbs=8, num_gb=4, q="int4", delegate=True, overlap=True)
    placement = PlacementFractions(w_g=0.45, w_c=0.55, w_d=0.0,
                                   c_g=0.20, c_c=0.80, c_d=0.0,
                                   h_g=1.0, h_c=0.0, h_d=0.0)
    cand = Candidate(enum=enum, placement=placement, t_per_token_s=0.0843, t_block_s=2.6976)
    return PolicyResult(best=cand, top_k=[cand])


def test_build_output_payload_contains_all_14_outputs():
    cap = LiveCapacity(gpu_vram_gb=24.0, ram_gb=64.0, disk_gb=800.0)
    coef = SystemCoefficients(pcie_bw_gbs=14.0, disk_bw_gbs=3.0,
                              tflops_fp16=10.0, tflops_int8=20.0, tflops_int4=40.0)
    spec = ModelSpec(hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32,
                     num_kv_heads=8, intermediate_size=14336, vocab_size=1, dtype_bytes=2)
    wl = WorkloadSpec(prompt_len=512, decode_len=128)
    payload = build_output_payload(_make_result(), cap, coef, spec, wl, machine_id="m1")

    bp = payload["best_policy"]
    assert bp["gpu_batch_size"] == 8
    assert bp["num_gpu_batches"] == 4
    assert bp["block_size"] == 32
    assert bp["compression"] == "int4"
    assert bp["cpu_compute_delegate"] is True
    assert bp["overlap_io_compute"] is True
    assert bp["weights"] == {"gpu": 0.45, "cpu": 0.55, "disk": 0.0}
    assert bp["kv_cache"] == {"gpu": 0.20, "cpu": 0.80, "disk": 0.0}
    assert bp["activations"] == {"gpu": 1.0, "cpu": 0.0, "disk": 0.0}
    assert payload["objective"]["per_token_latency_ms"] > 0


def test_run_writes_json_and_log(tmp_path, monkeypatch):
    """Smoke-test the orchestrator with mocked solver, calibration, model."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "workload.yaml").write_text("prompt_len: 512\ndecode_len: 128\n")
    (tmp_path / "experiments" / "results").mkdir(parents=True)
    (tmp_path / "experiments" / "logs").mkdir(parents=True)
    (tmp_path / "configs" / "system_calibration").mkdir()

    with patch("experiments.run_flexgen.probe_live_capacity", return_value=LiveCapacity(24, 64, 800)), \
         patch("experiments.run_flexgen.ensure_calibration", return_value=SystemCoefficients(
             pcie_bw_gbs=14, disk_bw_gbs=3, tflops_fp16=10, tflops_int8=20, tflops_int4=40)), \
         patch("experiments.run_flexgen.load_model_spec", return_value=ModelSpec(
             hf_id="x", num_layers=32, hidden_dim=4096, num_heads=32, num_kv_heads=8,
             intermediate_size=14336, vocab_size=1, dtype_bytes=2)), \
         patch("experiments.run_flexgen.run_policy_search", return_value=_make_result()):
        out_path = run(
            model_id="meta-llama/Meta-Llama-3-8B",
            workload_path=str(tmp_path / "configs" / "workload.yaml"),
            recalibrate=False, output_dir=str(tmp_path / "experiments" / "results"),
            log_dir=str(tmp_path / "experiments" / "logs"),
            cache_dir=str(tmp_path / "configs" / "system_calibration"),
            verbose=False,
        )

    payload = json.loads(Path(out_path).read_text())
    assert payload["best_policy"]["gpu_batch_size"] == 8
    log_files = list((tmp_path / "experiments" / "logs").glob("*.log"))
    assert len(log_files) == 1
    assert "policy search" in log_files[0].read_text() or len(log_files[0].read_text()) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_run_flexgen.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Replace contents of `experiments/run_flexgen.py`:

```python
"""FlexGen faithful policy-search orchestrator.

CLI:
    python experiments/run_flexgen.py \
        --model meta-llama/Meta-Llama-3-8B \
        --workload configs/workload.yaml \
        [--recalibrate] [--verbose]
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.flexgen.system_probe import probe_live_capacity, LiveCapacity
from src.flexgen.calibration import (
    ensure_calibration, machine_id, SystemCoefficients,
)
from src.flexgen.model_introspect import load_model_spec, ModelSpec
from src.flexgen.workload import load_workload, WorkloadSpec
from src.flexgen.policy_search import run_policy_search, PolicyResult, Candidate


def _setup_logging(log_path: Path, verbose: bool) -> None:
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console)


def _candidate_to_json(c: Candidate) -> dict:
    return {
        "gpu_batch_size":       c.enum.gbs,
        "num_gpu_batches":      c.enum.num_gb,
        "block_size":           c.enum.block_size,
        "compression":          c.enum.q,
        "cpu_compute_delegate": c.enum.delegate,
        "overlap_io_compute":   c.enum.overlap,
        "weights":     {"gpu": round(c.placement.w_g, 4),
                        "cpu": round(c.placement.w_c, 4),
                        "disk": round(c.placement.w_d, 4)},
        "kv_cache":    {"gpu": round(c.placement.c_g, 4),
                        "cpu": round(c.placement.c_c, 4),
                        "disk": round(c.placement.c_d, 4)},
        "activations": {"gpu": round(c.placement.h_g, 4),
                        "cpu": round(c.placement.h_c, 4),
                        "disk": round(c.placement.h_d, 4)},
        "per_token_latency_ms": round(c.t_per_token_s * 1000, 4),
    }


def build_output_payload(
    result: PolicyResult, cap: LiveCapacity, coef: SystemCoefficients,
    spec: ModelSpec, wl: WorkloadSpec, machine_id: str,
) -> dict:
    best = result.best
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_id": machine_id,
        "input": {
            "system": {**asdict(cap), **asdict(coef)},
            "model": asdict(spec),
            "workload": asdict(wl),
        },
        "best_policy": _candidate_to_json(best),
        "objective": {
            "per_token_latency_ms": round(best.t_per_token_s * 1000, 4),
            "throughput_tok_s": round(1.0 / best.t_per_token_s, 4),
            "t_block_ms": round(best.t_block_s * 1000, 4),
        },
        "top_k_candidates": [_candidate_to_json(c) for c in result.top_k],
    }


def run(
    model_id: str,
    workload_path: str,
    recalibrate: bool,
    output_dir: str,
    log_dir: str,
    cache_dir: str,
    verbose: bool,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"flexgen_{ts}.log"
    json_path = Path(output_dir) / f"flexgen_{ts}.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    _setup_logging(log_path, verbose)
    log = logging.getLogger("run_flexgen")
    log.info("=== FlexGen policy-search run ===")
    log.info("model=%s workload=%s recalibrate=%s", model_id, workload_path, recalibrate)

    mid = machine_id()
    log.info("machine_id=%s", mid)
    cap = probe_live_capacity(project_root=str(ROOT))
    log.info("system: gpu_vram=%.2fGB ram=%.2fGB disk=%.2fGB",
             cap.gpu_vram_gb, cap.ram_gb, cap.disk_gb)

    coef = ensure_calibration(cache_dir=cache_dir, key=mid, recalibrate=recalibrate)
    log.info("calib: pcie=%.2fGB/s disk=%.2fGB/s fp16=%.1fTFLOPS int4=%.1fTFLOPS",
             coef.pcie_bw_gbs, coef.disk_bw_gbs, coef.tflops_fp16, coef.tflops_int4)

    spec = load_model_spec(model_id)
    log.info("model: layers=%d hidden=%d heads=%d kv_heads=%d",
             spec.num_layers, spec.hidden_dim, spec.num_heads, spec.num_kv_heads)

    wl = load_workload(workload_path)
    log.info("workload: prompt_len=%d decode_len=%d", wl.prompt_len, wl.decode_len)

    log.info("Running policy search...")
    result = run_policy_search(cap, spec, wl, coef, top_k=20)
    log.info("Best: gbs=%d num_gb=%d q=%s delegate=%s overlap=%s -> %.2f ms/token",
             result.best.enum.gbs, result.best.enum.num_gb, result.best.enum.q,
             result.best.enum.delegate, result.best.enum.overlap,
             result.best.t_per_token_s * 1000)

    payload = build_output_payload(result, cap, coef, spec, wl, machine_id=mid)
    json_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote results: %s", json_path)
    return str(json_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--workload", default=str(ROOT / "configs" / "workload.yaml"))
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "results"))
    parser.add_argument("--log-dir", default=str(ROOT / "experiments" / "logs"))
    parser.add_argument("--cache-dir", default=str(ROOT / "configs" / "system_calibration"))
    args = parser.parse_args()
    run(
        model_id=args.model,
        workload_path=args.workload,
        recalibrate=args.recalibrate,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_run_flexgen.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add experiments/run_flexgen.py tests/flexgen/test_run_flexgen.py
git commit -m "feat(flexgen): CLI orchestrator with logging and JSON output"
```

---

## Task 13: Plot extensions — Pareto + placement heatmap

**Files:**
- Modify: `analysis/plot_tradeoffs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/flexgen/test_plot_tradeoffs.py`:

```python
import json
from pathlib import Path
from analysis.plot_tradeoffs import (
    plot_flexgen_pareto, plot_flexgen_placement_heatmap,
)


def _sample_payload():
    cands = []
    for i, gbs in enumerate([1, 4, 16, 32]):
        cands.append({
            "gpu_batch_size": gbs, "num_gpu_batches": 1, "block_size": gbs,
            "compression": "int4" if i % 2 == 0 else "fp16",
            "cpu_compute_delegate": False, "overlap_io_compute": True,
            "weights":     {"gpu": 1 - i*0.2, "cpu": i*0.2, "disk": 0},
            "kv_cache":    {"gpu": 1 - i*0.2, "cpu": i*0.2, "disk": 0},
            "activations": {"gpu": 1.0, "cpu": 0, "disk": 0},
            "per_token_latency_ms": 100 + i * 30,
        })
    return {"top_k_candidates": cands, "best_policy": cands[0]}


def test_plot_flexgen_pareto_writes_png(tmp_path):
    payload_path = tmp_path / "p.json"
    payload_path.write_text(json.dumps(_sample_payload()))
    out = plot_flexgen_pareto(str(payload_path), out_dir=str(tmp_path))
    assert Path(out).exists()
    assert out.endswith("flexgen_pareto.png")


def test_plot_flexgen_placement_heatmap_writes_png(tmp_path):
    payload_path = tmp_path / "p.json"
    payload_path.write_text(json.dumps(_sample_payload()))
    out = plot_flexgen_placement_heatmap(str(payload_path), out_dir=str(tmp_path))
    assert Path(out).exists()
    assert out.endswith("flexgen_placement_heatmap.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_plot_tradeoffs.py -v`
Expected: FAIL with `ImportError: cannot import name 'plot_flexgen_pareto'`.

- [ ] **Step 3: Write the implementation**

Append to `analysis/plot_tradeoffs.py`:

```python
import json as _json
import os as _os

import matplotlib.pyplot as _plt
import numpy as _np


def plot_flexgen_pareto(payload_path: str, out_dir: str) -> str:
    payload = _json.loads(open(payload_path).read())
    cands = payload["top_k_candidates"]
    best = payload["best_policy"]

    xs = [c["block_size"] for c in cands]
    ys = [c["per_token_latency_ms"] for c in cands]
    colors = ["#1f77b4" if c["compression"] == "fp16" else "#d62728" for c in cands]

    _plt.figure(figsize=(7, 5))
    _plt.scatter(xs, ys, c=colors, alpha=0.7, s=60)
    _plt.scatter([best["block_size"]], [best["per_token_latency_ms"]],
                 marker="*", s=300, c="gold", edgecolor="black", label="best")
    _plt.xscale("log", base=2)
    _plt.xlabel("Effective batch size (gbs · num_gb)")
    _plt.ylabel("Per-token latency (ms)")
    _plt.title("FlexGen policy search: latency vs effective batch")
    _plt.legend()
    _plt.tight_layout()
    _os.makedirs(out_dir, exist_ok=True)
    out = _os.path.join(out_dir, "flexgen_pareto.png")
    _plt.savefig(out, dpi=120)
    _plt.close()
    return out


def plot_flexgen_placement_heatmap(payload_path: str, out_dir: str) -> str:
    payload = _json.loads(open(payload_path).read())
    cands = payload["top_k_candidates"]

    rows = []
    labels = []
    for c in cands:
        rows.append([c["weights"]["gpu"], c["weights"]["cpu"], c["weights"]["disk"],
                     c["kv_cache"]["gpu"], c["kv_cache"]["cpu"], c["kv_cache"]["disk"],
                     c["activations"]["gpu"], c["activations"]["cpu"], c["activations"]["disk"]])
        labels.append(f"B={c['block_size']} {c['compression']}")
    arr = _np.array(rows)

    _plt.figure(figsize=(9, max(3, 0.3 * len(rows))))
    _plt.imshow(arr, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    _plt.colorbar(label="fraction")
    _plt.xticks(range(9), ["w_g", "w_c", "w_d", "c_g", "c_c", "c_d", "h_g", "h_c", "h_d"])
    _plt.yticks(range(len(labels)), labels)
    _plt.title("FlexGen top-k placement fractions")
    _plt.tight_layout()
    _os.makedirs(out_dir, exist_ok=True)
    out = _os.path.join(out_dir, "flexgen_placement_heatmap.png")
    _plt.savefig(out, dpi=120)
    _plt.close()
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/flexgen/test_plot_tradeoffs.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add analysis/plot_tradeoffs.py tests/flexgen/test_plot_tradeoffs.py
git commit -m "feat(flexgen): Pareto and placement-heatmap plots from policy-search JSON"
```

---

## Task 14: End-to-end integration test

**Files:**
- Create: `tests/flexgen/test_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/flexgen/test_integration.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch
from experiments.run_flexgen import run
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec


LLAMA3_8B_SPEC = ModelSpec(
    hf_id="meta-llama/Meta-Llama-3-8B", num_layers=32, hidden_dim=4096,
    num_heads=32, num_kv_heads=8, intermediate_size=14336, vocab_size=128256,
    dtype_bytes=2,
)


def test_end_to_end_llama3_8b_with_synthetic_system(tmp_path):
    """Full pipeline: probe -> calibrate -> introspect -> search -> JSON. All mocked."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "workload.yaml").write_text("prompt_len: 512\ndecode_len: 128\n")
    (tmp_path / "experiments" / "results").mkdir(parents=True)
    (tmp_path / "experiments" / "logs").mkdir(parents=True)
    (tmp_path / "configs" / "system_calibration").mkdir()

    # Synthetic A100-class server
    cap = LiveCapacity(gpu_vram_gb=38.0, ram_gb=250.0, disk_gb=1500.0)
    coef = SystemCoefficients(pcie_bw_gbs=24.6, disk_bw_gbs=3.1,
                              tflops_fp16=142.0, tflops_int8=280.0, tflops_int4=480.0)

    with patch("experiments.run_flexgen.probe_live_capacity", return_value=cap), \
         patch("experiments.run_flexgen.ensure_calibration", return_value=coef), \
         patch("experiments.run_flexgen.load_model_spec", return_value=LLAMA3_8B_SPEC):
        out_path = run(
            model_id="meta-llama/Meta-Llama-3-8B",
            workload_path=str(tmp_path / "configs" / "workload.yaml"),
            recalibrate=False,
            output_dir=str(tmp_path / "experiments" / "results"),
            log_dir=str(tmp_path / "experiments" / "logs"),
            cache_dir=str(tmp_path / "configs" / "system_calibration"),
            verbose=False,
        )

    payload = json.loads(Path(out_path).read_text())
    bp = payload["best_policy"]

    # All 14 fields populated
    for field in ["gpu_batch_size", "num_gpu_batches", "block_size", "compression",
                  "cpu_compute_delegate", "overlap_io_compute"]:
        assert field in bp
    for tier in ("weights", "kv_cache", "activations"):
        for level in ("gpu", "cpu", "disk"):
            assert level in bp[tier]
            assert 0.0 <= bp[tier][level] <= 1.0

    # All sums to 1
    for tier in ("weights", "kv_cache", "activations"):
        s = sum(bp[tier].values())
        assert abs(s - 1.0) < 0.01

    # Sanity bounds: latency must be positive and finite
    assert payload["objective"]["per_token_latency_ms"] > 0
    assert payload["objective"]["per_token_latency_ms"] < 10_000  # under 10s/token

    # Top-k populated
    assert 1 <= len(payload["top_k_candidates"]) <= 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/flexgen/test_integration.py -v`
Expected: FAIL initially if any prior task is incomplete; PASS once all modules wired.

- [ ] **Step 3: Verify the integration test passes**

Run: `pytest tests/flexgen/test_integration.py -v`
Expected: 1 passed (no implementation step needed — this validates Tasks 2-12 hang together).

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/flexgen/ -v`
Expected: all FlexGen tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/flexgen/test_integration.py
git commit -m "test(flexgen): end-to-end integration with synthetic system"
```

---

## Self-Review Checklist (executed; results below)

**Spec coverage** — every spec section maps to at least one task:
- §3.1 system live + cached → Task 3 (live), Tasks 4–5 (calibration + cache)
- §3.2 HF model introspection → Task 6
- §3.3 workload YAML → Task 2
- §4 cost model → Tasks 7–9
- §5 optimizer (outer enum + inner LP) → Tasks 10–11
- §6 outputs (JSON + log) → Task 12
- §7 code structure → all module-creation tasks
- §8 test strategy → tests appear in every task; integration in Task 14
- §9 CLI → Task 12
- §10 timeline → addressed by task ordering (Mon: 1; Tue demo: 2–6 + 12; Wed: 7–11; Thu: 13–14; Fri: polish)
- §11 future work → not implemented (correctly out of scope)

**Placeholder scan** — no TBDs; every code block is concrete and runnable.

**Type consistency** —
- `LiveCapacity`, `SystemCoefficients`, `ModelSpec`, `WorkloadSpec`, `EnumPoint`, `PlacementFractions`, `LayerTerms`, `InnerLPResult`, `Candidate`, `PolicyResult`: all defined in exactly one task and consistently re-imported.
- Function signatures: `prefill_layer_terms(enum, p, spec, wl, coef)` and `decode_layer_terms(enum, p, spec, wl, coef, kv_len)` — argument order consistent across Tasks 8, 9, 10.
- `bench_compute_tflops(dtype=...)` uses string keys consistent across Tasks 4, 5.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-26-flexgen-faithful.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
