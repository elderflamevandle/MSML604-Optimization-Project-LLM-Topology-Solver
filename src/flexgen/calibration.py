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
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = {"fp16": torch.float16, "int8": torch.float16, "int4": torch.float16}[dtype]
    else:
        device = "cpu"
        torch_dtype = torch.float32

    n = 4096 if device == "cuda" else 1024
    a = torch.randn(n, n, device=device, dtype=torch_dtype)
    b = torch.randn(n, n, device=device, dtype=torch_dtype)

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
        tflops *= 2.0
    elif dtype == "int4":
        tflops *= 4.0

    return tflops


def bench_pcie_bw_gbs(size_mb: int = 256, n_repeats: int = 3) -> float:
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
    bytes_moved = 2 * size_mb * 1024 * 1024
    return (bytes_moved / 1024**3) / median_t
