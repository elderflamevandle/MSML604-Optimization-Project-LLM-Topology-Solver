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
