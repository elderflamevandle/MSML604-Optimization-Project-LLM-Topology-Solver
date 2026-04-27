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

    assert cap.gpu_vram_gb == 10.0
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
