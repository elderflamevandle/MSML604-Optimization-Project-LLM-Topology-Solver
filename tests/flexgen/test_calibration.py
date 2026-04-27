import json
from pathlib import Path
from unittest.mock import patch
from src.flexgen.calibration import (
    SystemCoefficients,
    bench_compute_tflops,
    bench_pcie_bw_gbs,
    bench_disk_bw_gbs,
    machine_id,
    ensure_calibration,
)


def test_bench_compute_tflops_returns_positive_for_fp16():
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


@patch("src.flexgen.calibration.torch.cuda.is_available", return_value=True)
@patch("src.flexgen.calibration.torch.cuda.get_device_name", return_value="NVIDIA RTX 4050 Laptop")
@patch("src.flexgen.calibration.socket.gethostname", return_value="laptop-01")
def test_machine_id_combines_hostname_and_gpu(mock_host, mock_gpu, mock_avail):
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
