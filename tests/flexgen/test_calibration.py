from src.flexgen.calibration import (
    SystemCoefficients,
    bench_compute_tflops,
    bench_pcie_bw_gbs,
    bench_disk_bw_gbs,
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
