import json
from unittest.mock import MagicMock, patch

from pipeline import (
    PipelineTestRun,
    extract_decision_variables_14,
    run_flexgen_tests,
    write_pipeline_summary,
)


def _payload():
    return {
        "best_policy": {
            "gpu_batch_size": 8,
            "num_gpu_batches": 4,
            "block_size": 32,
            "compression": "int4",
            "cpu_compute_delegate": True,
            "overlap_io_compute": True,
            "weights": {"gpu": 0.45, "cpu": 0.55, "disk": 0.0},
            "kv_cache": {"gpu": 0.2, "cpu": 0.8, "disk": 0.0},
            "activations": {"gpu": 1.0, "cpu": 0.0, "disk": 0.0},
        },
        "objective": {
            "per_token_latency_ms": 84.3,
            "throughput_tok_s": 11.86,
            "t_block_ms": 2697.6,
        },
        "input": {
            "system": {"gpu_vram_gb": 24.0, "ram_gb": 64.0},
            "model": {"hf_id": "local-qwen"},
            "workload": {"prompt_len": 512, "decode_len": 128},
        },
    }


def test_extract_decision_variables_returns_14_values():
    decisions = extract_decision_variables_14(_payload())
    assert len(decisions) == 14
    assert decisions["gpu_batch_size"] == 8
    assert decisions["activations_disk"] == 0.0
    assert "block_size" not in decisions


def test_run_flexgen_tests_uses_current_python():
    completed = MagicMock(returncode=0)
    with patch("pipeline.subprocess.run", return_value=completed) as mock_run:
        result = run_flexgen_tests("tests/flexgen", quiet=True)

    assert result.returncode == 0
    command = mock_run.call_args.args[0]
    assert command[1:4] == ["-m", "pytest", "tests/flexgen"]
    assert command[-1] == "-q"


def test_write_pipeline_summary(tmp_path):
    path = write_pipeline_summary(
        output_dir=str(tmp_path),
        tests=PipelineTestRun(command=["python", "-m", "pytest"], returncode=0),
        flexgen_result_path="experiments/results/flexgen_x.json",
        flexgen_payload=_payload(),
        baseline_comparison_path="experiments/baseline_comparisons/baseline_x.json",
        inference_result_path=None,
    )

    data = json.loads((tmp_path / path.split("/")[-1]).read_text())
    assert len(data["decision_variables_14"]) == 14
    assert data["derived"]["block_size"] == 32
    assert data["tests"]["returncode"] == 0
    assert data["baseline_comparison_path"].endswith("baseline_x.json")
