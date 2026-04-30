from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.flexgen.config_file import (
    load_flexgen_config,
    override,
    require_section,
    require_value,
    resolve_repo_path,
)

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineTestRun:
    command: list[str]
    returncode: int
    skipped: bool = False


def run_flexgen_tests(test_target: str, quiet: bool) -> PipelineTestRun:
    command = [sys.executable, "-m", "pytest", test_target]
    if quiet:
        command.append("-q")

    completed = subprocess.run(command, cwd=str(ROOT))
    return PipelineTestRun(command=command, returncode=completed.returncode)


def extract_decision_variables_14(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload["best_policy"]
    return {
        "gpu_batch_size": best["gpu_batch_size"],
        "num_gpu_batches": best["num_gpu_batches"],
        "compression": best["compression"],
        "cpu_compute_delegate": best["cpu_compute_delegate"],
        "overlap_io_compute": best["overlap_io_compute"],
        "weights_gpu": best["weights"]["gpu"],
        "weights_cpu": best["weights"]["cpu"],
        "weights_disk": best["weights"]["disk"],
        "kv_cache_gpu": best["kv_cache"]["gpu"],
        "kv_cache_cpu": best["kv_cache"]["cpu"],
        "kv_cache_disk": best["kv_cache"]["disk"],
        "activations_gpu": best["activations"]["gpu"],
        "activations_cpu": best["activations"]["cpu"],
        "activations_disk": best["activations"]["disk"],
    }


def print_policy_summary(payload: dict[str, Any]) -> None:
    decisions = extract_decision_variables_14(payload)
    best = payload["best_policy"]
    objective = payload["objective"]
    system = payload["input"]["system"]

    print("\n=== FlexGen 14 policy parameters ===")
    for idx, (name, value) in enumerate(decisions.items(), start=1):
        print(f"{idx:02d}. {name}: {value}")

    print("\n=== Derived value ===")
    print(f"block_size: {best['block_size']}")

    print("\n=== Objective ===")
    print(f"per_token_latency_ms: {objective['per_token_latency_ms']}")
    print(f"throughput_tok_s: {objective['throughput_tok_s']}")
    print(f"t_block_ms: {objective['t_block_ms']}")

    print("\n=== Measured/probed system inputs ===")
    for name, value in system.items():
        print(f"{name}: {value}")


def write_pipeline_summary(
    output_dir: str,
    tests: PipelineTestRun,
    flexgen_result_path: str,
    flexgen_payload: dict[str, Any],
    inference_result_path: str | None,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"flexgen_pipeline_{ts}.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": asdict(tests),
        "flexgen_result_path": flexgen_result_path,
        "inference_result_path": inference_result_path,
        "decision_variables_14": extract_decision_variables_14(flexgen_payload),
        "derived": {
            "block_size": flexgen_payload["best_policy"]["block_size"],
        },
        "objective": flexgen_payload["objective"],
        "system": flexgen_payload["input"]["system"],
        "model": flexgen_payload["input"]["model"],
        "workload": flexgen_payload["input"]["workload"],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(path)


def run_optional_inference(args: argparse.Namespace) -> str | None:
    if not args.run_inference:
        return None

    from src.flexgen.qwen_inference import (
        InferenceConfig,
        run_qwen_inference,
        write_inference_result,
    )

    result = run_qwen_inference(InferenceConfig(
        model=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
    ))
    path = write_inference_result(result, args.output_dir)

    print("\n=== Actual Qwen inference ===")
    print(f"latency_s: {result.latency_s:.3f}")
    print(f"tokens_per_s: {result.tokens_per_s:.2f}")
    print(f"generated_tokens: {result.generated_tokens}")
    print(f"inference_result_path: {path}")
    print("\nGenerated text:\n")
    print(result.generated_text)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command FlexGen pipeline: tests, policy search, 14 parameters, optional Qwen inference."
    )
    parser.add_argument("--config", default="config_flexgen.yml",
                        help="YAML config file containing model path, test settings, and run parameters.")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model id or local Qwen folder.")
    parser.add_argument("--workload", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--test-target", default=None)
    parser.add_argument("--skip-tests", action="store_true", default=None)
    parser.add_argument("--test-verbose", action="store_true", default=None)
    parser.add_argument("--recalibrate", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_true", default=None)

    parser.add_argument("--run-inference", action="store_true", default=None,
                        help="Also load Qwen and generate text after policy search.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default=None,
                        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config, ROOT)
    config = load_flexgen_config(config_path)
    paths_cfg = require_section(config, "paths")
    tests_cfg = require_section(config, "tests")
    flexgen_cfg = require_section(config, "flexgen")
    inference_cfg = require_section(config, "inference")

    args.model = override(require_value(paths_cfg, "model", "paths"), args.model)
    args.workload = resolve_repo_path(
        override(require_value(paths_cfg, "workload", "paths"), args.workload),
        ROOT,
    )
    args.output_dir = resolve_repo_path(
        override(require_value(paths_cfg, "output_dir", "paths"), args.output_dir),
        ROOT,
    )
    args.log_dir = resolve_repo_path(
        override(require_value(paths_cfg, "log_dir", "paths"), args.log_dir),
        ROOT,
    )
    args.cache_dir = resolve_repo_path(
        override(require_value(paths_cfg, "cache_dir", "paths"), args.cache_dir),
        ROOT,
    )
    args.test_target = override(require_value(tests_cfg, "target", "tests"), args.test_target)
    args.skip_tests = override(bool(tests_cfg.get("skip", False)), args.skip_tests)
    test_quiet = bool(tests_cfg.get("quiet", True))
    if args.test_verbose:
        test_quiet = False
    args.recalibrate = override(bool(flexgen_cfg.get("recalibrate", False)), args.recalibrate)
    args.verbose = override(bool(flexgen_cfg.get("verbose", False)), args.verbose)
    args.run_inference = override(bool(inference_cfg.get("enabled", False)), args.run_inference)
    args.prompt = override(require_value(inference_cfg, "prompt", "inference"), args.prompt)
    args.max_new_tokens = int(override(
        require_value(inference_cfg, "max_new_tokens", "inference"),
        args.max_new_tokens,
    ))
    args.device = override(require_value(inference_cfg, "device", "inference"), args.device)
    args.device_map = override(inference_cfg.get("device_map"), args.device_map)
    args.dtype = override(require_value(inference_cfg, "dtype", "inference"), args.dtype)

    if args.skip_tests:
        tests = PipelineTestRun(command=[], returncode=0, skipped=True)
        print("Skipping FlexGen tests.")
    else:
        print(f"Running FlexGen tests: {args.test_target}")
        tests = run_flexgen_tests(args.test_target, quiet=test_quiet)
        if tests.returncode != 0:
            print(f"\nFlexGen tests failed with return code {tests.returncode}. Stopping.")
            sys.exit(tests.returncode)

    print("\nRunning FlexGen policy search...")
    from experiments.run_flexgen import run as run_flexgen

    flexgen_result_path = run_flexgen(
        model_id=args.model,
        workload_path=args.workload,
        recalibrate=args.recalibrate,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
    )

    flexgen_payload = json.loads(Path(flexgen_result_path).read_text(encoding="utf-8"))
    print_policy_summary(flexgen_payload)

    inference_result_path = run_optional_inference(args)

    summary_path = write_pipeline_summary(
        output_dir=args.output_dir,
        tests=tests,
        flexgen_result_path=flexgen_result_path,
        flexgen_payload=flexgen_payload,
        inference_result_path=inference_result_path,
    )

    print("\n=== Pipeline files ===")
    print(f"flexgen_result_path: {flexgen_result_path}")
    print(f"pipeline_summary_path: {summary_path}")


if __name__ == "__main__":
    main()
