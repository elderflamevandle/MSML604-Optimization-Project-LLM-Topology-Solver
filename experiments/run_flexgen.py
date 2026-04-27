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
