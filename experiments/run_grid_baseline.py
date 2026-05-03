"""
Grid search baseline runner — mirrors run_flexgen.py structure.

Outputs a JSON in the same schema as run_flexgen.py plus extra keys:
  "method": "grid_search"
  "search_stats": { n_total, n_feasible, n_infeasible, elapsed_s, inner_grid_size, step }

CLI:
    python experiments/run_grid_baseline.py --model models/smollm2-135m-instruct
    python experiments/run_grid_baseline.py --model Qwen/Qwen2-1.5B --verbose
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
from src.flexgen.calibration import ensure_calibration, machine_id, SystemCoefficients
from src.flexgen.model_introspect import load_model_spec, ModelSpec
from src.flexgen.workload import load_workload, WorkloadSpec
from src.flexgen.policy_search import PolicyResult, Candidate
from src.flexgen.grid_search_baseline import run_grid_search, GridSearchResult


def _setup_logging(log_path: Path, verbose: bool) -> None:
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)


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
    gs: GridSearchResult,
    cap: LiveCapacity,
    coef: SystemCoefficients,
    spec: ModelSpec,
    wl: WorkloadSpec,
    mid: str,
) -> dict:
    best = gs.policy_result.best
    return {
        "method": "grid_search",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_id": mid,
        "search_stats": {
            "n_total":          gs.n_total,
            "n_feasible":       gs.n_feasible,
            "n_infeasible":     gs.n_infeasible,
            "elapsed_s":        round(gs.elapsed_s, 3),
            "inner_grid_size":  gs.inner_grid_size,
            "step":             gs.step,
        },
        "input": {
            "system":   {**asdict(cap), **asdict(coef)},
            "model":    asdict(spec),
            "workload": asdict(wl),
        },
        "best_policy": _candidate_to_json(best),
        "objective": {
            "per_token_latency_ms": round(best.t_per_token_s * 1000, 4),
            "throughput_tok_s":     round(1.0 / best.t_per_token_s, 4),
            "t_block_ms":           round(best.t_block_s * 1000, 4),
        },
        "top_k_candidates": [_candidate_to_json(c) for c in gs.policy_result.top_k],
    }


def run(
    model_id: str,
    workload_path: str,
    recalibrate: bool,
    output_dir: str,
    log_dir: str,
    cache_dir: str,
    verbose: bool,
    sim_gpu_gb: float | None = None,
    sim_ram_gb: float | None = None,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"grid_baseline_{ts}.log"
    json_path = Path(output_dir) / f"grid_baseline_{ts}.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    _setup_logging(log_path, verbose)
    log = logging.getLogger("run_grid_baseline")
    log.info("=== Grid search baseline run ===")
    log.info("model=%s workload=%s recalibrate=%s", model_id, workload_path, recalibrate)

    mid = machine_id()
    cap = probe_live_capacity(project_root=str(ROOT))
    if sim_gpu_gb is not None or sim_ram_gb is not None:
        cap = LiveCapacity(
            gpu_vram_gb=sim_gpu_gb if sim_gpu_gb is not None else cap.gpu_vram_gb,
            ram_gb=sim_ram_gb if sim_ram_gb is not None else cap.ram_gb,
            disk_gb=cap.disk_gb,
        )
        log.info("SIMULATED capacity: gpu_vram=%.2fGB ram=%.2fGB disk=%.2fGB",
                 cap.gpu_vram_gb, cap.ram_gb, cap.disk_gb)
    else:
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

    log.info("Running grid search (240 outer × 3 375 inner = 810 000 evaluations)…")
    gs = run_grid_search(cap, spec, wl, coef, top_k=20)

    best = gs.policy_result.best
    log.info(
        "Best: gbs=%d num_gb=%d q=%s delegate=%s overlap=%s -> %.2f ms/token  (%.1fs elapsed)",
        best.enum.gbs, best.enum.num_gb, best.enum.q,
        best.enum.delegate, best.enum.overlap,
        best.t_per_token_s * 1000, gs.elapsed_s,
    )

    payload = build_output_payload(gs, cap, coef, spec, wl, mid)
    json_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote: %s", json_path)
    return str(json_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search baseline for FlexGen 14-parameter policy."
    )
    parser.add_argument("--model",       default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--workload",    default=str(ROOT / "configs" / "workload.yaml"))
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--output-dir",  default=str(ROOT / "experiments" / "results"))
    parser.add_argument("--log-dir",     default=str(ROOT / "experiments" / "logs"))
    parser.add_argument("--cache-dir",   default=str(ROOT / "configs" / "system_calibration"))
    parser.add_argument("--sim-gpu-gb", type=float, default=None,
                        help="Simulate this GPU VRAM (GB) instead of probing real hardware")
    parser.add_argument("--sim-ram-gb", type=float, default=None,
                        help="Simulate this RAM (GB) instead of probing real hardware")
    args = parser.parse_args()
    path = run(
        model_id=args.model,
        workload_path=args.workload,
        recalibrate=args.recalibrate,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        sim_gpu_gb=args.sim_gpu_gb,
        sim_ram_gb=args.sim_ram_gb,
    )
    print(f"\nGrid baseline result: {path}")


if __name__ == "__main__":
    main()
