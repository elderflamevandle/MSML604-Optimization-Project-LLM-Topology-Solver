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
