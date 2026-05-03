# FlexGen Faithful Extension — Brainstorming Q&A Log

**Topic:** Extend the simplified FlexGen LP into a faithful policy-search optimization that
adds GPU batch size, # GPU batches per block, CPU compute delegation, 4-bit compression,
and I/O–compute overlap as decision variables / configuration knobs. Read host system
configuration each run; write a per-run log file.

**Selected scope:** Option A — Faithful FlexGen policy search (per the original paper's
two-level cost model, minimizing per-token latency).

**Date started:** 2026-04-26

---

## Q1 — Modeling approach (resolved)

**Asked:** How faithful should the new formulation be?

**Options presented:**
- A. Faithful FlexGen policy search — implement the paper's two-level cost model (per-block
  compute + I/O latency, with `max(...)` when overlap is on), minimize per-token latency
  `T_block / (gbs · num_gb)` subject to memory + bandwidth constraints. Closest to the
  paper, hardest math (bilinear `gbs · num_gb`).
- B. MILP extension of the current toy LP — keep the placement-penalty objective, add the
  new knobs as additive penalty terms. Simple, but not faithful.
- C. Hybrid — outer enumeration over discrete knobs, inner LP for the 9 placement
  fractions. Matches what FlexGen actually does in practice (they grid-search the discrete
  part). Easier to debug.

**Recommendation given:** C.

**User decision:** **A** (faithful policy search).

---

## Q2 — Source of cost-model coefficients

**Asked:** Where do bandwidths, compute throughput, and model/workload specs come from?

**Options presented:**
- A. Micro-benchmark on first run, cache results to `configs/system_calibration.json`.
  Most accurate, but ~30-60 s on first run and benchmark noise.
- B. Hardcoded reference values for the RTX 4050 Laptop, with YAML override. Fast and
  deterministic.
- C. YAML config file only — no auto-detection.

**Recommendation given:** B.

**User decision:** Rejected B because the project will run on multiple servers — no
hardcoded values acceptable. Asked for an automated approach, leaning toward A but
inviting alternatives.

---

## Q2.1 — Refinement of A for multi-machine automation

**Asked:** What's the right shape of the auto-calibration so it works across servers?

**Proposed refinement (Option D):**
- **Live each run:** GPU VRAM, free system RAM, free disk via `psutil` + `torch.cuda`.
  These are volatile (depend on what else is running).
- **Cached per machine:** bandwidths (GPU↔CPU, CPU↔Disk) and compute throughput at
  fp16/int8/int4. Cache key = `{hostname}_{cuda_device_name}` so each new machine
  triggers a one-time calibration (~30 s) and subsequent runs reuse it.
- **Cache location:** `configs/system_calibration/{machine_id}.json` (gitignored).
- **CLI flag:** `--recalibrate` to force re-running (e.g. after hardware upgrade).
- **Calibration method:** torch matmul timings for compute (one per quant level),
  host↔device tensor copy timings for PCIe, temp-file write/read for disk. No reliance
  on `nvidia-smi` parsing or theoretical specs lookup tables.

**User decision:** **Approved.** "this look good ... I liked the way to catch real time
system infos." Constraint reaffirmed: tool will run on multiple servers, no hardcoded
values acceptable.

---

## Q3 — Model-config automation + scope of HF weights download

**Asked:** Plug-any-HF-model — does the user want config-only introspection, full weights
download for verification, or a complete FlexGen runtime with real inference?

**Options presented:**
- A. Optimizer-only, config-driven. Pull `config.json` from HF Hub, derive memory
  analytically, no weight download. Works for any causal-LM. Fast.
- B. A + verified sizes (sum from `model.safetensors.index.json`).
- C. Full FlexGen runtime — implement weight streaming, KV management, CPU-delegated
  attention, run real forward pass to validate optimizer. Multi-week build.
- D. A now, C later (separate phase / future work).

**Recommendation given:** D, starting with A.

**User decision:** **D** (with A this iteration). Reaffirmed timeline: 5 days total to
complete; 1-day progress showcase. Confirmed all 14 parameters from the original table
must appear as outputs.

**Constraint added:** must run on multi-GPU servers; no GPU memory concern.

---

## Q4 — Workload spec source

**Asked:** Where do prompt length, decode length, and concurrency come from?

**Options presented:**
- A. Hardcoded defaults in code (prompt=512, decode=128, concurrency=1).
- B. YAML at `configs/workload.yaml` with sensible defaults; CLI override.
- C. Trace-driven — sample distributions from `data/sharegpt_vicuna/`. Most realistic.

**Recommendation given:** B for the 5-day window, with `--workload` flag accepting
`sharegpt` as a future built-in preset (Option C-as-extension).

**User decision:** **B** for this iteration. **C kept on the recommendations / future-work
list** in the design doc.

---

## Status: ready to write design spec

All four clarifying questions resolved. Moving to design-doc generation at
`docs/superpowers/specs/2026-04-26-flexgen-faithful-design.md`.

