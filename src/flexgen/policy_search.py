from dataclasses import dataclass
from typing import Iterator
import logging

from src.flexgen.cost_model import EnumPoint, PlacementFractions
from src.flexgen.lp_formulation import solve_inner_lp, InnerLPResult
from src.flexgen.system_probe import LiveCapacity
from src.flexgen.calibration import SystemCoefficients
from src.flexgen.model_introspect import ModelSpec
from src.flexgen.workload import WorkloadSpec

logger = logging.getLogger(__name__)

GBS_GRID = (1, 2, 4, 8, 16, 32)
NUM_GB_GRID = (1, 2, 4, 8, 16)
QUANT_GRID = ("fp16", "int4")


@dataclass(frozen=True)
class Candidate:
    enum: EnumPoint
    placement: PlacementFractions
    t_per_token_s: float
    t_block_s: float


@dataclass(frozen=True)
class PolicyResult:
    best: Candidate
    top_k: list[Candidate]


def _enum_iter() -> Iterator[EnumPoint]:
    for gbs in GBS_GRID:
        for num_gb in NUM_GB_GRID:
            for q in QUANT_GRID:
                for delegate in (False, True):
                    for overlap in (False, True):
                        yield EnumPoint(gbs=gbs, num_gb=num_gb, q=q,
                                        delegate=delegate, overlap=overlap)


def run_policy_search(
    cap: LiveCapacity, spec: ModelSpec, wl: WorkloadSpec, coef: SystemCoefficients,
    top_k: int = 20,
) -> PolicyResult:
    candidates: list[Candidate] = []
    n_total, n_feasible, n_infeasible = 0, 0, 0
    for enum in _enum_iter():
        n_total += 1
        res: InnerLPResult = solve_inner_lp(enum, cap, spec, wl, coef)
        if res.status == "Optimal":
            n_feasible += 1
            candidates.append(Candidate(
                enum=enum, placement=res.placement,
                t_per_token_s=res.t_per_token_s, t_block_s=res.t_block_s,
            ))
        else:
            n_infeasible += 1

    if not candidates:
        raise RuntimeError(f"Policy search found no feasible config (n_total={n_total})")

    candidates.sort(key=lambda c: (c.t_per_token_s,
                                   -(c.placement.w_g + c.placement.c_g + c.placement.h_g)))
    logger.info(
        "policy search: total=%d feasible=%d infeasible=%d best_t=%.4fs",
        n_total, n_feasible, n_infeasible, candidates[0].t_per_token_s,
    )
    return PolicyResult(best=candidates[0], top_k=candidates[:top_k])
