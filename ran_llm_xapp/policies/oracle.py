from __future__ import annotations

from typing import Tuple

from ..config import ExperimentConfig
from ..metrics import score_allocation_proxy
from .base import Observation, Policy


class OraclePolicy(Policy):
    """Oracle (Exact-Soft-Opt): enumerate PRB allocations and pick best by proxy score."""

    name = "oracle"

    def __init__(self, *, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def select_action(self, obs: Observation) -> Tuple[int, int]:
        # Return (prb1, prb2) directly (runner treats oracle output as PRB allocation).
        best_score = float("-inf")
        best_prbs = (0, int(self.cfg.R_total))
        for prb1 in range(0, int(self.cfg.R_total) + 1):
            prb2 = int(self.cfg.R_total) - int(prb1)
            score, _extra = score_allocation_proxy(
                self.cfg,
                t=int(obs.t),
                sigma1=float(obs.sigma1),
                sigma2=float(obs.sigma2),
                prb1=int(prb1),
                prb2=int(prb2),
            )
            if score > best_score:
                best_score = float(score)
                best_prbs = (int(prb1), int(prb2))
        return best_prbs

