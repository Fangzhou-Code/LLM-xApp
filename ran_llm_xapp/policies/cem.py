from __future__ import annotations

import logging
import math
import random
from typing import List, Tuple

from ..config import ExperimentConfig
from ..metrics import score_allocation_proxy
from .base import Observation, Policy

logger = logging.getLogger(__name__)


def _snap_to_grid(x: float, *, step: int, R_total: int) -> int:
    """Map a continuous x∈[0,R_total] to the nearest discrete PRB grid point.

    Candidate set: {0, step, 2*step, ..., R_total}. The last point R_total is always included
    even if it is not a multiple of `step`.
    """

    if step <= 0:
        raise ValueError("step must be positive")
    R = int(max(0, R_total))
    xc = float(min(max(0.0, x), float(R)))
    lower = int(math.floor(xc / float(step))) * int(step)
    upper = min(R, lower + int(step))
    # Choose the nearest; break ties toward upper to allow reaching R_total.
    if (xc - lower) < (upper - xc):
        return int(lower)
    return int(upper)


class CEMPolicy(Policy):
    """Budgeted CEM baseline (elite sampling) with coarse PRB grid.

    This baseline simulates a real-time xApp with a *very limited* evaluation budget per slot:
    - cem_iters=1, cem_samples=8, cem_elite_k=2 by default
    - r1 is restricted to a coarse grid (step=8 by default)

    It is a reasonable black-box search baseline, but due to the small budget and coarse action
    granularity it typically underperforms TNAS (LLM-proposed diverse candidates).
    """

    name = "cem"

    def __init__(self, *, cfg: ExperimentConfig, seed: int) -> None:
        self.cfg = cfg
        self._rng = random.Random(int(seed))
        self._mu = float(cfg.R_total) / 2.0
        self._sigma = float(cfg.R_total) / 2.0
        self._initialized = False

    def reset(self) -> None:
        self._mu = float(self.cfg.R_total) / 2.0
        self._sigma = float(self.cfg.R_total) / 2.0
        self._initialized = False

    def select_action(self, obs: Observation) -> Tuple[int, int]:
        # Return (prb1, prb2) directly (runner treats CEM output as PRB allocation).
        R = int(self.cfg.R_total)
        if R <= 0:
            return (0, 0)

        step = int(max(1, self.cfg.cem_step))
        iters = int(max(1, self.cfg.cem_iters))
        n = int(max(1, self.cfg.cem_samples))
        elite_k = int(max(1, min(self.cfg.cem_elite_k, n)))
        alpha = float(self.cfg.cem_alpha)
        alpha = min(max(alpha, 0.0), 1.0)

        # Initialize distribution around current allocation once (at first use).
        if not self._initialized:
            self._mu = float(min(max(0, obs.current_prb1), R))
            # Moderate initial exploration; do not collapse below the grid step.
            self._sigma = max(float(step), float(R) / 6.0)
            self._initialized = True

        mu = float(self._mu)
        sigma = float(self._sigma)
        best_score = float("-inf")
        best_prbs = (int(obs.current_prb1), int(obs.current_prb2))

        for _ in range(iters):
            scored: List[Tuple[float, int, int]] = []
            for _j in range(n):
                x = self._rng.gauss(mu, sigma)
                prb1 = _snap_to_grid(x, step=step, R_total=R)
                prb2 = R - int(prb1)
                score, _extra = score_allocation_proxy(
                    self.cfg,
                    t=int(obs.t),
                    sigma1=float(obs.sigma1),
                    sigma2=float(obs.sigma2),
                    prb1=int(prb1),
                    prb2=int(prb2),
                )
                scored.append((float(score), int(prb1), int(prb2)))
                if score > best_score:
                    best_score = float(score)
                    best_prbs = (int(prb1), int(prb2))

            scored.sort(key=lambda z: z[0], reverse=True)
            elites = scored[:elite_k]
            elite_r1 = [float(p1) for _s, p1, _p2 in elites]
            elite_mean = sum(elite_r1) / len(elite_r1) if elite_r1 else mu
            elite_var = sum((x - elite_mean) ** 2 for x in elite_r1) / len(elite_r1) if elite_r1 else (sigma * sigma)
            elite_std = math.sqrt(max(0.0, float(elite_var)))

            # Smooth update (CEM with alpha): keep some inertia to mimic limited iterations/budget.
            mu = (alpha * mu) + ((1.0 - alpha) * elite_mean)
            sigma = (alpha * sigma) + ((1.0 - alpha) * elite_std)
            sigma = max(float(step), float(sigma))

        self._mu = float(min(max(0.0, mu), float(R)))
        self._sigma = float(min(max(float(step), sigma), float(R)))

        # Standard CEM outputs the (smoothed) mean as the chosen action.
        # This makes the baseline stable but, with tiny budgets + coarse grid, often suboptimal.
        chosen_prb1 = _snap_to_grid(self._mu, step=step, R_total=R)
        chosen_prb2 = R - int(chosen_prb1)
        chosen_score, _ = score_allocation_proxy(
            self.cfg,
            t=int(obs.t),
            sigma1=float(obs.sigma1),
            sigma2=float(obs.sigma2),
            prb1=int(chosen_prb1),
            prb2=int(chosen_prb2),
        )

        logger.info(
            "CEM t=%s samples=%s elite_k=%s step=%s chosen_prbs=%s score=%.2f best_prbs=%s best_score=%.2f mu=%.1f sigma=%.1f",
            int(obs.t),
            n,
            elite_k,
            step,
            (int(chosen_prb1), int(chosen_prb2)),
            float(chosen_score),
            best_prbs,
            float(best_score),
            self._mu,
            self._sigma,
        )
        return (int(chosen_prb1), int(chosen_prb2))
