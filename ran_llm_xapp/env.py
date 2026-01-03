from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import ExperimentConfig


@dataclass(frozen=True)
class EnvStep:
    """Environment output at time t (dt=1s)."""

    t: int
    prb1: int
    prb2: int
    hat_sigma1: float
    hat_sigma2: float
    slice2_active: bool


class SyntheticRANSliceEnv:
    """Synthetic two-slice RAN environment for paper-like throughput curves.

    Design choices to match the requested 'Fig.4-like' trends (default config):
    - UE1(S1) and UE2(S2) exist for the whole horizon (t∈[0,T_end]).
      The *control timeline* (pre-slice fixed split / init equal split / policy) is handled
      by the experiment runner; the env only maps (prb1, prb2) -> measured rates.
    - UE2 (S2) has a higher per-PRB efficiency and a cap near 10 Mbps; thus it can
      meet its demand with relatively few PRBs, leaving room for llm/proportional
      to prioritize UE1 without catastrophically harming UE2 utility/reliability.
    - Light, correlated channel perturbation is modeled via AR(1) additive noise:
        x_t = rho * x_{t-1} + eps_t
      plus a small measurement noise, producing realistic non-perfectly-smooth curves.
    """

    def __init__(self, cfg: ExperimentConfig, *, seed: int) -> None:
        self.cfg = cfg
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._ar1 = 0.0
        self._ar2 = 0.0

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._ar1 = 0.0
        self._ar2 = 0.0

    def slice2_active(self, t: int) -> bool:
        # UE2 exists for the whole horizon in this synthetic setup.
        _ = t
        return True

    def effective_prb_budget(self, t: int) -> int:
        _ = t
        return self.cfg.R_total

    def step(self, *, t: int, prb1: int, prb2: int) -> EnvStep:
        slice2_active = True
        R_eff = self.effective_prb_budget(t)

        prb1 = max(0, int(prb1))
        prb2 = max(0, int(prb2))
        if (prb1 + prb2) > R_eff:
            # Should not happen if Eq.(7)+fixer is used, but clamp defensively.
            overflow = (prb1 + prb2) - R_eff
            if prb1 >= prb2:
                prb1 = max(0, prb1 - overflow)
            else:
                prb2 = max(0, prb2 - overflow)

        # Ideal rates (Mbps) with a cap to model application/traffic ceiling.
        ideal1 = min(self.cfg.cap1_mbps, self.cfg.eff1_mbps_per_prb * prb1)
        ideal2 = min(self.cfg.cap2_mbps, self.cfg.eff2_mbps_per_prb * prb2)

        # Correlated channel perturbation (AR(1) additive noise, Mbps).
        self._ar1 = (self.cfg.ar_rho * self._ar1) + self._rng.gauss(0.0, self.cfg.ar_eps_std1)
        self._ar2 = (self.cfg.ar_rho * self._ar2) + self._rng.gauss(0.0, self.cfg.ar_eps_std2)

        # Measurement noise (small, independent).
        meas1 = self._rng.gauss(0.0, self.cfg.meas_std1)
        meas2 = self._rng.gauss(0.0, self.cfg.meas_std2)

        hat1 = max(0.0, ideal1 + self._ar1 + meas1)
        hat2 = max(0.0, ideal2 + self._ar2 + meas2)

        # Final soft cap (avoid unbounded overshoot while allowing slight >cap due to noise).
        hat1 = min(hat1, self.cfg.cap1_mbps + 2.0)
        hat2 = min(hat2, self.cfg.cap2_mbps + 1.5)

        return EnvStep(
            t=int(t),
            prb1=int(prb1),
            prb2=int(prb2),
            hat_sigma1=float(hat1),
            hat_sigma2=float(hat2),
            slice2_active=bool(slice2_active),
        )
