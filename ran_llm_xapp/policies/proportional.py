from __future__ import annotations

from typing import Tuple

from .base import Observation, Policy, clamp_action


class ProportionalPolicy(Policy):
    name = "proportional"

    def select_action(self, obs: Observation) -> Tuple[int, int]:
        if not obs.slice2_active:
            return (128, 1)
        # Simple proportional action using requested rates (sigma1:sigma2).
        a1 = int(round(obs.sigma1))
        a2 = int(round(obs.sigma2))
        return clamp_action(a1, a2)

