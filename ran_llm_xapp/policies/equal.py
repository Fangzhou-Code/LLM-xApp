from __future__ import annotations

from typing import Tuple

from .base import Observation, Policy


class EqualPolicy(Policy):
    name = "equal"

    def select_action(self, obs: Observation) -> Tuple[int, int]:
        if obs.slice2_active:
            return (64, 64)
        # When S2 is inactive, a2 is ignored by the env; keep it valid anyway.
        return (128, 1)

