from __future__ import annotations

import random
from typing import Tuple

from .base import Observation, Policy


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, *, seed: int) -> None:
        self._seed = int(seed)
        self._rng = random.Random(self._seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def select_action(self, obs: Observation) -> Tuple[int, int]:
        if not obs.slice2_active:
            return (128, 1)
        return (self._rng.randint(1, 128), self._rng.randint(1, 128))

