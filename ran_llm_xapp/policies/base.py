from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Observation:
    """Observation passed to policies at reconfiguration time."""

    t: int
    sigma1: float
    sigma2: float
    slice2_active: bool
    recent_hat_sigma1: Sequence[float]
    recent_hat_sigma2: Sequence[float]


@dataclass(frozen=True)
class SlotOutcome:
    """Outcome recorded at slot end (for OPRO history)."""

    k: int
    t_start: int
    t_end: int
    action: Tuple[int, int]
    prb1: int
    prb2: int
    sigma1: float
    sigma2: float
    mean_hat_sigma1: float
    mean_hat_sigma2: float
    V: float


def clamp_action(a1: int, a2: int) -> Tuple[int, int]:
    a1 = int(max(1, min(128, a1)))
    a2 = int(max(1, min(128, a2)))
    return a1, a2


class Policy(abc.ABC):
    """Policy interface: choose action A_k at each reconfig slot."""

    name: str

    def reset(self) -> None:
        return None

    @abc.abstractmethod
    def select_action(self, obs: Observation) -> Tuple[int, int]:
        """Return action A_k = (a1,a2), each in [1,128] integer."""

    def record_outcome(self, outcome: SlotOutcome) -> None:
        """Optional hook for policies that learn from outcomes (LLM-OPRO)."""

        _ = outcome
        return None

