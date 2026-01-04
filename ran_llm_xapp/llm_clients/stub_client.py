from __future__ import annotations

import json
import random
from typing import Optional

from .base import LLMClient


class StubLLMClient(LLMClient):
    """Heuristic stub that returns a valid JSON *candidate list* without any API key.

    It is intentionally simple and deterministic (given seed + prompt + temperature),
    designed to mimic the 'tnas improves UE1 after 200s while keeping UE2 near demand'
    behavior in the synthetic environment.
    """

    provider = "stub"

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(int(seed))

    def complete(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
        seed: Optional[int] = None,
    ) -> str:
        _ = (prompt, model, max_tokens, timeout_s)
        if seed is not None:
            # Make it reproducible per-run if caller provides seed.
            local_rng = random.Random(int(seed))
        else:
            local_rng = self._rng

        # Temperature-scaled exploration (small action jitter).
        jitter = int(round(local_rng.gauss(0.0, max(0.25, 1.5 * float(temperature)))))

        # Always provide 6 candidates; at least 3 have small a2 ∈ {4,8,12,16}.
        # We make actions sum to 128 so that Eq.(7) maps approximately to PRB=(a1,a2).
        base = [
            (124, 4),
            (120, 8),
            (116, 12),
            (112, 16),
            (103, 25),  # close to proportional 40:10 mapping
            (64, 64),   # equal
        ]
        candidates = []
        for a1, a2 in base:
            a1 = max(1, min(128, int(a1 + jitter)))
            a2 = max(1, min(128, int(a2 - jitter)))
            candidates.append({"a1": a1, "a2": a2, "reason": "stub candidate"})
        return json.dumps({"candidates": candidates})
