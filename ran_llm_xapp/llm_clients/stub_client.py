from __future__ import annotations

import json
import random
import re
from typing import Optional, Tuple

from .base import LLMClient


_RE_SIGMA = re.compile(r"requested rates .*?sigma1\s*=\s*([0-9.]+),\s*sigma2\s*=\s*([0-9.]+|N/A)", re.I)
_RE_UE1_MEAN = re.compile(r"UE1:\s*mean=([0-9.]+)", re.I)
_RE_UE2_MEAN = re.compile(r"UE2:\s*mean=([0-9.]+|N/A)", re.I)


class StubLLMClient(LLMClient):
    """Heuristic stub that returns a valid JSON action without any API key.

    It is intentionally simple and deterministic (given seed + prompt + temperature),
    designed to mimic the 'llm improves UE1 after 200s while keeping UE2 near demand'
    behavior in the synthetic environment.
    """

    provider = "stub"

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = random.Random(int(seed))

    def _parse_obs(self, prompt: str) -> Tuple[float, Optional[float], float, Optional[float]]:
        sigma1 = 40.0
        sigma2: Optional[float] = 10.0
        m = _RE_SIGMA.search(prompt)
        if m:
            sigma1 = float(m.group(1))
            sigma2_str = m.group(2)
            sigma2 = None if sigma2_str.upper() == "N/A" else float(sigma2_str)

        ue1_mean = 0.0
        ue2_mean: Optional[float] = None
        m1 = _RE_UE1_MEAN.search(prompt)
        if m1:
            ue1_mean = float(m1.group(1))
        m2 = _RE_UE2_MEAN.search(prompt)
        if m2:
            s = m2.group(1)
            ue2_mean = None if s.upper() == "N/A" else float(s)
        return sigma1, sigma2, ue1_mean, ue2_mean

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
        _ = (model, max_tokens, timeout_s)
        if seed is not None:
            # Make it reproducible per-run if caller provides seed.
            local_rng = random.Random(int(seed))
        else:
            local_rng = self._rng

        sigma1, sigma2, ue1_mean, ue2_mean = self._parse_obs(prompt)
        if sigma2 is None or ue2_mean is None:
            action = {"a1": 128, "a2": 1}
            return json.dumps(action)

        # Heuristic: keep UE2 comfortably above its utility threshold (~8.5 Mbps),
        # allocate the rest to UE1, with mild temperature-scaled jitter.
        desired_prb2 = 16
        if ue2_mean < 9.0:
            desired_prb2 = 24
        elif ue2_mean > (sigma2 + 0.3):
            desired_prb2 = 12

        # If UE1 is far below its demand, push more toward UE1 by shrinking PRB2.
        if ue1_mean < (0.85 * sigma1):
            desired_prb2 = max(10, desired_prb2 - 2)

        # Temperature-scaled exploration (integer jitter). Keep it mild so curves
        # remain paper-like (llm is relatively stable vs random).
        jitter = int(round(local_rng.gauss(0.0, max(0.5, 2.5 * float(temperature)))))
        # Keep UE2 minimally protected to avoid large reliability degradation.
        desired_prb2 = int(max(12, min(40, desired_prb2 + jitter)))
        desired_prb1 = int(max(1, 128 - desired_prb2))

        a1 = max(1, min(128, desired_prb1))
        a2 = max(1, min(128, desired_prb2))
        return json.dumps({"a1": int(a1), "a2": int(a2)})
