from __future__ import annotations

import abc
from typing import Optional


class LLMClient(abc.ABC):
    """Minimal LLM client interface for OPRO."""

    provider: str

    @abc.abstractmethod
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
        """Return raw text completion (expected to be a JSON object string)."""

