from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..config import ExperimentConfig
from ..io_utils import PromptResponseCache
from ..llm_clients.base import LLMClient
from ..prompts import HistoryExample, PromptObservation, build_meta_prompt, build_repair_prompt
from .base import Observation, Policy, SlotOutcome, clamp_action

logger = logging.getLogger(__name__)


_RE_A1 = re.compile(r"\"?a1\"?\s*:\s*([0-9]+)")
_RE_A2 = re.compile(r"\"?a2\"?\s*:\s*([0-9]+)")


@dataclass
class OptimizationRecord:
    """A single optimization history record (Algorithm 1)."""

    k: int
    action: Tuple[int, int]
    prb1: int
    prb2: int
    sigma1: float
    sigma2: float
    mean_hat_sigma1: float
    mean_hat_sigma2: float
    V: float


def _extract_json_object(text: str) -> Optional[str]:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and i < j:
        return s[i : j + 1]
    return None


def _parse_action(text: str) -> Optional[Tuple[int, int]]:
    """Robustly parse {"a1":int,"a2":int} from LLM output."""

    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "a1" in obj and "a2" in obj:
            a1 = int(obj["a1"])
            a2 = int(obj["a2"])
            return clamp_action(a1, a2)
    except Exception:
        pass

    blob = _extract_json_object(s)
    if blob:
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict) and "a1" in obj and "a2" in obj:
                return clamp_action(int(obj["a1"]), int(obj["a2"]))
        except Exception:
            pass

    m1 = _RE_A1.search(s)
    m2 = _RE_A2.search(s)
    if m1 and m2:
        return clamp_action(int(m1.group(1)), int(m2.group(1)))
    return None


class LLMOPROPolicy(Policy):
    name = "llm"

    def __init__(
        self,
        *,
        cfg: ExperimentConfig,
        llm_client: LLMClient,
        model: str,
        cache: PromptResponseCache,
        seed: int,
    ) -> None:
        self.cfg = cfg
        self.client = llm_client
        self.model = model
        self.cache = cache
        self.seed = int(seed)

        self.temperature = float(cfg.Tem_max)
        self._last_action: Optional[Tuple[int, int]] = None
        self.optimization_history: List[OptimizationRecord] = []
        self._last_cache_path: Optional[Path] = None

    def reset(self) -> None:
        self.temperature = float(self.cfg.Tem_max)
        self._last_action = None
        self.optimization_history = []
        self._last_cache_path = None

    def _history_examples(self) -> List[HistoryExample]:
        # Optimization history is stored sorted by V desc.
        examples: List[HistoryExample] = []
        for rec in self.optimization_history[: self.cfg.in_context_n_examples]:
            examples.append(
                HistoryExample(
                    k=rec.k,
                    V=rec.V,
                    a1=rec.action[0],
                    a2=rec.action[1],
                    prb1=rec.prb1,
                    prb2=rec.prb2,
                    mean_hat_sigma1=rec.mean_hat_sigma1,
                    mean_hat_sigma2=rec.mean_hat_sigma2,
                )
            )
        return examples

    def _complete_cached(self, *, prompt: str, temperature: float) -> str:
        hit = self.cache.get(
            provider=self.client.provider,
            model=self.model,
            temperature=temperature,
            prompt=prompt,
            extra={"seed": self.seed},
        )
        if hit is not None:
            self._last_cache_path = hit.cache_path
            return hit.response_text

        response = self.client.complete(
            prompt=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=self.cfg.llm_max_tokens,
            timeout_s=self.cfg.llm_timeout_s,
            seed=self.seed,
        )
        cache_path = self.cache.put(
            provider=self.client.provider,
            model=self.model,
            temperature=temperature,
            prompt=prompt,
            response_text=response,
            extra={"seed": self.seed},
        )
        self._last_cache_path = cache_path
        return response

    def select_action(self, obs: Observation) -> Tuple[int, int]:
        if not obs.slice2_active:
            self._last_action = (128, 1)
            return (128, 1)

        win1 = list(obs.recent_hat_sigma1)
        win2 = list(obs.recent_hat_sigma2)
        mean1 = sum(win1) / len(win1) if win1 else 0.0
        mean2 = sum(win2) / len(win2) if win2 else 0.0
        last1 = win1[-1] if win1 else 0.0
        last2 = win2[-1] if win2 else 0.0

        last_a1 = self._last_action[0] if self._last_action else None
        last_a2 = self._last_action[1] if self._last_action else None

        prompt_obs = PromptObservation(
            t=obs.t,
            sigma1=obs.sigma1,
            sigma2=obs.sigma2,
            slice2_active=obs.slice2_active,
            mean_hat_sigma1=float(mean1),
            mean_hat_sigma2=float(mean2),
            last_hat_sigma1=float(last1),
            last_hat_sigma2=float(last2),
            Tem_k=float(self.temperature),
            last_action_a1=last_a1,
            last_action_a2=last_a2,
        )
        prompt = build_meta_prompt(
            obs=prompt_obs,
            history_examples=self._history_examples(),
            in_context_top_n=self.cfg.in_context_top_n,
            require_new_action=True,
        )

        action: Optional[Tuple[int, int]] = None
        response = ""
        try:
            response = self._complete_cached(prompt=prompt, temperature=self.temperature)
            action = _parse_action(response)
        except Exception as e:
            logger.warning("LLM call failed; fallback to proportional. err=%s", e)

        if action is None and self.cfg.llm_parse_retry > 0:
            try:
                repair = build_repair_prompt(bad_response=response)
                response2 = self._complete_cached(prompt=repair, temperature=0.0)
                action = _parse_action(response2)
            except Exception as e:
                logger.warning("LLM repair call failed; fallback to proportional. err=%s", e)

        if action is None:
            snippet = (response or "").strip().replace("\n", "\\n")
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            logger.warning(
                "LLM output parse failed; fallback to proportional. cache=%s response=%s",
                str(self._last_cache_path) if self._last_cache_path else "N/A",
                snippet or "<empty>",
            )
            action = clamp_action(int(round(obs.sigma1)), int(round(obs.sigma2)))

        if self._last_action is not None and action == self._last_action:
            # Enforce "must be different" by nudging a1.
            a1, a2 = action
            a1 = a1 + 1 if a1 < 128 else a1 - 1
            action = clamp_action(a1, a2)

        self._last_action = action

        # Temperature decay (Table I).
        self.temperature = max(self.cfg.Tem_min, self.temperature - self.cfg.Tem_delta)
        return action

    def record_outcome(self, outcome: SlotOutcome) -> None:
        rec = OptimizationRecord(
            k=outcome.k,
            action=outcome.action,
            prb1=outcome.prb1,
            prb2=outcome.prb2,
            sigma1=outcome.sigma1,
            sigma2=outcome.sigma2,
            mean_hat_sigma1=outcome.mean_hat_sigma1,
            mean_hat_sigma2=outcome.mean_hat_sigma2,
            V=outcome.V,
        )
        self.optimization_history.append(rec)
        self.optimization_history.sort(key=lambda r: r.V, reverse=True)
