from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..config import ExperimentConfig
from ..io_utils import PromptResponseCache
from ..llm_clients.base import LLMClient
from ..metrics import action_to_prbs, effective_cap_mbps, get_g_function
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
    V_k: float
    prb2_min_est: int
    waste: int
    penalty: float
    V_k_soft: float


def _extract_json_object(text: str) -> Optional[str]:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and i < j:
        return s[i : j + 1]
    return None


def _extract_json_array(text: str) -> Optional[str]:
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        return s
    i = s.find("[")
    j = s.rfind("]")
    if i != -1 and j != -1 and i < j:
        return s[i : j + 1]
    return None


def _parse_action(text: str) -> Optional[Tuple[int, int]]:
    """Parse a single {"a1":int,"a2":int} from text (legacy/repair fallback)."""

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


def _parse_candidate_actions(text: str) -> Optional[List[Tuple[int, int]]]:
    """Robustly parse a JSON list of candidate actions from LLM output.

    Expected format (strict):
      [{"a1": 120, "a2": 8}, {"a1": 112, "a2": 16}, ...]

    Also accepts a wrapper object: {"candidates":[...]} as a recovery.
    """

    s = text.strip()
    obj: object
    try:
        obj = json.loads(s)
    except Exception:
        blob = _extract_json_array(s) or _extract_json_object(s)
        if not blob:
            return None
        try:
            obj = json.loads(blob)
        except Exception:
            return None

    if isinstance(obj, dict) and "candidates" in obj:
        obj = obj["candidates"]
    if not isinstance(obj, list):
        return None

    out: List[Tuple[int, int]] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        if "a1" not in item or "a2" not in item:
            continue
        try:
            out.append(clamp_action(int(item["a1"]), int(item["a2"])))
        except Exception:
            continue
    return out or None


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
        self._debug = os.getenv("RAN_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        self._last_candidates: List[Tuple[int, int]] = []
        self._last_proxy_scores: List[float] = []
        self._last_chosen_index: Optional[int] = None

    def reset(self) -> None:
        self.temperature = float(self.cfg.Tem_max)
        self._last_action = None
        self.optimization_history = []
        self._last_cache_path = None
        self._last_candidates = []
        self._last_proxy_scores = []
        self._last_chosen_index = None

    def _history_examples(self) -> List[HistoryExample]:
        # Optimization history is stored sorted by V_k_soft desc.
        examples: List[HistoryExample] = []
        for rec in self.optimization_history[: self.cfg.in_context_n_examples]:
            examples.append(
                HistoryExample(
                    k=rec.k,
                    V_k_soft=rec.V_k_soft,
                    a1=rec.action[0],
                    a2=rec.action[1],
                    prb1=rec.prb1,
                    prb2=rec.prb2,
                    mean_hat_sigma1=rec.mean_hat_sigma1,
                    mean_hat_sigma2=rec.mean_hat_sigma2,
                    prb2_min_est=rec.prb2_min_est,
                    waste=rec.waste,
                    penalty=rec.penalty,
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

        cap1_hard = self.cfg.cap1_hard_mbps
        cap2_hard = self.cfg.cap2_hard_mbps
        eff_cap1 = effective_cap_mbps(float(obs.sigma1), cap1_hard)
        eff_cap2 = effective_cap_mbps(float(obs.sigma2), cap2_hard)

        # Estimate UE2 minimal PRBs to reach its effective target (soft, used for prompt + proxy).
        prb2_curr = int(max(0, obs.current_prb2))
        eff2_est = max(float(self.cfg.eff2_mbps_per_prb), float(self.cfg.waste_eps))
        prb2_min_est = int(math.ceil(eff_cap2 / max(eff2_est, float(self.cfg.waste_eps)))) if eff_cap2 > 0 else 0
        prb2_min_est = max(0, prb2_min_est)

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
            cap1_hard_mbps=cap1_hard,
            cap2_hard_mbps=cap2_hard,
            eff_cap2_mbps=float(eff_cap2),
            current_prb1=int(max(0, obs.current_prb1)),
            current_prb2=prb2_curr,
            prb2_min_est=int(prb2_min_est),
            lambda_waste=float(self.cfg.lambda_waste),
            last_action_a1=last_a1,
            last_action_a2=last_a2,
        )
        prompt = build_meta_prompt(
            obs=prompt_obs,
            history_examples=self._history_examples(),
            in_context_top_n=self.cfg.in_context_top_n,
            require_new_action=True,
        )

        candidates: Optional[List[Tuple[int, int]]] = None
        response = ""
        try:
            response = self._complete_cached(prompt=prompt, temperature=self.temperature)
            candidates = _parse_candidate_actions(response)
        except Exception as e:
            logger.warning("LLM call failed; fallback to proportional. err=%s", e)

        if candidates is None and self.cfg.llm_parse_retry > 0:
            try:
                repair = build_repair_prompt(bad_response=response)
                response2 = self._complete_cached(prompt=repair, temperature=0.0)
                candidates = _parse_candidate_actions(response2)
            except Exception as e:
                logger.warning("LLM repair call failed; fallback to proportional. err=%s", e)

        if candidates is None:
            snippet = (response or "").strip().replace("\n", "\\n")
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            logger.warning(
                "LLM output parse failed; fallback to proportional. cache=%s response=%s",
                str(self._last_cache_path) if self._last_cache_path else "N/A",
                snippet or "<empty>",
            )
            candidates = [clamp_action(int(round(obs.sigma1)), int(round(obs.sigma2)))]

        # Ensure diversity / include aggressive small-a2 candidates for best-of-N selection.
        aggressive_set = {4, 8, 12, 16}
        seed_required = [clamp_action(128 - int(prb2_min_est), int(prb2_min_est))]
        seed_aggressive = [clamp_action(128 - a2, a2) for a2 in (4, 8, 12)]
        seed_mid = [
            clamp_action(int(round(obs.sigma1)), int(round(obs.sigma2))),  # proportional
            clamp_action(int(round(obs.sigma1)), max(1, int(round(obs.sigma2 * 1.5)))),
            (64, 64),  # equal (as a conservative fallback)
        ]

        combined = seed_required + list(seed_aggressive) + list(candidates) + seed_mid
        # De-dup while preserving order.
        uniq: List[Tuple[int, int]] = []
        for a in combined:
            if a not in uniq:
                uniq.append(a)

        # Ensure at least 6 candidates; pad with mild variations if needed.
        fillers = [clamp_action(120, 16), clamp_action(116, 12), clamp_action(112, 8), clamp_action(124, 4)]
        if len(uniq) < 6:
            for f in fillers:
                if f not in uniq:
                    uniq.append(f)
                if len(uniq) >= 6:
                    break
        if len(uniq) < 6:
            # As a last resort, generate additional deterministic variants.
            for a2 in range(2, 32, 2):
                cand = clamp_action(128 - a2, a2)
                if cand not in uniq:
                    uniq.append(cand)
                if len(uniq) >= 6:
                    break
        # Ensure at least 3 aggressive candidates remain in the first 6.
        first_six = uniq[:]
        aggressive_count = sum(1 for a in first_six[:6] if a[1] in aggressive_set)
        if aggressive_count < 3:
            for a2 in (4, 8, 12, 16):
                cand = clamp_action(128 - a2, a2)
                if cand in first_six[:6]:
                    continue
                # Replace a non-aggressive tail candidate.
                for j in range(5, -1, -1):
                    if first_six[j][1] not in aggressive_set:
                        first_six[j] = cand
                        aggressive_count += 1
                        break
                if aggressive_count >= 3:
                    break
        candidates = first_six[:6]

        # Best-of-N: choose by one-step soft proxy (includes UE2 waste penalty).
        g = get_g_function(self.cfg)
        prb1_curr = int(max(0, obs.current_prb1))
        eff1_est_raw = float(mean1) / max(prb1_curr, 1)
        eff1_est = max(eff1_est_raw, float(self.cfg.eff1_mbps_per_prb), float(self.cfg.waste_eps))

        proxy_scores: List[float] = []
        scored: List[Tuple[float, int, Tuple[int, int]]] = []
        for idx, cand in enumerate(candidates):
            prb1_c, prb2_c = action_to_prbs(cand, self.cfg.R_total)
            pred1 = min(eff_cap1, eff1_est * prb1_c)
            pred2 = min(eff_cap2, eff2_est * prb2_c)
            x1 = pred1 - eff_cap1
            x2 = pred2 - eff_cap2
            waste = max(0, int(prb2_c) - int(prb2_min_est))
            score = (self.cfg.beta1 * g(float(x1))) + (self.cfg.beta2 * g(float(x2))) - (
                float(self.cfg.lambda_waste) * float(waste * waste)
            )
            proxy_scores.append(float(score))
            scored.append((float(score), idx, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen_idx: Optional[int] = int(scored[0][1]) if scored else None
        chosen_action: Tuple[int, int] = scored[0][2] if scored else (128, 1)

        self._last_candidates = list(candidates)
        self._last_proxy_scores = list(proxy_scores)
        self._last_chosen_index = chosen_idx
        if self._debug:
            logger.info(
                "LLM best-of-N choose idx=%s action=%s prbs=%s scores=%s",
                str(chosen_idx) if chosen_idx is not None else "N/A",
                chosen_action,
                action_to_prbs(chosen_action, self.cfg.R_total),
                [round(s, 2) for s in proxy_scores],
            )

        self._last_action = chosen_action

        # Temperature decay (Table I).
        self.temperature = max(self.cfg.Tem_min, self.temperature - self.cfg.Tem_delta)
        return chosen_action

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
            V_k=outcome.V_k,
            prb2_min_est=outcome.prb2_min_est,
            waste=outcome.waste,
            penalty=outcome.penalty,
            V_k_soft=outcome.V_k_soft,
        )
        self.optimization_history.append(rec)
        self.optimization_history.sort(key=lambda r: r.V_k_soft, reverse=True)
