from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..config import ExperimentConfig
from ..io_utils import PromptResponseCache
from ..llm_clients.base import LLMClient
from ..metrics import action_to_prbs, effective_cap_mbps, score_allocation_proxy
from ..prompts import PromptObservation, build_tnas_prompt, build_tnas_repair_prompt
from .base import Observation, Policy, SlotOutcome, clamp_action

logger = logging.getLogger(__name__)


class RealScoreCritic:
    """Simplified critic that learns from real V_k_soft samples."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self._lr = float(cfg.tnas_real_score_lr)
        self.explore_prob = float(cfg.tnas_real_score_explore)
        self.weights = [0.0] * self._feature_dim()

    def _feature_dim(self) -> int:
        return 16

    def build_features(
        self,
        obs: Observation,
        action: Tuple[int, int],
        prbs: Tuple[int, int],
    ) -> Sequence[float]:
        prb1, prb2 = prbs
        total_prb = max(float(self.cfg.R_total), 1.0)
        denom = float(action[0] + action[1]) if (action[0] + action[1]) > 0 else 1.0
        ratio_a1 = float(action[0]) / denom

        mean1 = float(sum(obs.recent_hat_sigma1) / len(obs.recent_hat_sigma1)) if obs.recent_hat_sigma1 else 0.0
        mean2 = float(sum(obs.recent_hat_sigma2) / len(obs.recent_hat_sigma2)) if obs.recent_hat_sigma2 else 0.0

        eff_cap1 = effective_cap_mbps(float(obs.sigma1), self.cfg.cap1_hard_mbps)
        eff_cap2 = effective_cap_mbps(float(obs.sigma2), self.cfg.cap2_hard_mbps)
        shortfall1 = max(0.0, eff_cap1 - mean1)
        shortfall2 = max(0.0, eff_cap2 - mean2)
        base_cap1 = max(float(eff_cap1), 1e-6)
        base_cap2 = max(float(eff_cap2), 1e-6)

        features = [
            1.0,
            float(prb1) / total_prb,
            float(prb2) / total_prb,
            (float(prb1) - float(prb2)) / total_prb,
            ratio_a1,
            mean1 / base_cap1,
            mean2 / base_cap2,
            shortfall1 / base_cap1,
            shortfall2 / base_cap2,
            float(obs.sigma1) / base_cap1,
            float(obs.sigma2) / base_cap2,
            1.0 if obs.slice2_active else 0.0,
            (float(prb1) - float(obs.current_prb1)) / total_prb,
            (float(prb2) - float(obs.current_prb2)) / total_prb,
            abs(float(prb1) - float(obs.current_prb1)) / total_prb,
            abs(float(prb2) - float(obs.current_prb2)) / total_prb,
        ]
        return features

    def predict(self, features: Sequence[float]) -> float:
        return sum(float(w) * float(f) for w, f in zip(self.weights, features))

    def confidence(self, features: Sequence[float]) -> float:
        return max(1e-6, sum(float(f) ** 2 for f in features))

    def update(self, features: Sequence[float], reward: float) -> None:
        if not features:
            return
        target = float(reward)
        pred = self.predict(features)
        error = target - pred
        denom = sum(float(f) ** 2 for f in features) + 1e-6
        factor = self._lr * error / denom
        for i, value in enumerate(features):
            self.weights[i] += factor * float(value)


def _extract_json_object(text: str) -> Optional[str]:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and i < j:
        return s[i : j + 1]
    return None


def _parse_tnas_candidates(text: str) -> Optional[List[Tuple[int, int]]]:
    """Parse TNAS output JSON object and return candidate (a1,a2) list."""

    s = text.strip()
    obj: object
    try:
        obj = json.loads(s)
    except Exception:
        blob = _extract_json_object(s)
        if not blob:
            return None
        try:
            obj = json.loads(blob)
        except Exception:
            return None

    candidates_obj: object = obj
    if isinstance(obj, dict) and "candidates" in obj:
        candidates_obj = obj["candidates"]
    if not isinstance(candidates_obj, list):
        return None

    out: List[Tuple[int, int]] = []
    for item in candidates_obj:
        if not isinstance(item, dict):
            continue
        if "a1" not in item or "a2" not in item:
            continue
        try:
            out.append(clamp_action(int(item["a1"]), int(item["a2"])))
        except Exception:
            continue
    return out or None


class TNASPolicy(Policy):
    """TNAS (Top-N Action Sampling) policy: sample candidates from LLM and re-rank locally."""

    name = "tnas"

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
        self.model = str(model)
        self.cache = cache
        self.seed = int(seed)
        self.temperature = float(cfg.Tem_max)
        self._last_cache_path: Optional[Path] = None
        self._debug = os.getenv("RAN_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        self._rng = random.Random(self.seed)
        self._critic: Optional[RealScoreCritic] = RealScoreCritic(cfg) if cfg.tnas_use_real_score else None
        self._pending_features: Optional[Sequence[float]] = None

    def reset(self) -> None:
        self._last_cache_path = None

    def _complete_cached(self, *, prompt: str, temperature: float) -> str:
        max_tokens = int(self.cfg.llm_max_tokens)
        cache_extra = {"seed": self.seed, "max_tokens": max_tokens}
        hit = self.cache.get(
            provider=self.client.provider,
            model=self.model,
            temperature=temperature,
            prompt=prompt,
            extra=cache_extra,
        )
        if hit is not None:
            self._last_cache_path = hit.cache_path
            return hit.response_text

        response = self.client.complete(
            prompt=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=self.cfg.llm_timeout_s,
            seed=self.seed,
        )
        cache_path = self.cache.put(
            provider=self.client.provider,
            model=self.model,
            temperature=temperature,
            prompt=prompt,
            response_text=response,
            extra=cache_extra,
        )
        self._last_cache_path = cache_path
        return response

    def select_action(self, obs: Observation) -> Tuple[int, int]:
        if not obs.slice2_active:
            return (128, 1)

        win1 = list(obs.recent_hat_sigma1)
        win2 = list(obs.recent_hat_sigma2)
        mean1 = sum(win1) / len(win1) if win1 else 0.0
        mean2 = sum(win2) / len(win2) if win2 else 0.0
        last1 = win1[-1] if win1 else 0.0
        last2 = win2[-1] if win2 else 0.0

        eff_cap1 = effective_cap_mbps(float(obs.sigma1), self.cfg.cap1_hard_mbps)
        eff_cap2 = effective_cap_mbps(float(obs.sigma2), self.cfg.cap2_hard_mbps)
        shortfall1 = max(0.0, float(eff_cap1) - float(mean1))
        shortfall2 = max(0.0, float(eff_cap2) - float(mean2))

        prompt_obs = PromptObservation(
            t=int(obs.t),
            sigma1=float(obs.sigma1),
            sigma2=float(obs.sigma2),
            slice2_active=bool(obs.slice2_active),
            mean_hat_sigma1=float(mean1),
            mean_hat_sigma2=float(mean2),
            last_hat_sigma1=float(last1),
            last_hat_sigma2=float(last2),
            Tem_k=float(self.temperature),
            cap1_hard_mbps=self.cfg.cap1_hard_mbps,
            cap2_hard_mbps=self.cfg.cap2_hard_mbps,
            eff_cap1_mbps=float(eff_cap1),
            eff_cap2_mbps=float(eff_cap2),
            shortfall1=float(shortfall1),
            shortfall2=float(shortfall2),
            current_prb1=int(max(0, obs.current_prb1)),
            current_prb2=int(max(0, obs.current_prb2)),
            soft_p=int(self.cfg.soft_p),
            lambda1=float(self.cfg.lambda1),
            lambda2=float(self.cfg.lambda2),
            soft_enable_time=int(self.cfg.soft_enable_time),
        )
        prompt = build_tnas_prompt(obs=prompt_obs, top_n=int(self.cfg.tnas_top_n))

        candidates: Optional[List[Tuple[int, int]]] = None
        response = ""
        try:
            response = self._complete_cached(prompt=prompt, temperature=self.temperature)
            candidates = _parse_tnas_candidates(response)
        except Exception as e:
            logger.warning("TNAS LLM call failed; fallback to proportional. err=%s", e)

        if candidates is None and self.cfg.llm_parse_retry > 0:
            try:
                repair = build_tnas_repair_prompt(bad_response=response, top_n=int(self.cfg.tnas_top_n))
                response2 = self._complete_cached(prompt=repair, temperature=0.0)
                candidates = _parse_tnas_candidates(response2)
            except Exception as e:
                logger.warning("TNAS LLM repair call failed; fallback to proportional. err=%s", e)

        if not candidates:
            snippet = (response or "").strip().replace("\n", "\\n")
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            logger.warning(
                "TNAS output parse failed or empty; fallback to proportional. cache=%s response=%s",
                str(self._last_cache_path) if self._last_cache_path else "N/A",
                snippet or "<empty>",
            )
            return clamp_action(int(round(obs.sigma1)), int(round(obs.sigma2)))

        # Enforce Top-N for logging and fair local re-ranking: pad/truncate deterministically.
        top_n = int(max(1, self.cfg.tnas_top_n))
        seed: List[Tuple[int, int]] = [
            clamp_action(int(round(obs.sigma1)), int(round(obs.sigma2))),  # proportional
            (64, 64),  # equal
            (124, 4),
            (120, 8),
            (112, 16),
            (128, 1),  # extreme UE1
            (1, 128),  # extreme UE2
        ]
        combined = list(candidates) + seed
        uniq: List[Tuple[int, int]] = []
        for a in combined:
            if a not in uniq:
                uniq.append(a)
            if len(uniq) >= top_n:
                break
        candidates = uniq[:top_n]

        scores: List[float] = []
        prbs_list: List[Tuple[int, int]] = []
        feature_list: List[Sequence[float]] = []
        for idx, action in enumerate(candidates):
            prb1_c, prb2_c = action_to_prbs(action, self.cfg.R_total)
            prbs_list.append((int(prb1_c), int(prb2_c)))
            if self._critic:
                feat = self._critic.build_features(obs, action, (prb1_c, prb2_c))
                feature_list.append(feat)
                score = self._critic.predict(feat)
            else:
                score, _extra = score_allocation_proxy(
                    self.cfg,
                    t=int(obs.t),
                    sigma1=float(obs.sigma1),
                    sigma2=float(obs.sigma2),
                    prb1=int(prb1_c),
                    prb2=int(prb2_c),
                )
            scores.append(float(score))

        low_confidence = False
        best_score = float("-inf")
        chosen_idx = 0
        best_idx = 0
        if self._critic:
            explore = self._rng.random() < self._critic.explore_prob
            if explore:
                chosen_idx = int(self._rng.randrange(len(candidates)))
            else:
                chosen_idx = int(max(range(len(candidates)), key=lambda i: scores[i]))
            best_idx = chosen_idx
            best_score = scores[chosen_idx]
            best_feat = feature_list[chosen_idx]
            best_action = candidates[chosen_idx]
            best_prbs = prbs_list[chosen_idx]
            confidence = self._critic.confidence(best_feat)
            if confidence < float(self.cfg.tnas_confidence_threshold):
                low_confidence = True
                fallback_action = clamp_action(int(round(obs.sigma1)), int(round(obs.sigma2)))
                fallback_prbs = action_to_prbs(fallback_action, self.cfg.R_total)
                fallback_feat = self._critic.build_features(obs, fallback_action, fallback_prbs)
                best_score = self._critic.predict(fallback_feat)
                best_action = fallback_action
                best_prbs = fallback_prbs
                self._pending_features = fallback_feat
                best_mode = "fallback"
            else:
                self._pending_features = best_feat
                best_mode = "real"
        else:
            chosen_idx = int(max(range(len(candidates)), key=lambda i: scores[i]))
            best_idx = chosen_idx
            best_score = scores[chosen_idx]
            best_action = candidates[chosen_idx]
            best_prbs = prbs_list[chosen_idx]
            self._pending_features = None
            best_mode = "proxy"

        logger.info(
            "TNAS t=%s candidates=%s chosen_idx=%s action=%s prbs=%s score=%.2f mode=%s",
            int(obs.t),
            len(candidates),
            best_idx,
            best_action,
            best_prbs,
            best_score,
            best_mode,
        )
        if self._debug:
            logger.info("TNAS scores=%s", [round(s, 2) for s in scores])

        return best_action

    def record_outcome(self, outcome: SlotOutcome) -> None:
        super().record_outcome(outcome)
        if self._critic and self._pending_features is not None:
            try:
                reward = float(outcome.V_k_soft)
            except Exception:
                reward = float("nan")
            self._critic.update(self._pending_features, reward)
            self._pending_features = None
