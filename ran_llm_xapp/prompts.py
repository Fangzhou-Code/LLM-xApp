from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class HistoryExample:
    """A compact history item for in-context learning."""

    k: int
    V_k_soft: float
    a1: int
    a2: int
    prb1: int
    prb2: int
    sigma1: float
    sigma2: float
    eff_cap1: float
    eff_cap2: float
    mean_hat_sigma1: float
    mean_hat_sigma2: float
    shortfall1: float
    shortfall2: float
    penalty: float


@dataclass(frozen=True)
class PromptObservation:
    t: int
    sigma1: float
    sigma2: float
    slice2_active: bool
    mean_hat_sigma1: float
    mean_hat_sigma2: float
    last_hat_sigma1: float
    last_hat_sigma2: float
    Tem_k: float
    cap1_hard_mbps: Optional[float]
    cap2_hard_mbps: Optional[float]
    eff_cap1_mbps: float
    eff_cap2_mbps: float
    shortfall1: float
    shortfall2: float
    current_prb1: int
    current_prb2: int
    soft_p: int
    lambda1: float
    lambda2: float
    soft_enable_time: int
    last_action_a1: Optional[int] = None
    last_action_a2: Optional[int] = None


def _format_history(examples: Sequence[HistoryExample]) -> str:
    if not examples:
        return "None (no history yet)."
    lines = []
    for ex in examples:
        lines.append(
            f"- k={ex.k}, V_k_soft={ex.V_k_soft:.2f}, action={{\"a1\":{ex.a1},\"a2\":{ex.a2}}}, "
            f"PRB=({ex.prb1},{ex.prb2}), sigma=({ex.sigma1:.1f},{ex.sigma2:.1f}), "
            f"eff_cap=({ex.eff_cap1:.1f},{ex.eff_cap2:.1f}), shortfall=({ex.shortfall1:.2f},{ex.shortfall2:.2f}), "
            f"penalty={ex.penalty:.1f}, mean_hat_sigma=(UE1={ex.mean_hat_sigma1:.2f}, UE2={ex.mean_hat_sigma2:.2f})"
        )
    return "\n".join(lines)


def build_meta_prompt(
    *,
    obs: PromptObservation,
    history_examples: Sequence[HistoryExample],
    in_context_top_n: int,
    require_new_action: bool = True,
) -> str:
    """Build the OPRO meta-prompt (Algorithm 1 style).

    The policy expects the model to output a STRICT JSON list of candidates:
      [{"a1": <int 1..128>, "a2": <int 1..128>}, ...]  (exactly 6 items)
    """

    history_text = _format_history(history_examples[:in_context_top_n])
    last_action_note = ""
    if require_new_action and obs.last_action_a1 is not None and obs.last_action_a2 is not None:
        last_action_note = (
            f"\n- Do NOT output the same action 6 times. At least one candidate MUST differ from the previous action "
            f"{{\"a1\":{obs.last_action_a1},\"a2\":{obs.last_action_a2}}}."
        )

    cap1_str = "None" if obs.cap1_hard_mbps is None else f"{obs.cap1_hard_mbps:.3f}"
    cap2_str = "None" if obs.cap2_hard_mbps is None else f"{obs.cap2_hard_mbps:.3f}"

    return f"""You are optimizing a 5G RAN slicing controller.

Goal:
- Propose MULTIPLE candidate actions for the next slot and let the controller pick the best.
- The controller maximizes a soft objective V_k_soft = V_k + penalty. Higher V_k_soft is better.
  - V_k is the paper Eq.(8) score with g(x) = -x^2 (squared mismatch penalty + constants).
  - penalty is a soft shortfall penalty (NOT a hard rule):
      eff_cap_s = min(sigma_s, cap_s_hard)  (cap_s_hard=None means +inf)
      shortfall_s = max(0, eff_cap_s - hat_sigma_s)
      penalty = -lambda1 * (shortfall1^p) - lambda2 * (shortfall2^p)
    Notes:
    - When the total demand is feasible, prefer meeting BOTH slices (shortfalls near 0).
    - When infeasible (e.g., sigma1 increases), allow small UE2 shortfall to reduce UE1 shortfall,
      but the trade-off is governed by lambda1/lambda2 and p (no hard-coded rule).

Action constraints:
- a1 and a2 are integers in [1, 128].
- Output MUST be STRICT JSON LIST of exactly 6 objects. Each object must have keys exactly: a1, a2.
- Output ONLY the JSON list, no extra text, no markdown, no code fences.{last_action_note}
- Candidate diversity requirement:
  - Provide 6 diverse candidates.
  - At least 3 candidates must be "aggressive toward UE1" with small a2 in {{4,8,12,16}}.

Current observation:
- time t = {obs.t}
- requested rates (Mbps): sigma1 = {obs.sigma1:.3f}, sigma2 = {obs.sigma2:.3f}
- hard caps (Mbps): cap1_hard = {cap1_str}, cap2_hard = {cap2_str} (None means +inf)
- effective targets (Mbps): eff_cap1 = {obs.eff_cap1_mbps:.3f}, eff_cap2 = {obs.eff_cap2_mbps:.3f}
- current PRB allocation: prb1 = {obs.current_prb1}, prb2 = {obs.current_prb2}
- recent measured rate stats (Mbps):
  - UE1: mean={obs.mean_hat_sigma1:.3f}, last={obs.last_hat_sigma1:.3f}
  - UE2: mean={obs.mean_hat_sigma2:.3f}, last={obs.last_hat_sigma2:.3f}
- current shortfalls (Mbps): shortfall1={obs.shortfall1:.3f}, shortfall2={obs.shortfall2:.3f}
- soft-penalty params: p={obs.soft_p}, lambda1={obs.lambda1:.3f}, lambda2={obs.lambda2:.3f}
- soft penalty enabled for t >= {obs.soft_enable_time}
- current temperature Tem_k = {obs.Tem_k:.3f}

Top history examples (sorted by V_k_soft desc; best -> worst):
{history_text}

Now output ONLY the JSON list of 6 candidate actions:
"""


def build_repair_prompt(*, bad_response: str) -> str:
    return f"""Your previous output was not valid JSON (expected a JSON LIST of 6 action objects).

Rules:
- Output STRICT JSON LIST ONLY (no explanation).
- The list must contain exactly 6 items.
- Each item must be an object with keys exactly: a1, a2.
- a1/a2 values must be integers in [1, 128].
- At least 3 items must have a2 in {{4,8,12,16}}.

Bad output:
{bad_response}

Now output ONLY the corrected JSON list:
"""


def build_tnas_prompt(*, obs: PromptObservation, top_n: int) -> str:
    """Build TNAS prompt: ask model for Top-N diverse candidates, no history/iteration."""

    cap1_str = "None" if obs.cap1_hard_mbps is None else f"{obs.cap1_hard_mbps:.3f}"
    cap2_str = "None" if obs.cap2_hard_mbps is None else f"{obs.cap2_hard_mbps:.3f}"
    n = int(max(1, top_n))

    return f"""You are generating candidate actions for a 5G RAN slicing controller (TNAS: Top-N Action Sampling).

The controller will evaluate each candidate locally and execute the one with the highest score.
Higher score is better. The score is based on:
- Paper Eq.(8) V_k (with g(x)=-x^2), and
- A soft shortfall penalty (enabled for t >= {obs.soft_enable_time}):
    eff_cap_s = min(sigma_s, cap_s_hard)  (cap_s_hard=None means +inf)
    shortfall_s = max(0, eff_cap_s - hat_sigma_s)
    penalty = -lambda1 * (shortfall1^p) - lambda2 * (shortfall2^p)
    V_k_soft = V_k + penalty

Your job:
- Output {n} DIVERSE candidate actions so the controller can pick a strong one.
- Cover different allocation styles: UE1-prioritize, UE2-prioritize, balanced, and extreme edge cases.
- Keep each reason VERY short (<= 6 words).

Action constraints:
- Each candidate must be an object: {{\"a1\": int, \"a2\": int, \"reason\": \"...\"}}
- a1 and a2 must be integers in [1, 128].
- Output MUST be STRICT JSON OBJECT with exactly one key \"candidates\".
- \"candidates\" must be a JSON list with exactly {n} items.
- Output ONLY JSON. No extra text, no markdown, no code fences.

Current observation:
- time t = {obs.t}
- requested rates (Mbps): sigma1 = {obs.sigma1:.3f}, sigma2 = {obs.sigma2:.3f}
- hard caps (Mbps): cap1_hard = {cap1_str}, cap2_hard = {cap2_str}
- effective targets (Mbps): eff_cap1 = {obs.eff_cap1_mbps:.3f}, eff_cap2 = {obs.eff_cap2_mbps:.3f}
- recent measured rate stats (Mbps):
  - UE1: mean={obs.mean_hat_sigma1:.3f}, last={obs.last_hat_sigma1:.3f}
  - UE2: mean={obs.mean_hat_sigma2:.3f}, last={obs.last_hat_sigma2:.3f}
- current PRB allocation: prb1 = {obs.current_prb1}, prb2 = {obs.current_prb2}
- current shortfalls (Mbps): shortfall1={obs.shortfall1:.3f}, shortfall2={obs.shortfall2:.3f}
- soft-penalty params: p={obs.soft_p}, lambda1={obs.lambda1:.3f}, lambda2={obs.lambda2:.3f}
- temperature = {obs.Tem_k:.3f}

Now output ONLY the JSON object:
"""


def build_tnas_repair_prompt(*, bad_response: str, top_n: int) -> str:
    n = int(max(1, top_n))
    return f"""Your previous output was not valid JSON for TNAS.

Rules:
- Output STRICT JSON OBJECT ONLY.
- Must be exactly: {{\"candidates\": [ ... ]}}
- \"candidates\" must contain exactly {n} items.
- Each item must have keys exactly: a1, a2, reason.
- a1/a2 values must be integers in [1, 128].
- No extra text.

Bad output:
{bad_response}

Now output ONLY the corrected JSON object:
"""
