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
    mean_hat_sigma1: float
    mean_hat_sigma2: float
    prb2_min_est: int
    waste: int
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
    eff_cap2_mbps: float
    current_prb1: int
    current_prb2: int
    prb2_min_est: int
    lambda_waste: float
    last_action_a1: Optional[int] = None
    last_action_a2: Optional[int] = None


def _format_history(examples: Sequence[HistoryExample]) -> str:
    if not examples:
        return "None (no history yet)."
    lines = []
    for ex in examples:
        lines.append(
            f"- k={ex.k}, V_k_soft={ex.V_k_soft:.2f}, action={{\"a1\":{ex.a1},\"a2\":{ex.a2}}}, "
            f"PRB=({ex.prb1},{ex.prb2}), prb2_min_est={ex.prb2_min_est}, waste={ex.waste}, penalty={ex.penalty:.1f}, "
            f"mean_hat_sigma=(UE1={ex.mean_hat_sigma1:.2f}, UE2={ex.mean_hat_sigma2:.2f})"
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
  - penalty discourages wasting PRBs on UE2 after it is already near its effective target:
      prb2_min_est = ceil(eff_cap2 / max(eff2_est, eps))
      waste = max(0, prb2 - prb2_min_est)
      penalty = -lambda_waste * waste^2
    This is a soft preference (NOT a hard rule), but it makes UE2 diminishing returns explicit.

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
- UE2 effective target: eff_cap2 = min(sigma2, cap2_hard) = {obs.eff_cap2_mbps:.3f} Mbps
- current PRB allocation: prb1 = {obs.current_prb1}, prb2 = {obs.current_prb2}
- recent measured rate stats (Mbps):
  - UE1: mean={obs.mean_hat_sigma1:.3f}, last={obs.last_hat_sigma1:.3f}
  - UE2: mean={obs.mean_hat_sigma2:.3f}, last={obs.last_hat_sigma2:.3f}
- prb2_min_est (soft estimate) = {obs.prb2_min_est}
- lambda_waste = {obs.lambda_waste:.3f}
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
