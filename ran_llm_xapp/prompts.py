from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class HistoryExample:
    """A compact history item for in-context learning."""

    k: int
    V: float
    a1: int
    a2: int
    prb1: int
    prb2: int
    mean_hat_sigma1: float
    mean_hat_sigma2: float


@dataclass(frozen=True)
class PromptObservation:
    t: int
    sigma1: float
    sigma2: Optional[float]
    slice2_active: bool
    mean_hat_sigma1: float
    mean_hat_sigma2: float
    last_hat_sigma1: float
    last_hat_sigma2: float
    Tem_k: float
    last_action_a1: Optional[int] = None
    last_action_a2: Optional[int] = None


def _format_history(examples: Sequence[HistoryExample]) -> str:
    if not examples:
        return "None (no history yet)."
    lines = []
    for ex in examples:
        lines.append(
            f"- k={ex.k}, V={ex.V:.3f}, action={{\"a1\":{ex.a1},\"a2\":{ex.a2}}}, "
            f"PRB=({ex.prb1},{ex.prb2}), mean_hat_sigma=(UE1={ex.mean_hat_sigma1:.2f}, UE2={ex.mean_hat_sigma2:.2f})"
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

    The policy expects the model to output STRICT JSON:
      {"a1": <int 1..128>, "a2": <int 1..128>}
    """

    history_text = _format_history(history_examples[:in_context_top_n])
    last_action_note = ""
    if require_new_action and obs.last_action_a1 is not None and obs.last_action_a2 is not None:
        last_action_note = (
            f"\n- You MUST output an action different from the previous action "
            f"{{\"a1\":{obs.last_action_a1},\"a2\":{obs.last_action_a2}}}."
        )

    sigma2_str = f"{obs.sigma2:.3f}" if obs.sigma2 is not None else "N/A"
    mean2_str = f"{obs.mean_hat_sigma2:.3f}" if obs.slice2_active else "N/A"
    last2_str = f"{obs.last_hat_sigma2:.3f}" if obs.slice2_active else "N/A"

    return f"""You are optimizing a 5G RAN slicing controller.

Goal:
- Propose the next action A_k = [a1, a2] to maximize the evaluation score V_k.
- Higher V_k is better (it penalizes squared rate mismatch via g(x) = -x^2 and adds large constants).

Action constraints:
- a1 and a2 are integers in [1, 128].
- Output MUST be STRICT JSON with keys exactly: a1, a2.
- Output ONLY the JSON object, no extra text, no markdown, no code fences.{last_action_note}

Current observation:
- time t = {obs.t}
- requested rates (Mbps): sigma1 = {obs.sigma1:.3f}, sigma2 = {sigma2_str}
- recent measured rate stats (Mbps):
  - UE1: mean={obs.mean_hat_sigma1:.3f}, last={obs.last_hat_sigma1:.3f}
  - UE2: mean={mean2_str}, last={last2_str}
- current temperature Tem_k = {obs.Tem_k:.3f}

Top history examples (sorted by V desc):
{history_text}

Now output the next action as STRICT JSON:
"""


def build_repair_prompt(*, bad_response: str) -> str:
    return f"""Your previous output was not valid JSON.

Rules:
- Output STRICT JSON ONLY (no explanation).
- Keys must be exactly: a1, a2.
- Values must be integers in [1, 128].

Bad output:
{bad_response}

Now output ONLY the corrected JSON:
"""

