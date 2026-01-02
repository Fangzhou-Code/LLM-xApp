from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .config import ExperimentConfig


def utility_s1_sigmoid(hat_sigma_mbps: float, sigma_mbps: float, *, a: float, b: float) -> float:
    """Utility for high-priority slice S1 (paper Eq.(1), sigmoid)."""

    return 1.0 / (1.0 + math.exp(-((a * (hat_sigma_mbps - sigma_mbps)) + b)))


def utility_s2_log_ratio(hat_sigma_mbps: float, sigma_mbps: float, *, c: float) -> float:
    """Utility for low-priority slice S2 (paper Eq.(2), log-ratio)."""

    return math.log(hat_sigma_mbps + c) / math.log(sigma_mbps + c)


def g_neg_square(x: float) -> float:
    """Default g(x) used in the paper experiments: g(x) = -x^2."""

    return -(x * x)


def get_g_function(cfg: ExperimentConfig) -> Callable[[float], float]:
    if cfg.g_name == "neg_square":
        return g_neg_square
    raise ValueError(f"Unknown g_name={cfg.g_name!r}")


def action_to_prbs(action: Tuple[int, int], R_total: int) -> Tuple[int, int]:
    """Map action A_k=[a1,a2] to PRBs r_s^k (paper Eq.(7) + budget fixer).

    Eq.(7):
      r_s^k = ceil( R_total * a_s^k / (a1^k + a2^k) )

    Fixer:
      If ceil makes r1+r2 exceed R_total, decrement PRBs from the currently largest
      slice one-by-one until the constraint is satisfied.
    """

    a1, a2 = action
    if not (isinstance(a1, int) and isinstance(a2, int)):
        raise TypeError("action must be a pair of ints")
    if a1 < 1 or a1 > 128 or a2 < 1 or a2 > 128:
        raise ValueError("a1,a2 must be integers in [1,128]")
    if R_total < 0:
        raise ValueError("R_total must be non-negative")

    denom = a1 + a2
    r1 = int(math.ceil(R_total * a1 / denom)) if R_total > 0 else 0
    r2 = int(math.ceil(R_total * a2 / denom)) if R_total > 0 else 0

    # Budget fixer to ensure r1 + r2 <= R_total
    while (r1 + r2) > R_total:
        if r1 >= r2 and r1 > 0:
            r1 -= 1
        elif r2 > 0:
            r2 -= 1
        else:
            break

    # Final guards (non-negative ints, sum constraint)
    r1 = max(0, int(r1))
    r2 = max(0, int(r2))
    if (r1 + r2) > R_total:
        raise RuntimeError("PRB budget fixer failed to satisfy sum constraint")
    return r1, r2


def compute_utilities(
    cfg: ExperimentConfig,
    *,
    hat_sigma1: float,
    hat_sigma2: float,
    sigma1: float,
    sigma2: float,
    slice2_active: bool,
) -> Tuple[float, float]:
    """Compute (u1,u2) for a single time t; u2 is NaN if S2 is inactive."""

    u1 = utility_s1_sigmoid(hat_sigma1, sigma1, a=cfg.a, b=cfg.b)
    if slice2_active:
        u2 = utility_s2_log_ratio(hat_sigma2, sigma2, c=cfg.c)
    else:
        u2 = float("nan")
    return u1, u2


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def reliability_outage_fraction(
    u_series: Sequence[float], *, threshold: float, Tw: int
) -> List[float]:
    """Compute reliability θ_s^t as 'utility-below-threshold ratio' in a sliding window.

    Paper definition:
      θ_s^t = ( sum_{τ=t-Tw/2}^{t+Tw/2} 1[ u_s^τ <= u_th_s ] ) / Tw

    Boundary handling (explicit and fixed in this implementation):
    - We use a window of *nominal length* Tw centered at t:
        left = floor(Tw/2), right = Tw - left
        window indices = [t-left, t+right)  (end-exclusive)
      so the window length is exactly Tw when fully inside bounds.
    - When the window exceeds available indices, we truncate to valid indices and
      divide by the *effective* number of valid samples (not by Tw).
    - NaN/Inf utilities are ignored (and also reduce the effective divisor).
    """

    if Tw <= 0:
        raise ValueError("Tw must be positive")
    n = len(u_series)
    left = Tw // 2
    right = Tw - left

    out: List[float] = []
    for t in range(n):
        start = max(0, t - left)
        end = min(n, t + right)
        window = [u for u in u_series[start:end] if _is_finite(u)]
        if not window:
            out.append(float("nan"))
            continue
        bad = sum(1 for u in window if u <= threshold)
        out.append(bad / len(window))
    return out


def system_average(u1: float, u2: float, *, slice2_active: bool) -> float:
    """System metric: simple average over existing slices (unweighted)."""

    if slice2_active and _is_finite(u2):
        return 0.5 * (u1 + u2)
    return u1


def moving_average_trailing(series: Sequence[float], window: int) -> List[float]:
    """Trailing moving average with boundary truncation; ignores NaN/Inf values."""

    if window <= 0:
        raise ValueError("window must be positive")
    out: List[float] = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        chunk = [x for x in series[start : i + 1] if _is_finite(x)]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def evaluate_V_k(
    cfg: ExperimentConfig,
    *,
    t_k: int,
    hat_sigma1_series: Sequence[float],
    hat_sigma2_series: Sequence[float],
) -> float:
    """Compute V_k(o_k, A_k) (paper Eq.(8)) at a given slot-end time index t_k.

    Eq.(8):
      V_k = sum_{s∈S} [ β_s * g( (2/Tw) * sum_{t=t_k - Tw/2}^{t_k} (hatσ_s^t - σ_s^{t_k}) ) + γ_s ]

    Notes:
    - We interpret `t_k` as the *slot end* time index where Eq.(8) is evaluated.
    - Window boundary is truncated to valid indices.
    - S2 is included only when active at time t_k.
    """

    g = get_g_function(cfg)
    half = cfg.Tw // 2
    start = max(0, t_k - half)
    end = min(len(hat_sigma1_series) - 1, t_k)

    # Requested rates at time t_k
    sigma1_tk = cfg.sigma1
    sigma2_tk = cfg.sigma2
    slice2_active = t_k >= cfg.slice_init_time

    sum1 = 0.0
    sum2 = 0.0
    for t in range(start, end + 1):
        sum1 += hat_sigma1_series[t] - sigma1_tk
        if slice2_active:
            sum2 += hat_sigma2_series[t] - sigma2_tk

    x1 = (2.0 / cfg.Tw) * sum1
    V = (cfg.beta1 * g(x1)) + cfg.gamma1
    if slice2_active:
        x2 = (2.0 / cfg.Tw) * sum2
        V += (cfg.beta2 * g(x2)) + cfg.gamma2
    return V


@dataclass(frozen=True)
class TimeAverages:
    avg_u1: float
    avg_u2: float
    avg_sys_u: float
    avg_theta1: float
    avg_theta2: float
    avg_sys_theta: float


def _mean_ignore_nan(xs: Iterable[float]) -> float:
    vals = [x for x in xs if _is_finite(x)]
    return sum(vals) / len(vals) if vals else float("nan")


def compute_time_averages(
    cfg: ExperimentConfig,
    *,
    u1: Sequence[float],
    u2: Sequence[float],
    theta1: Sequence[float],
    theta2: Sequence[float],
    sys_u: Sequence[float],
    sys_theta: Sequence[float],
    start_t: int,
) -> TimeAverages:
    """Compute time-averaged metrics over t ∈ [start_t, T_end]."""

    start_idx = max(0, int(start_t // cfg.dt))
    return TimeAverages(
        avg_u1=_mean_ignore_nan(u1[start_idx:]),
        avg_u2=_mean_ignore_nan(u2[start_idx:]),
        avg_sys_u=_mean_ignore_nan(sys_u[start_idx:]),
        avg_theta1=_mean_ignore_nan(theta1[start_idx:]),
        avg_theta2=_mean_ignore_nan(theta2[start_idx:]),
        avg_sys_theta=_mean_ignore_nan(sys_theta[start_idx:]),
    )

