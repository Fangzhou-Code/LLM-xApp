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


def effective_cap_mbps(sigma_mbps: float, hard_cap_mbps: Optional[float]) -> float:
    """Return effective target cap: min(demand, hard_cap); hard_cap=None means +inf."""

    if hard_cap_mbps is None:
        return float(sigma_mbps)
    return float(min(sigma_mbps, hard_cap_mbps))


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
    sigma1_tk: Optional[float] = None,
    sigma2_tk: Optional[float] = None,
) -> float:
    """Compute V_k(o_k, A_k) (paper Eq.(8)) at a given slot-end time index t_k.

    Eq.(8):
      V_k = sum_{s∈S} [ β_s * g( (2/Tw) * sum_{t=t_k - Tw/2}^{t_k} (hatσ_s^t - σ_s^{t_k}) ) + γ_s ]

    Notes:
    - We interpret `t_k` as the *slot end* time index where Eq.(8) is evaluated.
    - Window boundary is truncated to valid indices.
    - This synthetic setup treats both slices as always present (S1+S2).
    """

    g = get_g_function(cfg)
    half = cfg.Tw // 2
    start = max(0, t_k - half)
    end = min(len(hat_sigma1_series) - 1, t_k)

    # Requested rates at time t_k (support time-varying sigma via explicit params).
    sigma1_tk = float(cfg.sigma1) if sigma1_tk is None else float(sigma1_tk)
    sigma2_tk = float(cfg.sigma2) if sigma2_tk is None else float(sigma2_tk)

    sum1 = 0.0
    sum2 = 0.0
    for t in range(start, end + 1):
        sum1 += hat_sigma1_series[t] - sigma1_tk
        sum2 += hat_sigma2_series[t] - sigma2_tk

    x1 = (2.0 / cfg.Tw) * sum1
    V = (cfg.beta1 * g(x1)) + cfg.gamma1
    x2 = (2.0 / cfg.Tw) * sum2
    V += (cfg.beta2 * g(x2)) + cfg.gamma2
    return V


@dataclass(frozen=True)
class SoftObjective:
    """Soft objective for OPRO: V_k plus a configurable shortfall penalty (post-enable time)."""

    V_k: float
    eff_cap1: float
    eff_cap2: float
    shortfall1: float
    shortfall2: float
    penalty: float
    V_k_soft: float
    # Backward-compat debug fields (kept in CSV; not used by V_k_soft here).
    prb2_min_est: int = 0
    waste: int = 0


def compute_prb2_waste_penalty(
    cfg: ExperimentConfig,
    *,
    mean_hat_sigma2: float,
    prb2: int,
) -> Tuple[int, int, float]:
    """Compute (prb2_min_est, waste, penalty) for UE2 PRB over-allocation.

    This is a *soft* penalty only; it is not a hard constraint.

    Spec:
      eff_cap2 = min(sigma2, cap2_hard)      (cap2_hard=None -> +inf)
      eff2_est = mean_hat_sigma2 / max(prb2, 1)
      prb2_min_est = ceil(eff_cap2 / max(eff2_est, eps))
      waste = max(0, prb2 - prb2_min_est)
      penalty = -lambda_waste * waste^2

    Implementation note (stability):
    - When UE2 is capped near its effective target, mean_hat_sigma2 stops growing with PRBs,
      making eff2_est artificially small and prb2_min_est unrealistically large. Also, correlated
      noise can temporarily make mean_hat_sigma2 appear too high, which would underestimate the
      required PRBs. For a stable "soft" penalty in this synthetic environment, we use the
      configured nominal efficiency `cfg.eff2_mbps_per_prb` as the efficiency estimate.
    """

    eff_cap2 = effective_cap_mbps(cfg.sigma2, cfg.cap2_hard_mbps)
    prb2_i = max(0, int(prb2))
    eff2_est = max(float(cfg.eff2_mbps_per_prb), float(cfg.waste_eps))
    prb2_min_est = int(math.ceil(eff_cap2 / max(eff2_est, float(cfg.waste_eps)))) if eff_cap2 > 0 else 0
    prb2_min_est = max(0, prb2_min_est)
    waste = max(0, prb2_i - prb2_min_est)
    penalty = -float(cfg.lambda_waste) * float(waste * waste)
    return prb2_min_est, waste, penalty


def evaluate_V_k_soft(
    cfg: ExperimentConfig,
    *,
    t_k: int,
    hat_sigma1_series: Sequence[float],
    hat_sigma2_series: Sequence[float],
    sigma1_tk: float,
    sigma2_tk: float,
    mean_hat_sigma1: float,
    mean_hat_sigma2: float,
) -> SoftObjective:
    """Compute soft score: V_k_soft = V_k - λ1*shortfall1^p - λ2*shortfall2^p.

    Where for each slice s:
      eff_cap_s^k = min(sigma_s^k, C_s)         (C_s is hard cap; None => +inf)
      shortfall_s^k = max(0, eff_cap_s^k - hat_sigma_s^k)

    This penalty is enabled only for t_k >= cfg.soft_enable_time (default=baseline_start_time).
    """

    V_k = evaluate_V_k(
        cfg,
        t_k=t_k,
        hat_sigma1_series=hat_sigma1_series,
        hat_sigma2_series=hat_sigma2_series,
        sigma1_tk=float(sigma1_tk),
        sigma2_tk=float(sigma2_tk),
    )

    eff_cap1 = effective_cap_mbps(float(sigma1_tk), cfg.cap1_hard_mbps)
    eff_cap2 = effective_cap_mbps(float(sigma2_tk), cfg.cap2_hard_mbps)
    shortfall1 = max(0.0, float(eff_cap1) - float(mean_hat_sigma1))
    shortfall2 = max(0.0, float(eff_cap2) - float(mean_hat_sigma2))

    penalty = 0.0
    if bool(cfg.use_soft_score) and int(t_k) >= int(cfg.soft_enable_time):
        p = int(cfg.soft_p)
        if p <= 0:
            raise ValueError("soft_p must be positive")
        penalty = -(
            float(cfg.lambda1) * float(shortfall1**p)
            + float(cfg.lambda2) * float(shortfall2**p)
        )

    return SoftObjective(
        V_k=float(V_k),
        eff_cap1=float(eff_cap1),
        eff_cap2=float(eff_cap2),
        shortfall1=float(shortfall1),
        shortfall2=float(shortfall2),
        penalty=float(penalty),
        V_k_soft=float(V_k + penalty),
        prb2_min_est=0,
        waste=0,
    )


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
