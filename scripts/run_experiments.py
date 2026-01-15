from __future__ import annotations

import sys
from pathlib import Path as _Path

# Allow running as a script: `python scripts/run_experiments.py`
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from ran_llm_xapp.config import ExperimentConfig, save_config_prefer_yaml
from ran_llm_xapp.env import SyntheticRANSliceEnv
from ran_llm_xapp.io_utils import PromptResponseCache, ensure_dir, load_dotenv, write_csv
from ran_llm_xapp.llm_clients import DeepSeekClient, OpenAIClient, StubLLMClient, GoogleClient
from ran_llm_xapp.metrics import (
    action_to_prbs,
    compute_severity_weighted_reliability_at_t,
    compute_time_averages,
    compute_utilities,
    effective_cap_mbps,
    evaluate_V_k_soft,
    outage_theta_fraction,
    reliability_from_outage_series,
    system_average,
    system_utility_weight,
    system_utility_weighted,
    system_reliability_severity,
)
from ran_llm_xapp.plotting import plot_fig4_grid, plot_fig4_single, plot_fig5_bars, plot_fig5_sys_curve
from ran_llm_xapp.policies import (
    EqualPolicy,
    Observation,
    ProportionalPolicy,
    RandomPolicy,
    SlotOutcome,
    TNASPolicy,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_experiments")


METHODS_ALL = ("equal", "random", "proportional")
# Methods considered valid when explicitly provided (e.g., --methods tnas).
METHODS_VALID = tuple(list(METHODS_ALL) + ["tnas"])


def _parse_methods(raw: Sequence[str]) -> List[str]:
    if len(raw) == 1 and raw[0].lower() in {"all", "all2"}:
        if raw[0].lower() == "all2":
            return list(METHODS_VALID)
        return list(METHODS_ALL)
    methods = [m.lower() for m in raw]
    unknown = [m for m in methods if m not in METHODS_VALID]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Valid: {METHODS_VALID} or 'all'/'all2'")
    # de-dup while preserving order
    out: List[str] = []
    for m in methods:
        if m not in out:
            out.append(m)
    return out


def _mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _default_model_for_provider(provider: str) -> str:
    provider = provider.lower()
    if provider == "openai":
        return "gpt-4o-mini"
    if provider == "deepseek":
        return "deepseek-chat"
    if provider == "google":
        return "gemini-3-pro"
    return "stub"


_PRETTY_METHOD_LABELS = {
    "equal": "Equal",
    "random": "Random",
    "proportional": "Proportional",
}


_PRETTY_MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "deepseek-v3.2": "DeepSeek-v3.2",
    "gemini-2.0-flash": "Gemini-2.0-Flash",
}


def _pretty_method_label(name: str) -> str:
    return _PRETTY_METHOD_LABELS.get(str(name).lower(), name)


def _pretty_model_label(name: str) -> str:
    key = str(name).strip().lower()
    return _PRETTY_MODEL_LABELS.get(key, name)


def _parse_llm_runs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """Parse `--llm-runs` into [(provider, model), ...]."""

    provider_default = str(args.provider).lower()
    model_default = str(args.model) if args.model else _default_model_for_provider(provider_default)

    llm_runs_raw = getattr(args, "llm_runs", None)
    if not llm_runs_raw:
        return [(provider_default, model_default)]

    runs: List[Tuple[str, str]] = []
    warned_openao = False
    for spec in llm_runs_raw:
        spec = str(spec).strip()
        if not spec:
            continue
        if ":" in spec:
            provider, model = spec.split(":", 1)
            provider = provider.strip().lower()
            model = model.strip()
            if not model:
                model = _default_model_for_provider(provider)
        else:
            provider = provider_default
            model = spec

        if provider == "openao":
            provider = "openai"
            warned_openao = True

        if provider not in {"openai", "deepseek", "google", "stub"}:
            raise SystemExit(f"Unknown provider in --llm-runs: {provider!r}")
        if not model:
            raise SystemExit("Model name cannot be empty in --llm-runs.")
        runs.append((provider, model))

    # de-dup while preserving order
    out: List[Tuple[str, str]] = []
    for r in runs:
        if r not in out:
            out.append(r)
    if warned_openao:
        logger.warning("Provider 'openao' was treated as 'openai' (typo tolerance).")
    return out


def _estimate_prb_need(*, sigma_mbps: float, cap_hard_mbps: float | None, eff_mbps_per_prb: float) -> int:
    eff_cap = effective_cap_mbps(float(sigma_mbps), cap_hard_mbps)
    eff = float(eff_mbps_per_prb)
    if eff_cap <= 0:
        return 0
    if eff <= 0:
        return 10**9
    return int(math.ceil(eff_cap / eff))


def _warn_if_schedule_not_sane(cfg: ExperimentConfig) -> None:
    """Light sanity check to ensure stageA is feasible and stageB is infeasible (with margin)."""

    # Stage A starts at baseline_start_time; stage B starts at the next schedule change.
    starts = sorted({int(s[0]) for s in (cfg.demand_schedule or []) if isinstance(s, (list, tuple)) and len(s) >= 3})
    stage_a_t = int(cfg.baseline_start_time)
    stage_b_t: int | None = None
    for t0 in starts:
        if t0 > stage_a_t:
            stage_b_t = int(t0)
            break
    if stage_b_t is None:
        logger.warning("Demand schedule sanity check skipped (no stage-B change point found).")
        return

    margin = int(getattr(cfg, "schedule_margin_prb", 8))

    s1a, s2a = cfg.sigma_at(stage_a_t)
    s1b, s2b = cfg.sigma_at(stage_b_t)

    need1a = _estimate_prb_need(sigma_mbps=s1a, cap_hard_mbps=cfg.cap1_hard_mbps, eff_mbps_per_prb=cfg.eff1_mbps_per_prb)
    need2a = _estimate_prb_need(sigma_mbps=s2a, cap_hard_mbps=cfg.cap2_hard_mbps, eff_mbps_per_prb=cfg.eff2_mbps_per_prb)
    need1b = _estimate_prb_need(sigma_mbps=s1b, cap_hard_mbps=cfg.cap1_hard_mbps, eff_mbps_per_prb=cfg.eff1_mbps_per_prb)
    need2b = _estimate_prb_need(sigma_mbps=s2b, cap_hard_mbps=cfg.cap2_hard_mbps, eff_mbps_per_prb=cfg.eff2_mbps_per_prb)

    total = int(cfg.R_total)
    if (need1a + need2a) > (total - margin):
        logger.warning(
            "Demand schedule stageA may be infeasible: t=%s sigma=(%.1f,%.1f) prb_need=(%s,%s) R_total=%s margin=%s",
            stage_a_t,
            s1a,
            s2a,
            need1a,
            need2a,
            total,
            margin,
        )
    if (need1b + need2b) < (total + margin):
        logger.warning(
            "Demand schedule stageB may be feasible: t=%s sigma=(%.1f,%.1f) prb_need=(%s,%s) R_total=%s margin=%s",
            stage_b_t,
            s1b,
            s2b,
            need1b,
            need2b,
            total,
            margin,
        )

    logger.info(
        "Demand schedule check: stageA t=%s sigma=(%.1f,%.1f) needPRB=(%s,%s); stageB t=%s sigma=(%.1f,%.1f) needPRB=(%s,%s); R_total=%s",
        stage_a_t,
        s1a,
        s2a,
        need1a,
        need2a,
        stage_b_t,
        s1b,
        s2b,
        need1b,
        need2b,
        total,
    )


def _build_policy(
    method: str,
    *,
    cfg: ExperimentConfig,
    seed: int,
    llm_provider: str,
    llm_model: str,
    cache: PromptResponseCache,
) -> object:
    if method == "equal":
        return EqualPolicy()
    if method == "proportional":
        return ProportionalPolicy()
    if method == "random":
        # Policy RNG should not consume the env RNG sequence.
        return RandomPolicy(seed=seed + 12345)
    if method == "tnas":
        if llm_provider == "openai":
            client = OpenAIClient()
            if (not client.api_key) or (not getattr(client, "base_url", None)):
                logger.warning("OPENAI_API_KEY/OPENAI_BASE_URL missing; falling back to stub provider.")
                client = StubLLMClient(seed=seed + 222)
        elif llm_provider == "deepseek":
            client = DeepSeekClient()
            if (not client.api_key) or (not getattr(client, "base_url", None)):
                logger.warning("DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL missing; falling back to stub provider.")
                client = StubLLMClient(seed=seed + 333)
        elif llm_provider == "google":
            client = GoogleClient()
            if (not client.api_key) or (not getattr(client, "base_url", None)):
                logger.warning("GOOGLE_API_KEY/GOOGLE_BASE_URL missing; falling back to stub provider.")
                client = StubLLMClient(seed=seed + 444)
        else:
            client = StubLLMClient(seed=seed + 444)
        return TNASPolicy(cfg=cfg, llm_client=client, model=llm_model, cache=cache, seed=seed + 555)
    raise RuntimeError(f"Unhandled method={method}")


def run_single_method(
    *,
    method: str,
    cfg: ExperimentConfig,
    seed: int,
    llm_provider: str,
    llm_model: str,
    cache: PromptResponseCache,
) -> Dict[str, List[float]]:
    env = SyntheticRANSliceEnv(cfg, seed=seed)
    env.reset()

    policy = _build_policy(method, cfg=cfg, seed=seed, llm_provider=llm_provider, llm_model=llm_model, cache=cache)
    if hasattr(policy, "reset"):
        policy.reset()

    times = list(range(0, cfg.T_end + 1, cfg.dt))

    prb1_series: List[int] = []
    prb2_series: List[int] = []
    hat1_series: List[float] = []
    hat2_series: List[float] = []
    u1_series: List[float] = []
    u2_series: List[float] = []
    slice2_active_series: List[bool] = []
    sigma1_series: List[float] = []
    sigma2_series: List[float] = []
    eff_cap1_series: List[float] = []
    eff_cap2_series: List[float] = []
    shortfall1_series: List[float] = []
    shortfall2_series: List[float] = []
    prb2_min_est_series: List[float] = []
    waste_series: List[float] = []
    penalty_series: List[float] = []
    V_k_soft_series: List[float] = []

    # Track slot state for V_k + OPRO history update.
    slot_k = 0
    slot_start_t = times[0]
    current_action: Tuple[int, int] = (128, 1)
    current_prbs: Tuple[int, int] = (cfg.pre_slice_prb1, cfg.pre_slice_prb2)

    def _recent_window(series: Sequence[float], t: int, window: int) -> List[float]:
        start = max(0, t - window + 1)
        return list(series[start : t + 1])

    for t in times:
        slice2_active = True
        slice2_active_series.append(True)
        sigma1_t, sigma2_t = cfg.sigma_at(t)
        sigma1_series.append(float(sigma1_t))
        sigma2_series.append(float(sigma2_t))
        eff_cap1_series.append(float(effective_cap_mbps(float(sigma1_t), cfg.cap1_hard_mbps)))
        eff_cap2_series.append(float(effective_cap_mbps(float(sigma2_t), cfg.cap2_hard_mbps)))
        shortfall1_series.append(float("nan"))
        shortfall2_series.append(float("nan"))
        prb2_min_est_series.append(float("nan"))
        waste_series.append(float("nan"))
        penalty_series.append(float("nan"))
        V_k_soft_series.append(float("nan"))

        # Reconfig slot: decide action at fixed interval boundaries.
        if (t == times[0]) or (t % cfg.reconfig_interval == 0):
            # Three-stage allocation timeline:
            # (i)  0~slice_init_time: fixed PRB split to match ~30/~10 Mbps
            # (ii) slice_init_time~baseline_start_time: init evenly split (64/64)
            # (iii) baseline_start_time~end: method policy takes effect (equal starts at slice init)
            if t < cfg.slice_init_time:
                current_action = (int(cfg.pre_slice_prb1), int(cfg.pre_slice_prb2))
                current_prbs = (int(cfg.pre_slice_prb1), int(cfg.pre_slice_prb2))
            elif t < cfg.baseline_start_time:
                current_action = (64, 64)
                current_prbs = action_to_prbs(current_action, cfg.R_total)
            elif method == "equal":
                current_action = (64, 64)
                current_prbs = action_to_prbs(current_action, cfg.R_total)
            else:
                # Observation uses recent *measured* window (trailing, not centered).
                win1 = _recent_window(hat1_series, len(hat1_series) - 1, cfg.Tw) if hat1_series else []
                win2 = _recent_window(hat2_series, len(hat2_series) - 1, cfg.Tw) if hat2_series else []
                obs = Observation(
                    t=t,
                    sigma1=float(sigma1_t),
                    sigma2=float(sigma2_t),
                    slice2_active=True,
                    current_prb1=int(current_prbs[0]),
                    current_prb2=int(current_prbs[1]),
                    recent_hat_sigma1=win1,
                    recent_hat_sigma2=win2,
                )
                current_action = policy.select_action(obs)
                current_prbs = action_to_prbs(current_action, cfg.R_total)

            slot_start_t = t
            slot_k = int(t // cfg.reconfig_interval)

        # Environment sampling at time t under current PRBs.
        step = env.step(t=t, prb1=current_prbs[0], prb2=current_prbs[1])
        prb1_series.append(step.prb1)
        prb2_series.append(step.prb2)
        hat1_series.append(step.hat_sigma1)
        hat2_series.append(step.hat_sigma2)

        u1, u2 = compute_utilities(
            cfg,
            hat_sigma1=step.hat_sigma1,
            hat_sigma2=step.hat_sigma2,
            sigma1=float(sigma1_t),
            sigma2=float(sigma2_t),
            slice2_active=True,
        )
        u1_series.append(u1)
        u2_series.append(u2)

        # Slot end: compute V_k and record to policy history (OPRO).
        is_slot_end = ((t - slot_start_t) == (cfg.reconfig_interval - 1)) or (t == times[-1])
        if is_slot_end:
            win_start = max(0, t - cfg.Tw + 1)
            mean_hat1 = _mean(hat1_series[win_start : t + 1])
            mean_hat2 = _mean(hat2_series[win_start : t + 1]) if slice2_active else float("nan")
            # Soft objective V_k_soft = V_k + penalty (penalty enabled only post-baseline).
            soft = evaluate_V_k_soft(
                cfg,
                t_k=t,
                hat_sigma1_series=hat1_series,
                hat_sigma2_series=hat2_series,
                sigma1_tk=float(sigma1_t),
                sigma2_tk=float(sigma2_t),
                mean_hat_sigma1=float(mean_hat1),
                mean_hat_sigma2=float(mean_hat2) if slice2_active else 0.0,
            )
            shortfall1_series[-1] = float(soft.shortfall1)
            shortfall2_series[-1] = float(soft.shortfall2)
            prb2_min_est_series[-1] = float(soft.prb2_min_est)
            waste_series[-1] = float(soft.waste)
            penalty_series[-1] = float(soft.penalty)
            V_k_soft_series[-1] = float(soft.V_k_soft)
            # --- Slot-level 回填（slot-level backfill） ---
            try:
                start = max(0, int(slot_start_t))
            except Exception:
                start = 0
            end = int(t)

            def _fill(series, value, s, e):
                if s > e or not series:
                    return
                last = min(e, len(series) - 1)
                if last < s:
                    return
                v = float(value)
                series[s : last + 1] = [v] * (last - s + 1)

            _fill(shortfall1_series, soft.shortfall1, start, end)
            _fill(shortfall2_series, soft.shortfall2, start, end)
            _fill(prb2_min_est_series, soft.prb2_min_est, start, end)
            _fill(waste_series, soft.waste, start, end)
            _fill(penalty_series, soft.penalty, start, end)
            _fill(V_k_soft_series, soft.V_k_soft, start, end)
            # --- End 回填 ---
            policy.record_outcome(
                SlotOutcome(
                    k=slot_k,
                    t_start=slot_start_t,
                    t_end=t,
                    action=current_action,
                    prb1=current_prbs[0],
                    prb2=current_prbs[1],
                    sigma1=float(sigma1_t),
                    sigma2=float(sigma2_t),
                    mean_hat_sigma1=mean_hat1,
                    mean_hat_sigma2=mean_hat2,
                    V_k=float(soft.V_k),
                    prb2_min_est=int(soft.prb2_min_est),
                    waste=int(soft.waste),
                    penalty=float(soft.penalty),
                    V_k_soft=float(soft.V_k_soft),
                )
            )

    outage_theta1_series = outage_theta_fraction(u1_series, threshold=cfg.u_th1, Tw=cfg.Tw)
    outage_theta2_series = outage_theta_fraction(u2_series, threshold=cfg.u_th2, Tw=cfg.Tw)
    w1 = system_utility_weight(cfg)
    sys_u_series = [
        system_utility_weighted(
            u1_series[i],
            u2_series[i],
            slice2_active=slice2_active_series[i],
            weight1=w1,
        )
        for i in range(len(times))
    ]
    system_outage_theta_series = [
        system_average(outage_theta1_series[i], outage_theta2_series[i], slice2_active=slice2_active_series[i])
        for i in range(len(times))
    ]
    reliability1_series = reliability_from_outage_series(outage_theta1_series)
    reliability2_series = reliability_from_outage_series(outage_theta2_series)
    system_reliability_series = reliability_from_outage_series(system_outage_theta_series)
    # severity-weighted system reliability (uses shortfall magnitudes and outage fractions)
    system_reliability_severity_series = system_reliability_severity(
        cfg,
        outage_theta1=outage_theta1_series,
        outage_theta2=outage_theta2_series,
        shortfall1=shortfall1_series,
        shortfall2=shortfall2_series,
    )
    # severity-weighted per-slice reliability (apply the same severity formula with the other slice set to 0).
    lam1 = float(cfg.sys_lambda1) if cfg.sys_lambda1 is not None else float(cfg.lambda1)
    lam2 = float(cfg.sys_lambda2) if cfg.sys_lambda2 is not None else float(cfg.lambda2)
    sev_p = int(cfg.sys_reliability_p)
    sev_eps = float(cfg.sys_reliability_eps)
    reliability1_severity_series = [
        compute_severity_weighted_reliability_at_t(
            float(shortfall1_series[i]),
            0.0,
            float(outage_theta1_series[i]),
            0.0,
            lambda1=lam1,
            lambda2=0.0,
            p=sev_p,
            eps=sev_eps,
        )
        for i in range(len(times))
    ]
    reliability2_severity_series = [
        compute_severity_weighted_reliability_at_t(
            0.0,
            float(shortfall2_series[i]),
            0.0,
            float(outage_theta2_series[i]),
            lambda1=0.0,
            lambda2=lam2,
            p=sev_p,
            eps=sev_eps,
        )
        for i in range(len(times))
    ]

    return {
        "t": [float(t) for t in times],
        "prb1": [float(x) for x in prb1_series],
        "prb2": [float(x) for x in prb2_series],
        "hat_sigma1": hat1_series,
        "hat_sigma2": hat2_series,
        "sigma1": sigma1_series,
        "sigma2": sigma2_series,
        "eff_cap1": eff_cap1_series,
        "eff_cap2": eff_cap2_series,
        "shortfall1": shortfall1_series,
        "shortfall2": shortfall2_series,
        "u1": u1_series,
        "u2": u2_series,
        # Legacy names kept for CSV backward-compat (θ is outage fraction).
        "theta1": outage_theta1_series,
        "theta2": outage_theta2_series,
        "sys_u": sys_u_series,
        "sys_theta": system_outage_theta_series,
        # Explicit outage / reliability series (preferred for new code + plots).
        "outage_theta1": outage_theta1_series,
        "outage_theta2": outage_theta2_series,
        "system_outage_theta": system_outage_theta_series,
        "reliability1": reliability1_series,
        "reliability2": reliability2_series,
        "system_reliability": system_reliability_series,
        "reliability1_severity": reliability1_severity_series,
        "reliability2_severity": reliability2_severity_series,
        "system_reliability_severity": system_reliability_severity_series,
        "prb2_min_est": prb2_min_est_series,
        "waste": waste_series,
        "penalty": penalty_series,
        "V_k_soft": V_k_soft_series,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Methods to run: all | all2 | equal random proportional tnas",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--provider", type=str, default="stub", choices=["openai", "deepseek", "google", "stub"])
    parser.add_argument("--model", type=str, default=None, help="Model name (default depends on --provider)")
    parser.add_argument(
        "--llm-runs",
        nargs="+",
        default=None,
        help="Optional: run TNAS with multiple models/providers. "
        "Each item is 'provider:model' or 'model' (uses --provider). "
        "Example: --llm-runs openai:gpt-4o deepseek:deepseek-v3.2",
    )
    parser.add_argument("--reconfig-interval", type=int, default=None)
    parser.add_argument("--Tw", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    # Load keys from `.env` if present (without overriding real environment).
    dotenv_loaded = load_dotenv()
    if dotenv_loaded:
        logger.info("Loaded .env from: %s", dotenv_loaded)

    methods = _parse_methods(args.methods)
    try:
        methods_raw = [m.lower() for m in args.methods]
    except Exception:
        methods_raw = []
    if getattr(args, "llm_runs", None) and "tnas" not in methods:
        methods.append("tnas")
        logger.info("--llm-runs provided: adding 'tnas' so the improved TNAS variants run as well.")
    out_dir = ensure_dir(args.out)

    cfg = ExperimentConfig()
    overrides = {}
    if args.reconfig_interval is not None:
        overrides["reconfig_interval"] = int(args.reconfig_interval)
    if args.Tw is not None:
        overrides["Tw"] = int(args.Tw)
    if args.smooth_window is not None:
        overrides["smooth_window"] = int(args.smooth_window)
    if overrides:
        cfg = cfg.with_overrides(**overrides)

    _warn_if_schedule_not_sane(cfg)

    save_config_prefer_yaml(cfg, out_dir)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "llm_cache")
    cache = PromptResponseCache(cache_dir)

    default_provider = str(args.provider)
    default_model = str(args.model) if args.model else _default_model_for_provider(default_provider)

    need_tnas = "tnas" in methods
    if need_tnas and not args.llm_runs:
        raise SystemExit("TNAS selected but --llm-runs was not provided. Example: --llm-runs openai:gpt-4o deepseek:deepseek-v3.2")
    llm_runs = _parse_llm_runs(args) if args.llm_runs else []

    results_by_method: Dict[str, Dict[str, List[float]]] = {}
    averages_by_method: Dict[str, Dict[str, float]] = {}
    tnas_variant_keys: List[str] = []
    display_name_by_key: Dict[str, str] = {}

    for method in methods:
        run_specs: List[Tuple[str, str, str]] = []
        if method != "tnas":
            pretty = _pretty_method_label(method)
            run_specs.append((method, default_provider, default_model))
            display_name_by_key[method] = pretty
        else:
            for prov, model in llm_runs:
                key = f"tnas_{prov}_{_sanitize_name(model)}"
                tnas_variant_keys.append(key)
                # Figure legend label: show only the model name, normalized per requirement.
                display_name_by_key[key] = _pretty_model_label(model)
                run_specs.append((key, prov, model))

        for method_key, llm_provider, llm_model in run_specs:
            logger.info("Running method=%s ...", method_key)
            res = run_single_method(
                method=method,
                cfg=cfg,
                seed=int(args.seed),
                llm_provider=llm_provider,
                llm_model=llm_model,
                cache=cache,
            )
            results_by_method[method_key] = {k: list(v) for k, v in res.items()}

            # CSV output
            rows: List[Dict[str, object]] = []
            for i, t in enumerate(res["t"]):
                rows.append(
                    {
                        "t": int(t),
                        "method": method_key,
                        "prb1": int(res["prb1"][i]),
                        "prb2": int(res["prb2"][i]),
                        "sigma1": float(res["sigma1"][i]),
                        "sigma2": float(res["sigma2"][i]),
                        "eff_cap1": float(res["eff_cap1"][i]),
                        "eff_cap2": float(res["eff_cap2"][i]),
                        "shortfall1": float(res["shortfall1"][i]),
                        "shortfall2": float(res["shortfall2"][i]),
                        "prb2_min_est": float(res["prb2_min_est"][i]),
                        "waste": float(res["waste"][i]),
                        "penalty": float(res["penalty"][i]),
                        "V_k_soft": float(res["V_k_soft"][i]),
                        "hat_sigma1": float(res["hat_sigma1"][i]),
                        "hat_sigma2": float(res["hat_sigma2"][i]),
                        "u1": float(res["u1"][i]),
                        "u2": float(res["u2"][i]),
                        # Legacy θ fields (outage fraction; kept for backward-compat).
                        "theta1": float(res["theta1"][i]),
                        "theta2": float(res["theta2"][i]),
                        "sys_u": float(res["sys_u"][i]),
                        "sys_theta": float(res["sys_theta"][i]),
                        # Explicit outage / reliability fields (preferred).
                        "outage_theta1": float(res["outage_theta1"][i]),
                        "outage_theta2": float(res["outage_theta2"][i]),
                        "system_outage_theta": float(res["system_outage_theta"][i]),
                        "reliability1": float(res["reliability1"][i]),
                        "reliability2": float(res["reliability2"][i]),
                        "system_reliability": float(res["system_reliability"][i]),
                        "reliability1_severity": float(res.get("reliability1_severity", [float("nan")])[i]),
                        "reliability2_severity": float(res.get("reliability2_severity", [float("nan")])[i]),
                        "system_reliability_severity": float(res.get("system_reliability_severity", [float("nan")])[i]),
                    }
                )
            write_csv(
                out_dir / f"timeseries_{method_key}.csv",
                fieldnames=[
                    "t",
                    "method",
                    "prb1",
                    "prb2",
                    "sigma1",
                    "sigma2",
                    "eff_cap1",
                    "eff_cap2",
                    "shortfall1",
                    "shortfall2",
                    "prb2_min_est",
                    "waste",
                    "penalty",
                    "V_k_soft",
                    "hat_sigma1",
                    "hat_sigma2",
                    "u1",
                    "u2",
                    "theta1",
                    "theta2",
                    "sys_u",
                    "sys_theta",
                    "outage_theta1",
                    "outage_theta2",
                    "system_outage_theta",
                    "reliability1",
                    "reliability2",
                    "system_reliability",
                    "reliability1_severity",
                    "reliability2_severity",
                    "system_reliability_severity",
                ],
                rows=rows,
            )

            avgs = compute_time_averages(
                cfg,
                u1=res["u1"],
                u2=res["u2"],
                outage_theta1=res["outage_theta1"],
                outage_theta2=res["outage_theta2"],
                sys_u=res["sys_u"],
                system_outage_theta=res["system_outage_theta"],
                system_reliability_severity=res.get("system_reliability_severity", None),
                start_t=cfg.baseline_start_time,
            )
            averages_by_method[method_key] = {
                "UE1": float(avgs.avg_u1),
                "UE2": float(avgs.avg_u2),
                "System": float(avgs.avg_sys_u),
                "UE1_outage_theta": float(avgs.avg_outage_theta1),
                "UE2_outage_theta": float(avgs.avg_outage_theta2),
                "System_outage_theta": float(avgs.avg_system_outage_theta),
                "UE1_reliability": float(avgs.avg_reliability1),
                "UE2_reliability": float(avgs.avg_reliability2),
                "System_reliability": float(avgs.avg_system_reliability),
                "System_reliability_severity": float(avgs.avg_system_reliability_severity),
            }
            # Additional severity-weighted per-slice averages (not part of TimeAverages yet).
            start_idx = max(0, int(cfg.baseline_start_time // cfg.dt))
            r1_sev = res.get("reliability1_severity", None)
            r2_sev = res.get("reliability2_severity", None)
            if isinstance(r1_sev, list):
                averages_by_method[method_key]["UE1_reliability_severity"] = float(_mean(r1_sev[start_idx:]))
            if isinstance(r2_sev, list):
                averages_by_method[method_key]["UE2_reliability_severity"] = float(_mean(r2_sev[start_idx:]))

    # Plotting
    methods_order: List[str] = []
    baseline_keys = [k for k in ("equal", "random", "proportional") if k in results_by_method]
    tnas_keys = [k for k in tnas_variant_keys if k in results_by_method]

    methods_order.extend(baseline_keys)
    for k in tnas_keys:
        if k not in methods_order:
            methods_order.append(k)

    # Fig.5a/5b: keep all curves (baselines + all LLM variants).
    methods_order_sys = list(methods_order)

    # Always write a per-method Fig4 panel for every produced method/variant.
    for k in methods_order:
        plot_fig4_single(
            cfg=cfg,
            method=k,
            result=results_by_method[k],
            display_name=display_name_by_key.get(k, k),
            out_path=str(out_dir / f"fig4_{k}.png"),
        )

    # Write a single combined Fig4 grid containing all produced methods/variants.
    if methods_order:
        plot_fig4_grid(
            cfg=cfg,
            results_by_method=results_by_method,
            methods_order=methods_order,
            display_names=display_name_by_key,
            out_path=str(out_dir / "fig4.png"),
        )

    plot_fig5_sys_curve(
        cfg=cfg,
        results_by_method=results_by_method,
        methods_order=methods_order_sys,
        series_key="sys_u",
        out_path=str(out_dir / "fig5a_sys_utility.png"),
        title="Fig.5a Smoothed System Utility",
        ylabel="System Utility",
        display_names=display_name_by_key,
        legend_loc="upper right",
        legend_bbox_to_anchor=None,
    )
    plot_fig5_sys_curve(
        cfg=cfg,
        results_by_method=results_by_method,
        methods_order=methods_order_sys,
        series_key="u1",
        out_path=str(out_dir / "fig5a_ue1_utility.png"),
        title="Fig.5a Smoothed UE1 Utility",
        ylabel="UE1 Utility",
        display_names=display_name_by_key,
        legend_loc="upper right",
        legend_bbox_to_anchor=None,
    )
    plot_fig5_sys_curve(
        cfg=cfg,
        results_by_method=results_by_method,
        methods_order=methods_order_sys,
        series_key="u2",
        out_path=str(out_dir / "fig5a_ue2_utility.png"),
        title="Fig.5a Smoothed UE2 Utility",
        ylabel="UE2 Utility",
        display_names=display_name_by_key,
        legend_loc="upper right",
        legend_bbox_to_anchor=None,
    )
    plot_fig5_sys_curve(
        cfg=cfg,
        results_by_method=results_by_method,
        methods_order=methods_order_sys,
        series_key="system_reliability_severity",
        out_path=str(out_dir / "fig5b_sys_reliability_severity.png"),
        title="Fig.5b Smoothed Severity-weighted System Reliability",
        ylabel="Severity-weighted System Reliability",
        display_names=display_name_by_key,
        legend_loc="upper right",
        legend_bbox_to_anchor=None,
    )

    plot_fig5_bars(
        averages_by_method={m: averages_by_method[m] for m in methods_order},
        methods_order=methods_order,
        keys=["UE1", "UE2", "System"],
        title=f"Fig.5c Time-averaged Utility (t≥{cfg.baseline_start_time}s)",
        ylabel="Utility",
        out_path=str(out_dir / "fig5c_avg_utility.png"),
        display_names=display_name_by_key,
    )
    plot_fig5_bars(
        averages_by_method={
            m: {
                "UE1": float(averages_by_method[m].get("UE1_reliability_severity", float("nan"))),
                "UE2": float(averages_by_method[m].get("UE2_reliability_severity", float("nan"))),
                "System": float(averages_by_method[m].get("System_reliability_severity", float("nan"))),
            }
            for m in methods_order
        },
        methods_order=methods_order,
        keys=["UE1", "UE2", "System"],
        title=f"Fig.5d Time-averaged Severity-weighted Reliability (t≥{cfg.baseline_start_time}s; higher is better)",
        ylabel="Severity-weighted Reliability",
        out_path=str(out_dir / "fig5d_avg_reliability_severity.png"),
        display_names=display_name_by_key,
    )

    logger.info("Done. Outputs written to: %s", out_dir)


if __name__ == "__main__":
    main()
