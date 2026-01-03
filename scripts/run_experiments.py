from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from ran_llm_xapp.config import ExperimentConfig, save_config_prefer_yaml
from ran_llm_xapp.env import SyntheticRANSliceEnv
from ran_llm_xapp.io_utils import PromptResponseCache, ensure_dir, load_dotenv, write_csv
from ran_llm_xapp.llm_clients import DeepSeekClient, OpenAIClient, StubLLMClient
from ran_llm_xapp.metrics import (
    action_to_prbs,
    compute_time_averages,
    compute_utilities,
    reliability_outage_fraction,
    system_average,
    evaluate_V_k,
)
from ran_llm_xapp.plotting import plot_fig4_grid, plot_fig4_single, plot_fig5_bars, plot_fig5_sys_curve
from ran_llm_xapp.policies import (
    EqualPolicy,
    LLMOPROPolicy,
    Observation,
    ProportionalPolicy,
    RandomPolicy,
    SlotOutcome,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_experiments")


METHODS_ALL = ("random", "equal", "proportional", "llm")


def _parse_methods(raw: Sequence[str]) -> List[str]:
    if len(raw) == 1 and raw[0].lower() == "all":
        return list(METHODS_ALL)
    methods = [m.lower() for m in raw]
    unknown = [m for m in methods if m not in METHODS_ALL]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Valid: {METHODS_ALL} or 'all'")
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
    return "stub"


def _parse_llm_runs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    """Parse `--llm-runs` into [(provider, model), ...]."""

    provider_default = str(args.provider).lower()
    model_default = str(args.model) if args.model else _default_model_for_provider(provider_default)

    llm_runs_raw = getattr(args, "llm_runs", None)
    if not llm_runs_raw:
        return [(provider_default, model_default)]

    runs: List[Tuple[str, str]] = []
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

        if provider not in {"openai", "deepseek", "stub"}:
            raise SystemExit(f"Unknown provider in --llm-runs: {provider!r}")
        if not model:
            raise SystemExit("Model name cannot be empty in --llm-runs.")
        runs.append((provider, model))

    # de-dup while preserving order
    out: List[Tuple[str, str]] = []
    for r in runs:
        if r not in out:
            out.append(r)
    return out


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
    if method == "llm":
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
        else:
            client = StubLLMClient(seed=seed + 444)
        return LLMOPROPolicy(cfg=cfg, llm_client=client, model=llm_model, cache=cache, seed=seed + 555)
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
                    sigma1=cfg.sigma1,
                    sigma2=cfg.sigma2,
                    slice2_active=True,
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
            sigma1=cfg.sigma1,
            sigma2=cfg.sigma2,
            slice2_active=True,
        )
        u1_series.append(u1)
        u2_series.append(u2)

        # Slot end: compute V_k and record to policy history (OPRO).
        is_slot_end = ((t - slot_start_t) == (cfg.reconfig_interval - 1)) or (t == times[-1])
        if is_slot_end:
            # V_k uses Eq.(8) evaluated at slot end time index t.
            V = evaluate_V_k(cfg, t_k=t, hat_sigma1_series=hat1_series, hat_sigma2_series=hat2_series)
            win_start = max(0, t - cfg.Tw + 1)
            mean_hat1 = _mean(hat1_series[win_start : t + 1])
            mean_hat2 = _mean(hat2_series[win_start : t + 1]) if slice2_active else float("nan")
            policy.record_outcome(
                SlotOutcome(
                    k=slot_k,
                    t_start=slot_start_t,
                    t_end=t,
                    action=current_action,
                    prb1=current_prbs[0],
                    prb2=current_prbs[1],
                    sigma1=cfg.sigma1,
                    sigma2=cfg.sigma2,
                    mean_hat_sigma1=mean_hat1,
                    mean_hat_sigma2=mean_hat2,
                    V=V,
                )
            )

    theta1_series = reliability_outage_fraction(u1_series, threshold=cfg.u_th1, Tw=cfg.Tw)
    theta2_series = reliability_outage_fraction(u2_series, threshold=cfg.u_th2, Tw=cfg.Tw)
    sys_u_series = [
        system_average(u1_series[i], u2_series[i], slice2_active=slice2_active_series[i]) for i in range(len(times))
    ]
    sys_theta_series = [
        system_average(theta1_series[i], theta2_series[i], slice2_active=slice2_active_series[i])
        for i in range(len(times))
    ]

    return {
        "t": [float(t) for t in times],
        "prb1": [float(x) for x in prb1_series],
        "prb2": [float(x) for x in prb2_series],
        "hat_sigma1": hat1_series,
        "hat_sigma2": hat2_series,
        "sigma1": [float(cfg.sigma1) for _ in times],
        "sigma2": [float(cfg.sigma2) for _ in times],
        "u1": u1_series,
        "u2": u2_series,
        "theta1": theta1_series,
        "theta2": theta2_series,
        "sys_u": sys_u_series,
        "sys_theta": sys_theta_series,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", required=True, help="Methods to run: all | random equal proportional llm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--provider", type=str, default="stub", choices=["openai", "deepseek", "stub"])
    parser.add_argument("--model", type=str, default=None, help="Model name (default depends on --provider)")
    parser.add_argument(
        "--llm-runs",
        nargs="+",
        default=None,
        help="Optional: run llm with multiple models/providers. "
        "Each item is 'provider:model' or 'model' (uses --provider). "
        "Example: --llm-runs openai:gpt-4o-mini deepseek:deepseek-chat",
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

    save_config_prefer_yaml(cfg, out_dir)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "llm_cache")
    cache = PromptResponseCache(cache_dir)

    default_provider = str(args.provider)
    default_model = str(args.model) if args.model else _default_model_for_provider(default_provider)
    llm_runs = _parse_llm_runs(args)

    results_by_method: Dict[str, Dict[str, List[float]]] = {}
    averages_by_method: Dict[str, Dict[str, float]] = {}
    llm_variant_keys: List[str] = []

    for method in methods:
        run_specs: List[Tuple[str, str, str]] = []
        if method != "llm":
            run_specs.append((method, default_provider, default_model))
        else:
            for prov, model in llm_runs:
                if len(llm_runs) == 1:
                    key = "llm"
                else:
                    key = f"llm_{prov}_{_sanitize_name(model)}"
                llm_variant_keys.append(key)
                run_specs.append((key, prov, model))

        for method_key, llm_provider, llm_model in run_specs:
            logger.info("Running method=%s ...", method_key)
            res = run_single_method(
                method="llm" if method == "llm" else method,
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
                        "hat_sigma1": float(res["hat_sigma1"][i]),
                        "hat_sigma2": float(res["hat_sigma2"][i]),
                        "sigma1": float(res["sigma1"][i]),
                        "sigma2": float(res["sigma2"][i]),
                        "u1": float(res["u1"][i]),
                        "u2": float(res["u2"][i]),
                        "theta1": float(res["theta1"][i]),
                        "theta2": float(res["theta2"][i]),
                        "sys_u": float(res["sys_u"][i]),
                        "sys_theta": float(res["sys_theta"][i]),
                    }
                )
            write_csv(
                out_dir / f"timeseries_{method_key}.csv",
                fieldnames=[
                    "t",
                    "method",
                    "prb1",
                    "prb2",
                    "hat_sigma1",
                    "hat_sigma2",
                    "sigma1",
                    "sigma2",
                    "u1",
                    "u2",
                    "theta1",
                    "theta2",
                    "sys_u",
                    "sys_theta",
                ],
                rows=rows,
            )

            avgs = compute_time_averages(
                cfg,
                u1=res["u1"],
                u2=res["u2"],
                theta1=res["theta1"],
                theta2=res["theta2"],
                sys_u=res["sys_u"],
                sys_theta=res["sys_theta"],
                start_t=cfg.baseline_start_time,
            )
            averages_by_method[method_key] = {
                "UE1": float(avgs.avg_u1),
                "UE2": float(avgs.avg_u2),
                "System": float(avgs.avg_sys_u),
                "UE1_theta": float(avgs.avg_theta1),
                "UE2_theta": float(avgs.avg_theta2),
                "System_theta": float(avgs.avg_sys_theta),
            }

    # Plotting
    methods_order: List[str] = []
    baseline_keys = [k for k in ("random", "equal", "proportional") if k in results_by_method]
    llm_keys = [k for k in llm_variant_keys if k in results_by_method]

    methods_order.extend(baseline_keys)
    for k in llm_keys:
        if k not in methods_order:
            methods_order.append(k)

    # Always write a per-method Fig4 panel for every produced method/variant.
    for k in methods_order:
        plot_fig4_single(
            cfg=cfg,
            method=k,
            result=results_by_method[k],
            out_path=str(out_dir / f"fig4_{k}.png"),
        )

    # Write a single combined Fig4 grid containing all produced methods/variants.
    if methods_order:
        plot_fig4_grid(
            cfg=cfg,
            results_by_method=results_by_method,
            methods_order=methods_order,
            out_path=str(out_dir / "fig4.png"),
        )

    plot_fig5_sys_curve(
        cfg=cfg,
        results_by_method=results_by_method,
        methods_order=methods_order,
        series_key="sys_u",
        out_path=str(out_dir / "fig5a_sys_utility.png"),
        title="Fig.5a Smoothed System Utility",
        ylabel="System Utility (moving avg)",
    )
    plot_fig5_sys_curve(
        cfg=cfg,
        results_by_method=results_by_method,
        methods_order=methods_order,
        series_key="sys_theta",
        out_path=str(out_dir / "fig5b_sys_reliability.png"),
        title="Fig.5b Smoothed System Reliability (Outage Fraction; lower is better)",
        ylabel="System Reliability θ (moving avg)",
    )

    plot_fig5_bars(
        averages_by_method={m: averages_by_method[m] for m in methods_order},
        methods_order=methods_order,
        keys=["UE1", "UE2", "System"],
        title=f"Fig.5c Time-averaged Utility (t≥{cfg.baseline_start_time}s)",
        ylabel="Utility (avg)",
        out_path=str(out_dir / "fig5c_avg_utility.png"),
    )
    plot_fig5_bars(
        averages_by_method={
            m: {"UE1": averages_by_method[m]["UE1_theta"], "UE2": averages_by_method[m]["UE2_theta"], "System": averages_by_method[m]["System_theta"]}
            for m in methods_order
        },
        methods_order=methods_order,
        keys=["UE1", "UE2", "System"],
        title=f"Fig.5d Time-averaged Reliability θ (t≥{cfg.baseline_start_time}s; lower is better)",
        ylabel="Reliability θ (avg outage fraction)",
        out_path=str(out_dir / "fig5d_avg_reliability.png"),
    )

    logger.info("Done. Outputs written to: %s", out_dir)


if __name__ == "__main__":
    main()
