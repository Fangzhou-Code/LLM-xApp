from __future__ import annotations

import argparse
import csv
import math
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from ran_llm_xapp.config import ExperimentConfig
from ran_llm_xapp.io_utils import PromptResponseCache, ensure_dir, load_dotenv
from scripts.run_experiments import run_single_method


OUT_DIR = Path("outputs/seed_sweep")
OUT_CSV = OUT_DIR / "seed_sweep.csv"
OUT_PDF_UTILITY = OUT_DIR / "seed_sweep_sys_utility.pdf"
OUT_PDF_RELIABILITY = OUT_DIR / "seed_sweep_sys_reliability.pdf"


def _mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if (x == x and not math.isinf(float(x)))]
    return sum(vals) / float(len(vals)) if vals else float("nan")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "model", "mean_system_utility", "mean_system_reliability"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_bar_pdf(
    *,
    out_pdf: Path,
    seeds: List[int],
    models: List[str],
    values: Dict[Tuple[int, str], float],
    ylabel: str,
) -> None:
    """Bar chart PDF using Matplotlib to match Fig4/Fig5 Times New Roman exactly."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Matplotlib (with numpy) is required to render Times New Roman consistently. "
            "Please run this script in the same Python environment you use for Fig4/Fig5."
        ) from e

    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = ["#000000", "#2C939A", "#A6A3A4"]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 3.8))

    x = list(range(len(seeds)))
    n_models = max(1, len(models))
    group_span = 0.86
    width = group_span / float(n_models)
    offsets = [(i - (n_models - 1) / 2.0) * width for i in range(n_models)]

    for i, model in enumerate(models):
        vals = [float(values.get((seed, model), float("nan"))) for seed in seeds]
        ax.bar([xi + offsets[i] for xi in x], vals, width=width * 0.95, label=model, color=colors[i % len(colors)])

    # Keep the legend inside the plot box but higher to reduce overlap with bars.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(3, len(models)),
        frameon=True,
        borderaxespad=0.1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    # Add extra headroom so the in-box legend doesn't overlap the tallest bars.
    ax.set_ylim(0.0, 1.25)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.02, 0.98))
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def _parse_llm_runs(raw: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        if ":" not in s:
            raise SystemExit(f"--llm-runs entries must be provider:model, got: {s!r}")
        prov, model = s.split(":", 1)
        prov = prov.strip().lower()
        model = model.strip()
        if prov not in {"openai", "deepseek", "google"}:
            raise SystemExit(f"Unsupported provider in --llm-runs: {prov!r} (expected openai/deepseek/google)")
        if not model:
            raise SystemExit(f"Empty model in --llm-runs entry: {s!r}")
        out.append((prov, model))
    if not out:
        raise SystemExit("--llm-runs must contain at least one provider:model entry")
    return out


def _require_env_for_provider(provider: str) -> None:
    provider = provider.lower()
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
            raise SystemExit("Missing OPENAI_API_KEY/OPENAI_BASE_URL; required for real LLM runs.")
    elif provider == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DEEPSEEK_BASE_URL"):
            raise SystemExit("Missing DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL; required for real LLM runs.")
    elif provider == "google":
        if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_BASE_URL"):
            raise SystemExit("Missing GOOGLE_API_KEY/GOOGLE_BASE_URL; required for real LLM runs.")


def _quiet_logs() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("run_experiments").setLevel(logging.WARNING)
    logging.getLogger("ran_llm_xapp.policies.tnas").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed sweep for TNAS system utility/reliability with real LLMs.")
    parser.add_argument(
        "--llm-runs",
        nargs="+",
        required=True,
        help="List of provider:model entries, e.g. openai:gpt-4o-mini deepseek:deepseek-v3.2 google:gemini-3-pro",
    )
    args = parser.parse_args()

    load_dotenv()
    _quiet_logs()

    out_dir = ensure_dir(OUT_DIR)
    cache = PromptResponseCache(out_dir / "llm_cache")

    seeds = [0, 3, 6, 9]
    llm_runs = _parse_llm_runs(list(args.llm_runs))
    for prov, _model in llm_runs:
        _require_env_for_provider(prov)
    models = [model for _prov, model in llm_runs]

    # Reduce LLM call count.
    cfg = ExperimentConfig().with_overrides(reconfig_interval=50)
    rows: List[Dict[str, object]] = []
    values_u: Dict[Tuple[int, str], float] = {}
    values_r: Dict[Tuple[int, str], float] = {}

    total_runs = len(seeds) * len(llm_runs)
    run_i = 0
    for seed in seeds:
        for (prov, model) in llm_runs:
            run_i += 1
            print(f"[{run_i}/{total_runs}] seed={seed} provider={prov} model={model} ...", flush=True)
            res = run_single_method(
                method="tnas",
                cfg=cfg,
                seed=int(seed),
                llm_provider=str(prov),
                llm_model=str(model),
                cache=cache,
            )
            t = [int(x) for x in res["t"]]
            sys_u = [float(x) for x in res["sys_u"]]
            sys_r = [float(x) for x in res["system_reliability"]]
            start_t = int(cfg.baseline_start_time)
            vals_u = [u for tv, u in zip(t, sys_u) if tv >= start_t]
            vals_r = [r for tv, r in zip(t, sys_r) if tv >= start_t]
            mu = _mean(vals_u)
            mr = _mean(vals_r)
            values_u[(int(seed), str(model))] = float(mu)
            values_r[(int(seed), str(model))] = float(mr)
            rows.append(
                {
                    "seed": int(seed),
                    "model": str(model),
                    "mean_system_utility": float(mu),
                    "mean_system_reliability": float(mr),
                }
            )

    _write_csv(OUT_CSV, rows)
    _write_bar_pdf(out_pdf=OUT_PDF_UTILITY, seeds=seeds, models=models, values=values_u, ylabel="System Utility")
    _write_bar_pdf(out_pdf=OUT_PDF_RELIABILITY, seeds=seeds, models=models, values=values_r, ylabel="System Reliability")
    print(f"DONE: wrote {OUT_CSV}", flush=True)
    print(f"FIGS={OUT_PDF_UTILITY}; {OUT_PDF_RELIABILITY}", flush=True)


if __name__ == "__main__":
    main()
