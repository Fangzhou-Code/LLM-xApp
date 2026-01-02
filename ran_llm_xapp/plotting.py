from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from .config import ExperimentConfig
from .metrics import moving_average_trailing


def plot_fig4_grid(
    *,
    cfg: ExperimentConfig,
    results_by_method: Mapping[str, Mapping[str, Sequence[float]]],
    methods_order: Sequence[str],
    out_path: str,
) -> None:
    """Plot Fig.4-style 2x2 grid: throughput curves for each method."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, method in enumerate(methods_order):
        ax = axes[i]
        res = results_by_method[method]
        t = list(res["t"])
        ue1 = list(res["hat_sigma1"])
        ue2_raw = list(res["hat_sigma2"])
        # UE2 does not exist before slice init; avoid drawing a misleading flat line at 0.
        ue2 = [ue2_raw[j] if t[j] >= cfg.slice_init_time else float("nan") for j in range(len(t))]

        ax.plot(t, ue1, label="UE1 / S1", linewidth=1.5)
        ax.plot(t, ue2, label="UE2 / S2", linewidth=1.5)
        ax.axvline(cfg.slice_init_time, color="k", linestyle="--", linewidth=1.0)
        ax.axvline(cfg.baseline_start_time, color="k", linestyle=":", linewidth=1.0)
        ax.set_title(f"{panel_labels[i]} {method}")
        ax.grid(True, alpha=0.3)
        if i % 2 == 0:
            ax.set_ylabel("Measured data rate (Mbps)")
        if i >= 2:
            ax.set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Fig.4-style Measured Data Rate (UE1 vs UE2)\n-- slice init @100s, : baseline start @200s")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig4_single(
    *,
    cfg: ExperimentConfig,
    method: str,
    result: Mapping[str, Sequence[float]],
    out_path: str,
) -> None:
    """Plot a single Fig.4-style panel for one method."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = list(result["t"])
    ue1 = list(result["hat_sigma1"])
    ue2_raw = list(result["hat_sigma2"])
    ue2 = [ue2_raw[j] if t[j] >= cfg.slice_init_time else float("nan") for j in range(len(t))]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, ue1, label="UE1 / S1", linewidth=1.5)
    ax.plot(t, ue2, label="UE2 / S2", linewidth=1.5)
    ax.axvline(cfg.slice_init_time, color="k", linestyle="--", linewidth=1.0, label="slice init")
    ax.axvline(cfg.baseline_start_time, color="k", linestyle=":", linewidth=1.0, label="baseline start")
    ax.set_title(method)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measured data rate (Mbps)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig5_sys_curve(
    *,
    cfg: ExperimentConfig,
    results_by_method: Mapping[str, Mapping[str, Sequence[float]]],
    methods_order: Sequence[str],
    series_key: str,
    out_path: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot Fig.5a/5b style smoothed system curve."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for method in methods_order:
        res = results_by_method[method]
        t = list(res["t"])
        raw = list(res[series_key])
        smooth = moving_average_trailing(raw, cfg.smooth_window)
        ax.plot(t, smooth, label=method, linewidth=1.7)

    ax.axvline(cfg.slice_init_time, color="k", linestyle="--", linewidth=1.0)
    ax.axvline(cfg.baseline_start_time, color="k", linestyle=":", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig5_bars(
    *,
    averages_by_method: Mapping[str, Mapping[str, float]],
    methods_order: Sequence[str],
    keys: Sequence[str],
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    """Plot grouped bar chart for UE1/UE2/System time-averaged metrics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(keys)  # e.g., ["UE1","UE2","System"]
    x = list(range(len(groups)))
    width = 0.18

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for i, method in enumerate(methods_order):
        vals = [averages_by_method[method][k] for k in groups]
        ax.bar([xi + (i - (len(methods_order) - 1) / 2) * width for xi in x], vals, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
