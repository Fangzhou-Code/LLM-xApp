from __future__ import annotations

import math
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
    """Plot Fig.4-style grid (variable size): throughput curves for each method/variant."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = len(methods_order)
    if n_panels <= 0:
        raise ValueError("methods_order must contain at least one method.")

    # Make a single combined figure that can include multiple LLM variants.
    if n_panels <= 4:
        ncols = 2
    elif n_panels <= 6:
        ncols = 3
    elif n_panels <= 12:
        ncols = 4
    else:
        ncols = 5
    nrows = int(math.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
    )
    if isinstance(axes, list):
        flat_axes = axes
    else:
        try:
            flat_axes = axes.flatten().tolist()
        except Exception:
            flat_axes = [axes]

    def _panel_label(i: int) -> str:
        if 0 <= i < 26:
            return f"({chr(ord('a') + i)})"
        return f"({i + 1})"

    for i, method in enumerate(methods_order):
        ax = flat_axes[i]
        res = results_by_method[method]
        t = list(res["t"])
        ue1 = list(res["hat_sigma1"])
        ue2 = list(res["hat_sigma2"])

        ax.plot(t, ue1, label="UE1 / S1", linewidth=1.5)
        ax.plot(t, ue2, label="UE2 / S2", linewidth=1.5)
        ax.axvline(cfg.slice_init_time, color="k", linestyle="--", linewidth=1.0)
        alloc_start_t = cfg.slice_init_time if method == "equal" else cfg.baseline_start_time
        ax.axvline(alloc_start_t, color="k", linestyle=":", linewidth=1.0)
        ax.set_title(f"{_panel_label(i)} {method}")
        ax.grid(True, alpha=0.3)
        if (i % ncols) == 0:
            ax.set_ylabel("Measured data rate (Mbps)")
        if i >= (ncols * (nrows - 1)):
            ax.set_xlabel("Time (s)")

    # Turn off any unused axes.
    for j in range(n_panels, len(flat_axes)):
        flat_axes[j].axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Fig.4-style Measured Data Rate (UE1 vs UE2)\n"
        "-- slice init @100s, : allocation starts (equal@100s, others@200s)"
    )
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
    ue2 = list(result["hat_sigma2"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, ue1, label="UE1 / S1", linewidth=1.5)
    ax.plot(t, ue2, label="UE2 / S2", linewidth=1.5)
    ax.axvline(cfg.slice_init_time, color="k", linestyle="--", linewidth=1.0, label="slice init")
    alloc_start_t = cfg.slice_init_time if method == "equal" else cfg.baseline_start_time
    ax.axvline(alloc_start_t, color="k", linestyle=":", linewidth=1.0, label="allocation starts")
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
