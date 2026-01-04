from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

from .config import ExperimentConfig
from .metrics import moving_average_trailing


# Explicit UE colors (paper-style, consistent across machines/styles)
UE1_COLOR = "#000080"  # navy
UE2_COLOR = "#B22222"  # firebrick


def _ceil_to_step(x: float, step: int) -> int:
    if step <= 0:
        raise ValueError("step must be positive")
    return int(math.ceil(float(x) / float(step)) * step)


def _fig4_ymax(cfg: ExperimentConfig, *, tick_step: int = 5) -> int:
    """Return a common y-axis max for Fig.4 plots (start from 0, tick step fixed)."""

    schedule = cfg.demand_schedule or [(0, cfg.sigma1, cfg.sigma2)]
    sigmas1 = [float(cfg.sigma1)]
    sigmas2 = [float(cfg.sigma2)]
    for entry in schedule:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        try:
            sigmas1.append(float(entry[1]))
            sigmas2.append(float(entry[2]))
        except Exception:
            continue

    max_sigma1 = max(sigmas1) if sigmas1 else float(cfg.sigma1)
    max_sigma2 = max(sigmas2) if sigmas2 else float(cfg.sigma2)

    # Match env.py soft caps: hat1 <= eff_cap1 + 2.0; hat2 <= eff_cap2 + 1.5.
    cap1 = max_sigma1 if cfg.cap1_hard_mbps is None else min(max_sigma1, float(cfg.cap1_hard_mbps))
    cap2 = max_sigma2 if cfg.cap2_hard_mbps is None else min(max_sigma2, float(cfg.cap2_hard_mbps))
    ymax = max(cap1 + 2.0, cap2 + 1.5, 10.0)
    return max(tick_step, _ceil_to_step(ymax, tick_step))


def _apply_fig4_yaxis(ax, *, cfg: ExperimentConfig, tick_step: int = 5) -> None:
    ymax = _fig4_ymax(cfg, tick_step=tick_step)
    ax.set_ylim(0.0, float(ymax))
    ax.set_yticks(list(range(0, int(ymax) + 1, int(tick_step))))


def _demand_change_times(cfg: ExperimentConfig) -> List[int]:
    """Return demand schedule change times strictly after baseline_start_time."""

    times: List[int] = []
    for entry in cfg.demand_schedule or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        try:
            t0 = int(entry[0])
        except Exception:
            continue
        if t0 > int(cfg.baseline_start_time) and t0 not in times:
            times.append(t0)
    times.sort()
    return times


def plot_fig4_grid(
    *,
    cfg: ExperimentConfig,
    results_by_method: Mapping[str, Mapping[str, Sequence[float]]],
    methods_order: Sequence[str],
    display_names: Mapping[str, str] | None = None,
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
        title_name = display_names.get(method, method) if display_names else method
        t = list(res["t"])
        ue1 = list(res["hat_sigma1"])
        ue2 = list(res["hat_sigma2"])

        ax.plot(t, ue1, label="UE1 / S1", linewidth=1.5, color=UE1_COLOR)
        ax.plot(t, ue2, label="UE2 / S2", linewidth=1.5, color=UE2_COLOR)
        ax.axvline(cfg.slice_init_time, color="k", linestyle="--", linewidth=1.0)
        ax.axvline(cfg.baseline_start_time, color="k", linestyle=":", linewidth=1.0)
        for t0 in _demand_change_times(cfg):
            ax.axvline(t0, color="0.5", linestyle="-.", linewidth=1.0)
        _apply_fig4_yaxis(ax, cfg=cfg, tick_step=5)
        ax.set_title(f"{_panel_label(i)} {title_name}")
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
        "-- slice init @100s, : allocation starts @200s, -. demand change"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig4_single(
    *,
    cfg: ExperimentConfig,
    method: str,
    result: Mapping[str, Sequence[float]],
    display_name: str | None = None,
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
    ax.plot(t, ue1, label="UE1 / S1", linewidth=1.5, color=UE1_COLOR)
    ax.plot(t, ue2, label="UE2 / S2", linewidth=1.5, color=UE2_COLOR)
    ax.axvline(cfg.slice_init_time, color="k", linestyle="--", linewidth=1.0, label="slice init")
    ax.axvline(cfg.baseline_start_time, color="k", linestyle=":", linewidth=1.0, label="allocation starts")
    for t0 in _demand_change_times(cfg):
        ax.axvline(t0, color="0.5", linestyle="-.", linewidth=1.0, label="demand change")
    _apply_fig4_yaxis(ax, cfg=cfg, tick_step=5)
    ax.set_title(display_name or method)
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
    display_names: Mapping[str, str] | None = None,
) -> None:
    """Plot Fig.5a/5b style smoothed system curve."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for method in methods_order:
        res = results_by_method[method]
        label = display_names.get(method, method) if display_names else method
        t = list(res["t"])
        raw = list(res[series_key])
        smooth = moving_average_trailing(raw, cfg.smooth_window)
        ax.plot(t, smooth, label=label, linewidth=1.7)

    ax.axvline(cfg.slice_init_time, color="k", linestyle="--", linewidth=1.0)
    ax.axvline(cfg.baseline_start_time, color="k", linestyle=":", linewidth=1.0)
    for t0 in _demand_change_times(cfg):
        ax.axvline(t0, color="0.5", linestyle="-.", linewidth=1.0)
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
    display_names: Mapping[str, str] | None = None,
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
        label = display_names.get(method, method) if display_names else method
        vals = [averages_by_method[method][k] for k in groups]
        ax.bar([xi + (i - (len(methods_order) - 1) / 2) * width for xi in x], vals, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
