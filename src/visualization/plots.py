from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.analysis.metrics import IdentificationMetrics, ThresholdPoint

_BG        = "#F7F8FC"
_SURFACE   = "#FFFFFF"
_BORDER    = "#E2E6EF"
_TEXT_MAIN = "#1C2135"
_TEXT_SUB  = "#5A6480"
_GRID_CLR  = "#DDE2EE"

_PALETTE = [
    "#3B6FE8",
    "#E8603B",
    "#2BAB7E",
    "#A855F7",
    "#F59E0B",
    "#0EA5E9",
    "#EF4444",
    "#10B981",
]

_BLUE  = _PALETTE[0]
_RED   = _PALETTE[1]
_GREEN = _PALETTE[2]

_METRIC_LABELS: dict[str, str] = {
    "success_rate":         "Success Rate",
    "coverage":             "Coverage",
    "conditional_accuracy": "Cond. Accuracy",
    "score_separation":     "Score Sep.",
    "abstention_rate":      "Abstention Rate",
}
_DEFAULT_METRICS = ["success_rate", "coverage", "conditional_accuracy", "score_separation"]

_FS_TITLE = 15
_FS_SUPT  = 14
_FS_AXIS  = 11
_FS_TICK  = 9.5
_FS_ANNOT = 9.0
_FS_SMALL = 8.5


def _apply_base_style(ax: plt.Axes, *, ymax: float | None = 1.05) -> None:
    ax.set_facecolor(_SURFACE)
    ax.figure.set_facecolor(_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_BORDER)
    ax.tick_params(colors=_TEXT_SUB, labelsize=_FS_TICK, length=3, pad=4)
    ax.xaxis.label.set_color(_TEXT_MAIN)
    ax.yaxis.label.set_color(_TEXT_MAIN)
    ax.grid(axis="y", color=_GRID_CLR, linewidth=0.8, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    if ymax is not None:
        ax.set_ylim(0, ymax)


def _pct_fmt() -> mticker.PercentFormatter:
    return mticker.PercentFormatter(xmax=1, decimals=0)


def _set_title(ax: plt.Axes, text: str, *, suptitle: bool = False) -> None:
    if suptitle:
        ax.figure.suptitle(
            text, fontsize=_FS_SUPT, fontweight="bold", color=_TEXT_MAIN, y=1.02
        )
    else:
        ax.set_title(
            text, fontsize=_FS_TITLE, fontweight="bold", color=_TEXT_MAIN, pad=16
        )


def _bar_labels(
        ax: plt.Axes,
        bars,
        *,
        errors: np.ndarray | None = None,
        pad: float = 0.012,
        fs: float = _FS_ANNOT,
) -> None:
    """Place a percentage label just above each bar (or above the error cap).

    Args:
        bars:   Return value of ``ax.bar(…)``.
        errors: Per-bar ±error values (same length as bars); label is pushed
                above the error cap so it never collides with the cap marker.
        pad:    Additional gap (in data units) above the bar top / error cap.
        fs:     Font size.
    """
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if np.isnan(h) or h == 0:
            continue
        err = float(errors[i]) if errors is not None and errors[i] > 0 else 0.0
        y = h + err + pad
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{h:.2%}",
            ha="center", va="bottom",
            fontsize=fs, color=_TEXT_MAIN, fontweight="semibold",
            clip_on=False,
        )


def _bar_ymax(
        values: np.ndarray,
        errors: np.ndarray,
        *,
        label_headroom: float = 0.09,
) -> float:
    """Upper y-limit that gives bars + error caps + bar-value labels room to breathe."""
    tops = np.abs(values) + np.abs(errors)
    return float(tops.max()) + label_headroom


def _save(fig: plt.Figure, save_path: Path | None) -> plt.Figure:
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=180, facecolor=_BG)
    return fig

def plot_success_rate_bar(
        scenario_metrics: Dict[str, IdentificationMetrics],
        metrics: Optional[List[str]] = None,
        title: str = "Attack Scenario Comparison",
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing key identification metrics across scenarios."""
    if metrics is None:
        metrics = _DEFAULT_METRICS

    scenario_names = list(scenario_metrics.keys())
    n_m = len(metrics)
    n_s = len(scenario_names)

    values = np.zeros((n_m, n_s))
    errors = np.zeros((n_m, n_s))
    for si, name in enumerate(scenario_names):
        d = scenario_metrics[name].to_dict()
        for mi, met in enumerate(metrics):
            raw = d.get(met, 0.0)
            values[mi, si] = abs(raw)
            errors[mi, si] = abs(d.get(f"std_{met}", 0.0))

    fig_w = max(10.0, n_m * n_s * 0.9 + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))

    gw = 0.72
    bw = gw / n_s
    group_pos = np.arange(n_m, dtype=float)
    patches = []

    for si, name in enumerate(scenario_names):
        offsets = group_pos - gw / 2 + bw * (si + 0.5)
        err_vec = errors[:, si]
        has_err = err_vec.any()
        bars = ax.bar(
            offsets, values[:, si],
            width = bw * 0.86,
            color = _PALETTE[si % len(_PALETTE)],
            alpha = 0.88,
            yerr = err_vec if has_err else None,
            capsize = 4,
            error_kw = dict(elinewidth=1.4, ecolor="#555", alpha=0.75),
            zorder = 3,
        )
        _bar_labels(ax, bars, errors=err_vec if has_err else None)
        patches.append(mpatches.Patch(
            color=_PALETTE[si % len(_PALETTE)], alpha=0.88, label=name
        ))

    for x in group_pos[:-1] + 0.5:
        ax.axvline(x, color=_BORDER, linewidth=0.9, linestyle="--", zorder=1)

    ymax = min(1.32, _bar_ymax(values, errors, label_headroom=0.09))
    _apply_base_style(ax, ymax=ymax)

    ax.set_xticks(group_pos)
    rotation = 0 if n_m <= 4 else 20
    ax.set_xticklabels(
        [_METRIC_LABELS.get(m, m) for m in metrics],
        fontsize=_FS_TICK + 0.5, fontweight="medium",
        color=_TEXT_MAIN, rotation=rotation,
        ha="right" if rotation else "center",
    )
    ax.tick_params(axis="x", length=0, pad=6)
    ax.yaxis.set_major_formatter(_pct_fmt())
    ax.set_ylabel("Value", fontsize=_FS_AXIS, labelpad=8)
    _set_title(ax, title)

    legend = ax.legend(
        handles=patches, title="Scenario",
        title_fontsize=_FS_SMALL, fontsize=_FS_SMALL,
        frameon=True, framealpha=0.95, edgecolor=_BORDER,
        bbox_to_anchor=(1.01, 1), loc="upper left",
    )
    legend.get_title().set_color(_TEXT_MAIN)

    fig.tight_layout(pad=1.8, rect=(0, 0, 0.86, 1))
    return _save(fig, save_path)


def plot_accuracy_coverage(
        sweep: List[ThresholdPoint],
        scenario_label: str = "",
        title: str = "Accuracy / Coverage Trade-off",
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Accuracy/coverage curve for a single scenario with dual y-axes."""
    pts = sorted(sweep, key=lambda p: p.threshold)
    thresholds = np.array([p.threshold for p in pts])
    accuracy = np.array([p.conditional_accuracy for p in pts])
    coverage = np.array([p.coverage for p in pts])
    best = int(np.argmax(accuracy * coverage))

    full_title = f"{title}  —  {scenario_label}" if scenario_label else title

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    fig.set_facecolor(_BG)
    ax2 = ax1.twinx()

    ax1.fill_between(thresholds, accuracy, alpha=0.08, color=_BLUE, zorder=1)
    ax2.fill_between(thresholds, coverage, alpha=0.06, color=_RED,  zorder=1)

    l1, = ax1.plot(thresholds, accuracy, color=_BLUE, linewidth=2.4,
                   label="Conditional Accuracy", zorder=4)
    l2, = ax2.plot(thresholds, coverage,  color=_RED,  linewidth=2.4,
                   linestyle=(0, (5, 3)), label="Coverage", zorder=4)

    ax1.axvline(thresholds[best], color=_TEXT_SUB,
                linestyle=":", linewidth=1.4, zorder=2)
    for _ax, arr, col in ((ax1, accuracy, _BLUE), (ax2, coverage, _RED)):
        _ax.scatter([thresholds[best]], [arr[best]],
                    color=col, s=90, zorder=6,
                    edgecolors=_SURFACE, linewidths=1.8)

    ax1.annotate(
        f"  θ* = {thresholds[best]:.3f}",
        xy=(thresholds[best], 0.03),
        xycoords=("data", "axes fraction"),
        ha="left", va="bottom",
        fontsize=_FS_ANNOT, color=_TEXT_SUB, fontweight="medium",
        clip_on=False,
    )

    for _ax in (ax1, ax2):
        _ax.set_facecolor(_SURFACE)
        _ax.spines["top"].set_visible(False)
        _ax.spines["bottom"].set_color(_BORDER)
        _ax.tick_params(labelsize=_FS_TICK, length=3, pad=4)
        _ax.set_ylim(-0.02, 1.10)
        _ax.yaxis.set_major_formatter(_pct_fmt())

    ax1.spines["left"].set_color(_BLUE)
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_color(_RED)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", colors=_BLUE)
    ax2.tick_params(axis="y", colors=_RED)
    ax1.set_ylabel("Conditional Accuracy", fontsize=_FS_AXIS,
                   labelpad=8, color=_BLUE)
    ax2.set_ylabel("Coverage", fontsize=_FS_AXIS, labelpad=8, color=_RED)
    ax1.set_xlabel("Decision Threshold", fontsize=_FS_AXIS,
                   labelpad=8, color=_TEXT_MAIN)
    ax1.tick_params(axis="x", colors=_TEXT_SUB)
    ax1.grid(color=_GRID_CLR, linewidth=0.8, linestyle="--", zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title(full_title, fontsize=_FS_TITLE, fontweight="bold",
                  color=_TEXT_MAIN, pad=16)

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()],
               loc="lower left", fontsize=_FS_SMALL,
               frameon=True, framealpha=0.94, edgecolor=_BORDER)

    fig.tight_layout(pad=1.8)
    return _save(fig, save_path)

def plot_accuracy_coverage_multi(
        sweeps: Dict[str, List[ThresholdPoint]],
        title: str = "Accuracy / Coverage — Multiple Scenarios",
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid accuracy/coverage curves with mean ± 1 std band."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    fig.set_facecolor(_BG)

    configs = [
        ("conditional_accuracy", "Conditional Accuracy"),
        ("coverage",             "Coverage"),
    ]

    all_t = [
        np.array([p.threshold
                  for p in sorted(sw, key=lambda p: p.threshold)])
        for sw in sweeps.values()
    ]
    t_min = max(t[0] for t in all_t)
    t_max = min(t[-1] for t in all_t)
    common_t = np.linspace(t_min, t_max, 400)

    for ax, (field, ylabel) in zip(axes, configs):
        curves: list[np.ndarray] = []

        for idx, (label, sweep) in enumerate(sweeps.items()):
            pts = sorted(sweep, key=lambda p: p.threshold)
            t_arr = np.array([p.threshold for p in pts])
            v_arr = np.array([getattr(p, field) for p in pts])
            vi = np.interp(common_t, t_arr, v_arr)
            curves.append(vi)
            ax.plot(common_t, vi,
                    color=_PALETTE[idx % len(_PALETTE)],
                    linewidth=1.7, alpha=0.72, label=label, zorder=3)

        stk = np.stack(curves)
        mean = stk.mean(axis=0)
        std = stk.std(axis=0)
        ax.plot(common_t, mean, color=_TEXT_MAIN, linewidth=2.6,
                linestyle="--", label="Mean", zorder=5)
        ax.fill_between(
            common_t,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=_TEXT_MAIN, alpha=0.10, label="±1 std", zorder=2,
        )

        _apply_base_style(ax, ymax=1.10)
        ax.grid(axis="both", color=_GRID_CLR, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Decision Threshold", fontsize=_FS_AXIS,
                      labelpad=8, color=_TEXT_MAIN)
        ax.set_ylabel(ylabel, fontsize=_FS_AXIS, labelpad=8)
        ax.yaxis.set_major_formatter(_pct_fmt())

        if ax is axes[-1]:
            ax.legend(
                fontsize=_FS_SMALL, frameon=True,
                framealpha=0.94, edgecolor=_BORDER,
                bbox_to_anchor=(1.01, 1), loc="upper left",
            )
        else:
            ax.legend(fontsize=_FS_SMALL, frameon=True,
                      framealpha=0.94, edgecolor=_BORDER, loc="best")

    _set_title(axes[0], title, suptitle=True)
    fig.tight_layout(pad=1.8, rect=(0, 0, 0.87, 1))
    return _save(fig, save_path)


def plot_score_distribution(
        results: List,
        title: str = "Correlation Score Distribution",
        bins: int = 40,
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid histograms of correlation scores for correct vs. incorrect IDs."""
    correct = np.array([r.correlation_score for r in results
                           if getattr(r, "correct", False)])
    incorrect = np.array([r.correlation_score for r in results
                           if not getattr(r, "correct", False)])

    all_sc = np.concatenate([correct, incorrect])
    edges = np.linspace(all_sc.min(), all_sc.max(), bins + 1)

    sep = (float(correct.mean()) - float(incorrect.mean())
           if len(correct) and len(incorrect) else 0.0)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    ax.hist(correct,   bins=edges, alpha=0.52, color=_GREEN,
            label=f"Correct   (n = {len(correct):,})",
            density=True, edgecolor=_SURFACE, linewidth=0.5, zorder=3)
    ax.hist(incorrect, bins=edges, alpha=0.52, color=_RED,
            label=f"Incorrect (n = {len(incorrect):,})",
            density=True, edgecolor=_SURFACE, linewidth=0.5, zorder=3)

    for scores, col in ((correct, _GREEN), (incorrect, _RED)):
        if len(scores):
            cnts, es = np.histogram(scores, bins=edges, density=True)
            mids     = (es[:-1] + es[1:]) / 2
            ax.step(np.r_[es[0], mids, es[-1]], np.r_[0, cnts, 0],
                    color=col, linewidth=1.6, alpha=0.9, zorder=4)

    _apply_base_style(ax, ymax=None)
    ax.autoscale(axis="y")
    ax.margins(y=0.14)
    fig.canvas.draw()
    y_top = ax.get_ylim()[1]
    x_span = float(all_sc.max() - all_sc.min())

    offsets_y = [y_top * 0.97, y_top * 0.82]
    for i, (scores, col, ha) in enumerate(
            ((correct, _GREEN, "left"), (incorrect, _RED, "right"))
    ):
        if not len(scores):
            continue
        mu      = float(scores.mean())
        nudge   = x_span * 0.012
        ax.axvline(mu, color=col, linestyle="--",
                   linewidth=1.8, alpha=0.9, zorder=5)
        ax.text(
            mu + (nudge if ha == "left" else -nudge),
            offsets_y[i],
            f"μ = {mu:.3f}",
            color=col, fontsize=_FS_ANNOT, va="top", ha=ha,
            fontweight="semibold", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=_SURFACE,
                      edgecolor=col, alpha=0.88, linewidth=0.9),
        )

    badge_col = _GREEN if sep >= 0 else _RED
    ax.text(
        0.974, 0.972,
        f"Separation\n{sep:+.4f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=_FS_ANNOT + 0.5, color=badge_col, fontweight="semibold",
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=_SURFACE,
                  edgecolor=badge_col, alpha=0.96, linewidth=1.2),
    )

    ax.grid(axis="both", color=_GRID_CLR, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Correlation Score", fontsize=_FS_AXIS,
                  labelpad=8, color=_TEXT_MAIN)
    ax.set_ylabel("Density", fontsize=_FS_AXIS, labelpad=8)
    ax.legend(fontsize=_FS_SMALL, frameon=True,
              framealpha=0.94, edgecolor=_BORDER, loc="upper left")
    _set_title(ax, title)

    fig.tight_layout(pad=1.8)
    return _save(fig, save_path)


def plot_seed_variance_box(
        scenario_per_seed: Dict[str, List[IdentificationMetrics]],
        metric: str = "success_rate",
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Box plot showing seed-to-seed variance of one metric for each scenario."""
    label = _METRIC_LABELS.get(metric, metric)
    if title is None:
        title = f"Seed Variance  —  {label}"

    names = list(scenario_per_seed.keys())
    data  = [[abs(m.to_dict()[metric]) for m in scenario_per_seed[n]] for n in names]
    n     = len(names)

    fig, ax = plt.subplots(figsize=(max(7.5, n * 1.9 + 2.5), 5.5))

    bp = ax.boxplot(
        data,
        patch_artist = True,
        notch = False,
        widths = 0.46,
        zorder = 3,
        medianprops = dict(color=_TEXT_MAIN, linewidth=2.2),
        whiskerprops = dict(linewidth=1.3, linestyle=(0, (4, 3))),
        capprops = dict(linewidth=1.5),
        flierprops = dict(marker="o", markersize=4,
                            linestyle="none", alpha=0.5),
    )

    for patch, color in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.50)
        patch.set_linewidth(1.3)

    for element in ("whiskers", "caps"):
        for line, color in zip(
            bp[element],
            [c for c in _PALETTE for _ in (0, 1)][:len(bp[element])],
        ):
            line.set_color(color)
            line.set_alpha(0.7)

    rng = np.random.default_rng(42)
    for xi, (vals, color) in enumerate(zip(data, _PALETTE), start=1):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            np.full(len(vals), xi) + jitter, vals,
            color=color, s=36, zorder=5,
            alpha=0.88, edgecolors=_SURFACE, linewidths=1.0,
        )

    for xi, vals in enumerate(data, start=1):
        med     = float(np.median(vals))
        is_last = (xi == n)
        x_off   = -0.30 if is_last else +0.30
        ha      = "right" if is_last else "left"
        ax.text(
            xi + x_off, med,
            f"{med:.2%}",
            ha=ha, va="center",
            fontsize=_FS_ANNOT, color=_TEXT_MAIN,
            fontweight="semibold", zorder=6, clip_on=False,
        )

    all_vals = [v for vals in data for v in vals]
    ymin = max(0.0, min(all_vals) - 0.06)
    ymax = min(1.0, max(all_vals) + 0.10)

    _apply_base_style(ax, ymax=None)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(_pct_fmt())
    ax.set_xticks(range(1, n + 1))
    rotation = 0 if n <= 4 else 18
    ax.set_xticklabels(
        names, fontsize=_FS_TICK, color=_TEXT_MAIN,
        rotation=rotation, ha="right" if rotation else "center",
    )
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylabel(label, fontsize=_FS_AXIS, labelpad=8)
    _set_title(ax, title)

    fig.tight_layout(pad=1.8)
    return _save(fig, save_path)