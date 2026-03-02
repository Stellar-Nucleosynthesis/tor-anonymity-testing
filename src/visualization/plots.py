from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

MetricsDict = Dict[str, Any]
ScenarioMetrics = Dict[str, MetricsDict]
ThresholdSweep = List[Dict[str, float]]

def plot_success_rate_bar(
    scenario_metrics: ScenarioMetrics,
    metrics: Optional[List[str]] = None,
    title: str = "Attack Scenario Comparison",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing key identification metrics across scenarios.

    Each group of bars corresponds to one metric; each bar within the group
    corresponds to one scenario.  Error bars are drawn when ``std_{metric}``
    keys are present in the metric dict (i.e. after seed aggregation).

    Args:
        scenario_metrics: Mapping of scenario label → metrics dict.  Should
            contain at least ``success_rate``, ``coverage``, and
            ``conditional_accuracy``.
        metrics: Subset of metric keys to display.  Defaults to
            ``["success_rate", "coverage", "conditional_accuracy",
               "score_separation"]``.
        title: Figure title.
        save_path: When provided, the figure is saved here before returning.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_success_rate_line(
    parameter_values: List[float],
    scenario_metrics: List[MetricsDict],
    parameter_label: str = "Parameter",
    metric_keys: Optional[List[str]] = None,
    title: str = "Success Rate vs Parameter",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Line chart of one or more metrics as a function of a single parameter.

    Useful for sweeping a single configuration knob (e.g. ``guard_fraction``)
    across otherwise-identical scenarios and visualizing how performance
    changes.  Multiple metric lines can be overlaid on the same axes.

    Args:
        parameter_values: Ordered list of parameter values on the x-axis (e.g.
            ``[0.05, 0.10, 0.20, 0.30]``).
        scenario_metrics: One metrics dict per parameter value, in the same
            order as *parameter_values*.
        parameter_label: x-axis label string.
        metric_keys: Metric keys to plot as separate lines.  Defaults to
            ``["success_rate", "coverage", "conditional_accuracy"]``.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_seed_variance_box(
    scenario_per_seed: Dict[str, List[MetricsDict]],
    metric: str = "success_rate",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Box plot showing seed-to-seed variance for each scenario.

    One box per scenario, y-axis is the chosen metric across seeds.  Reveals
    whether performance differences between scenarios are consistent or
    artifacts of a particular seed.

    Args:
        scenario_per_seed: Mapping of scenario label → list of per-seed metric
            dicts (one dict per seed run).
        metric: The metric key to plot on the y-axis.
        title: Figure title.  Defaults to ``"Seed Variance — {metric}"``.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_significance_heatmap(
    scenario_labels: List[str],
    p_value_matrix: np.ndarray,
    title: str = "Pairwise Statistical Significance (Mann-Whitney)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of Mann-Whitney p-values for all pairs of scenarios.

    Cells are colored green (significant, p < 0.05) or red (not significant).
    The diagonal is always white (trivially p = 1).  Scenario labels appear on
    both axes.

    Args:
        scenario_labels: Ordered list of scenario labels; must match the row/
            column order of *p_value_matrix*.
        p_value_matrix: Square matrix of shape ``(n, n)`` where entry
            ``[i, j]`` is the p-value from ``statistical_significance(a, b)``.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Single-simulation diagnostics
# ---------------------------------------------------------------------------

def plot_accuracy_coverage(
    sweep: ThresholdSweep,
    scenario_label: str = "",
    title: str = "Accuracy / Coverage Trade-off",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Accuracy/coverage curve for a single simulation run.

    Plots ``conditional_accuracy`` on the primary y-axis and ``coverage`` on
    a secondary y-axis, both as functions of the decision threshold.  A
    vertical dashed line marks the threshold at which the product
    (accuracy × coverage) is maximized — a simple operating-point heuristic.

    Args:
        sweep: Output of ``compute_threshold_sweep`` for one scenario.
        scenario_label: Appended to the title in parentheses when non-empty.
        title: Base figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_score_distribution(
    results: List[Any],
    title: str = "Correlation Score Distribution",
    bins: int = 40,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid histograms of correlation scores for correct vs. incorrect identifications.

    A clear separation between the two distributions indicates good
    discriminability.  ``score_separation`` from ``compute_identification_metrics``
    is annotated on the plot.

    Args:
        results: List of ``DeanonymizationResult`` objects.
        title: Figure title.
        bins: Number of histogram bins.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_timing_distribution(
    scenario_results: Dict[str, List[Any]],
    metric: str = "time_to_identify",
    title: str = "Time-to-Identify Distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Side-by-side box plots of identification timing across scenarios.

    One box per scenario.  A separate panel can show only the successful
    identifications.

    Args:
        scenario_results: Mapping of scenario label → list of
            ``DeanonymizationResult`` objects.
        metric: Attribute of ``DeanonymizationResult`` to plot.  Must be a
            numeric field (``time_to_identify`` or ``correlation_score``).
        title: Figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_accuracy_coverage_multi(
    sweeps: Dict[str, ThresholdSweep],
    title: str = "Accuracy / Coverage — Multiple Seeds",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid accuracy/coverage curves for multiple seeds or scenarios.

    Each curve corresponds to one entry in *sweeps*.  A shaded band can be
    drawn across seeds when the keys represent seed indices.

    Args:
        sweeps: Mapping of label → threshold sweep list (output of
            ``compute_threshold_sweep``).
        title: Figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_seed_success_timeline(
    seed_labels: List[str],
    seed_metrics: List[MetricsDict],
    metric: str = "success_rate",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Line chart of a metric in seed order, with optional confidence band.

    Useful for checking whether performance drifts across simulation seeds or
    remains stable.

    Args:
        seed_labels: Ordered seed labels (e.g. ``["seed_0", "seed_1", …]``).
        seed_metrics: One metrics dict per seed in the same order.
        metric: Metric key to plot on the y-axis.
        title: Figure title.  Defaults to ``"{metric} Across Seeds"``.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_guard_exit_matrix(
    guard_fractions: List[float],
    exit_fractions: List[float],
    success_rate_matrix: np.ndarray,
    metric_label: str = "Success Rate",
    title: str = "Guard × Exit Fraction — Success Rate",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of success rate as a function of guard and exit adversary fractions.

    Rows are guard fractions, columns are exit fractions.  Cell color
    encodes the metric value (darker = higher).  Annotates each cell with
    the exact value.

    Args:
        guard_fractions: List of guard-fraction values (y-axis).
        exit_fractions: List of exit-fraction values (x-axis).
        success_rate_matrix: 2-D array of shape
            ``(len(guard_fractions), len(exit_fractions))``.
        metric_label: Colour-bar label and annotation unit.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_correlation_matrix(
    correlation_matrix: np.ndarray,
    guard_ids: Optional[List[str]] = None,
    exit_ids: Optional[List[str]] = None,
    title: str = "Traffic Correlation Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of pairwise correlation scores between guard and exit profiles.

    Rows are guard observations, columns are exit observations.  The
    ground-truth diagonal (where guard and exit belong to the same circuit) is
    outlined when it can be inferred.

    Args:
        correlation_matrix: Array of shape ``(n_guards, n_exits)`` where entry
            ``[i, j]`` is the correlation score between guard profile *i* and
            exit profile *j*.
        guard_ids: Row labels (circuit IDs for guard-side profiles).
        exit_ids: Column labels (circuit IDs for exit-side profiles).
        title: Figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError