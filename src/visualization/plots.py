from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.metrics import IdentificationMetrics, ThresholdPoint


def plot_success_rate_bar(
        scenario_metrics: Dict[str, IdentificationMetrics],
        metrics: Optional[List[str]] = None,
        title: str = "Attack Scenario Comparison",
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing key identification metrics across scenarios.

    Each group of bars corresponds to one metric; each bar within the group
    corresponds to one scenario.  Error bars are drawn when ``seed_variance``
    is provided alongside the metrics (i.e. ``std_*`` fields are non-zero).

    Args:
        scenario_metrics: Mapping of scenario display label →
            ``IdentificationMetrics``.
        metrics: Subset of ``IdentificationMetrics`` field names to display.
            Defaults to ``["success_rate", "coverage",
            "conditional_accuracy", "score_separation"]``.
        title: Figure title.
        save_path: When provided, the figure is saved here before returning.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_accuracy_coverage(
        sweep: List[ThresholdPoint],
        scenario_label: str = "",
        title: str = "Accuracy / Coverage Trade-off",
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Accuracy/coverage curve for a single scenario.

    Plots ``conditional_accuracy`` on the primary y-axis and ``coverage`` on
    a secondary y-axis, both as functions of the decision threshold.  A
    vertical dashed line marks the threshold that maximizes
    accuracy × coverage — a simple operating-point heuristic.

    Args:
        sweep: Output of ``compute_threshold_sweep`` for one scenario.
        scenario_label: Appended to the title when non-empty.
        title: Base figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_accuracy_coverage_multi(
        sweeps: Dict[str, List[ThresholdPoint]],
        title: str = "Accuracy / Coverage — Multiple Scenarios",
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid accuracy/coverage curves for multiple scenarios or seeds.

    Each curve corresponds to one entry in *sweeps*.  A bold dashed mean line
    and ±1 std shaded band are drawn across all curves.

    Args:
        sweeps: Mapping of display label → ``List[ThresholdPoint]``.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_score_distribution(
        results: List,
        title: str = "Correlation Score Distribution",
        bins: int = 40,
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid histograms of correlation scores for correct vs. incorrect IDs.

    A clear separation between the two distributions indicates good
    discriminability.  The ``score_separation`` value is annotated on the
    plot.

    Args:
        results: List of ``DeanonymizationResult`` objects carrying
            ``correlation_score`` and ``successful`` / ``correct`` attributes.
        title: Figure title.
        bins: Number of histogram bins.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError


def plot_seed_variance_box(
        scenario_per_seed: Dict[str, List[IdentificationMetrics]],
        metric: str = "success_rate",
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
) -> plt.Figure:
    """Box plot showing seed-to-seed variance of one metric for each scenario.

    One box per scenario on the x-axis, the chosen metric on the y-axis.
    Reveals whether performance differences between scenarios are consistent
    or artifacts of a particular seed.

    Args:
        scenario_per_seed: Mapping of display label →
            list of per-seed ``IdentificationMetrics``.
        metric: Field name on ``IdentificationMetrics`` to plot.
        title: Figure title.  Defaults to ``"Seed Variance — {metric}"``.
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

    Rows are guard fractions (y-axis), columns are exit fractions (x-axis).
    Each cell is annotated with the exact value.

    Args:
        guard_fractions: Sorted list of guard-fraction values.
        exit_fractions: Sorted list of exit-fraction values.
        success_rate_matrix: 2-D array of shape
            ``(len(guard_fractions), len(exit_fractions))``.
        metric_label: Colour-bar label.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    raise NotImplementedError