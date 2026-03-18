from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

@dataclass
class IdentificationMetrics:
    """Primary top-1 identification metrics for one scenario / seed."""
    total_observed: int
    attempted: int
    correct: int
    success_rate: float
    coverage: float
    abstention_rate: float
    conditional_accuracy: float
    mean_score_correct: float
    mean_score_incorrect: float
    score_separation: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdPoint:
    """Identification metrics evaluated at a single decision threshold."""
    threshold: float
    coverage: float
    conditional_accuracy: float
    success_rate: float
    attempted: int
    correct: int


@dataclass
class SeedVarianceMetrics:
    """Aggregated mean / std / min / max of a scalar metric across seeds."""
    num_seeds: int
    stats: Dict[str, Dict[str, float]]

    def mean(self, key: str) -> float:
        return self.stats[key]["mean"]

    def std(self, key: str) -> float:
        return self.stats[key]["std"]

    def min(self, key: str) -> float:
        return self.stats[key]["min"]

    def max(self, key: str) -> float:
        return self.stats[key]["max"]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"num_seeds": self.num_seeds}
        for key, s in self.stats.items():
            for stat, val in s.items():
                out[f"{stat}_{key}"] = val
        return out


@dataclass
class MetricComparison:
    """Comparison of one metric across multiple scenarios."""
    values: Dict[str, float]
    best_scenario: str
    mean: float
    std: float


@dataclass
class ScenarioComparison:
    """Result of comparing identification metrics across multiple scenarios."""
    scenarios: List[str]
    metrics_comparison: Dict[str, MetricComparison]

    def best(self, metric: str) -> str:
        return self.metrics_comparison[metric].best_scenario

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenarios": self.scenarios,
            "metrics_comparison": {
                k: asdict(v) for k, v in self.metrics_comparison.items()
            },
        }


@dataclass
class StatisticalSignificanceResult:
    """Result of a two-sample significance test between two scenarios."""
    test: str
    statistic: float
    p_value: float
    significant: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_identification_metrics(
    results: List[Any],
) -> IdentificationMetrics:
    """Compute primary top-1 identification metrics.

    Args:
        results: List of ``DeanonymizationResult`` objects.

    Returns:
        ``IdentificationMetrics`` dataclass instance.
    """
    if not results:
        return IdentificationMetrics(
            total_observed=0,
            attempted=0,
            correct=0,
            success_rate=0.0,
            coverage=0.0,
            abstention_rate=1.0,
            conditional_accuracy=0.0,
            mean_score_correct=0.0,
            mean_score_incorrect=0.0,
            score_separation=0.0,
        )

    attempted = [r for r in results if r.attempted]

    n_total = len(results)
    n_attempted = len(attempted)
    n_correct = sum(1 for r in attempted if r.successful)

    coverage = n_attempted / n_total if n_total > 0 else 0.0
    conditional_accuracy = n_correct / n_attempted if n_attempted > 0 else 0.0
    success_rate = n_correct / n_total if n_total > 0 else 0.0

    scores_correct = [r.correlation_score for r in attempted if r.successful]
    scores_incorrect = [r.correlation_score for r in attempted if not r.successful]
    mean_correct = float(np.mean(scores_correct)) if scores_correct else 0.0
    mean_incorrect = float(np.mean(scores_incorrect)) if scores_incorrect else 0.0

    return IdentificationMetrics(
        total_observed = n_total,
        attempted = n_attempted,
        correct = n_correct,
        success_rate = float(success_rate),
        coverage = float(coverage),
        abstention_rate = float(1.0 - coverage),
        conditional_accuracy = float(conditional_accuracy),
        mean_score_correct = mean_correct,
        mean_score_incorrect = mean_incorrect,
        score_separation = float(mean_correct - mean_incorrect),
    )


def compute_threshold_sweep(
    results: List[Any],
    n_thresholds: int = 100,
    threshold_range: Optional[Tuple[float, float]] = None,
) -> List[ThresholdPoint]:
    """Evaluate accuracy and coverage at a range of decision thresholds.

    Args:
        results: List of ``DeanonymizationResult`` objects.
        n_thresholds: Number of evenly-spaced threshold values to evaluate.
        threshold_range: ``(lo, hi)`` override.  Defaults to the score range
            padded by 1 % on each side.

    Returns:
        List of ``ThresholdPoint`` instances sorted by ascending threshold.
    """
    if not results:
        return []

    n_total = len(results)
    attempted = [r for r in results if r.attempted]
    if not attempted:
        return []

    scores = np.array([r.correlation_score for r in attempted])

    if threshold_range is not None:
        lo, hi = threshold_range
    else:
        lo, hi = float(scores.min()), float(scores.max())
        pad = max((hi - lo) * 0.01, 1e-6)
        lo -= pad
        hi += pad

    sweep: List[ThresholdPoint] = []
    for t in np.linspace(lo, hi, n_thresholds):
        mask = scores >= t
        n_att = int(mask.sum())
        n_correct = sum(1 for r, a in zip(attempted, mask) if a and r.successful)
        sweep.append(ThresholdPoint(
            threshold = float(t),
            coverage = n_att / n_total if n_total > 0 else 0.0,
            conditional_accuracy = n_correct / n_att if n_att > 0 else 0.0,
            success_rate = n_correct / n_total if n_total > 0 else 0.0,
            attempted = n_att,
            correct = n_correct,
        ))
    return sweep


def compute_seed_variance(
    per_seed_metrics: List[IdentificationMetrics],
) -> SeedVarianceMetrics:
    """Summarize variance of identification metrics across simulation seeds.

    Args:
        per_seed_metrics: One ``IdentificationMetrics`` per seed.

    Returns:
        ``SeedVarianceMetrics`` instance.
    """
    if not per_seed_metrics:
        return SeedVarianceMetrics(num_seeds=0, stats={})

    scalar_fields = [f.name for f in fields(IdentificationMetrics)
                     if f.type in ("float", "int")]

    stats: Dict[str, Dict[str, float]] = {}
    for key in scalar_fields:
        vals = np.array([getattr(m, key) for m in per_seed_metrics], dtype=float)
        stats[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return SeedVarianceMetrics(num_seeds=len(per_seed_metrics), stats=stats)


def compare_scenarios(
    scenario_metrics: Dict[str, IdentificationMetrics],
) -> ScenarioComparison:
    """Compare identification metrics across multiple attack scenarios.

    Args:
        scenario_metrics: Mapping of scenario label → ``IdentificationMetrics``.

    Returns:
        ``ScenarioComparison`` dataclass instance.
    """
    lower_is_better = {"abstention_rate", "mean_time_s", "mean_score_incorrect"}

    key_metrics = [
        "success_rate",
        "coverage",
        "conditional_accuracy",
        "abstention_rate",
        "score_separation",
        "correct",
        "attempted",
    ]

    comparisons: Dict[str, MetricComparison] = {}
    for metric in key_metrics:
        values: Dict[str, float] = {
            label: float(getattr(m, metric))
            for label, m in scenario_metrics.items()
            if hasattr(m, metric)
        }
        if not values:
            continue
        best = (min(values, key=values.__getitem__)
                if metric in lower_is_better
                else max(values, key=values.__getitem__))
        arr = np.array(list(values.values()))
        comparisons[metric] = MetricComparison(
            values = values,
            best_scenario = best,
            mean = float(np.mean(arr)),
            std = float(np.std(arr)),
        )

    return ScenarioComparison(
        scenarios = list(scenario_metrics.keys()),
        metrics_comparison = comparisons,
    )


def statistical_significance(
    results_a: List[Any],
    results_b: List[Any],
) -> StatisticalSignificanceResult:
    """Test whether two scenarios differ significantly in identification success.

    Uses a Mann-Whitney U-test on correlation scores — valid for unequal
    sample sizes, no normality assumption required.

    Args:
        results_a: ``DeanonymizationResult`` list from scenario A.
        results_b: ``DeanonymizationResult`` list from scenario B.

    Returns:
        ``StatisticalSignificanceResult`` dataclass instance.
    """
    from scipy.stats import mannwhitneyu

    scores_a = np.array([r.correlation_score for r in results_a])
    scores_b = np.array([r.correlation_score for r in results_b])
    stat, p  = mannwhitneyu(scores_a, scores_b, alternative="two-sided")

    return StatisticalSignificanceResult(
        test = "mann_whitney_u",
        statistic = float(stat),
        p_value = float(p),
        significant = bool(p < 0.05),
    )