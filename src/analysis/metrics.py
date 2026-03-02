from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_identification_metrics(
    results: List[Any],
    total_observed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute primary top-1 identification metrics.

    Args:
        results: List of ``DeanonymizationResult`` objects.
        total_observed: Total guard profiles given to the attack before
            candidate pre-filtering.  When ``None``, ``len(results)`` is used
            as the denominator, so ``coverage = 1.0``.

    Returns:
        Flat dict with the following keys:

        total_observed, attempted, correct,
        success_rate, coverage, abstention_rate, conditional_accuracy,
        mean_score_correct, mean_score_incorrect, score_separation.
    """
    if not results:
        return _empty_identification_metrics(total_observed or 0)

    n_total = total_observed if total_observed is not None else len(results)
    n_attempted = len(results)
    n_correct = sum(1 for r in results if r.successful)

    coverage = n_attempted / n_total if n_total > 0 else 0.0
    conditional_accuracy = n_correct / n_attempted if n_attempted > 0 else 0.0
    success_rate = n_correct / n_total if n_total > 0 else 0.0

    scores_correct = [r.correlation_score for r in results if r.successful]
    scores_incorrect = [r.correlation_score for r in results if not r.successful]
    mean_correct = float(np.mean(scores_correct)) if scores_correct else 0.0
    mean_incorrect = float(np.mean(scores_incorrect)) if scores_incorrect else 0.0

    return {
        "total_observed":       n_total,
        "attempted":            n_attempted,
        "correct":              n_correct,
        "success_rate":         float(success_rate),
        "coverage":             float(coverage),
        "abstention_rate":      float(1.0 - coverage),
        "conditional_accuracy": float(conditional_accuracy),
        "mean_score_correct":   mean_correct,
        "mean_score_incorrect": mean_incorrect,
        "score_separation":     float(mean_correct - mean_incorrect),
    }


def compute_timing_metrics(results: List[Any]) -> Dict[str, float]:
    """Compute timing statistics over all attempted identifications.

    Both successful and unsuccessful results are included so the caller can
    understand the full cost of running the attack.

    Args:
        results: List of ``DeanonymizationResult`` objects.

    Returns:
        Dict with ``{mean,median,p95,min,max,std}_time_s`` for all results,
        and ``correct_{mean,median,p95,min,max,std}_time_s`` for successful
        identifications only.  All values are wall-clock seconds.
    """
    if not results:
        return {k: 0.0 for k in _timing_keys()}

    times_all = np.array([r.time_to_identify for r in results])
    times_correct = np.array([r.time_to_identify for r in results if r.successful])

    def _stats(arr: np.ndarray, prefix: str = "") -> Dict[str, float]:
        if len(arr) == 0:
            return {f"{prefix}{s}": 0.0
                    for s in ("mean_time_s", "median_time_s", "p95_time_s",
                              "min_time_s",  "max_time_s",  "std_time_s")}
        return {
            f"{prefix}mean_time_s":   float(np.mean(arr)),
            f"{prefix}median_time_s": float(np.median(arr)),
            f"{prefix}p95_time_s":    float(np.percentile(arr, 95)),
            f"{prefix}min_time_s":    float(np.min(arr)),
            f"{prefix}max_time_s":    float(np.max(arr)),
            f"{prefix}std_time_s":    float(np.std(arr)),
        }

    return {**_stats(times_all), **_stats(times_correct, "correct_")}


def compute_threshold_sweep(
    results: List[Any],
    total_observed: Optional[int] = None,
    n_thresholds: int = 100,
    threshold_range: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, float]]:
    """Evaluate accuracy and coverage at a range of decision thresholds.

    For each threshold *t*, a prediction is made only when the best
    correlation score ≥ *t*.  As *t* increases:

    - coverage falls (the attack abstains more often),
    - conditional accuracy rises (only high-confidence guesses pass).

    Args:
        results: List of ``DeanonymizationResult`` objects, each carrying
            ``correlation_score`` (score at the chosen threshold) and
            ``successful``.
        total_observed: Total guard profiles before pre-filtering.
            Defaults to ``len(results)``.
        n_thresholds: Number of evenly-spaced threshold values to evaluate.
        threshold_range: ``(lo, hi)`` override.  Defaults to the score range
            in *results* padded by 1 % on each side.

    Returns:
        List of dicts sorted by ascending threshold.  Each dict contains:
        ``threshold``, ``coverage``, ``conditional_accuracy``,
        ``success_rate``, ``attempted``, ``correct``.
    """
    if not results:
        return []

    n_total = total_observed if total_observed is not None else len(results)
    scores  = np.array([r.correlation_score for r in results])

    if threshold_range is not None:
        lo, hi = threshold_range
    else:
        lo, hi = float(scores.min()), float(scores.max())
        pad = max((hi - lo) * 0.01, 1e-6)
        lo -= pad
        hi += pad

    sweep: List[Dict[str, float]] = []
    for t in np.linspace(lo, hi, n_thresholds):
        mask = scores >= t
        attempted = int(mask.sum())
        correct = sum(1 for r, a in zip(results, mask) if a and r.successful)

        sweep.append({
            "threshold": float(t),
            "coverage": attempted / n_total if n_total > 0 else 0.0,
            "conditional_accuracy": correct / attempted if attempted > 0 else 0.0,
            "success_rate": correct / n_total if n_total > 0 else 0.0,
            "attempted": attempted,
            "correct": correct,
        })
    return sweep


def compute_seed_variance(per_seed_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize variance of identification metrics across simulation seeds.

    Args:
        per_seed_metrics: One metric dict per seed, each produced by
            ``compute_identification_metrics``.

    Returns:
        Dict with ``mean_*``, ``std_*``, ``min_*``, ``max_*`` for every scalar
        metric key present in at least one seed dict, plus ``num_seeds``.
    """
    if not per_seed_metrics:
        return {"num_seeds": 0}

    scalar_keys = {
        k for d in per_seed_metrics
        for k, v in d.items()
        if isinstance(v, (int, float)) and not np.isnan(float(v))
    }

    out: Dict[str, Any] = {"num_seeds": len(per_seed_metrics)}
    for key in sorted(scalar_keys):
        vals = np.array(
            [d[key] for d in per_seed_metrics
             if key in d and isinstance(d[key], (int, float))],
            dtype=float,
        )
        if len(vals) == 0:
            continue
        out[f"mean_{key}"] = float(np.mean(vals))
        out[f"std_{key}"]  = float(np.std(vals))
        out[f"min_{key}"]  = float(np.min(vals))
        out[f"max_{key}"]  = float(np.max(vals))

    return out


def compare_scenarios(
    scenario_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare identification metrics across multiple attack scenarios.

    Args:
        scenario_metrics: Mapping of scenario label → metric dict as produced
            by ``compute_identification_metrics`` (optionally merged with
            ``compute_timing_metrics``).

    Returns:
        Dict with:

        * ``scenarios`` — list of labels.
        * ``metrics_comparison`` — per-metric sub-dict with ``values``,
          ``best_scenario``, ``mean``, and ``std`` across scenarios.
    """
    lower_is_better = {"abstention_rate", "mean_time_s", "mean_score_incorrect"}

    key_metrics = [
        "success_rate",
        "coverage",
        "conditional_accuracy",
        "abstention_rate",
        "score_separation",
        "mean_time_s",
        "correct",
        "attempted",
    ]

    comparison: Dict[str, Any] = {
        "scenarios": list(scenario_metrics.keys()),
        "metrics_comparison": {},
    }

    for metric in key_metrics:
        values: Dict[str, float] = {
            label: float(m[metric])
            for label, m in scenario_metrics.items()
            if metric in m and isinstance(m.get(metric), (int, float))
        }
        if not values:
            continue

        best = min(values, key=values.get) if metric in lower_is_better \
            else max(values, key=values.get)
        arr  = np.array(list(values.values()))

        comparison["metrics_comparison"][metric] = {
            "values":        values,
            "best_scenario": best,
            "mean":          float(np.mean(arr)),
            "std":           float(np.std(arr)),
        }

    return comparison


def statistical_significance(
    results_a: List[Any],
    results_b: List[Any],
) -> Dict[str, Any]:
    """Test whether two scenarios differ significantly in identification success.

    Uses a Mann-Whitney U-test on correlation scores — valid for unequal
    sample sizes and makes no normality assumption.

    Args:
        results_a: ``DeanonymizationResult`` list from scenario A.
        results_b: ``DeanonymizationResult`` list from scenario B.

    Returns:
        Dict with ``statistic``, ``p_value``, ``significant`` (p < 0.05).
    """
    from scipy.stats import mannwhitneyu

    scores_a = np.array([r.correlation_score for r in results_a])
    scores_b = np.array([r.correlation_score for r in results_b])
    stat, p  = mannwhitneyu(scores_a, scores_b, alternative="two-sided")

    return {
        "test":        "mann_whitney_u",
        "statistic":   float(stat),
        "p_value":     float(p),
        "significant": bool(p < 0.05),
    }


def _empty_identification_metrics(total_observed: int) -> Dict[str, Any]:
    return {
        "total_observed":       total_observed,
        "attempted":            0,
        "correct":              0,
        "success_rate":         0.0,
        "coverage":             0.0,
        "abstention_rate":      1.0,
        "conditional_accuracy": 0.0,
        "mean_score_correct":   0.0,
        "mean_score_incorrect": 0.0,
        "score_separation":     0.0,
    }


def _timing_keys() -> List[str]:
    prefixes = ("", "correct_")
    suffixes = ("mean_time_s", "median_time_s", "p95_time_s",
                "min_time_s",  "max_time_s",    "std_time_s")
    return [p + s for p in prefixes for s in suffixes]