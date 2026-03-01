from typing import Optional, List, Dict
from dataclasses import dataclass
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

from src.analysis.metrics import compute_basic_metrics, compute_roc_metrics


@dataclass
class DeanonymizationResult:
    """Single deanonymization attempt result"""
    client_id: str
    circuit_id: str
    true_guard: str
    true_exit: str
    predicted_guard: Optional[str]
    predicted_exit: Optional[str]
    confidence: float
    correlation_score: float
    time_to_identify: float
    successful: bool


def compute_time_to_deanonymize(results: List[DeanonymizationResult]) -> Dict[str, float]:
    """
    Compute statistics about time to deanonymize.

    Args:
        results: List of deanonymization results

    Returns:
        Dictionary with timing statistics
    """
    successful_times = [r.time_to_identify for r in results if r.successful]

    if not successful_times:
        return {
            'mean_time': float('inf'),
            'median_time': float('inf'),
            'min_time': float('inf'),
            'max_time': float('inf'),
            'std_time': 0
        }

    return {
        'mean_time': np.mean(successful_times),
        'median_time': np.median(successful_times),
        'min_time': np.min(successful_times),
        'max_time': np.max(successful_times),
        'std_time': np.std(successful_times),
        'total_successful': len(successful_times),
        'success_rate': len(successful_times) / len(results)
    }


def evaluate_attack(
        results: List[DeanonymizationResult]
) -> Dict[str, any]:
    """
    Comprehensive evaluation of an attack.

    Args:
        results: List of deanonymization results

    Returns:
        Dictionary with all metrics
    """
    y_true = np.array([int(r.successful == r.decision) for r in results])
    y_scores = np.array([r.correlation_score for r in results])
    y_pred = np.array([1 if r.decision else 0 for r in results])

    metrics = {}

    metrics.update(compute_basic_metrics(y_true, y_pred))

    if len(np.unique(y_true)) > 1:
        metrics.update(compute_roc_metrics(y_true, y_scores))

    metrics.update(compute_time_to_deanonymize(results))

    confidences = [r.confidence for r in results]
    metrics['mean_confidence'] = np.mean(confidences)
    metrics['median_confidence'] = np.median(confidences)

    return metrics


def compare_scenarios(scenario_results: Dict[str, Dict[str, any]]) -> Dict[str, any]:
    """
    Compare multiple attack scenarios.

    Args:
        scenario_results: Dictionary mapping scenario names to their metrics

    Returns:
        Comparison analysis
    """
    comparison = {
        'scenarios': list(scenario_results.keys()),
        'metrics_comparison': {}
    }

    key_metrics = ['roc_auc', 'f1_score', 'precision', 'recall',
                   'success_rate', 'mean_time']

    for metric in key_metrics:
        values = {}
        for scenario, results in scenario_results.items():
            if metric in results:
                values[scenario] = results[metric]

        if values:
            comparison['metrics_comparison'][metric] = {
                'values': values,
                'best_scenario': max(values, key=values.get) if metric != 'mean_time'
                else min(values, key=values.get),
                'mean': np.mean(list(values.values())),
                'std': np.std(list(values.values()))
            }

    return comparison


def statistical_significance(
        results1: List[DeanonymizationResult],
        results2: List[DeanonymizationResult],
        test: str = 'wilcoxon'
) -> Dict[str, float]:
    """
    Test statistical significance between two attack scenarios.

    Args:
        results1: Results from first scenario
        results2: Results from second scenario
        test: Statistical test to use ('wilcoxon' or 'mann_whitney')

    Returns:
        Dictionary with test statistics
    """
    scores1 = np.array([r.correlation_score for r in results1])
    scores2 = np.array([r.correlation_score for r in results2])

    if test == 'mann_whitney':
        statistic, p_value = mannwhitneyu(scores1, scores2)
    elif test == 'wilcoxon':
        if len(scores1) != len(scores2):
            raise ValueError("Wilcoxon requires equal sample sizes")
        else:
            statistic, p_value = wilcoxon(scores1, scores2)
    else:
        raise ValueError("Test must be 'wilcoxon' or 'mann_whitney'")

    return {
        'test': test,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }