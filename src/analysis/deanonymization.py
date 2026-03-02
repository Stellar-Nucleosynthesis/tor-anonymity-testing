from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.analysis.metrics import (
    compare_scenarios,
    compute_identification_metrics,
    compute_seed_variance,
    compute_timing_metrics,
    statistical_significance,
)


@dataclass
class DeanonymizationResult:
    """One top-1 identification attempt for a single guard-side circuit.

    Attributes:
        client_id: Unique identifier for the relay.
        circuit_id: Scoped ``"hostname/local_cid"`` string from the guard log.
        true_guard: Hostname of the actual guard relay used by this circuit.
        true_exit: Hostname of the actual exit relay used by this circuit.
        predicted_guard: Guard hostname the attack attributed this circuit to,
            or ``None`` when the adversary does not control the guard.
        predicted_exit: Exit hostname ranked first by the attack, or ``None``
            when no candidate cleared the threshold.
        confidence: Normalised confidence in the top-1 prediction (0–1).
        correlation_score: Raw primary correlation score for the top-ranked
            candidate (e.g. cross-correlation coefficient).
        time_to_identify: Wall-clock seconds spent on this circuit.
        successful: ``True`` when the top-ranked candidate is the correct
            exit and the score clears the decision threshold.
    """

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


def evaluate_attack(
    results: List[DeanonymizationResult],
    total_observed: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute identification and timing metrics for one attack run.

    Args:
        results: Deanonymization results for one seed or aggregated across
            seeds.
        total_observed: Total guard profiles the attack received before
            candidate pre-filtering.  When ``None``, ``len(results)`` is used
            as the denominator so ``coverage = 1.0``.

    Returns:
        Merged dict from ``compute_identification_metrics`` and
        ``compute_timing_metrics``.  All keys are documented in
        ``src/analysis/metrics.py``.
    """
    metrics = compute_identification_metrics(results, total_observed=total_observed)
    metrics.update(compute_timing_metrics(results))
    return metrics

__all__ = [
    "DeanonymizationResult",
    "evaluate_attack",
    "compare_scenarios",
    "compute_seed_variance",
    "statistical_significance",
]