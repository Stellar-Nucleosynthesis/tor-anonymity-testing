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
        seed: Simulation seed.
        origin_id: Unique identifier for the origin relay.
        circuit_id: Scoped local circuit identifier from the origin.
        attempted: ``True`` if attempt to deanonymize was made.
        successful: ``True`` when the top-ranked candidate is the correct
            exit and the score clears the decision threshold.
        confidence: Normalised confidence in the top-1 prediction (0–1).
        correlation_score: Raw correlation score for the top-ranked
            candidate (e.g. cross-correlation coefficient).
        time_to_identify: Wall-clock seconds spent on this circuit.
    """
    seed: str
    group: Optional[str]
    origin_id: str
    circuit_id: str
    attempted: bool
    successful: bool
    confidence: Optional[float] = None
    correlation_score: Optional[float] = None
    time_to_identify: Optional[float] = None


def evaluate_attack(results: List[DeanonymizationResult]) -> Dict[str, Any]:
    """Compute identification and timing metrics for one attack run.

    Args:
        results: Deanonymization results for one seed or aggregated across
            seeds.

    Returns:
        Merged dict from ``compute_identification_metrics`` and
        ``compute_timing_metrics``.  All keys are documented in
        ``src/analysis/metrics.py``.
    """
    metrics = compute_identification_metrics(results)
    metrics.update(compute_timing_metrics(results))
    return metrics

__all__ = [
    "DeanonymizationResult",
    "evaluate_attack",
    "compare_scenarios",
    "compute_seed_variance",
    "statistical_significance",
]