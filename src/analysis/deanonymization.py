from dataclasses import dataclass
from typing import List, Optional

from src.analysis.metrics import IdentificationMetrics, compute_identification_metrics


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


def evaluate_attack(results: List[DeanonymizationResult]) -> IdentificationMetrics:
    """Compute identification and timing metrics for one attack run.

    Args:
        results: Deanonymization results for one seed or aggregated across
            seeds.

    Returns:
        An object that contains statistical metrics of an attack.
    """
    metrics = compute_identification_metrics(results)
    return metrics