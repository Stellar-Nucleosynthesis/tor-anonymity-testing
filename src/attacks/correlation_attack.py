from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.attacks.base_attack import AttackConfig, BaseAttack
from src.analysis.correlation import CorrelationAnalyzer, CorrelationConfig

@dataclass
class CorrelationAttackConfig(AttackConfig):
    """Configuration for a traffic correlation attack

    Attributes:
        correlation_method: A method names passed to ``CorrelationAnalyzer``.
        correlation_threshold: Decision thresholds passed to ``CorrelationAnalyzer``.
        time_window: Correlation time window passed to ``CorrelationAnalyzer``.
        bin_size: Width in seconds of each time bin used for traffic
            histogramming.
    """
    correlation_method: str = "cross_correlation"
    correlation_threshold: float = 0.9
    time_window: float = 30
    bin_size: float = 0.1


class CorrelationAttack(BaseAttack, ABC):
    """Abstract class for Tor traffic correlation attack simulations.
    """

    ATTACK_NAME: str = "traffic_correlation_attack"

    def __init__(self, config: CorrelationAttackConfig, workspace: Optional[Path] = None):
        """Initialize the attack with a configuration and optional workspace.

        Args:
            config: Scenario configuration including adversary fractions,
                correlation method, and seed count.
            workspace: Root directory for intermediate files. Defaults to
                ``./workspace`` relative to the current working directory.
        """
        super().__init__(config, workspace)
        analyzer_conf = CorrelationConfig(
            method=config.correlation_method,
            time_window=config.time_window,
            threshold=config.correlation_threshold
        )
        self._analyzer = CorrelationAnalyzer(analyzer_conf)