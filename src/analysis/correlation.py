import dataclasses

import numpy as np
from scipy import signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging


@dataclass
class TrafficProfile:
    """Represents traffic observations from a single observation point"""
    circuit_id: str
    timestamps: np.ndarray
    packet_sizes: np.ndarray
    byte_counts: np.ndarray
    packet_counts: np.ndarray
    first_packet_time: float
    last_packet_time: float
    total_bytes: int
    total_packets: int
    observation_point: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def dtw_distance(
        profile1: TrafficProfile,
        profile2: TrafficProfile
) -> float:
    """
    Compute Dynamic Time Warping distance between profiles.

    Args:
        profile1: First traffic profile
        profile2: Second traffic profile

    Returns:
        DTW distance (lower is more similar)
    """
    seq1 = np.column_stack([
        profile1.timestamps - profile1.first_packet_time,
        profile1.packet_sizes
    ])

    seq2 = np.column_stack([
        profile2.timestamps - profile2.first_packet_time,
        profile2.packet_sizes
    ])

    distance, _ = fastdtw(seq1, seq2, dist=euclidean)
    return float(distance)


def flow_fingerprint(profile: TrafficProfile) -> np.ndarray:
    """
    Generate a fingerprint for a traffic flow.

    Args:
        profile: Traffic profile

    Returns:
        Feature vector representing the flow
    """
    features = list()

    features.append(profile.total_bytes)
    features.append(profile.total_packets)
    features.append(profile.last_packet_time - profile.first_packet_time)

    if len(profile.packet_sizes) > 0:
        features.append(np.mean(profile.packet_sizes))
        features.append(np.std(profile.packet_sizes))
        features.append(np.median(profile.packet_sizes))
        features.append(np.max(profile.packet_sizes))
        features.append(np.min(profile.packet_sizes))
    else:
        features.extend([0, 0, 0, 0, 0])

    if len(profile.timestamps) > 1:
        inter_packet_times = np.diff(profile.timestamps)
        features.append(np.mean(inter_packet_times))
        features.append(np.std(inter_packet_times))
        features.append(np.median(inter_packet_times))
    else:
        features.extend([0, 0, 0])

    return np.array(features)


def bin_traffic(
        profile: TrafficProfile,
        start_time: float,
        bin_size: float,
        num_bins: int
) -> np.ndarray:
    """
    Bin traffic into time windows.

    Args:
        profile: Traffic profile to bin
        start_time: Start time for binning
        bin_size: Size of each bin in seconds
        num_bins: Number of bins

    Returns:
        Array of byte counts per bin
    """
    bins = np.zeros(num_bins)

    for timestamp, size in zip(profile.timestamps, profile.packet_sizes):
        bin_idx = int((timestamp - start_time) / bin_size)
        if 0 <= bin_idx < num_bins:
            bins[bin_idx] += size

    return bins


def cross_correlation(
        profile1: TrafficProfile,
        profile2: TrafficProfile,
        normalize: bool = True
) -> Tuple[float, float]:
    """
    Compute cross-correlation between two traffic profiles.

    Args:
        profile1: First traffic profile (e.g., from guard)
        profile2: Second traffic profile (e.g., from exit)
        normalize: Whether to normalize correlation

    Returns:
        Tuple of (correlation_coefficient, time_lag)
    """
    bin_size = 1

    min_time = min(profile1.first_packet_time, profile2.first_packet_time)
    max_time = max(profile1.last_packet_time, profile2.last_packet_time)
    duration = max_time - min_time
    num_bins = int(duration / bin_size) + 1

    bins1 = bin_traffic(profile1, min_time, bin_size, num_bins)
    bins2 = bin_traffic(profile2, min_time, bin_size, num_bins)

    correlation = signal.correlate(bins1, bins2, mode='full')

    if normalize:
        norm = np.sqrt(np.sum(bins1 ** 2) * np.sum(bins2 ** 2))
        if norm > 0:
            correlation = correlation / norm

    peak_idx = np.argmax(np.abs(correlation))
    peak_corr = correlation[peak_idx]
    lag = (peak_idx - len(bins2) + 1) * bin_size

    return float(peak_corr), float(lag)


def time_shift_search(
        profile1: TrafficProfile,
        profile2: TrafficProfile,
        max_shift: float = 5.0
) -> Tuple[float, float]:
    """
    Search for optimal time shift between profiles.

    Args:
        profile1: First traffic profile
        profile2: Second traffic profile
        max_shift: Maximum time shift to search (seconds)

    Returns:
        Tuple of (best_correlation, optimal_shift)
    """
    shifts = np.linspace(-max_shift, max_shift, 100)
    best_corr = -1
    best_shift = 0

    for shift in shifts:
        shifted_profile = TrafficProfile(
            circuit_id=profile2.circuit_id,
            timestamps=profile2.timestamps + shift,
            packet_sizes=profile2.packet_sizes,
            byte_counts=profile2.byte_counts,
            packet_counts=profile2.packet_counts,
            first_packet_time=profile2.first_packet_time + shift,
            last_packet_time=profile2.last_packet_time + shift,
            total_bytes=profile2.total_bytes,
            total_packets=profile2.total_packets,
            observation_point=profile2.observation_point,
            metadata=profile2.metadata
        )

        corr, _ = cross_correlation(profile1, shifted_profile)

        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return float(best_corr), float(best_shift)


@dataclass
class CorrelationConfig:
    methods: List[str] = field(default_factory=lambda: ["cross_correlation"])
    primary: str | None = None
    time_window: float = 300.0
    thresholds: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CorrelationConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        base = {k: v for k, v in d.items() if k in known and k != "extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**base, extra=extra)


class CorrelationAnalyzer:
    """
    Analyzes traffic patterns to identify correlations between
    different observation points.
    """

    def __init__(self, config: CorrelationConfig):
        """
        Initialize correlation analyzer.

        Args:
            config: Correlation configuration
        """
        self.config = config
        self.methods = list(config.methods)
        self.primary = config.primary
        if self.primary and self.primary not in self.methods:
            self.methods.append(self.primary)
        self.time_window = config.time_window
        self.thresholds = config.thresholds
        self.logger = logging.getLogger("CorrelationAnalyzer")

    def correlate_profiles(
            self,
            profile1: TrafficProfile,
            profile2: TrafficProfile
    ) -> Dict[str, float]:
        """
        Correlate two profiles using configured methods.

        Args:
            profile1: First traffic profile
            profile2: Second traffic profile

        Returns:
            Dictionary of correlation scores by method
        """
        results = {}

        if 'cross_correlation' in self.methods:
            corr, lag = cross_correlation(profile1, profile2)
            results['cross_correlation'] = corr
            results['cross_correlation_lag'] = lag

        if 'time_shift_search' in self.methods:
            corr, shift = time_shift_search(profile1, profile2)
            results['time_shift_correlation'] = corr
            results['optimal_shift'] = shift

        if 'dtw' in self.methods:
            distance = dtw_distance(profile1, profile2)
            results['dtw_distance'] = distance

        if 'flow_fingerprinting' in self.methods:
            fp1 = flow_fingerprint(profile1)
            fp2 = flow_fingerprint(profile2)
            similarity = np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2))
            results['fingerprint_similarity'] = similarity

        return results

    def is_match(
            self,
            correlation_scores: Dict[str, float]
    ) -> Tuple[float, bool]:
        """
        Determine if correlation scores indicate a match.

        Args:
            correlation_scores: Dictionary of correlation scores

        Returns:
            A tuple of ``(score, is_match)`` where ``score`` is
            correlation coefficient, and ``is_match`` is ``True``
            when all configured thresholds are satisfied.
        """
        prim_score = correlation_scores.get(self.primary, 0)
        for method, score in correlation_scores.items():
            if method == 'cross_correlation':
                if score < self.thresholds.get('cross_correlation', 0.9):
                    return prim_score, False
            elif method == 'dtw_distance':
                if score > self.thresholds.get('dtw_distance', 100):
                    return prim_score, False
            elif method == 'optimal_shift':
                if abs(score) > self.thresholds.get('time_shift_max', 5.0):
                    return prim_score, False
            elif method == 'flow_fingerprinting':
                if abs(score) > self.thresholds.get('flow_fingerprinting', 5.0):
                    return prim_score, False

        return prim_score, True

    def batch_correlate(self,
                        entry_profiles: List[TrafficProfile],
                        exit_profiles: List[TrafficProfile]) -> np.ndarray:
        """
        Compute correlation matrix between all entry and exit profiles.

        Args:
            entry_profiles: List of profiles from guard observations
            exit_profiles: List of profiles from exit observations

        Returns:
            Correlation matrix of shape (len(entry_profile), len(exit_profile))
        """
        n_entries = len(entry_profiles)
        n_exits = len(exit_profiles)

        correlation_matrix = np.zeros((n_entries, n_exits))

        self.logger.info(f"Computing {n_entries}x{n_exits} correlation matrix...")

        for i, guard_prof in enumerate(entry_profiles):
            for j, exit_prof in enumerate(exit_profiles):
                scores = self.correlate_profiles(guard_prof, exit_prof)
                correlation_matrix[i, j] = scores.get('cross_correlation', 0)

        return correlation_matrix
