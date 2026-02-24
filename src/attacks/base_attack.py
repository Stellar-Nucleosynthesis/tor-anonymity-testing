import abc
import dataclasses
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.analysis.correlation import CorrelationAnalyzer, TrafficProfile
from src.analysis.deanonymization import DeanonymizationResult, evaluate_attack
from src.analysis.metrics import compute_roc_metrics

@dataclass
class AttackConfig:
    """Generic configuration shared by all attack types.

    Additional scenario-specific keys are stored in ``extra`` and forwarded
    to the concrete subclass via ``BaseAttack.configure``.

    Attributes:
        name: Unique scenario identifier used in filenames and reports.
        description: Human-readable description of the scenario.
        adversary_guard_fraction: Fraction of guard relays controlled by the
            adversary, in the range [0, 1].
        adversary_exit_fraction: Fraction of exit relays controlled by the
            adversary, in the range [0, 1].
        adversary_middle_fraction: Fraction of middle relays controlled by
            the adversary, in the range [0, 1].
        num_seeds: Number of independent simulation seeds to average over.
        correlation_methods: Ordered list of method names passed to
            ``CorrelationAnalyzer``.
        correlation_thresholds: Per-method decision thresholds passed to
            ``CorrelationAnalyzer``.
        extra: Free-form dict for scenario-specific parameters that do not
            belong to the common fields.
    """

    name: str
    description: str = ""
    adversary_guard_fraction: float = 0.10
    adversary_exit_fraction: float = 0.10
    adversary_middle_fraction: float = 0.0
    num_seeds: int = 3
    correlation_methods: List[str] = field(
        default_factory=lambda: ["cross_correlation", "flow_fingerprinting"]
    )
    correlation_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"cross_correlation": 0.7, "dtw_distance": 100}
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AttackConfig":
        """Construct an ``AttackConfig`` from a plain dictionary.

        Known field names are mapped to dataclass fields; any remaining keys
        are collected into ``extra``.

        Args:
            d: Dictionary of configuration values, typically loaded from a
                YAML scenario file.

        Returns:
            A fully populated ``AttackConfig`` instance.
        """
        known = {k for k in dataclasses.fields(cls)}
        base = {k: v for k, v in d.items() if k in known and k != "extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**base, extra=extra)


@dataclass
class AttackResult:
    """Aggregated result produced by ``BaseAttack.run``.

    Contains raw per-circuit deanonymization results plus computed evaluation
    metrics so the runner can render reports without knowing attack internals.

    Attributes:
        attack_name: Value of ``BaseAttack.ATTACK_NAME`` for the producing
            attack class.
        scenario_label: Human-readable label for the scenario variant.
        config: The ``AttackConfig`` used for this run.
        deanon_results: One ``DeanonymizationResult`` per circuit, aggregated
            across all seeds.
        metrics: Flat metrics dict produced by ``evaluate_attack`` (keys such
            as ``roc_auc``, ``f1_score``, ``success_rate``).
        per_seed_metrics: Per-seed metric dicts for variance analysis.
        elapsed_seconds: Wall-clock seconds for the complete ``run`` call.
        extra_info: Arbitrary metadata populated by the subclass via
            ``_extra_info``.
    """

    attack_name: str
    scenario_label: str
    config: AttackConfig
    deanon_results: List[DeanonymizationResult]
    metrics: Dict[str, Any]
    per_seed_metrics: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    extra_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def roc_auc(self) -> float:
        """float: ROC-AUC score, or 0.0 if not yet computed."""
        return float(self.metrics.get("roc_auc", 0.0))

    @property
    def f1_score(self) -> float:
        """float: F1 score, or 0.0 if not yet computed."""
        return float(self.metrics.get("f1_score", 0.0))

    @property
    def success_rate(self) -> float:
        """float: Fraction of circuits successfully deanonymized."""
        return float(self.metrics.get("success_rate", 0.0))

    def summary(self) -> str:
        """Return a compact multi-line summary string suitable for logging.

        Returns:
            A newline-joined string with attack name, circuit count, seed
            count, ROC-AUC, F1, success rate, and elapsed time.
        """
        lines = [
            f"Attack : {self.attack_name} — {self.scenario_label}",
            f"Circuits: {len(self.deanon_results)}  "
            f"Seeds: {self.config.num_seeds}",
            f"ROC-AUC : {self.roc_auc:.4f}",
            f"F1      : {self.f1_score:.4f}",
            f"Success : {self.success_rate:.2%}",
            f"Time    : {self.elapsed_seconds:.1f}s",
        ]
        return "\n".join(lines)


class BaseAttack(abc.ABC):
    """Abstract base class for all Tor deanonymization attack simulations.

    Subclasses implement a specific attack vector (e.g. Guard+Exit correlation,
    Guard+AS, watermarking) by overriding ``_run_single_seed`` and
    ``_build_adversary_relay_list``.

    Lifecycle:
        1. Instantiate with an ``AttackConfig``.
        2. Call ``configure`` to perform scenario-specific setup.
        3. Call ``run``, which iterates over seeds, aggregates
           ``DeanonymizationResult`` objects, and evaluates metrics.
        4. Inspect the returned ``AttackResult``.

    Subclass responsibilities:
        _run_single_seed: Given a parsed simulation output directory and a
            seed index, extract traffic profiles for each observation point
            and return a list of ``DeanonymizationResult`` objects.
        _build_adversary_relay_list: Return relay fingerprints the adversary
            controls, derived from network topology or tornettools staging data.
        _load_profiles_from_cell_stats: Override if the default CELL_STATS
            parser does not cover the scenario's observation points.

    Attributes:
        ATTACK_NAME: Human-readable name used in reports. Set by each
            concrete subclass.
        config: The ``AttackConfig`` this instance was created with.
        workspace: Root directory for intermediate files and caches.
        logger: Logger named after the concrete subclass.
    """

    ATTACK_NAME: str = "base"

    def __init__(self, config: AttackConfig, workspace: Optional[Path] = None):
        """Initialize the attack with a configuration and optional workspace.

        Args:
            config: Scenario configuration including adversary fractions,
                correlation methods, and seed count.
            workspace: Root directory for intermediate files. Defaults to
                ``./workspace`` relative to the current working directory.
        """
        self.config = config
        self.workspace = Path(workspace) if workspace else Path("./workspace")
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._analyzer = CorrelationAnalyzer(
            {
                "methods": config.correlation_methods,
                "thresholds": config.correlation_thresholds,
            }
        )


    def configure(self, **kwargs: Any) -> None:
        """Perform scenario-specific initialization after construction.

        Override to pre-generate adversary relay lists, inject tornettools
        configuration patches, or validate scenario prerequisites. The default
        implementation merges ``kwargs`` into ``config.extra``.

        Args:
            **kwargs: Arbitrary keyword arguments stored in ``config.extra``.
        """
        self.config.extra.update(kwargs)
        self.logger.debug(f"{self.ATTACK_NAME}.configure() called with %s", kwargs)

    def run(self, simulation_dirs: List[Path], *, label: str = "") -> AttackResult:
        """Execute the attack across all provided simulation output directories.

        Each directory must correspond to one seed run produced by
        ``SimulationOrchestrator``. A mismatch between the number of
        directories and ``config.num_seeds`` is tolerated with a warning.

        Args:
            simulation_dirs: One directory per seed. The list length should
                match ``config.num_seeds``.
            label: Human-readable label for this scenario variant, e.g.
                ``"guard_frac=0.2"``. Falls back to ``config.name`` when
                empty.

        Returns:
            An ``AttackResult`` containing all per-circuit results, aggregated
            metrics, and per-seed metric breakdowns.
        """
        if len(simulation_dirs) != self.config.num_seeds:
            self.logger.warning(
                "Expected %d seed dirs, got %d — proceeding anyway.",
                self.config.num_seeds,
                len(simulation_dirs),
            )

        t0 = time.perf_counter()
        all_results: List[DeanonymizationResult] = []
        per_seed_metrics: List[Dict[str, Any]] = []

        for seed_idx, sim_dir in enumerate(simulation_dirs):
            self.logger.info(
                "[%s] Seed %d/%d — %s",
                self.ATTACK_NAME,
                seed_idx + 1,
                len(simulation_dirs),
                sim_dir,
            )
            seed_results = self._run_single_seed(sim_dir, seed=seed_idx)
            all_results.extend(seed_results)

            if seed_results:
                try:
                    per_seed_metrics.append(evaluate_attack(seed_results))
                except (ValueError, IndexError) as exc:
                    self.logger.debug(
                        "Per-seed metric computation skipped (seed=%d): %s",
                        seed_idx,
                        exc,
                    )

        metrics = evaluate_attack(all_results) if all_results else {}

        y_true = np.array([1 if r.successful else 0 for r in all_results])
        y_scores = np.array([r.correlation_score for r in all_results])
        if len(np.unique(y_true)) > 1:
            metrics.update(compute_roc_metrics(y_true, y_scores))

        elapsed = time.perf_counter() - t0

        result = AttackResult(
            attack_name=self.ATTACK_NAME,
            scenario_label=label or self.config.name,
            config=self.config,
            deanon_results=all_results,
            metrics=metrics,
            per_seed_metrics=per_seed_metrics,
            elapsed_seconds=elapsed,
            extra_info=self._extra_info(),
        )

        self.logger.info("[%s] Finished.\n%s", self.ATTACK_NAME, result.summary())
        return result


    @abc.abstractmethod
    def _run_single_seed(self, sim_dir: Path, *, seed: int) -> List[DeanonymizationResult]:
        """Analyze one seed's simulation output and return per-circuit results.

        Args:
            sim_dir: Path to the tornettools/Shadow output directory for this
                seed, as produced by ``SimulationOrchestrator``.
            seed: Zero-based seed index used for logging and reproducibility
                tracking.

        Returns:
            A list of ``DeanonymizationResult`` objects, one per analyzed
            circuit. An empty list is acceptable when no compromised circuits
            are found.
        """

    @abc.abstractmethod
    def _build_adversary_relay_list(
        self, network_data: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Derive adversary-controlled relay IDs from network topology data.

        Args:
            network_data: Parsed content of the ``relayinfo_*.json`` staging
                file produced by ``tornettools stage``, or any topology dict
                the subclass chooses to pass.

        Returns:
            A tuple of ``(guard_ids, middle_ids, exit_ids)`` where each
            element is a list of relay fingerprint strings controlled by the
            adversary.
        """


    _TOR_CELL_BYTES: int = 512

    # Compiled regexes for raw OnionTrace log lines.
    # Format:
    #   <unix_float> [oniontrace] <hostname> <SEVERITY> 650 CELL_STATS ID=<n> ...
    #   <unix_float> [oniontrace] <hostname> <SEVERITY> 650 CIRC <n> BUILT <path> ...
    _RE_CELL_STATS: re.Pattern = re.compile(
        r'^(?P<ts>\d+\.\d+)\s+\[oniontrace]\s+\S+\s+\S+\s+'
        r'650 CELL_STATS\s+ID=(?P<cid>\d+)'
        r'.*?\bInboundAdded=(?P<in_added>\d+)'
        r'.*?\bOutboundAdded=(?P<out_added>\d+)'
    )
    _RE_CIRC_BUILT: re.Pattern = re.compile(
        r'^(?P<ts>\d+\.\d+)\s+\[oniontrace]\s+(?P<host>\S+)\s+\S+\s+'
        r'650 CIRC\s+(?P<cid>\d+)\s+BUILT\s+(?P<path>\S+)'
    )

    def _load_profiles_from_cell_stats(
        self,
        shadow_hosts_dir: Path,
        observation_point: str,
        relay_hostnames: List[str],
    ) -> List[TrafficProfile]:
        """Parse raw OnionTrace logs and return one ``TrafficProfile`` per circuit per relay.

        This method reads the **raw** per-host ``oniontrace.log`` files that
        Shadow writes to ``shadow.data/hosts/<hostname>/oniontrace.log`` and
        extracts ``CELL_STATS`` control-port events, which give per-circuit
        cell counts at each relay hop.

        **Cell-to-byte conversion**

        Tor cells are fixed at 512 bytes (Tor spec §0.1). ``CELL_STATS``
        reports cell counts, so ``bytes = InboundAdded × 512`` (guard) or
        ``bytes = OutboundAdded × 512`` (exit).

        **Circuit identity**

        Circuit IDs in ``CELL_STATS`` are relay-local integers; the same
        integer may refer to different circuits on different relays. ``CIRC
        BUILT`` events in the same log files carry the full three-hop path
        as ``$FP~nickname`` triples, which are identical on every relay in
        the circuit. The canonical circuit key is therefore the fingerprint
        tuple ``(guard_fp, middle_fp, exit_fp)`` derived from ``CIRC BUILT``.

        **Two-pass algorithm**

        Pass 1 — read ``CIRC BUILT`` events from every relay log in
        ``shadow_hosts_dir``. Build a table mapping
        ``(hostname, local_circuit_id)`` → canonical circuit key.

        Pass 2 — read ``CELL_STATS`` events only from the logs of
        ``relay_hostnames``. Accumulate cell counts into a
        ``{canonical_key: {timestamp: cells}}`` dict, then convert to
        ``TrafficProfile`` objects.

        Args:
            shadow_hosts_dir: Path to the ``shadow.data/hosts/`` directory
                produced by the Shadow simulator. Every subdirectory is
                expected to contain an ``oniontrace.log`` file.
            observation_point: Either ``"guard"`` or ``"exit"``. Controls
                which ``CELL_STATS`` field is extracted and is stored in
                each returned ``TrafficProfile``.
            relay_hostnames: Shadow hostnames of the adversary relays to
                include, e.g. ``["relay1", "relay3"]``. Only logs
                from these hosts are read in pass 2.

        Returns:
            A list of ``TrafficProfile`` objects. Each profile represents
            one circuit observed at one adversary relay, identified by its
            canonical ``(guard_fp, middle_fp, exit_fp)`` key stored in
            ``circuit_id``. Returns an empty list when no ``CELL_STATS``
            events are found or no logs exist.

        Raises:
            FileNotFoundError: If ``shadow_hosts_dir`` does not exist.
        """
        if not shadow_hosts_dir.is_dir():
            raise FileNotFoundError(f"shadow_hosts_dir not found: {shadow_hosts_dir}")

        cell_field = ("in_added" if observation_point == "guard" else "out_added")

        circ_key_table: Dict[Tuple[str, str], Tuple[str, str, str]] = {}

        all_log_paths = list(shadow_hosts_dir.glob("*/oniontrace.log"))
        if not all_log_paths:
            self.logger.warning(
                "No oniontrace.log files found under %s. "
                "Check that Shadow wrote host logs to this directory.",
                shadow_hosts_dir,
            )
            return []

        for log_path in all_log_paths:
            hostname = log_path.parent.name
            for ts, cid, path_str in self._iter_circ_built(log_path):
                key = self._canonical_circuit_key(path_str)
                if key is not None:
                    circ_key_table[(hostname, cid)] = key
        self.logger.debug(
            "Pass 1 complete: %d (host, cid) → canonical-key entries from %d log files.",
            len(circ_key_table),
            len(all_log_paths),
        )
        if not circ_key_table:
            self.logger.warning(
                "No CIRC BUILT events found in any oniontrace.log. "
                "Verify that OnionTrace is subscribing to CIRC events."
            )
            return []

        accumulator: Dict[
            Tuple[str, str, str],
            Dict[str, Dict[float, int]],
        ] = {}
        hostname_set = set(relay_hostnames)
        cell_stats_seen = 0

        for log_path in all_log_paths:
            hostname = log_path.parent.name
            if hostname not in hostname_set:
                continue
            for ts, cid, in_added, out_added in self._iter_cell_stats(log_path):
                lookup = circ_key_table.get((hostname, cid))
                if lookup is None:
                    continue

                cells = int(in_added) if cell_field == "in_added" else int(out_added)
                bytes_val = cells * self._TOR_CELL_BYTES

                relay_map = accumulator.setdefault(lookup, {})
                ts_map = relay_map.setdefault(hostname, {})
                ts_bucket = float(int(float(ts)))
                ts_map[ts_bucket] = ts_map.get(ts_bucket, 0) + bytes_val
                cell_stats_seen += 1

        self.logger.debug(
            "Pass 2 complete: %d CELL_STATS events across %d adversary relay(s).",
            cell_stats_seen,
            len(hostname_set),
        )
        if cell_stats_seen == 0:
            self.logger.warning(
                "No CELL_STATS events found in adversary relay logs. "
                "Ensure 'TestingEnableCellStatsEvent 1' is set in the Tor "
                "configuration used by the simulation (see configuration guide)."
            )
            return []

        profiles: List[TrafficProfile] = []

        for canon_key, relay_map in accumulator.items():
            guard_fp, middle_fp, exit_fp = canon_key
            circuit_label = f"{guard_fp[:8]}..{exit_fp[:8]}"

            for relay_host, ts_map in relay_map.items():
                if not ts_map:
                    continue

                pairs = sorted(ts_map.items())
                timestamps  = np.array([p[0] for p in pairs], dtype=float)
                byte_vals   = np.array([p[1] for p in pairs], dtype=float)
                cumulative  = np.cumsum(byte_vals)

                profiles.append(
                    TrafficProfile(
                        circuit_id=circuit_label,
                        timestamps=timestamps,
                        packet_sizes=byte_vals,
                        byte_counts=cumulative,
                        packet_counts=np.arange(
                            1, len(byte_vals) + 1, dtype=float
                        ),
                        first_packet_time=float(timestamps[0]),
                        last_packet_time=float(timestamps[-1]),
                        total_bytes=int(cumulative[-1]),
                        total_packets=len(byte_vals),
                        observation_point=observation_point,
                        metadata={
                            "guard_fp":    guard_fp,
                            "middle_fp":   middle_fp,
                            "exit_fp":     exit_fp,
                            "relay_host":  relay_host,
                            "cell_field":  cell_field,
                            "canon_key":   canon_key,
                        },
                    )
                )

        self.logger.debug(
            "Built %d TrafficProfile(s) for observation_point='%s'.",
            len(profiles),
            observation_point,
        )
        return profiles


    def _iter_circ_built(self, log_path: Path) -> Iterator[Tuple[str, str, str]]:
        """Yield ``(timestamp, local_circuit_id, path_string)`` for every ``CIRC BUILT`` event.

        Iterates the raw OnionTrace log line by line without loading it into
        memory, making it safe for large Shadow simulation logs.

        Args:
            log_path: Path to a single ``oniontrace.log`` file.

        Yields:
            Tuples of ``(ts, cid, path_str)`` where ``ts`` is the raw
            timestamp string, ``cid`` is the local circuit ID string, and
            ``path_str`` is the ``$FP~nick,...`` path field verbatim.
        """
        try:
            with log_path.open(errors="replace") as fh:
                for line in fh:
                    m = self._RE_CIRC_BUILT.match(line)
                    if m:
                        yield m.group("ts"), m.group("cid"), m.group("path")
        except OSError as exc:
            self.logger.warning("Could not read %s: %s", log_path, exc)

    def _iter_cell_stats(self, log_path: Path) -> Iterator[Tuple[str, str, str, str]]:
        """Yield ``(timestamp, local_circuit_id, in_added, out_added)`` for every ``CELL_STATS`` event.

        Iterates the raw OnionTrace log line by line without loading it into
        memory.

        Args:
            log_path: Path to a single ``oniontrace.log`` file.

        Yields:
            Tuples of ``(ts, cid, in_added, out_added)`` where all values are
            raw strings as they appear in the log. The caller is responsible
            for converting to numeric types.
        """
        try:
            with log_path.open(errors="replace") as fh:
                for line in fh:
                    m = self._RE_CELL_STATS.match(line)
                    if m:
                        yield (
                            m.group("ts"),
                            m.group("cid"),
                            m.group("in_added"),
                            m.group("out_added"),
                        )
        except OSError as exc:
            self.logger.warning("Could not read %s: %s", log_path, exc)

    @staticmethod
    def _canonical_circuit_key(path_str: str,) -> Optional[Tuple[str, str, str]]:
        """Parse a ``CIRC BUILT`` path string into a canonical three-hop fingerprint tuple.

        The path field in a ``CIRC BUILT`` event has the form::

            $FP1~nickname1,$FP2~nickname2,$FP3~nickname3

        The tuple ``(guard_fp, middle_fp, exit_fp)`` is identical on every
        relay that is part of the circuit and therefore serves as a globally
        unique circuit identifier across all relay logs.

        Args:
            path_str: The raw path field from a ``CIRC BUILT`` log line,
                e.g. ``"$AABB~relay1,$AACC~relay2,$AADD~relay3"``.

        Returns:
            A three-element tuple of uppercase hex fingerprint strings, or
            ``None`` when the path does not contain exactly three hops (e.g.
            for two-hop directory circuits or malformed lines).
        """
        hops = path_str.split(",")
        if len(hops) != 3:
            return None
        fps = []
        for hop in hops:
            fp_part = hop.split("~")[0].lstrip("$")
            fps.append(fp_part.upper())
        return fps[0], fps[1], fps[2]

    def _correlate_and_decide(
        self,
        guard_profile: TrafficProfile,
        exit_profile: TrafficProfile,
    ) -> Tuple[float, bool]:
        """Correlate a guard–exit profile pair and return the score and decision.

        Delegates to ``CorrelationAnalyzer`` configured from
        ``config.correlation_methods`` and ``config.correlation_thresholds``.
        The primary score is taken from ``cross_correlation`` if available,
        falling back to ``fingerprint_similarity``.

        Args:
            guard_profile: Traffic profile observed at the guard relay.
            exit_profile: Traffic profile observed at the exit relay.

        Returns:
            A tuple of ``(score, is_match)`` where ``score`` is the primary
            correlation coefficient in [0, 1] and ``is_match`` is ``True``
            when all configured thresholds are satisfied.
        """
        scores = self._analyzer.correlate_profiles(guard_profile, exit_profile)
        primary_score = scores.get(
            "cross_correlation", scores.get("fingerprint_similarity", 0.0)
        )
        is_match = self._analyzer.is_match(scores)
        return float(primary_score), is_match

    def _extra_info(self) -> Dict[str, Any]:
        """Return additional metadata to include in ``AttackResult.extra_info``.

        Override in subclasses to surface scenario-specific diagnostics such
        as adversary relay counts or theoretical deanonymization probability.

        Returns:
            A dictionary of serializable values. Returns an empty dict by
            default.
        """
        return {}


    @staticmethod
    def _select_adversary_relays(
        all_relay_ids: List[str],
        fraction: float,
        rng: Optional[np.random.Generator] = None,
    ) -> List[str]:
        """Randomly select a fraction of relay IDs as adversary-controlled.

        Args:
            all_relay_ids: Full list of candidate relay fingerprints.
            fraction: Proportion to select, in the range [0, 1]. Values
                ``<= 0`` always return an empty list.
            rng: NumPy random generator for reproducible selection. A new
                generator with a random seed is used when ``None``.

        Returns:
            A list of relay fingerprint strings of length
            ``max(1, int(len(all_relay_ids) * fraction))``, or an empty list
            when ``all_relay_ids`` is empty or ``fraction <= 0``.
        """
        if not all_relay_ids or fraction <= 0:
            return []
        rng = rng or np.random.default_rng()
        n = max(1, int(len(all_relay_ids) * fraction))
        return list(rng.choice(all_relay_ids, size=min(n, len(all_relay_ids)), replace=False))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.config.name!r}, "
            f"guard_frac={self.config.adversary_guard_fraction}, "
            f"exit_frac={self.config.adversary_exit_fraction})"
        )