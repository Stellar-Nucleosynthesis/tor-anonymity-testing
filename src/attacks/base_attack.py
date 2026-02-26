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
        _load_profiles_from_oniontrace: Override if the default ``CIRC_BW``
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
        self,
        network_data: Dict[str, Any]
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

    _RE_CIRC_BW: re.Pattern = re.compile(
        r'CIRC_BW\s+'
        r'ID=(?P<cid>\d+)\s+'
        r'READ=(?P<read>\d+)\s+'
        r'WRITTEN=(?P<written>\d+)\s+'
        r'TIME=(?P<time_us>\d+)'
    )

    def _load_profiles_from_oniontrace(
        self,
        shadow_hosts_dir: Path,
        observation_point: str,
        relay_filter: Optional[List[str]] = None,
    ) -> List[TrafficProfile]:
        """Parse OnionTrace logs and return one ``TrafficProfile`` per circuit per relay.

        Reads the raw per-host OnionTrace log files written by Shadow to
        ``shadow.data/hosts/<hostname>/*oniontrace*.log`` and extracts
        ``CIRC_BW`` control-port events.  ``CIRC_BW`` is a standard Tor event
        (no testing flags required) emitted every time bytes are queued on a
        circuit, providing the per-circuit, sub-second byte counts needed for
        timing correlation.

        Args:
            shadow_hosts_dir: Path to the ``shadow.data/hosts/`` directory
                produced by the Shadow simulator. Each subdirectory must
                correspond to one simulated host and contain an OnionTrace log.
            observation_point: Either ``"guard"`` or ``"exit"``. Selects which
                ``CIRC_BW`` byte field to extract and is stored verbatim in
                each returned ``TrafficProfile``.

        Returns:
            A list of ``TrafficProfile`` objects, one per circuit per relay.
            Each profile's ``circuit_id`` is ``"<hostname>/<local_cid>"``.
            Returns an empty list when ``shadow_hosts_dir`` contains no
            matching logs or no ``CIRC_BW`` events are found.

        Raises:
            FileNotFoundError: If ``shadow_hosts_dir`` is not an existing
                directory.
        """
        if not shadow_hosts_dir.is_dir():
            raise FileNotFoundError(
                f"shadow_hosts_dir not found: {shadow_hosts_dir}"
            )

        bw_field = "read" if observation_point == "guard" else "written"

        if not relay_filter:
            self.logger.warning(
                "No relays to process under %s.", shadow_hosts_dir
            )
            return []

        profiles: List[TrafficProfile] = []
        total_events = 0

        fingerprint_to_name = {}
        for d in shadow_hosts_dir.iterdir():
            fp = self._find_host_fingerprint(d)
            if fp:
                fingerprint_to_name[fp] = d.name

        for fp in relay_filter:
            hostname = fingerprint_to_name.get(fp)
            if hostname is None:
                self.logger.warning("Relay directory not found for relay: %s — skipping.", fp)
                continue
            log_path = self._find_relay_oniontrace_log(shadow_hosts_dir / hostname)
            if log_path is None:
                self.logger.warning("No OnionTrace log found in %s — skipping.", hostname)
                continue

            circuit_data: Dict[str, List[Tuple[float, int]]] = {}

            for cid, read_b, written_b, time_us in self._iter_circ_bw(log_path):
                bytes_val = read_b if bw_field == "read" else written_b
                if bytes_val == 0:
                    continue

                ts_s = time_us / 1_000_000.0
                circuit_data.setdefault(cid, []).append((ts_s, bytes_val))
                total_events += 1

            for local_cid, events in circuit_data.items():
                profile = self._build_circ_bw_profile(
                    events=events,
                    circuit_id=f"{hostname}/{local_cid}",
                    observation_point=observation_point,
                    relay_hostname=hostname,
                )
                if profile is not None:
                    profiles.append(profile)

        self.logger.debug(
            "Loaded %d profile(s) from %d relay(s) (%d CIRC_BW events, point=%s).",
            len(profiles),
            len(relay_filter),
            total_events,
            observation_point,
        )
        return profiles

    def _iter_circ_bw(self, log_path: Path) -> Iterator[Tuple[str, int, int, int]]:
        """Yield parsed fields from every ``CIRC_BW`` line in an OnionTrace log.

        Iterates the file line-by-line without loading it into memory so it is
        safe for large Shadow simulation logs.

        Args:
            log_path: Path to a single OnionTrace log file.

        Yields:
            Tuples of ``(circuit_id, read_bytes, written_bytes, time_us)``
            where ``circuit_id`` is the raw local circuit ID string,
            ``read_bytes`` and ``written_bytes`` are integers, and
            ``time_us`` is the event timestamp in microseconds since the Unix
            epoch.
        """
        try:
            with log_path.open(errors="replace") as fh:
                for line in fh:
                    m = self._RE_CIRC_BW.search(line)
                    if m:
                        yield (
                            m.group("cid"),
                            int(m.group("read")),
                            int(m.group("written")),
                            int(m.group("time_us")),
                        )
        except OSError as exc:
            self.logger.warning("Could not read %s: %s", log_path, exc)

    @staticmethod
    def _build_circ_bw_profile(
        events: List[Tuple[float, int]],
        circuit_id: str,
        observation_point: str,
        relay_hostname: str,
    ) -> Optional[TrafficProfile]:
        """Build a ``TrafficProfile`` from a list of ``(timestamp_s, bytes)`` events.

        Events are sorted by timestamp before building the arrays so that
        out-of-order log lines (which can occur near Shadow process-wake
        boundaries) do not corrupt the timeseries.

        Args:
            events: List of ``(timestamp_seconds, byte_count)`` pairs for one
                circuit on one relay.  Must contain at least one entry.
            circuit_id: Scoped circuit identifier (``"<hostname>/<local_cid>"``),
                stored in ``TrafficProfile.circuit_id``.
            observation_point: ``"guard"`` or ``"exit"``, stored in
                ``TrafficProfile.observation_point``.
            relay_hostname: Shadow hostname of the relay, stored in
                ``TrafficProfile.metadata``.

        Returns:
            A populated ``TrafficProfile``, or ``None`` if ``events`` is empty.
        """
        if not events:
            return None

        events_sorted = sorted(events, key=lambda e: e[0])
        timestamps = np.array([e[0] for e in events_sorted], dtype=float)
        byte_counts = np.array([e[1] for e in events_sorted], dtype=float)
        cumulative = np.cumsum(byte_counts)

        return TrafficProfile(
            circuit_id=circuit_id,
            timestamps=timestamps,
            packet_sizes=byte_counts,
            byte_counts=cumulative,
            packet_counts=np.arange(1, len(byte_counts) + 1, dtype=float),
            first_packet_time=float(timestamps[0]),
            last_packet_time=float(timestamps[-1]),
            total_bytes=int(cumulative[-1]),
            total_packets=len(byte_counts),
            observation_point=observation_point,
            metadata={
                "relay_hostname": relay_hostname,
                "local_circuit_id": circuit_id.split("/", 1)[-1],
            },
        )


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

    def _load_adversary_relays_from_hosts(
            self,
            hosts_dir: Path,
            logger: logging.Logger
    ) -> None:
        """Parse a Shadow directory  with hosts and populate the adversary relay sets.

        Calls ``_parse_consensus_dir`` to read relay flags, then passes the
        result to ``_build_adversary_relay_list`` and stores the selected
        fingerprints in ``_adversary_guards`` and ``_adversary_exits``.

        Args:
            hosts_dir: Path to the root of a shadow.data directory.
            logger: The output for event logging.
        """
        try:
            network_data = self._parse_hosts_dir(hosts_dir, logger)
            if not network_data:
                logger.warning(
                    "No relay data parsed from %s; adversary lists remain empty.",
                    hosts_dir,
                )
                return
            guards, _, exits = self._build_adversary_relay_list(network_data)
            self._adversary_guards = guards
            self._adversary_exits = exits
        except (ValueError, OSError) as exc:
            self.logger.error(
                "Failed to load adversary relays from hosts dir %s: %s",
                hosts_dir, exc,
            )

    @staticmethod
    def _select_adversary_relays(
        all_relay_fps: List[str],
        fraction: float,
        rng: Optional[np.random.Generator] = None,
    ) -> List[str]:
        """Randomly select a fraction of relay IDs as adversary-controlled.

        Args:
            all_relay_fps: Full list of candidate relay fingerprints.
            fraction: Proportion to select, in the range [0, 1]. Values
                ``<= 0`` always return an empty list.
            rng: NumPy random generator for reproducible selection. A new
                generator with a random seed is used when ``None``.

        Returns:
            A list of relay fingerprint strings of length
            ``max(1, int(len(all_relay_ids) * fraction))``, or an empty list
            when ``all_relay_ids`` is empty or ``fraction <= 0``.
        """
        if not all_relay_fps or fraction <= 0:
            return []
        rng = rng or np.random.default_rng()
        n = max(1, int(len(all_relay_fps) * fraction))
        return list(rng.choice(all_relay_fps, size=min(n, len(all_relay_fps)), replace=False))

    @staticmethod
    def _find_shadow_data_hosts(sim_dir: Path) -> Optional[Path]:
        """Search common locations for the ``shadow.data/hosts`` directory.

        Args:
            sim_dir: Root of the simulation output directory for this seed.

        Returns:
            Path to the first matching ``shadow.data/hosts`` directory, or
            ``None`` when none is found.
        """
        for p in sim_dir.rglob("shadow.data/hosts"):
            if p.is_dir():
                return p
        return None

    @classmethod
    def _parse_hosts_dir(
            cls,
            hosts_dir: Path,
            logger: logging.Logger,
    ) -> Dict[str, Dict[str, Any]]:
        """Parse all active hosts under shadow.data directory.

        Args:
            hosts_dir: Path: Path to shadow.data simulation directory, e.g.
                ``Path("seed0/shadow.data")``.

        Returns:
            A merged relay metadata dict in the same format as
            ``_parse_host_dir``. Returns an empty dict when no host
            directories are found.

        Raises:
            ValueError: If ``hosts_dir`` is not an existing directory.
        """
        if not hosts_dir.is_dir():
            raise ValueError(
                f"hosts_dir is not a directory: {hosts_dir}"
            )

        host_dirs = list(hosts_dir.iterdir())
        if len(host_dirs) == 0:
            logger.warning("No host directories found under %s", hosts_dir)
            return {}

        logger.info(
            "Parsing %d host directories from %s …",
            len(host_dirs), hosts_dir,
        )

        merged: Dict[str, Dict[str, Any]] = {}
        for path in host_dirs:
            merged.update(cls._parse_shadow_host(cls, path, logger))

        guards = sum(1 for m in merged.values() if "Guard" in m["flags"])
        exits = sum(1 for m in merged.values() if "Exit" in m["flags"])
        middles = len(merged)
        logger.info(
            "Parsed %d unique relays — Guard: %d, Exit: %d, Middle: %d",
            len(merged), guards, exits, middles,
        )
        return merged

    @staticmethod
    def _parse_shadow_host(
            cls,
            path: Path,
            logger: logging.Logger
    ) -> Dict[str, Dict[str, Any]]:
        """Parse a single Shadow host directory and extract relay metadata.

        Args:
            path: Path to a host directory.

        Returns:
            A dict mapping hex fingerprint strings to metadata dicts. Each
            metadata dict contains:

            * ``"nickname"`` (str): the relay nickname.
            * ``"flags"`` (List[str]): consensus flags, e.g.
              ``["Guard", "Exit", "Fast", "Running", "Stable", "Valid"]``.

            Returns an empty dict if the directory cannot be read.
        """
        current_fp = cls._find_host_fingerprint(path)
        if not current_fp:
            return {}
        metadata: Dict[str, Any] = {
            "nickname": path.name,
            "flags": [],
        }

        prim_path = path / "cached-consensus"
        fallback_path = path / "cached-microdesc-consensus"
        cons_path = prim_path if prim_path.exists() else fallback_path
        if not cons_path.exists():
            return {}

        try:
            with cons_path.open(errors="replace") as fh:
                for raw_line in fh:
                    line = raw_line.rstrip("\n")
                    if line.startswith("known-flags "):
                        parts = line.split()
                        metadata["flags"] = parts[1:]
        except OSError as exc:
            logger.warning("Could not read host consensus file %s: %s", path, exc)
            return {}

        return {current_fp: metadata}

    @staticmethod
    def _find_host_fingerprint(host_dir: Path) -> Optional[str]:
        """Find the host's fingerprint from its shadow directory

        Args:
            host_dir: Directory with the shadow host.

        Returns:
            Fingerprint of the host found in ``host_dir``.
        """
        if not host_dir.is_dir():
            return None
        fp = host_dir / "fingerprint"
        if not fp.is_file():
            return None
        try:
            with fp.open("r", encoding="utf-8") as f:
                _, fp = f.readline().split(None, 1)
                return fp.strip()
        except ValueError:
            return None

    @staticmethod
    def _find_relay_oniontrace_log(relay_dir: Path) -> Optional[Path]:
        """Return the OnionTrace log file inside a Shadow relay directory.

        Args:
            relay_dir: Path to one host's directory under
                ``shadow.data/hosts/``.

        Returns:
            Path to the first matching log file, or ``None`` when no
            OnionTrace log exists in ``relay_dir``.
        """
        p = relay_dir / "oniontrace.1003.stdout"
        return p if p.is_file() else None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.config.name!r}, "
            f"guard_frac={self.config.adversary_guard_fraction}, "
            f"exit_frac={self.config.adversary_exit_fraction})"
        )