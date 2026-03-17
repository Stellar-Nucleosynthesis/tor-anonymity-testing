import abc
import glob
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.attacks.correlation_attack import CorrelationAttackConfig, CorrelationAttack
from src.analysis.correlation import TrafficProfile

@dataclass
class RelayCompromiseAttackConfig(CorrelationAttackConfig):
    """Configuration for a relay compromise attack.

    Attributes:
        adversary_guard_fraction: Fraction of guard relays controlled by the
            adversary, in the range [0, 1].
        adversary_exit_fraction: Fraction of exit relays controlled by the
            adversary, in the range [0, 1].
        adversary_middle_fraction: Fraction of middle relays controlled by
            the adversary, in the range [0, 1].
        max_guard_profiles: Maximum number of analyzed profiles that were
            collected from the guard relays.
        max_middle_profiles: Maximum number of analyzed profiles that were
            collected from the middle relays.
        max_exit_profiles: Maximum number of analyzed profiles that were
            collected from the exit relays.
        deanon_circ_frac: Fraction of analyzed circuits that meet the
            requirements for deanonymization to be successful.
    """
    adversary_guard_fraction: float = 0.10
    adversary_exit_fraction: float = 0.10
    adversary_middle_fraction: float = 0.0
    max_guard_profiles: Optional[int] = None,
    max_middle_profiles: Optional[int] = None,
    max_exit_profiles: Optional[int] = None,
    deanon_circ_frac: Optional[float] = None,


@dataclass
class RelayMetadata:
    """Class that describes a Tor relay in Tor hierarchy

    Attributes:
        fingerprint: Unique relay fingerprint, corresponding
            to a SHA1 hash of relay RSA key.
        flags: List of Tor relay flags.
    """
    fingerprint: str
    flags: List[str]


class RelayCompromiseAttack(CorrelationAttack):
    """Abstract base class for Tor attack simulations with relay compromises.
    """

    ATTACK_NAME: str = "relay_compromise_attack"

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
            element is a list of relay nicknames controlled by the
            adversary.
        """


    _RE_OR_CIRC_BW: re.Pattern = re.compile(
        r'CIRC_BW OR_STAT\s+'
        r'OR_CIRC_ID=(?P<or_cid>\d+)\s+'
        r'READ=(?P<read>\d+)\s+'
        r'WRITTEN=(?P<written>\d+)\s+'
        r'ResearchID=(?P<rid>[0-9a-f]+)\s+'
        r'TIME=(?P<time>\S+)'
    )

    def _load_profiles_from_oniontrace(
            self,
            shadow_hosts_dir: Path,
            observation_point: str,
            relay_filter: Optional[List[str]] = None,
    ) -> List[TrafficProfile]:
        """Parse OnionTrace logs and return one ``TrafficProfile`` per circuit per relay.

        Reads OR-side ``CIRC_BW OR_STAT`` control-port events emitted by relay
        nodes.  Each event carries an ``OR_CIRC_ID`` - the relay's local circuit
        identifier - which is used as ``TrafficProfile.circuit_id``,
        matching the key format used by ``_build_ground_truth``.

        Args:
            shadow_hosts_dir: Path to the ``shadow.data/hosts/`` directory.
            observation_point: Either ``"guard"`` or ``"exit"``. Selects which
                byte field to extract: ``READ`` bytes for guards (traffic
                arriving from the client), ``WRITTEN`` bytes for exits (traffic
                arriving from the destination).
            relay_filter: Hostnames of adversary relays to process. Only
                directories whose name appears in this list are parsed.

        Returns:
            A list of ``TrafficProfile`` objects, one per OR_CIRC_ID per relay.
            ``TrafficProfile.circuit_id`` is ``"hostname/or_cid"``.
            Returns an empty list when no matching logs or events are found.

        Raises:
            FileNotFoundError: If ``shadow_hosts_dir`` is not an existing
                directory.
        """
        if not shadow_hosts_dir.is_dir():
            raise FileNotFoundError(
                f"shadow_hosts_dir not found: {shadow_hosts_dir}"
            )

        if not relay_filter:
            self.logger.warning(
                "No relays to process under %s.", shadow_hosts_dir
            )
            return []

        bw_field = "read" if observation_point == "guard" else "written"

        profiles: List[TrafficProfile] = []
        total_events = 0

        for hostname in relay_filter:
            log_path = self._find_relay_oniontrace_log(shadow_hosts_dir / hostname)
            if log_path is None:
                self.logger.debug("No OnionTrace log found in %s - skipping.", hostname)
                continue

            circuit_data: Dict[str, List[Tuple[float, int]]] = {}

            try:
                with log_path.open(errors="replace") as fh:
                    for line in fh:
                        m = self._RE_OR_CIRC_BW.search(line)
                        if not m:
                            continue
                        bytes_val = int(m.group(bw_field))
                        if bytes_val == 0:
                            continue
                        ts = datetime.fromisoformat(m.group("time"))
                        timestamp_s = (ts.hour * 3600 + ts.minute * 60
                                       + ts.second + ts.microsecond * 1e-6)
                        circuit_data.setdefault(m.group("or_cid"), []).append(
                            (timestamp_s, bytes_val)
                        )
                        total_events += 1
            except OSError as exc:
                self.logger.warning("Could not read %s: %s", log_path, exc)
                continue

            for or_cid, events in circuit_data.items():
                profile = self._build_circ_bw_profile(
                    events=events,
                    circuit_id=f"{or_cid}",
                    observation_point=observation_point,
                    relay_hostname=hostname,
                )
                if profile is not None:
                    profiles.append(profile)

        self.logger.info(
            "Loaded %d profile(s) from %d relay(s) (%d CIRC_BW events, point=%s).",
            len(profiles),
            len(relay_filter),
            total_events,
            observation_point,
        )
        return profiles


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
            hostname=relay_hostname,
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
        score, lag = self._analyzer.correlate_profiles(guard_profile, exit_profile)
        return score, self._analyzer.is_match(score, lag)


    def _load_adversary_relays_from_hosts(
            self,
            hosts_dir: Path,
            logger: logging.Logger
    ) -> None:
        """Parse a Shadow directory  with hosts and populate the adversary relay sets.

        Calls ``_parse_consensus_dir`` to read relay flags, then passes the
        result to ``_build_adversary_relay_list`` and stores the selected
        relay nicknames in ``_adversary_guards`` and ``_adversary_exits``.

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
        all_relay_names: List[str],
        fraction: float,
        rng: Optional[np.random.Generator] = None,
    ) -> List[str]:
        """Randomly select a fraction of relay IDs as adversary-controlled.

        Args:
            all_relay_names: Full list of candidate relay nicknames.
            fraction: Proportion to select, in the range [0, 1]. Values
                ``<= 0`` always return an empty list.
            rng: NumPy random generator for reproducible selection. A new
                generator with a random seed is used when ``None``.

        Returns:
            A list of relay nicknames of length
            ``max(1, int(len(all_relay_ids) * fraction))``, or an empty list
            when ``all_relay_names`` is empty or ``fraction <= 0``.
        """
        if not all_relay_names or fraction <= 0:
            return []
        rng = rng or np.random.default_rng()
        n = max(1, int(len(all_relay_names) * fraction))
        return list(rng.choice(all_relay_names, size=min(n, len(all_relay_names)), replace=False))


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
    ) -> Dict[str, RelayMetadata]:
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
            "Parsing %d host directories from %s ...",
            len(host_dirs), hosts_dir,
        )

        merged: Dict[str, RelayMetadata] = {}
        for path in host_dirs:
            merged.update(cls._parse_shadow_host(cls, path, logger))

        guards = sum(1 for m in merged.values() if "Guard" in m.flags)
        exits = sum(1 for m in merged.values() if "Exit" in m.flags)
        middles = len(merged)
        logger.info(
            "Parsed %d unique relays - Guard: %d, Exit: %d, Middle: %d",
            len(merged), guards, exits, middles,
        )
        return merged


    @staticmethod
    def _parse_shadow_host(
            cls,
            path: Path,
            logger: logging.Logger
    ) -> Dict[str, RelayMetadata]:
        """Parse a single Shadow host directory and extract relay metadata.

        Args:
            path: Path to a host directory.

        Returns:
            A dict mapping relay hostname to metadata object.
            Returns an empty dict if the directory cannot be read.
        """
        hostname = path.name
        flags: List[str] = []
        fp = cls._find_host_fingerprint(path)
        if not fp:
            return {}

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
                        flags = line.split()[1:]
        except OSError as exc:
            logger.warning("Could not read host consensus file %s: %s", path, exc)
            return {}

        return {hostname: RelayMetadata(hostname, flags)}


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
        p = relay_dir / "oniontrace.*.stdout"
        files = glob.glob(str(p))
        if not files:
            return None
        return Path(files[0])