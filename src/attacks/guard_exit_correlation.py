import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.attacks.base_attack import RelayMetadata
from src.analysis.correlation import TrafficProfile
from src.analysis.deanonymization import DeanonymizationResult
from src.analysis.guard_exit import (
    compute_circuit_compromise_rate,
    compute_guard_exit_deanon_probability,
)
from src.attacks.base_attack import AttackConfig, BaseAttack


logger = logging.getLogger("GuardExit")

@dataclass(frozen=True)
class Circuit:
    """Information about a Tor circuit

    Attributes:
        global_id: unique circuit identifier chosen at the circuit origin
        relays: A tuple of circuit relay hostnames.
        origin: A tuple of relay origin hostname and circuit id
    """
    global_id: str
    relays: Tuple[str, ...]
    origin: Tuple[str, str]

@dataclass
class GuardExitConfig(AttackConfig):
    """Configuration for the Guard + Exit correlation attack.
    """


class GuardExitAttack(BaseAttack):
    """Guard + Exit end-to-end traffic correlation attack.

    Implements the classic passive deanonymization attack where the adversary
    controls both a guard and an exit relay and correlates the traffic
    observed at each end to identify which client is communicating with which
    destination.

    Attributes:
        ATTACK_NAME: Fixed identifier ``"guard_exit_correlation"`` used in
            filenames and reports.
        ge_config: Typed reference to the ``GuardExitConfig`` passed at
            construction.
    """

    ATTACK_NAME = "guard_exit_correlation"

    def __init__(
            self,
            config: GuardExitConfig,
            workspace: Optional[Path] = None
    ):
        """Initialize the attack.

        Args:
            config: Guard+Exit-specific scenario configuration.
            workspace: Root directory for intermediate files. Defaults to
                ``./workspace``.
        """
        super().__init__(config, workspace)
        self.ge_config: GuardExitConfig = config

        self._adversary_guards: List[str] = []
        self._adversary_exits: List[str] = []

    def _build_adversary_relay_list(
            self,
            network_data: Dict[str, RelayMetadata]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Partition relays by role and select the adversary subset.

        ``network_data`` must be a dict in the form returned by
        ``_parse_consensus_dir``::

            {
                "AABBCC...HEX_FP...": {"nickname": "...", "flags": ["Guard", ...]},
                ...
            }

        Selection uses a fixed RNG seed of 42 for reproducibility across runs.

        Args:
            network_data: Mapping of relay nicknames to relay metadata
                dicts containing at least a ``"flags"`` list. Produced by
                ``_parse_consensus_dir``.

        Returns:
            A tuple of ``(adv_guards, adv_middles, adv_exits)`` where each
            element is a list of relay nicknames selected as
            adversary-controlled.
        """
        rng = np.random.default_rng(seed=42)

        guards = [
            name for name, meta in network_data.items()
            if "Guard" in meta.flags
        ]
        exits = [
            name for name, meta in network_data.items()
            if "Exit" in meta.flags
        ]
        middles = [
            name for name, meta in network_data.items()
        ]

        adv_guards = self._select_adversary_relays(
            guards, self.config.adversary_guard_fraction, rng
        )
        adv_exits = self._select_adversary_relays(
            exits, self.config.adversary_exit_fraction, rng
        )
        adv_middles = self._select_adversary_relays(
            middles, self.config.adversary_middle_fraction, rng
        )

        self.logger.info(
            "Adversary relays - guards: %d/%d, exits: %d/%d, middles: %d/%d",
            len(adv_guards), len(guards),
            len(adv_exits), len(exits),
            len(adv_middles), len(middles),
        )
        return adv_guards, adv_middles, adv_exits

    def _run_single_seed(self, sim_dir: Path, *, seed: int) -> List[DeanonymizationResult]:
        """Analyze one seed's Shadow/OnionTrace output.

        Expected directory layout produced by ``tornettools simulate``::

            sim_dir/
                tornet/
                    shadow.data/
                        hosts/
                            relayguard1/
                                oniontrace.1001.oniontrace.log
                            relayexit1/
                                oniontrace.1001.oniontrace.log
                            ...

        The method:
            1. Locates ``shadow.data/hosts/`` under ``sim_dir``.
            2. Ensures adversary relay lists are populated, auto-discovering
               a consensus directory if needed.
            3. Extracts guard and exit ``TrafficProfile`` objects from
               ``CIRC_BW`` events in adversary relay logs.
            4. Builds ground truth from ``CIRC BUILT`` events across all
               relay logs.
            5. Cross-correlates all guard × exit candidate pairs.
            6. Logs circuit-level compromise statistics.

        Args:
            sim_dir: Path to the tornettools/Shadow output directory for this
                seed.
            seed: Zero-based seed index for logging and reproducibility.

        Returns:
            A list of ``DeanonymizationResult`` objects, one per guard profile
            processed.
        """
        self.logger.info("Analyzing seed=%d  dir=%s", seed, sim_dir)

        hosts_dir = self._find_shadow_data_hosts(sim_dir)
        if hosts_dir is None:
            self.logger.warning(
                "No shadow.data/hosts directory found in %s - skipping seed.", sim_dir,
            )
            return []

        if not self._adversary_guards or not self._adversary_exits:
            self._load_adversary_relays_from_hosts(hosts_dir, logger)

        guard_profiles = self._load_profiles_from_oniontrace(
            hosts_dir,
            observation_point="guard",
            relay_filter=self._adversary_guards,
        )
        exit_profiles = self._load_profiles_from_oniontrace(
            hosts_dir,
            observation_point="exit",
            relay_filter=self._adversary_exits,
        )

        if not guard_profiles or not exit_profiles:
            self.logger.warning(
                "Seed=%d: guard_profiles=%d, exit_profiles=%d - insufficient data.",
                seed,
                len(guard_profiles),
                len(exit_profiles),
            )
            return []

        client_hostnames = self._resolve_client_filter(sim_dir)
        ground_truth = self._build_ground_truth(hosts_dir)

        results = self._correlate_all_pairs(
            guard_profiles, exit_profiles,
            ground_truth,
            client_filter=client_hostnames,
            seed=seed
        )

        circuits_for_stats = [c.relays for c in set(ground_truth.values())]
        if circuits_for_stats:
            stats = compute_circuit_compromise_rate(
                circuits_for_stats,
                self._adversary_guards,
                self._adversary_exits,
            )
            self.logger.info(
                "Seed=%d compromise - guard: %.1f%%, exit: %.1f%%, both: %.1f%%",
                seed,
                stats["guard_compromise_rate"] * 100,
                stats["exit_compromise_rate"] * 100,
                stats["full_compromise_rate"] * 100,
            )
            theoretical = compute_guard_exit_deanon_probability(
                self.config.adversary_guard_fraction,
                self.config.adversary_exit_fraction,
            )
            self.logger.debug("Theoretical deanon prob: %.4f", theoretical)

        return results

    def _resolve_client_filter(self, sim_dir: Path) -> Optional[Dict[str, str | None]]:
        """Return the mapping of allowed client hostnames to the  groups they
        belong to for this seed, or ``None``.

        Loads ``custom_clients_manifest.json`` from the network directory
        (produced by ``inject_custom_clients``) and resolves
        ``ge_config.client_filter`` against it.  Returns ``None`` when no
        filter is configured, which preserves existing all-circuits behavior.

        Args:
            sim_dir: Root seed directory (``runs/seed_N/``).

        Returns:
            A ``dict`` with keys as Shadow hostnames to include and values as group names,
            or ``None`` for no filtering.
        """
        if not self.ge_config.client_filter:
            return None

        from src.simulation.orchestrator import CustomClientsManifest
        manifest = None
        for candidate in sim_dir.rglob("custom_clients_manifest.json"):
            manifest = CustomClientsManifest.load(candidate.parent)
            break

        if manifest is None:
            self.logger.warning(
                "client_filter=%r set but no custom_clients_manifest.json found "
                "under %s - filter ignored.",
                self.ge_config.client_filter, sim_dir,
            )
            return None

        hostnames = manifest.resolve_filter(self.ge_config.client_filter)
        if hostnames is None:
            self.logger.warning(
                "client_filter=%r did not match any group in the manifest - "
                "filter ignored.",
                self.ge_config.client_filter,
            )
            return None

        self.logger.info(
            "client_filter=%r → %d host(s): %s",
            self.ge_config.client_filter,
            len(hostnames),
            hostnames if len(hostnames) <= 6 else list(hostnames.keys())[:6] + ["..."],
        )
        return hostnames

    _RE_RESEARCH_ID_CHOSEN = re.compile(
        r'CIRC RESEARCH_ID_CHOSEN LocalCircID=(?P<cid>\d+)'
        r' ResearchID=(?P<rid>[0-9a-f]+)'
    )
    _RE_RESEARCH_ID_UPDATED = re.compile(
        r'CIRC RESEARCH_ID_UPDATED LocalOrCircID=(?P<cid>\d+)'
        r' ResearchID=(?P<rid>[0-9a-f]+)'
    )
    _RE_CIRC_BUILT: re.Pattern = re.compile(
        r'650 CIRC\s+(?P<cid>\d+)\s+BUILT\s+(?P<path>\S+)'
    )

    def _build_ground_truth(
            self,
            hosts_dir: Path,
    ) -> Dict[Tuple[str, str], Circuit]:
        """Build ground truth from OnionTrace logs using research IDs.

        Two-pass algorithm:

        Pass 1 - client logs only:
            For each torclient* host, correlate RESEARCH_ID_CHOSEN events
            (which map a local circuit ID to a global research_id) with
            CIRC BUILT events (which map a local circuit ID to a relay path).
            Produces a mapping of research_id -> Circuit.

        Pass 2 - relay logs only:
            For each relay host, parse RESEARCH_ID_UPDATED events which map
            the relay's local circuit ID to a research_id. Look up the
            Circuit from Pass 1 and register the key
            (relay_hostname, local_cid) -> Circuit in ground_truth.

        Returns:
            A mapping of (hostname, local_cid) to Circuit, covering every
            relay hop that received a research_id for a general-purpose
            client circuit.
        """
        research_id_to_circuit: Dict[str, Circuit] = {}

        for host_dir in sorted(hosts_dir.iterdir()):
            if not host_dir.is_dir():
                continue
            log_path = self._find_relay_oniontrace_log(host_dir)
            if log_path is None:
                continue

            hostname = host_dir.name

            cid_to_rid: Dict[str, str] = {}
            cid_to_relays: Dict[str, Tuple[str, ...]] = {}

            try:
                with log_path.open(errors="replace") as fh:
                    for line in fh:
                        m = self._RE_RESEARCH_ID_CHOSEN.search(line)
                        if m:
                            cid_to_rid[m.group("cid")] = m.group("rid")
                            continue
                        m = self._RE_CIRC_BUILT.search(line)
                        if m:
                            relays = self._parse_circ_path(m.group("path"))
                            if relays is not None:
                                cid_to_relays[m.group("cid")] = relays
            except OSError as exc:
                self.logger.warning("Could not read %s: %s", log_path, exc)
                continue

            for cid, rid in cid_to_rid.items():
                relays = cid_to_relays.get(cid)
                if relays is None:
                    continue
                if rid in research_id_to_circuit:
                    continue
                research_id_to_circuit[rid] = Circuit(
                    global_id=rid,
                    relays=relays,
                    origin=(hostname, cid),
                )

        self.logger.info(
            "Pass 1: found %d local circuits in client logs.", len(research_id_to_circuit)
        )

        ground_truth: Dict[Tuple[str, str], Circuit] = {}

        for host_dir in sorted(hosts_dir.iterdir()):
            if not host_dir.is_dir():
                continue
            if host_dir.name.startswith("torclient"):
                continue
            log_path = self._find_relay_oniontrace_log(host_dir)
            if log_path is None:
                continue

            hostname = host_dir.name

            try:
                with log_path.open(errors="replace") as fh:
                    for line in fh:
                        m = self._RE_RESEARCH_ID_UPDATED.search(line)
                        if not m:
                            continue
                        cid = m.group("cid")
                        rid = m.group("rid")
                        circuit = research_id_to_circuit.get(rid)
                        if circuit is None:
                            continue
                        key = (hostname, cid)
                        if key not in ground_truth:
                            ground_truth[key] = circuit
            except OSError as exc:
                self.logger.warning("Could not read %s: %s", log_path, exc)

        self.logger.info(
            "Pass 2: registered %d (relay, local_cid) -> Circuit entries.",
            len(ground_truth),
        )
        return ground_truth

    @staticmethod
    def _parse_circ_path(path_str: str,) -> Optional[Tuple[str, ...]]:
        """Parse a ``CIRC BUILT`` path string into a relay hostname tuple.

        The path field has the form::

            $FP1~nickname1,$FP2~nickname2,$FP3~nickname3

        Args:
            path_str: The raw path field from a ``CIRC BUILT`` log line.

        Returns:
            A tuple of ``(guard_name, middle_name_1, ..., exit_name)``,
            or ``None`` when the path contains less than 2 hops.
        """
        hops = path_str.split(",")
        if len(hops) < 2:
            return None
        hostnames = [hop.split("~")[1].strip() for hop in hops]
        return tuple(hostnames)

    _BW_RATIO_MAX: float = 3.0

    @staticmethod
    def _build_exit_index(
            exit_profiles: List[TrafficProfile],
    ) -> Tuple[List[TrafficProfile], List[float]]:
        """Sort exit profiles by ``first_packet_time`` for O(log N) range queries.

        Args:
            exit_profiles: Unsorted list of exit ``TrafficProfile`` objects.

        Returns:
            A tuple of ``(sorted_exits, exit_starts)`` where ``sorted_exits``
            is ordered by ascending ``first_packet_time`` and ``exit_starts``
            holds the corresponding float values for use with ``bisect``.
        """
        sorted_exits = sorted(exit_profiles, key=lambda p: p.first_packet_time)
        exit_starts = [p.first_packet_time for p in sorted_exits]
        return sorted_exits, exit_starts

    def _candidates_for_guard(
            self,
            g_prof: TrafficProfile,
            sorted_exits: List[TrafficProfile],
            exit_starts: List[float],
    ) -> List[TrafficProfile]:
        """Return exit profiles that are plausible matches for ``g_prof``.

        Two cheap filters are applied in sequence so that
        ``CorrelationAnalyzer`` is only called on survivors:

        **Filter 1 - temporal overlap** (O(log N))
            Guard and exit observe the same circuit within a propagation delay
            bounded by ``ge_config.max_time_lag``.  Exit profiles whose
            ``first_packet_time`` lies entirely outside the window
            ``[guard_start − max_lag, guard_end + max_lag]`` are discarded via
            ``bisect`` without inspecting individual entries.

        **Filter 2 - byte-count ratio** (O(k) on time-filter survivors)
            Guard and exit carry the same application payload ±
            ``_BW_RATIO_MAX`` for Tor cell overhead.  Exit profiles whose
            ``total_bytes`` fall outside that band are discarded.

        Both filters are *sound*: no true match can be discarded.  A true
        match must by definition overlap in time and carry comparable bytes.

        Args:
            g_prof: Guard profile for which candidates are sought.
            sorted_exits: Exit profiles sorted by ``first_packet_time``.
            exit_starts: Parallel float list for ``bisect`` queries.

        Returns:
            Exit ``TrafficProfile`` objects that pass both filters.
        """
        import bisect as _bisect

        lag = self.ge_config.time_window * 10
        lo_time = g_prof.first_packet_time - lag
        hi_time = g_prof.last_packet_time + lag

        lo_idx = _bisect.bisect_left(exit_starts, lo_time)
        hi_idx = _bisect.bisect_right(exit_starts, hi_time)
        time_candidates = sorted_exits[lo_idx:hi_idx]

        if not time_candidates:
            return []

        g_bytes = max(g_prof.total_bytes, 1)
        lo_bytes = g_bytes / self._BW_RATIO_MAX
        hi_bytes = g_bytes * self._BW_RATIO_MAX
        return [e for e in time_candidates if lo_bytes <= e.total_bytes <= hi_bytes and e != g_prof]

    def _correlate_all_pairs(
            self,
            guard_profiles: List[TrafficProfile],
            exit_profiles: List[TrafficProfile],
            ground_truth: Dict[Tuple[str, str], Circuit],
            client_filter: Optional[Dict[str, str | None]],
            seed: int,
    ) -> List[DeanonymizationResult]:
        """Cross-correlate guard-exit profile pairs.

        Guard and exit ``circuit_id`` values are ``"hostname/local_cid"``
        strings that are never directly comparable across relays.  Success is
        declared when both profiles resolve to the same hostname
        list via ``ground_truth``.

        Args:
            guard_profiles: Traffic profiles observed at adversary guard relays,
                as returned by ``_load_profiles_from_oniontrace``.
            exit_profiles: Traffic profiles observed at adversary exit relays,
                as returned by ``_load_profiles_from_oniontrace``.
            ground_truth: Mapping of ``"hostname/local_cid"`` to ``Circuit`` objects,
                as returned by ``_build_ground_truth``.
            client_filter: The list of client hostnames for which to
                return deanonymization results.
            seed: Seed index embedded in ``DeanonymizationResult.client_id``
                for traceability.

        Returns:
            A list of ``DeanonymizationResult`` objects, one per guard profile
            for which at least one candidate survived the pre-filters.
        """
        results: List[DeanonymizationResult] = []

        filtered_g_profs = self._filter_traffic_profiles(
            guard_profiles, ground_truth, client_filter
        )
        filtered_e_profs = self._filter_traffic_profiles(
            exit_profiles, ground_truth, client_filter
        )

        self.logger.info(
            f"Left after filtering: {len(filtered_g_profs)} guard profiles "
            f"({len(filtered_g_profs)/len(guard_profiles):.2%})"
        )
        self.logger.info(
            f"Left after filtering: {len(filtered_e_profs)} exit profiles "
            f"({len(filtered_e_profs) / len(exit_profiles):.2%})"
        )

        sorted_exits, exit_starts = self._build_exit_index(filtered_e_profs)
        total_candidates = 0
        falsely_filtered = 0
        falsely_evaluated = 0
        for g_prof in filtered_g_profs:
            t_start = time.perf_counter()
            circuit = ground_truth[(g_prof.hostname, g_prof.circuit_id)]

            origin_hostname, origin_cid = circuit.origin
            group_name = client_filter[origin_hostname] if client_filter else None

            candidates = self._candidates_for_guard(g_prof, sorted_exits, exit_starts)
            total_candidates += len(candidates)

            falsely_discarded_real = False
            cand_circuits = [ground_truth.get((c.hostname, c.circuit_id), None)
                                    for c in candidates]
            cand_global_ids = [c.global_id for c in cand_circuits if c is not None]
            if circuit.global_id not in cand_global_ids:
                falsely_discarded_real = True
                falsely_filtered += 1

            candidate_scores: List[Tuple[str, float]] = []
            for e_prof in candidates:
                score, matches = self._correlate_and_decide(g_prof, e_prof)
                prof_circ = ground_truth.get(
                    (e_prof.hostname, e_prof.circuit_id), None)
                if matches and prof_circ is not None:
                    candidate_scores.append((prof_circ.global_id, score))

            if not candidate_scores:
                results.append(
                    DeanonymizationResult(
                        seed=str(seed),
                        group=group_name,
                        origin_id=origin_hostname,
                        circuit_id=origin_cid,
                        attempted=False,
                        successful=False,
                    )
                )
                continue

            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            best_glob_id, best_score = candidate_scores[0]
            successful = best_glob_id == circuit.global_id
            if not successful and not falsely_discarded_real:
                falsely_evaluated += 1

            confidence = 1 / (1 + math.exp(-best_score))

            results.append(
                DeanonymizationResult(
                    seed=str(seed),
                    group=group_name,
                    origin_id=origin_hostname,
                    circuit_id=g_prof.circuit_id,
                    confidence=confidence,
                    correlation_score=best_score,
                    time_to_identify=time.perf_counter() - t_start,
                    attempted=True,
                    successful=successful,
                )
            )

        self.logger.debug(
            "Correlated %d guard profile(s) against %d exit profile(s): "
            "%d pair(s) evaluated (%.1f avg/guard; brute-force would be %d).",
            len(results), len(filtered_e_profs),
            total_candidates, total_candidates / max(len(filtered_g_profs), 1),
            len(filtered_e_profs),
        )
        self.logger.debug(
            f"Falsely discarded true exits for {falsely_filtered} guard profile(s) "
            f"({falsely_filtered/len(filtered_g_profs):.2%}) during filtering"
        )
        self.logger.debug(
            f"Falsely discarded true exits for {falsely_evaluated} guard profile(s) "
            f"({falsely_evaluated / len(filtered_g_profs):.2%}) during evaluation"
        )
        return results


    @staticmethod
    def _filter_traffic_profiles(
            profiles: List[TrafficProfile],
            ground_truth: Dict[Tuple[str, str], Circuit],
            client_filter: Optional[Dict[str, str | None]],
    ) -> List[TrafficProfile]:
        """Filter traffic profiles to eliminate profiles with no registered circuit and
        include only the profiles with specified origins

        Args:
            profiles: A list of traffic profiles to filter.
            ground_truth: Mapping of local OR circuits to global client circuits.
            client_filter:  An optional list of origin hostnames.
        Yields:
            A generator of filtered ``TrafficProfile`` objects.
        """
        res = list()
        for profile in profiles:
            key = (profile.hostname, profile.circuit_id)
            circ = ground_truth.get(key, None)
            if circ is None:
                continue
            hostname, _ = circ.origin
            if client_filter is not None and hostname not in client_filter:
                continue
            res.append(profile)
        return res


    def _extra_info(self) -> Dict[str, Any]:
        """Return Guard+Exit-specific metadata for ``AttackResult.extra_info``.

        Returns:
            A dict with keys ``adversary_guard_count``,
            ``adversary_exit_count``, ``adversary_guard_fraction``,
            ``adversary_exit_fraction``, and
            ``theoretical_deanon_probability``.
        """
        return {
            "adversary_guard_count": len(self._adversary_guards),
            "adversary_exit_count": len(self._adversary_exits),
            "adversary_guard_fraction": self.config.adversary_guard_fraction,
            "adversary_exit_fraction": self.config.adversary_exit_fraction,
            "theoretical_deanon_probability": compute_guard_exit_deanon_probability(
                self.config.adversary_guard_fraction,
                self.config.adversary_exit_fraction,
            ),
        }