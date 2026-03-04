import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from src.analysis.correlation import TrafficProfile
from src.analysis.deanonymization import DeanonymizationResult
from src.analysis.guard_exit import (
    compute_circuit_compromise_rate,
    compute_guard_exit_deanon_probability,
)
from src.attacks.base_attack import AttackConfig, BaseAttack


logger = logging.getLogger("GuardExit")

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
        synthetic: When ``True``, synthetic results are generated when no
            Shadow host directory is found instead of skipping the seed.
    """

    ATTACK_NAME = "guard_exit_correlation"

    _RE_CIRC_BUILT: re.Pattern = re.compile(
        r'650 CIRC\s+(?P<cid>\d+)\s+BUILT\s+(?P<path>\S+)'
    )

    def __init__(
            self,
            config: GuardExitConfig,
            workspace: Optional[Path] = None,
            synthetic: bool = False,
    ):
        """Initialize the attack.

        Args:
            config: Guard+Exit-specific scenario configuration.
            workspace: Root directory for intermediate files. Defaults to
                ``./workspace``.
            synthetic: When ``True``, missing Shadow host directories trigger
                synthetic result generation instead of an empty return.
        """
        super().__init__(config, workspace)
        self.ge_config: GuardExitConfig = config
        self.synthetic = synthetic

        self._adversary_guards: List[str] = []
        self._adversary_exits: List[str] = []

    def _build_adversary_relay_list(
            self,
            network_data: Dict[str, Any]
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
            if "Guard" in meta.get("flags", [])
        ]
        exits = [
            name for name, meta in network_data.items()
            if "Exit" in meta.get("flags", [])
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
            1. Locates ``shadow.data/hosts/`` under ``sim_dir`` (falls back to
               synthetic data if ``self.synthetic`` is ``True`` and no
               directory is found).
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
            processed. Returns synthetic results when ``self.synthetic`` is
            ``True`` and no Shadow host directory is found.
        """
        self.logger.info("Analyzing seed=%d  dir=%s", seed, sim_dir)

        hosts_dir = self._find_shadow_data_hosts(sim_dir)
        if hosts_dir is None:
            if self.synthetic:
                return self._generate_synthetic_results(seed)
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
            return self._generate_synthetic_results(seed)

        client_hostnames = self._resolve_client_filter(sim_dir)
        ground_truth, path_to_client = self._build_ground_truth(
            hosts_dir, client_filter=client_hostnames
        )

        results = self._correlate_all_pairs(
            guard_profiles, exit_profiles, ground_truth, path_to_client, seed=seed
        )

        circuits_for_stats = list(ground_truth.values())
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

    def _resolve_client_filter(self, sim_dir: Path) -> Optional[set]:
        """Return the set of allowed client hostnames for this seed, or ``None``.

        Loads ``custom_clients_manifest.json`` from the network directory
        (produced by ``inject_custom_clients``) and resolves
        ``ge_config.client_filter`` against it.  Returns ``None`` when no
        filter is configured, which preserves existing all-circuits behavior.

        Args:
            sim_dir: Root seed directory (``runs/seed_N/``).

        Returns:
            A ``set`` of Shadow hostnames to include, or ``None`` for no
            filtering.
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
                "under %s — filter ignored.",
                self.ge_config.client_filter, sim_dir,
            )
            return None

        hostnames = manifest.resolve_filter(self.ge_config.client_filter)
        if hostnames is None:
            self.logger.warning(
                "client_filter=%r did not match any group in the manifest — "
                "filter ignored.",
                self.ge_config.client_filter,
            )
            return None

        self.logger.info(
            "client_filter=%r → %d host(s): %s",
            self.ge_config.client_filter,
            len(hostnames),
            hostnames if len(hostnames) <= 6 else hostnames[:6] + ["..."],
        )
        return set(hostnames)

    def _build_ground_truth(
            self,
            hosts_dir: Path,
            client_filter: Optional[set] = None,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str, str], str]]:
        """Build ground truth and a path-to-client mapping from OnionTrace logs.

        Iterates every OnionTrace log under ``hosts_dir`` and extracts
        ``CIRC BUILT`` events.

        Returns:
            A tuple of:
            - ``ground_truth``: mapping ``"hostname/local_cid"`` to
              ``{"guard", "middle", "exit"}`` relay hostnames.
            - ``path_to_client``: mapping ``(guard, middle, exit)`` to the
              torclient hostname that built that path.  Used to tag
              ``DeanonymizationResult.client_id`` with the originating client
              so that results can be split by group in the analysis step.
              First-seen semantics when multiple clients use the same path.
        """
        ground_truth: Dict[str, Dict[str, str]] = {}
        path_to_client: Dict[Tuple[str, str, str], str] = {}

        for host_dir in sorted(hosts_dir.iterdir()):
            if not host_dir.is_dir():
                continue
            log_path = self._find_relay_oniontrace_log(host_dir)
            if log_path is None:
                continue

            hostname = host_dir.name

            is_client = not any(
                hostname.startswith(pfx)
                for pfx in ("relayguard", "relayexit", "relay")
            )
            if client_filter is not None and is_client and hostname not in client_filter:
                continue
            for local_cid, path_str in self._iter_circ_built(log_path):
                key = f"{hostname}/{local_cid}"
                if key in ground_truth:
                    continue
                parsed = self._parse_circ_path(path_str)
                if parsed is not None:
                    guard_name, middle_name, exit_name = parsed
                    ground_truth[key] = {
                        "guard":  guard_name,
                        "middle": middle_name,
                        "exit":   exit_name,
                    }
                    if hostname.startswith("torclient"):
                        path_key = (guard_name, middle_name, exit_name)
                        if path_key not in path_to_client:
                            path_to_client[path_key] = hostname

        self.logger.debug(
            "Built ground truth for %d (hostname, cid) entries; "
            "%d path-client mappings from %s.",
            len(ground_truth), len(path_to_client), hosts_dir,
        )
        return ground_truth, path_to_client

    def _iter_circ_built(self, log_path: Path) -> Iterator[Tuple[str, str]]:
        """Yield ``(local_circuit_id, path_string)`` for every ``CIRC BUILT`` event.

        Iterates the log line-by-line without loading it into memory.

        Args:
            log_path: Path to a single OnionTrace log file.

        Yields:
            Tuples of ``(local_cid, path_str)`` where ``local_cid`` is the
            Tor-process-local circuit ID string and ``path_str`` is the raw
            ``$FP~nickname,...`` path field from the event.
        """
        try:
            with log_path.open(errors="replace") as fh:
                for line in fh:
                    m = self._RE_CIRC_BUILT.search(line)
                    if m:
                        yield m.group("cid"), m.group("path")
        except OSError as exc:
            self.logger.warning("Could not read %s: %s", log_path, exc)

    @staticmethod
    def _parse_circ_path(path_str: str,) -> Optional[Tuple[str, str, str]]:
        """Parse a ``CIRC BUILT`` path string into a relay ID tuple.

        The path field has the form::

            $FP1~nickname1,$FP2~nickname2,$FP3~nickname3

        Args:
            path_str: The raw path field from a ``CIRC BUILT`` log line.

        Returns:
            A tuple of ``(guard_name, middle_name, exit_name)`` uppercase hex
            strings, or ``None`` when the path does not contain exactly three
            hops (e.g. two-hop directory circuits or malformed lines).
        """
        hops = path_str.split(",")
        if len(hops) != 3:
            return None
        hostnames = [hop.split("~")[1].strip() for hop in hops]
        return hostnames[0], hostnames[1], hostnames[2]

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

        **Filter 1 — temporal overlap** (O(log N))
            Guard and exit observe the same circuit within a propagation delay
            bounded by ``ge_config.max_time_lag``.  Exit profiles whose
            ``first_packet_time`` lies entirely outside the window
            ``[guard_start − max_lag, guard_end + max_lag]`` are discarded via
            ``bisect`` without inspecting individual entries.

        **Filter 2 — byte-count ratio** (O(k) on time-filter survivors)
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
        return [e for e in time_candidates if lo_bytes <= e.total_bytes <= hi_bytes]

    def _correlate_all_pairs(
            self,
            guard_profiles: List[TrafficProfile],
            exit_profiles: List[TrafficProfile],
            ground_truth: Dict[str, Dict[str, str]],
            path_to_client: Dict[Tuple[str, str, str], str],
            seed: int,
    ) -> List[DeanonymizationResult]:
        """Cross-correlate guard-exit profile pairs.

        Guard and exit ``circuit_id`` values are ``"hostname/local_cid"``
        strings that are never directly comparable across relays.  Success is
        declared when both profiles resolve to the same canonical
        ``(guard_fp, middle_fp, exit_fp)`` triple via ``ground_truth``.

        Args:
            guard_profiles: Traffic profiles observed at adversary guard relays,
                as returned by ``_load_profiles_from_oniontrace``.
            exit_profiles: Traffic profiles observed at adversary exit relays,
                as returned by ``_load_profiles_from_oniontrace``.
            ground_truth: Mapping of ``"hostname/local_cid"`` to relay
                hostname dicts, as returned by ``_build_ground_truth``.
            path_to_client: mapping (built by ``_build_ground_truth``)
                used to embed the originating torclient hostname in each result's
                ``client_id`` field as
                ``"client_seed{N}_{client_hostname}__{guard_host}/{local_cid}"``.
            seed: Seed index embedded in ``DeanonymizationResult.client_id``
                for traceability.

        Returns:
            A list of ``DeanonymizationResult`` objects, one per guard profile
            for which at least one candidate survived the pre-filters.
        """
        results: List[DeanonymizationResult] = []

        sorted_exits, exit_starts = self._build_exit_index(exit_profiles)

        total_candidates = 0

        falsely_filtered = 0
        falsely_evaluated = 0
        for g_prof in guard_profiles:
            t_start = time.perf_counter()
            g_canon = ground_truth.get(g_prof.circuit_id)
            true_guard = (g_canon or {}).get("guard", "unknown")
            true_exit = (g_canon or {}).get("exit", "unknown")

            client_hostname = "unknown"
            if g_canon:
                path_key = (g_canon["guard"], g_canon["middle"], g_canon["exit"])
                client_hostname = path_to_client.get(path_key, "unknown")

            candidates = self._candidates_for_guard(g_prof, sorted_exits, exit_starts)
            total_candidates += len(candidates)

            falsely_discarded_real = False
            if true_exit not in [(ground_truth.get(c.circuit_id) or {}).get("exit", "unknown") for c in candidates]:
                falsely_discarded_real = True
                falsely_filtered += 1

            if not candidates:
                continue

            candidate_scores: List[Tuple[str, float]] = []
            for e_prof in candidates:
                score, matches = self._correlate_and_decide(g_prof, e_prof)
                if matches:
                    candidate_scores.append((e_prof.circuit_id, score))

            if not candidate_scores:
                results.append(
                    DeanonymizationResult(
                        client_id=f"client_seed{seed}_{client_hostname}__{g_prof.circuit_id}",
                        circuit_id=g_prof.circuit_id,
                        true_guard=true_guard,
                        true_exit=true_exit,
                        attempted=False,
                        successful=False,
                    )
                )
                continue

            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            best_cid, best_score = candidate_scores[0]

            best_canon = ground_truth.get(best_cid)
            successful = (
                    g_canon is not None
                    and best_canon is not None
                    and g_canon == best_canon
            )

            if not successful and not falsely_discarded_real:
                falsely_evaluated += 1

            predicted_exit_name = (
                best_canon.get("exit")
                if successful else None
            )

            confidence = 1 / (1 + math.exp(-best_score))

            results.append(
                DeanonymizationResult(
                    client_id=f"client_seed{seed}_{client_hostname}__{g_prof.circuit_id}",
                    circuit_id=g_prof.circuit_id,
                    true_guard=true_guard,
                    true_exit=true_exit,
                    predicted_guard=true_guard if g_canon else None,
                    predicted_exit=predicted_exit_name,
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
            len(guard_profiles), len(exit_profiles),
            total_candidates, total_candidates / max(len(guard_profiles), 1),
            len(guard_profiles) * len(exit_profiles),
        )
        self.logger.debug(
            f"Falsely discarded true exits for {falsely_filtered} guard profile(s) "
            f"({falsely_filtered/len(guard_profiles):.2%}) during filtering"
        )
        self.logger.debug(
            f"Falsely discarded true exits for {falsely_evaluated} guard profile(s) "
            f"({falsely_evaluated / len(guard_profiles):.2%}) during evaluation"
        )
        return results


    def _generate_synthetic_results(self, seed: int) -> List[DeanonymizationResult]:
        """Generate synthetic deanonymization results for demo and unit testing.

        Produces 200 circuits per seed using score distributions calibrated to
        the configured adversary fractions so that ROC/AUC plots remain
        meaningful even without real simulation logs.

        Compromised circuits (guard and exit both adversary-controlled) receive
        scores drawn from N(0.82, 0.08); uncompromised circuits receive scores
        from N(0.35, 0.15). Both distributions are clipped to [0, 1].

        Args:
            seed: Seed index used to seed the NumPy RNG, ensuring different
                seeds produce different but reproducible results.

        Returns:
            A list of 200 synthetic ``DeanonymizationResult`` objects.
        """
        self.logger.info(
            "Seed=%d: generating synthetic Guard+Exit results "
            "(guard_frac=%.2f, exit_frac=%.2f)",
            seed,
            self.config.adversary_guard_fraction,
            self.config.adversary_exit_fraction,
        )
        rng = np.random.default_rng(seed=seed + 1000)
        n_circuits = 200
        results: List[DeanonymizationResult] = []

        p_compromise = (
            self.config.adversary_guard_fraction
            * self.config.adversary_exit_fraction
        )

        for i in range(n_circuits):
            is_compromised = rng.random() < p_compromise
            if is_compromised:
                score = float(np.clip(rng.normal(0.82, 0.08), 0, 1))
            else:
                score = float(np.clip(rng.normal(0.35, 0.15), 0, 1))

            threshold = 0.7
            successful = is_compromised and score >= threshold

            results.append(
                DeanonymizationResult(
                    client_id=f"synthetic_seed{seed}_client{i}",
                    circuit_id=f"circ_{seed}_{i}",
                    true_guard=f"guard_{rng.integers(0, 100):03d}",
                    true_exit=f"exit_{rng.integers(0, 100):03d}",
                    predicted_guard=(
                        f"guard_{rng.integers(0, 100):03d}" if successful else None
                    ),
                    predicted_exit=(
                        f"exit_{rng.integers(0, 100):03d}" if successful else None
                    ),
                    confidence=float(np.clip(score + rng.normal(0, 0.05), 0, 1)),
                    correlation_score=score,
                    time_to_identify=float(rng.exponential(0.5)),
                    attempted=True,
                    successful=successful,
                )
            )

        return results

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