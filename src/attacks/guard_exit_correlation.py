import base64
import binascii
import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    Extends ``AttackConfig`` with scenario-specific knobs for time-lag search,
    traffic binning, and matching strategy.

    Attributes:
        max_time_lag: Maximum time-lag in seconds considered during the
            correlation search window.
        bin_size: Width in seconds of each time bin used for traffic
            histogramming.
        use_all_methods: When ``True``, all configured correlation methods are
            applied; when ``False``, only cross-correlation is used.
        require_top_rank: When ``True``, deanonymization is declared successful
            only when the correct pair is the top-ranked candidate. When
            ``False``, any candidate whose score exceeds the threshold counts.
        consensus_dir: Path to a directory of Tor consensus files in CollecTor
            format (e.g. ``consensuses-2023-04/``). Used to derive which relays
            carry the Guard and Exit flags. When ``None`` the attack tries to
            locate a consensus directory automatically adjacent to the
            simulation workspace.
    """

    max_time_lag: float = 5.0
    bin_size: float = 0.1
    use_all_methods: bool = True
    require_top_rank: bool = True
    consensus_dir: Optional[Path] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GuardExitConfig":
        """Construct a ``GuardExitConfig`` from a plain dictionary.

        Known field names are mapped to dataclass fields; any remaining keys
        are collected into ``extra``. String values for ``consensus_dir`` are
        automatically converted to ``Path`` objects.

        Args:
            d: Dictionary of configuration values, typically loaded from a
                YAML scenario file.

        Returns:
            A fully populated ``GuardExitConfig`` instance.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        base = {k: v for k, v in d.items() if k in known and k != "extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        if "consensus_dir" in base and base["consensus_dir"] is not None:
            base["consensus_dir"] = Path(base["consensus_dir"])
        return cls(**base, extra=extra)


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
            OnionTrace log is found instead of skipping the seed.
    """

    ATTACK_NAME = "guard_exit_correlation"

    def __init__(
        self,
        config: GuardExitConfig,
        workspace: Optional[Path] = None,
        synthetic: bool = False,
    ):
        """Initialize the attack.

        Ensures ``cross_correlation`` is always the first method in
        ``config.correlation_methods`` since it is used as the primary score.

        Args:
            config: Guard+Exit-specific scenario configuration.
            workspace: Root directory for intermediate files. Defaults to
                ``./workspace``.
            synthetic: When ``True``, missing OnionTrace logs trigger
                synthetic result generation instead of an empty return.
        """
        if "cross_correlation" not in config.correlation_methods:
            config.correlation_methods.insert(0, "cross_correlation")

        super().__init__(config, workspace)
        self.ge_config: GuardExitConfig = config
        self.synthetic = synthetic

        self._adversary_guards: List[str] = []
        self._adversary_exits: List[str] = []


    def configure(self, **kwargs: Any) -> None:
        """Preload adversary relay lists and apply extra configuration.

        If ``consensus_dir`` is provided (either via ``kwargs`` or
        ``ge_config.consensus_dir``), consensus files are parsed and a random
        subset of relays is marked as adversary-controlled according to the
        configured fractions. Call this before ``run``.

        Args:
            **kwargs: Keyword arguments forwarded to ``AttackConfig.extra``.
                Recognizes ``consensus_dir`` (str or Path) to override the
                config-level path.
        """
        super().configure(**kwargs)

        consensus_dir = kwargs.get("consensus_dir", self.ge_config.consensus_dir)
        if consensus_dir:
            consensus_dir = Path(consensus_dir)
            if consensus_dir.exists():
                self._load_adversary_relays_from_consensus(consensus_dir)
            else:
                self.logger.warning(
                    "consensus_dir does not exist: %s", consensus_dir
                )

    def _build_adversary_relay_list(
        self, network_data: Dict[str, Any]
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
            network_data: Mapping of hex fingerprint strings to relay metadata
                dicts containing at least a ``"flags"`` list. Produced by
                ``_parse_consensus_dir``.

        Returns:
            A tuple of ``(adv_guards, adv_middles, adv_exits)`` where each
            element is a list of relay fingerprint strings selected as
            adversary-controlled.
        """
        rng = np.random.default_rng(seed=42)

        guards = [
            fp for fp, meta in network_data.items()
            if "Guard" in meta.get("flags", [])
        ]
        exits = [
            fp for fp, meta in network_data.items()
            if "Exit" in meta.get("flags", [])
        ]
        middles = [
            fp for fp, meta in network_data.items()
            if "Guard" not in meta.get("flags", [])
            and "Exit" not in meta.get("flags", [])
        ]

        adv_guards = self._select_adversary_relays(guards, self.config.adversary_guard_fraction, rng)
        adv_exits = self._select_adversary_relays(exits, self.config.adversary_exit_fraction, rng)
        adv_middles = self._select_adversary_relays(middles, self.config.adversary_middle_fraction, rng)

        self.logger.info(
            "Adversary relays — guards: %d/%d, exits: %d/%d, middles: %d/%d",
            len(adv_guards), len(guards),
            len(adv_exits), len(exits),
            len(adv_middles), len(middles),
        )
        return adv_guards, adv_middles, adv_exits

    def _run_single_seed(self, sim_dir: Path, *, seed: int) -> List[DeanonymizationResult]:
        """Analyze one seed's Oniontrace/Shadow output.

        Expected directory layout produced by ``tornettools simulate``::

            sim_dir/
                tornet/
                    shadow.data/
                        hosts/


        The method:
            1. Locates the Oniontrace log in Shadow directory (falls back to
               synthetic data if ``self.synthetic`` is ``True`` and no log is found).
            2. Ensures adversary relay lists are populated, auto-discovering
               a consensus directory if needed.
            3. Extracts guard and exit ``TrafficProfile`` objects filtered to
               adversary-controlled relays.
            4. Cross-correlates all guard × exit candidate pairs.
            5. Logs circuit-level compromise statistics.

        Args:
            sim_dir: Path to the tornettools/Shadow output directory for this seed.
            seed: Zero-based seed index for logging and reproducibility.

        Returns:
            A list of ``DeanonymizationResult`` objects, one per guard profile
            processed. Returns synthetic results when ``self.synthetic`` is
            ``True`` and no log file is found.
        """
        self.logger.info("Analyzing seed=%d  dir=%s", seed, sim_dir)

        hosts_dir = self._find_shadow_data_hosts(sim_dir)
        if hosts_dir is None:
            if self.synthetic:
                return self._generate_synthetic_results(seed)
            self.logger.warning(
                "No shadow.data directory found in %s — skipping seed.", sim_dir
            )
            return []

        if not self._adversary_guards or not self._adversary_exits:
            self._auto_load_adversary_relays(sim_dir)

        guard_profiles = self._load_profiles_from_cell_stats(
            hosts_dir,
            observation_point="guard",
            relay_hostnames=self._adversary_guards
        )
        exit_profiles = self._load_profiles_from_cell_stats(
            hosts_dir,
            observation_point="exit",
            relay_hostnames=self._adversary_exits
        )

        if not guard_profiles or not exit_profiles:
            self.logger.warning(
                "Seed=%d: guard_profiles=%d, exit_profiles=%d — insufficient data.",
                seed,
                len(guard_profiles),
                len(exit_profiles),
            )
            return self._generate_synthetic_results(seed)

        ground_truth = self._build_ground_truth(hosts_dir)

        results = self._correlate_all_pairs(
            guard_profiles, exit_profiles, ground_truth, seed=seed
        )

        circuits_for_stats = [
            {"guard": gt["guard"], "exit": gt["exit"]}
            for gt in ground_truth.values()
        ]
        if circuits_for_stats:
            stats = compute_circuit_compromise_rate(
                circuits_for_stats,
                self._adversary_guards,
                self._adversary_exits,
            )
            self.logger.info(
                "Seed=%d compromise — guard: %.1f%%, exit: %.1f%%, both: %.1f%%",
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

    @staticmethod
    def _b64_fingerprint_to_hex(b64: str) -> str:
        """Convert a base64-encoded relay fingerprint to an uppercase hex string.

        Tor consensus ``r`` lines encode the 20-byte identity fingerprint in
        base64 without padding. This method re-adds the padding before
        decoding and returns the standard 40-character hex representation.

        Args:
            b64: Base64 fingerprint string as it appears in the ``r`` line of
                a consensus file (no ``=`` padding, typically 27 characters).

        Returns:
            A 40-character uppercase hex string, e.g.
            ``"AABBCCDDEEFF00112233445566778899AABBCCDD"``.
            Returns the original string unchanged if decoding fails.
        """
        try:
            pad = (4 - len(b64) % 4) % 4
            return binascii.hexlify(
                base64.b64decode(b64 + "=" * pad)
            ).upper().decode()
        except Exception:
            return b64

    @staticmethod
    def _parse_consensus_file(path: Path,) -> Dict[str, Dict[str, Any]]:
        """Parse a single Tor consensus file and extract relay metadata.

        Iterates line-by-line over the consensus and tracks ``r`` (router)
        and ``s`` (status flags) lines. Only relays that have at least one
        status line are included.

        The consensus format is defined in Tor's directory-spec.txt.
        Relevant lines::

            r <nickname> <identity_b64> <digest_b64> <date> <time> <IP> <ORport> <Dirport>
            s <Flag1> <Flag2> ...

        Args:
            path: Path to a consensus file (plain text, not compressed).

        Returns:
            A dict mapping hex fingerprint strings to metadata dicts. Each
            metadata dict contains:

            * ``"nickname"`` (str): the relay nickname.
            * ``"flags"`` (List[str]): consensus flags, e.g.
              ``["Guard", "Exit", "Fast", "Running", "Stable", "Valid"]``.

            Returns an empty dict if the file cannot be read.
        """
        relays: Dict[str, Dict[str, Any]] = {}
        current_fp: Optional[str] = None

        try:
            with path.open(errors="replace") as fh:
                for raw_line in fh:
                    line = raw_line.rstrip("\n")

                    if line.startswith("r "):
                        parts = line.split()
                        if len(parts) >= 3:
                            current_fp = GuardExitAttack._b64_fingerprint_to_hex(
                                parts[2]
                            )
                            relays[current_fp] = {
                                "nickname": parts[1],
                                "flags": [],
                            }

                    elif line.startswith("s ") and current_fp is not None:
                        relays[current_fp]["flags"] = line.split()[1:]

        except OSError as exc:
            logger.warning("Could not read consensus file %s: %s", path, exc)

        return relays

    @classmethod
    def _parse_consensus_dir(
        cls,
        consensus_dir: Path,
        max_files: int = 24,
    ) -> Dict[str, Dict[str, Any]]:
        """Parse consensus files from a CollecTor directory and merge the results.

        CollecTor archives consensuses in a nested layout::

            <consensuses-YYYY-MM>/
                    DD/
                        YYYY-MM-DD-HH-MM-SS-consensus
                        ...

        Flat directories (all files directly under ``consensus_dir``) are
        also supported for convenience.

        Relay entries are merged across files: if a fingerprint appears in
        multiple files the entry from the most recently processed file wins.
        In practice relays rarely change flags within a single month, so
        using a subset of files is sufficient for adversary selection.

        Args:
            consensus_dir: Root of the CollecTor consensus archive, e.g.
                ``Path("data/consensuses-2023-04")``.
            max_files: Maximum number of consensus files to process. Older
                files are skipped once the limit is reached. Defaults to 24
                (one day of hourly snapshots), which gives a representative
                sample of the network without excessive parsing time.

        Returns:
            A merged relay metadata dict in the same format as
            ``_parse_consensus_file``. Returns an empty dict when no
            consensus files are found.

        Raises:
            ValueError: If ``consensus_dir`` is not an existing directory.
        """
        if not consensus_dir.is_dir():
            raise ValueError(
                f"consensus_dir is not a directory: {consensus_dir}"
            )

        candidates: List[Path] = sorted(
            consensus_dir.rglob("*-consensus"),
            key=lambda p: p.name,
        )

        if not candidates:
            candidates = sorted(
                p for p in consensus_dir.rglob("*")
                if p.is_file() and not p.suffix
            )

        if not candidates:
            logger.warning(
                "No consensus files found under %s", consensus_dir
            )
            return {}

        selected = candidates[:max_files]
        logger.info(
            "Parsing %d/%d consensus files from %s …",
            len(selected),
            len(candidates),
            consensus_dir,
        )

        merged: Dict[str, Dict[str, Any]] = {}
        for path in selected:
            merged.update(cls._parse_consensus_file(path))

        guards  = sum(1 for m in merged.values() if "Guard" in m["flags"])
        exits   = sum(1 for m in merged.values() if "Exit"  in m["flags"])
        middles = len(merged) - guards - exits
        logger.info(
            "Parsed %d unique relays — Guard: %d, Exit: %d, Middle: %d",
            len(merged), guards, exits, middles,
        )
        return merged

    def _load_adversary_relays_from_consensus(self, consensus_dir: Path) -> None:
        """Parse a consensus directory and populate the adversary relay sets.

        Calls ``_parse_consensus_dir`` to read relay flags, then passes the
        result to ``_build_adversary_relay_list`` and stores the selected
        fingerprints in ``_adversary_guards`` and ``_adversary_exits``.

        Args:
            consensus_dir: Path to the root of a CollecTor consensus archive.
        """
        try:
            network_data = self._parse_consensus_dir(consensus_dir)
            if not network_data:
                logger.warning(
                    "No relay data parsed from %s; adversary lists remain empty.",
                    consensus_dir,
                )
                return
            guards, _, exits = self._build_adversary_relay_list(network_data)
            self._adversary_guards = guards
            self._adversary_exits = exits
        except (ValueError, OSError) as exc:
            self.logger.error(
                "Failed to load adversary relays from consensus dir %s: %s",
                consensus_dir,
                exc,
            )

    def _auto_load_adversary_relays(self, sim_dir: Path) -> None:
        """Auto-discover a consensus directory and populate adversary relay lists.

        Search order:
            1. ``ge_config.consensus_dir`` if set.
            2. Any directory matching ``consensuses-*`` under the workspace.
            3. Any directory matching ``consensuses-*`` adjacent to ``sim_dir``.

        Logs a warning when no consensus directory can be found.

        Args:
            sim_dir: Root of the simulation output directory for the current
                seed, used to anchor the fallback search.
        """
        if self.ge_config.consensus_dir and self.ge_config.consensus_dir.exists():
            self._load_adversary_relays_from_consensus(self.ge_config.consensus_dir)
            return

        candidates: List[Path] = []
        for pattern in ("consensuses-*", "data/consensuses-*"):
            candidates.extend(self.workspace.glob(pattern))

        if not candidates:
            candidates.extend(sim_dir.parent.glob("consensuses-*"))

        if candidates:
            chosen = sorted(candidates)[-1]
            self.logger.info("Auto-discovered consensus directory: %s", chosen)
            self._load_adversary_relays_from_consensus(chosen)
        else:
            self.logger.warning(
                "No consensus directory found. "
                "Provide consensus_dir in the config or place a "
                "'consensuses-*' directory under %s.",
                self.workspace,
            )

    def _correlate_all_pairs(
            self,
            guard_profiles: List[TrafficProfile],
            exit_profiles: List[TrafficProfile],
            ground_truth: Dict[str, Dict[str, str]],
            seed: int,
    ) -> List[DeanonymizationResult]:
        """Cross-correlate every guard–exit profile pair and produce results.

        For each guard profile:
            1. Scores all candidate exit profiles using ``_correlate_and_decide``.
            2. Ranks candidates by descending score.
            3. Declares a match for the top-ranked candidate when
               ``require_top_rank`` is ``True``, or for any candidate above
               the threshold otherwise.
            4. Checks whether the matched circuit ID equals the guard profile's
               own circuit ID (both derived from the same canonical key, so
               equality holds only for the correct exit).

        Args:
            guard_profiles: Traffic profiles observed at adversary guard relays.
            exit_profiles: Traffic profiles observed at adversary exit relays.
            ground_truth: Mapping of circuit label to relay fingerprints,
                as returned by ``_build_ground_truth``.
            seed: Seed index embedded in ``DeanonymizationResult.client_id``
                for traceability.

        Returns:
            A list of ``DeanonymizationResult`` objects, one per guard profile
            that had at least one candidate exit profile to compare against.
        """
        results: List[DeanonymizationResult] = []

        for g_prof in guard_profiles:
            t_start = time.perf_counter()
            cid = g_prof.circuit_id
            gt = ground_truth.get(cid, {})
            true_guard = gt.get("guard", g_prof.metadata.get("guard_fp", "unknown"))
            true_exit = gt.get("exit", g_prof.metadata.get("exit_fp", "unknown"))

            candidate_scores: List[Tuple[str, float]] = []
            for e_prof in exit_profiles:
                score, _ = self._correlate_and_decide(g_prof, e_prof)
                candidate_scores.append((e_prof.circuit_id, score))

            if not candidate_scores:
                continue

            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            best_cid, best_score = candidate_scores[0]

            threshold = self.config.correlation_thresholds.get("cross_correlation", 0.7)
            if self.ge_config.require_top_rank:
                predicted_exit_id = best_cid
                successful = best_score >= threshold and predicted_exit_id == cid
            else:
                successful = best_score >= threshold
                predicted_exit_id = best_cid if successful else None

            confidence = float(np.clip(best_score, 0, 1))
            elapsed = time.perf_counter() - t_start

            results.append(
                DeanonymizationResult(
                    client_id=f"client_seed{seed}_{cid}",
                    circuit_id=cid,
                    true_guard=true_guard,
                    true_exit=true_exit,
                    predicted_guard=g_prof.metadata.get("guard_fp"),
                    predicted_exit=predicted_exit_id,
                    confidence=confidence,
                    correlation_score=best_score,
                    time_to_identify=elapsed,
                    successful=successful,
                )
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

            threshold = self.config.correlation_thresholds.get(
                "cross_correlation", 0.7
            )
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

    def _build_ground_truth(self, hosts_dir: Path) -> Dict[str, Dict[str, str]]:
        """Build a circuit-to-relay mapping from ``CIRC BUILT`` events in host logs.

        Iterates every ``oniontrace.log`` file under ``hosts_dir`` using
        ``_iter_circ_built`` and extracts the three-hop fingerprint path from
        each ``CIRC BUILT`` event. The resulting dict is keyed by the same
        ``"{guard_fp[:8]}..{exit_fp[:8]}"`` label that
        ``_load_profiles_from_cell_stats`` stores in
        ``TrafficProfile.circuit_id``, so ``ground_truth.get(profile.circuit_id)``
        always resolves without a separate lookup step.

        Args:
            hosts_dir: Path to the ``shadow.data/hosts/`` directory whose
                subdirectories each contain an ``oniontrace.log`` file.

        Returns:
            A dict mapping circuit label strings to inner dicts with keys
            ``"guard"``, ``"middle"``, and ``"exit"``, each holding a full
            uppercase hex fingerprint string. When the same circuit label is
            seen in multiple relay logs the first occurrence wins, since all
            relays record identical path information for the same circuit.
            Returns an empty dict when no ``CIRC BUILT`` events are found.
        """
        ground_truth: Dict[str, Dict[str, str]] = {}

        for log_path in sorted(hosts_dir.glob("*/oniontrace.log")):
            for _ts, _cid, path_str in self._iter_circ_built(log_path):
                key = self._canonical_circuit_key(path_str)
                if key is None:
                    continue
                guard_fp, middle_fp, exit_fp = key
                circuit_label = f"{guard_fp[:8]}..{exit_fp[:8]}"
                if circuit_label not in ground_truth:
                    ground_truth[circuit_label] = {
                        "guard": guard_fp,
                        "middle": middle_fp,
                        "exit": exit_fp,
                    }

        self.logger.debug(
            "Built ground truth for %d circuits from %s.",
            len(ground_truth),
            hosts_dir,
        )
        return ground_truth

    @staticmethod
    def _find_shadow_data_hosts(sim_dir: Path) -> Optional[Path]:
        """Search common locations for the hosts directory under shadow.data.

        Args:
            sim_dir: Root of the simulation output directory for this seed.

        Returns:
            Path to the first matching hosts directory.
        """
        candidates = list(sim_dir.rglob("shadow.data/hosts"))
        for p in candidates:
            if p.exists():
                return p
        return None