import abc
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analysis.deanonymization import DeanonymizationResult, evaluate_attack

@dataclass
class AttackConfig:
    """Generic configuration shared by all attack types.

    Additional scenario-specific keys are stored in ``extra`` and forwarded
    to the concrete subclass via ``BaseAttack.configure``.

    Attributes:
        name: Unique scenario identifier used in filenames and reports.
        description: Human-readable description of the scenario.
        num_seeds: Number of independent simulation seeds to average over.
        client_filter: When set, restricts ground-truth building (and therefore
            all correlation and metrics) to circuits originating from a specific
            set of client hosts. Accepted formats:
            * ``"group:<n>"``  - all hosts in that injection group, resolved
              via ``custom_clients_manifest.json`` written by
              ``inject_custom_clients``.
            * ``"host:h1,h2"`` - explicit comma-separated Shadow hostnames.
            * ``None`` (default) - no filtering; all client circuits are used.
        extra: Free-form dict for scenario-specific parameters that do not
            belong to the common fields.
    """
    name: str
    description: str = ""
    num_seeds: int = 3
    client_filter: Optional[str] = None
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
            as ``success_rate``, ``coverage``, ``conditional_accuracy``).
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
    def success_rate(self) -> float:
        """float: Fraction of circuits successfully deanonymized."""
        return float(self.metrics.get("success_rate", 0.0))

    @property
    def coverage(self) -> float:
        """coverage: Fraction of circuits For which a prediction was made."""
        return float(self.metrics.get("coverage", 0.0))

    @property
    def conditional_accuracy(self) -> float:
        """float: Accuracy of deanonymization attempts."""
        return float(self.metrics.get("conditional_accuracy", 0.0))

    def summary(self) -> str:
        """Return a compact multi-line summary string suitable for logging.

        Returns:
            A newline-joined string with attack name, circuit count, seed
            count, success rate, coverage, conditional accuracy.
        """
        lines = [
            f"Attack : {self.attack_name} - {self.scenario_label}",
            f"Circuits: {len(self.deanon_results)}  "
            f"Seeds: {self.config.num_seeds}",
            f"Success : {self.success_rate:.2%}",
            f"Coverage : {self.coverage:.2%}",
            f"Conditional accuracy : {self.conditional_accuracy:.2%}",
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
                correlation method, and seed count.
            workspace: Root directory for intermediate files. Defaults to
                ``./workspace`` relative to the current working directory.
        """
        self.config = config
        self.workspace = Path(workspace) if workspace else Path("./workspace")
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


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
                "Expected %d seed dirs, got %d - proceeding anyway.",
                self.config.num_seeds,
                len(simulation_dirs),
            )

        t0 = time.perf_counter()
        all_results: List[DeanonymizationResult] = []
        per_seed_metrics: List[Dict[str, Any]] = []

        for seed_idx, sim_dir in enumerate(simulation_dirs):
            self.logger.info(
                "[%s] Seed %d/%d - %s",
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