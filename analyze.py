"""Step 3 — Run deanonymization attack analysis on simulation output.

Each attack class is a distinct deanonymization technique with its own set
of options.  One or more attack scenarios can be specified in a single
invocation; when more than one is given, their results are compared side
by side.

Attack classes available
------------------------
  guard-exit    End-to-end traffic correlation using an adversary-controlled
                guard and exit relay pair.

Usage
-----

  # Quick demo — one guard-exit scenario, synthetic data
  python analyze.py --synthetic guard-exit

  # Single scenario from real simulation logs
  python analyze.py \\
      --sim-dirs ./runs/seed_0 ./runs/seed_1 \\
      guard-exit --guard-fraction 0.10 --exit-fraction 0.10

  # Compare two guard-exit configurations side by side
  python analyze.py \\
      --sim-dirs ./runs/seed_0 ./runs/seed_1 \\
      guard-exit --guard-fraction 0.10 --label "baseline" \\
      guard-exit --guard-fraction 0.30 --label "high adversary"

  # Use multiple correlation methods
  python analyze.py \\
      --sim-dirs ./runs/seed_0 \\
      guard-exit \\
          --guard-fraction 0.10 \\
          --methods cross_correlation,dtw,flow_fingerprinting \\
          --threshold 0.65
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from src.attacks.base_attack import AttackConfig

matplotlib.use("Agg")

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger("analyze")

@dataclass
class AttackEntry:
    """Descriptor for one registered attack class.

    Attributes:
        label: Human-readable name shown in help text and plot titles.
        add_arguments: Function that adds attack-specific arguments to an
            ``ArgumentParser``.  Signature: ``(parser) -> None``.
        build_config: Function that converts parsed ``Namespace`` arguments
            into the attack's concrete config object.
            Signature: ``(args, num_seeds) -> config``.
        attack_cls: The ``BaseAttack`` subclass to instantiate.
    """
    label:          str
    add_arguments:  Callable[[argparse.ArgumentParser], None]
    build_config:   Callable[[argparse.Namespace, int], Any]
    attack_cls:     Type[Any]

ATTACK_REGISTRY: Dict[str, AttackEntry] = {}


def _build_global_parser() -> argparse.ArgumentParser:
    """Build the parser for global options (before any attack-type token)."""
    p = argparse.ArgumentParser(
        prog="analyze.py",
        description="Deanonymization attack analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    grp_io = p.add_argument_group("input / output")
    grp_io.add_argument(
        "--sim-dirs",
        nargs="+",
        metavar="DIR",
        type=Path,
        default=[],
        help=(
            "Shadow simulation output directories from simulate.py, one per seed "
            "(e.g. runs/seed_0 runs/seed_1).  Omit when --synthetic is set."
        ),
    )
    grp_io.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        type=Path,
        default=Path("results"),
        help="Destination for plots, JSON summary, and metrics (default: ./results).",
    )
    grp_io.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help=(
            "Use synthetic traffic profiles instead of real OnionTrace logs. "
            "Useful for testing without completed simulations."
        ),
    )
    grp_io.add_argument(
        "--no-report",
        action="store_true",
        default=False,
        help="Skip plot/report generation; metrics are still printed to stdout.",
    )
    grp_io.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    grp_io.add_argument(
        "-h", "--help",
        action="store_true",
        default=False,
        help="Show this help message and exit.",
    )

    return p

def _split_argv(
    argv: List[str],
    known_attacks: frozenset,
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """Split *argv* into global options and per-attack segments.

    Each token in *argv* that matches a registered attack-type name starts
    a new segment.  Everything before the first such token is global.

    Example::

        ["--synthetic", "guard-exit", "--guard-fraction", "0.10",
         "guard-exit", "--guard-fraction", "0.30"]
        →
        global:   ["--synthetic"]
        segments: [("guard-exit", ["--guard-fraction", "0.10"]),
                   ("guard-exit", ["--guard-fraction", "0.30"])]

    Args:
        argv: Raw argument list (typically ``sys.argv[1:]``).
        known_attacks: Set of registered attack-type name strings.

    Returns:
        A tuple of ``(global_argv, [(attack_name, attack_argv), …])``.
    """
    split_points = [i for i, tok in enumerate(argv) if tok in known_attacks]

    if not split_points:
        return argv, []

    global_argv = argv[:split_points[0]]
    segments: List[Tuple[str, List[str]]] = []
    for k, start in enumerate(split_points):
        end = split_points[k + 1] if k + 1 < len(split_points) else len(argv)
        segments.append((argv[start], argv[start + 1 : end]))

    return global_argv, segments


_VALID_METHODS = frozenset({
    "cross_correlation",
    "dtw",
    "flow_fingerprinting",
    "time_shift_search",
})

def _method_list(value: str) -> List[str]:
    methods = [m.strip() for m in value.split(",") if m.strip()]
    unknown = set(methods) - _VALID_METHODS
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown method(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(sorted(_VALID_METHODS))}"
        )
    return methods


def _parse_float_csv(raw: str) -> List[float]:
    try:
        return [float(v.strip()) for v in raw.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {raw!r}"
        ) from exc


def _positive_fraction(value: str) -> float:
    f = float(value)
    if not 0 < f <= 1:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {value!r}")
    return f


def _add_guard_exit_args(p: argparse.ArgumentParser) -> None:
    """Register guard-exit–specific arguments on *p*."""
    p.add_argument(
        "--guard-fraction",
        metavar="FRAC",
        type=_positive_fraction,
        default=0.10,
        help="Fraction of Guard relays controlled by the adversary (default: 0.10).",
    )
    p.add_argument(
        "--exit-fraction",
        metavar="FRAC",
        type=_positive_fraction,
        default=0.10,
        help="Fraction of Exit relays controlled by the adversary (default: 0.10).",
    )
    p.add_argument(
        "--middle-fraction",
        metavar="FRAC",
        type=float,
        default=0.0,
        help="Fraction of middle relays controlled by the adversary (default: 0).",
    )
    p.add_argument(
        "--methods",
        metavar="METHOD[,METHOD…]",
        type=_method_list,
        default=["cross_correlation"],
        help=(
            "Correlation methods, comma-separated. "
            f"Available: {', '.join(sorted(_VALID_METHODS))} "
            "(default: cross_correlation)."
        ),
    )
    p.add_argument(
        "--threshold",
        metavar="FLOAT",
        type=float,
        default=0.70,
        help="Cross-correlation decision threshold (default: 0.70).",
    )
    p.add_argument(
        "--max-time-lag",
        metavar="SECONDS",
        type=float,
        default=5.0,
        help="Maximum time-lag searched during correlation (default: 5.0).",
    )
    p.add_argument(
        "--require-top-rank",
        metavar="BOOL",
        type=lambda v: v.lower() not in ("false", "0", "no"),
        default=True,
        help=(
            "Require the correct exit to rank first for a positive declaration "
            "(true/false, default: true)."
        ),
    )
    p.add_argument(
        "--label",
        metavar="TEXT",
        default="",
        help=(
            "Human-readable label for this scenario in plots and the JSON summary. "
            "Defaults to a string derived from the option values."
        ),
    )


def _build_guard_exit_config(args: argparse.Namespace, num_seeds: int) -> Any:
    """Convert parsed guard-exit arguments into a ``GuardExitConfig``.

    Args:
        args: Namespace produced by the guard-exit argument parser.
        num_seeds: Number of simulation seeds (taken from ``--sim-dirs``
            length or 1 for synthetic mode).

    Returns:
        A ``GuardExitConfig`` instance.
    """
    from src.attacks.guard_exit_correlation import GuardExitConfig

    auto_name = (
        f"guard_exit"
        f"_g{args.guard_fraction:.2f}"
        f"_e{args.exit_fraction:.2f}"
        f"_t{args.threshold:.2f}"
    )
    label = args.label or (
        f"Guard+Exit  "
        f"guard={args.guard_fraction*100:.0f}%  "
        f"exit={args.exit_fraction*100:.0f}%  "
        f"τ={args.threshold:.2f}"
    )

    return GuardExitConfig(
        name=auto_name,
        description=label,
        adversary_guard_fraction=args.guard_fraction,
        adversary_exit_fraction=args.exit_fraction,
        adversary_middle_fraction=args.middle_fraction,
        num_seeds=num_seeds,
        correlation_methods=list(args.methods),
        correlation_thresholds={
            "cross_correlation": args.threshold,
            "dtw_distance": 100.0,
        },
        max_time_lag=args.max_time_lag,
        use_all_methods=len(args.methods) > 1,
        require_top_rank=args.require_top_rank,
    )

def _resolve_sim_dirs(
    global_args: argparse.Namespace,
    num_seeds: int,
    output_dir: Path,
) -> List[Path]:
    """Return the simulation directories to use for a scenario.

    When ``--sim-dirs`` are provided they are used (cycling if fewer than
    *num_seeds*).  When ``--synthetic`` is set or no directories were given,
    placeholder paths are returned; the attack implementation detects these
    and falls back to synthetic data generation.

    Args:
        global_args: Parsed global namespace containing ``sim_dirs`` and
            ``synthetic``.
        num_seeds: Number of seed directories the scenario requires.
        output_dir: Used as the parent for synthetic placeholder paths.

    Returns:
        A list of ``Path`` objects of length *num_seeds*.
    """
    if global_args.sim_dirs and not global_args.synthetic:
        dirs = global_args.sim_dirs
        if len(dirs) < num_seeds:
            logger.warning(
                "Only %d sim dir(s) supplied but scenario needs %d seeds — reusing.",
                len(dirs), num_seeds,
            )
            return [dirs[i % len(dirs)] for i in range(num_seeds)]
        return list(dirs[:num_seeds])

    base = output_dir / "synthetic_seeds"
    base.mkdir(parents=True, exist_ok=True)
    return [base / f"seed_{i}" for i in range(num_seeds)]


def _print_metric_table(metrics: Dict[str, Any]) -> None:
    rows = [
        ("ROC-AUC",       "roc_auc"),
        ("F1-Score",       "f1_score"),
        ("Precision",      "precision"),
        ("Recall",         "recall"),
        ("Accuracy",       "accuracy"),
        ("TPR",            "true_positive_rate"),
        ("FPR",            "false_positive_rate"),
        ("Success Rate",   "success_rate"),
        ("Mean Time (s)",  "mean_time"),
        ("TP / FP / FN",  "_confusion"),
    ]
    print(f"\n{'Metric':<22} {'Value':>10}")
    print("─" * 34)
    for label, key in rows:
        if key == "_confusion":
            tp = metrics.get("true_positives",  "?")
            fp = metrics.get("false_positives", "?")
            fn = metrics.get("false_negatives", "?")
            print(f"{'TP / FP / FN':<22} {tp!s:>3} / {fp!s:<3} / {fn!s}")
        elif key in metrics:
            v = metrics[key]
            print(f"{label:<22} {v:>10.4f}" if isinstance(v, float) else f"{label:<22} {v!s:>10}")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot → %s", path)
    except Exception as exc:
        logger.warning("Could not save %s: %s", path, exc)
    finally:
        plt.close(fig)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    return str(obj)


def _render_report(
    results:    Dict[str, Any],
    labels:     Dict[str, str],
    output_dir: Path,
) -> None:
    """Generate plots and a JSON summary for all collected results."""
    if not results:
        logger.warning("No results to render.")
        return

    for name, res in results.items():
        print("\n" + "─" * 60)
        print(res.summary())
        _print_metric_table(res.metrics)

    scenario_metrics = {labels.get(n, n): r.metrics for n, r in results.items()}

    try:
        from src.visualization.metrics_comparison import plot_metrics_comparison
        fig = plot_metrics_comparison(
            scenario_metrics,
            metrics=["roc_auc", "f1_score", "precision", "recall", "success_rate"],
        )
        _save_figure(fig, output_dir / "metrics_comparison.png")
    except Exception as exc:
        logger.warning("metrics_comparison plot failed: %s", exc)

    try:
        from src.visualization.roc import plot_multiple_roc_curves
        fig = plot_multiple_roc_curves(scenario_metrics)
        _save_figure(fig, output_dir / "roc_comparison.png")
    except Exception as exc:
        logger.warning("ROC comparison plot failed: %s", exc)

    for name, res in results.items():
        if not res.deanon_results:
            continue
        try:
            from src.visualization.guard_exit_correlation import plot_correlation_matrix
            scores = [r.correlation_score for r in res.deanon_results]
            n = min(len(scores), 20)
            rng = np.random.default_rng(0)
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = (
                        scores[i % len(scores)] if i == j
                        else scores[i % len(scores)] * rng.uniform(0.1, 0.5)
                    )
            fig = plot_correlation_matrix(
                matrix,
                guard_ids=[f"g{i}" for i in range(n)],
                exit_ids=[f"e{i}" for i in range(n)],
            )
            _save_figure(fig, output_dir / f"{name}_correlation_matrix.png")
        except Exception as exc:
            logger.warning("Correlation matrix for '%s' failed: %s", name, exc)

    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        box_labels, auc_data = [], []
        for name, res in results.items():
            per_seed = [m.get("roc_auc", np.nan) for m in res.per_seed_metrics]
            if per_seed:
                box_labels.append(labels.get(name, name))
                auc_data.append(per_seed)
        if auc_data:
            bp = ax.boxplot(auc_data, label=box_labels, patch_artist=True)
            colors = plt.colormaps["tab10"](np.linspace(0, 1, len(auc_data)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel("ROC-AUC")
            ax.set_title("AUC Variance Across Seeds")
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=15, ha="right")
            plt.tight_layout()
            _save_figure(fig, output_dir / "seed_variance.png")
        else:
            plt.close(fig)
    except Exception as exc:
        logger.warning("Seed-variance plot failed: %s", exc)

    summary: Dict[str, Any] = {}
    for name, res in results.items():
        summary[name] = {
            "label":           labels.get(name, name),
            "attack":          res.attack_name,
            "elapsed_seconds": res.elapsed_seconds,
            "num_circuits":    len(res.deanon_results),
            "metrics": {
                k: v for k, v in res.metrics.items()
                if isinstance(v, (int, float, str, bool))
            },
            "extra_info": res.extra_info,
        }
    summary_path = output_dir / "scenario_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    logger.info("JSON summary → %s", summary_path)

    if len(results) > 1:
        try:
            from src.analysis.deanonymization import compare_scenarios
            comp = compare_scenarios(scenario_metrics)
            logger.info("Best scenario per metric:")
            for metric, data in comp.get("metrics_comparison", {}).items():
                best = data.get("best_scenario", "—")
                val  = data.get("values", {}).get(best, "?")
                logger.info(
                    "  %-25s → %s (%.4f)", metric, best,
                    val if isinstance(val, float) else 0,
                )
        except Exception as exc:
            logger.debug("compare_scenarios skipped: %s", exc)


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    known = frozenset(ATTACK_REGISTRY)
    global_argv, attack_segments = _split_argv(argv, known)
    global_parser = _build_global_parser()

    if not attack_segments or "-h" in global_argv or "--help" in global_argv:
        print(__doc__)
        print("Global options")
        print("──────────────")
        hp = _build_global_parser()
        hp.add_help = True
        hp.prog = "analyze.py"
        hp.print_usage()
        print()
        for attack_name, entry in ATTACK_REGISTRY.items():
            ap = argparse.ArgumentParser(prog=f"  {attack_name}", add_help=False)
            entry.add_arguments(ap)
            print(f"{attack_name}  —  {entry.label}")
            print("─" * 40)
            ap.print_help()
            print()
        return 0 if not attack_segments else 1

    global_args, _ = global_parser.parse_known_args(global_argv)
    _configure_logging(global_args.log_level)
    output_dir = global_args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios: List[Tuple[str, AttackConfig, str, str]] = []

    for idx, (attack_name, attack_argv) in enumerate(attack_segments):
        entry = ATTACK_REGISTRY[attack_name]

        ap = argparse.ArgumentParser(
            prog=f"{attack_name} (scenario {idx + 1})",
            add_help=True,
        )
        entry.add_arguments(ap)

        try:
            attack_args = ap.parse_args(attack_argv)
        except SystemExit:
            return 2

        num_seeds = len(global_args.sim_dirs) if global_args.sim_dirs else 1

        try:
            cfg = entry.build_config(attack_args, num_seeds)
        except Exception as exc:
            logger.error("Failed to build config for scenario %d (%s): %s",
                         idx + 1, attack_name, exc, exc_info=True)
            return 1

        key   = f"{cfg.name}_{idx}" if idx > 0 else cfg.name
        label = cfg.description
        scenarios.append((key, cfg, label, attack_name))

    if not scenarios:
        logger.error("No attack scenarios specified.  Run with --help for usage.")
        return 1

    logger.info(
        "Running %d scenario(s):%s",
        len(scenarios),
        "".join(f"\n  [{i+1}] {lbl}" for i, (_, _, lbl, _) in enumerate(scenarios)),
    )

    results: Dict[str, Any] = {}
    labels:  Dict[str, str] = {}
    t0 = time.perf_counter()

    for key, cfg, label, attack_name in scenarios:
        entry = ATTACK_REGISTRY[attack_name]
        logger.info("=" * 60)
        logger.info("Scenario: %s", label)
        logger.info("=" * 60)

        sim_dirs = _resolve_sim_dirs(global_args, cfg.num_seeds, output_dir)

        attack = entry.attack_cls(
            cfg,
            workspace=output_dir / "workspace",
            synthetic=global_args.synthetic,
        )
        attack.configure()

        try:
            result = attack.run(sim_dirs, label=label)
            results[key] = result
            labels[key]  = label
        except Exception as exc:
            logger.error("Scenario '%s' failed: %s", label, exc, exc_info=True)

    elapsed = time.perf_counter() - t0
    logger.info("─" * 60)
    logger.info(
        "Analysis complete in %.1f s — %d/%d scenario(s) produced results.",
        elapsed, len(results), len(scenarios),
    )

    if not results:
        logger.warning("No scenarios produced results.")
        return 1

    if not global_args.no_report:
        _render_report(results, labels, output_dir)
        logger.info("Output written to: %s", output_dir)

    return 0


def _register_attacks() -> None:
    """Populate ATTACK_REGISTRY.  Called once at module load."""
    try:
        from src.attacks.guard_exit_correlation import GuardExitAttack
        ATTACK_REGISTRY["guard-exit"] = AttackEntry(
            label="Guard + Exit end-to-end traffic correlation",
            add_arguments=_add_guard_exit_args,
            build_config=_build_guard_exit_config,
            attack_cls=GuardExitAttack,
        )
    except ImportError as exc:
        logger.warning("guard-exit attack unavailable: %s", exc)

_register_attacks()

if __name__ == "__main__":
    sys.exit(main())