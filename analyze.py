"""Step 3 - Run deanonymization attack analysis on simulation output.

Each attack class is a distinct deanonymization technique with its own set
of options. One or more attack scenarios can be specified in a single
invocation; when more than one is given, their results are compared side
by side.

Metrics used
------------
  success_rate            - fraction of all observed circuits correctly
                            identified (unconditional).
  coverage                - fraction of circuits for which a prediction was
                            made (at least one candidate found).
  conditional_accuracy    - fraction of predictions that were correct.
  score_separation        - mean(score|correct) − mean(score|incorrect),
                            a measure of discriminability.
  timing                  - mean / median / p95 wall-clock seconds per circuit.

Attack classes available
------------------------
  guard-exit    End-to-end traffic correlation using an adversary-controlled
                guard and exit relay pair.

Usage
-----

  # Single scenario from real simulation logs
  python analyze.py \\
      --sim-dirs ./runs/seed_0 ./runs/seed_1 \\
      guard-exit --guard-fraction 0.10 --exit-fraction 0.10

  # Compare two guard-exit configurations side by side
  python analyze.py \\
      --sim-dirs ./runs/seed_0 ./runs/seed_1 \\
      guard-exit --guard-fraction 0.10 --label "baseline" \\
      guard-exit --guard-fraction 0.30 --label "high adversary"

  # Generate specific plots only
  python analyze.py \\
      --sim-dirs ./runs/seed_0 ./runs/seed_1 \\
      --plots success_bar accuracy_coverage score_dist \\
      guard-exit --guard-fraction 0.15

  # Generate all available plots
  python analyze.py \\
      --sim-dirs ./runs/seed_0 ./runs/seed_1 \\
      --plots all guard-exit
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.deanonymization import DeanonymizationResult
from src.analysis.metrics import compare_scenarios, compute_seed_variance, compute_threshold_sweep, ThresholdPoint, IdentificationMetrics
from src.visualization import plots as vplots

matplotlib.use("Agg")

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger("analyze")

_ALL_PLOTS = frozenset({
    "success_bar",          # grouped bar chart - metrics across scenarios
    "accuracy_coverage",    # threshold-sweep curve for each scenario
    "score_dist",           # histogram of scores: correct vs. incorrect
    "seed_variance",        # box plot of per-seed success rates
    "ge_matrix",            # guard-fraction × exit-fraction heatmap (guard-exit only)
})

_PLOT_HELP = (
    "Plots to generate.  Pass individual names or 'all'.  "
    f"Available: {', '.join(sorted(_ALL_PLOTS))}.  "
    "Default: success_bar accuracy_coverage."
)

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
    label: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    build_config: Callable[[argparse.Namespace, int], Any]
    attack_cls: Type[Any]

ATTACK_REGISTRY: Dict[str, AttackEntry] = {}


def _build_global_parser() -> argparse.ArgumentParser:
    """Build the parser for global options (before any attack-type token)."""
    p = argparse.ArgumentParser(
        prog="analyze.py",
        description="Deanonymization attack analysis (top-1 identification).",
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
            "Shadow simulation output directories from simulate.py, one per seed. "
        ),
    )
    grp_io.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        type=Path,
        default=Path("results"),
        help="Destination for plots and JSON summary (default: ./results).",
    )
    grp_io.add_argument(
        "--plots",
        nargs="+", metavar="PLOT",
        default=["success_bar", "accuracy_coverage"],
        help=_PLOT_HELP,
    )
    grp_io.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip all plot generation; metrics JSON is still written.",
    )
    grp_io.add_argument(
        "--sweep-thresholds",
        metavar="N", type=int, default=100,
        help="Number of threshold values for accuracy/coverage sweep (default: 100).",
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

        ["--output-dir", "./res", "guard-exit", "--guard-fraction", "0.10",
         "guard-exit", "--guard-fraction", "0.30"]
        →
        global:   ["--output-dir", "./res"]
        segments: [("guard-exit", ["--guard-fraction", "0.10"]),
                   ("guard-exit", ["--guard-fraction", "0.30"])]
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


def _positive_fraction(value: str) -> float:
    f = float(value)
    if not 0 < f <= 1:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {value!r}")
    return f


def _add_guard_exit_args(p: argparse.ArgumentParser) -> None:
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
        "--max-guard-profiles",
        type=int,
        default=None,
        help="Maximum number of guard profiles to test for this scenario.",
    )

    p.add_argument(
        "--max-exit-profiles",
        type=int,
        default=None,
        help="Maximum number of exit profiles to test for this scenario.",
    )

    p.add_argument(
        "--deanon-circuit-fraction",
        type=_positive_fraction,
        default=None,
        metavar="FRAC",
        help="Total fraction of analyzed circuits that have both guard and exit"
             "compromised. Unnecessary circuits are trimmed.",
    )
    p.add_argument(
        "--method",
        metavar="METHOD",
        type=str,
        default="cross_correlation",
        help=(
            "Traffic correlation method."
            f"Available: {', '.join(sorted(_VALID_METHODS))} "
            "(default: cross_correlation)."
        ),
    )
    p.add_argument(
        "--threshold",
        metavar="FLOAT",
        type=float,
        default=0.70,
        help="Correlation decision threshold (default: 0.70).",
    )
    p.add_argument(
        "--max-time-lag",
        metavar="SECONDS",
        type=float,
        default=5.0,
        help="Maximum time-lag searched during correlation (default: 5.0).",
    )
    p.add_argument(
        "--bin-size",
        metavar="SECONDS",
        type=float,
        default=1.0,
        help=(
            "Width in seconds of each time bin used for traffic "
            "histogramming."
        ),
    )
    p.add_argument(
        "--label",
        metavar="TEXT",
        default="",
        help=(
            "Human-readable label for this scenario in plots and JSON. "
            "Auto-generated from option values when omitted."
        ),
    )
    p.add_argument(
        "--client-filter",
        metavar="SPEC",
        default=None,
        help=(
            "Restrict analysis to circuits from a specific group of client hosts. "
            "Formats: 'group:<name>' (injection group from custom_clients_manifest.json), "
            "'host:h1,h2' (explicit Shadow hostnames). "
            "When omitted all client circuits are used."
        ),
    )


def _build_guard_exit_config(args: argparse.Namespace, num_seeds: int) -> Any:
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
        max_guard_profiles=args.max_guard_profiles,
        max_exit_profiles=args.max_exit_profiles,
        deanon_circ_frac=args.deanon_circuit_fraction,
        num_seeds=num_seeds,
        correlation_method=args.method,
        correlation_threshold=args.threshold,
        time_window=args.max_time_lag,
        bin_size=args.bin_size,
        client_filter=args.client_filter,
    )

def _resolve_sim_dirs(
    global_args: argparse.Namespace,
    num_seeds: int
) -> List[Path]:
    if global_args.sim_dirs:
        dirs = global_args.sim_dirs
        if len(dirs) < num_seeds:
            logger.warning(
                "Only %d sim dir(s) supplied but scenario needs %d seeds - reusing.",
                len(dirs), num_seeds,
            )
            return [dirs[i % len(dirs)] for i in range(num_seeds)]
        return list(dirs[:num_seeds])

    logger.error("Simulation directory not specified - exiting.")
    raise ValueError("Simulation directory not specified.")


def _print_metric_table(metrics: IdentificationMetrics) -> None:
    rows = [
        ("Total observed",       "total_observed"),
        ("Attempted",            "attempted"),
        ("Correctly identified", "correct"),
        ("Success rate",         "success_rate"),
        ("Coverage",             "coverage"),
        ("Abstention rate",      "abstention_rate"),
        ("Conditional accuracy", "conditional_accuracy"),
        ("Score separation",     "score_separation"),
    ]
    print(f"\n  {'Metric':<28} {'Value':>10}")
    print("  " + "─" * 34)
    for label, key in rows:
        v = getattr(metrics, key, None)
        if v is None:
            continue
        if isinstance(v, float):
            print(f"  {label:<28} {v:>10.4f}")
        else:
            print(f"  {label:<28} {v!s:>10}")

def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    return str(obj)


def _resolve_plot_set(raw: List[str]) -> frozenset:
    if "all" in raw:
        return _ALL_PLOTS
    unknown = set(raw) - _ALL_PLOTS
    if unknown:
        raise ValueError(
            f"Unknown plot(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(_ALL_PLOTS))}"
        )
    return frozenset(raw)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot → %s", path)
    except Exception as exc:
        logger.warning("Could not save %s: %s", path, exc)
    finally:
        plt.close(fig)


def _render_report(
        results: Dict[str, Any],
        labels: Dict[str, str],
        sweeps: Dict[str, List[ThresholdPoint]],
        deanon_map: Dict[str, List[Any]],
        per_seed_map: Dict[str, List[IdentificationMetrics]],
        output_dir: Path,
        plot_set: frozenset,
) -> None:
    """Generate plots and write the JSON summary.

    Args:
        results:      scenario_key → AttackResult.
        labels:       scenario_key → display label.
        sweeps:       scenario_key → list of ThresholdPoint (from
                      ``compute_threshold_sweep``).
        deanon_map:   scenario_key → list of DeanonymizationResult.
        per_seed_map: scenario_key → list of per-seed IdentificationMetrics.
        output_dir:   destination directory.
        plot_set:     set of plot names to generate.  Valid names:
                      ``"success_bar"``, ``"accuracy_coverage"``,
                      ``"score_dist"``, ``"seed_variance"``, ``"ge_matrix"``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, res in results.items():
        print("\n" + "─" * 60)
        print(res.summary())
        _print_metric_table(res.metrics)

    scenario_metrics: Dict[str, IdentificationMetrics] = {
        labels.get(k, k): r.metrics for k, r in results.items()
    }

    if "success_bar" in plot_set:
        try:
            fig = vplots.plot_success_rate_bar(scenario_metrics)
            _save_figure(fig, output_dir / "success_rate_bar.png")
        except NotImplementedError:
            logger.debug("plot_success_rate_bar not implemented — skipping.")
        except Exception as exc:
            logger.warning("success_bar plot failed: %s", exc)

    if "accuracy_coverage" in plot_set and sweeps:
        named_sweeps = {labels.get(k, k): sweeps[k] for k in sweeps if k in sweeps}
        try:
            if len(named_sweeps) == 1:
                label, sweep = next(iter(named_sweeps.items()))
                fig = vplots.plot_accuracy_coverage(sweep, scenario_label=label)
                _save_figure(fig, output_dir / "accuracy_coverage.png")
            else:
                fig = vplots.plot_accuracy_coverage_multi(named_sweeps)
                _save_figure(fig, output_dir / "accuracy_coverage_multi.png")
        except NotImplementedError:
            logger.debug("accuracy_coverage plots not implemented — skipping.")
        except Exception as exc:
            logger.warning("accuracy_coverage plot failed: %s", exc)

    if "score_dist" in plot_set:
        for key, dres in deanon_map.items():
            if not dres:
                continue
            try:
                fig = vplots.plot_score_distribution(
                    dres,
                    title=f"Score Distribution — {labels.get(key, key)}",
                )
                safe = key.replace(" ", "_").replace("/", "_")
                _save_figure(fig, output_dir / f"{safe}_score_dist.png")
            except NotImplementedError:
                logger.debug("plot_score_distribution not implemented — skipping.")
                break
            except Exception as exc:
                logger.warning("score_dist for '%s' failed: %s", key, exc)

    if "seed_variance" in plot_set:
        per_seed_named = {
            labels.get(k, k): v
            for k, v in per_seed_map.items()
            if v
        }
        if per_seed_named:
            try:
                fig = vplots.plot_seed_variance_box(per_seed_named)
                _save_figure(fig, output_dir / "seed_variance.png")
            except NotImplementedError:
                logger.debug("plot_seed_variance_box not implemented — skipping.")
            except Exception as exc:
                logger.warning("seed_variance plot failed: %s", exc)
        else:
            logger.debug("seed_variance: no multi-seed data available — skipping.")

    if "ge_matrix" in plot_set:
        ge_results = {
            k: r for k, r in results.items()
            if "adversary_guard_fraction" in r.extra_info
               and "adversary_exit_fraction" in r.extra_info
        }
        if len(ge_results) >= 4:
            try:
                gf_set = sorted({
                    r.extra_info["adversary_guard_fraction"]
                    for r in ge_results.values()
                })
                ef_set = sorted({
                    r.extra_info["adversary_exit_fraction"]
                    for r in ge_results.values()
                })
                matrix = np.zeros((len(gf_set), len(ef_set)))
                for r in ge_results.values():
                    gi = gf_set.index(r.extra_info["adversary_guard_fraction"])
                    ei = ef_set.index(r.extra_info["adversary_exit_fraction"])
                    matrix[gi, ei] = r.metrics.success_rate
                fig = vplots.plot_guard_exit_matrix(gf_set, ef_set, matrix)
                _save_figure(fig, output_dir / "guard_exit_matrix.png")
            except NotImplementedError:
                logger.debug("plot_guard_exit_matrix not implemented — skipping.")
            except Exception as exc:
                logger.warning("ge_matrix plot failed: %s", exc)
        else:
            logger.debug(
                "ge_matrix requires ≥4 scenarios with varying guard/exit fractions "
                "— skipping (found %d).", len(ge_results),
            )

    if len(results) > 1:
        comp = compare_scenarios(scenario_metrics)
        logger.info("Best scenario per metric:")
        for metric, mc in comp.metrics_comparison.items():
            val = mc.values.get(mc.best_scenario, 0.0)
            logger.info(
                "  %-28s → %-35s %.4f",
                metric, mc.best_scenario, val,
            )

    summary: Dict[str, Any] = {}
    for key, res in results.items():
        seed_variance = compute_seed_variance(per_seed_map.get(key, []))
        summary[key] = {
            "label": labels.get(key, key),
            "attack": res.attack_name,
            "elapsed_seconds": res.elapsed_seconds,
            "num_circuits": len(res.deanon_results),
            "metrics": res.metrics.to_dict(),
            "seed_variance": seed_variance.to_dict(),
            "extra_info": res.extra_info,
            "threshold_sweep": [
                {
                    "threshold": p.threshold,
                    "coverage": p.coverage,
                    "conditional_accuracy": p.conditional_accuracy,
                    "success_rate": p.success_rate,
                    "attempted": p.attempted,
                    "correct": p.correct,
                }
                for p in sweeps.get(key, [])
            ],
        }

    summary_path = output_dir / "scenario_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    logger.info("JSON summary → %s", summary_path)


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
        hp.print_usage()
        print()
        for name, entry in ATTACK_REGISTRY.items():
            ap = argparse.ArgumentParser(prog=f"  {name}", add_help=False)
            entry.add_arguments(ap)
            print(f"{name}  -  {entry.label}")
            print("─" * 40)
            ap.print_help()
            print()
        return 0 if not attack_segments else 1

    global_args, _ = global_parser.parse_known_args(global_argv)
    _configure_logging(global_args.log_level)
    output_dir = global_args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if global_args.no_plots:
        plot_set: frozenset = frozenset()
    else:
        try:
            plot_set = _resolve_plot_set(global_args.plots)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

    scenarios: List[Tuple[str, Any, str, str]] = []

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
            logger.error("Config build failed for scenario %d (%s): %s",
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
    labels: Dict[str, str] = {}
    sweeps: Dict[str, Any] = {}
    deanon_map: Dict[str, List[Any]] = {}
    per_seed_map: Dict[str, List[IdentificationMetrics]] = {}

    t0 = time.perf_counter()

    for key, cfg, label, attack_name in scenarios:
        entry = ATTACK_REGISTRY[attack_name]
        logger.info("=" * 60)
        logger.info("Scenario: %s", label)
        logger.info("=" * 60)

        sim_dirs = _resolve_sim_dirs(global_args, cfg.num_seeds)
        attack = entry.attack_cls(
            cfg,
            workspace=output_dir / "workspace",
        )
        attack.configure()

        try:
            result = attack.run(sim_dirs, label=label)
        except Exception as exc:
            logger.error("Scenario '%s' failed: %s", label, exc, exc_info=True)
            continue

        results[key]      = result
        labels[key]       = label
        deanon_map[key]   = result.deanon_results
        per_seed_map[key] = result.per_seed_metrics

        sweeps[key] = compute_threshold_sweep(
            result.deanon_results,
            n_thresholds=global_args.sweep_thresholds
        )

    elapsed = time.perf_counter() - t0
    logger.info("─" * 60)
    logger.info(
        "Analysis complete in %.1f s - %d/%d scenario(s) produced results.",
        elapsed, len(results), len(scenarios),
    )

    if not results:
        logger.warning("No scenarios produced results.")
        return 1

    _render_report(
        results, labels, sweeps, deanon_map, per_seed_map, output_dir, plot_set
    )
    logger.info("Output written to: %s", output_dir)
    return 0


def _split_results_by_group(
    all_results: List[DeanonymizationResult],
) -> Dict[str, List[DeanonymizationResult]]:
    """Split DeanonymizationResult objects by the group encoded in client_id.

    The group is embedded by GuardExitAttack._correlate_all_pairs as
    "..._grp:{name}_...".  Results whose client_id contains no such
    tag are placed in the "general" bucket.

    Args:
        all_results: Flat list of DeanonymizationResult objects.

    Returns:
        Dict mapping group name to list of results for that group.
    """
    groups: Dict[str, List[Any]] = {}
    for r in all_results:
        group = r.group if r.group else "general"
        groups.setdefault(group, []).append(r)
    return groups


def _register_attacks() -> None:
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