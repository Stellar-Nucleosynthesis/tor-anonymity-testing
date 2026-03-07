"""Step 2 - Run Shadow/tornettools network simulations.

Runs the tornettools pipeline for each seed:

    stage (once, shared) → generate → [inject custom clients] → simulate → parse

The stage step processes relay descriptors and produces deterministic output
for a given month.  It runs once into ``_stage/`` and is reused across all
seeds.  Generate, simulate, and parse run independently per seed.

Custom clients
--------------
The optional ``--client-groups FILE`` argument accepts a JSON file that
describes one or more *groups* of custom client hosts.  For each group a
fraction of the tornettools-generated ``torclient`` hosts is replaced with
hosts that run user-defined processes (a modified Tor binary, a probe tool,
a traffic generator, etc.) before Shadow starts.

A *client group manifest* is written next to ``shadow.yaml`` in every seed
directory.  The manifest records which Shadow hostnames belong to which group
so that the analysis step can filter to a specific group's circuits.

Client groups JSON format
-------------------------
The file must contain a JSON array.  Each element describes one group::

    [
      {
        "name":     "probe",
        "fraction": 0.05,
        "tor_binary": "/path/to/modified-tor",
        "torrc_append": {
          "CircuitBuildTimeout": "10",
          "NewCircuitPeriod": "15"
        },
        "replace_default_traffic": true,
        "processes": [
          {
            "path":       "/path/to/probe-tool",
            "args":       "--socks {socks_port} --log {data_dir}/probe.log",
            "start_time": "90s"
          }
        ]
      },
      {
        "name":     "baseline",
        "fraction": 0.05,
        "processes": []
      }
    ]

Placeholders available in ``args``:
    {hostname}      - Shadow host name, e.g. "torclient3"
    {socks_port}    - Tor SOCKS port for this host
    {control_port}  - Tor ControlPort for this host
    {data_dir}      - Absolute path to the host's template data directory
    {torrc_dir}     - Alias for {data_dir}

Usage
-----

  # Single seed, 1% scale, 600 s simulation
  python simulate.py --data-dir ./data --output-dir ./runs

  # Three seeds with two custom client groups
  python simulate.py \\
      --data-dir ./data --output-dir ./runs \\
      --num-seeds 3 --network-scale 0.01 --sim-time 1800 \\
      --client-groups ./my_groups.json

  # Specific seeds only
  python simulate.py \\
      --data-dir ./data --output-dir ./runs \\
      --seeds 0 2 5 --client-groups ./my_groups.json

  # Different month
  python simulate.py --data-dir ./data --month 2024-01 --output-dir ./runs
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.simulation.orchestrator import (
    CustomClientGroup,
    SimulationOrchestrator,
)

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("simulate")

def _dir_must_exist(value: str) -> Path:
    p = Path(value)
    if not p.is_dir():
        raise argparse.ArgumentTypeError(f"directory not found: {value!r}")
    return p


def _positive_float(value: str) -> float:
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value!r}")
    return f


def _file_must_exist(value: str) -> Path:
    p = Path(value)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {value!r}")
    return p

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simulate.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    grp_data = p.add_argument_group("Tor data (produced by download.py)")
    grp_data.add_argument(
        "--data-dir",
        metavar="DIR",
        type=_dir_must_exist,
        required=True,
        help="Root of the downloaded Tor data directory.",
    )
    grp_data.add_argument(
        "--month", "-m",
        metavar="YYYY-MM",
        default="2023-04",
        help="Month of relay data to use (default: 2023-04).",
    )

    grp_sim = p.add_argument_group("simulation parameters")
    grp_sim.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        type=Path,
        default=Path("runs"),
        help=(
            "Parent output directory. Subdirectories seed_0/, seed_1/, ... "
            "are created for each seed (default: ./runs)."
        ),
    )
    grp_sim.add_argument(
        "--num-seeds",
        metavar="N",
        type=int,
        default=1,
        help="Number of seeds to run as 0, 1, ... N-1 (default: 1).",
    )
    grp_sim.add_argument(
        "--seeds",
        nargs="+",
        metavar="N",
        type=int,
        default=None,
        help="Explicit seed integers (e.g. --seeds 0 2 7). Overrides --num-seeds.",
    )
    grp_sim.add_argument(
        "--network-scale",
        metavar="SCALE",
        type=_positive_float,
        default=0.01,
        help="Fraction of the public Tor network to simulate (default: 0.01 = 1%%).",
    )
    grp_sim.add_argument(
        "--sim-time",
        metavar="SECONDS",
        type=int,
        default=600,
        help="Shadow simulation stop-time in seconds (default: 600).",
    )

    grp_inject = p.add_argument_group(
        "custom client injection",
        description=(
            "Replace a fraction of generated torclient hosts with user-defined "
            "hosts before Shadow runs. See module docstring for the JSON format."
        ),
    )
    grp_inject.add_argument(
        "--client-groups",
        metavar="FILE", type=_file_must_exist, default=None,
        help=(
            "JSON file describing one or more custom client groups to inject. "
            "A custom_clients_manifest.json is written per seed for use by "
            "analyze.py --client-filter."
        ),
    )
    grp_inject.add_argument(
        "--injection-seed",
        metavar="N", type=int, default=42,
        help=(
            "RNG seed for reproducible host selection during injection "
            "(default: 42). Each simulation seed uses injection_seed + seed_idx."
        ),
    )

    grp_tools = p.add_argument_group("tool paths")
    grp_tools.add_argument(
        "--tornettools",
        metavar="CMD",
        default="tornettools",
        help="tornettools executable or full path (default: tornettools).",
    )
    grp_tools.add_argument(
        "--tor-binary",
        metavar="PATH",
        type=Path,
        default=None,
        help=(
            "Path to the tor binary. "
            "Auto-detected as <data-dir>/tor/src/app/tor when omitted."
        ),
    )
    grp_tools.add_argument(
        "--tor-gencert",
        metavar="PATH",
        type=Path,
        default=None,
        help=(
            "Path to tor-gencert. "
            "Auto-detected as <data-dir>/tor/src/app/tor-gencert when omitted."
        ),
    )

    grp_opt = p.add_argument_group("optional pipeline steps")
    grp_opt.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Run 'tornettools plot' after each simulation.",
    )
    grp_opt.add_argument(
        "--archive",
        action="store_true",
        default=False,
        help="Run 'tornettools archive' after each simulation (produces .tar.xz).",
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)

    data_dir   = args.data_dir
    month      = args.month
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    consensus_dir   = data_dir / f"consensuses-{month}"
    server_desc_dir = data_dir / f"server-descriptors-{month}"
    userstats_file  = data_dir / "userstats-relay-country.csv"
    tmodel_dir      = data_dir / "tmodel-ccs2018.github.io"
    onionperf_data  = data_dir / f"onionperf-{month}"
    bandwidth_data  = data_dir / f"bandwidth-{month}.csv"
    geoip_path      = data_dir / "tor" / "src" / "config" / "geoip"

    required = [
        (consensus_dir,   f"consensuses-{month}/"),
        (server_desc_dir, f"server-descriptors-{month}/"),
        (userstats_file,  "userstats-relay-country.csv"),
        (tmodel_dir,      "tmodel-ccs2018.github.io/"),
    ]
    missing = [f"  {p}  ({label})" for p, label in required if not p.exists()]
    if missing:
        logger.error(
            "Required data not found under %s:\n%s\n"
            "Run first: python download.py --month %s --output-dir %s",
            data_dir, "\n".join(missing), month, data_dir,
        )
        return 1

    tor_binary  = args.tor_binary
    tor_gencert = args.tor_gencert
    if tor_binary is None:
        candidate = data_dir / "tor" / "src" / "app" / "tor"
        if candidate.exists():
            tor_binary = candidate
            logger.info("Auto-detected tor binary: %s", tor_binary)
    if tor_gencert is None:
        candidate = data_dir / "tor" / "src" / "app" / "tor-gencert"
        if candidate.exists():
            tor_gencert = candidate

    client_groups: List[CustomClientGroup] = []
    if args.client_groups:
        try:
            client_groups = SimulationOrchestrator.load_client_groups(
                args.client_groups
            )
        except Exception as exc:
            logger.error(
                "Failed to load client groups from %s: %s",
                args.client_groups, exc,
            )
            return 1

        total_frac = sum(g.fraction for g in client_groups)
        logger.info(
            "Loaded %d client group(s) from %s  (total fraction: %.1f%%):  %s",
            len(client_groups),
            args.client_groups,
            total_frac * 100,
            ", ".join(f"{g.name}={g.fraction*100:.1f}%" for g in client_groups),
        )
        if total_frac > 1.0:
            logger.error(
                "Total client-group fraction (%.2f) exceeds 1.0 - aborting.",
                total_frac,
            )
            return 1

    seed_list: List[int] = (
        args.seeds if args.seeds is not None
        else list(range(args.num_seeds))
    )

    logger.info(
        "Simulating %d seed(s): %s | scale=%.4f | sim-time=%ds",
        len(seed_list), seed_list, args.network_scale, args.sim_time,
    )

    stage_workspace = output_dir / "_stage"
    orc_stage = SimulationOrchestrator(
        workspace=stage_workspace,
        tornettools_cmd=args.tornettools,
        tor_binary=str(tor_binary)  if tor_binary  else None,
        tor_gencert_binary=str(tor_gencert) if tor_gencert else None,
    )

    logger.info("─" * 60)
    logger.info("Stage step (shared across all seeds)...")
    logger.info("─" * 60)
    try:
        staged = orc_stage.stage_network_data(
            output_dir=stage_workspace,
            consensus_dir=consensus_dir,
            server_desc_dir=server_desc_dir,
            userstats_file=userstats_file,
            tmodel_dir=tmodel_dir,
            onionperf_data=onionperf_data if onionperf_data.exists() else None,
            bandwidth_data=bandwidth_data if bandwidth_data.exists() else None,
            geoip_path=geoip_path         if geoip_path.exists()     else None,
        )
    except Exception as exc:
        logger.error("Stage step failed: %s", exc, exc_info=True)
        return 1

    for key in ("relayinfo", "userinfo", "networkinfo"):
        if not staged.get(key):
            logger.error(
                "Stage produced incomplete output - missing '%s'. "
                "Check tornettools stage logs in %s.",
                key, stage_workspace,
            )
            return 1

    failed: List[int] = []

    for seed_idx in seed_list:
        seed_dir = output_dir / f"seed_{seed_idx}"
        logger.info("─" * 60)
        logger.info("Seed %d  →  %s", seed_idx, seed_dir)
        logger.info("─" * 60)

        orc = SimulationOrchestrator(
            workspace=seed_dir,
            tornettools_cmd=args.tornettools,
            tor_binary=str(tor_binary) if tor_binary else None,
            tor_gencert_binary=str(tor_gencert) if tor_gencert else None,
        )

        try:
            logger.info("[seed %d] generate...", seed_idx)
            network_dir = orc.generate_network(
                relayinfo_file=staged["relayinfo"],
                userinfo_file=staged["userinfo"],
                networkinfo_file=staged["networkinfo"],
                tmodel_dir=tmodel_dir,
                network_scale=args.network_scale,
                prefix="tornet",
                output_dir=seed_dir,
                geoip_path=geoip_path if geoip_path.exists() else None
            )

            if client_groups:
                logger.info(
                    "[seed %d] injecting %d client group(s)...",
                    seed_idx, len(client_groups),
                )
                manifest = orc.inject_custom_clients(
                    network_dir=network_dir,
                    groups=client_groups,
                    rng_seed=args.injection_seed + seed_idx,
                )
                for gname, hosts in manifest.groups.items():
                    display = hosts if len(hosts) <= 5 else hosts[:5] + ["..."]
                    logger.info(
                        "[seed %d]   group '%s': %d host(s) - %s",
                        seed_idx, gname, len(hosts), display,
                    )

            logger.info("[seed %d] simulate (%ds)...", seed_idx, args.sim_time)
            orc.run_simulation(
                network_dir,
                additional_args=["--args", f"--stop-time {args.sim_time}"],
            )

            logger.info("[seed %d] parse...", seed_idx)
            orc.parse_results(network_dir)

            if args.plot:
                logger.info("[seed %d] plot...", seed_idx)
                orc.plot_results(network_dir)

            if args.archive:
                logger.info("[seed %d] archive...", seed_idx)
                orc.archive_results(network_dir)

            logger.info("[seed %d] ✓ complete: %s", seed_idx, seed_dir)

        except Exception as exc:
            logger.error("[seed %d] failed: %s", seed_idx, exc, exc_info=True)
            failed.append(seed_idx)

    succeeded = [s for s in seed_list if s not in failed]
    logger.info("─" * 60)
    logger.info(
        "%d/%d seed(s) succeeded.  Output: %s",
        len(succeeded), len(seed_list), output_dir,
    )
    if succeeded:
        dirs_arg = " ".join(str(output_dir / f"seed_{s}") for s in succeeded)
        lines = [
            f"python analyze.py --sim-dirs {dirs_arg} --output-dir ./results \\"
        ]
        if client_groups:
            lines.append(
                "    guard-exit  "
                + "  ".join(f"# --client-filter group:{g.name}" for g in client_groups)
            )
        else:
            lines.append("    guard-exit")
        logger.info("Next step:\n  %s", "\n  ".join(lines))

    if failed:
        logger.warning("Failed seeds: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())