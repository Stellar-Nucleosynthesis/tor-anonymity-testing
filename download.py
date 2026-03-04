"""Step 1 — Download Tor relay data from CollecTor and Tor Metrics.

Downloads consensus snapshots, server descriptors, user statistics,
OnionPerf data, and the bandwidth CSV for the chosen month, then
optionally clones and builds the Tor source tree required by
tornettools generate.

All files are placed under --output-dir.  Already-present files
and already-extracted archives are skipped automatically.

Usage
-----

  # Download April 2023 data, build Tor from source
  python download.py --month 2023-04 --output-dir ./data

  # Tor binaries already installed; skip the build
  python download.py --month 2023-04 --output-dir ./data --skip-tor-build

  # Try a different month without re-downloading unchanged files
  python download.py --month 2024-01 --output-dir ./data
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("download")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--month", "-m",
        metavar="YYYY-MM",
        default="2023-04",
        help="Month of relay data to download (default: 2023-04).",
    )
    p.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        type=Path,
        default=Path("data"),
        help=(
            "Root directory for all downloaded and extracted files. "
            "Created automatically if absent (default: ./data)."
        ),
    )
    p.add_argument(
        "--skip-tor-build",
        action="store_true",
        default=False,
        help=(
            "Skip cloning and building the Tor source tree. "
            "Use when tor / tor-gencert are already on PATH."
        ),
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

    try:
        from fetching import prepare_tor_data as ptd
    except ImportError as exc:
        logger.error(
            "Cannot import prepare_tor_data.py — "
            "make sure it is in the same directory.\n  %s", exc
        )
        return 1

    ptd.setup_logging(args.log_level)
    log = logging.getLogger("prepare_tor_data")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    month: str = args.month

    log.info("=" * 60)
    log.info("Tor Data Download   month=%s  →  %s", month, output_dir)
    log.info("=" * 60)

    collector_base = "https://collector.torproject.org/archive"
    metrics_base   = "https://metrics.torproject.org"

    downloads: List[Tuple[str, str, bool]] = [
        (
            f"{collector_base}/relay-descriptors/consensuses/"
            f"consensuses-{month}.tar.xz",
            f"consensuses-{month}.tar.xz",
            True,
        ),
        (
            f"{collector_base}/relay-descriptors/server-descriptors/"
            f"server-descriptors-{month}.tar.xz",
            f"server-descriptors-{month}.tar.xz",
            True,
        ),
        (
            f"{collector_base}/onionperf/onionperf-{month}.tar.xz",
            f"onionperf-{month}.tar.xz",
            True,
        ),
        (
            f"{metrics_base}/userstats-relay-country.csv",
            "userstats-relay-country.csv",
            False,
        ),
        (
            f"{metrics_base}/bandwidth.csv?start={month}-01&end={month}-30",
            f"bandwidth-{month}.csv",
            False,
        ),
    ]

    log.info("Step 1/3: Downloading relay data...")
    for url, filename, do_extract in downloads:
        dest = output_dir / filename
        if dest.exists():
            log.info("  already present: %s", dest.name)
        else:
            if not ptd.download_file(url, dest, log):
                log.error("Download failed (%s) — aborting.", filename)
                return 1

        if do_extract:
            extracted = output_dir / dest.name.replace(".tar.xz", "")
            if not extracted.exists():
                if not ptd.extract_archive(dest, output_dir, log):
                    log.error("Extraction failed (%s) — aborting.", filename)
                    return 1
            else:
                log.info("  already extracted: %s", extracted.name)

    log.info("Step 2/3: Cloning TModel traffic model...")
    if not ptd.clone_tmodel(output_dir, log):
        log.error("TModel clone failed — aborting.")
        return 1

    if not args.skip_tor_build:
        log.info("Step 3/3: Building Tor from source (may take several minutes)...")
        if not ptd.clone_tor_source(output_dir, log):
            log.error("Tor build failed — aborting.")
            return 1
    else:
        log.info("Step 3/3: Skipping Tor build (--skip-tor-build).")

    log.info("=" * 60)
    log.info("Download complete.  Data directory: %s", output_dir)
    log.info(
        "Next step:\n  python simulate.py --data-dir %s --output-dir ./runs",
        output_dir,
    )
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())