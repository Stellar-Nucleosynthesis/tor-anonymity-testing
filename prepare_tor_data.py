import argparse
import os
import subprocess
import logging
from pathlib import Path
import sys


def setup_logging(log_level: str = "INFO"):
    """Setup logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def download_file(url: str, output_path: Path, logger: logging.Logger):
    """Download a file using wget"""
    logger.info(f"Downloading {url}...")

    cmd = ['wget', '-O', str(output_path), url]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Downloaded to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path, logger: logging.Logger):
    """Extract tar archive"""
    logger.info(f"Extracting {archive_path}...")

    cmd = ['tar', '-xaf', str(archive_path), '-C', output_dir]

    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Extracted {archive_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def clone_tmodel(output_dir: Path, logger: logging.Logger):
    """Clone TModel repository"""
    tmodel_dir = output_dir / "tmodel-ccs2018.github.io"

    if tmodel_dir.exists():
        logger.info(f"TModel already exists at {tmodel_dir}")
        return True

    logger.info("Cloning TModel repository...")

    cmd = ['git', 'clone', 'https://github.com/tmodel-ccs2018/tmodel-ccs2018.github.io.git']

    try:
        subprocess.run(cmd, cwd=output_dir, check=True)
        logger.info(f"Cloned TModel to {tmodel_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone TModel: {e}")
        return False


def clone_tor_source(output_dir: Path, logger: logging.Logger):
    """Clone and build Tor source"""
    tor_dir = output_dir / "tor"

    if tor_dir.exists():
        logger.info(f"Tor source already exists at {tor_dir}")
        return True

    logger.info("Cloning Tor source...")

    cmd = ['git', 'clone', 'https://git.torproject.org/tor.git']
    try:
        subprocess.run(cmd, cwd=output_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone Tor: {e}")
        return False

    logger.info("Building Tor (this may take a while)...")

    build_commands = [
        ['./autogen.sh'],
        ['./configure', '--disable-asciidoc', '--disable-unittests', '--disable-manpage', '--disable-html-manual'],
        ['make', f'-j{os.cpu_count() or 1}']
    ]

    for cmd in build_commands:
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=tor_dir, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build Tor: {e}")
            return False

    logger.info(f"Tor built successfully at {tor_dir}")
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Prepare Tor network data for simulations"
    )
    parser.add_argument(
        '--month',
        type=str,
        default='2023-04',
        help='Month to download data for (format: YYYY-MM, default: 2023-04)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data'),
        help='Output directory for downloaded data (default: data/)'
    )
    parser.add_argument(
        '--skip-tor-build',
        action='store_true',
        help='Skip Tor source download and build'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger('prepare_tor_data')

    logger.info("=" * 70)
    logger.info("Tor Data Preparation")
    logger.info("=" * 70)
    logger.info(f"Month: {args.month}")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 70 + "\\n")

    args.output.mkdir(parents=True, exist_ok=True)

    collector_base = "https://collector.torproject.org/archive"
    metrics_base = "https://metrics.torproject.org"

    files_to_download = [
        {
            'url': f"{collector_base}/relay-descriptors/consensuses/consensuses-{args.month}.tar.xz",
            'filename': f'consensuses-{args.month}.tar.xz',
            'extract': True
        },
        {
            'url': f"{collector_base}/relay-descriptors/server-descriptors/server-descriptors-{args.month}.tar.xz",
            'filename': f'server-descriptors-{args.month}.tar.xz',
            'extract': True
        },
        {
            'url': f"{collector_base}/onionperf/onionperf-{args.month}.tar.xz",
            'filename': f'onionperf-{args.month}.tar.xz',
            'extract': True
        },
        {
            'url': f"{metrics_base}/userstats-relay-country.csv",
            'filename': 'userstats-relay-country.csv',
            'extract': False
        },
        {
            'url': f"{metrics_base}/bandwidth.csv?start={args.month}-01&end={args.month}-30",
            'filename': f'bandwidth-{args.month}.csv',
            'extract': False
        }
    ]

    logger.info("Step 1: Downloading Tor network data...")
    for file_info in files_to_download:
        output_path = args.output / file_info['filename']

        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
        else:
            if not download_file(file_info['url'], output_path, logger):
                logger.error(f"Failed to download {file_info['filename']}")
                return 1

        if file_info['extract'] and output_path.suffix == '.xz':
            extracted_dir = args.output / output_path.stem.replace('.tar', '')
            if not extracted_dir.exists():
                if not extract_archive(output_path, args.output, logger):
                    logger.error(f"Failed to extract {file_info['filename']}")
                    return 1
            else:
                logger.info(f"Already extracted: {extracted_dir}")

    logger.info("\\nStep 2: Getting TModel traffic model...")
    if not clone_tmodel(args.output, logger):
        logger.error("Failed to get TModel")
        return 1

    if not args.skip_tor_build:
        logger.info("\\nStep 3: Getting Tor source...")
        if not clone_tor_source(args.output, logger):
            logger.error("Failed to get Tor source")
            return 1

        tor_bin_path = args.output / "tor" / "src" / "core" / "or"
        tor_app_path = args.output / "tor" / "src" / "app"
        tor_tools_path = args.output / "tor" / "src" / "tools"

        logger.info("\\n" + "=" * 70)
        logger.info("IMPORTANT: Add Tor binaries to your PATH")
        logger.info("=" * 70)
        logger.info("Run this command:")
        logger.info(f"export PATH=$PATH:{tor_bin_path}:{tor_app_path}:{tor_tools_path}")
        logger.info("=" * 70)

    logger.info("\\n" + "=" * 70)
    logger.info("Data preparation complete!")
    logger.info("=" * 70)
    logger.info("\\nNext steps:")
    logger.info("1. Add Tor binaries to PATH (see above)")
    logger.info("2. Run tornettools stage with your data:")
    logger.info(f"   tornettools stage \\\\")
    logger.info(f"       {args.output}/consensuses-{args.month} \\\\")
    logger.info(f"       {args.output}/server-descriptors-{args.month} \\\\")
    logger.info(f"       {args.output}/userstats-relay-country.csv \\\\")
    logger.info(f"       {args.output}/tmodel-ccs2018.github.io \\\\")
    logger.info(f"       --onionperf_data_path {args.output}/onionperf-{args.month} \\\\")
    logger.info(f"       --bandwidth_data_path {args.output}/bandwidth-{args.month}.csv")
    logger.info("\\n" + "=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
