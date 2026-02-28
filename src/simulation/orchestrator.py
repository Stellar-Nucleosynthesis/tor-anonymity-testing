"""
Simulation Orchestrator

Manages the execution of Shadow simulations using the official tornettools workflow:
1. Stage: Process relay and user data
2. Generate: Create Shadow network configuration
3. Simulate: Run Shadow with TGen and OnionTrace
4. Parse: Process simulation results
5. Plot: Generate visualizations (optional)
6. Archive: Package results (optional)
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

def setup_logging(log_level: str = "INFO"):
    """Setup logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class SimulationOrchestrator:
    """
    Orchestrates Shadow simulation execution following official tornettools workflow.

    Workflow:
        1. tornettools stage - Process Tor network data
        2. tornettools generate - Create Shadow config
        3. tornettools simulate - Run Shadow simulation
        4. tornettools parse - Parse results
        5. tornettools plot - Generate graphs (optional)
    """

    def __init__(
            self,
            workspace: Path = Path("./workspace"),
            tornettools_cmd: str = "tornettools",
            tor_binary: Optional[str] = None,
            tor_gencert_binary: Optional[str] = None):
        """
        Initialize orchestrator.

        Args:
            workspace: Working directory for simulations
            tornettools_cmd: Command to run tornettools (e.g., 'tornettools' or path to script)
            tor_binary: Path to tor binary (required for generate step)
            tor_gencert_binary: Path to tor-gencert binary (required for generate step)
        """
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.tornettools_cmd = tornettools_cmd
        self.tor_binary = tor_binary
        self.tor_gencert_binary = tor_gencert_binary

        setup_logging("INFO")
        self.logger = logging.getLogger('SimulationOrchestrator')

        self.staged_data_dir = self.workspace / "staged_data"
        self.staged_data_dir.mkdir(exist_ok=True)

    def stage_network_data(
            self,
            consensus_dir: Path,
            server_desc_dir: Path,
            userstats_file: Path,
            tmodel_dir: Path,
            onionperf_data: Optional[Path] = None,
            bandwidth_data: Optional[Path] = None,
            geoip_path: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Run tornettools stage to process Tor network data.

        This processes real Tor consensus, server descriptors, and user statistics
        to create staged files for network generation.

        Args:
            consensus_dir: Directory with consensus files (e.g., consensuses-2023-04/)
            server_desc_dir: Directory with server descriptors
            userstats_file: CSV file with user statistics
            tmodel_dir: Directory with TModel traffic model data
            onionperf_data: Optional OnionPerf data directory
            bandwidth_data: Optional bandwidth CSV file
            geoip_path: Optional path to geoip file

        Returns:
            Dictionary with paths to staged files
        """
        self.logger.info("Running tornettools stage...")

        cmd = [
            self.tornettools_cmd,
            'stage',
            str(consensus_dir),
            str(server_desc_dir),
            str(userstats_file),
            str(tmodel_dir),
            '--prefix', str(self.staged_data_dir),
        ]

        if onionperf_data:
            cmd.extend(['--onionperf_data_path', str(onionperf_data)])

        if bandwidth_data:
            cmd.extend(['--bandwidth_data_path', str(bandwidth_data)])

        if geoip_path:
            cmd.extend(['--geoip_path', str(geoip_path)])

        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=1800
            )

            self.logger.info("tornettools stage completed successfully")
            if result.stdout:
                self.logger.debug(f"Output: {result.stdout}")

            staged_files : Dict[str, Path | None] = {
                'relayinfo': None,
                'userinfo': None,
                'networkinfo': None,
                'tor_metrics': None
            }

            for file in self.staged_data_dir.glob('relayinfo_*.json'):
                staged_files['relayinfo'] = file
            for file in self.staged_data_dir.glob('userinfo_*.json'):
                staged_files['userinfo'] = file
            for file in self.staged_data_dir.glob('networkinfo_*.gml'):
                staged_files['networkinfo'] = file
            for file in self.staged_data_dir.glob('tor_metrics_*.json'):
                staged_files['tor_metrics'] = file

            self.logger.info(f"Staged files: {staged_files}")
            return staged_files

        except subprocess.CalledProcessError as e:
            self.logger.error(f"tornettools stage failed: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            self.logger.error("tornettools stage timed out")
            raise

    def generate_network(
            self,
            relayinfo_file: Path,
            userinfo_file: Path,
            networkinfo_file: Path,
            tmodel_dir: Path,
            network_scale: float = 0.01,
            prefix: str = "tornet",
            output_dir: Optional[Path] = None,
            additional_args: Optional[List[str]] = None
    ) -> Path:
        """
        Run tornettools generate to create Shadow network configuration.

        Args:
            relayinfo_file: Staged relay info JSON file
            userinfo_file: Staged user info JSON file
            networkinfo_file: Staged network info GML file
            tmodel_dir: Directory with TModel traffic model data
            network_scale: Scale of network (0.01 = 1% of public Tor)
            prefix: Prefix for output directory
            output_dir: Base output directory (defaults to workspace)
            additional_args: Additional arguments to pass to tornettools generate

        Returns:
            Path to generated network directory
        """
        self.logger.info(f"Running tornettools generate (scale={network_scale})...")

        if output_dir is None:
            output_dir = self.workspace

        env = os.environ.copy()
        if self.tor_binary or self.tor_gencert_binary:
            tor_dir = Path(self.tor_binary).parent if self.tor_binary else None
            if tor_dir and str(tor_dir) not in env['PATH']:
                env['PATH'] = f"{tor_dir}:{env['PATH']}"
                self.logger.info(f"Added {tor_dir} to PATH")
        else:
            self.logger.warning("Tor binaries not specified - assuming they're in PATH")

        cmd = [
            self.tornettools_cmd,
            'generate',
            str(relayinfo_file),
            str(userinfo_file),
            str(networkinfo_file),
            str(tmodel_dir),
            '--network_scale', str(network_scale),
            '--prefix', str(output_dir / prefix),
            '--events', "CIRC,CIRC_BW"
        ]

        if additional_args:
            cmd.extend(additional_args)

        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600
            )

            self.logger.info("tornettools generate completed successfully")
            if result.stdout:
                self.logger.debug(f"Output: {result.stdout}")

            network_dir = output_dir / f"{prefix}-{network_scale}"

            if not network_dir.exists():
                network_dir = output_dir / prefix

            if not network_dir.exists():
                raise FileNotFoundError(f"Generated network directory not found at {network_dir}")

            self.logger.info(f"Generated network at: {network_dir}")
            return network_dir

        except subprocess.CalledProcessError as e:
            self.logger.error(f"tornettools generate failed: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            self.logger.error("tornettools generate timed out")
            raise

    def run_simulation(
            self,
            network_dir: Path,
            additional_args: Optional[List[str]] = None
    ) -> Path:
        """
        Run tornettools simulate to execute Shadow simulation.

        Args:
            network_dir: Generated network directory from tornettools generate
            additional_args: Additional arguments to pass to tornettools simulate

        Returns:
            Path to simulation output directory (same as network_dir)
        """
        self.logger.info(f"Running tornettools simulate on {network_dir}...")

        cmd = [
            self.tornettools_cmd,
            'simulate',
            str(network_dir)
        ]

        if additional_args:
            cmd.extend(additional_args)

        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200
            )

            self.logger.info("tornettools simulate completed successfully")
            if result.stdout:
                self.logger.debug(f"Output: {result.stdout}")

            return network_dir

        except subprocess.CalledProcessError as e:
            self.logger.error(f"tornettools simulate failed: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            self.logger.error("tornettools simulate timed out")
            raise

    def parse_results(
            self,
            network_dir: Path,
            additional_args: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run tornettools parse to process simulation results.

        Args:
            network_dir: Network directory with simulation results
            additional_args: Additional arguments to pass to tornettools parse

        Returns:
            Dictionary with parsed results information
        """
        self.logger.info(f"Running tornettools parse on {network_dir}...")

        cmd = [
            self.tornettools_cmd,
            'parse',
            str(network_dir)
        ]

        if additional_args:
            cmd.extend(additional_args)

        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=1800
            )

            self.logger.info("tornettools parse completed successfully")
            if result.stdout:
                self.logger.debug(f"Output: {result.stdout}")

            parsed_files = {
                'parsed_dir': network_dir / 'parsed',
                'tgen_stats': None,
                'oniontrace_stats': None
            }

            if (network_dir / 'parsed').exists():
                for file in (network_dir / 'parsed').glob('*.json'):
                    if 'tgen' in file.name:
                        parsed_files['tgen_stats'] = file
                    elif 'oniontrace' in file.name:
                        parsed_files['oniontrace_stats'] = file

            self.logger.info(f"Parsed files: {parsed_files}")
            return parsed_files

        except subprocess.CalledProcessError as e:
            self.logger.error(f"tornettools parse failed: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            self.logger.error("tornettools parse timed out")
            raise

    def plot_results(
            self,
            network_dir: Path,
            tor_metrics_path: Optional[Path] = None,
            prefix: str = "plots",
            additional_args: Optional[List[str]] = None
    ) -> Path:
        """
        Run tornettools plot to generate performance graphs.

        Args:
            network_dir: Network directory with parsed results
            tor_metrics_path: Optional path to Tor metrics JSON for comparison
            prefix: Prefix for plot output directory
            additional_args: Additional arguments to pass to tornettools plot

        Returns:
            Path to plots directory
        """
        self.logger.info(f"Running tornettools plot on {network_dir}...")

        plots_dir = network_dir / prefix

        cmd = [
            self.tornettools_cmd,
            'plot',
            str(network_dir),
            '--prefix', str(plots_dir)
        ]

        if tor_metrics_path:
            cmd.extend(['--tor_metrics_path', str(tor_metrics_path)])

        if additional_args:
            cmd.extend(additional_args)

        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600
            )

            self.logger.info("tornettools plot completed successfully")
            if result.stdout:
                self.logger.debug(f"Output: {result.stdout}")

            return plots_dir

        except subprocess.CalledProcessError as e:
            self.logger.error(f"tornettools plot failed: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            self.logger.error("tornettools plot timed out")
            raise

    def archive_results(
            self,
            network_dir: Path,
            additional_args: Optional[List[str]] = None
    ) -> Path:
        """
        Run tornettools archive to package results.

        Args:
            network_dir: Network directory to archive
            additional_args: Additional arguments to pass to tornettools archive

        Returns:
            Path to archive file
        """
        self.logger.info(f"Running tornettools archive on {network_dir}...")

        cmd = [
            self.tornettools_cmd,
            'archive',
            str(network_dir)
        ]

        if additional_args:
            cmd.extend(additional_args)

        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600
            )

            self.logger.info("tornettools archive completed successfully")
            if result.stdout:
                self.logger.debug(f"Output: {result.stdout}")

            archive_file = network_dir.parent / f"{network_dir.name}.tar.xz"
            return archive_file

        except subprocess.CalledProcessError as e:
            self.logger.error(f"tornettools archive failed: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            self.logger.error("tornettools archive timed out")
            raise

    def run_complete_pipeline(self,
                             consensus_dir: Path,
                             server_desc_dir: Path,
                             userstats_file: Path,
                             tmodel_dir: Path,
                             network_scale: float,
                             seed: int,
                             prefix: str = "tornet",
                             onionperf_data: Optional[Path] = None,
                             bandwidth_data: Optional[Path] = None,
                             geoip_path: Optional[Path] = None,
                             tor_metrics_path: Optional[Path] = None,
                             run_plot: bool = True,
                             run_archive: bool = False) -> Dict[str, Any]:
        """
        Run complete simulation pipeline including staging.

        This method runs the full tornettools workflow:
        1. Stage - Process raw Tor network data
        2. Generate - Create Shadow network configuration
        3. Simulate - Run Shadow simulation
        4. Parse - Extract statistics
        5. Plot - Generate graphs (optional)
        6. Archive - Package results (optional)

        Args:
            consensus_dir: Directory with consensus files (e.g., consensuses-2023-04/)
            server_desc_dir: Directory with server descriptors
            userstats_file: CSV file with user statistics
            tmodel_dir: TModel directory
            network_scale: Network scale (e.g., 0.01 for 1%)
            seed: Random seed
            prefix: Prefix for network directory
            onionperf_data: Optional OnionPerf data directory
            bandwidth_data: Optional bandwidth CSV file
            geoip_path: Optional path to geoip file
            tor_metrics_path: Optional Tor metrics for plotting
            run_plot: Whether to run plot step
            run_archive: Whether to run archive step

        Returns:
            Dictionary with all results and paths
        """
        self.logger.info(f"Running complete pipeline (scale={network_scale}, seed={seed})")
        self.logger.info("Pipeline: STAGE → GENERATE → SIMULATE → PARSE → PLOT")

        results : Dict[str, Any] = {
            'seed': seed,
            'network_scale': network_scale,
            'timestamp': datetime.now().isoformat(),
            'pipeline_steps': []
        }

        self.logger.info("Step 1/5: Staging network data...")
        staged_files = self.stage_network_data(
            consensus_dir=consensus_dir,
            server_desc_dir=server_desc_dir,
            userstats_file=userstats_file,
            tmodel_dir=tmodel_dir,
            onionperf_data=onionperf_data,
            bandwidth_data=bandwidth_data,
            geoip_path=geoip_path
        )
        results['staged_files'] = {k: str(v) if v else None for k, v in staged_files.items()}
        results['pipeline_steps'].append('stage')
        self.logger.info("✓ Staging completed")

        self.logger.info(f"Step 2/5: Generating network (scale={network_scale})...")
        network_dir = self.generate_network(
            relayinfo_file=staged_files['relayinfo'],
            userinfo_file=staged_files['userinfo'],
            networkinfo_file=staged_files['networkinfo'],
            tmodel_dir=tmodel_dir,
            network_scale=network_scale,
            prefix=prefix
        )
        results['network_dir'] = str(network_dir)
        results['pipeline_steps'].append('generate')
        self.logger.info("✓ Network generation completed")

        self.logger.info("Step 3/5: Running Shadow simulation...")
        self.run_simulation(network_dir)
        results['simulation_completed'] = True
        results['pipeline_steps'].append('simulate')
        self.logger.info("✓ Simulation completed")

        self.logger.info("Step 4/5: Parsing results...")
        parsed_files = self.parse_results(network_dir)
        results['parsed_files'] = {k: str(v) if v else None for k, v in parsed_files.items()}
        results['pipeline_steps'].append('parse')
        self.logger.info("✓ Parsing completed")

        if run_plot:
            self.logger.info("Step 5/5: Generating plots...")
            plots_dir = self.plot_results(network_dir, tor_metrics_path=tor_metrics_path)
            results['plots_dir'] = str(plots_dir)
            results['pipeline_steps'].append('plot')
            self.logger.info("✓ Plotting completed")
        else:
            self.logger.info("Step 5/5: Skipping plot generation")

        if run_archive:
            self.logger.info("Archiving results...")
            archive_file = self.archive_results(network_dir)
            results['archive_file'] = str(archive_file)
            results['pipeline_steps'].append('archive')
            self.logger.info("✓ Archiving completed")

        self.logger.info(f"✓ Complete pipeline finished: {network_dir}")
        self.logger.info(f"Completed steps: {' → '.join(results['pipeline_steps'])}")
        return results


def check_tornettools_installation():
    """Check if tornettools is installed and accessible"""
    try:
        result = subprocess.run(['tornettools', '-h'], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_tornettools_version():
    """Get tornettools version"""
    try:
        result = subprocess.run(['tornettools', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None