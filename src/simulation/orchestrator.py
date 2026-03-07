import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass
class ClientProcess:
    """One process entry to be added to a Shadow host.

    Args:
        path: Absolute path to the executable.
        args: Argument string. The following placeholders are substituted
            per-host at injection time:
              ``{hostname}``      - Shadow host name, e.g. ``"torclient3"``.
              ``{socks_port}``    - SOCKS port parsed from the host's tor args.
              ``{control_port}``  - ControlPort parsed from the host's tor args.
              ``{data_dir}``      - Absolute path to the host's data directory
                                    inside ``shadow.data.template/``.
              ``{torrc_dir}``     - Same as ``{data_dir}`` (convenience alias).
        start_time: Shadow virtual start time, e.g. ``"300s"`` or ``300``.
        environment: Extra environment variables for this process.
    """
    path: str
    args: str = ""
    start_time: Any = "300"
    environment: Dict[str, str] = field(default_factory=dict)

    def to_shadow_entry(self, substitutions: Dict[str, Any]) -> Dict[str, Any]:
        """Return a Shadow YAML process dict with placeholders filled in."""
        entry: Dict[str, Any] = {
            "path": self.path,
            "args": self.args.format(**substitutions),
            "start_time": self.start_time,
        }
        if self.environment:
            entry["environment"] = dict(self.environment)
        return entry


@dataclass
class CustomClientGroup:
    """Configuration for one group of custom Shadow client hosts.

    A group replaces a ``fraction`` of the tornettools-generated ``torclient``
    hosts with hosts that run ``processes`` instead of (or in addition to) the
    default tgen traffic generator.

    Args:
        name: Short identifier used in the manifest and as a filter key during
            analysis (e.g. ``"probe"``, ``"modified-tor"``).
        fraction: Fraction of all torclient hosts to replace (0, 1].
        processes: Replacement process list. Tor and OnionTrace from the
            original host are preserved; these entries replace tgen/tgenrs.
        tor_binary: If set, the host's tor process path is replaced with this
            binary.
        torrc_append: Key-value pairs appended to the host's torrc as
            ``Key Value`` lines. Applied after the tornettools-generated torrc
            so they override any earlier setting.
        replace_default_traffic: When ``True`` (default) the tgen/tgenrs
            process is removed and replaced by ``processes``. When ``False``
            ``processes`` are appended alongside the existing traffic generator.
        count: Exact number of hosts to replace. Takes precedence over
            ``fraction`` when set. Useful when the total number of clients is
            small and rounding produces unexpected results.
    """
    name: str
    fraction: float
    processes: List[ClientProcess] = field(default_factory=list)
    tor_binary: Optional[str] = None
    torrc_append: Dict[str, str] = field(default_factory=dict)
    replace_default_traffic: bool = True
    count: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError(
                f"CustomClientGroup '{self.name}': fraction must be in (0, 1], "
                f"got {self.fraction}"
            )
        if not self.name.isidentifier():
            raise ValueError(
                f"Group name '{self.name}' is not a valid identifier. "
                "Use letters, digits, and underscores only."
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CustomClientGroup":
        """Construct a ``CustomClientGroup`` from a plain dictionary.

        The ``processes`` key should be a list of dicts with ``path``, ``args``,
        ``start_time``, and optionally ``environment``.
        """
        processes = [
            ClientProcess(**p) for p in d.pop("processes", [])
        ]
        return cls(processes=processes, **d)

    def compute_count(self, total_clients: int) -> int:
        """Return the number of hosts this group should replace."""
        if self.count is not None:
            return min(self.count, total_clients)
        return max(1, round(self.fraction * total_clients))


MANIFEST_FILENAME = "custom_clients_manifest.json"

@dataclass
class CustomClientsManifest:
    """Maps group names to the Shadow hostnames they control.

    Written to ``<network_dir>/custom_clients_manifest.json`` after injection.
    ``analyze.py`` loads this file to resolve ``--client-filter group:<name>``
    arguments into concrete hostname sets.

    Attributes:
        groups: ``{group_name: [hostname, ...]}``.
        injection_time: ISO timestamp of when the manifest was written.
        total_clients: Total number of torclient hosts in the simulation.
    """
    groups: Dict[str, List[str]]
    injection_time: str
    total_clients: int

    def save(
            self, network_dir: Path) -> Path:
        path = network_dir / MANIFEST_FILENAME
        path.write_text(json.dumps(self.__dict__, indent=2))
        return path

    @classmethod
    def load(cls, network_dir: Path) -> Optional["CustomClientsManifest"]:
        path = network_dir / MANIFEST_FILENAME
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(**data)

    def resolve_filter(self, filter_spec: str) -> Optional[List[str]]:
        """Resolve a filter spec to a list of hostnames.

        Supported formats:
          ``group:probe``                    - all hosts in the named group.
          ``host:torclient3,torclient7``     - explicit comma-separated hostnames.

        Returns ``None`` when the spec format is unrecognized.
        """
        if filter_spec.startswith("group:"):
            name = filter_spec[len("group:"):]
            return self.groups.get(name)
        if filter_spec.startswith("host:"):
            return [h.strip() for h in filter_spec[len("host:"):].split(",")]
        return None


class SimulationOrchestrator:
    """Orchestrates the Shadow simulation pipeline.

    Workflow::

        stage_network_data()
        generate_network()
        inject_custom_clients()   ← optional
        run_simulation()
        parse_results()
        plot_results()            ← optional
        archive_results()         ← optional
    """

    def __init__(
            self,
            workspace: Path = Path("./workspace"),
            tornettools_cmd: str = "tornettools",
            tor_binary: Optional[str] = None,
            tor_gencert_binary: Optional[str] = None
    ):
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
        self.logger = logging.getLogger("SimulationOrchestrator")

    def stage_network_data(
            self,
            output_dir: Path,
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
            output_dir: Output directory for staged files
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
            '--prefix', str(output_dir),
        ]
        if onionperf_data:
            cmd.extend(['--onionperf_data_path', str(onionperf_data)])

        if bandwidth_data:
            cmd.extend(['--bandwidth_data_path', str(bandwidth_data)])

        if geoip_path:
            cmd.extend(['--geoip_path', str(geoip_path)])

        self._run_cmd(cmd, timeout=1800, step="stage")

        staged: Dict[str, Optional[Path]] = dict()
        for f in output_dir.glob("relayinfo_*.json"):
            staged["relayinfo"] = f
        for f in output_dir.glob("userinfo_*.json"):
            staged["userinfo"] = f
        for f in output_dir.glob("networkinfo_*.gml"):
            staged["networkinfo"] = f
        for f in output_dir.glob("tor_metrics_*.json"):
            staged["tor_metrics"] = f

        self.logger.info("Staged files: %s", {k: str(v) for k, v in staged.items()})
        return staged

    def generate_network(
            self,
            relayinfo_file: Path,
            userinfo_file: Path,
            networkinfo_file: Path,
            tmodel_dir: Path,
            network_scale: float = 0.01,
            prefix: str = "tornet",
            output_dir: Optional[Path] = None,
            geoip_path: Optional[Path] = None,
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
            geoip_path: Optional path to geoip file
            additional_args: Additional arguments to pass to tornettools generate

        Returns:
            Path to generated network directory
        """
        self.logger.info(f"Running tornettools generate (scale={network_scale:.2%})...")

        if output_dir is None:
            output_dir = self.workspace

        network_dir = output_dir / prefix

        cmd = [
            self.tornettools_cmd,
            'generate',
            str(relayinfo_file),
            str(userinfo_file),
            str(networkinfo_file),
            str(tmodel_dir),
            '--network_scale', str(network_scale),
            '--prefix', str(network_dir),
            '--events', "CIRC,CIRC_BW",
        ]
        if self.tor_binary:
            cmd.extend(['--tor', str(self.tor_binary)])
        if self.tor_gencert_binary:
            cmd.extend(['--torgencert', str(self.tor_gencert_binary)])
        if geoip_path:
            cmd.extend(['--geoip_path', str(geoip_path)])
        if additional_args:
            cmd.extend(additional_args)

        self._run_cmd(cmd, timeout=3600, step="generate")

        tor_abs_path = os.path.abspath(self.tor_binary)
        if network_dir.exists():
            self.logger.info("Generated network: %s", network_dir)
            if self.tor_binary:
                shadow_config = network_dir / "shadow.config.yaml"
                if shadow_config.exists() and self.tor_binary:
                    self.logger.info("Patching Tor binary in shadow config")
                    self._patch_shadow_tor_path(shadow_config, Path(tor_abs_path))
                else:
                    self.logger.warning(
                        "shadow.config.yaml not found at %s",
                        shadow_config
                    )
            return network_dir

        raise FileNotFoundError(
            f"Generated network directory not found under {output_dir}."
        )


    def inject_custom_clients(
            self,
            network_dir: Path,
            groups: List[CustomClientGroup],
            rng_seed: int = 42
    ) -> CustomClientsManifest:
        """Patch ``shadow.yaml`` to replace a fraction of client hosts.

        For each group the method:

        1. Identifies all ``torclient*`` hosts generated by tornettools.
        2. Randomly selects ``group.compute_count(n_clients)`` of them.
        3. Replaces their tgen/tgenrs process with ``group.processes``.
        4. Optionally replaces the tor binary path.
        5. Optionally appends lines to the host's torrc.
        6. Writes a ``custom_clients_manifest.json`` next to ``shadow.yaml``.

        Args:
            network_dir: The directory produced by ``generate_network`` that
                contains ``shadow.yaml``.
            groups: List of ``CustomClientGroup`` objects. Groups must
                have unique names. Total replaced hosts must not exceed the
                number of client hosts in the simulation.
            rng_seed: Seed for reproducible host selection.

        Returns:
            The written ``CustomClientsManifest``.

        Raises:
            FileNotFoundError: When ``shadow.yaml`` is not found.
            ValueError: When group names are not unique or the total count
                exceeds the number of available client hosts.
        """
        shadow_yaml = self._find_shadow_yaml(network_dir)
        self.logger.info("Injecting custom clients into %s", shadow_yaml)

        with shadow_yaml.open() as fh:
            cfg = yaml.safe_load(fh)

        hosts: Dict[str, Any] = cfg.get("hosts", {})
        client_hosts = [h for h, v in hosts.items() if self._is_client_host(h, v)]
        n_clients = len(client_hosts)
        self.logger.info("Found %d client host(s) in shadow.yaml.", n_clients)

        if n_clients == 0:
            raise ValueError(
                "No client hosts found in shadow.yaml. "
                "Expected hosts whose name matches 'torclient*' or that "
                "contain a tgen/tgenrs process."
            )

        names = [g.name for g in groups]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate group names: {names}")

        total_replacing = sum(g.compute_count(n_clients) for g in groups)
        if total_replacing > n_clients:
            raise ValueError(
                f"Groups request {total_replacing} host replacements but only "
                f"{n_clients} client hosts are available."
            )

        rng = np.random.default_rng(seed=rng_seed)
        available = list(client_hosts)
        rng.shuffle(available)

        manifest_groups: Dict[str, List[str]] = {}

        for group in groups:
            n = group.compute_count(n_clients)
            sel = available[:n]
            available = available[n:]

            self.logger.info(
                "Group '%s': replacing %d host(s): %s",
                group.name, len(sel), sel,
            )

            for hostname in sel:
                host_entry = hosts[hostname]
                data_dir = self._host_data_dir(network_dir, hostname)
                ports = self._parse_tor_ports(host_entry)
                substitutions = {
                    "hostname": hostname,
                    "socks_port": ports["socks_port"] or 9000,
                    "control_port": ports["control_port"] or 9001,
                    "data_dir": str(data_dir),
                    "torrc_dir": str(data_dir),
                }

                self._apply_group_to_host(
                    host_entry, group, substitutions, data_dir
                )

            manifest_groups[group.name] = sel

        cfg["hosts"] = hosts
        with shadow_yaml.open("w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

        manifest = CustomClientsManifest(
            groups=manifest_groups,
            injection_time=datetime.now().isoformat(),
            total_clients=n_clients,
        )
        manifest_path = manifest.save(network_dir)
        self.logger.info(
            "Manifest written to %s (groups: %s)",
            manifest_path,
            {k: len(v) for k, v in manifest_groups.items()},
        )
        return manifest

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
        self.logger.info(f"Running tornettools simulate on {network_dir}")

        cmd = [
            self.tornettools_cmd,
            'simulate',
            str(network_dir)
        ]
        if additional_args:
            cmd.extend(additional_args)

        self._run_cmd(cmd, timeout=7200, step="simulate")
        return network_dir


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
        self._run_cmd(cmd, timeout=1800, step="parse")

        parsed: Dict[str, Any] = {
            "parsed_dir": network_dir / "parsed"
        }
        if (network_dir / "parsed").exists():
            for f in (network_dir / "parsed").glob("*.json"):
                if "tgen" in f.name:
                    parsed["tgen_stats"] = f
                elif "oniontrace" in f.name:
                    parsed["oniontrace_stats"] = f
        return parsed

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

        self._run_cmd(cmd, timeout=600, step="plot")
        return plots_dir


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

        self._run_cmd(cmd, timeout=600, step="archive")
        return network_dir.parent / f"{network_dir.name}.tar.xz"

    @staticmethod
    def load_client_groups(path: Path) -> List[CustomClientGroup]:
        """Load a list of ``CustomClientGroup`` objects from a JSON file.

        The file must contain a JSON array. Each element is an object with
        the fields of ``CustomClientGroup``; the ``processes`` key is a list of
        objects matching ``ClientProcess``.
        """
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, list):
            raise ValueError(f"{path}: expected a JSON array, got {type(raw).__name__}")
        return [CustomClientGroup.from_dict(entry) for entry in raw]

    def _run_cmd(
            self,
            cmd: List[str],
            timeout: int,
            step: str,
            env: Optional[Dict[str, str]] = None
    ) -> None:
        self.logger.info("Executing: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if result.stdout:
                self.logger.debug("%s stdout: %s", step, result.stdout[:2000])
        except subprocess.CalledProcessError as exc:
            self.logger.error("%s failed:\n%s", step, exc.stderr)
            raise
        except subprocess.TimeoutExpired:
            self.logger.error("%s timed out after %ds", step, timeout)
            raise

    @staticmethod
    def _find_shadow_yaml(network_dir: Path) -> Path:
        for candidate in (
            network_dir / "shadow.yaml",
            network_dir / "shadow.config.yaml",
        ):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"shadow.yaml not found under {network_dir}. "
            "Run generate_network first."
        )

    @staticmethod
    def _is_client_host(hostname: str, host_entry: Dict[str, Any]) -> bool:
        """Return True for hosts that are traffic-generating clients."""
        if hostname.startswith("torclient"):
            return True
        for proc in host_entry.get("processes", []):
            path = str(proc.get("path", ""))
            if "tgen" in path or "tgenrs" in path:
                return True
        return False

    @staticmethod
    def _parse_tor_ports(host_entry: Dict[str, Any]) -> Dict[str, Optional[int]]:
        """Extract SocksPort and ControlPort from the host's tor process args."""
        for proc in host_entry.get("processes", []):
            path = str(proc.get("path", ""))
            if "tor" in path and "oniontrace" not in path:
                args = str(proc.get("args", ""))
                socks = re.search(r"--SocksPort\s+(\d+)", args)
                ctrl = re.search(r"--ControlPort\s+(\d+)", args)
                return {
                    "socks_port": int(socks.group(1)) if socks else None,
                    "control_port": int(ctrl.group(1)) if ctrl else None,
                }
        return {"socks_port": None, "control_port": None}

    @staticmethod
    def _host_data_dir(network_dir: Path, hostname: str) -> Path:
        """Return the template data directory for a Shadow host."""
        for candidate in (
            network_dir / "shadow.data.template" / "hosts" / hostname,
            network_dir / hostname,
        ):
            if candidate.exists():
                return candidate
        return network_dir / "shadow.data.template" / "hosts" / hostname

    @staticmethod
    def _patch_shadow_tor_path(shadow_config: Path, tor_binary: Path):
        tor_binary = str(tor_binary.resolve())

        with open(shadow_config, "r") as f:
            data = yaml.safe_load(f)

        def patch(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "path" and isinstance(v, str) and v.endswith("/tor"):
                        obj[k] = tor_binary
                    else:
                        patch(v)
            elif isinstance(obj, list):
                for item in obj:
                    patch(item)

        patch(data)

        with open(shadow_config, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def _apply_group_to_host(
            self,
            host_entry: Dict[str, Any],
            group: CustomClientGroup,
            substitutions: Dict[str, Any],
            data_dir: Path,
    ) -> None:
        """Modify *host_entry* in-place according to *group* settings."""
        processes: List[Dict[str, Any]] = host_entry.get("processes", [])

        if group.tor_binary:
            for proc in processes:
                path = str(proc.get("path", ""))
                if "tor" in path and "oniontrace" not in path and "tgen" not in path:
                    self.logger.debug(
                        " %s: replacing tor binary %s → %s",
                        substitutions["hostname"], proc["path"], group.tor_binary,
                    )
                    proc["path"] = group.tor_binary
                    break

        if group.torrc_append:
            self._append_to_torrc(data_dir, group.torrc_append, group.name)

        if group.replace_default_traffic:
            processes = [
                p for p in processes
                if "tgen" not in str(p.get("path", ""))
                and "tgenrs" not in str(p.get("path", ""))
            ]

        for client_proc in group.processes:
            processes.append(client_proc.to_shadow_entry(substitutions))

        host_entry["processes"] = processes

    def _append_to_torrc(
        self,
        data_dir: Path,
        options: Dict[str, str],
        group_name: str,
    ) -> None:
        """Append *options* to the host's torrc as ``%include`` of a side-file."""
        data_dir.mkdir(parents=True, exist_ok=True)

        extra_path = data_dir / f"torrc.{group_name}.extra"
        lines = [f"{k} {v}" for k, v in options.items()]
        extra_path.write_text("\n".join(lines) + "\n")

        torrc_path = data_dir / "torrc"
        if torrc_path.exists():
            existing = torrc_path.read_text()
            include_line = f"%include {extra_path.name}"
            if include_line not in existing:
                torrc_path.write_text(existing.rstrip("\n") + f"\n{include_line}\n")
        else:
            self.logger.debug(
                "torrc not found at %s - extra options file written but not included.",
                torrc_path,
            )