"""
Typed Configuration Classes

Provides type-safe access to configuration values, replacing
dictionary-based access with proper dataclasses. This enables:
- IDE autocompletion and type checking
- Validation at config load time
- Clear documentation of available settings
"""

import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class APIEndpoint:
    """Configuration for a single API endpoint."""
    url: str
    name: str = "default"


@dataclass
class APIConfig:
    """API connection configuration."""
    endpoint: str = "http://localhost:8000/api/v1"
    endpoints: List[APIEndpoint] = field(default_factory=list)
    retry_attempts: int = 3
    timeout: int = 30

    def get_endpoints(self) -> List[APIEndpoint]:
        """
        Get list of endpoints to submit to.

        Returns endpoints list if configured, otherwise wraps
        the single endpoint in a list.
        """
        if self.endpoints:
            return self.endpoints
        return [APIEndpoint(url=self.endpoint, name="default")]


@dataclass
class ClientConfig:
    """Client identification configuration."""
    username: str = "default_user"
    cpu_name: str = "default_machine"


@dataclass
class ExecutionConfig:
    """Execution environment configuration."""
    output_dir: str = "data/outputs"
    residue_dir: str = "data/residues"
    failed_uploads_dir: str = "data/failed_uploads"
    preserve_failed_uploads: bool = True
    save_raw_output: bool = True
    queue_dir: str = "data/queue"  # Persistent submission queue for failed API operations

    def ensure_dirs_exist(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.residue_dir).mkdir(parents=True, exist_ok=True)
        Path(self.failed_uploads_dir).mkdir(parents=True, exist_ok=True)
        Path(self.queue_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    file: str = "data/logs/ecm_client.log"
    level: str = "INFO"
    log_factors_found: bool = True

    def ensure_log_dir_exists(self) -> None:
        """Create log directory if it doesn't exist."""
        Path(self.file).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class FactorDBConfig:
    """FactorDB integration configuration (aliquot_wrapper --factordb)."""
    cookie: Optional[str] = None  # fdbuser cookie value for authenticated requests


@dataclass
class AliquotTrackerConfig:
    """Aliquot tracker integration configuration (aliquot_wrapper --tracker)."""
    url: Optional[str] = None  # Tracker base URL, e.g. https://aliquot.example.com
    api_key: Optional[str] = None  # X-Api-Key for verified attribution (optional)
    submitter: Optional[str] = None  # Anonymous handle when no api_key (default: client.username)


@dataclass
class GPUConfig:
    """Nested GPU-specific tuning under programs.gmp_ecm.gpu.

    Distinct from the flat `gpu_enabled` / `gpu_device` / `gpu_curves` fields
    on GMPECMConfig (which were added before nesting became necessary). New
    GPU-only knobs go here.
    """
    curves_per_batch: int = 1000  # Curves to request per GPU batch in stage1-only mode


@dataclass
class GMPECMConfig:
    """GMP-ECM program configuration."""
    path: str = "ecm"
    default_b1: int = 110000000
    default_b2: Optional[int] = None
    default_curves: int = 1
    early_termination: bool = True
    gpu_enabled: bool = False
    gpu_device: int = 0
    gpu_curves: Optional[int] = None
    workers: int = 8  # Parallel workers (multiprocess ECM, stage2 threads)
    stage2_max_b1: Optional[int] = None  # Max B1 for stage 2 residues (RAM limit)
    max_batch: Optional[int] = None  # Max curves per GPU batch in two-stage t-level mode
    pm1_b1: int = 2900000000
    pm1_b2: int = 1000000000000000
    pp1_b1: int = 110000000
    pp1_b2: int = 500000000000
    gpu: GPUConfig = field(default_factory=GPUConfig)


@dataclass
class YAFUConfig:
    """YAFU program configuration."""
    path: str = "yafu"
    threads: int = 8


@dataclass
class CADOConfig:
    """CADO-NFS program configuration."""
    path: str = "~/cado-nfs/cado-nfs.py"
    threads: int = 4
    working_dir: str = "~/cado-nfs"


@dataclass
class TLevelBinaryConfig:
    """T-level calculator configuration."""
    path: str = "bin/t-level.exe" if sys.platform == 'win32' else "bin/t-level"


@dataclass
class ProgramsConfig:
    """All program configurations."""
    gmp_ecm: GMPECMConfig = field(default_factory=GMPECMConfig)
    yafu: YAFUConfig = field(default_factory=YAFUConfig)
    cado_nfs: CADOConfig = field(default_factory=CADOConfig)
    t_level: TLevelBinaryConfig = field(default_factory=TLevelBinaryConfig)


@dataclass
class AppConfig:
    """
    Root configuration object containing all settings.

    This is the main configuration class that aggregates all
    sub-configurations. Use TypedConfigLoader to create instances
    from YAML files.

    Usage:
        config = TypedConfigLoader().load("client.yaml")
        print(config.programs.gmp_ecm.path)
        print(config.api.endpoint)
    """
    api: APIConfig = field(default_factory=APIConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    programs: ProgramsConfig = field(default_factory=ProgramsConfig)
    factordb: FactorDBConfig = field(default_factory=FactorDBConfig)
    aliquot_tracker: AliquotTrackerConfig = field(default_factory=AliquotTrackerConfig)

    def ensure_dirs_exist(self) -> None:
        """Create all required directories."""
        self.execution.ensure_dirs_exist()
        self.logging.ensure_log_dir_exists()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config back to dictionary format.

        This is useful for backward compatibility with code
        that still expects dictionary access.
        """
        return {
            'api': {
                'endpoint': self.api.endpoint,
                'endpoints': [{'url': e.url, 'name': e.name} for e in self.api.endpoints],
                'retry_attempts': self.api.retry_attempts,
                'timeout': self.api.timeout,
            },
            'client': {
                'username': self.client.username,
                'cpu_name': self.client.cpu_name,
            },
            'execution': {
                'output_dir': self.execution.output_dir,
                'residue_dir': self.execution.residue_dir,
                'failed_uploads_dir': self.execution.failed_uploads_dir,
                'preserve_failed_uploads': self.execution.preserve_failed_uploads,
                'save_raw_output': self.execution.save_raw_output,
                'queue_dir': self.execution.queue_dir,
            },
            'logging': {
                'file': self.logging.file,
                'level': self.logging.level,
                'log_factors_found': self.logging.log_factors_found,
            },
            'programs': {
                'gmp_ecm': {
                    'path': self.programs.gmp_ecm.path,
                    'default_b1': self.programs.gmp_ecm.default_b1,
                    'default_b2': self.programs.gmp_ecm.default_b2,
                    'default_curves': self.programs.gmp_ecm.default_curves,
                    'early_termination': self.programs.gmp_ecm.early_termination,
                    'gpu_enabled': self.programs.gmp_ecm.gpu_enabled,
                    'gpu_device': self.programs.gmp_ecm.gpu_device,
                    'gpu_curves': self.programs.gmp_ecm.gpu_curves,
                    'workers': self.programs.gmp_ecm.workers,
                    'stage2_max_b1': self.programs.gmp_ecm.stage2_max_b1,
                    'max_batch': self.programs.gmp_ecm.max_batch,
                    'pm1_b1': self.programs.gmp_ecm.pm1_b1,
                    'pm1_b2': self.programs.gmp_ecm.pm1_b2,
                    'pp1_b1': self.programs.gmp_ecm.pp1_b1,
                    'pp1_b2': self.programs.gmp_ecm.pp1_b2,
                    'gpu': {
                        'curves_per_batch': self.programs.gmp_ecm.gpu.curves_per_batch,
                    },
                },
                'yafu': {
                    'path': self.programs.yafu.path,
                    'threads': self.programs.yafu.threads,
                },
                'cado_nfs': {
                    'path': self.programs.cado_nfs.path,
                    'threads': self.programs.cado_nfs.threads,
                    'working_dir': self.programs.cado_nfs.working_dir,
                },
                't_level': {
                    'path': self.programs.t_level.path,
                },
            },
            'factordb': {
                'cookie': self.factordb.cookie,
            },
            'aliquot_tracker': {
                'url': self.aliquot_tracker.url,
                'api_key': self.aliquot_tracker.api_key,
                'submitter': self.aliquot_tracker.submitter,
            },
        }


class TypedConfigLoader:
    """
    Load configuration from YAML into typed dataclasses.

    This class wraps ConfigManager to provide type-safe access
    to configuration values.

    Usage:
        loader = TypedConfigLoader()
        config = loader.load("client.yaml")
        print(config.programs.gmp_ecm.path)
    """

    def load(self, config_path: str) -> AppConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Typed AppConfig instance
        """
        from .config_manager import ConfigManager

        # Load raw config using existing ConfigManager
        manager = ConfigManager()
        raw_config = manager.load_config(config_path)

        return self._parse_config(raw_config)

    def _parse_config(self, raw: Dict[str, Any]) -> AppConfig:
        """Parse raw dictionary into typed config."""
        return AppConfig(
            api=self._parse_api(raw.get('api', {})),
            client=self._parse_client(raw.get('client', {})),
            execution=self._parse_execution(raw.get('execution', {})),
            logging=self._parse_logging(raw.get('logging', {})),
            programs=self._parse_programs(raw.get('programs', {})),
            factordb=self._parse_factordb(raw.get('factordb', {})),
            aliquot_tracker=self._parse_aliquot_tracker(raw.get('aliquot_tracker', {})),
        )

    def _parse_factordb(self, raw: Dict[str, Any]) -> FactorDBConfig:
        """Parse FactorDB configuration."""
        return FactorDBConfig(
            cookie=raw.get('cookie'),
        )

    def _parse_aliquot_tracker(self, raw: Dict[str, Any]) -> AliquotTrackerConfig:
        """Parse aliquot tracker configuration. URL normalization (trailing
        slash) is owned by AliquotTrackerClient, the sole consumer."""
        return AliquotTrackerConfig(
            url=raw.get('url'),
            api_key=raw.get('api_key'),
            submitter=raw.get('submitter'),
        )

    def _parse_api(self, raw: Dict[str, Any]) -> APIConfig:
        """Parse API configuration."""
        endpoints = []
        if 'endpoints' in raw:
            for ep in raw['endpoints']:
                url = ep.get('url', '')
                # If 'name' is omitted, fall back to URL (matches legacy dict
                # behavior in BaseWrapper._ensure_api_clients).
                endpoints.append(APIEndpoint(
                    url=url,
                    name=ep.get('name') or url or 'default',
                ))

        return APIConfig(
            endpoint=raw.get('endpoint', 'http://localhost:8000/api/v1'),
            endpoints=endpoints,
            retry_attempts=raw.get('retry_attempts', 3),
            timeout=raw.get('timeout', 30),
        )

    def _parse_client(self, raw: Dict[str, Any]) -> ClientConfig:
        """Parse client configuration."""
        return ClientConfig(
            username=raw.get('username', 'default_user'),
            cpu_name=raw.get('cpu_name', 'default_machine'),
        )

    def _parse_execution(self, raw: Dict[str, Any]) -> ExecutionConfig:
        """Parse execution configuration."""
        return ExecutionConfig(
            output_dir=raw.get('output_dir', 'data/outputs'),
            residue_dir=raw.get('residue_dir', 'data/residues'),
            failed_uploads_dir=raw.get('failed_uploads_dir', 'data/failed_uploads'),
            preserve_failed_uploads=raw.get('preserve_failed_uploads', True),
            save_raw_output=raw.get('save_raw_output', True),
            queue_dir=raw.get('queue_dir', 'data/queue'),
        )

    def _parse_logging(self, raw: Dict[str, Any]) -> LoggingConfig:
        """Parse logging configuration."""
        return LoggingConfig(
            file=raw.get('file', 'data/logs/ecm_client.log'),
            level=raw.get('level', 'INFO'),
            log_factors_found=raw.get('log_factors_found', True),
        )

    def _parse_programs(self, raw: Dict[str, Any]) -> ProgramsConfig:
        """Parse programs configuration."""
        return ProgramsConfig(
            gmp_ecm=self._parse_gmp_ecm(raw.get('gmp_ecm', {})),
            yafu=self._parse_yafu(raw.get('yafu', {})),
            cado_nfs=self._parse_cado(raw.get('cado_nfs', {})),
            t_level=self._parse_tlevel(raw.get('t_level', {})),
        )

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        """Safely convert a config value to int, handling scientific notation strings."""
        if value is None:
            return default
        return int(float(value))

    @staticmethod
    def _safe_optional_int(value: Any) -> Optional[int]:
        """Safely convert a config value to Optional[int], handling scientific notation."""
        if value is None:
            return None
        return int(float(value))

    def _parse_gmp_ecm(self, raw: Dict[str, Any]) -> GMPECMConfig:
        """Parse GMP-ECM configuration with safe numeric casting."""
        return GMPECMConfig(
            path=raw.get('path', 'ecm'),
            default_b1=self._safe_int(raw.get('default_b1'), 110000000),
            default_b2=self._safe_optional_int(raw.get('default_b2')),
            default_curves=self._safe_int(raw.get('default_curves'), 1),
            early_termination=raw.get('early_termination', True),
            gpu_enabled=raw.get('gpu_enabled', False),
            gpu_device=self._safe_int(raw.get('gpu_device'), 0),
            gpu_curves=self._safe_optional_int(raw.get('gpu_curves')),
            workers=self._safe_int(raw.get('workers', raw.get('stage2_workers')), 8),
            stage2_max_b1=self._safe_optional_int(raw.get('stage2_max_b1')),
            max_batch=self._safe_optional_int(raw.get('max_batch')),
            pm1_b1=self._safe_int(raw.get('pm1_b1'), 2900000000),
            pm1_b2=self._safe_int(raw.get('pm1_b2'), 1000000000000000),
            pp1_b1=self._safe_int(raw.get('pp1_b1'), 110000000),
            pp1_b2=self._safe_int(raw.get('pp1_b2'), 500000000000),
            gpu=self._parse_gpu(raw.get('gpu', {})),
        )

    def _parse_gpu(self, raw: Dict[str, Any]) -> GPUConfig:
        """Parse nested GPU configuration."""
        return GPUConfig(
            curves_per_batch=self._safe_int(raw.get('curves_per_batch'), 1000),
        )

    def _parse_yafu(self, raw: Dict[str, Any]) -> YAFUConfig:
        """Parse YAFU configuration."""
        return YAFUConfig(
            path=raw.get('path', 'yafu'),
            threads=raw.get('threads', 8),
        )

    def _parse_cado(self, raw: Dict[str, Any]) -> CADOConfig:
        """Parse CADO-NFS configuration."""
        return CADOConfig(
            path=raw.get('path', '~/cado-nfs/cado-nfs.py'),
            threads=raw.get('threads', 4),
            working_dir=raw.get('working_dir', '~/cado-nfs'),
        )

    def _parse_tlevel(self, raw: Dict[str, Any]) -> TLevelBinaryConfig:
        """Parse t-level configuration."""
        return TLevelBinaryConfig(
            path=raw.get('path', 'bin/t-level'),
        )
