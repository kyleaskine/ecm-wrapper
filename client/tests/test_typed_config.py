#!/usr/bin/env python3
"""
Tests for the typed configuration layer (lib/typed_config.py).

Focuses on:
- Round-trip preservation: load YAML -> AppConfig -> to_dict() should reproduce
  every field that was in the YAML.
- Defaults: fields absent from YAML get the documented dataclass defaults.
- Schema coverage: every field used by production code is reachable on the
  typed surface (regression guard against half-finished migrations).
"""
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.typed_config import (
    TypedConfigLoader,
    AppConfig,
    GMPECMConfig,
    GPUConfig,
    ExecutionConfig,
)


def _load_yaml(yaml_text: str) -> AppConfig:
    """Helper: write yaml_text to a temp file and load it through TypedConfigLoader."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        f.write(yaml_text)
        path = f.name
    try:
        return TypedConfigLoader().load(path)
    finally:
        Path(path).unlink()


# ==================== Schema additions (Phase 2 step 1) ====================


def test_gmp_ecm_max_batch_parses_from_yaml():
    cfg = _load_yaml("""
programs:
  gmp_ecm:
    max_batch: 2304
""")
    assert cfg.programs.gmp_ecm.max_batch == 2304


def test_gmp_ecm_max_batch_defaults_to_none():
    cfg = _load_yaml("programs: {gmp_ecm: {}}")
    assert cfg.programs.gmp_ecm.max_batch is None


def test_gmp_ecm_max_batch_accepts_scientific_notation():
    cfg = _load_yaml("""
programs:
  gmp_ecm:
    max_batch: 2.5e3
""")
    assert cfg.programs.gmp_ecm.max_batch == 2500


def test_gmp_ecm_stage2_max_b1_parses_from_yaml():
    """Regression guard: _parse_gmp_ecm previously dropped this field on read."""
    cfg = _load_yaml("""
programs:
  gmp_ecm:
    stage2_max_b1: 110000000
""")
    assert cfg.programs.gmp_ecm.stage2_max_b1 == 110000000


def test_gmp_ecm_gpu_subdict_parses():
    cfg = _load_yaml("""
programs:
  gmp_ecm:
    gpu:
      curves_per_batch: 3072
""")
    assert isinstance(cfg.programs.gmp_ecm.gpu, GPUConfig)
    assert cfg.programs.gmp_ecm.gpu.curves_per_batch == 3072


def test_gmp_ecm_gpu_subdict_default_when_absent():
    cfg = _load_yaml("programs: {gmp_ecm: {}}")
    assert cfg.programs.gmp_ecm.gpu.curves_per_batch == 1000


def test_execution_queue_dir_parses_from_yaml():
    cfg = _load_yaml("""
execution:
  queue_dir: /custom/queue
""")
    assert cfg.execution.queue_dir == "/custom/queue"


def test_execution_queue_dir_default():
    cfg = _load_yaml("execution: {}")
    assert cfg.execution.queue_dir == "data/queue"


# ==================== Round-trip preservation ====================


def test_to_dict_includes_max_batch():
    cfg = AppConfig()
    cfg.programs.gmp_ecm.max_batch = 4096
    d = cfg.to_dict()
    assert d['programs']['gmp_ecm']['max_batch'] == 4096


def test_to_dict_includes_stage2_max_b1():
    cfg = AppConfig()
    cfg.programs.gmp_ecm.stage2_max_b1 = 50000000
    d = cfg.to_dict()
    assert d['programs']['gmp_ecm']['stage2_max_b1'] == 50000000


def test_to_dict_includes_gpu_subdict():
    cfg = AppConfig()
    cfg.programs.gmp_ecm.gpu.curves_per_batch = 2304
    d = cfg.to_dict()
    assert d['programs']['gmp_ecm']['gpu'] == {'curves_per_batch': 2304}


def test_to_dict_includes_queue_dir():
    cfg = AppConfig()
    cfg.execution.queue_dir = "/var/cache/ecm/queue"
    d = cfg.to_dict()
    assert d['execution']['queue_dir'] == "/var/cache/ecm/queue"


def test_api_endpoint_name_falls_back_to_url():
    """Unnamed endpoints use URL as their name (matches legacy dict behavior)."""
    cfg = _load_yaml("""
api:
  endpoints:
    - url: http://example.com/api
""")
    assert cfg.api.endpoints[0].url == "http://example.com/api"
    assert cfg.api.endpoints[0].name == "http://example.com/api"


def test_api_endpoint_explicit_name_wins():
    cfg = _load_yaml("""
api:
  endpoints:
    - url: http://example.com/api
      name: production
""")
    assert cfg.api.endpoints[0].name == "production"


def test_full_round_trip_preserves_all_new_fields():
    """Load -> mutate -> to_dict -> reload produces equivalent typed config."""
    yaml_text = """
api:
  endpoint: http://example.com/api
  timeout: 45
  retry_attempts: 5
client:
  username: tester
  cpu_name: testbox
execution:
  output_dir: /tmp/out
  residue_dir: /tmp/res
  failed_uploads_dir: /tmp/fail
  preserve_failed_uploads: false
  save_raw_output: false
  queue_dir: /tmp/queue
logging:
  file: /tmp/log
  level: DEBUG
  log_factors_found: false
programs:
  gmp_ecm:
    path: /opt/ecm
    default_b1: 50000
    default_curves: 100
    early_termination: false
    gpu_enabled: true
    gpu_device: 1
    gpu_curves: 5000
    workers: 16
    stage2_max_b1: 110000000
    max_batch: 2304
    pm1_b1: 1000000000
    pp1_b1: 100000000
    gpu:
      curves_per_batch: 2304
  yafu:
    path: /opt/yafu
    threads: 16
  cado_nfs:
    path: /opt/cado
    threads: 8
    working_dir: /opt/cado/work
  t_level:
    path: /opt/bin/t-level
factordb:
  cookie: fdbcookie123
aliquot_tracker:
  url: https://aliquot.example.com
  api_key: trackerkey456
  submitter: tester-handle
"""
    cfg1 = _load_yaml(yaml_text)
    d = cfg1.to_dict()

    # Round-trip through the dict -> typed loader path
    import yaml as pyyaml
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        pyyaml.safe_dump(d, f)
        path = f.name
    try:
        cfg2 = TypedConfigLoader().load(path)
    finally:
        Path(path).unlink()

    # All the previously-buggy or newly-added fields survive
    assert cfg2.programs.gmp_ecm.max_batch == 2304
    assert cfg2.programs.gmp_ecm.stage2_max_b1 == 110000000
    assert cfg2.programs.gmp_ecm.gpu.curves_per_batch == 2304
    assert cfg2.execution.queue_dir == "/tmp/queue"
    assert cfg2.factordb.cookie == "fdbcookie123"
    assert cfg2.aliquot_tracker.url == "https://aliquot.example.com"
    assert cfg2.aliquot_tracker.api_key == "trackerkey456"
    assert cfg2.aliquot_tracker.submitter == "tester-handle"

    # Sanity: pre-existing fields also survive
    assert cfg2.api.endpoint == "http://example.com/api"
    assert cfg2.programs.gmp_ecm.path == "/opt/ecm"
    assert cfg2.programs.yafu.threads == 16
