# Client CLAUDE.md

Implementation details for the ECM client components. See also the [root CLAUDE.md](../CLAUDE.md) for project overview and dev commands.

## Entry Points (Two Separate Scripts by Design)

| Script | Purpose | Composite Source | Residue Handling |
|--------|---------|------------------|------------------|
| `ecm_client.py` | Server-coordinated work | From server API | Upload/download via server |
| `ecm_wrapper.py` | Local/manual factorization | `--composite` argument | Local files only |

**`ecm_client.py`** - Server-coordinated modes:
- Auto-work: Continuously requests work from server
- `--composite`: Target specific composite (server provides t-level info)
- `--pm1`: P-1 sweep (1 curve per composite, B1 from target t-level)
- `--pp1`: P+1 sweep (3 curves per composite, configurable with `--pp1-curves`)
- `--p1`: Combined P-1 + P+1 sweep per composite
- `--stage1-only`: Upload residues to server
- `--stage2-only`: Download residues from server (with `--min-b1`/`--max-b1` filters)

**`ecm_wrapper.py`** - Local/manual modes:
- Requires `--composite` argument
- `--stage2-only <file>`: Process a local residue file
- `--stage1-only` with `--upload`: Save locally and optionally upload

### Parser Architecture
Both entry points use parsers defined in `lib/arg_parser.py`:
- `create_client_parser()` - For `ecm_client.py` (server-coordinated)
- `create_ecm_parser()` - For `ecm_wrapper.py` (local/manual)
- `create_yafu_parser()` - For `yafu_wrapper.py`

Note: `--stage2-only` has different semantics in each parser:
- In `create_client_parser()`: Boolean flag (downloads from server)
- In `create_ecm_parser()`: String path to local residue file

### Submission Flag Behavior
- **`ecm_wrapper.py`**: Default = no submission. Use `--submit` to opt-in.
- **`ecm_client.py`**: Default = always submits. Use `--no-submit` to opt-out.

## V2 API (Config-Based Architecture)

The ECM wrapper uses a modern config-based API (v2) with typed configuration objects and strongly-typed return values.

### Configuration Classes (`lib/ecm_config.py`)

**ECMConfig** - Standard ECM execution:
```python
config = ECMConfig(
    composite="123456789012345",
    b1=50000,
    b2=5000000,         # Optional, None = GMP-ECM default
    curves=100,
    sigma=None,          # Optional: int or "3:N" string format
    parametrization=1,   # 1=CPU (Montgomery), 3=GPU (Twisted Edwards)
    threads=1,
    verbose=False,
    use_gpu=False,       # Auto-sets parametrization=3 if True
    method="ecm"         # 'ecm', 'pm1', 'pp1'
)
result = wrapper.run_ecm_v2(config)
```

**TwoStageConfig** - GPU stage 1 + CPU stage 2:
```python
config = TwoStageConfig(
    composite="123456789012345",
    b1=110000000,
    b2=11000000000000,
    stage1_curves=3000,
    stage2_curves_per_residue=1000,
    stage1_parametrization=3,  # GPU
    stage2_parametrization=1,  # CPU
    threads=8,
    save_residues="residues/output.txt",  # Optional path
    no_submit=False
)
result = wrapper.run_two_stage_v2(config)
```

**MultiprocessConfig** - Parallel CPU workers:
```python
config = MultiprocessConfig(
    composite="123456789012345",
    b1=50000,
    total_curves=1000,
    curves_per_process=100,
    num_processes=8,       # Auto-detects CPU count if None
    parametrization=1,
    verbose=False
)
result = wrapper.run_multiprocess_v2(config)
```

**TLevelConfig** - Progressive t-level targeting:
```python
config = TLevelConfig(
    composite="123456789012345",
    target_t_level=35.0,
    b1_strategy="optimal",  # 'optimal', 'conservative', 'aggressive'
    parametrization=1,       # Auto-switches to 3 if use_two_stage=True
    threads=1,
    use_two_stage=False,
    project="my-project",
    no_submit=False,
    work_id=None             # For auto-work batch submissions
)
result = wrapper.run_tlevel_v2(config)
```

### FactorResult Return Type (`lib/ecm_config.py:166`)

All v2 methods return a `FactorResult` object:
```python
result = wrapper.run_ecm_v2(config)
result.success       # bool
result.factors       # List[str]
result.sigmas        # List[Optional[str]]
result.curves_run    # int
result.execution_time  # float
result.raw_output    # Optional[str]

for factor, sigma in result.factor_sigma_pairs:
    print(f"Factor {factor} found with sigma {sigma}")
```

### Config Validation (ECMConfigValidation Mixin)

All config classes inherit from `ECMConfigValidation` mixin:
- `_validate_composite(composite)` - Ensures composite is non-empty
- `_validate_b1(b1)` - Ensures B1 is positive
- `_validate_method(method)` - Validates method is 'ecm', 'pm1', or 'pp1'
- `_validate_parametrization(param)` - Validates param is 0-3

## WorkMode Pattern (lib/work_modes.py)

Auto-work mode (`ecm_client.py`) uses a Strategy pattern:
- **StandardAutoWorkMode** - Regular ECM work from server assignments
- **P1WorkMode** - P-1/P+1 sweep across composites (`--pm1`, `--pp1`, `--p1`)
- **Stage1ProducerMode** - GPU stage 1 only, uploads residues to server
- **Stage2ConsumerMode** - Downloads residues from server, runs stage 2

Each mode implements the `WorkMode` abstract base class with template method `run()`:
```python
class WorkMode(ABC):
    def run(self) -> int:
        while self.should_continue():
            work = self.request_work()
            result = self.execute_work(work)
            self.submit_results(work, result)
            self.complete_work(work)
        return self.completed_count
```

## Auto-Work Mode

**API Methods** (`lib/api_client.py`):
- `get_ecm_work()` - Request work from `/ecm-work` endpoint
- `get_p1_work()` - Request P-1/P+1 work from `/p1-work` endpoint
- `complete_work()` - Mark work complete via `POST /work/{work_id}/complete`
- `abandon_work()` - Release work via `DELETE /work/{work_id}`
- `upload_residue()` - Upload residue file to server (decoupled two-stage)
- `get_residue_work()` - Request stage 2 work from residue pool
- `download_residue()` - Download residue file for stage 2 processing
- `complete_residue()` - Mark stage 2 complete, supersede stage 1
- `abandon_residue()` - Release residue claim

## P-1/P+1 Sweep Mode (2026-02)

**New design** (`P1WorkMode` in `lib/work_modes.py`):
- Three mutually exclusive flags: `--pm1`, `--pp1`, `--p1`
- Uses dedicated `/p1-work` endpoint that filters composites by PM1/PP1 coverage
- Server calculates B1 one step above composite's target t-level, returns in response
  - Example: target_t=48 → next entry ≥ 48 is t50 (B1=43M) → one above = t55 → B1=110M
  - Client caps by config `pm1_b1` / `pp1_b1` values
- B2 omitted (GMP-ECM uses its default ratio)
- Per composite: PM1 runs 1 curve, PP1 runs N curves (default 3, `--pp1-curves`)
- If PM1 finds a factor, PP1 is skipped

**Client API**: `get_p1_work()` in `lib/api_client.py`, `request_p1_work()` in `lib/work_helpers.py`
**B1 lookup**: `get_b1_above_tlevel()` in `lib/ecm_math.py`
**Config values**: `typed_config.py` handles scientific notation strings via `_safe_int()`

## Decoupled Two-Stage ECM

**Workflow:**
```
Stage 1 (GPU): Request work → Run stage 1 (B2=0) → Submit → Upload residue
Stage 2 (CPU): Request residue → Download → Run stage 2 → Submit → Complete (supersedes stage 1)
```

**Usage:**
```bash
python3 ecm_client.py --stage1-only --b1 110000000 --curves 3000 --gpu
python3 ecm_client.py --stage2-only --b2 11000000000000 --workers 8
```

## T-Level Calculation Details

T-level mode uses a hybrid approach:

1. **Cached transitions** (`lib/ecm_math.py:168`): Standard 5-digit increments (t20→t25, t25→t30, etc.)
   - Separate caches for parametrization 1 (CPU) and 3 (GPU)
2. **Direct t-level binary calls**: Non-standard targets (e.g., t35→t38.75)
   - Uses `calculate_curves_to_target_direct()` with `-w`, `-t`, `-b`, `-p` flags
3. **Parametrization awareness**: p=1 (CPU) and p=3 (GPU) give different t-levels for same curves

## Output Parsing

- **ECM**: `parse_ecm_output_multiple()` (lib/parsing_utils.py:61)
  - Pattern: `r'Factor found in step \d+: (\d+)'`
  - Composite factor filtering and multi-pattern matching with deduplication
- **YAFU**: `parse_yafu_ecm_output()` and `parse_yafu_auto_factors()` (lib/parsing_utils.py:141-200)
  - Multiple patterns for P/Q notation and factor formats
- **Shared**: Unified subprocess execution via `BaseWrapper.run_subprocess_with_parsing()` (lib/base_wrapper.py:278)

## ResultsBuilder (`lib/results_builder.py`)

Unified builder for constructing ECM results dictionaries:
```python
from lib.results_builder import ResultsBuilder, results_for_ecm, results_for_stage1

results = (ResultsBuilder('123456789', 'ecm')
           .with_b1(50000).with_b2(5000000)
           .with_curves(100, 75)
           .with_parametrization(3)
           .with_execution_time(123.45)
           .build())

# Stage1-only
results = results_for_stage1('123', b1=50000, curves=100, param=1).build()
```

**Key methods:** `with_b1()`, `with_b2()`, `with_curves()`, `with_factors()`, `with_single_factor()`, `add_raw_output()`, `with_parametrization()`, `with_execution_time()`, `as_stage1_only()`, `as_two_stage()`, `as_multiprocess()`, `build()`, `build_no_truncate()`

**Factory functions:** `results_for_ecm()`, `results_for_stage1()`

**Migration note**: `BaseWrapper.create_base_results()` is deprecated - use ResultsBuilder for new code.

## Graceful Shutdown

Multi-level Ctrl+C handling for Stage 2 and CPU Stage 1 execution:
- **1st Ctrl+C**: Workers finish current chunks, then submit
- **2nd Ctrl+C**: Workers stop after current curve, then submit
- **3rd Ctrl+C**: Immediate abort

Implementation: `lib/ecm_executor.py` uses `shutdown_level` counter and shared `stop_event`

## Error Handling and Timeouts
- **Subprocess timeouts**: 1 hour for ECM, 2-4 hours for YAFU
- **API submission retries**: Exponential backoff with configurable attempts
- **Raw output preservation**: All outputs saved to `data/outputs/`
- **Failed submission persistence**: JSON files in `data/results/` for manual retry

## File Organization
All data directories are auto-created on first use:
- **Raw outputs**: `data/outputs/` (configured via `execution.output_dir`)
- **Logs**: `data/logs/ecm_client.log` (configured via `logging.file`)
- **Factors found**: `data/factors_found.txt` (human-readable), `data/factors.json` (machine-readable)
- **Residue files**: `data/residues/` (configured via `execution.residue_dir`)
- **Failed submissions**: `data/results/` (for retry via `resend_failed.py`)

## Scientific Notation Support
B1/B2 parameters accept scientific notation: `--b1 26e7`, `--b2 4e11`
Supports: lowercase/uppercase e, decimals (2.6e8), explicit + sign (26e+7)

## Recent Client Bug Fixes

### Residue Upload Permanent-Failure Handling (2026-06)
- `lib/api_client.py`: `upload_residue()` raises `PermanentUploadError` on any
  non-duplicate **4xx** (composite factored, stage-1 attempt gone, invalid file).
  **5xx** still returns `None` (transient → retry). Duplicate-400 returns the
  `{"already_uploaded": True}` success dict.
- `lib/submission_queue.py`: `_retry_item()` discards queue items on
  `PermanentUploadError` instead of retrying to the 200-attempt cap.
- `lib/ecm_executor.py`: direct stage-1 upload path skips queueing on a permanent
  rejection. See server-side fix in `server/CLAUDE.md`.

### Two-Stage ECM Improvements
- Exit code 8 (factor found) treated as success in stage 1
- Factor found in stage 1 correctly submits with `b2=0`
- Pipeline submits combined stage1 + stage2 execution time

### Residue File Handling
- `lib/residue_manager.py` auto-detects GPU (single-line) and CPU (multi-line) formats
- Checks first 5 lines for `METHOD=ECM; SIGMA=...; ...` pattern

### FactorDB Integration (aliquot_wrapper.py)
- 3 automatic retries with exponential backoff for transient errors
- All operations logged to `data/logs/ecm_client.log`

### Aliquot Sequence Factorization
- Miller-Rabin primality tests after trial division AND after ECM
- CADO-NFS failure detection prevents partial result submission
- Early termination when cofactor is already prime

### Pipeline Batch Processing
- No results submitted when stage 2 fails
- `None` return from stage 2 means "no factor found" (success), not failure
