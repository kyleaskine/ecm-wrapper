# ECM Distributed Client

Standalone Python clients for distributed integer factorization using GMP-ECM, YAFU, and CADO-NFS.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install requests pyyaml
   ```

2. **Configure your client:**
   - Copy `client.local.yaml.example` to `client.local.yaml`
   - Edit `client.local.yaml` with your machine-specific settings (username, binary paths, API endpoints)
   - The system automatically merges `client.yaml` (defaults) with `client.local.yaml` (overrides)

3. **Run factorization:**
   ```bash
   # ECM with GMP-ECM (standard mode)
   python3 ecm-wrapper.py --composite "123456789012345" --curves 100 --b1 50000

   # ECM with two-stage GPU/CPU pipeline
   python3 ecm-wrapper.py --composite "123456789012345" --two-stage --curves 1000 --b1 50000

   # ECM with multiprocess parallelization
   python3 ecm-wrapper.py --composite "123456789012345" --multiprocess --workers 4 --curves 1000

   # ECM targeting specific t-level
   python3 ecm-wrapper.py --composite "123456789012345" --tlevel 30

   # Auto-work mode - continuously get work from server
   python3 ecm-wrapper.py --auto-work                          # Use server t-levels
   python3 ecm-wrapper.py --auto-work --work-count 5           # Process 5 assignments
   python3 ecm-wrapper.py --auto-work --tlevel 35              # Override with client t-level
   python3 ecm-wrapper.py --auto-work --multiprocess --workers 8  # Multiprocess mode

   # P-1 with GMP-ECM
   python3 ecm-wrapper.py --composite "123456789012345" --method pm1 --b1 1000000

   # YAFU automatic factorization
   python3 yafu-wrapper.py --composite "123456789012345" --mode auto

   # YAFU P-1 factorization
   python3 yafu-wrapper.py --composite "123456789012345" --mode pm1 --b1 1000000
   ```

4. **Batch processing:**
   ```bash
   # Run from client/ directory
   scripts/run_batch.sh              # ECM batch
   scripts/run_pm1_batch.sh          # GMP-ECM P-1 batch
   scripts/run_pm1_batch_yafu.sh     # YAFU P-1 batch

   # GPU/CPU pipeline batch processing
   python3 scripts/run_batch_pipeline.py --input numbers.txt --b1 50000 --curves 1000
   ```

5. **Resend failed submissions:**
   ```bash
   python3 resend_failed.py --dry-run  # Preview what would be resent
   python3 resend_failed.py            # Actually resend and mark as complete
   ```

## Core Wrappers

### `ecm-wrapper.py` - GMP-ECM Factorization
Comprehensive wrapper for GMP-ECM with multiple execution modes:

**Modes:**
- **Standard**: Run N curves with specified B1/B2
- **Two-stage**: GPU Stage 1 + multi-threaded CPU Stage 2 (optimal for large-scale ECM)
- **Multiprocess**: Multi-core CPU parallelization
- **T-level**: Progressive ECM targeting specific t-level (e.g., t30, t35)
- **Auto-work**: Continuously request and process work assignments from server
- **Stage 2 only**: Resume from existing residue files

**Methods:** ECM (default), P-1 (`--method pm1`), P+1 (`--method pp1`)

**GPU Support:** Optional GPU acceleration for Stage 1 (`--gpu` flag)

**Key Features:**
- Multiple factors per run (automatically detects and logs all factors)
- Automatic composite factor recursion (fully factors composite results)
- Primality testing (Miller-Rabin)
- Residue file management for two-stage processing
- Parametrization support (0, 1, 2, 3 for different curve families)

### `yafu-wrapper.py` - YAFU Multi-Method Factorization
Wrapper for YAFU with support for multiple factorization methods:

**Modes:**
- `ecm` - Elliptic Curve Method
- `pm1` - Pollard's P-1
- `pp1` - Williams's P+1
- `auto` - Automatic method selection (tries multiple approaches)
- `siqs` - Self-initializing quadratic sieve
- `nfs` - Number field sieve

**Key Features:**
- Automatic temporary file cleanup
- Thread pool configuration
- Real-time verbose output streaming
- Intelligent pretesting for ECM mode

### `cado-wrapper.py` - CADO-NFS Large Number Factorization
Wrapper for CADO-NFS (Number Field Sieve) for large factorizations:

**Key Features:**
- Thread pool management
- Designed for integration with aliquot sequences
- Output parsing for NFS-specific factor formats

### `aliquot-wrapper.py` - Aliquot Sequence Calculator
Progressive factorization of aliquot sequences (n → σ(n) - n → ...):

**Factorization Strategy:**
1. Trial division for small factors
2. ECM for medium factors
3. YAFU for larger composites
4. CADO-NFS for very large composites

**Key Features:**
- **Primality checks**: Miller-Rabin tests after trial division AND after ECM
- **CADO failure detection**: Detects crashes and avoids submitting partial results
- **FactorDB integration**: Automatic factor submission with retry logic (3 attempts, exponential backoff)
- **Sequence termination**: Stops on primes, perfect power cycles, or sociable chains
- **Progress tracking**: Maintains full factorization state

## Supporting Infrastructure

### Configuration System
- **`client.yaml`**: Default configuration (checked into git, generic paths)
- **`client.local.yaml`**: Machine-specific overrides (gitignored)
  - Automatically detected and deep-merged with `client.yaml`
  - Use `client.local.yaml.example` as template
  - Contains: username, CPU name, binary paths, API endpoints, GPU settings

### Core Utilities (lib/)

Implementation modules are organized in the `lib/` directory for clean separation from user-facing scripts.

- **`lib/base_wrapper.py`**: Shared base class for all wrappers
  - Configuration loading via ConfigManager
  - Logging setup (file + console)
  - API client initialization (supports multiple endpoints)
  - Factor logging (text + JSON formats)
  - Subprocess execution with timeout handling

- **`lib/config_manager.py`**: YAML configuration with deep merge
  - Loads defaults from `client.yaml`
  - Auto-detects and merges `client.local.yaml`
  - Validates required configuration keys

- **`lib/api_client.py`**: API communication with retry logic
  - Exponential backoff retry (configurable attempts)
  - Failed submission persistence to `data/results/`
  - Multi-endpoint support for redundancy
  - Health check endpoint

- **`lib/parsing_utils.py`**: Optimized output parsing
  - Pre-compiled regex patterns (ECMPatterns, YAFUPatterns)
  - Multiple factor extraction with deduplication
  - Sigma value extraction with 4-level fallback
  - Curve progress tracking

- **`lib/arg_parser.py`**: Unified argument parsing
  - Comprehensive validation (mode compatibility, required fields)
  - Default value resolution from config
  - GPU flag resolution logic

- **`lib/residue_manager.py`**: ECM residue file operations
  - Format auto-detection (GPU single-line vs CPU multi-line)
  - Residue file splitting for parallel Stage 2
  - Metadata extraction (N, B1, curve count)

- **`lib/result_processor.py`**: Factor processing and logging
  - Factor deduplication
  - Composite factor recursion
  - Unified logging interface

- **`lib/stage2_executor.py`**: Multi-threaded Stage 2 execution
  - Worker pool management
  - Early termination on factor discovery
  - Progress interval reporting

- **`lib/ecm_config.py`**: Configuration dataclasses for ECM execution
- **`lib/ecm_math.py`**: Mathematical utilities (trial division, primality, t-level)
- **`lib/ecm_executor.py`**: Command building and execution engine
- **`lib/ecm_pipeline.py`**: Multi-stage pipeline orchestration
- **`lib/ecm_worker_process.py`**: Multiprocess worker encapsulation

## Helper Scripts

- **`resend_failed.py`**: Retry failed API submissions
  - Scans `data/results/` for unsubmitted results
  - Dry-run mode for testing (`--dry-run`)
  - Marks files as completed after successful submission

- **`scripts/run_batch_pipeline.py`**: GPU/CPU pipeline batch processing
  - GPU workers for Stage 1 (save residues)
  - CPU workers for Stage 2 (process residues)
  - Queue-based coordination for maximum throughput

## Data Directory Structure

All outputs auto-created in `data/`:
- **`data/logs/`** - Client logs (`ecm_client.log`)
- **`data/outputs/`** - Raw program outputs (timestamped)
- **`data/factors_found.txt`** - Human-readable factor log
- **`data/factors.json`** - Machine-readable factor log
- **`data/residues/`** - ECM residue files (two-stage mode)
- **`data/results/`** - Failed API submissions (for retry)

## Testing

Run the test suite to validate parsing logic and utilities:

```bash
cd client
pytest tests/ -v                           # Run all tests
pytest tests/test_factorization.py -v     # Test parsing logic
pytest tests/test_config_manager.py -v    # Test configuration
pytest tests/test_api_client.py -v        # Test API client
```

**Test Coverage:**
- `test_factorization.py`: ECM output parsing, factor extraction (263 lines)
- `test_config_manager.py`: Config loading and merging (319 lines)
- `test_api_client.py`: API payload building (262 lines)
- `test_residue_manager.py`: Residue file parsing (130 lines)
- `test_sigma_matching.py`: Sigma value extraction (122 lines)

## Advanced Usage Examples

### Two-Stage ECM with GPU
```bash
# Stage 1 on GPU, Stage 2 on 8 CPU cores
python3 ecm-wrapper.py --composite "123...456" \
  --two-stage --curves 10000 --b1 50000 --b2 12500000 \
  --stage2-workers 8 --gpu-device 0
```

### Resume from Residue File
```bash
# Run Stage 2 only on existing residues
python3 ecm-wrapper.py --stage2-only residue_file.txt \
  --b1 50000 --b2 12500000 --stage2-workers 8
```

### Multiprocess ECM
```bash
# Use 16 CPU cores for parallel ECM
python3 ecm-wrapper.py --composite "123...456" \
  --multiprocess --workers 16 --curves 10000 --b1 50000
```

### T-Level Targeting
```bash
# Progressive ECM to reach t30 (uses optimal B1 values)
python3 ecm-wrapper.py --composite "123...456" --tlevel 30
```

### Test Without API Submission
```bash
# Run locally without submitting results
python3 ecm-wrapper.py --composite "123...456" --curves 100 --no-submit
```

## Configuration

Edit `client.local.yaml` (create from `client.local.yaml.example`) to set:
- **API endpoints**: Single or multiple for redundancy
- **Binary paths**: Full paths to `ecm`, `yafu`, `cado-nfs.py`
- **Client identification**: Your username and CPU name (for tracking)
- **Default parameters**: B1/B2 values for ECM/P-1/P+1
- **GPU settings**: Device ID, curves per GPU batch
- **Logging**: Log level, output directories

## Documentation

See **`../CLAUDE.md`** (root) for comprehensive documentation including:
- Full development guide (client + server)
- Architecture overview
- Recent bug fixes and improvements
- Database schema
- Binary dependencies
- Implementation details