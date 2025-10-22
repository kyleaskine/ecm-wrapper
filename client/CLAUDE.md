# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python wrapper clients for distributed integer factorization using:
- **GMP-ECM**: Elliptic Curve Method factorization via the ecm-wrapper.py script
- **YAFU**: Multi-method factorization (ECM, SIQS, NFS) via the yafu-wrapper.py script

Both wrappers submit factorization results to a centralized API server and are designed for computational cryptography research and distributed factorization projects.

## Core Architecture

### Configuration System
- **`client.yaml`**: Default configuration (checked into git)
  - Contains sensible defaults for all settings
  - Works out of the box with generic paths
  - Should not contain sensitive or machine-specific data

- **`client.local.yaml`**: Local overrides (gitignored)
  - Optional file for machine-specific settings
  - Overrides values from client.yaml
  - Use `client.local.yaml.example` as a template
  - Contains:
    - Your username and machine name
    - Production API endpoints
    - Full paths to ECM/YAFU binaries
    - CPU/GPU specific settings

Configuration loading:
1. Loads `client.yaml` (defaults)
2. Merges `client.local.yaml` if it exists (deep merge)
3. Local settings override defaults

### Wrapper Classes
- **ECMWrapper** (ecm-wrapper.py:14): Handles GMP-ECM execution with curve-by-curve control and two-stage processing
- **YAFUWrapper** (yafu-wrapper.py:8): Manages YAFU factorization in both ECM and automatic modes
- **BaseWrapper** (base_wrapper.py:17): Shared functionality for subprocess execution, result parsing, and API submission

### Result Submission
Both wrappers submit structured results to API endpoints with:
- Client identification and program metadata
- Factorization parameters and execution metrics
- Raw program output for debugging
- Automatic retry with exponential backoff

## Common Development Commands

### Running ECM Factorization
```bash
python3 ecm-wrapper.py --composite "123456789012345" --curves 100 --b1 50000
```

### Running YAFU Factorization  
```bash
# ECM mode
python3 yafu-wrapper.py --composite "123456789012345" --mode ecm --curves 100

# P-1 factorization
python3 yafu-wrapper.py --composite "123456789012345" --mode pm1 --b1 1000000

# Automatic factorization
python3 yafu-wrapper.py --composite "123456789012345" --mode auto

# Specific method (SIQS/NFS)
python3 yafu-wrapper.py --composite "123456789012345" --mode nfs
```

### Batch Processing
```bash
# Run from client/ directory
scripts/run_pm1_batch_yafu.sh    # YAFU P-1 batch
scripts/run_pm1_batch.sh         # GMP-ECM P-1 batch
scripts/run_batch.sh             # ECM batch
```

### Testing Without API Submission
```bash
python3 ecm-wrapper.py --composite "123456789012345" --no-submit
python3 yafu-wrapper.py --composite "123456789012345" --no-submit
```

### Code Quality Checks
When making code changes, always run both syntax and type checks:
```bash
# Basic syntax check (catches syntax errors only)
python3 -m py_compile *.py

# Type checking (catches type hint issues, undefined variables in annotations)
python3 -m mypy --ignore-missing-imports *.py

# OR use pylint for comprehensive linting
pylint *.py
```

**Important**: `py_compile` alone is insufficient - it does not validate type hints or catch undefined names in type annotations. Always use mypy or pylint for thorough validation, especially after refactoring.

## Key Implementation Details

### Output Parsing
- **ECM**: Multiple factor detection with prime factor filtering using `parse_ecm_output_multiple()` (parsing_utils.py:61)
- **YAFU**: Unified parsing functions `parse_yafu_ecm_output()` and `parse_yafu_auto_factors()` (parsing_utils.py:141-200)
- **Shared Infrastructure**: Unified subprocess execution via `BaseWrapper.run_subprocess_with_parsing()` (base_wrapper.py:254)

### Error Handling
- Subprocess timeouts: 1 hour for ECM, 2-4 hours for YAFU
- API submission retries with exponential backoff
- Raw output preservation for debugging

### File Organization
- **Raw outputs**: `data/outputs/` (configured via `execution.output_dir`)
- **Logs**: `data/logs/ecm_client.log` (configured via `logging.file`)
- **Factors found**: `data/factors_found.txt` (hardcoded)
- **Residue files**: `data/residues/` (configured via `execution.residue_dir`)
  - Auto-generated when using two-stage mode without `--save-residues`
  - Filename format: `residue_<composite_hash>_<timestamp>.txt`
  - Override with `--save-residues /path/to/custom.txt`
- **Failed submissions**: `data/results/` (for retry)
- All directories created automatically on first use

## Dependencies

Required Python packages: `subprocess`, `yaml`, `requests`, `pathlib`
External binaries: GMP-ECM and YAFU must be installed and accessible via configured paths

## Recent Improvements

### ECM Two-Stage Processing
- **Exit code handling**: Stage 1 now correctly treats factor discovery (exit code 8) as success
- **B2 accuracy**: When factor found in stage 1, results submitted with `b2=0` (stage 2 never ran)
- **GPU residue format**: Auto-detects and handles GPU single-line residue file format

### Aliquot Sequence Factorization (`aliquot-wrapper.py`)
- **Primality checks**: Miller-Rabin tests after trial division AND after ECM to avoid wasting time
- **CADO-NFS failure detection**: Properly detects CADO crashes and stops instead of submitting partial results
- **FactorDB integration enhancements**:
  - 3 automatic retries with exponential backoff (1s, 2s) for transient errors (502, timeouts)
  - Comprehensive logging to `data/logs/ecm_client.log`
  - Partial failure tracking (some factors succeed, others fail)
  - View logs: `grep "FactorDB" data/logs/ecm_client.log`

### Batch Pipeline (`scripts/run_batch_pipeline.py`)
- **GPU format support**: Fixed residue file splitting for GPU-generated files
- **Timing accuracy**: Submits combined stage1 + stage2 execution time
- **No false failures**: Fixed detection - `None` means "no factor found" (success), not failure
- **Testing mode**: Use `--no-submit` flag to test without submitting results

### Residue File Manager (`residue_manager.py`)
- **Format auto-detection**: Handles both GPU (single-line) and CPU (multi-line) residue formats
- **GPU format**: `METHOD=ECM; PARAM=3; SIGMA=...; B1=...; N=...; X=...; ...` (all on one line)
- **CPU format**: Separate `N=`, `B1=`, `SIGMA=` lines with multi-line curve blocks
- **Debug logging**: Shows detected format and first 20 lines on parsing failures