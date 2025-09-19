# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python wrapper clients for distributed integer factorization using:
- **GMP-ECM**: Elliptic Curve Method factorization via the ecm-wrapper.py script
- **YAFU**: Multi-method factorization (ECM, SIQS, NFS) via the yafu-wrapper.py script

Both wrappers submit factorization results to a centralized API server and are designed for computational cryptography research and distributed factorization projects.

## Core Architecture

### Configuration System
- `client.yaml`: Central configuration file containing:
  - Client identity and machine specs
  - API endpoints and retry settings
  - Program paths for ecm/yafu binaries
  - Logging and output preferences
  - Default factorization parameters

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
- Raw outputs saved to `data/outputs/` with timestamps
- Logs written to `data/logs/ecm_client.log`
- Factors logged to `data/factors_found.txt`
- Residue files saved to `data/residues/`
- Failed submissions saved to `data/results/`
- Configuration validation on startup

## Dependencies

Required Python packages: `subprocess`, `yaml`, `requests`, `pathlib`
External binaries: GMP-ECM and YAFU must be installed and accessible via configured paths