# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**See also:** [`client/CLAUDE.md`](./client/CLAUDE.md) for client implementation details, [`server/CLAUDE.md`](./server/CLAUDE.md) for server implementation details.

## Project Overview

ECM Coordination Middleware - a minimal, focused system for coordinating distributed ECM factorization work:
- **Client components**: Standalone Python wrappers for GMP-ECM and YAFU factorization
- **Server component**: FastAPI-based coordination middleware with PostgreSQL backend
- **Architecture**: Lightweight middleware that projects can integrate for ECM work coordination
- **Focus**: Pure ECM coordination, t-level progress tracking, and work assignment

## Development Commands

### Client Development
```bash
# Install client dependencies
pip install requests pyyaml

# Run ECM factorization with GMP-ECM
python3 client/ecm_wrapper.py --composite "123456789012345" --curves 100 --b1 50000

# Run YAFU factorization (various modes)
python3 client/yafu_wrapper.py --composite "123456789012345" --mode ecm --curves 100
python3 client/yafu_wrapper.py --composite "123456789012345" --mode pm1 --b1 1000000
python3 client/yafu_wrapper.py --composite "123456789012345" --mode auto

# Test without API submission (ecm_wrapper.py defaults to no submission)
python3 client/ecm_wrapper.py --composite "123456789012345"

# Submit results to API (opt-in for ecm_wrapper.py)
python3 client/ecm_wrapper.py --composite "123456789012345" --submit

# Stage 1 only with residue upload to server (manual mode)
python3 client/ecm_wrapper.py --composite "123456789012345" --b1 50000 --curves 100 --stage1-only --upload

# Auto-work mode - continuously request and process work from server
python3 client/ecm_client.py                    # Use server's target t-levels
python3 client/ecm_client.py --work-count 5     # Process 5 assignments then exit
python3 client/ecm_client.py --tlevel 35        # Override with client t-level
python3 client/ecm_client.py --b1 50000 --b2 5000000 --curves 100  # Override with B1/B2
python3 client/ecm_client.py --b1 26e7 --b2 4e11 --curves 100      # Scientific notation support
python3 client/ecm_client.py --two-stage --b1 50000 --b2 5000000   # GPU two-stage mode
python3 client/ecm_client.py --multiprocess --workers 8            # Multiprocess mode
python3 client/ecm_client.py --min-digits 60 --max-digits 80       # Filter by size

# P-1/P+1 sweep mode - run PM1/PP1 across composites
python3 client/ecm_client.py --pm1                               # P-1 only (1 curve per composite)
python3 client/ecm_client.py --pp1                               # P+1 only (3 curves per composite)
python3 client/ecm_client.py --p1                                # P-1 + P+1 per composite
python3 client/ecm_client.py --p1 --work-count 10                # Process 10 composites then exit
python3 client/ecm_client.py --pp1 --pp1-curves 5                # Custom P+1 curve count

# Decoupled two-stage mode - separate GPU and CPU workers
python3 client/ecm_client.py --stage1-only --b1 110000000 --curves 3000 --gpu  # GPU producer
python3 client/ecm_client.py --stage2-only --b2 11000000000000 --workers 8  # CPU consumer

# Run batch processing scripts
cd client/scripts/
./run_batch.sh                    # ECM batch
./run_pm1_batch.sh               # GMP-ECM P-1 batch
./run_pm1_batch_yafu.sh          # YAFU P-1 batch

# Resend failed submissions
python3 resend_failed.py --dry-run  # Test without marking files
python3 resend_failed.py            # Submit and mark as completed
```

### Server Development

**Quick Start (Recommended)**
```bash
cd server/

# PostgreSQL runs natively on the default port 5432 (DATABASE_URL is set
# in server/.env). Start the API server:
source venv/bin/activate
export DATABASE_URL="postgresql://ecm_user:ecm_password@localhost:5432/ecm_distributed"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Setup Commands**
```bash
# Install server dependencies (first time only)
cd server/
pip install -r requirements.txt

# Database operations
alembic revision --autogenerate -m "Description"  # Create new migration
alembic upgrade head                               # Apply migrations
alembic downgrade -1                              # Rollback one migration
```

**Alternative Setup (Docker PostgreSQL)**
```bash
# Start the containerized PostgreSQL (exposed on host port 5434)
docker-compose -f docker-compose.dev.yml up -d postgres

export DATABASE_URL="postgresql://ecm_user:ecm_password@localhost:5434/ecm_distributed"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing and Validation
```bash
# Test server health
curl http://localhost:8000/health

# View API documentation
# Open: http://localhost:8000/docs

# Monitor client logs
tail -f client/scripts/ecm_client.log

# Test API endpoints
curl http://localhost:8000/api/v1/composites
curl -X POST http://localhost:8000/api/v1/results/ecm \
  -H "Content-Type: application/json" \
  -d '{"client_id": "test", "composite": "123", "factors": ["3", "41"]}'

# Run unit tests
cd server
source venv/bin/activate
pytest tests/test_number_utils.py -v           # Test number utilities
pytest tests/ -v  # All server tests

cd ../client
pytest test_factorization.py -v               # Test parsing logic
```

## Architecture Overview

### ECM Coordination Model
```
┌─────────────────────┐    HTTP/API     ┌─────────────────────┐
│   Client (Python)  │◄──────────────►│ ECM Middleware      │
│   • GMP-ECM        │                 │   • Work assignment │
│   • YAFU           │                 │   • T-level tracking│
│   • Batch scripts  │                 │   • Progress monitor│
└─────────────────────┘                 └─────────────────────┘
                                                 │
                                        ┌───────────────────┐
                                        │ Any Project Can   │
                                        │ Submit Numbers    │
                                        │ Get Results       │
                                        └───────────────────┘
```

### Key Components

#### Client Components
- **ECMWrapper** (client/ecm_wrapper.py:14): GMP-ECM execution with multiple modes
- **YAFUWrapper** (client/yafu_wrapper.py:8): YAFU multi-method factorization coordination
- **CADOWrapper** (client/cado_wrapper.py): Number Field Sieve for large numbers
- **AliquotWrapper** (client/aliquot_wrapper.py): Aliquot sequence calculator with FactorDB integration
- **BaseWrapper** (client/lib/base_wrapper.py:17): Shared base class

#### Server Components
- **API Server** (server/app/main.py): FastAPI middleware with coordination endpoints
- **Database Models** (server/app/models/): Minimal schema focused on ECM coordination
- **API Routes** (server/app/api/v1/): RESTful endpoints for work assignment and results
- **T-Level Services** (server/app/services/): T-level calculation and progress tracking

### Configuration System
- **client.yaml**: Default client configuration (API endpoints, binary paths, default parameters)
  - Contains sensible defaults that work out of the box; checked into git
- **client.local.yaml**: Local overrides for client.yaml (gitignored, machine-specific settings)
  - Deep merges with client.yaml (local settings override defaults)
  - **Important**: Always pass `client.yaml` as config path - BaseWrapper handles the merge
- **server/app/config.py**: Server configuration (database URL, API settings, residue storage path)
- **docker-compose.yml**: Full system deployment configuration
- **alembic.ini**: Database migration configuration

### Binary Dependencies
- **GMP-ECM**: Configure path in client.yaml `programs.gmp_ecm.path`
- **YAFU**: Configure path in client.yaml `programs.yafu.path`
- **t-level binary**: Deployed to `server/bin/t-level` for t-level calculations
- **PARI/GP**: Installed in Docker container for elliptic curve group order calculations

## ECM Coordination Workflow

1. **Project submits numbers**: Upload composites with target t-levels via API or admin interface
2. **Work assignment**: Clients request ECM work assignments with optimal B1/B2 parameters
3. **Client execution**: Wrapper scripts execute GMP-ECM/YAFU binaries with assigned parameters
4. **Progress tracking**: T-level progress updated as curves complete via `/submit_result`
5. **Factor discovery**: Numbers marked as factored when factors found
   - Group order calculation via PARI/GP for all parametrizations (0, 1, 2, 3)
6. **Result delivery**: Projects retrieve factorization results via API

## Key Features

### Auto-Work Mode
Clients continuously request and process work from the server. See `client/CLAUDE.md` for details.

### Decoupled Two-Stage ECM
GPU and CPU workers run stage 1 and stage 2 independently. Stage 1 produces residues uploaded to server; stage 2 consumers download and process them. See `client/CLAUDE.md` for client details and `server/CLAUDE.md` for server endpoints.

### P-1/P+1 Sweep Mode (2026-02)
Flags: `--pm1`, `--pp1`, `--p1`. Uses `/p1-work` endpoint. B1 calculated one step above target t-level. See `client/CLAUDE.md` for client details and `server/CLAUDE.md` for server endpoint.

### Multi-Factor Batch Submission
All factors from a single ECM run are submitted in one API call via `factors_found` list. Server processes all factors before updating composite state.

### Database Connection
- **Default URL**: `postgresql://ecm_user:ecm_password@localhost:5432/ecm_distributed` (native PostgreSQL on the default port; also set in `server/.env`)
- **Docker alternative**: containerized PostgreSQL exposed on port 5434 (host) → 5432 (container)
- **Environment**: Set `DATABASE_URL` to override default connection string

### Code Quality Checks
When making code changes, run a type checker — `py_compile` only catches syntax errors and is not sufficient on its own.

Both `mypy` and `pyright` are installed via pipx (`~/.local/bin/`). They overlap on real bugs; differences:
- **mypy** is quieter and focused on argument/assignment type mismatches.
- **pyright** adds stricter flow-narrowing analysis (catches more `Optional` leaks) and flags duck-typed mocks in tests. More noise, also more coverage.

Typical invocations from `client/`:
```bash
mypy --ignore-missing-imports .                 # Quieter, focused checks
pyright                                          # Stricter, more findings
```

For `server/`, both checkers must resolve imports (and mypy's pydantic plugin) from the
server venv — the pipx-installed binaries fail without it:
```bash
cd server/
venv/bin/python -m mypy app/                     # mypy is installed in the venv
pyright --pythonpath venv/bin/python app/        # point pipx pyright at the venv
```
Server schemas use `Field(default=None, ...)` (keyword form) — pyright does not
recognize the positional `Field(None, ...)` form as a default and will emit false
"Argument missing" errors if positional defaults are reintroduced.

Pyright auto-discovers source roots; mypy needs `--ignore-missing-imports` until a `mypy.ini` / `pyproject.toml` config is added (server/ has `mypy.ini`).
