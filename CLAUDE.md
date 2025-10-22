# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
python3 client/ecm-wrapper.py --composite "123456789012345" --curves 100 --b1 50000

# Run YAFU factorization (various modes)
python3 client/yafu-wrapper.py --composite "123456789012345" --mode ecm --curves 100
python3 client/yafu-wrapper.py --composite "123456789012345" --mode pm1 --b1 1000000
python3 client/yafu-wrapper.py --composite "123456789012345" --mode auto

# Test without API submission
python3 client/ecm-wrapper.py --composite "123456789012345" --no-submit

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

# Start PostgreSQL (uses existing data volume)
docker-compose -f docker-compose.dev.yml up -d postgres

# Start API server
source venv/bin/activate
export DATABASE_URL="postgresql://ecm_user:ecm_password@localhost:5434/ecm_distributed"
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

**Alternative Setup (Local PostgreSQL)**
```bash
# Set up local database on default port 5432
createdb ecm_distributed
createuser ecm_user -P  # password: ecm_password

# Start with local PostgreSQL
export DATABASE_URL="postgresql://ecm_user:ecm_password@localhost:5432/ecm_distributed"
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
```

### Server Refactoring Documentation

**IMPORTANT:** The server codebase underwent significant refactoring (Phase 1 & 2) on 2025-10-21:

- **[REFACTORING_GUIDE.md](./server/REFACTORING_GUIDE.md)** - Complete migration guide with examples
- **[REFACTORING_QUICK_REFERENCE.md](./server/REFACTORING_QUICK_REFERENCE.md)** - Quick lookup for new patterns

**Key changes:**
- ✅ Unified service architecture with dependency injection
- ✅ Centralized error handling utilities
- ✅ Centralized calculation utilities
- ✅ Eliminated 300-400 lines of duplicate code
- ✅ All routes use dependency injection (no module-level singletons)

**If you're adding new routes or services, follow the patterns in the refactoring guide.**

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
- **ECMWrapper** (client/ecm-wrapper.py): GMP-ECM process management and result parsing
- **YAFUWrapper** (client/yafu-wrapper.py): YAFU multi-method factorization coordination
- **API Server** (server/app/main.py): FastAPI middleware with coordination endpoints
- **Database Models** (server/app/models/): Minimal schema focused on ECM coordination
- **API Routes** (server/app/api/v1/): RESTful endpoints for work assignment and results
- **T-Level Services** (server/app/services/): T-level calculation and progress tracking

### Configuration System
- **client.yaml**: Default client configuration (API endpoints, binary paths, default parameters)
  - Contains sensible defaults that work out of the box
  - Checked into git for version control
- **client.local.yaml**: Local overrides for client.yaml (gitignored, machine-specific settings)
  - Deep merges with client.yaml (local settings override defaults)
  - Use `client.local.yaml.example` as template
  - Auto-detected by BaseWrapper and arg_parser
  - **Important**: Always pass `client.yaml` as config path - BaseWrapper handles the merge
- **resend_failed.py**: Inherits from BaseWrapper to reuse config loading logic
- **server/app/config.py**: Server configuration (database URL, API settings)
- **docker-compose.yml**: Full system deployment configuration
- **alembic.ini**: Database migration configuration

### Middleware Architecture Layers
```
┌─────────────────────────────────────────────────────────────┐
│                   ECM Coordination API                     │
├─────────────────────────────────────────────────────────────┤
│                      Core Endpoints                        │
│   • /composites   • /results/ecm   • /work   • /admin      │
├─────────────────────────────────────────────────────────────┤
│                   Minimal Services                         │
│   • WorkAssignment   • TLevelCalculation   • Dashboard     │
├─────────────────────────────────────────────────────────────┤
│                   Simplified Models                        │
│   • Composites   • ECMAttempts   • Factors   • Clients     │
├─────────────────────────────────────────────────────────────┤
│                    PostgreSQL Database                     │
└─────────────────────────────────────────────────────────────┘
```

## ECM Coordination Workflow

1. **Project submits numbers**: Upload composites with target t-levels via API or admin interface
   - Bulk upload via CSV/text file (`/admin/composites/upload`)
   - Structured upload with metadata (`/admin/composites/bulk-structured`)
   - Can update existing composites: `current_composite`, `priority`, `is_fully_factored`, `is_prime`, `has_snfs_form`, `snfs_difficulty`
2. **Work assignment**: Clients request ECM work assignments with optimal B1/B2 parameters
3. **Client execution**: Wrapper scripts execute GMP-ECM/YAFU binaries with assigned parameters
4. **Progress tracking**: T-level progress updated as curves complete via `/submit_result`
5. **Factor discovery**: Numbers marked as factored when factors found
   - **Group order calculation**: Elliptic curve group orders automatically calculated via PARI/GP
   - Supports all parametrizations (0, 1, 2, 3) with proper curve construction
   - Prime factorization of group order computed for mathematical analysis
6. **Result delivery**: Projects retrieve factorization results via API
7. **Manual curve submission**: Upload ECM curves via `/submit_result` endpoint with full metadata

## Minimal Database Schema

Essential tables for ECM coordination:
- `composites`: Numbers with t-level progress (id, number, digit_length, target_t_level, current_t_level, is_prime, is_fully_factored, priority)
- `ecm_attempts`: Individual ECM curve attempts with B1/B2 parameters and parametrization (0-3)
- `factors`: Discovered factors with discovery methods, sigma values, and elliptic curve group orders
  - `sigma`: Sigma value that found this factor (for reproducibility)
  - `group_order`: Calculated elliptic curve group order (via PARI/GP)
  - `group_order_factorization`: Prime factorization of the group order
- `work_assignments`: Active work assignments to clients
- `clients`: Registered client information and capabilities
- `projects`: Optional organizational structure for campaigns

### Key Schema Updates
- **ecm_attempts.parametrization**: ECM parametrization type (0, 1, 2, or 3) - affects t-level calculations
  - Parametrization 1: Montgomery curves (CPU default)
  - Parametrization 3: Twisted Edwards curves (GPU default)
- **factors.sigma**: Sigma value that found the factor (for reproducibility)
- **ecm_attempts.b2**: Can be NULL (use GMP-ECM default) or 0 (stage 1 only)

## Binary Dependencies

### Client Dependencies
- **GMP-ECM**: Configure path in client.yaml `programs.gmp_ecm.path`
- **YAFU**: Configure path in client.yaml `programs.yafu.path`
- Both programs must be compiled and accessible on client machines

### Server Dependencies
- **t-level binary**: Deployed to `server/bin/t-level` for t-level calculations
- **PARI/GP**: Installed in Docker container for elliptic curve group order calculations
- **group.gp script**: PARI/GP script deployed to `server/bin/group.gp` for FindGroupOrder function

## Result Parsing Patterns

- **ECM factor extraction**: `r'Factor found in step \d+: (\d+)'` (ecm-wrapper.py:108)
- **YAFU output parsing**: Multiple patterns for P/Q notation and factor formats (yafu-wrapper.py:144-198)
- **Timeout handling**: 1 hour for ECM, 2-4 hours for YAFU operations

## Recent Bug Fixes and Improvements

### Two-Stage ECM Improvements
- **Exit code handling**: Fixed stage 1 to treat factor discovery (exit code 8) as success, not failure
- **B2 parameter accuracy**: When factor found in stage 1, now correctly submits with `b2=0` (stage 2 never ran)
- **Timing accuracy**: Pipeline now submits combined stage1 + stage2 execution time

### Residue File Handling
- **GPU format support**: `residue_manager.py` now auto-detects and handles both GPU (single-line) and CPU (multi-line) residue file formats
- **Format detection**: Checks first 5 lines for `METHOD=ECM; SIGMA=...; ...` pattern to identify GPU format

### FactorDB Integration (aliquot-wrapper.py)
- **Retry logic**: 3 automatic retries with exponential backoff (1s, 2s) for transient server errors (502, etc.)
- **Enhanced logging**: All FactorDB operations logged to `client/data/logs/ecm_client.log` with:
  - Success/failure status for each factor submission
  - Partial failure tracking (some factors succeed, others fail)
  - Retry attempt logging with countdowns
- **View logs**: `grep "FactorDB" client/data/logs/ecm_client.log`

### Aliquot Sequence Factorization
- **Primality checks**: Added Miller-Rabin primality tests after trial division AND after ECM
- **CADO-NFS failure detection**: Now properly detects when CADO crashes and stops sequence instead of submitting partial results
- **Early termination**: Avoids wasting compute on ECM/CADO when cofactor is already prime

### Pipeline Batch Processing
- **Failure handling**: No longer submits results when stage 2 fails (e.g., residue file split errors)
- **No false failures**: Fixed detection logic - `None` return from stage 2 means "no factor found" (success), not failure

### Server Dashboard Improvements
- **Group order display**: Composite details page now shows elliptic curve group order data for factors
  - Shows: Factor, Sigma, Group Order, Group Order Factorization
  - Only displayed when group order information is available (requires sigma and parametrization)
- **Deduplicated factors**: Work summary now shows unique factors sorted numerically
  - Same factor appearing in multiple attempts now shown only once

## Important File Locations

### Server Structure
- **Main application**: `server/app/main.py` - FastAPI app setup and middleware
- **Configuration**: `server/app/config.py` - Environment settings with Pydantic
- **Database setup**: `server/app/database.py` - SQLAlchemy engine and session
- **Models**: `server/app/models/*.py` - Database table definitions
- **API schemas**: `server/app/schemas/*.py` - Request/response validation
- **API routes**: `server/app/api/v1/*.py` - Core API endpoints (submit, work, stats, factors)
- **Admin routes**: `server/app/api/v1/admin/*.py` - Modular admin endpoints
  - `dashboard.py` - Admin dashboard and summary stats
  - `composites.py` - Composite upload, bulk operations, CRUD
  - `work.py` - Work assignment management
  - `projects.py` - Project organization
  - `maintenance.py` - T-level recalculation utilities
- **Services**: `server/app/services/*.py` - Business logic layer
  - `composite_manager.py` - Composite CRUD, bulk loading, updates
  - `composites.py` - Core composite operations
  - `factors.py` - Factor validation and management
  - `t_level_calculator.py` - ECM t-level calculations
  - `group_order.py` - Elliptic curve group order calculation using PARI/GP
- **Utilities**: `server/app/utils/*.py` - Shared utilities
  - `serializers.py` - Database model to API dict conversion
  - `query_helpers.py` - Reusable database query patterns
  - `html_helpers.py` - Template formatting and HTML escaping
- **Templates**: `server/app/templates/` - Jinja2 HTML templates
  - `base.html` - Shared CSS and layout
  - `admin/` - Admin dashboard templates
  - `public/` - Public dashboard templates
  - `components/` - Reusable UI components
- **Migrations**: `server/migrations/` - Alembic database migrations

### Security Features
- **Admin authentication**: All admin endpoints require API key via `X-Admin-Key` header
- **Timing attack protection**: Constant-time key comparison using `secrets.compare_digest()`
- **File upload limits**: 10 MB maximum file size on bulk upload endpoints
- **Input validation**: UTF-8 encoding validation, Pydantic schema validation
- **Error sanitization**: Generic error messages to clients, detailed logging server-side
- **SQL injection protection**: SQLAlchemy ORM with parameterized queries
- **XSS protection**: Jinja2 auto-escaping with explicit `esc()` for HTML output

### Client Structure
- **Main wrappers**: `client/ecm-wrapper.py`, `client/yafu-wrapper.py`
- **Configuration**: `client/client.yaml` - Binary paths and API settings
- **Base classes**: `client/base_wrapper.py` - Shared wrapper functionality
- **Utilities**: `client/parsing_utils.py`, `client/arg_parser.py`
- **Batch scripts**: `client/scripts/` - Automated processing workflows

### Database Connection
- **Default URL**: `postgresql://ecm_user:ecm_password@localhost:5432/ecm_distributed`
- **Docker port**: PostgreSQL exposed on port 5434 (host) → 5432 (container)
- **Environment**: Set `DATABASE_URL` to override default connection string