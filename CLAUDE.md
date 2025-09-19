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
```

### Server Development
```bash
# Install server dependencies
cd server/
pip install -r requirements.txt

# Set up local database
createdb ecm_distributed
createuser ecm_user -P  # password: ecm_password

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Full system with Docker
docker-compose up

# Database operations
alembic revision --autogenerate -m "Description"  # Create new migration
alembic upgrade head                               # Apply migrations
alembic downgrade -1                              # Rollback one migration
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
- **client.yaml**: Client configuration (API endpoints, binary paths, default parameters)
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
2. **Work assignment**: Clients request ECM work assignments with optimal B1/B2 parameters
3. **Client execution**: Wrapper scripts execute GMP-ECM/YAFU binaries with assigned parameters
4. **Progress tracking**: T-level progress updated as curves complete
5. **Factor discovery**: Numbers marked as factored when factors found
6. **Result delivery**: Projects retrieve factorization results via API

## Minimal Database Schema

Essential tables for ECM coordination:
- `composites`: Numbers with t-level progress (id, number, digit_length, target_t_level, current_t_level, is_prime, is_fully_factored, priority)
- `ecm_attempts`: Individual ECM curve attempts with B1/B2 parameters
- `factors`: Discovered factors with discovery methods
- `work_assignments`: Active work assignments to clients
- `clients`: Registered client information and capabilities
- `projects`: Optional organizational structure for campaigns

## Binary Dependencies

- **GMP-ECM**: Configure path in client.yaml `programs.gmp_ecm.path`
- **YAFU**: Configure path in client.yaml `programs.yafu.path`
- Both programs must be compiled and accessible on client machines

## Result Parsing Patterns

- **ECM factor extraction**: `r'Factor found in step \d+: (\d+)'` (ecm-wrapper.py:108)
- **YAFU output parsing**: Multiple patterns for P/Q notation and factor formats (yafu-wrapper.py:144-198)
- **Timeout handling**: 1 hour for ECM, 2-4 hours for YAFU operations

## Important File Locations

### Server Structure
- **Main application**: `server/app/main.py` - FastAPI app setup and middleware
- **Configuration**: `server/app/config.py` - Environment settings with Pydantic
- **Database setup**: `server/app/database.py` - SQLAlchemy engine and session
- **Models**: `server/app/models/*.py` - Database table definitions
- **API schemas**: `server/app/schemas/*.py` - Request/response validation
- **Services**: `server/app/services/*.py` - Business logic layer
- **Migrations**: `server/migrations/` - Alembic database migrations

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