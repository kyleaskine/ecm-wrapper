# ECM Coordination Middleware

A lightweight, distributed ECM factorization coordination system with client wrappers for GMP-ECM and YAFU.

## Quick Start

### Run Server with Docker (5 minutes)

```bash
cd server/
cp .env.example .env

# Generate secure keys
echo "DB_PASSWORD=$(openssl rand -hex 32)" >> .env
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env
echo "ADMIN_API_KEY=$(openssl rand -hex 32)" >> .env

# Start services
docker-compose -f docker-compose.simple.yml up -d

# Access admin dashboard
# Open http://localhost:8000/api/v1/admin/login
# Enter your ADMIN_API_KEY from .env
```

### Run Client

```bash
cd client/

# Install dependencies
pip install requests pyyaml

# Configure client (copy example and edit with your settings)
cp client.local.yaml.example client.local.yaml
# Edit client.local.yaml with your API endpoint and binary paths

# Run ECM factorization (standard mode)
python3 ecm-wrapper.py --composite "123456789012345" --curves 100 --b1 50000

# Run ECM with GPU acceleration and two-stage processing
python3 ecm-wrapper.py --composite "123456789012345" --two-stage --curves 1000 --b1 50000

# Run ECM with multiprocess parallelization
python3 ecm-wrapper.py --composite "123456789012345" --multiprocess --workers 4 --curves 1000

# Run ECM targeting t-level (progressive approach)
python3 ecm-wrapper.py --composite "123456789012345" --tlevel 30

# Auto-work mode - continuously get work from server
python3 ecm-wrapper.py --auto-work                          # Use server t-levels
python3 ecm-wrapper.py --auto-work --work-count 5           # Process 5 assignments
python3 ecm-wrapper.py --auto-work --tlevel 35              # Override with client t-level
python3 ecm-wrapper.py --auto-work --b1 50000 --b2 5000000  # Override with B1/B2

# Run YAFU automatic factorization
python3 yafu-wrapper.py --composite "123456789012345" --mode auto

# Run YAFU P-1 factorization
python3 yafu-wrapper.py --composite "123456789012345" --mode pm1 --b1 1000000

# Resend failed API submissions
python3 resend_failed.py --dry-run  # Preview first
python3 resend_failed.py            # Actually resend

# Run tests
pytest tests/ -v
```

## Project Structure

```
ecm-wrapper/
├── server/                      # FastAPI coordination server
│   ├── app/                     # Application code
│   │   ├── api/v1/              # API routes (submit, work, admin, factors)
│   │   ├── models/              # Database models
│   │   ├── schemas/             # Request/response schemas
│   │   ├── services/            # Business logic layer
│   │   ├── templates/           # Jinja2 HTML templates
│   │   └── utils/               # Shared utilities
│   ├── migrations/              # Alembic database migrations
│   ├── bin/                     # Binary utilities (t-level, PARI/GP scripts)
│   ├── tests/                   # Server unit tests
│   ├── docker-compose.simple.yml  # Easy Docker setup
│   ├── DEPLOYMENT.md            # Detailed deployment guide
│   ├── REFACTORING_GUIDE.md     # Server refactoring documentation
│   └── .env.example             # Environment template
│
├── client/                      # Python factorization clients
│   ├── ecm-wrapper.py           # GMP-ECM wrapper (standard, two-stage, multiprocess, t-level)
│   ├── yafu-wrapper.py          # YAFU wrapper (ECM, P-1, P+1, SIQS, NFS, Auto)
│   ├── cado-wrapper.py          # CADO-NFS wrapper (Number Field Sieve)
│   ├── aliquot-wrapper.py       # Aliquot sequence calculator with FactorDB integration
│   ├── base_wrapper.py          # Shared base class for all wrappers
│   ├── config_manager.py        # Configuration loading with deep merge
│   ├── api_client.py            # API communication with retry logic
│   ├── parsing_utils.py         # Output parsing with pre-compiled patterns
│   ├── arg_parser.py            # Unified argument parsing
│   ├── residue_manager.py       # Residue file operations (GPU/CPU formats)
│   ├── result_processor.py      # Factor deduplication and logging
│   ├── stage2_executor.py       # Multi-threaded Stage 2 execution
│   ├── ecm_worker_process.py    # Multiprocess worker coordination
│   ├── resend_failed.py         # Retry failed API submissions
│   ├── client.yaml              # Default configuration (checked into git)
│   ├── client.local.yaml.example # Local config template (machine-specific)
│   ├── tests/                   # Client unit tests (1,100+ lines)
│   ├── scripts/                 # Batch processing scripts
│   │   ├── run_batch.sh         # ECM batch processing
│   │   ├── run_pm1_batch.sh     # GMP-ECM P-1 batch
│   │   └── run_batch_pipeline.py # GPU/CPU pipeline batch processing
│   ├── data/                    # Auto-created data directories
│   │   ├── logs/                # Client logs
│   │   ├── outputs/             # Raw program outputs
│   │   ├── residues/            # ECM residue files
│   │   └── results/             # Failed API submissions
│   ├── README.md                # Client documentation
│   └── CLAUDE.md                # Detailed client guide
│
├── CLAUDE.md                    # Project overview and development guide
└── README.md                    # This file
```

## Features

### Server
- **Work Coordination**: Assign ECM work to distributed clients
- **T-Level Tracking**: Monitor progress toward factorization goals with SNFS difficulty support
- **SNFS Support**: Track special number form status and GNFS-equivalent difficulty
- **Number Tracking**: Maintain link between original number form and current composite as factors are found
- **Group Order Calculation**: Automatic elliptic curve group order computation via PARI/GP for mathematical analysis
- **Admin Dashboard**: Web-based management interface with secure login
- **API Security**: Admin endpoints protected by API key authentication
- **REST API**: Full OpenAPI documentation at `/docs`
- **Automated Deployment**: GitHub Actions integration for production deployment

### Client
- **GMP-ECM Support**: Comprehensive wrapper with multiple execution modes:
  - **Standard Mode**: Run N curves with specified B1/B2 parameters
  - **Two-Stage Mode**: GPU Stage 1 + multi-threaded CPU Stage 2 for optimal performance
  - **Multiprocess Mode**: Multi-core CPU parallelization with worker pools
  - **T-Level Mode**: Progressive ECM targeting specific t-levels (e.g., t30, t35)
  - **Stage 2 Resume**: Continue from existing residue files
- **YAFU Support**: Multi-method factorization (ECM, P-1, P+1, SIQS, NFS, Auto)
- **CADO-NFS Support**: Number Field Sieve for large composite factorization
- **Aliquot Sequences**: Progressive factorization of aliquot sequences with FactorDB integration
- **Multiple Factor Handling**: Properly submits all factors found in single run with individual sigma values
- **Parametrization Tracking**: Tracks ECM parametrization (0-3) for accurate t-level calculations
- **Automatic Factor Recursion**: Fully factors composite results using trial division + ECM
- **Primality Testing**: Miller-Rabin tests to avoid wasting time on primes
- **GPU Acceleration**: Optional GPU support for ECM Stage 1
- **Batch Processing**: Pipeline scripts for automated GPU/CPU processing
- **Result Submission**: Automatic submission to coordination server with retry on failure and multi-endpoint support
- **Failed Submission Recovery**: `resend_failed.py` script to retry failed submissions
- **Comprehensive Testing**: 1,100+ lines of unit tests covering parsing, config, and API

## Documentation

- **Development Guide**: [CLAUDE.md](CLAUDE.md) - Comprehensive guide for Claude Code (client + server development commands, architecture, implementation details)
- **Production Deployment**: [PRODUCTION_DEPLOY.md](PRODUCTION_DEPLOY.md) - Automated deployment with GitHub Actions
- **Server Setup**: [server/DEPLOYMENT.md](server/DEPLOYMENT.md) - Local development and Docker setup
- **Server Refactoring**: [server/REFACTORING_GUIDE.md](server/REFACTORING_GUIDE.md) - Architecture patterns and migration guide
- **Client Quick Reference**: [client/README.md](client/README.md) - Client usage examples and testing
- **API Documentation**: `/docs` endpoint - Interactive OpenAPI documentation when server is running

## Security Features

✅ Admin authentication via API keys
✅ XSS protection with HTML escaping
✅ CORS configured for public API
✅ Environment-based secrets management
✅ Docker secrets support for production

## API Endpoints

### Public (No authentication)
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /api/v1/submit_result` - Submit factorization results
- `GET /api/v1/work` - Get work assignments

### Admin (Requires X-Admin-Key header)
- `GET /api/v1/admin/login` - Admin login page
- `GET /api/v1/admin/dashboard` - Admin dashboard
- `POST /api/v1/admin/composites/upload` - Upload composites
- All other `/api/v1/admin/*` endpoints

## Architecture

```
┌─────────────────────┐    HTTP/API     ┌─────────────────────┐
│   Client (Python)  │◄──────────────►│ ECM Middleware      │
│   • GMP-ECM        │                 │   • Work assignment │
│   • YAFU           │                 │   • T-level tracking│
│   • CADO-NFS       │                 │   • Progress monitor│
│   • Aliquot seqs   │                 │   • Group orders    │
│   • Batch scripts  │                 │   • Admin dashboard │
└─────────────────────┘                 └─────────────────────┘
                                                 │
                                        ┌───────────────────┐
                                        │   PostgreSQL      │
                                        │   • Composites    │
                                        │   • ECM attempts  │
                                        │   • Factors       │
                                        │   • Work queue    │
                                        └───────────────────┘
```

## Recent Improvements

### Client Enhancements (2025-10)
- ✅ **Multi-factor submission**: Properly handles multiple factors found in single ECM run with individual sigmas
- ✅ **Two-stage ECM improvements**: Fixed exit code handling (factor in Stage 1 = success, not failure)
- ✅ **Residue file handling**: Auto-detects GPU (single-line) vs CPU (multi-line) formats
- ✅ **FactorDB integration**: 3 automatic retries with exponential backoff for transient errors
- ✅ **Primality checks**: Miller-Rabin tests after trial division AND after ECM
- ✅ **CADO-NFS failure detection**: Properly detects crashes, avoids submitting partial results
- ✅ **Pipeline batch processing**: Fixed Stage 2 failure handling and timing accuracy

### Server Refactoring (2025-10)
- ✅ **Unified service architecture**: Dependency injection throughout (no module-level singletons)
- ✅ **Centralized error handling**: Consistent error response patterns
- ✅ **Calculation utilities**: Eliminated 300-400 lines of duplicate code
- ✅ **Group order calculation**: Automatic elliptic curve group order via PARI/GP
- ✅ **Admin dashboard improvements**: Auto-refresh, deduplicated factors, multi-factor indicators

### Testing Infrastructure
- **Client tests**: 1,100+ lines covering factorization parsing, config management, API payloads
- **Server tests**: Unit tests for number utilities, transaction handling, business logic
- Run with: `pytest tests/ -v`

## Requirements

### Server
- **Docker & Docker Compose** (recommended for quick start)
- **OR Manual Setup**: Python 3.11+, PostgreSQL 15+, PARI/GP (for group order calculations)
- **Development**: pytest (for running tests)

### Client
- **Python 3.8+**
- **Required packages**: `requests`, `pyyaml` (install via: `pip install requests pyyaml`)
- **Factorization binaries** (configure paths in `client.local.yaml`):
  - GMP-ECM binary (for `ecm-wrapper.py`) - Required
  - YAFU binary (for `yafu-wrapper.py`) - Optional
  - CADO-NFS (for `cado-wrapper.py` and `aliquot-wrapper.py`) - Optional
- **Testing**: pytest (install via: `pip install pytest`)

## Development

### Running Tests
```bash
# Client tests
cd client/
pytest tests/ -v

# Server tests (requires database)
cd server/
source venv/bin/activate
pytest tests/ --ignore=tests/test_transactions.py -v
```

### Code Quality
The codebase follows these principles:
- **DRY (Don't Repeat Yourself)**: Recent refactoring eliminated 300-400 lines of duplication
- **Dependency Injection**: Server uses DI throughout for testability
- **Type Hints**: Comprehensive typing for better IDE support
- **Pre-compiled Patterns**: Regex patterns compiled once for performance
- **Comprehensive Error Handling**: Retry logic with exponential backoff

### Potential Refactoring Opportunities
See architectural analysis in project for details:
1. **Extract T-Level Calculator** (ecm-wrapper.py lines 951-1120) into `tlevel_calculator.py`
2. **Extract Factorization Utilities** (lines 776-950) into `factorization_utils.py`
3. **Consider Renaming** hyphenated files (`ecm-wrapper.py` → `ecm_wrapper.py`) for standard Python imports

## Support

- **GitHub Issues**: Report bugs or request features at the project repository
- **API Documentation**: `/docs` endpoint when server is running - Interactive OpenAPI docs
- **Deployment Help**: See `server/DEPLOYMENT.md` for detailed setup instructions
- **Client Guide**: See `client/CLAUDE.md` for comprehensive client usage
- **Development Guide**: See root `CLAUDE.md` for development commands and architecture