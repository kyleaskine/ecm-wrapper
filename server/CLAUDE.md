# Server CLAUDE.md

Implementation details for the ECM coordination server. See also the [root CLAUDE.md](../CLAUDE.md) for project overview and dev commands.

## Refactoring Documentation

**IMPORTANT:** The server codebase underwent significant refactoring (Phase 1 & 2) on 2025-10-21:

- **[REFACTORING_GUIDE.md](./REFACTORING_GUIDE.md)** - Complete migration guide with examples
- **[REFACTORING_QUICK_REFERENCE.md](./REFACTORING_QUICK_REFERENCE.md)** - Quick lookup for new patterns

**Key changes:**
- Unified service architecture with dependency injection
- Centralized error handling and calculation utilities
- All routes use dependency injection (no module-level singletons)

**If you're adding new routes or services, follow the patterns in the refactoring guide.**

## Server Structure

- **Main application**: `app/main.py` - FastAPI app setup and middleware
- **Configuration**: `app/config.py` - Environment settings with Pydantic
  - `RESIDUE_STORAGE_PATH`: Directory for storing residue files (default: `data/residues`)
- **Database setup**: `app/database.py` - SQLAlchemy engine and session
- **Models**: `app/models/*.py` - Database table definitions
- **API schemas**: `app/schemas/*.py` - Request/response validation
- **API routes**: `app/api/v1/*.py` - Core API endpoints (submit, work, stats, factors, residues)
- **Admin routes**: `app/api/v1/admin/*.py` - Modular admin endpoints
  - `dashboard.py` - Admin dashboard and summary stats
  - `composites.py` - Composite upload, bulk operations, CRUD
  - `work.py` - Work assignment management
  - `projects.py` - Project organization
  - `maintenance.py` - T-level recalculation utilities
- **Services**: `app/services/*.py` - Business logic layer
  - `composite_manager.py` - Composite CRUD, bulk loading, updates
  - `composites.py` - Core composite operations
  - `factors.py` - Factor validation and management
  - `t_level_calculator.py` - ECM t-level calculations (excludes superseded attempts)
  - `residue_manager.py` - Residue file storage, parsing, and lifecycle management
  - `group_order.py` - Elliptic curve group order calculation using PARI/GP
- **Utilities**: `app/utils/*.py` - Shared utilities
  - `serializers.py` - Database model to API dict conversion
  - `query_helpers.py` - Reusable database query patterns
  - `html_helpers.py` - Template formatting and HTML escaping
- **Templates**: `app/templates/` - Jinja2 HTML templates
  - `base.html` - Shared CSS and layout
  - `admin/` - Admin dashboard templates
  - `public/` - Public dashboard templates
  - `components/` - Reusable UI components
- **Migrations**: `migrations/` - Alembic database migrations

### Middleware Architecture Layers
```
┌─────────────────────────────────────────────────────────────┐
│                   ECM Coordination API                     │
├─────────────────────────────────────────────────────────────┤
│                      Core Endpoints                        │
│  /composites  /submit  /work  /residues  /admin  /stats    │
├─────────────────────────────────────────────────────────────┤
│                   Minimal Services                         │
│  WorkAssignment  TLevelCalculation  ResidueManager  Dash   │
├─────────────────────────────────────────────────────────────┤
│                   Simplified Models                        │
│  Composites  ECMAttempts  Factors  ECMResidues  Clients    │
├─────────────────────────────────────────────────────────────┤
│                    PostgreSQL Database                     │
└─────────────────────────────────────────────────────────────┘
```

## Database Schema

Essential tables for ECM coordination:

### composites
Numbers with t-level progress: `id`, `number`, `digit_length`, `target_t_level`, `current_t_level`, `prior_t_level`, `is_prime`, `is_fully_factored`, `priority`
- `prior_t_level`: Work done before import (starting point via `-w` flag)
- `current_t_level`: True combined t-level from prior + new work
- T-levels are logarithmic/probabilistic, not additive (t40 + t40 ≈ t41, not t80)

### ecm_attempts
Individual ECM curve attempts with B1/B2 parameters and parametrization (0-3)
- `parametrization`: 1 = Montgomery (CPU default), 3 = Twisted Edwards (GPU default)
- `superseded_by`: Links stage 1 attempt to stage 2 that replaced it
- `b2`: Can be NULL (GMP-ECM default) or 0 (stage 1 only)

### ecm_residues
Stage 1 residue files for decoupled two-stage ECM:
- `composite_id`, `stage1_attempt_id`: Links to composite and stage 1 attempt
- `b1`, `parametrization`, `curve_count`: Parsed from residue file
- `storage_path`, `file_size_bytes`, `checksum`: File storage metadata
- `status`: 'available', 'claimed', 'completed', 'expired'
- `claimed_by`, `claimed_at`, `expires_at`: Lifecycle tracking

### factors
Discovered factors with discovery methods:
- `sigma`: Sigma value that found this factor (for reproducibility)
- `group_order`: Calculated elliptic curve group order (via PARI/GP)
- `group_order_factorization`: Prime factorization of the group order

### Other tables
- `work_assignments`: Active work assignments to clients
- `clients`: Registered client information and capabilities
- `projects`: Optional organizational structure for campaigns

## Security Features

- **Admin authentication**: All admin endpoints require API key via `X-Admin-Key` header
- **Timing attack protection**: Constant-time key comparison using `secrets.compare_digest()`
- **Submission validation**: Only composites already in database can receive ECM results
- **File upload limits**: 10 MB maximum on bulk upload endpoints
- **Input validation**: UTF-8 encoding validation, Pydantic schema validation
- **Error sanitization**: Generic error messages to clients, detailed logging server-side
- **SQL injection protection**: SQLAlchemy ORM with parameterized queries
- **XSS protection**: Jinja2 auto-escaping with explicit `esc()` for HTML output

## Server API Endpoints

### Composite Management
- Bulk upload via CSV/text file (`/admin/composites/upload`)
- Structured upload with metadata (`/admin/composites/bulk-structured`)
- Can update existing composites: `current_composite`, `priority`, `is_fully_factored`, `is_prime`, `has_snfs_form`, `snfs_difficulty`

### Work Assignment
- `GET /ecm-work` - Request ECM work assignment
- `POST /work/{work_id}/complete` - Mark work complete
- `DELETE /work/{work_id}` - Release/abandon work

### P-1/P+1 Work (2026-02)
- `GET /p1-work` in `app/api/v1/ecm_work.py`
  - Query params: `client_id`, `method` (pm1/pp1/p1), `priority`, `min_target_tlevel`, `max_target_tlevel`, `work_type`
  - Returns: `work_id`, `composite`, `pm1_b1`, `pp1_b1`, etc.
- Server calculates B1 one step above target t-level
- Filters out composites that already have PM1/PP1 at the required B1 level
  - Uses SQL `NOT EXISTS` subqueries against `ecm_attempts` (hits composite_id,method index)
  - No `ecm_progress < 1.0` filter — PM1/PP1 valuable even after ECM target reached
- B1 lookup: `get_b1_above_tlevel()` in `app/constants.py`

### Residue Endpoints (`app/api/v1/residues.py`)
- `POST /residues/upload` - Upload stage 1 residue file
- `GET /residues/work` - Request stage 2 work
- `GET /residues/{id}/download` - Download residue file
- `POST /residues/{id}/complete` - Mark complete, supersede stage 1
- `DELETE /residues/{id}/claim` - Release residue claim

### Multi-Factor Batch Submission
- `factors_found` list in `ResultsSchema` (`app/schemas/submit.py`)
- Server processes all factors BEFORE updating composite state
- Factors divided out sequentially from a running cofactor
- Skips factors that don't divide (handles composite factors gracefully)

## Recent Server Bug Fixes

### Server-Side Residue Validation (2026-01)
- Factor found: Always accepted
- No factor: Must complete at least 75% of assigned curves
- Invalid completions: rejected (400) with the claim left in place; it
  expires via `cleanup_expired_claims` rather than being released in the same
  transaction (the rejection rolls that transaction back, so an in-band
  release would be undone while the response claimed success)
- Implementation: `app/services/residue_manager.py:complete_residue()`

### Testing Status Page Improvements
- Server-side pagination: 200 per page, ~0.5s load (was 10+ seconds)
- Server-side filtering: T-level range, priority threshold, SNFS difficulty
- Eliminated N+1 query problem with aggregated GROUP BY query

### Dashboard Improvements
- Group order display on composite details page
- Deduplicated factors in work summary (sorted numerically)
- Multi-factor indicators: `[+N more]` badge in attempts tables
- Delete button with confirmation on admin composite details
- Admin dashboard auto-refreshes every 30 seconds
