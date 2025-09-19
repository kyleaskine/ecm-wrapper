# ECM Coordination Middleware

FastAPI-based middleware for coordinating distributed ECM factorization work. This system provides a minimal, focused API for projects to submit numbers, track t-level progress, and retrieve factorization results.

## Quick Start

### Development Setup

1. **Install dependencies:**
   ```bash
   cd server/
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL:**
   ```bash
   # Create database and user
   createdb ecm_distributed
   createuser ecm_user -P  # Enter password: ecm_password
   ```

3. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```

4. **Start the server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the interfaces:**
   - API Documentation: http://localhost:8000/docs
   - Admin Dashboard: http://localhost:8000/api/v1/admin/dashboard
   - User Dashboard: http://localhost:8000/api/v1/dashboard/
   - Health Check: http://localhost:8000/health

### Production Setup

1. **Environment variables:**
   ```bash
   export DATABASE_URL="postgresql://user:pass@host:5432/ecm_distributed"
   export SECRET_KEY="your-secure-secret-key"
   ```

2. **Run with gunicorn:**
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

## API Endpoints

### Core ECM Coordination
- `POST /api/v1/composites/bulk` - Submit numbers for factorization with target t-levels
- `GET /api/v1/work/request` - Request ECM work assignments
- `POST /api/v1/results/ecm` - Submit ECM attempt results and t-level progress
- `GET /api/v1/composites` - List composites and their progress

### Administration
- `GET /api/v1/admin/dashboard` - Admin dashboard interface
- `POST /api/v1/admin/composites/upload` - Bulk upload composites from files
- `POST /api/v1/admin/composites/calculate-t-levels` - Calculate target t-levels
- `GET /api/v1/admin/stats/summary` - System health and statistics

### Monitoring
- `GET /api/v1/dashboard/` - User dashboard interface
- `GET /api/v1/stats/overview` - System overview statistics
- `GET /health` - Health check endpoint

## Minimal Database Schema

The middleware uses PostgreSQL with a simplified schema focused on ECM coordination:
- `composites` - Numbers with t-level progress tracking (id, number, digit_length, target_t_level, current_t_level, is_prime, is_fully_factored, priority)
- `ecm_attempts` - Individual ECM curve attempts with B1/B2 parameters
- `factors` - Discovered factors with discovery methods
- `work_assignments` - Active work assignments to clients
- `clients` - Registered client information and capabilities
- `projects` - Optional project organization

## ECM Coordination Workflow

This middleware provides a simple workflow for any project that needs distributed ECM factorization:

1. **Submit Numbers**: Projects submit composite numbers with optional target t-levels via API or admin interface
2. **Work Assignment**: ECM clients request work and receive optimal B1/B2 parameters for their assigned composites
3. **Progress Tracking**: As clients complete ECM curves, t-level progress is automatically tracked
4. **Factor Discovery**: When factors are found, composites are marked as factored
5. **Result Retrieval**: Projects can monitor progress and retrieve results via API or dashboards

### Example Usage

```bash
# Submit a composite for factorization
curl -X POST http://localhost:8000/api/v1/composites/bulk \
  -H "Content-Type: application/json" \
  -d '[{"number": "12345678901234567890123456789", "target_t_level": 25.0, "priority": 1}]'

# Client requests work
curl http://localhost:8000/api/v1/work/request?client_id=my_client

# Submit ECM results
curl -X POST http://localhost:8000/api/v1/results/ecm \
  -H "Content-Type: application/json" \
  -d '{"work_assignment_id": "abc123", "curves_completed": 100, "b1": 50000, "factors_found": []}'
```

## Configuration

Edit `app/config.py` or use environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `SECRET_KEY` - API security key
- `DEFAULT_WORK_TIMEOUT_MINUTES` - Work assignment timeout