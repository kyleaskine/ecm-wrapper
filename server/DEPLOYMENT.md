# ECM Coordination Server - Deployment Guide

> **For Production Deployment**: See [../PRODUCTION_DEPLOY.md](../PRODUCTION_DEPLOY.md) for automated GitHub Actions deployment to Digital Ocean.

## Quick Start with Docker (Recommended - Local/Development)

### Prerequisites
- Docker and Docker Compose installed
- 2GB+ RAM
- 10GB+ disk space

### Setup Steps

1. **Clone the repository and navigate to server directory**
   ```bash
   cd server/
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   ```

3. **Generate secure secrets**
   ```bash
   # Generate strong passwords/keys (Linux/Mac)
   echo "DB_PASSWORD=$(openssl rand -hex 32)" >> .env
   echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env
   echo "ADMIN_API_KEY=$(openssl rand -hex 32)" >> .env

   # Or manually edit .env and replace the placeholder values
   ```

4. **Start the services**
   ```bash
   docker-compose -f docker-compose.simple.yml up -d
   ```

5. **Verify deployment**
   ```bash
   # Check service status
   docker-compose -f docker-compose.simple.yml ps

   # View logs
   docker-compose -f docker-compose.simple.yml logs -f api

   # Test health endpoint
   curl http://localhost:8000/health
   ```

6. **Access the admin dashboard**
   - Open browser to `http://localhost:8000/api/v1/admin/login`
   - Enter your `ADMIN_API_KEY` from the `.env` file
   - You'll be redirected to the dashboard
   - Key is stored in browser sessionStorage for the session
   - Use the "Logout" button in the top-right to clear the session

### Management Commands

```bash
# Stop services
docker-compose -f docker-compose.simple.yml stop

# Start services
docker-compose -f docker-compose.simple.yml start

# Restart services
docker-compose -f docker-compose.simple.yml restart

# View logs
docker-compose -f docker-compose.simple.yml logs -f

# Stop and remove containers (data persists in volume)
docker-compose -f docker-compose.simple.yml down

# Stop and REMOVE ALL DATA (including database)
docker-compose -f docker-compose.simple.yml down -v
```

### Updating to New Version

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose -f docker-compose.simple.yml up -d --build

# Database migrations run automatically on startup
```

---

## Standalone Installation (Without Docker)

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- GCC compiler (for some Python packages)

### Setup Steps

1. **Install PostgreSQL**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install postgresql postgresql-contrib

   # macOS
   brew install postgresql@15
   ```

2. **Create database and user**
   ```bash
   sudo -u postgres psql
   ```

   In PostgreSQL shell:
   ```sql
   CREATE DATABASE ecm_distributed;
   CREATE USER ecm_user WITH PASSWORD 'your_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE ecm_distributed TO ecm_user;
   \q
   ```

3. **Set up Python environment**
   ```bash
   cd server/
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials and secrets
   ```

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the server**
   ```bash
   # Use 2 workers to keep the server responsive during heavy background operations
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
   ```

7. **Access the dashboard**
   - Open `http://localhost:8000/api/v1/admin/login`
   - Enter your `ADMIN_API_KEY` from the `.env` file
   - You'll be redirected to the dashboard

---

## Production Deployment

### Recommended Configuration

For production deployment, use `docker-compose.prod.yml` which includes:
- Secret management via Docker secrets
- NGINX reverse proxy
- Health checks
- Automatic restarts

### Security Checklist

- [ ] Use strong, randomly-generated passwords (32+ characters)
- [ ] Set `ADMIN_API_KEY` to a secure random value
- [ ] Set `SECRET_KEY` to a secure random value
- [ ] Use HTTPS with valid SSL certificate (via NGINX/Caddy/Traefik)
- [ ] Configure firewall to only allow necessary ports
- [ ] Regularly update Docker images and system packages
- [ ] Enable PostgreSQL SSL connections for production
- [ ] Set up automated backups of PostgreSQL volume
- [ ] Monitor logs for suspicious activity
- [ ] Consider rate limiting at reverse proxy level

### Backup and Restore

**Backup database:**
```bash
docker-compose -f docker-compose.simple.yml exec postgres \
  pg_dump -U ecm_user ecm_distributed > backup_$(date +%Y%m%d).sql
```

**Restore database:**
```bash
docker-compose -f docker-compose.simple.yml exec -T postgres \
  psql -U ecm_user ecm_distributed < backup_20240101.sql
```

---

## API Endpoints

### Public Endpoints (No authentication required)
- `GET /health` - Health check
- `GET /` - API information
- `GET /docs` - Interactive API documentation
- `POST /api/v1/submit_result` - Submit factorization results
- `GET /api/v1/ecm-work` - Get an ECM work assignment
- `GET /api/v1/p1-work` - Get a P-1/P+1 work assignment
- All work management endpoints (`/api/v1/work/*`)

### Admin Endpoints (Require X-Admin-Key header)
- `GET /api/v1/admin/dashboard` - Admin web dashboard
- `GET /api/v1/admin/composites/status` - Queue status
- `POST /api/v1/admin/composites/upload` - Upload composite numbers
- `POST /api/v1/admin/composites/bulk` - Bulk add composites
- `DELETE /api/v1/admin/composites/{id}` - Remove composite
- All other `/api/v1/admin/*` endpoints

### Using Admin API with curl

```bash
# Set your admin key
export ADMIN_KEY="your-admin-api-key-here"

# Get queue status
curl -H "X-Admin-Key: $ADMIN_KEY" http://localhost:8000/api/v1/admin/composites/status

# Upload composites
curl -X POST -H "X-Admin-Key: $ADMIN_KEY" \
  -F "file=@composites.txt" \
  http://localhost:8000/api/v1/admin/composites/upload
```

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose -f docker-compose.simple.yml logs

# Check if port 8000 is already in use
sudo lsof -i :8000
```

### Database connection errors
```bash
# Verify PostgreSQL is running
docker-compose -f docker-compose.simple.yml ps postgres

# Check PostgreSQL logs
docker-compose -f docker-compose.simple.yml logs postgres

# Test database connection
docker-compose -f docker-compose.simple.yml exec postgres \
  psql -U ecm_user -d ecm_distributed -c "SELECT 1;"
```

### Admin dashboard returns 401 Unauthorized
- Verify your `ADMIN_API_KEY` matches what's set in `.env`
- Check browser console for errors
- Clear browser sessionStorage and try again
- Restart API container after changing `.env`

### Performance issues
- Increase PostgreSQL shared_buffers if handling many composites
- Monitor container resource usage: `docker stats`
- Consider vertical scaling (more RAM/CPU) or external PostgreSQL

---

## Support

For issues or questions:
- Check existing GitHub issues
- Review API documentation at `/docs`
- Enable debug logging: set `LOG_LEVEL=DEBUG` in `.env`