FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY alembic.ini .
COPY migrations/ migrations/

# Create bin directory and copy t-level executable
RUN mkdir -p /app/bin
COPY bin/t-level /app/bin/t-level
RUN chmod +x /app/bin/t-level

# Run database migrations on startup
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]