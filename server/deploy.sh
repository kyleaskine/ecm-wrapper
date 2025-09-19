#!/bin/bash

# ECM Distributed Server Deployment Script
# Usage: ./deploy.sh [production|staging]

set -e

ENVIRONMENT=${1:-production}
DOMAIN=${2:-your-domain.com}

echo "🚀 Deploying ECM Distributed Server to $ENVIRONMENT"

# Check if running as root or with docker permissions
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Error: Cannot access Docker. Run with sudo or add user to docker group."
    exit 1
fi

# Create secrets directory if it doesn't exist
mkdir -p secrets

# Generate secrets if they don't exist
if [ ! -f secrets/postgres_password.txt ]; then
    echo "🔐 Generating PostgreSQL password..."
    openssl rand -base64 32 > secrets/postgres_password.txt
    chmod 600 secrets/postgres_password.txt
fi

if [ ! -f secrets/api_secret_key.txt ]; then
    echo "🔐 Generating API secret key..."
    openssl rand -base64 64 > secrets/api_secret_key.txt
    chmod 600 secrets/api_secret_key.txt
fi

# Update domain in nginx config
echo "🌐 Configuring domain: $DOMAIN"
sed -i "s/your-domain.com/$DOMAIN/g" nginx.conf
sed -i "s/your-domain.com/$DOMAIN/g" docker-compose.prod.yml

# Pull latest images
echo "📦 Pulling latest Docker images..."
docker-compose -f docker-compose.prod.yml pull

# Stop existing services
echo "🛑 Stopping existing services..."
docker-compose -f docker-compose.prod.yml down

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose -f docker-compose.prod.yml up -d --build

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🩺 Checking service health..."
if curl -f http://localhost/health >/dev/null 2>&1; then
    echo "✅ API server is healthy"
else
    echo "❌ API server health check failed"
    docker-compose -f docker-compose.prod.yml logs api
    exit 1
fi

# Display status
echo "📊 Service status:"
docker-compose -f docker-compose.prod.yml ps

echo "🎉 Deployment complete!"
echo "📱 Dashboard: https://$DOMAIN/api/v1/dashboard/"
echo "📚 API Docs: https://$DOMAIN/docs"
echo "🔍 Health Check: https://$DOMAIN/health"

# Show logs
echo "📝 Recent logs:"
docker-compose -f docker-compose.prod.yml logs --tail=50