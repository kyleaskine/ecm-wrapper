#!/bin/bash

# Digital Ocean Droplet Setup Script for ECM Distributed Server
# Run this script on a fresh Ubuntu 22.04 droplet

set -e

echo "🐧 Setting up ECM Distributed Server on Digital Ocean Droplet"

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $USER

# Install Docker Compose
echo "🐙 Installing Docker Compose..."
apt install -y docker-compose-plugin

# Install other essentials
echo "🛠️  Installing essential tools..."
apt install -y git nginx certbot python3-certbot-nginx ufw htop curl wget

# Configure firewall
echo "🔥 Configuring firewall..."
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

# Create deployment directory
echo "📁 Creating deployment directory..."
mkdir -p /opt/ecm-distributed
chown $USER:$USER /opt/ecm-distributed

# Clone repository (you'll need to update this URL)
echo "📥 Cloning repository..."
cd /opt/ecm-distributed
git clone https://github.com/YOUR_USERNAME/ecm-wrapper.git .

# Set up SSL with Let's Encrypt (requires domain to be pointed to droplet)
read -p "Enter your domain name (e.g., ecm.yourdomain.com): " DOMAIN
if [ ! -z "$DOMAIN" ]; then
    echo "🔒 Setting up SSL certificate for $DOMAIN..."
    certbot --nginx -d $DOMAIN --non-interactive --agree-tos -m admin@$DOMAIN
    
    # Update nginx config with actual domain
    sed -i "s/your-domain.com/$DOMAIN/g" server/nginx.conf
fi

# Create systemd service for auto-start
echo "⚙️  Creating systemd service..."
cat > /etc/systemd/system/ecm-distributed.service << EOF
[Unit]
Description=ECM Distributed Factorization Server
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ecm-distributed/server
ExecStart=/usr/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl enable ecm-distributed.service

# Set up log rotation
echo "📜 Setting up log rotation..."
cat > /etc/logrotate.d/ecm-distributed << EOF
/opt/ecm-distributed/server/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF

# Create deployment user
echo "👤 Creating deployment user..."
adduser --system --group --shell /bin/bash deploy
usermod -aG docker deploy
chown -R deploy:deploy /opt/ecm-distributed

# Generate SSH key for GitHub Actions (display public key)
echo "🔑 Generating SSH key for deployment..."
sudo -u deploy ssh-keygen -t ed25519 -f /home/deploy/.ssh/id_ed25519 -N ""
echo "
📋 Add this public key to your server's authorized_keys for GitHub Actions:
"
cat /home/deploy/.ssh/id_ed25519.pub
echo "

📝 GitHub Secrets to configure:
- DEPLOY_SSH_KEY: (the private key above)
- SERVER_HOST: $(curl -s ifconfig.me)
- SERVER_USER: deploy
- DOMAIN_NAME: $DOMAIN

🎯 Next steps:
1. Point your domain DNS to: $(curl -s ifconfig.me)
2. Configure GitHub secrets
3. Push code to trigger deployment
4. Access dashboard at: https://$DOMAIN/api/v1/dashboard/
"

echo "✅ Droplet setup complete!"