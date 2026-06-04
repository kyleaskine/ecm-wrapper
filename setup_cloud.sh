#!/bin/bash
# ECM Client Setup Script for Cloud Instances (vast.ai, etc.)
# Usage: curl -sSL https://ecm.kyleaskine.com/downloads/setup_cloud.sh | bash
#    or: wget -qO- https://ecm.kyleaskine.com/downloads/setup_cloud.sh | bash

set -e  # Exit on error

echo "============================================================"
echo "  ECM Factorization Client - Cloud Instance Setup"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Check/Install Dependencies
# ============================================================
echo "📦 Checking dependencies..."

# Check if running as root or with sudo access
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Install git if needed
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    $SUDO apt-get update -qq
    $SUDO apt-get install -y git
fi

# Install python3 and pip if needed
if ! command -v python3 &> /dev/null; then
    echo "Installing python3..."
    $SUDO apt-get install -y python3 python3-pip
fi

echo "✓ Dependencies ready"

# ============================================================
# Step 2: User Configuration
# ============================================================
echo ""
echo "============================================================"
read -p "📝 Enter your username: " USERNAME
read -p "🖥️  Enter machine name (optional, default: $(hostname)): " MACHINE_NAME
MACHINE_NAME=${MACHINE_NAME:-$(hostname)}

read -p "⭐ Enter priority filter (default: 5): " PRIORITY_VALUE
PRIORITY_VALUE=${PRIORITY_VALUE:-5}

API_ENDPOINT="https://ecm.kyleaskine.com/api/v1"
echo "🌐 Using API endpoint: $API_ENDPOINT"
echo "============================================================"
echo ""

# ============================================================
# Step 3: Setup Directory
# ============================================================
INSTALL_DIR="$HOME/ecm-wrapper"
echo "📁 Setting up in: $INSTALL_DIR"

if [ -d "$INSTALL_DIR" ]; then
    echo "⚠️  Directory exists. Removing old installation..."
    rm -rf "$INSTALL_DIR"
fi

# Clone repository
echo "📦 Cloning ecm-wrapper repository..."
git clone -q https://github.com/kyleaskine/ecm-wrapper.git "$INSTALL_DIR"
cd "$INSTALL_DIR/client"
echo "✓ Repository cloned"

# Create data directory
mkdir -p data
echo "✓ Data directory created"

# ============================================================
# Step 4: Download ECM Binary
# ============================================================
# Single GPU-universal binary: CUDA runtime is statically linked and the binary
# carries GPU SASS for every arch (sm_50-sm_120). Nothing CUDA-specific needs
# to be installed on the host - just an NVIDIA driver (CUDA 12+ class). The CPU
# side still must be built for a portable x86-64 ISA.
echo ""
echo "⬇️  Downloading ECM binary (universal: static cudart, sm_50-sm_120)..."
ECM_DOWNLOAD_URL="https://ecm.kyleaskine.com/downloads/ecm/ecm.gz"
ECM_PATH="$HOME/ecm"

if wget -q --show-progress "$ECM_DOWNLOAD_URL" -O "${ECM_PATH}.gz"; then
    gunzip -f "${ECM_PATH}.gz"
    chmod +x "$ECM_PATH"
else
    echo "❌ Failed to download ECM binary from $ECM_DOWNLOAD_URL"
    exit 1
fi

# Verify installation with a tiny CPU-only run.
if [ -x "$ECM_PATH" ]; then
    set +e
    ECM_VERIFY_OUTPUT=$(printf '1\n' | "$ECM_PATH" 1 2>&1)
    ECM_VERIFY_STATUS=$?
    set -e
    if [[ "$ECM_VERIFY_OUTPUT" != *"GMP-ECM"* ]]; then
        echo "❌ ECM binary failed verification (exit $ECM_VERIFY_STATUS)"
        printf '%s\n' "$ECM_VERIFY_OUTPUT" | head -20
        if [ "$ECM_VERIFY_STATUS" -eq 132 ]; then
            echo ""
            echo "The downloaded binary contains CPU instructions unsupported by this host."
            echo "Rebuild/publish ECM with portable CPU flags, e.g. -march=x86-64 -mtune=generic."
            if command -v lscpu &> /dev/null; then
                lscpu | grep -E 'Model name|Flags' || true
            fi
        fi
        exit 1
    fi
    ECM_VERSION_STR=$(printf '%s\n' "$ECM_VERIFY_OUTPUT" | head -1)
    echo "✓ ECM binary installed: $ECM_VERSION_STR"
else
    echo "❌ ECM binary installation failed"
    exit 1
fi

# ============================================================
# Step 5: Install Python Dependencies
# ============================================================
echo ""
echo "📚 Installing Python dependencies..."
# Use `python3 -m pip` so packages land in the same interpreter that runs the
# client (vast.ai images often ship with conda + system Python side-by-side).
# Fall back to --break-system-packages for PEP 668 environments.
if ! python3 -m pip install requests pyyaml; then
    echo "⚠️  pip install failed, retrying with --break-system-packages..."
    python3 -m pip install --break-system-packages requests pyyaml
fi
echo "✓ Dependencies installed (requests, pyyaml)"
python3 -c "import requests, yaml; print(f'   requests {requests.__version__}, pyyaml {yaml.__version__}')"

# ============================================================
# Step 6: Detect GPU
# ============================================================
echo ""
echo "🎮 Checking for GPU..."
GPU_ENABLED="false"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_ENABLED="true"
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "✓ GPU detected: $GPU_INFO"
    else
        echo "ℹ️  nvidia-smi found but no GPU detected"
    fi
else
    echo "ℹ️  No GPU detected (CPU mode)"
fi

# ============================================================
# Step 7: Create Configuration
# ============================================================
echo ""
echo "⚙️  Creating client.local.yaml..."

cat > client.local.yaml << EOF
# Cloud Instance Configuration
# Generated: $(date)

api:
  endpoint: "$API_ENDPOINT"
  timeout: 30

client:
  username: "$USERNAME"
  cpu_name: "$MACHINE_NAME"

programs:
  gmp_ecm:
    path: "$ECM_PATH"
    gpu_enabled: $GPU_ENABLED
    gpu_device: 0

# Logging configuration
logging:
  level: "INFO"
  file: "data/logs/ecm_client.log"
  console: true
EOF

echo "✓ Configuration file created"

# ============================================================
# Step 8: Setup Complete - Display Summary
# ============================================================
echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo "Username:      $USERNAME"
echo "Machine:       $MACHINE_NAME"
echo "API Endpoint:  $API_ENDPOINT"
echo "ECM Binary:    $ECM_PATH"
echo "Binary:        universal (static cudart, sm_50-sm_120)"
echo "GPU:           $GPU_ENABLED"
echo "Working Dir:   $INSTALL_DIR/client"
echo "============================================================"
echo ""
echo "🚀 Ready to run ECM factorization!"
echo ""
echo "Example commands:"
echo ""
echo "  # Change to client directory"
echo "  cd $INSTALL_DIR/client"
echo ""
echo "  # Test with a small number (ecm_wrapper.py doesn't submit by default)"
echo "  python3 ecm_wrapper.py --composite \"123456789012345\" --curves 10 --b1 11000"
echo ""
echo "  # Auto-work mode (progressive strategy, stage 1 only; client selects B1)"
echo "  python3 ecm_client.py --work-type progressive --stage1-only --priority $PRIORITY_VALUE -v"
echo ""
echo "  # Auto-work with specific count"
echo "  python3 ecm_client.py --work-type progressive --work-count 10 --stage1-only --priority $PRIORITY_VALUE -v"
echo ""
if [ "$GPU_ENABLED" = "true" ]; then
echo "  # Auto-work with GPU (stage 1 only)"
echo "  python3 ecm_client.py --work-type progressive --stage1-only --priority $PRIORITY_VALUE -v"
echo ""
fi
echo "  # Auto-work with multiprocess (CPU only)"
echo "  python3 ecm_client.py --work-type progressive --multiprocess --workers 8 --stage1-only --priority $PRIORITY_VALUE -v"
echo ""
echo "============================================================"
echo ""

# Optional: Offer to start auto-work immediately
read -p "Start auto-work mode now? [y/N]: " START_NOW
if [[ "$START_NOW" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting auto-work mode (progressive strategy)..."
    echo "Press Ctrl+C to stop"
    echo ""
    cd "$INSTALL_DIR/client"
    if [ "$GPU_ENABLED" = "true" ]; then
        python3 ecm_client.py --work-type progressive --stage1-only --priority $PRIORITY_VALUE -v
    else
        python3 ecm_client.py --work-type progressive --multiprocess --stage1-only --priority $PRIORITY_VALUE -v
    fi
fi
