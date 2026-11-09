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
# Step 4: Detect GPU
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
# Step 5: Build GMP-ECM from source
# ============================================================
# github.com/kyleaskine/gmp-ecm has GPU stage 1 *and* GPU stage 2 (the old
# prebuilt universal binary only had GPU stage 1). It is built here for this
# machine's GPU, so it needs the CUDA toolkit (nvcc): use a CUDA "devel" image.
echo ""
echo "🔨 Building GMP-ECM from github.com/kyleaskine/gmp-ecm..."
ECM_SRC="$HOME/gmp-ecm"
ECM_PATH="$ECM_SRC/ecm"

echo "Installing build tools (gcc, autotools, GMP)..."
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq build-essential autoconf automake libtool m4 libgmp-dev > /dev/null

CONFIGURE_ARGS=()
GPU_BUILD_DESC="CPU only"

# The GPU build cannot go ahead: offer a CPU-only build instead, or exit.
offer_cpu_only() {
    read -p "Build a CPU-only GMP-ECM instead? [y/N]: " CPU_ONLY
    if [[ ! "$CPU_ONLY" =~ ^[Yy]$ ]]; then
        exit 1
    fi
    GPU_ENABLED="false"
}

# Print the first of the directories (arguments 2...) that contains file $1.
find_dir_with() {
    local file=$1 dir
    shift
    for dir in "$@"; do
        if [ -e "$dir/$file" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

# Locate what GMP-ECM's configure needs besides nvcc, for the toolkit
# version of $NVCC: cuda.h + cuda_runtime.h (CUDA_INCLUDE), libcudart.so
# (CUDA_LIB), and a libcuda.so that gcc can link a test program against.
# (apt's cuda-nvcc pulls them in, but other installs may lack them or keep
# them outside $CUDA_HOME), so several layouts are searched. When only the
# driver's libcuda.so.1 is installed, CUDA_STUBS is the toolkit's link-time
# stub for it. Fails with the missing files in CUDA_MISSING.
find_cuda_dev_files() {
    local ver_root="/usr/local/cuda-$CUDA_VER"
    CUDA_MISSING=()
    CUDA_STUBS=""
    CUDA_INCLUDE=$(find_dir_with cuda.h "$CUDA_HOME/include" "$CUDA_HOME"/targets/*/include \
                   "$ver_root/include" "$ver_root"/targets/*/include || true)
    if [ -z "$CUDA_INCLUDE" ]; then
        CUDA_MISSING+=(cuda.h)
    elif [ ! -e "$CUDA_INCLUDE/cuda_runtime.h" ]; then
        CUDA_MISSING+=(cuda_runtime.h)
    fi
    CUDA_LIB=$(find_dir_with libcudart.so "$CUDA_HOME/lib64" "$CUDA_HOME/lib" \
               "$CUDA_HOME"/targets/*/lib "$CUDA_HOME/lib/$(uname -m)-linux-gnu" \
               "$ver_root/lib64" "$ver_root"/targets/*/lib || true)
    if [ -z "$CUDA_LIB" ]; then
        CUDA_MISSING+=(libcudart.so)
    fi
    # configure links with -L$CUDA_LIB plus gcc's own search path, which
    # includes $LIBRARY_PATH (that is how CUDA -devel images find the stub).
    if { [ -z "$CUDA_LIB" ] || [ ! -e "$CUDA_LIB/libcuda.so" ]; } \
       && [ "$(gcc -print-file-name=libcuda.so)" = "libcuda.so" ]; then
        CUDA_STUBS=$(find_dir_with libcuda.so ${CUDA_LIB:+"$CUDA_LIB/stubs"} \
                     "$CUDA_HOME/lib64/stubs" "$CUDA_HOME"/targets/*/lib/stubs \
                     "$ver_root"/targets/*/lib/stubs || true)
        if [ -z "$CUDA_STUBS" ]; then
            CUDA_MISSING+=(libcuda.so)
        fi
    fi
    [ ${#CUDA_MISSING[@]} -eq 0 ]
}

if [ "$GPU_ENABLED" = "true" ]; then
    NVCC=$(command -v nvcc || true)
    if [ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
        NVCC=/usr/local/cuda/bin/nvcc
    fi
    if [ -z "$NVCC" ]; then
        NVCC=$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)
    fi
    if [ -z "$NVCC" ]; then
        echo "❌ A GPU is present but the CUDA toolkit (nvcc) is not installed."
        echo "   GMP-ECM's GPU code has to be compiled with nvcc. Use an image with the"
        echo "   CUDA toolkit (e.g. nvidia/cuda:12.8.0-devel-ubuntu22.04 or a pytorch"
        echo "   -devel image), or install the toolkit and rerun this script."
        offer_cpu_only
    fi
fi

if [ "$GPU_ENABLED" = "true" ]; then
    CUDA_HOME=$(dirname "$(dirname "$NVCC")")
    CUDA_VER=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "✓ CUDA toolkit: release $CUDA_VER ($CUDA_HOME)"

    if ! find_cuda_dev_files; then
        # In NVIDIA's apt repository, cuda-cudart-dev-X-Y has cuda.h,
        # cuda_runtime.h and libcudart.so, and pulls in cuda-driver-dev-X-Y
        # (the libcuda.so stub).
        echo "❌ The CUDA $CUDA_VER toolkit is missing ${CUDA_MISSING[*]}, needed to build"
        echo "   the GPU code (searched under $CUDA_HOME and /usr/local/cuda-$CUDA_VER)."
        echo "   Install them (apt-get install cuda-cudart-dev-${CUDA_VER/./-}) or use a CUDA"
        echo "   -devel image, then rerun this script."
        offer_cpu_only
    fi
fi

if [ "$GPU_ENABLED" = "true" ]; then
    echo "✓ CUDA headers: $CUDA_INCLUDE, libraries: $CUDA_LIB"
    if [ -n "$CUDA_STUBS" ]; then
        # Only the driver's libcuda.so.1 is installed, which configure's test
        # link cannot use; the binary still loads the real driver at run time.
        export LIBRARY_PATH="$CUDA_STUBS${LIBRARY_PATH:+:$LIBRARY_PATH}"
    fi

    # The driver must support the toolkit's CUDA version, or every GPU run
    # fails -- check now rather than after the long build.
    # Header says "CUDA Version: 12.4", or "CUDA UMD Version: 13.4" on newer drivers.
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null \
                  | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9.]*\).*/\2/p' | head -1)
    if [ -n "$DRIVER_CUDA" ] && [ -n "$CUDA_VER" ] \
       && [ "$(printf '%s\n' "$CUDA_VER" "$DRIVER_CUDA" | sort -V | tail -1)" != "$DRIVER_CUDA" ]; then
        echo "❌ CUDA toolkit $CUDA_VER is newer than this NVIDIA driver supports (CUDA $DRIVER_CUDA)."
        echo "   The GPU code would build but not run. Use an image with CUDA <= $DRIVER_CUDA,"
        echo "   or a machine with a newer driver."
        read -p "Build anyway (e.g. CUDA forward-compatibility libraries are installed)? [y/N]: " BUILD_ANYWAY
        if [[ ! "$BUILD_ANYWAY" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # One compute capability per GPU model present (7.5 -> 75). A single
    # one is built on its own (much faster); mixed GPUs get every arch.
    GPU_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | tr -d ' .' | grep -E '^[0-9]+$' | sort -u || true)
    if [ "$(printf '%s\n' "$GPU_ARCHS" | grep -c .)" -eq 1 ]; then
        NVCC_ARCHS=$("$NVCC" --list-gpu-arch 2>/dev/null || true)
        if [ -n "$NVCC_ARCHS" ] && ! printf '%s\n' "$NVCC_ARCHS" | grep -qx "compute_$GPU_ARCHS"; then
            echo "❌ This CUDA toolkit cannot compile for this GPU (sm_$GPU_ARCHS)."
            echo "   Use an image with a newer CUDA toolkit."
            exit 1
        fi
        CONFIGURE_ARGS+=("--enable-gpu=$GPU_ARCHS")
        GPU_BUILD_DESC="GPU sm_$GPU_ARCHS"
    else
        CONFIGURE_ARGS+=("--enable-gpu")
        GPU_BUILD_DESC="GPU (all architectures)"
    fi
    # Pass the directories found above rather than --with-cuda=$CUDA_HOME,
    # which assumes $CUDA_HOME/include and $CUDA_HOME/lib64.
    CONFIGURE_ARGS+=("--with-cuda-bin=$(dirname "$NVCC")"
                     "--with-cuda-include=$CUDA_INCLUDE"
                     "--with-cuda-lib=$CUDA_LIB")
    # configure treats any compiler warning in its tests as a failure
    # (AC_LANG_WERROR) and builds with -Wundef. CUDA 12.0/12.1's cuda.h has an
    # unguarded "#if __cplusplus", so it reports "checking for cuda.h... no"
    # even though the header is there. As an -isystem directory, its headers
    # raise no warnings. CPPFLAGS (unlike CFLAGS) keeps configure's default
    # -O3 -march=native; /usr/include is a system directory already.
    if [ "$CUDA_INCLUDE" != "/usr/include" ]; then
        CONFIGURE_ARGS+=("CPPFLAGS=-isystem $CUDA_INCLUDE${CPPFLAGS:+ $CPPFLAGS}")
    fi
fi

rm -rf "$ECM_SRC"
git clone -q --depth 1 https://github.com/kyleaskine/gmp-ecm.git "$ECM_SRC"
BUILD_LOG="$ECM_SRC/build.log"
echo "Configuring and compiling ($GPU_BUILD_DESC, $(nproc) cores)."
echo "This takes several minutes (the GPU code is slow to compile); log: $BUILD_LOG"
if ! ( cd "$ECM_SRC" \
       && autoreconf -i \
       && ./configure "${CONFIGURE_ARGS[@]}" \
       && make -j"$(nproc)" ) > "$BUILD_LOG" 2>&1; then
    echo "❌ GMP-ECM build failed. Last lines of $BUILD_LOG:"
    tail -25 "$BUILD_LOG"
    exit 1
fi
echo "✓ GMP-ECM built"

# Verify with a tiny CPU-only run.
set +e
ECM_VERIFY_OUTPUT=$(printf '1\n' | "$ECM_PATH" 1 2>&1)
ECM_VERIFY_STATUS=$?
set -e
if [[ "$ECM_VERIFY_OUTPUT" != *"GMP-ECM"* ]]; then
    echo "❌ ECM binary failed verification (exit $ECM_VERIFY_STATUS)"
    printf '%s\n' "$ECM_VERIFY_OUTPUT" | head -20
    exit 1
fi
ECM_VERSION_STR=$(printf '%s\n' "$ECM_VERIFY_OUTPUT" | head -1)
echo "✓ ECM binary works: $ECM_VERSION_STR"

# Verify both GPU stages on a C50 (no factor is found at B1=1000; it only has
# to run stage 1 and stage 2 on the GPU without an error).
if [ "$GPU_ENABLED" = "true" ]; then
    set +e
    GPU_TEST_OUTPUT=$(echo 10000000000000000000000083000000000000000000000091 \
                      | timeout 300 "$ECM_PATH" -gpu -gpucurves 64 1000 2>&1)
    set -e
    if [[ "$GPU_TEST_OUTPUT" != *"Step 2 took"*"of GPU time"* ]] \
       || [[ "$GPU_TEST_OUTPUT" == *"Error"* ]]; then
        echo "❌ GPU test run failed:"
        printf '%s\n' "$GPU_TEST_OUTPUT" | tail -20
        echo "   (A CUDA toolkit newer than the NVIDIA driver supports fails here:"
        echo "   compare 'nvcc --version' with the CUDA version 'nvidia-smi' reports.)"
        exit 1
    fi
    echo "✓ GPU stage 1 and stage 2 work"
fi

# ============================================================
# Step 6: Install Python Dependencies
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
echo "Build:         $GPU_BUILD_DESC (source: $ECM_SRC)"
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
if [ "$GPU_ENABLED" = "true" ]; then
echo "  # Auto-work mode (progressive strategy, GPU stage 1 only; client selects B1)"
echo "  python3 ecm_client.py --work-type progressive --stage1-only --priority $PRIORITY_VALUE -v"
echo ""
echo "  # Auto-work with specific count"
echo "  python3 ecm_client.py --work-type progressive --work-count 10 --stage1-only --priority $PRIORITY_VALUE -v"
echo ""
echo "  # Auto-work with both stages on the GPU (server t-level targets)"
echo "  python3 ecm_client.py --standard --gpu --priority $PRIORITY_VALUE -v"
echo ""
echo "  # Process GPU stage 1 residues uploaded by other clients (stage 2 on the GPU)"
echo "  python3 ecm_client.py --stage2-only --gpu --b2 -1"
echo ""
else
echo "  # Auto-work with multiprocess (progressive strategy, server t-level targets)"
echo "  python3 ecm_client.py --work-type progressive --multiprocess --priority $PRIORITY_VALUE -v"
echo ""
echo "  # Auto-work with specific count"
echo "  python3 ecm_client.py --work-type progressive --multiprocess --work-count 10 --priority $PRIORITY_VALUE -v"
echo ""
echo "  # Process GPU stage 1 residues uploaded by other clients (stage 2 on the CPU)"
echo "  python3 ecm_client.py --stage2-only --workers $(nproc)"
echo ""
fi
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
        python3 ecm_client.py --work-type progressive --multiprocess --priority $PRIORITY_VALUE -v
    fi
fi
