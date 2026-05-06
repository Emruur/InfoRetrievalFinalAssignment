#!/usr/bin/env bash
set -e

ENV_PATH="/data/s4402146/conda-envs/ir-assignment"
PYTHON_VERSION="3.11"

mkdir -p "/data/s4402146/conda-envs"

if [ -d "$ENV_PATH" ]; then
    echo "==> Conda env at $ENV_PATH already exists, updating..."
else
    echo "==> Creating conda env at $ENV_PATH (Python $PYTHON_VERSION)"
    conda create -y -p "$ENV_PATH" python=$PYTHON_VERSION
fi

export PIP_CACHE_DIR="/data/s4402146/.pip-cache"
mkdir -p "$PIP_CACHE_DIR"

conda run -p "$ENV_PATH" --no-capture-output bash <<'INNER'
set -e

export PIP_CACHE_DIR="/data/s4402146/.pip-cache"
mkdir -p "$PIP_CACHE_DIR"

echo "==> Upgrading pip"
pip install --upgrade pip

OS="$(uname -s)"
ARCH="$(uname -m)"

detect_cuda_version() {
    # 1. Try nvidia-smi (may fail if driver/library mismatch)
    if command -v nvidia-smi &>/dev/null; then
        VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
        if [ -n "$VER" ]; then echo "$VER"; return; fi
    fi

    # 2. Try nvcc
    if command -v nvcc &>/dev/null; then
        VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
        if [ -n "$VER" ]; then echo "$VER"; return; fi
    fi

    # 3. Try CUDA version file
    for f in /usr/local/cuda/version.json /usr/local/cuda/version.txt; do
        if [ -f "$f" ]; then
            VER=$(grep -oP '"version"\s*:\s*"\K[0-9]+\.[0-9]+' "$f" 2>/dev/null \
               || grep -oP 'CUDA Version \K[0-9]+\.[0-9]+' "$f" 2>/dev/null | head -1)
            if [ -n "$VER" ]; then echo "$VER"; return; fi
        fi
    done

    echo ""
}

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "==> Apple Silicon — installing PyTorch with MPS support"
    pip install torch torchvision

elif [ -e /dev/nvidia0 ] || command -v nvidia-smi &>/dev/null || command -v nvcc &>/dev/null; then
    CUDA_VER=$(detect_cuda_version)

    if [ -z "$CUDA_VER" ]; then
        echo "WARNING: NVIDIA GPU detected but could not determine CUDA version."
        echo "         Defaulting to CUDA 12.1 PyTorch wheel."
        echo "         Override by setting CUDA_VER=12.x before running this script."
        CUDA_VER="12.1"
    fi

    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    echo "==> CUDA $CUDA_VER detected"

    if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    elif [ "$CUDA_MAJOR" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi

    echo "==> Installing PyTorch from $TORCH_INDEX"
    pip install torch torchvision --index-url "$TORCH_INDEX"
    pip install bitsandbytes==0.43.1

else
    echo "==> No GPU detected — CPU-only PyTorch"
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
fi

echo "==> Installing core dependencies"
pip install \
    sentence-transformers==3.4.1 \
    pytrec_eval \
    numpy \
    tqdm

echo "==> Verifying installation"
python - <<'EOF'
import torch

print(f"PyTorch version : {torch.__version__}")
if torch.cuda.is_available():
    print(f"Backend         : CUDA {torch.version.cuda}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif torch.backends.mps.is_available():
    print("Backend         : MPS (Apple Silicon)")
else:
    print("Backend         : CPU only")

import sentence_transformers
print(f"sentence-transformers: {sentence_transformers.__version__}")
EOF

INNER

echo ""
echo "Setup complete. Activate with:"
echo "  conda activate $ENV_PATH"
echo "Then run:"
echo "  python p1.py"
