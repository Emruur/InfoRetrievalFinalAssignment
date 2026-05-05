#!/usr/bin/env bash
set -e

VENV_DIR=".venv"
PYTHON="python3"

echo "==> Creating virtual environment in $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

# Detect CUDA via nvidia-smi
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    # Parse major.minor version from driver, map to closest PyTorch wheel
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

    echo "==> Detected CUDA $CUDA_VER"

    # Choose the highest supported torch CUDA wheel
    if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_EXTRA="cu124"
    elif [ "$CUDA_MAJOR" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_EXTRA="cu121"
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_EXTRA="cu118"
    else
        echo "WARNING: CUDA $CUDA_VER is older than 11.8 — falling back to CPU torch build."
        TORCH_INDEX=""
        TORCH_EXTRA="cpu"
    fi

    if [ -n "$TORCH_INDEX" ]; then
        echo "==> Installing PyTorch (CUDA $TORCH_EXTRA)"
        pip install torch torchvision --index-url "$TORCH_INDEX"
    else
        pip install torch torchvision
    fi

    echo "==> Installing bitsandbytes (CUDA required)"
    pip install bitsandbytes==0.43.1

else
    echo "==> No CUDA GPU detected — installing CPU-only PyTorch"
    echo "    Training will work but will be very slow on CPU."
    echo "    Consider reducing max_train_samples to 5e4 in p1.py."
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
    # bitsandbytes is CUDA-only; skip it
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
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Running on CPU — no GPU acceleration.")

import sentence_transformers
print(f"sentence-transformers: {sentence_transformers.__version__}")
EOF

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo "Then run:"
echo "  python p1.py"
