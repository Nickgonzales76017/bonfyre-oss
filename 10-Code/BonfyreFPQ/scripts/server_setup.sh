#!/bin/bash
# server_setup.sh — Bootstrap a cheap VPS for BonfyreFPQ batch conversion
# Tested on: Ubuntu 22.04/24.04 (Hetzner, OVH, Vultr, DigitalOcean)
# Recommended: Hetzner CCX33 (8 vCPU, 32GB RAM, 240GB NVMe) — €36/mo
# Alternative: OVH Rise-1 (6 core, 64GB RAM, 2TB) — ~$60/mo
#
# For models up to 14B: 32GB RAM, 200GB disk
# For 70B: 64GB RAM, 500GB disk

set -euo pipefail

echo "=== BonfyreFPQ Server Setup ==="

# System packages
apt-get update -qq
apt-get install -y -qq build-essential git python3 python3-pip libopenblas-dev curl

# Python packages
pip3 install --break-system-packages huggingface_hub safetensors

# HuggingFace CLI
pip3 install --break-system-packages "huggingface_hub[cli]"

# Clone and build
cd /opt
if [ ! -d BonfyreFPQ ]; then
    git clone https://github.com/Nickgonzales76017/bonfyre.git BonfyreFPQ
    cd BonfyreFPQ/10-Code/BonfyreFPQ
else
    cd BonfyreFPQ/10-Code/BonfyreFPQ
    git pull
fi

make clean && make -j$(nproc)
echo "  Binary: $(ls -la bonfyre-fpq)"

# Symlink binary to PATH
ln -sf "$(pwd)/bonfyre-fpq" /usr/local/bin/bonfyre-fpq
ln -sf "$(pwd)/bonfyre-fpqx" /usr/local/bin/bonfyre-fpqx 2>/dev/null || true

# HF login
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN"
    echo "  HF logged in"
fi

# Work directory
mkdir -p /workspace/fpq-convert
export WORK_DIR=/workspace/fpq-convert

echo ""
echo "=== Setup Complete ==="
echo "  bonfyre-fpq: $(bonfyre-fpq --version 2>&1 | head -1 || echo 'built')"
echo "  Work dir: /workspace/fpq-convert"
echo ""
echo "Run conversions:"
echo "  export HF_TOKEN=hf_xxx"
echo "  cd /opt/BonfyreFPQ/10-Code/BonfyreFPQ"
echo "  bash scripts/mass_convert.sh"
echo ""
echo "Or single model:"
echo "  bash scripts/batch_convert_hf.sh meta-llama/Llama-3.2-1B"
