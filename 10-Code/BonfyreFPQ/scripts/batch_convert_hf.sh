#!/bin/bash
# batch_convert_hf.sh — Download HF model, convert to v12 .fpq, upload to NICKO/
# Usage: ./batch_convert_hf.sh <hf_repo_id> [--bits 3]
# Example: ./batch_convert_hf.sh meta-llama/Llama-3.2-1B
# Requires: bonfyre-fpq binary, huggingface-cli, ~2× model size in disk

set -euo pipefail

HF_REPO="${1:?Usage: $0 <hf_repo_id> [--bits N]}"
BITS="${3:-3}"
HF_TOKEN="${HF_TOKEN:-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}"
HF_ORG="${HF_ORG:-NICKO}"
WORK_DIR="${WORK_DIR:-/tmp/fpq-convert}"
BINARY="${BINARY:-./bonfyre-fpq}"

# Derive names
MODEL_NAME=$(echo "$HF_REPO" | tr '/' '-')
OUTPUT_REPO="${HF_ORG}/${MODEL_NAME}-BonfyreFPQ-v12"
SRC_DIR="${WORK_DIR}/src/${MODEL_NAME}"
OUT_DIR="${WORK_DIR}/out/${MODEL_NAME}"

mkdir -p "$SRC_DIR" "$OUT_DIR"

echo "=== BonfyreFPQ v12 Batch Converter ==="
echo "  Source: $HF_REPO"
echo "  Target: $OUTPUT_REPO"
echo "  Work:   $WORK_DIR"
echo ""

# Step 1: Download
echo "[1/4] Downloading ${HF_REPO}..."
huggingface-cli download "$HF_REPO" \
    --local-dir "$SRC_DIR" \
    --include "*.safetensors" "*.json" \
    --token "$HF_TOKEN" \
    --quiet

SAFETENSORS=($(find "$SRC_DIR" -name "*.safetensors" | sort))
echo "  Found ${#SAFETENSORS[@]} safetensors file(s)"

# Step 2: Convert each shard
echo "[2/4] Converting to .fpq v12..."
TOTAL_SRC=0
TOTAL_FPQ=0
for ST in "${SAFETENSORS[@]}"; do
    BASENAME=$(basename "$ST")
    FPQ_NAME="${BASENAME%.safetensors}.fpq"
    echo "  Converting: $BASENAME"
    
    "$BINARY" convert-fpq "$ST" "${OUT_DIR}/${FPQ_NAME}" --bits "$BITS"
    
    SRC_SIZE=$(stat -f%z "$ST" 2>/dev/null || stat -c%s "$ST")
    FPQ_SIZE=$(stat -f%z "${OUT_DIR}/${FPQ_NAME}" 2>/dev/null || stat -c%s "${OUT_DIR}/${FPQ_NAME}")
    RATIO=$(python3 -c "print(f'{$SRC_SIZE/$FPQ_SIZE:.1f}')")
    echo "    ${SRC_SIZE} → ${FPQ_SIZE} (${RATIO}×)"
    
    TOTAL_SRC=$((TOTAL_SRC + SRC_SIZE))
    TOTAL_FPQ=$((TOTAL_FPQ + FPQ_SIZE))
done

TOTAL_RATIO=$(python3 -c "print(f'{$TOTAL_SRC/$TOTAL_FPQ:.1f}')")
echo "  Total: $(python3 -c "print(f'{$TOTAL_SRC/1e9:.2f}')") GB → $(python3 -c "print(f'{$TOTAL_FPQ/1e9:.2f}')") GB (${TOTAL_RATIO}×)"

# Step 3: Create HF repo + upload
echo "[3/4] Uploading to ${OUTPUT_REPO}..."
python3 << PYEOF
from huggingface_hub import HfApi, create_repo
import os, glob

api = HfApi(token="$HF_TOKEN")
repo_id = "$OUTPUT_REPO"
out_dir = "$OUT_DIR"
hf_repo = "$HF_REPO"

create_repo(repo_id, repo_type="model", token=api.token, exist_ok=True)

# Generate README
fpq_files = sorted(glob.glob(os.path.join(out_dir, "*.fpq")))
total_size = sum(os.path.getsize(f) for f in fpq_files)
n_files = len(fpq_files)

card = f"""---
tags: [bonfyre-fpq, v12, rans-entropy, e8-lattice, quantization]
base_model: {hf_repo}
---
# {hf_repo.split('/')[-1]} — BonfyreFPQ v12 Native

| Property | Value |
|----------|-------|
| Format | .fpq v12 (rANS entropy + E8 lattice + 6-bit tiles) |
| Base model | [{hf_repo}](https://huggingface.co/{hf_repo}) |
| Files | {n_files} |
| Total size | {total_size/1e9:.2f} GB |

## Usage
\`\`\`bash
# Decode to safetensors
./bonfyre-fpqx decode model.fpq  # → model.safetensors (BF16)
\`\`\`

## About BonfyreFPQ
E8 lattice quantization with FWHT spectral transform and rANS entropy coding.
[GitHub](https://github.com/Nickgonzales76017/bonfyre)
"""

api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id,
                commit_message="Add model card")

for fpath in fpq_files:
    fname = os.path.basename(fpath)
    sz = os.path.getsize(fpath)
    print(f"  Uploading {fname} ({sz/1e9:.2f} GB)...")
    api.upload_file(path_or_fileobj=fpath, path_in_repo=fname, repo_id=repo_id,
                    commit_message=f"Add {fname}")
    print(f"  Done: {fname}")

print("Upload complete")
PYEOF

# Step 4: Cleanup source (keep .fpq)
echo "[4/4] Cleaning up source files..."
rm -rf "$SRC_DIR"
echo ""
echo "=== DONE ==="
echo "  Repo: https://huggingface.co/${OUTPUT_REPO}"
echo "  Local: ${OUT_DIR}/"
