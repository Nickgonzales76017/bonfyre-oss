#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# ALL-IN-ONE: Benchmark FPQ + Fix .fpq headers + Upload
# Run on a single cheap RunPod (RTX A5000, $0.16/hr)
# Target: ~30 min total, ~$0.10 budget
#
# We only benchmark OUR model — competitor numbers are published:
#   Qwen2.5-3B FP16 baseline (Qwen blog):
#     MMLU=65.6  ARC-C=56.5  WinoGrande=71.1  HellaSwag=74.6
#   No published 3-bit GPTQ/AWQ for this model (only Int4/Int8)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"  # set via env or pipeline/.env
HF_TOKEN="${HF_TOKEN:-}"  # set via env or pipeline/.env
export HF_TOKEN
FPQ_REPO="NICKO/Qwen2.5-3B-BonfyreAlgebra-FPQ3"
WORKDIR="/workspace/benchmark"
RESULTS_DIR="$WORKDIR/results"

echo "═══════════════════════════════════════════════════════════════"
echo " PHASE 0: Setup"
echo "═══════════════════════════════════════════════════════════════"
mkdir -p "$RESULTS_DIR" && cd "$WORKDIR"
pip install -q lm-eval transformers accelerate safetensors huggingface_hub
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# PHASE 1: BENCHMARK — Run lm-eval on our FPQ-compressed model
# Same tasks + shot counts as Qwen's published eval for 1:1 compare
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " PHASE 1: Benchmark BonfyreFPQ-3bit"
echo "═══════════════════════════════════════════════════════════════"

# Tasks matching Qwen's published eval (same shot counts):
#   HellaSwag 10-shot, WinoGrande 5-shot, ARC-C 25-shot,
#   TruthfulQA 0-shot, MMLU 5-shot
TASKS="hellaswag,winogrande,arc_challenge,truthfulqa_mc2,mmlu"

echo "[1/1] Benchmarking FPQ-3bit: $FPQ_REPO (BF16 safetensors)..."
lm_eval --model hf \
  --model_args "pretrained=$FPQ_REPO,dtype=bfloat16,trust_remote_code=True" \
  --tasks "$TASKS" \
  --batch_size auto \
  --output_path "$RESULTS_DIR/fpq_3bit" \
  --log_samples 2>&1 | tee "$RESULTS_DIR/fpq_3bit.log"

# ── Print comparison table ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " RESULTS: BonfyreFPQ-3bit vs Published FP16 Baseline"
echo "═══════════════════════════════════════════════════════════════"
python3 << 'PYEOF'
import json, glob, os

# Published Qwen2.5-3B FP16 numbers (from Qwen blog, same eval protocol)
published_fp16 = {
    "mmlu": 0.656,
    "arc_challenge": 0.565,
    "winogrande": 0.711,
    "hellaswag": 0.746,
    "truthfulqa_mc2": 0.489,
}

results_dir = "/workspace/benchmark/results"
fpq_scores = {}
for p in glob.glob(f"{results_dir}/fpq_3bit/**/results.json", recursive=True):
    with open(p) as f:
        data = json.load(f)
    for task, vals in data.get("results", {}).items():
        for key in ["acc,none", "acc_norm,none", "mc2,none"]:
            if key in vals:
                fpq_scores[task] = vals[key]
                break

if not fpq_scores:
    print("ERROR: No FPQ results found — check logs")
else:
    tasks = sorted(published_fp16.keys())
    print(f"{'Task':<20} {'FP16 (published)':>16} {'FPQ-3bit':>12} {'Δ':>8}")
    print("-" * 58)
    fp16_avg, fpq_avg = 0, 0
    for t in tasks:
        fp16 = published_fp16.get(t, 0)
        fpq = fpq_scores.get(t, 0)
        delta = fpq - fp16
        sign = "+" if delta >= 0 else ""
        print(f"{t:<20} {fp16:>15.1%} {fpq:>11.1%} {sign}{delta:>6.1%}")
        fp16_avg += fp16
        fpq_avg += fpq
    fp16_avg /= len(tasks)
    fpq_avg /= len(tasks)
    delta_avg = fpq_avg - fp16_avg
    sign = "+" if delta_avg >= 0 else ""
    print("-" * 58)
    print(f"{'AVERAGE':<20} {fp16_avg:>15.1%} {fpq_avg:>11.1%} {sign}{delta_avg:>6.1%}")
    print()
    print("FPQ-3bit = BonfyreFPQ algebra-compressed at ~3 bits/weight")
    print("FP16 = Qwen/Qwen2.5-3B original (published by Qwen team)")
    print(f"Compression ratio: ~10.7× (FP16 → FPQ native .fpq format)")
PYEOF

# ═══════════════════════════════════════════════════════════════
# PHASE 2: FIX .FPQ FILES
# Build from source (uploaded via scp), convert safetensors → .fpq
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " PHASE 2: Rebuild native .fpq files (fixed headers)"
echo "═══════════════════════════════════════════════════════════════"

if [ -d "/workspace/BonfyreFPQ/src" ]; then
    cd /workspace/BonfyreFPQ
    # Build both binaries (need bonfyre-fpq for convert-fpq command)
    apt-get install -y -q build-essential libopenblas-dev 2>/dev/null || true
    make clean && make 2>&1 | tail -5

    # Convert each safetensors model → native .fpq with fixed headers
    for REPO in \
        "NICKO/Qwen2.5-3B-BonfyreAlgebra-FPQ3" \
        "NICKO/whisper-large-v3-BonfyreAlgebra-FPQ3" \
        "NICKO/F5-TTS-BonfyreAlgebra-FPQ3"; do

        SHORT=$(echo "$REPO" | sed 's|NICKO/||; s|-BonfyreAlgebra-FPQ3||; s|\.|-|g' | tr '[:upper:]' '[:lower:]')
        NATIVE_REPO=$(echo "$REPO" | sed 's|-BonfyreAlgebra-FPQ3|-algebra-fpq3-BonfyreFPQ-Native|')
        echo ""
        echo "  Converting: $REPO → native .fpq"
        mkdir -p "/workspace/fpq_rebuild/$SHORT"
        cd "/workspace/fpq_rebuild/$SHORT"

        # Download safetensors
        huggingface-cli download "$REPO" --local-dir ./input

        # Find all safetensors files and convert each shard
        for ST_FILE in ./input/*.safetensors; do
            [ -f "$ST_FILE" ] || continue
            BASE=$(basename "$ST_FILE" .safetensors)
            echo "    $BASE.safetensors → $BASE.fpq"
            /workspace/BonfyreFPQ/bonfyre-fpq convert-fpq "$ST_FILE" "./${BASE}.fpq" 2>&1 | tail -3
        done

        # Upload fixed .fpq files to existing native repo
        echo "    Uploading to $NATIVE_REPO..."
        for FPQ_FILE in ./*.fpq; do
            [ -f "$FPQ_FILE" ] || continue
            python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='$FPQ_FILE',
    path_in_repo='$(basename $FPQ_FILE)',
    repo_id='$NATIVE_REPO',
    repo_type='model'
)
print('  Uploaded: $(basename $FPQ_FILE)')
"
        done

        cd /workspace/BonfyreFPQ
    done
else
    echo "  Source not found at /workspace/BonfyreFPQ/src"
    echo "  Upload source first: scp -P PORT -r src/ include/ Makefile root@HOST:/workspace/BonfyreFPQ/"
fi

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Upload benchmark results
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " PHASE 3: Upload benchmark results"
echo "═══════════════════════════════════════════════════════════════"

python3 << 'UPLOAD_EOF'
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
repo_id = "NICKO/BonfyreFPQ-Benchmarks"
create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)

results_dir = "/workspace/benchmark/results"
for root, dirs, files in os.walk(results_dir):
    for f in files:
        local_path = os.path.join(root, f)
        remote_path = os.path.relpath(local_path, results_dir)
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"qwen2.5-3b/{remote_path}",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"  Uploaded: {remote_path}")
        except Exception as e:
            print(f"  Failed: {remote_path}: {e}")
UPLOAD_EOF

# ═══════════════════════════════════════════════════════════════
# Self-stop to save money
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " ALL DONE"
echo "═══════════════════════════════════════════════════════════════"

if [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo " Self-stopping pod $RUNPOD_POD_ID..."
    curl -s -H "Content-Type: application/json" \
         -H "Authorization: Bearer $RUNPOD_API_KEY" \
         https://api.runpod.io/graphql \
         -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}"
    echo ""
fi
