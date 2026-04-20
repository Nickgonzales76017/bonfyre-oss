#!/usr/bin/env bash
# scripts/extract_fragment.sh — generalized fragment extraction for any promoted family.
#
# Extracts layers 0:2 (384→256 Gemm+ReLU) from a family's ONNX model,
# quantizes to BQFP, aligns against the parent full model, and benchmarks.
#
# This generalizes the T04-specific fragment_mesh.sh to work with ANY family
# whose model.onnx lives in a standard bonfyre-72/diversity runs directory.
#
# Usage:
#   bash scripts/extract_fragment.sh --family T15 [options]
#
# Required:
#   --family FAMILY_ID    family to extract fragment from (e.g. T15, T16)
#
# Optional:
#   --onnx PATH           explicit path to source model.onnx
#                         (auto-detected from --runs-dir if omitted)
#   --runs-dir DIR        root runs directory (default: /tmp/bonfyre-72/runs or
#                         /tmp/bonfyre-diversity/runs, whichever has the family)
#   --models-dir DIR      output dir for .bqfp and align (default: /tmp/bonfyre-families)
#   --range RANGE         layer range to extract (default: 0:2)
#   --n-anchors N         FPQx anchor count (default: 256)
#   --no-bench            skip SLI benchmark after extraction
#
# Output:
#   <models-dir>/<FAMILY>-frag.bqfp           quantized fragment
#   <models-dir>/align-<FAMILY>frag-<FAMILY>/ alignment vs full model
#
# macOS bash 3.2 compatible.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
FAMILY=""
ONNX_PATH=""
RUNS_DIR=""
MODELS_DIR="/tmp/bonfyre-families"
RANGE="0:2"
N_ANCHORS=256
NO_BENCH=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --family)     FAMILY="$2";    shift 2 ;;
        --onnx)       ONNX_PATH="$2"; shift 2 ;;
        --runs-dir)   RUNS_DIR="$2";  shift 2 ;;
        --models-dir) MODELS_DIR="$2";shift 2 ;;
        --range)      RANGE="$2";     shift 2 ;;
        --n-anchors)  N_ANCHORS="$2"; shift 2 ;;
        --no-bench)   NO_BENCH=1;     shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$FAMILY" ]]; then
    echo "Usage: extract_fragment.sh --family FAMILY_ID [--onnx PATH] [options]"
    exit 1
fi

# ── tool paths ────────────────────────────────────────────────────────────────
LAYER_BIN="$REPO_ROOT/cmd/BonfyreLayer/bonfyre-layer"
QUANT_BIN="$REPO_ROOT/cmd/BonfyreQuant/bonfyre-quant"
FPQX_BIN="$REPO_ROOT/cmd/BonfyreFPQX/bonfyre-fpqx"
SLI_BIN="$REPO_ROOT/cmd/BonfyreSLI/bonfyre-sli"

for bin in "$LAYER_BIN" "$QUANT_BIN" "$FPQX_BIN" "$SLI_BIN"; do
    if [[ ! -x "$bin" ]]; then
        echo "[extract_fragment] ERROR: missing binary: $bin"
        exit 1
    fi
done

# ── auto-detect ONNX path ─────────────────────────────────────────────────────
if [[ -z "$ONNX_PATH" ]]; then
    # Try standard run dirs in priority order
    CANDIDATE_ROOTS=(
        "/tmp/bonfyre-72/runs"
        "/tmp/bonfyre-diversity/runs"
        "/tmp/bonfyre-domain"
    )

    # Also respect explicit --runs-dir
    if [[ -n "$RUNS_DIR" ]]; then
        CANDIDATE_ROOTS=("$RUNS_DIR" "${CANDIDATE_ROOTS[@]}")
    fi

    for root in "${CANDIDATE_ROOTS[@]}"; do
        if [[ ! -d "$root" ]]; then continue; fi
        # Find best run for this family: prefer largest n, then cnn_dm, then ag_news
        # Pattern: <root>/<FAMILY>-C-<dataset>-<n>/train/model.onnx
        best=""
        best_n=0
        for run_dir in "$root"/${FAMILY}-C-*; do
            [[ -d "$run_dir" ]] || continue
            onnx="$run_dir/train/model.onnx"
            [[ -f "$onnx" ]] || continue
            # Extract n from dirname like T15-C-cnn_dm-1000
            n_val=$(echo "$run_dir" | grep -oE '[0-9]+$' || echo "0")
            if [[ "$n_val" -gt "$best_n" ]]; then
                best="$onnx"
                best_n="$n_val"
            fi
        done
        if [[ -n "$best" ]]; then
            ONNX_PATH="$best"
            echo "[extract_fragment] auto-detected: $ONNX_PATH"
            break
        fi
    done

    if [[ -z "$ONNX_PATH" ]]; then
        echo "[extract_fragment] ERROR: cannot find model.onnx for family $FAMILY"
        echo "  Try: --onnx /path/to/model.onnx"
        exit 1
    fi
fi

if [[ ! -f "$ONNX_PATH" ]]; then
    echo "[extract_fragment] ERROR: model.onnx not found: $ONNX_PATH"
    exit 1
fi

# ── derived paths ─────────────────────────────────────────────────────────────
RANGE_TAG=$(echo "$RANGE" | tr ':' '-')          # "0:2" → "0-2"
FRAG_DIR="/tmp/bonfyre-fragment/${FAMILY}-frag-L${RANGE_TAG}"
FRAG_ONNX="$FRAG_DIR/layer_fragment.onnx"
FRAG_BQFP="$MODELS_DIR/${FAMILY}-frag.bqfp"
PARENT_BQFP="$MODELS_DIR/${FAMILY}.bqfp"
ALIGN_OUT="$MODELS_DIR/align-${FAMILY}frag-${FAMILY}"
TEST_OUT="/tmp/bonfyre-fragment-test-${FAMILY}-$$"

mkdir -p "$FRAG_DIR" "$MODELS_DIR" "$TEST_OUT"

PASS=0; FAIL=0

echo "==================================================================="
echo " extract_fragment.sh"
echo "  family     : $FAMILY"
echo "  source     : $ONNX_PATH"
echo "  range      : $RANGE  (→ 384→256 Gemm+ReLU typical)"
echo "  output     : $FRAG_BQFP"
echo "==================================================================="
echo ""

# ── Step 1: Pull layer fragment ───────────────────────────────────────────────
echo "── Step 1: pull-layer $RANGE from ${FAMILY}.onnx ───────────────────────"
if [[ -f "$FRAG_ONNX" ]]; then
    echo "  (cached) $FRAG_ONNX"
else
    "$LAYER_BIN" pull-layer "$ONNX_PATH" --range "$RANGE" --out "$FRAG_DIR" \
        || { echo "  ERROR: pull-layer failed"; exit 1; }
fi
"$LAYER_BIN" inspect "$FRAG_ONNX" 2>&1 | grep -E "Nodes:|Parameters:|Inputs:|Outputs:" | head -6
echo ""

# ── Step 2: BQFP compress fragment ───────────────────────────────────────────
echo "── Step 2: bonfyre-quant compress fragment → ${FAMILY}-frag.bqfp ───────"
if [[ -f "$FRAG_BQFP" ]]; then
    echo "  (cached) $FRAG_BQFP"
else
    "$QUANT_BIN" compress "$FRAG_ONNX" --out "$FRAG_BQFP" \
        || { echo "  ERROR: quant compress failed"; FAIL=$((FAIL+1)); exit 1; }
fi
SZ=$(du -k "$FRAG_BQFP" | cut -f1)
echo "  ${FAMILY}-frag.bqfp: ${SZ}KB"
PASS=$((PASS+1))
echo ""

# ── Step 3: FPQx align fragment vs full model ─────────────────────────────────
echo "── Step 3: bonfyre-fpqx align fragment ↔ full $FAMILY ─────────────────"
if [[ -d "$ALIGN_OUT" ]]; then
    echo "  (cached) $ALIGN_OUT"
elif [[ ! -f "$PARENT_BQFP" ]]; then
    echo "  WARNING: parent ${FAMILY}.bqfp not found ($PARENT_BQFP)"
    echo "  Skipping alignment — run 'bonfyre-quant compress' on full model first"
    FAIL=$((FAIL+1))
else
    mkdir -p "$ALIGN_OUT"
    "$FPQX_BIN" align "$FRAG_BQFP" "$PARENT_BQFP" \
        --out "$ALIGN_OUT" --n-anchors "$N_ANCHORS" \
        || { echo "  ERROR: fpqx align failed"; FAIL=$((FAIL+1)); }
fi
if [[ -f "$ALIGN_OUT/fpqx_alignment.json" ]]; then
    COS=$(python3 -c "
import json
d = json.load(open('$ALIGN_OUT/fpqx_alignment.json'))
print(f\"cosine_mean={d.get('cosine_mean','?'):.6f}  n_anchors={d.get('n_anchors','?')}\")
" 2>/dev/null || echo "?")
    echo "  alignment: $COS"
    # Eval Procrustes preservation
    PROC=$("$FPQX_BIN" eval "$FRAG_BQFP" "$ALIGN_OUT/fpqx_alignment.json" 2>/dev/null | \
        grep "mean cosine" | awk -F: '{print $2}' | awk '{print $1}' || echo "?")
    echo "  procrustes preservation: $PROC"
    PASS=$((PASS+1))
fi
echo ""

# ── Step 4: SLI benchmark (fragment throughput) ───────────────────────────────
if [[ $NO_BENCH -eq 0 ]]; then
    echo "── Step 4: SLI benchmark (fragment) ────────────────────────────────────"
    "$SLI_BIN" bench --model "$FRAG_BQFP" --n 1000 2>&1 | grep -E "throughput|ms|vectors" || true
    echo ""
fi

# ── Step 5: Verify round-trip (write test vectors → run fragment → check) ─────
echo "── Step 5: Round-trip verification ─────────────────────────────────────"
TEST_IN="$TEST_OUT/test_vecs.bin"
TEST_FRAG_OUT="$TEST_OUT/frag_out.bin"
python3 - "$TEST_IN" <<'PYEOF'
import struct, sys, random
path = sys.argv[1]
n, dim = 4, 384
random.seed(42)
with open(path, "wb") as f:
    f.write(struct.pack("<II", n, dim))
    for _ in range(n):
        row = [random.gauss(0, 1) for _ in range(dim)]
        norm = sum(x*x for x in row)**0.5
        row = [x/norm for x in row]
        f.write(struct.pack(f"<{dim}f", *row))
print(f"  test vectors: {n} × {dim} → {path}")
PYEOF
if "$SLI_BIN" run --in "$TEST_IN" --model "$FRAG_BQFP" --out "$TEST_FRAG_OUT" 2>&1; then
    python3 - "$TEST_IN" "$TEST_FRAG_OUT" <<'PYEOF'
import struct, math, sys

def read_vecs(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return [list(struct.unpack(f"<{d}f", f.read(d*4))) for _ in range(n)], d

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na>1e-12 and nb>1e-12 else 0.0

raw, dr = read_vecs(sys.argv[1])
out, do = read_vecs(sys.argv[2])
cos_vals = [cosine(raw[i][:do], out[i]) for i in range(min(len(raw),len(out)))]
avg = sum(cos_vals)/len(cos_vals)
print(f"  fragment output: {len(out)} × {do}  raw→frag cosine_mean={avg:.4f}")
PYEOF
    PASS=$((PASS+1))
else
    echo "  WARNING: sli run failed"
    FAIL=$((FAIL+1))
fi
rm -rf "$TEST_OUT"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "==================================================================="
echo " extract_fragment.sh complete: ${FAMILY}-frag"
echo "  PASS=$PASS  FAIL=$FAIL"
echo "  fragment bqfp : $FRAG_BQFP"
if [[ -f "$ALIGN_OUT/fpqx_alignment.json" ]]; then
    echo "  alignment dir : $ALIGN_OUT"
fi
echo ""
echo " To use this fragment in demo.py:"
echo "   --models-dir $MODELS_DIR   (T${FAMILY}-frag.bqfp auto-detected by SLI)"
echo "==================================================================="
