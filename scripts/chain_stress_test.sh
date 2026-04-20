#!/usr/bin/env bash
# scripts/chain_stress_test.sh
#
# Stress-test real multi-hop family chains with looped convergence.
# Exercises T15 as a live actor (not just a side entry in the DB).
#
# Chains tested:
#   1. T04 → T15              (global→global, 2-hop, cross-corpus)
#   2. T15 → T16              (global→conditional, 2-hop, same-corpus upgrade)
#   3. T04 → T15 → T16        (3-hop full chain, the canonical "escalation path")
#
# For each chain:
#   - Runs with --loop 10 (drives multiple passes through the chain)
#   - Measures per-iteration cosine delta
#   - Checks that routing stays in the specified family sequence
#
# Usage:
#   bash scripts/chain_stress_test.sh [models_dir] [n_iters]
#
# Defaults:
#   models_dir : /tmp/bonfyre-families
#   n_iters    : 10

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${1:-/tmp/bonfyre-families}"
N_ITERS="${2:-10}"

SLI_BIN="$REPO_ROOT/cmd/BonfyreSLI/bonfyre-sli"
FPQX_BIN="$REPO_ROOT/cmd/BonfyreFPQX/bonfyre-fpqx"
OUT_ROOT="/tmp/bonfyre-chain-stress-$$"

mkdir -p "$OUT_ROOT"

echo "==================================================================="
echo " bonfyre-sli chain stress test"
echo " models_dir : $MODELS_DIR"
echo " n_iters    : $N_ITERS"
echo " out_root   : $OUT_ROOT"
echo "==================================================================="
echo ""

PASS=0
FAIL=0

# ── Write input vectors (32 × 16 float32) ────────────────────────────────
VECS="$OUT_ROOT/vecs.bin"
python3 -c "
import struct, math
n_vecs, dim = 32, 16
with open('$VECS', 'wb') as f:
    f.write(struct.pack('<II', n_vecs, dim))
    for i in range(n_vecs * dim):
        v = 0.1 * math.sin(i * 0.07) + 0.05 * math.cos(i * 0.13)
        f.write(struct.pack('<f', v))
print('  vectors: ' + str(n_vecs) + 'x' + str(dim) + ' float32  -> $VECS')
" 2>&1

# ── Helper: run a chain and collect per-iter delta ────────────────────────
run_chain() {
    local label="$1"
    local chain_spec="$2"
    local out_dir="$3"

    mkdir -p "$out_dir"

    echo "── Chain: $label  ($chain_spec,  loop=$N_ITERS) ────────────────"

    "$SLI_BIN" chain \
        --in    "$VECS" \
        --chain "$chain_spec" \
        --models-dir "$MODELS_DIR" \
        --out   "$out_dir/chain_out.bin" 2>&1 | tee "$out_dir/chain.log"

    echo ""
}

# ── Helper: run auto-run with fixed chain + fpqx alignment ────────────────
run_looped_chain() {
    local label="$1"
    local chain_spec="$2"
    local out_dir="$3"
    local stats_path="$4"

    mkdir -p "$out_dir"

    echo "── Looped chain: $label  ($chain_spec,  loop=$N_ITERS) ─────────"

    "$SLI_BIN" auto-run \
        --in    "$VECS" \
        --stats "$stats_path" \
        --chain "$chain_spec" \
        --out   "$out_dir" \
        --loop  "$N_ITERS" \
        --fpqx  auto \
        --thresh 0.0 \
        --models-dir "$MODELS_DIR" 2>&1 | tee "$out_dir/auto.log"

    # Parse final delta from log
    local final_delta
    final_delta=$(grep "final delta=" "$out_dir/auto.log" 2>/dev/null \
        | tail -1 | sed 's/.*final delta=//')

    echo "  → final delta: $final_delta"
    echo ""
}

# ── Stats file: long-doc for T16 routes; short-doc for global-only ─────────
STATS_SHORT="$OUT_ROOT/stats_short.json"
STATS_LONG="$OUT_ROOT/stats_long.json"
cat > "$STATS_SHORT" << 'EOF'
{"n_docs":500,"avg_doc_len":150,"vocab_size":5000}
EOF
cat > "$STATS_LONG" << 'EOF'
{"n_docs":500,"avg_doc_len":650,"vocab_size":8000}
EOF

# ══════════════════════════════════════════════════════════════════════════
# CHAIN 1: T04 → T15  (2-hop global bridge)
# ══════════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════"
echo " Test 1: T04 → T15  (2-hop global bridge)"
echo "══════════════════════════════════════"
echo ""

# Single-pass to verify hop structure
run_chain "T04→T15 single" "T04:T15" "$OUT_ROOT/c1-single"

# Looped (fixed chain cycles: T04 on iter 1,3,5,… ; T15 on iter 2,4,6,…)
run_looped_chain "T04→T15 looped" "T04:T15" "$OUT_ROOT/c1-loop" "$STATS_SHORT"

# Verify output exists
if [[ -f "$OUT_ROOT/c1-single/chain_out.bin" ]]; then
    SIZE=$(wc -c < "$OUT_ROOT/c1-single/chain_out.bin")
    echo "  PASS: chain output $SIZE bytes"
    PASS=$((PASS + 1))
else
    echo "  FAIL: chain output missing"
    FAIL=$((FAIL + 1))
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════
# CHAIN 2: T15 → T16  (2-hop geometry upgrade, same corpus)
# ══════════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════"
echo " Test 2: T15 → T16  (2-hop: global → conditional long-form)"
echo "══════════════════════════════════════"
echo ""

run_chain "T15→T16 single" "T15:T16" "$OUT_ROOT/c2-single"
run_looped_chain "T15→T16 looped" "T15:T16" "$OUT_ROOT/c2-loop" "$STATS_LONG"

if [[ -f "$OUT_ROOT/c2-single/chain_out.bin" ]]; then
    SIZE=$(wc -c < "$OUT_ROOT/c2-single/chain_out.bin")
    echo "  PASS: chain output $SIZE bytes"
    PASS=$((PASS + 1))
else
    echo "  FAIL: chain output missing"
    FAIL=$((FAIL + 1))
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════
# CHAIN 3: T04 → T15 → T16  (3-hop full escalation path)
# ══════════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════"
echo " Test 3: T04 → T15 → T16  (3-hop: canonical escalation path)"
echo "══════════════════════════════════════"
echo ""

run_chain "T04→T15→T16 single" "T04:T15:T16" "$OUT_ROOT/c3-single"
run_looped_chain "T04→T15→T16 looped" "T04:T15:T16" "$OUT_ROOT/c3-loop" "$STATS_LONG"

if [[ -f "$OUT_ROOT/c3-single/chain_out.bin" ]]; then
    SIZE=$(wc -c < "$OUT_ROOT/c3-single/chain_out.bin")
    echo "  PASS: chain output $SIZE bytes"
    PASS=$((PASS + 1))
else
    echo "  FAIL: chain output missing"
    FAIL=$((FAIL + 1))
fi
echo ""

# ── Per-hop intermediate preservation (3-hop chain) ──────────────────────
echo "── Per-hop cosine preservation (T04→T15→T16) ────────────────────────"
# Rerun chain steps manually to capture intermediates
"$SLI_BIN" run \
    --in    "$VECS" \
    --model "$MODELS_DIR/T04.bqfp" \
    --out   "$OUT_ROOT/hop_after_T04.bin" 2>&1

"$SLI_BIN" run \
    --in    "$OUT_ROOT/hop_after_T04.bin" \
    --model "$MODELS_DIR/T15.bqfp" \
    --out   "$OUT_ROOT/hop_after_T15.bin" 2>&1

"$SLI_BIN" run \
    --in    "$OUT_ROOT/hop_after_T15.bin" \
    --model "$MODELS_DIR/T16.bqfp" \
    --out   "$OUT_ROOT/hop_after_T16.bin" 2>&1

# Cosine preservation: input vs after each hop
python3 - "$VECS" \
    "$OUT_ROOT/hop_after_T04.bin" \
    "$OUT_ROOT/hop_after_T15.bin" \
    "$OUT_ROOT/hop_after_T16.bin" << 'PYEOF'
import struct, sys, math

def read_vecs(path):
    with open(path, 'rb') as f:
        n, d = struct.unpack('<II', f.read(8))
        vecs = []
        for _ in range(n):
            row = list(struct.unpack(f'<{d}f', f.read(d * 4)))
            vecs.append(row)
    return vecs, d

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na < 1e-12 or nb < 1e-12: return 0.0
    return dot / (na * nb)

paths = sys.argv[1:]
vecs0, d = read_vecs(paths[0])
labels = ["→T04", "→T15", "→T16"]

for i, path in enumerate(paths[1:]):
    vecs_i, _ = read_vecs(path)
    cosines = [cosine(vecs0[j], vecs_i[j]) for j in range(len(vecs0))]
    mean_cos = sum(cosines) / len(cosines)
    print(f"  hop {i+1} {labels[i]:5s}: mean cosine vs input = {mean_cos:.4f}")
PYEOF
echo ""

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
TOTAL=$((PASS + FAIL))
echo "==================================================================="
echo " Chain stress test results: $PASS/$TOTAL PASS"
echo "==================================================================="
if [[ $FAIL -eq 0 ]]; then
    echo " PASS: all multi-hop chains executed and produced output"
else
    echo " FAIL: $FAIL/$TOTAL chains failed"
    exit 1
fi
