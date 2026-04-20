#!/usr/bin/env bash
# scripts/stability_test.sh
#
# Long-loop stability test for bonfyre-sli auto-run.
# Runs 50-iteration convergence loops under two geometry regimes:
#   A — short-doc  (avg_doc_len=150 → routes to T04)
#   B — long-doc   (avg_doc_len=650 → routes to T16)
#
# For each iteration:
#   - records the routing family and cosine-delta
#   - outputs CSV: iter,regime,family,delta
#
# Usage:
#   bash scripts/stability_test.sh [models_dir] [n_iters]
#
# Defaults:
#   models_dir : /tmp/bonfyre-families
#   n_iters    : 50

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${1:-/tmp/bonfyre-families}"
N_ITERS="${2:-50}"

SLI_BIN="$REPO_ROOT/cmd/BonfyreSLI/bonfyre-sli"
OUT_ROOT="/tmp/bonfyre-stability-$$"
CSV="$OUT_ROOT/convergence.csv"

mkdir -p "$OUT_ROOT"

echo "==================================================================="
echo " bonfyre-sli long-loop stability test"
echo " models_dir : $MODELS_DIR"
echo " n_iters    : $N_ITERS"
echo " out_root   : $OUT_ROOT"
echo "==================================================================="
echo ""

# ── Helper: write corpus_stats.json ─────────────────────────────────────
write_stats() {
    local path="$1" n_docs="$2" avg_doc_len="$3" vocab_size="$4"
    cat > "$path" << STATSEOF
{"n_docs":$n_docs,"avg_doc_len":$avg_doc_len,"vocab_size":$vocab_size}
STATSEOF
}

# ── Helper: write random-like input vectors (n × dim, float32 LE) ────────
write_vectors() {
    local path="$1" n="$2" dim="$3"
    python3 -c "
import struct, math
n, d = $n, $dim
with open('$path', 'wb') as f:
    f.write(struct.pack('<II', n, d))
    for i in range(n * d):
        v = 0.1 * math.sin(i * 0.07) + 0.05 * math.cos(i * 0.13)
        f.write(struct.pack('<f', v))
"
}

# ── Helper: run auto-run and emit CSV rows ────────────────────────────────
run_regime() {
    local regime="$1" stats="$2" vecs="$3" out_dir="$4"
    mkdir -p "$out_dir"

    echo "── Regime $regime ($N_ITERS iterations) ────────────────────────────"
    # --thresh 0.0 disables convergence early-stop so all N_ITERS run
    "$SLI_BIN" auto-run \
        --stats "$stats" \
        --in    "$vecs" \
        --out   "$out_dir" \
        --loop  "$N_ITERS" \
        --chain auto \
        --fpqx  auto \
        --thresh 0.0 \
        --models-dir "$MODELS_DIR" 2>&1 \
    | tee "$out_dir/raw.log" \
    | tee "$out_dir/raw.log"

    # Parse the raw log into CSV rows
    grep "iter " "$out_dir/raw.log" | while IFS= read -r line; do
        iter=$(echo "$line"   | sed -n 's/.*iter[[:space:]]*\([0-9]*\)\/.*/\1/p')
        family=$(echo "$line" | sed -n 's/.*→[[:space:]]*\([A-Z0-9]*\)[[:space:]]*(delta.*/\1/p')
        delta=$(echo "$line"  | sed -n 's/.*delta=\([^)]*\).*/\1/p')
        [[ "$delta" == "n/a" ]] && delta=""
        [[ -n "$iter" && -n "$family" ]] && echo "$iter,$regime,$family,$delta" >> "$CSV"
    done
    echo ""
}

# ── Shared input vectors (16 × 16) ───────────────────────────────────────
VECS="$OUT_ROOT/vecs.bin"
write_vectors "$VECS" 16 16

# ── Regime A: short-doc corpus (avg_doc_len=150 → T04) ───────────────────
STATS_A="$OUT_ROOT/stats_short.json"
write_stats "$STATS_A" 500 150 5000
run_regime "A-short" "$STATS_A" "$VECS" "$OUT_ROOT/regime-a"

# ── Regime B: long-doc corpus (avg_doc_len=650 → T16) ────────────────────
STATS_B="$OUT_ROOT/stats_long.json"
write_stats "$STATS_B" 500 650 8000
run_regime "B-long" "$STATS_B" "$VECS" "$OUT_ROOT/regime-b"

# ── Summary ───────────────────────────────────────────────────────────────
echo "==================================================================="
echo " Convergence summary"
echo "==================================================================="
echo ""
echo "CSV: $CSV"
echo ""
echo "iter,regime,family,delta"
cat "$CSV"
echo ""

# Per-regime stats
for regime in A-short B-long; do
    lines=$(grep ",$regime," "$CSV" 2>/dev/null || true)
    if [[ -z "$lines" ]]; then continue; fi

    n=$(echo "$lines" | wc -l | tr -d ' ')
    # Deltas (skip empty first-iter)
    deltas=$(echo "$lines" | awk -F',' '$4!="" {print $4}')
    if [[ -n "$deltas" ]]; then
        min_d=$(echo "$deltas" | sort -n | head -1)
        max_d=$(echo "$deltas" | sort -n | tail -1)
        avg_d=$(echo "$deltas" | awk '{s+=$1;c++} END{printf "%.4f", s/c}')
        last_d=$(echo "$deltas" | tail -1)
    else
        min_d="-" ; max_d="-" ; avg_d="-" ; last_d="-"
    fi

    family=$(echo "$lines" | awk -F',' '{print $3}' | sort | uniq -c | sort -rn | awk '{print $2"("$1")"}' | head -1)
    echo "Regime $regime:"
    echo "  iters     : $n"
    echo "  family    : $family (dominant)"
    echo "  delta min : $min_d"
    echo "  delta max : $max_d"
    echo "  delta avg : $avg_d"
    echo "  delta last: $last_d"
    echo ""
done

echo "==================================================================="
echo " PASS: long-loop stability complete"
echo "==================================================================="
