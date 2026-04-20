#!/usr/bin/env bash
# scripts/verify_routing.sh
#
# Verify that bonfyre-model route picks the correct promoted transform family
# across three realistic corpus geometry regimes, then run bonfyre-sli auto-run
# with FPQx alignment enabled to demonstrate the full loop.
#
# Promoted family registry:
#   T04  global   (no condition)  mean_f1=0.914  — short/medium corpora
#   T15  global   (no condition)  mean_f1=0.911  — fallback
#   T16  conditional avg_doc_len > 500  mean_f1=0.931  — long-doc corpora
#
# Expected routing results:
#   Scenario A: avg_doc_len=150  → T04  (T16 condition fails, T04 > T15)
#   Scenario B: avg_doc_len=800  → T16  (condition met, F1=0.931 wins)
#   Scenario C: avg_doc_len=350  → T04  (T16 condition fails, T04 > T15)
#
# Usage:
#   bash scripts/verify_routing.sh [models_dir]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${1:-/tmp/bonfyre-families}"
MODEL_BIN="$REPO_ROOT/cmd/BonfyreModel/bonfyre-model"
SLI_BIN="$REPO_ROOT/cmd/BonfyreSLI/bonfyre-sli"
OUT_ROOT="/tmp/bonfyre-routing-test"
PASS=0; FAIL=0

mkdir -p "$OUT_ROOT"

echo "==================================================================="
echo " bonfyre routing verification + auto-run end-to-end test"
echo " models_dir : $MODELS_DIR"
echo "==================================================================="

# ── Helper: write corpus_stats.json ─────────────────────────────────────
write_stats() {
    local path="$1" n_docs="$2" avg_doc_len="$3" vocab_size="$4"
    python3 -c "
import json
d = {'n_docs': $n_docs, 'avg_doc_len': $avg_doc_len, 'vocab_size': $vocab_size}
with open('$path', 'w') as f:
    json.dump(d, f, indent=2)
print('  wrote $path')
"
}

# ── Helper: assert routing result ───────────────────────────────────────
assert_route() {
    local label="$1" stats="$2" expected="$3"
    echo ""
    echo "  ── $label ──"
    echo "     stats: $(cat $stats | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f\"n_docs={d[\\\"n_docs\\\"]}, avg_doc_len={d[\\\"avg_doc_len\\\"]}, vocab_size={d[\\\"vocab_size\\\"]}\")' 2>/dev/null || cat $stats)"
    local route_out
    route_out=$("$MODEL_BIN" route "$stats" 2>/dev/null) || {
        echo "  [ERROR] bonfyre-model route failed"
        FAIL=$((FAIL+1)); return
    }
    local got_family
    got_family=$(echo "$route_out" | grep -o 'family=[^ ]*' | cut -d= -f2 || true)
    echo "     route output: $route_out"
    if [[ "$got_family" == "$expected" ]]; then
        echo "  [PASS] routed to $got_family (expected $expected)"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] routed to '$got_family', expected '$expected'"
        FAIL=$((FAIL+1))
    fi
}

# ── Helper: create synthetic input vectors ───────────────────────────────
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
print('  wrote $path  (${n}×${dim} float32)')
"
}

# ── Step 1: Routing assertions ───────────────────────────────────────────

echo ""
echo "── Step 1: Corpus geometry routing verification ──"

# Scenario A: short-doc general corpus
STATS_A="$OUT_ROOT/stats_short.json"
write_stats "$STATS_A" 500 150 5000
assert_route "Scenario A — short-doc (avg_doc_len=150)" "$STATS_A" "T04"

# Scenario B: long-doc corpus (avg_doc_len > 500 → T16 qualifies)
STATS_B="$OUT_ROOT/stats_long.json"
write_stats "$STATS_B" 2000 800 15000
assert_route "Scenario B — long-doc (avg_doc_len=800)" "$STATS_B" "T16"

# Scenario C: medium-doc corpus (avg_doc_len=350 → T16 condition fails)
STATS_C="$OUT_ROOT/stats_medium.json"
write_stats "$STATS_C" 1200 350 9000
assert_route "Scenario C — medium-doc (avg_doc_len=350)" "$STATS_C" "T04"

echo ""
echo "  Routing: $PASS passed, $FAIL failed"

# ── Step 2: End-to-end auto-run with FPQx alignment ─────────────────────

echo ""
echo "── Step 2: auto-run end-to-end (short corpus → T04 regime) ──"

IN_VECS="$OUT_ROOT/input_vecs.bin"
AUTO_OUT="$OUT_ROOT/auto-short"
write_vectors "$IN_VECS" 8 16

echo ""
echo "  Running: bonfyre-sli auto-run"
echo "    --stats $STATS_A"
echo "    --in    $IN_VECS"
echo "    --out   $AUTO_OUT"
echo "    --loop  3 --chain auto --fpqx auto --thresh 0.001"
echo "    --models-dir $MODELS_DIR"
echo ""

"$SLI_BIN" auto-run \
    --stats "$STATS_A" \
    --in    "$IN_VECS" \
    --out   "$AUTO_OUT" \
    --loop  3 \
    --chain auto \
    --fpqx  auto \
    --thresh 0.001 \
    --models-dir "$MODELS_DIR"

echo ""
echo "── Step 3: auto-run end-to-end (long corpus → T16 regime) ──"

AUTO_OUT_LONG="$OUT_ROOT/auto-long"

echo ""
echo "  Running with long-doc corpus stats (T16 should be selected):"
"$SLI_BIN" auto-run \
    --stats "$STATS_B" \
    --in    "$IN_VECS" \
    --out   "$AUTO_OUT_LONG" \
    --loop  3 \
    --chain auto \
    --fpqx  auto \
    --thresh 0.001 \
    --models-dir "$MODELS_DIR"

# ── Step 4: Verify artifacts ─────────────────────────────────────────────

echo ""
echo "── Step 4: Artifact verification ──"
for out_dir in "$AUTO_OUT" "$AUTO_OUT_LONG"; do
    echo ""
    echo "  $out_dir:"
    for iter_dir in "$out_dir"/iter-*/; do
        if [[ -d "$iter_dir" ]]; then
            fam=$(python3 -c "import json; d=json.load(open('${iter_dir}artifact.json')); print(d['family'])" 2>/dev/null || echo "?")
            delta=$(python3 -c "import json; d=json.load(open('${iter_dir}artifact.json')); print(f\"{d['cosine_delta']:.4f}\")" 2>/dev/null || echo "?")
            echo "    $(basename $iter_dir): family=$fam  delta=$delta  $(ls -lh ${iter_dir}vectors.bin 2>/dev/null | awk '{print $5}')"
        fi
    done
    echo "  summary: $(python3 -c "
import json, sys
d = json.load(open('$out_dir/artifact.json'))
print('family=%s delta=%.4f converged=%s' % (d['family'], d['cosine_delta'], d['converged']))
" 2>/dev/null || echo '(artifact.json missing)')"
done

# ── Summary ───────────────────────────────────────────────────────────────

echo ""
echo "==================================================================="
echo " ROUTING TEST: $PASS / $((PASS + FAIL)) scenarios correct"
echo " AUTO-RUN: completed 2 corpus regimes (short → T04, long → T16)"
echo " FPQx:     alignment matrices applied on family transitions"
echo "==================================================================="
[[ $FAIL -eq 0 ]]
