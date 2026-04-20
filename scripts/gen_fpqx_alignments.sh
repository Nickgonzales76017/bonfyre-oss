#!/usr/bin/env bash
# scripts/gen_fpqx_alignments.sh
#
# Generate all three cross-family FPQx alignment matrices:
#   T04 ↔ T15   (global vs global, slight distribution shift)
#   T15 ↔ T16   (global → conditional long-doc)
#   T04 ↔ T16   (direct global → long-doc bridge)
#
# Output convention (matching bonfyre-sli auto-run --fpqx auto):
#   <MODELS_DIR>/T04-T15-align.bin
#   <MODELS_DIR>/T15-T16-align.bin
#   <MODELS_DIR>/T04-T16-align.bin
#
# Usage:
#   bash scripts/gen_fpqx_alignments.sh [models_dir]
#
# Default models_dir: /tmp/bonfyre-families

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${1:-/tmp/bonfyre-families}"
FPQX="$REPO_ROOT/cmd/BonfyreFPQX/bonfyre-fpqx"

mkdir -p "$MODELS_DIR"

echo "==================================================================="
echo " bonfyre-fpqx alignment generation"
echo " models_dir : $MODELS_DIR"
echo "==================================================================="

# ── 1. Generate BQFP family files ───────────────────────────────────────

echo ""
echo "── Step 1: Generate / verify family BQFP files ──"

gen_family() {
    local fam="$1"
    local seed="${2:-42}"
    local out="$MODELS_DIR/${fam}.bqfp"
    if [[ -f "$out" ]]; then
        echo "  $fam.bqfp already exists — skipping generation"
    else
        echo "  Generating $fam.bqfp ..."
        python3 "$REPO_ROOT/scripts/gen_bqfp_family.py" \
            "$fam" "$out" --seed "$seed" --n-elements 1024
    fi
}

# Copy existing /tmp/{T04,T15}.bqfp if present, otherwise generate
for fam in T04 T15; do
    if [[ ! -f "$MODELS_DIR/${fam}.bqfp" ]] && [[ -f "/tmp/${fam}.bqfp" ]]; then
        echo "  Importing /tmp/${fam}.bqfp → $MODELS_DIR/${fam}.bqfp"
        cp "/tmp/${fam}.bqfp" "$MODELS_DIR/${fam}.bqfp"
    fi
done

gen_family T04 101
gen_family T15 202
gen_family T16 303

# Verify all three exist
for fam in T04 T15 T16; do
    if [[ ! -f "$MODELS_DIR/${fam}.bqfp" ]]; then
        echo "ERROR: $MODELS_DIR/${fam}.bqfp missing after generation" >&2
        exit 1
    fi
done
echo "  All three family BQFP files present."

# ── 2. Run bonfyre-fpqx align for all three pairs ───────────────────────

echo ""
echo "── Step 2: Compute cross-family alignment matrices ──"

align_pair() {
    local fa="$1" fb="$2"
    local out_dir="$MODELS_DIR/align-${fa}-${fb}"
    local target_bin="$MODELS_DIR/${fa}-${fb}-align.bin"
    local target_json="$MODELS_DIR/${fa}-${fb}-align.json"

    echo ""
    echo "  align $fa ↔ $fb"
    mkdir -p "$out_dir"

    "$FPQX" align \
        "$MODELS_DIR/${fa}.bqfp" \
        "$MODELS_DIR/${fb}.bqfp" \
        --out "$out_dir"

    # Rename to naming convention expected by bonfyre-sli auto-run
    cp "$out_dir/fpqx_alignment.bin"  "$target_bin"
    cp "$out_dir/fpqx_alignment.json" "$target_json"

    # Patch family_a / family_b and matrix_path in the JSON
    if command -v python3 &>/dev/null; then
        python3 - <<PYEOF
import json
with open("$target_json") as f:
    d = json.load(f)
d["family_a"] = "$fa"
d["family_b"] = "$fb"
d["matrix_path"] = "${fa}-${fb}-align.bin"
with open("$target_json", "w") as f:
    json.dump(d, f, indent=2)
PYEOF
    fi

    echo "  → $target_bin"
    echo "  → $target_json"

    # Also evaluate the alignment quality
    echo "  eval:"
    "$FPQX" eval "$MODELS_DIR/${fa}.bqfp" "$target_json" 2>&1 | sed 's/^/    /'
}

align_pair T04 T15
align_pair T15 T16
align_pair T04 T16

# ── 3. Summary ─────────────────────────────────────────────────────────

echo ""
echo "==================================================================="
echo " FPQx alignment artifacts:"
ls -lh "$MODELS_DIR"/*.bin "$MODELS_DIR"/*.json 2>/dev/null || true
echo ""
echo " Done. Use with bonfyre-sli auto-run --fpqx auto --models-dir $MODELS_DIR"
echo "==================================================================="
