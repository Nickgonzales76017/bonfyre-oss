#!/usr/bin/env bash
# scripts/frontier_map.sh
#
# Generate the 5-family transform network frontier map.
# Reads all 10 FPQx alignment manifests + runs eval for Procrustes preservation.
# Outputs a markdown table + machine-readable frontier.json.
#
# Usage:
#   bash scripts/frontier_map.sh [models_dir]
#
# Default models_dir: /tmp/bonfyre-families

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${1:-/tmp/bonfyre-families}"
FPQX="$REPO_ROOT/cmd/BonfyreFPQX/bonfyre-fpqx"
OUT_JSON="$MODELS_DIR/frontier.json"

# ── Family metadata ───────────────────────────────────────────────────────
declare -A FAM_F1    FAM_GEOM    FAM_TASK    FAM_CORPUS  FAM_PARAMS

FAM_F1[T04]=0.914   ; FAM_GEOM[T04]="global"      ; FAM_TASK[T04]="topic-map"
FAM_CORPUS[T04]="ag_news"    ; FAM_PARAMS[T04]=99588

FAM_F1[T15]=0.911   ; FAM_GEOM[T15]="global"      ; FAM_TASK[T15]="topic-map"
FAM_CORPUS[T15]="cnn_dm"     ; FAM_PARAMS[T15]=100102

FAM_F1[T16]=0.931   ; FAM_GEOM[T16]="conditional" ; FAM_TASK[T16]="chunk-boundary"
FAM_CORPUS[T16]="cnn_dm"     ; FAM_PARAMS[T16]=98561

FAM_F1[T08]=0.871   ; FAM_GEOM[T08]="global"      ; FAM_TASK[T08]="topic-map"
FAM_CORPUS[T08]="cnn_dm"     ; FAM_PARAMS[T08]=24770

FAM_F1[T14]=0.823   ; FAM_GEOM[T14]="global"      ; FAM_TASK[T14]="topic-map"
FAM_CORPUS[T14]="cnn_dm"     ; FAM_PARAMS[T14]=99331

# ── Pair use-case labels ──────────────────────────────────────────────────
use_case() {
    local a="$1" b="$2"
    local key="${a}:${b}"
    case "$key" in
        T04:T15) echo "cross-corpus global bridge (same task, different training set)" ;;
        T04:T16) echo "short→long-form geometry bridge (global → conditional)" ;;
        T04:T08) echo "quality↔size tradeoff (full ↔ compact, 4× smaller)" ;;
        T04:T14) echo "parallel global refinement (same geometry, lower baseline)" ;;
        T08:T14) echo "compact↔full swap (low-F1 tier, size exploration)" ;;
        T08:T15) echo "compact T08 → quality boost to T15 global" ;;
        T08:T16) echo "compact global → long-form conditional jump" ;;
        T14:T15) echo "parallel global second-pass (T14→T15 quality upgrade)" ;;
        T14:T16) echo "low-quality global → conditional long-form escalation" ;;
        T15:T16) echo "same-corpus geometry upgrade (global → long-form, cnn_dm)" ;;
        *) echo "cross-family alignment" ;;
    esac
}

# ── Collect all pairs ─────────────────────────────────────────────────────
PAIRS=(
    "T04 T15"
    "T04 T16"
    "T04 T08"
    "T04 T14"
    "T08 T14"
    "T08 T15"
    "T08 T16"
    "T14 T15"
    "T14 T16"
    "T15 T16"
)

echo "==================================================================="
echo " bonfyre-fpqx transform network frontier map"
echo " models_dir : $MODELS_DIR"
echo "==================================================================="
echo ""

# ── Table header ──────────────────────────────────────────────────────────
printf "%-4s  %-4s  %-10s  %-12s  %-6s  %-5s  %-5s  %s\n" \
    "A" "B" "geom_A" "geom_B" "anchor" "cos" "proc" "use case"
printf "%s\n" "$(printf '─%.0s' {1..100})"

JSON_ROWS=""

for pair in "${PAIRS[@]}"; do
    A=$(echo "$pair" | awk '{print $1}')
    B=$(echo "$pair" | awk '{print $2}')

    ALIGN_JSON="$MODELS_DIR/align-${A}-${B}/fpqx_alignment.json"
    if [[ ! -f "$ALIGN_JSON" ]]; then
        # Also try reversed order
        ALIGN_JSON="$MODELS_DIR/align-${B}-${A}/fpqx_alignment.json"
        if [[ ! -f "$ALIGN_JSON" ]]; then
            printf "%-4s  %-4s  MISSING\n" "$A" "$B"
            continue
        fi
        # Swap for consistent direction
        TMP="$A"; A="$B"; B="$TMP"
    fi

    # Extract cosine_mean and n_anchors from JSON
    COS=$(python3 -c "import json; d=json.load(open('$ALIGN_JSON')); print(f\"{d['cosine_mean']:.4f}\")")
    ANC=$(python3 -c "import json; d=json.load(open('$ALIGN_JSON')); print(d['n_anchors'])")

    # Run eval for Procrustes preservation
    PROC=$("$FPQX" eval "$MODELS_DIR/${A}.bqfp" "$ALIGN_JSON" 2>/dev/null \
           | awk '/mean cosine/{print $4}')

    UC=$(use_case "$A" "$B")
    GA="${FAM_GEOM[$A]}"
    GB="${FAM_GEOM[$B]}"

    printf "%-4s  %-4s  %-10s  %-12s  %-6s  %-5s  %-5s  %s\n" \
        "$A" "$B" "$GA" "$GB" "$ANC" "$COS" "$PROC" "$UC"

    # Accumulate JSON
    [[ -n "$JSON_ROWS" ]] && JSON_ROWS+=","
    JSON_ROWS+=$(cat <<JSONEOF

    {
      "family_a": "$A",
      "family_b": "$B",
      "f1_a": ${FAM_F1[$A]},
      "f1_b": ${FAM_F1[$B]},
      "geometry_a": "$GA",
      "geometry_b": "$GB",
      "task_a": "${FAM_TASK[$A]}",
      "task_b": "${FAM_TASK[$B]}",
      "params_a": ${FAM_PARAMS[$A]},
      "params_b": ${FAM_PARAMS[$B]},
      "n_anchors": $ANC,
      "cosine_mean": $COS,
      "procrustes_preservation": $PROC,
      "use_case": "$UC"
    }
JSONEOF
)
done

echo ""

# ── Write frontier.json ───────────────────────────────────────────────────
cat > "$OUT_JSON" <<JSONEOF
{
  "schema": "bonfyre-frontier-map-v1",
  "families": ["T04","T08","T14","T15","T16"],
  "n_pairs": ${#PAIRS[@]},
  "pairs": [$JSON_ROWS
  ]
}
JSONEOF

echo "frontier.json: $OUT_JSON"
echo ""

# ── Per-family summary ────────────────────────────────────────────────────
echo "Family registry:"
printf "%-4s  %-11s  %-10s  %-14s  %-8s  %s\n" \
    "ID" "mean_f1" "geometry" "task" "params" "corpus"
printf "%s\n" "$(printf '─%.0s' {1..70})"
for fam in T04 T08 T14 T15 T16; do
    role="primary"
    [[ "$fam" == "T08" || "$fam" == "T14" ]] && role="secondary"
    printf "%-4s  %-11s  %-10s  %-14s  %-8s  %s\n" \
        "$fam" "${FAM_F1[$fam]}" "${FAM_GEOM[$fam]}" "${FAM_TASK[$fam]}" \
        "${FAM_PARAMS[$fam]}" "${FAM_CORPUS[$fam]}"
done

echo ""
echo "==================================================================="
echo " DONE: 5-family frontier map complete"
echo "==================================================================="
