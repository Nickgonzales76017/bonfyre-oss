#!/usr/bin/env bash
# scripts/run_domain_matrix.sh — multi-domain expansion matrix runner
#
# Runs domain_pack.sh for each configured domain/family pair,
# then regenerates the global frontier map and produces a summary report.
#
# This is the scaling companion to run_diversity.sh.
# Where run_diversity.sh sweeps task × corpus × scale to find new families,
# run_domain_matrix.sh deploys proven domain packs into the live mesh.
#
# Wave 1 domains (immediate, labeled corpora, no user data required):
#   finance  → T20  (financial_phrasebank, 3-class sentiment)
#   science  → T21  (scientific_papers/arxiv, cluster-mode)
#
# Wave 2 domains (short-term, also fully public):
#   legal    → T22  (lex_glue/unfair_tos, 8-class ToS classification)
#   health   → T23  (pubmed_qa/pqa_labeled, yes/no/maybe)
#
# Usage:
#   bash scripts/run_domain_matrix.sh [options]
#
# Options:
#   --wave N            run wave 1 only (default: 1), or 2 for both waves
#   --n N               corpus size per domain (default: 1000)
#   --out-root DIR      root output dir (default: /tmp/bonfyre-domain)
#   --models-dir DIR    BQFP/align dir (default: /tmp/bonfyre-families)
#   --dry-run           print what would run, don't execute
#   --skip-existing     skip domains whose model.onnx already exists
#
# Output:
#   <out-root>/domain_matrix_results.tsv   per-domain results
#   <out-root>/domain_matrix_report.txt    human-readable summary
#   <models-dir>/domain_families.json      registered domain family metadata
#   <models-dir>/frontier.json             updated global frontier map
#
# macOS bash 3.2 compatible.

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
WAVE=1
N=1000
OUT_ROOT="/tmp/bonfyre-domain"
MODELS_DIR="/tmp/bonfyre-families"
DRY_RUN=0
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --wave)         WAVE="$2";        shift 2 ;;
        --n)            N="$2";           shift 2 ;;
        --out-root)     OUT_ROOT="$2";    shift 2 ;;
        --models-dir)   MODELS_DIR="$2";  shift 2 ;;
        --dry-run)      DRY_RUN=1;        shift   ;;
        --skip-existing) SKIP_EXISTING=1; shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── domain matrix definition ──────────────────────────────────────────────────
# Format: "domain:family_id:description"
WAVE1_DOMAINS=(
    "finance:T20:financial_phrasebank 3-class sentiment"
    "science:T21:arxiv abstract cluster-mode"
)
WAVE2_DOMAINS=(
    "legal:T22:lex_glue unfair_tos 8-class ToS"
    "health:T23:pubmed_qa yes/no/maybe"
)

if [[ $WAVE -ge 2 ]]; then
    ALL_DOMAINS=("${WAVE1_DOMAINS[@]}" "${WAVE2_DOMAINS[@]}")
else
    ALL_DOMAINS=("${WAVE1_DOMAINS[@]}")
fi

mkdir -p "$OUT_ROOT" "$MODELS_DIR"

RESULTS_TSV="$OUT_ROOT/domain_matrix_results.tsv"
REPORT_TXT="$OUT_ROOT/domain_matrix_report.txt"

# Write TSV header (only if new)
if [[ ! -f "$RESULTS_TSV" ]]; then
    printf "domain\tfamily\tn_corpus\fn1_labels\pass\twall_s\tbqfp_kb\tfrag_kb\talign_cosine\n" \
        > "$RESULTS_TSV"
fi

echo "=================================================================="
echo " run_domain_matrix.sh — Bonfyre domain expansion"
echo "  wave       : $WAVE  (${#ALL_DOMAINS[@]} domains)"
echo "  corpus n   : $N"
echo "  out-root   : $OUT_ROOT"
echo "  models-dir : $MODELS_DIR"
echo "  dry-run    : $DRY_RUN"
echo "=================================================================="
echo ""

TOTAL=0; PASSED=0; FAILED=0
START_TS=$(date +%s)

for entry in "${ALL_DOMAINS[@]}"; do
    # Parse "domain:family:description"
    DOMAIN=$(echo "$entry" | cut -d: -f1)
    FAMILY=$(echo "$entry" | cut -d: -f2)
    DESC=$(echo "$entry" | cut -d: -f3-)

    TOTAL=$((TOTAL+1))
    echo "══════════════════════════════════════════════════════"
    echo " DOMAIN: $DOMAIN → $FAMILY  ($DESC)"
    echo "══════════════════════════════════════════════════════"

    MODEL_ONNX="$OUT_ROOT/${DOMAIN}-${N}/run/train/model.onnx"
    if [[ $SKIP_EXISTING -eq 1 ]] && [[ -f "$MODEL_ONNX" ]]; then
        echo "  (skip-existing) model.onnx found: $MODEL_ONNX"
        PASSED=$((PASSED+1))
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] would run: domain_pack.sh --domain $DOMAIN --family $FAMILY --n $N"
        continue
    fi

    PACK_START=$(date +%s)
    bash "$REPO_ROOT/scripts/domain_pack.sh" \
        --domain   "$DOMAIN" \
        --family   "$FAMILY" \
        --n        "$N" \
        --out-root "$OUT_ROOT" \
        --models-dir "$MODELS_DIR" \
        --skip-corpus \
        2>&1 \
    && DOMAIN_EXIT=0 || DOMAIN_EXIT=$?
    PACK_END=$(date +%s)
    WALL_S=$((PACK_END - PACK_START))

    METRICS_FILE="$OUT_ROOT/${DOMAIN}-${N}/domain_pack_metrics.json"
    if [[ -f "$METRICS_FILE" ]] && [[ $DOMAIN_EXIT -eq 0 ]]; then
        N_CORPUS=$(python3 -c "import json; d=json.load(open('$METRICS_FILE')); print(d.get('n_corpus',0))" 2>/dev/null || echo "?")
        N_LABELS=$(python3 -c "import json; d=json.load(open('$METRICS_FILE')); print(d.get('n_labels',0))" 2>/dev/null || echo "?")
        PK_PASS=$(python3 -c  "import json; d=json.load(open('$METRICS_FILE')); print(d.get('pass',0))" 2>/dev/null || echo "?")
        BQFP_KB=$(du -k "$MODELS_DIR/${FAMILY}.bqfp" 2>/dev/null | cut -f1 || echo "?")
        FRAG_KB=$(du -k "$MODELS_DIR/${FAMILY}-frag.bqfp" 2>/dev/null | cut -f1 || echo "?")
        ALIGN_JSON="$MODELS_DIR/align-${FAMILY}-T04/fpqx_alignment.json"
        COS=$(python3 -c "import json; d=json.load(open('$ALIGN_JSON')); print(round(d.get('cosine_mean',0),4))" 2>/dev/null || echo "?")

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$DOMAIN" "$FAMILY" "$N_CORPUS" "$N_LABELS" "$PK_PASS" \
            "$WALL_S" "$BQFP_KB" "$FRAG_KB" "$COS" \
            >> "$RESULTS_TSV"

        echo "  result: corpus=$N_CORPUS labels=$N_LABELS bqfp=${BQFP_KB}KB cosine=${COS} wall=${WALL_S}s"
        PASSED=$((PASSED+1))
    else
        printf "%s\t%s\tN/A\tN/A\tFAIL\t%s\tN/A\tN/A\tN/A\n" \
            "$DOMAIN" "$FAMILY" "$WALL_S" >> "$RESULTS_TSV"
        echo "  FAILED (exit=$DOMAIN_EXIT)"
        FAILED=$((FAILED+1))
    fi
    echo ""
done

END_TS=$(date +%s)
WALL_TOTAL=$((END_TS - START_TS))

# ── Regenerate frontier map ────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 ]] && [[ $PASSED -gt 0 ]]; then
    echo "── Regenerating global frontier map ─────────────────────────────────"
    python3 "$REPO_ROOT/scripts/frontier_map.py" "$MODELS_DIR" 2>&1 | tail -5
    echo ""
fi

# ── Generate report ────────────────────────────────────────────────────────────
python3 - "$RESULTS_TSV" "$REPORT_TXT" "$WAVE" "$PASSED" "$FAILED" "$WALL_TOTAL" <<'PYEOF'
import sys, csv, json

tsv_path, report_path, wave, n_pass, n_fail, wall_s = sys.argv[1:]
n_pass = int(n_pass); n_fail = int(n_fail); wall_s = int(wall_s)

rows = []
try:
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
except Exception:
    pass

lines = []
lines.append("=" * 70)
lines.append(f" Bonfyre domain_matrix report  (wave {wave})")
lines.append("=" * 70)
lines.append(f"  total domains : {n_pass + n_fail}")
lines.append(f"  passed        : {n_pass}")
lines.append(f"  failed        : {n_fail}")
lines.append(f"  wall time     : {wall_s//60}m{wall_s%60}s")
lines.append("")

if rows:
    lines.append(f"  {'domain':<10}  {'family':<6}  {'corpus':<8}  "
                 f"{'labels':<7}  {'bqfp_kb':<8}  {'frag_kb':<7}  {'cosine':<7}  {'wall_s':<7}")
    lines.append("  " + "─" * 66)
    for r in rows:
        domain  = r.get("domain", "?")
        family  = r.get("family", "?")
        n_c     = r.get("n_corpus", "?")
        n_l     = r.get("n_labels", "?")
        bqfp_kb = r.get("bqfp_kb", "?")
        frag_kb = r.get("frag_kb", "?")
        cos     = r.get("align_cosine", "?")
        ws      = r.get("wall_s", "?")
        lines.append(f"  {domain:<10}  {family:<6}  {n_c:<8}  "
                     f"{n_l:<7}  {bqfp_kb:<8}  {frag_kb:<7}  {cos:<7}  {ws:<7}")

lines.append("")
lines.append("Promotion checklist (per domain):")
lines.append("  [ ] f1_vs_consensus >= 0.85  → check run/train/metrics.json")
lines.append("  [ ] bqfp generated           → <models-dir>/<FAMILY>.bqfp")
lines.append("  [ ] fragment extracted        → <models-dir>/<FAMILY>-frag.bqfp")
lines.append("  [ ] alignment computed        → align-<FAMILY>-T04/fpqx_alignment.json")
lines.append("  [ ] frontier.json updated     → <models-dir>/frontier.json")
lines.append("  [ ] geometry_condition set    → bonfyre-model DB entry")
lines.append("  [ ] FAMILY_HEADS entry        → scripts/demo.py")
lines.append("")
lines.append("Next steps:")
lines.append("  Run: bash scripts/extract_fragment.sh --family T15")
lines.append("  Run: bash scripts/extract_fragment.sh --family T16")
lines.append("  Add T20/T21 to FAMILY_HEADS in scripts/demo.py")
lines.append("  Run: python3 scripts/demo.py --models-dir /tmp/bonfyre-families \\")
lines.append("         --metrics-out /tmp/bonfyre-domain/metrics.json")
lines.append("=" * 70)

report = "\n".join(lines)
print(report)

with open(report_path, "w") as f:
    f.write(report + "\n")
print(f"\nreport → {report_path}")
PYEOF

echo ""
echo "=================================================================="
echo " domain matrix done"
echo "  TOTAL=$TOTAL  PASSED=$PASSED  FAILED=$FAILED"
echo "  wall=${WALL_TOTAL}s"
echo "  results: $RESULTS_TSV"
echo "  report : $REPORT_TXT"
echo "=================================================================="
