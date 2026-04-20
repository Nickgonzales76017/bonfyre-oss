#!/usr/bin/env bash
# run_matrix.sh — run the 16-experiment collapse calibration matrix.
#
# Experiments (4 tasks × 2 datasets × 2 corpus sizes = 16):
#   Tasks:    T04 (topic-map)  T07 (chunk-boundary)
#   Datasets: ag_news  cnn_dm
#   Sizes:    250 docs  500 docs
#
# (T14 requires external NER teacher models; excluded from the baseline matrix.
#  Add it manually once you have model paths to inject via --stage-opts.)
#
# Usage
#   ./scripts/run_matrix.sh [--out-root /tmp/matrix] [--bonfyre-run bonfyre-run]
#
# Output
#   <out-root>/<task>/<dataset>-<n>/   — full pipeline output
#   <out-root>/results.tsv             — ranked summary table
#   <out-root>/results.txt             — human-readable ranked report
#
# Requirements: bonfyre-run on PATH, python3, pip install datasets sentence-transformers torch onnx onnxruntime

set -euo pipefail

# ── args ───────────────────────────────────────────────────────────────────────
OUT_ROOT="/tmp/bonfyre-matrix"
BF_RUN="bonfyre-run"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --out-root)    OUT_ROOT="$2";  shift 2 ;;
        --bonfyre-run) BF_RUN="$2";   shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_ROOT"
RESULTS_TSV="$OUT_ROOT/results.tsv"
echo -e "experiment\ttask\tdataset\tn_docs\tf1_vs_consensus\tlatency_ratio\tn_params\tpass\tcollapse_time_s" \
    > "$RESULTS_TSV"

# ── matrix definition ──────────────────────────────────────────────────────────
TASKS=("T04" "T07")
DATASETS=("ag_news" "cnn_dm")
SIZES=(250 500)

# ── helpers ────────────────────────────────────────────────────────────────────
prep_corpus() {
    local dataset=$1 n=$2 dir=$3
    if [[ -d "$dir" ]] && [[ $(ls "$dir"/*.txt 2>/dev/null | wc -l) -ge $n ]]; then
        echo "[matrix] corpus already present: $dir ($n docs)"
        return
    fi
    echo "[matrix] downloading $dataset ($n docs) → $dir"
    python3 "$SCRIPT_DIR/prep_corpus.py" --dataset "$dataset" --out "$dir" --n "$n"
}

extract_metric() {
    # extract a single key from metrics.json; returns "N/A" if absent
    local json_path=$1 key=$2
    python3 -c "
import json, sys
try:
    d = json.load(open('$json_path'))
    print(d.get('$key', 'N/A'))
except Exception:
    print('N/A')
"
}

run_experiment() {
    local task=$1 dataset=$2 n=$3
    local exp_id="${task}-${dataset}-${n}"
    local corpus_dir="$OUT_ROOT/corpus/${dataset}-${n}"
    local run_out="$OUT_ROOT/runs/${exp_id}"
    local metrics_path="$run_out/train/metrics.json"

    echo ""
    echo "════════════════════════════════════════"
    echo " EXPERIMENT: $exp_id"
    echo "════════════════════════════════════════"

    prep_corpus "$dataset" "$n" "$corpus_dir"

    if [[ -f "$metrics_path" ]]; then
        echo "[matrix] already ran — skipping (delete $run_out to rerun)"
    else
        mkdir -p "$run_out"
        # bonfyre-run resolves recipe by code and runs stages from repo root
        (cd "$REPO_ROOT" && "$BF_RUN" "$task" "$corpus_dir" --out "$run_out") \
            || echo "[matrix] WARNING: $exp_id exited non-zero (calibration mode — continuing)"
    fi

    # collect results (may be absent if pipeline failed before train stage)
    if [[ -f "$metrics_path" ]]; then
        local f1=$(extract_metric "$metrics_path" "f1_vs_consensus")
        local lr=$(extract_metric "$metrics_path" "latency_ratio")
        local np=$(extract_metric "$metrics_path" "n_params")
        local ps=$(extract_metric "$metrics_path" "pass")
        local ct=$(extract_metric "$metrics_path" "collapse_time_s")
        echo -e "${exp_id}\t${task}\t${dataset}\t${n}\t${f1}\t${lr}\t${np}\t${ps}\t${ct}" \
            >> "$RESULTS_TSV"
        echo "[matrix] $exp_id → f1=$f1  latency_ratio=$lr  pass=$ps"
    else
        echo "[matrix] $exp_id → no metrics (train stage did not complete)"
        echo -e "${exp_id}\t${task}\t${dataset}\t${n}\tN/A\tN/A\tN/A\tN/A\tN/A" \
            >> "$RESULTS_TSV"
    fi
}

# ── main loop ─────────────────────────────────────────────────────────────────
START_TS=$(date +%s)

for task in "${TASKS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for n in "${SIZES[@]}"; do
            run_experiment "$task" "$dataset" "$n"
        done
    done
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

# ── rank and report ────────────────────────────────────────────────────────────
REPORT="$OUT_ROOT/results.txt"
python3 - "$RESULTS_TSV" "$REPORT" "$ELAPSED" << 'PYEOF'
import sys, csv, math

tsv_path, report_path, elapsed_str = sys.argv[1], sys.argv[2], sys.argv[3]
elapsed = int(elapsed_str)

rows = []
with open(tsv_path) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        rows.append(row)

def safe_float(v, default=0.0):
    try: return float(v)
    except: return default

# Score: higher f1 + lower latency_ratio is better.
# score = f1 - 0.3 * latency_ratio  (latency matters but quality dominates)
for r in rows:
    f1 = safe_float(r['f1_vs_consensus'])
    lr = safe_float(r['latency_ratio'], default=1.0)
    r['_score'] = f1 - 0.3 * lr

rows.sort(key=lambda r: r['_score'], reverse=True)

lines = []
lines.append("=" * 60)
lines.append("COLLAPSE CALIBRATION MATRIX — RESULTS")
lines.append(f"Total wall time: {elapsed//60}m {elapsed%60}s")
lines.append("=" * 60)
lines.append("")
lines.append(f"{'RANK':<5} {'EXPERIMENT':<28} {'F1':>6} {'LAT_R':>6} {'PARAMS':>8} {'PASS':<6} {'SCORE':>7}")
lines.append("-" * 70)

for i, r in enumerate(rows, 1):
    f1   = r['f1_vs_consensus']
    lr   = r['latency_ratio']
    np_  = r['n_params']
    ps   = r['pass']
    sc   = f"{r['_score']:.4f}"
    lines.append(f"{i:<5} {r['experiment']:<28} {f1:>6} {lr:>6} {np_:>8} {ps:<6} {sc:>7}")

lines.append("")
lines.append("WINNERS (top 3):")
for r in rows[:3]:
    lines.append(f"  → {r['experiment']}  score={r['_score']:.4f}  f1={r['f1_vs_consensus']}  latency_ratio={r['latency_ratio']}")

lines.append("")
lines.append("Next step: rerun winners on cnn_dm if not already done,")
lines.append("then drop --calibration from those recipe JSONs.")

report = "\n".join(lines)
print(report)
with open(report_path, "w") as f:
    f.write(report + "\n")
PYEOF

echo ""
echo "[matrix] results → $RESULTS_TSV"
echo "[matrix] report  → $REPORT"
