#!/usr/bin/env bash
# run_matrix.sh — run the collapse calibration matrix.
#
# Experiments (2 tasks × 2 datasets × per-task sizes):
#   Tasks:    T04 (topic-map)      sizes: 250 500 1000
#             T07 (chunk-boundary) sizes: 250 500 1000 2000
#   Datasets: ag_news  cnn_dm
#
# Goal: map collapse behavior across task × corpus × scale.
# Not selecting winners — increasing resolution of the map.
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
# T04-C / T07-C: calibration variants — use Python synth teachers,
# no fastText model or bonfyre-embed ONNX install required.
TASKS=("T04-C" "T07-C")
DATASETS=("ag_news" "cnn_dm")
# Per-task sizes: T07 is the structural candidate — run all 4 scales.
# T04-C deferred at 2000 until cnn_dm cluster-mode is confirmed stable.
task_sizes() {
    case "$1" in
        T04-C) echo "250 500 1000 2000" ;;
        T07-C) echo "250 500" ;;
        *)     echo "250 500" ;;
    esac
}

# ── helpers ────────────────────────────────────────────────────────────────────
prep_corpus() {
    local dataset=$1 n=$2 dir=$3
    if [[ -d "$dir" ]] && [[ $(ls "$dir"/*.txt 2>/dev/null | wc -l) -ge $n ]]; then
        echo "[matrix] corpus already present: $dir ($n docs)"
        return
    fi
    echo "[matrix] downloading $dataset ($n docs) → $dir"
    local label_flag=""
    [[ "$dataset" == "ag_news" ]] && label_flag="--write-labels"
    python3 "$SCRIPT_DIR/prep_corpus.py" --dataset "$dataset" --out "$dir" --n "$n" $label_flag
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
        # Run from REPO_ROOT so scripts/ relative paths resolve inside bonfyre-run stages
        (cd "$REPO_ROOT" && "$BF_RUN" "$task" "$corpus_dir" --out "$run_out") \
            || echo "[matrix] WARNING: $exp_id exited non-zero (calibration mode — continuing)"
    fi

    # Strip -C suffix for task column (T04-C → T04, T07-C → T07)
    local task_base="${task%-C}"
    # collect results (may be absent if pipeline failed before train stage)
    if [[ -f "$metrics_path" ]]; then
        local f1=$(extract_metric "$metrics_path" "f1_vs_consensus")
        local lr=$(extract_metric "$metrics_path" "latency_ratio")
        local np=$(extract_metric "$metrics_path" "n_params")
        local ps=$(extract_metric "$metrics_path" "pass")
        local ct=$(extract_metric "$metrics_path" "collapse_time_s")
        echo -e "${exp_id}\t${task_base}\t${dataset}\t${n}\t${f1}\t${lr}\t${np}\t${ps}\t${ct}" \
            >> "$RESULTS_TSV"
        echo "[matrix] $exp_id → f1=$f1  latency_ratio=$lr  pass=$ps"
    else
        echo "[matrix] $exp_id → no metrics (train stage did not complete)"
        echo -e "${exp_id}\t${task_base}\t${dataset}\t${n}\tN/A\tN/A\tN/A\tN/A\tN/A" \
            >> "$RESULTS_TSV"
    fi
}

# ── main loop ─────────────────────────────────────────────────────────────────
START_TS=$(date +%s)

for task in "${TASKS[@]}"; do
    read -ra sizes <<< "$(task_sizes "$task")"
    for dataset in "${DATASETS[@]}"; do
        for n in "${sizes[@]}"; do
            run_experiment "$task" "$dataset" "$n"
        done
    done
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

# ── rank, stability, and report ────────────────────────────────────────────────
REPORT="$OUT_ROOT/results.txt"
STABILITY_TSV="$OUT_ROOT/stability.tsv"
python3 - "$RESULTS_TSV" "$REPORT" "$STABILITY_TSV" "$ELAPSED" << 'PYEOF'
import sys, csv, math, statistics

tsv_path, report_path, stability_path, elapsed_str = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
elapsed = int(elapsed_str)

rows = []
with open(tsv_path) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        rows.append(row)

def safe_float(v, default=0.0):
    try: return float(v)
    except: return default

# Primary sort score: higher f1 + lower latency_ratio is better
for r in rows:
    f1 = safe_float(r['f1_vs_consensus'])
    lr = safe_float(r['latency_ratio'], default=1.0)
    r['_score'] = f1 - 0.3 * lr

rows.sort(key=lambda r: r['_score'], reverse=True)

# ── stability analysis ────────────────────────────────────────────────────────
# For each task (T04, T07), compute:
#   score_mean, score_variance — across ALL 4 experiments for that task
#   corpus_variance            — variance across the 2 datasets (ag_news vs cnn_dm),
#                                averaged over both sample sizes
#   scale_variance             — variance across the 2 sample sizes (250 vs 500),
#                                averaged over both datasets
#
# Low variance = structurally stable = real transform family.
# High variance = sensitive to corpus or scale = lucky winner risk.

tasks = sorted(set(r['task'] for r in rows))

stability_rows = []
for task in tasks:
    task_rows = [r for r in rows if r['task'] == task]
    scores    = [r['_score'] for r in task_rows]

    score_mean = statistics.mean(scores) if scores else 0.0
    score_var  = statistics.variance(scores) if len(scores) > 1 else 0.0

    # corpus variance: for each n_docs, take score diff between datasets; average the squared diffs
    corpus_diffs = []
    for n in sorted(set(r['n_docs'] for r in task_rows)):
        by_corpus = {r['dataset']: r['_score'] for r in task_rows if r['n_docs'] == n}
        vals = list(by_corpus.values())
        if len(vals) > 1:
            corpus_diffs.append(statistics.variance(vals))
    corpus_var = statistics.mean(corpus_diffs) if corpus_diffs else 0.0

    # scale variance: for each dataset, take score diff between n_docs; average
    scale_diffs = []
    for ds in sorted(set(r['dataset'] for r in task_rows)):
        by_scale = {r['n_docs']: r['_score'] for r in task_rows if r['dataset'] == ds}
        vals = list(by_scale.values())
        if len(vals) > 1:
            scale_diffs.append(statistics.variance(vals))
    scale_var = statistics.mean(scale_diffs) if scale_diffs else 0.0

    # Stability score: penalise both variance dimensions equally
    # Higher = more stable (better)
    stability_score = score_mean - 0.5 * corpus_var - 0.5 * scale_var

    stability_rows.append({
        'task':             task,
        'score_mean':       round(score_mean, 4),
        'score_var':        round(score_var, 6),
        'corpus_var':       round(corpus_var, 6),
        'scale_var':        round(scale_var, 6),
        'stability_score':  round(stability_score, 4),
        'n_experiments':    len(task_rows),
    })

stability_rows.sort(key=lambda r: r['stability_score'], reverse=True)

# write stability.tsv
stab_fields = ['task','score_mean','score_var','corpus_var','scale_var','stability_score','n_experiments']
with open(stability_path, 'w') as f:
    f.write('\t'.join(stab_fields) + '\n')
    for r in stability_rows:
        f.write('\t'.join(str(r[k]) for k in stab_fields) + '\n')

# ── results report ────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 65)
lines.append("COLLAPSE CALIBRATION MATRIX — RESULTS")
lines.append(f"Total wall time: {elapsed//60}m {elapsed%60}s")
lines.append("=" * 65)
lines.append("")
lines.append(f"{'RANK':<5} {'EXPERIMENT':<28} {'F1':>6} {'LAT_R':>6} {'PARAMS':>8} {'PASS':<6} {'SCORE':>7}")
lines.append("-" * 65)

for i, r in enumerate(rows, 1):
    f1   = r['f1_vs_consensus']
    lr   = r['latency_ratio']
    np_  = r['n_params']
    ps   = r['pass']
    sc   = f"{r['_score']:.4f}"
    lines.append(f"{i:<5} {r['experiment']:<28} {f1:>6} {lr:>6} {np_:>8} {ps:<6} {sc:>7}")

lines.append("")
lines.append("WINNERS (top 3 by score):")
for r in rows[:3]:
    lines.append(f"  → {r['experiment']}  score={r['_score']:.4f}  f1={r['f1_vs_consensus']}  latency_ratio={r['latency_ratio']}")

lines.append("")
lines.append("=" * 65)
lines.append("STABILITY ANALYSIS (structural collapseability)")
lines.append("Low corpus_var + low scale_var = real transform family")
lines.append("=" * 65)
lines.append("")
lines.append(f"{'TASK':<8} {'MEAN':>7} {'SCORE_V':>9} {'CORP_V':>9} {'SCALE_V':>9} {'STAB':>8}")
lines.append("-" * 55)
for r in stability_rows:
    lines.append(
        f"{r['task']:<8} {r['score_mean']:>7} {r['score_var']:>9} "
        f"{r['corpus_var']:>9} {r['scale_var']:>9} {r['stability_score']:>8}"
    )

lines.append("")
lines.append("RECOMMENDED NEXT STEP:")
if stability_rows:
    winner = stability_rows[0]
    lines.append(f"  Promote {winner['task']} (stability_score={winner['stability_score']})")
    lines.append(f"  → rerun on cnn_dm at 1000 docs")
    lines.append(f"  → drop --calibration from recipes/{winner['task']}.json")
    lines.append(f"  → re-enable pass/fail gate")

report = "\n".join(lines)
print(report)
with open(report_path, "w") as f:
    f.write(report + "\n")
PYEOF

echo ""
echo "[matrix] results   → $RESULTS_TSV"
echo "[matrix] stability → $STABILITY_TSV"
echo "[matrix] report    → $REPORT"
