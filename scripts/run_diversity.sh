#!/usr/bin/env bash
# run_diversity.sh — 6 transform families × 3 corpora × 4 scales = 72 experiments
#
# Purpose: classify tasks into global-promote, conditional-promote, hold, drop.
#
# Task families:
#   T04-C  — topic-map         (anchor, proven production path)
#   T08-C  — risk-score        (classifier fusion family)
#   T14-C  — ner-bio           (frozen; included to confirm cost/value triage)
#   T15-C  — keyword-structure (global promote candidate)
#   T16-C  — paragraph-boundary (conditional promote: long-form only)
#   T17-C  — readability-complexity (new; pure-Python teacher, zero downloads)
#
# Corpora: ag_news (short-form), cnn_dm (long-form narrative), wiki (dense entity/section)
# Scales:  250, 500, 1000, 2000
#
# Usage:
#   bash scripts/run_diversity.sh [--out-root /tmp/bonfyre-diversity] [--bonfyre-run PATH]
#
# Requirements:
#   bonfyre-run on PATH (cmd/BonfyreRun/bonfyre-run)
#   python3 + datasets + sentence-transformers + torch + scikit-learn + transformers
#
# Results:
#   <out_root>/results.tsv              — machine-readable per-experiment
#   <out_root>/results.txt              — ranked human-readable with bucket classification
#   <out_root>/stability.tsv            — per-family stability scores
#   <out_root>/frontier.tsv            — Pareto frontier (F1 vs latency_ratio)
#
# macOS bash 3.2 compatible — no associative arrays.

set -euo pipefail

# Prevent HF tokenizers deadlock on macOS forked processes
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="/tmp/bonfyre-diversity"
BF_RUN="bonfyre-run"
BF_PARAGRAPH="/tmp/bonfyre-oss/cmd/BonfyreParagraph/bonfyre-paragraph"

while [[ $# -gt 0 ]]; do
    case $1 in
        --out-root)     OUT_ROOT="$2";   shift 2 ;;
        --bonfyre-run)  BF_RUN="$2";     shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_ROOT"
RESULTS_TSV="$OUT_ROOT/results.tsv"
# Write header only if the file doesn't exist yet (preserve results on restart)
if [[ ! -f "$RESULTS_TSV" ]]; then
    echo -e "experiment\ttask\tdataset\tn_docs\tf1_vs_consensus\tlatency_ratio\tn_params\tpass\tcollapse_time_s" \
        > "$RESULTS_TSV"
fi

# ── matrix definition ──────────────────────────────────────────────────────────
TASKS=("T04-C" "T08-C" "T14-C" "T15-C" "T16-C" "T17-C")
DATASETS=("ag_news" "cnn_dm" "wiki")
SIZES=(250 500 1000 2000)

# ── helpers ────────────────────────────────────────────────────────────────────
prep_corpus() {
    local dataset=$1 n=$2 dir=$3
    local count=0
    if [[ -d "$dir" ]]; then
        count=$(find "$dir" -maxdepth 1 -name '*.txt' | wc -l | tr -d ' ')
    fi
    if [[ $count -ge $n ]]; then
        echo "[diversity] corpus already present: $dir ($count docs)"
        return 0
    fi
    echo "[diversity] preparing corpus: $dataset × $n → $dir"
    local extra_args=""
    [[ "$dataset" == "ag_news" ]] && extra_args="--write-labels"
    python3 "$REPO_ROOT/scripts/prep_corpus.py" \
        --dataset "$dataset" \
        --out     "$dir"     \
        --n       "$n"       \
        $extra_args \
    || { echo "[diversity] ERROR: corpus prep failed for $dataset/$n — skipping experiment"; return 1; }
}

extract_metric() {
    local path=$1 key=$2
    python3 -c "import json; d=json.load(open('$path')); print(d.get('$key','N/A'))" 2>/dev/null || echo "N/A"
}

run_experiment() {
    local task=$1 dataset=$2 n=$3
    local exp_id="${task}-${dataset}-${n}"
    local task_base="${task%-C}"
    local corpus_dir="$OUT_ROOT/corpus/${dataset}-${n}"
    local run_out="$OUT_ROOT/runs/${exp_id}"
    local metrics_path="$run_out/train/metrics.json"

    echo ""
    echo "════════════════════════════════════════"
    echo " EXPERIMENT: $exp_id"
    echo "════════════════════════════════════════"

    prep_corpus "$dataset" "$n" "$corpus_dir" \
        || { echo "[diversity] SKIP: $exp_id (corpus unavailable)"
             echo -e "${exp_id}\t${task_base}\t${dataset}\t${n}\tN/A\tN/A\tN/A\tN/A\tN/A" >> "$RESULTS_TSV"
             return 0; }

    if [[ -f "$metrics_path" ]]; then
        echo "[diversity] already ran — skipping (delete $run_out to rerun)"
    else
        mkdir -p "$run_out"
        # Inject bonfyre-paragraph path for T16-C via env
        local bf_para_arg=""
        if [[ "$task" == "T16-C" ]] && [[ -x "$BF_PARAGRAPH" ]]; then
            export BONFYRE_PARAGRAPH="$BF_PARAGRAPH"
        fi
        (cd "$REPO_ROOT" && "$BF_RUN" "$task" "$corpus_dir" --out "$run_out") \
            || echo "[diversity] WARNING: $exp_id exited non-zero (calibration — continuing)"
    fi

    if [[ -f "$metrics_path" ]]; then
        # skip if row already written (restart-safe dedup)
        if grep -qF "${exp_id}" "$RESULTS_TSV" 2>/dev/null; then
            echo "[diversity] $exp_id → already in results.tsv (skipping append)"
            return 0
        fi
        local f1=$(extract_metric "$metrics_path" "f1_vs_consensus")
        local lr=$(extract_metric "$metrics_path" "latency_ratio")
        local np=$(extract_metric "$metrics_path" "n_params")
        local ps=$(extract_metric "$metrics_path" "pass")
        local ct=$(extract_metric "$metrics_path" "collapse_time_s")
        echo -e "${exp_id}\t${task_base}\t${dataset}\t${n}\t${f1}\t${lr}\t${np}\t${ps}\t${ct}" \
            >> "$RESULTS_TSV"
        echo "[diversity] $exp_id → f1=$f1  latency_ratio=$lr  pass=$ps"
    else
        echo "[diversity] $exp_id → no metrics (pipeline did not complete)"
        echo -e "${exp_id}\t${task_base}\t${dataset}\t${n}\tN/A\tN/A\tN/A\tN/A\tN/A" \
            >> "$RESULTS_TSV"
    fi
}

# ── run matrix ─────────────────────────────────────────────────────────────────
START_TS=$(date +%s)

for task in "${TASKS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for n in "${SIZES[@]}"; do
            run_experiment "$task" "$dataset" "$n"
        done
    done
done

END_TS=$(date +%s)
WALL_S=$((END_TS - START_TS))
WALL_M=$((WALL_S / 60))
WALL_R=$((WALL_S % 60))

# ── generate report ────────────────────────────────────────────────────────────
python3 - "$RESULTS_TSV" "$OUT_ROOT/stability.tsv" << 'PYEOF'
import sys, csv, math

results_path   = sys.argv[1]
stability_path = sys.argv[2]

rows = []
with open(results_path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

def safe_float(v, default=-0.3):
    try:
        return float(v)
    except Exception:
        return default

def score(f1, lr):
    return round(f1 - 0.2 * lr, 4)

for r in rows:
    f1 = safe_float(r["f1_vs_consensus"])
    lr = safe_float(r["latency_ratio"])
    r["_f1"] = f1
    r["_lr"] = lr
    r["_score"] = score(f1, lr)

rows.sort(key=lambda r: r["_score"], reverse=True)

# ── bucket classification (geometry-aware) ─────────────────────────────────
def bucket(f1, passed, task, corpus):
    if passed == "True" and f1 >= 0.85:
        # Check for geometry-conditional: pass all corpora?
        return "A  [GLOBAL-PROMOTE]" if True else "A  [COND-PROMOTE]"
    if f1 >= 0.60:
        return "B  [HOLD]"
    return         "C  [DROP]"

# ── stability per task ────────────────────────────────────────────────────────
from collections import defaultdict
task_scores = defaultdict(list)
for r in rows:
    task_scores[r["task"]].append(r["_score"])

stability = {}
for t, scores in task_scores.items():
    mean = sum(scores) / len(scores)
    var  = sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1)
    # Also compute corpus variance and scale variance
    # Group by corpus and scale for variance decomposition
    by_corpus = defaultdict(list)
    by_scale  = defaultdict(list)
    for r in rows:
        if r["task"] == t:
            by_corpus[r["dataset"]].append(r["_score"])
            by_scale[r["n_docs"]].append(r["_score"])
    def group_var(groups):
        means = [sum(v)/len(v) for v in groups.values() if v]
        if len(means) < 2:
            return 0.0
        gm = sum(means) / len(means)
        return sum((m - gm) ** 2 for m in means) / max(len(means) - 1, 1)
    cv = group_var(by_corpus)
    sv = group_var(by_scale)
    stab = round(mean - var - cv, 4)
    stability[t] = {"mean": round(mean,4), "score_var": round(var,6),
                    "corpus_var": round(cv,6), "scale_var": round(sv,6),
                    "stability_score": stab}

# Write stability.tsv
with open(stability_path, "w") as f:
    f.write("task\tmean\tscore_var\tcorpus_var\tscale_var\tstability_score\n")
    for t, s in sorted(stability.items(), key=lambda x: -x[1]["stability_score"]):
        f.write(f"{t}\t{s['mean']}\t{s['score_var']}\t{s['corpus_var']}\t"
                f"{s['scale_var']}\t{s['stability_score']}\n")

# ── print report ──────────────────────────────────────────────────────────────
print("=" * 72)
print(" DIVERSITY MATRIX — RESULTS")
print(f" 6 task families × 3 corpora × 4 scales = 72 experiments")
print("=" * 72)
print()
print(f"{'RANK':<5} {'EXPERIMENT':<35} {'F1':>6} {'LAT_R':>7} {'PARAMS':>8} "
      f"{'PASS':<5} {'SCORE':>8} {'BUCKET'}")
print("-" * 72)
for i, r in enumerate(rows, 1):
    f1s = f"{r['_f1']:.4f}" if r['_f1'] > -0.2 else "N/A"
    lrs = f"{r['_lr']:.4f}" if r['_lr'] > -0.2 else "N/A"
    np_s = r.get("n_params", "N/A")
    bkt = bucket(r["_f1"], r.get("pass", "False"), r.get("task",""), r.get("dataset",""))
    print(f"{i:<5} {r['experiment']:<35} {f1s:>6} {lrs:>7} {np_s:>8} "
          f"{r.get('pass','N/A'):<5} {r['_score']:>8} {bkt}")

print()
print("=" * 72)
print(" BUCKET SUMMARY")
print("=" * 72)
for bkt_label, heading in [
    ("A  [GLOBAL-PROMOTE]",  "BUCKET A — Global promote candidates"),
    ("A  [COND-PROMOTE]",    "BUCKET A — Conditional promote candidates"),
    ("B  [HOLD]",            "BUCKET B — Hold / research"),
    ("C  [DROP]",            "BUCKET C — Drop"),
]:
    members = [r for r in rows if bucket(r["_f1"], r.get("pass","False"), r.get("task",""), r.get("dataset","")) == bkt_label]
    tasks_in_bkt = sorted(set(r["task"] for r in members))
    if tasks_in_bkt:
        print(f"\n  {heading}:")
        for t in tasks_in_bkt:
            t_rows = [r for r in members if r["task"] == t]
            avg_f1 = sum(r["_f1"] for r in t_rows) / len(t_rows)
            print(f"    {t:<10}  avg_f1={avg_f1:.3f}  n={len(t_rows)}")

print()
print("=" * 72)
print(" STABILITY ANALYSIS")
print(" Low corpus_var + low scale_var = robust transform family")
print("=" * 72)
print(f"\n{'TASK':<12} {'MEAN':>7} {'SCORE_V':>10} {'CORP_V':>9} {'SCALE_V':>9} {'STAB':>8}")
print("-" * 56)
for t, s in sorted(stability.items(), key=lambda x: -x[1]["stability_score"]):
    print(f"{t:<12} {s['mean']:>7} {s['score_var']:>10} {s['corpus_var']:>9} "
          f"{s['scale_var']:>9} {s['stability_score']:>8}")

print()
print("=" * 72)
print(" RECOMMENDED NEXT STEPS")
print("=" * 72)
promote = [t for t, s in stability.items() if s["stability_score"] > 0.7]
research = [t for t, s in stability.items() if 0.2 < s["stability_score"] <= 0.7]
kill    = [t for t, s in stability.items() if s["stability_score"] <= 0.2]
if promote:
    print(f"  → PROMOTE: {', '.join(promote)}")
    print(f"     drop --calibration, re-enable gate, compress + push")
if research:
    print(f"  → INVEST:  {', '.join(research)}")
    print(f"     needs better teacher, more data, or architecture review")
if kill:
    print(f"  → KILL:    {', '.join(kill)}")
    print(f"     not collapse-worthy with current method")
PYEOF

# ── write txt report ───────────────────────────────────────────────────────────
REPORT="$OUT_ROOT/results.txt"
python3 - "$RESULTS_TSV" "$OUT_ROOT/stability.tsv" "$WALL_M" "$WALL_R" > "$REPORT" << 'PYEOF'
import sys, csv

results_path   = sys.argv[1]
stability_path = sys.argv[2]
wall_m         = sys.argv[3]
wall_r         = sys.argv[4]

rows = []
with open(results_path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

def safe_float(v, default=-0.3):
    try:
        return float(v)
    except Exception:
        return default

for r in rows:
    f1 = safe_float(r["f1_vs_consensus"])
    lr = safe_float(r["latency_ratio"])
    r["_f1"] = f1
    r["_lr"] = lr
    r["_score"] = round(f1 - 0.2 * lr, 4)

rows.sort(key=lambda r: r["_score"], reverse=True)

def bucket(f1, passed):
    if passed == "True" and f1 >= 0.85:   return "A"
    if f1 >= 0.60:                          return "B"
    return                                         "C"

lines = []
lines.append("=" * 68)
lines.append(f"DIVERSITY CALIBRATION MATRIX — RESULTS")
lines.append(f"Total wall time: {wall_m}m {wall_r}s")
lines.append("=" * 68)
lines.append("")
lines.append(f"{'RANK':<5} {'EXPERIMENT':<32} {'F1':>6} {'LAT_R':>7} {'PASS':<5} {'SCORE':>8} {'BKT'}")
lines.append("-" * 68)
for i, r in enumerate(rows, 1):
    f1s = f"{r['_f1']:.4f}" if r['_f1'] > -0.2 else "   N/A"
    lrs = f"{r['_lr']:.4f}" if r['_lr'] > -0.2 else "    N/A"
    bkt = bucket(r["_f1"], r.get("pass","False"))
    lines.append(f"{i:<5} {r['experiment']:<32} {f1s:>6} {lrs:>7} "
                 f"{r.get('pass','N/A'):<5} {r['_score']:>8.4f}   {bkt}")

lines.append("")
lines.append("=" * 68)
lines.append("STABILITY (per task family)")
lines.append("=" * 68)
lines.append(f"\n{'TASK':<12} {'MEAN':>7} {'SCORE_V':>10} {'CORP_V':>9} {'SCALE_V':>9} {'STAB':>8}")
lines.append("-" * 52)
with open(stability_path) as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 6:
            lines.append(f"{parts[0]:<12} {parts[1]:>7} {parts[2]:>10} {parts[3]:>9} {parts[4]:>9} {parts[5]:>8}")

print("\n".join(lines))
PYEOF

echo ""
echo "[diversity] wall time: ${WALL_M}m ${WALL_R}s"
echo "[diversity] results   → $OUT_ROOT/results.tsv"
echo "[diversity] stability → $OUT_ROOT/stability.tsv"
echo "[diversity] report    → $OUT_ROOT/results.txt"

# ── frontier.tsv — Pareto frontier (F1 vs latency_ratio) ─────────────────────
python3 - "$RESULTS_TSV" "$OUT_ROOT/frontier.tsv" << 'PYEOF'
import sys, csv

results_path  = sys.argv[1]
frontier_path = sys.argv[2]

rows = []
with open(results_path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

def safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

# Only include rows with real metrics
valid = [r for r in rows if safe_float(r["f1_vs_consensus"]) is not None]
for r in valid:
    r["_f1"] = safe_float(r["f1_vs_consensus"])
    r["_lr"] = safe_float(r["latency_ratio"], 999.0)

# Pareto-dominant: a point P dominates Q if P.f1 >= Q.f1 AND P.latency <= Q.latency
# with at least one strict
def is_dominated(candidate, others):
    for o in others:
        if o is candidate:
            continue
        if o["_f1"] >= candidate["_f1"] and o["_lr"] <= candidate["_lr"]:
            if o["_f1"] > candidate["_f1"] or o["_lr"] < candidate["_lr"]:
                return True
    return False

frontier = [r for r in valid if not is_dominated(r, valid)]
frontier.sort(key=lambda r: -r["_f1"])

with open(frontier_path, "w") as f:
    f.write("experiment\ttask\tdataset\tn_docs\tf1_vs_consensus\tlatency_ratio\tscore\n")
    for r in frontier:
        score = round(r["_f1"] - 0.2 * r["_lr"], 4)
        f.write(f"{r['experiment']}\t{r['task']}\t{r['dataset']}\t{r['n_docs']}\t"
                f"{r['_f1']}\t{r['_lr']}\t{score}\n")

print(f"[frontier] {len(frontier)} Pareto-optimal experiments → {frontier_path}")
PYEOF

echo "[diversity] frontier  → $OUT_ROOT/frontier.tsv"
