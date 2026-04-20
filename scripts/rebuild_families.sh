#!/usr/bin/env bash
# scripts/rebuild_families.sh — Bonfyre self-rewrite trigger
#
# Re-collapses the top N families, updates fragments, refreshes alignments,
# and regenerates the frontier map.  Called by auto_evolve.py when the
# escalation rate exceeds the critical threshold, or triggered manually.
#
# This is the "rewrite" step in the self-evolution loop.  It does NOT replace
# the existing families — it re-trains them from scratch using the original
# corpus locations (or the failure corpus if --from-failures is set), then
# writes updated .bqfp + fragment files to models-dir.
#
# Design rules:
#   - Idempotent: re-run produces identical structure (just fresher weights)
#   - Reversible: originals backed up to models-dir/backup_<timestamp>/
#   - Observable: detailed log written to memory-dir/graph/rebuild_log.txt
#   - Incremental: --families selects which families to rebuild (default: all)
#
# Usage:
#   bash scripts/rebuild_families.sh [options]
#
# Options:
#   --models-dir DIR     BQFP model store (default: /tmp/bonfyre-families)
#   --memory-dir DIR     Bonfyre memory dir (default: /tmp/bonfyre-memory)
#   --out-root DIR       Collapse output root (default: /tmp/bonfyre-rebuild)
#   --families LIST      Comma-separated family IDs to rebuild (default: T04,T15,T16)
#   --n N                Corpus size per family (default: 1000)
#   --anchor T04         Anchor family for alignment (default: T04)
#   --from-failures      Re-collapse using failure corpus from memory
#   --skip-align         Skip FPQx alignment (faster, but frontier will be stale)
#   --skip-frontier      Skip frontier map regeneration
#   --dry-run            Print what would be done, do not execute
#
# macOS bash 3.2 compatible.

set -euo pipefail
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PREFIX="[rebuild_families]"

# ── Defaults ────────────────────────────────────────────────────────────────

MODELS_DIR="/tmp/bonfyre-families"
MEMORY_DIR="/tmp/bonfyre-memory"
OUT_ROOT="/tmp/bonfyre-rebuild"
FAMILIES="T04,T15,T16"
N=1000
ANCHOR="T04"
FROM_FAILURES=0
SKIP_ALIGN=0
SKIP_FRONTIER=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir)    MODELS_DIR="$2";    shift 2 ;;
        --memory-dir)    MEMORY_DIR="$2";    shift 2 ;;
        --out-root)      OUT_ROOT="$2";      shift 2 ;;
        --families)      FAMILIES="$2";      shift 2 ;;
        --n)             N="$2";             shift 2 ;;
        --anchor)        ANCHOR="$2";        shift 2 ;;
        --from-failures) FROM_FAILURES=1;    shift   ;;
        --skip-align)    SKIP_ALIGN=1;       shift   ;;
        --skip-frontier) SKIP_FRONTIER=1;    shift   ;;
        --dry-run)       DRY_RUN=1;          shift   ;;
        *) echo "$LOG_PREFIX Unknown arg: $1"; exit 1 ;;
    esac
done

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$MEMORY_DIR/graph"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rebuild_log_${TS}.txt"
touch "$LOG_FILE"

log() { echo "$LOG_PREFIX $*" | tee -a "$LOG_FILE"; }
dry() { if [[ $DRY_RUN -eq 1 ]]; then log "[DRY-RUN] $*"; return 0; fi; return 1; }
run() { if dry "would run: $*"; then return; fi; "$@" >> "$LOG_FILE" 2>&1; }

log "Rebuild started at $TS"
log "  models-dir  : $MODELS_DIR"
log "  memory-dir  : $MEMORY_DIR"
log "  out-root    : $OUT_ROOT"
log "  families    : $FAMILIES"
log "  n           : $N"
log "  anchor      : $ANCHOR"
log "  from_failures: $FROM_FAILURES"
log "  dry_run     : $DRY_RUN"

# ── Validate binaries ────────────────────────────────────────────────────────

BONFYRE_RUN="$REPO_ROOT/cmd/BonfyreRun/bonfyre-run"
BONFYRE_QUANT="$REPO_ROOT/cmd/BonfyreQuant/bonfyre-quant"
BONFYRE_LAYER="$REPO_ROOT/cmd/BonfyreLayer/bonfyre-layer"
BONFYRE_FPQX="$REPO_ROOT/cmd/BonfyreFPQX/bonfyre-fpqx"

for BIN in "$BONFYRE_RUN" "$BONFYRE_QUANT"; do
    if [[ ! -x "$BIN" ]]; then
        log "ERROR: required binary not found or not executable: $BIN"
        log "       Build it first with:  make -C cmd/$(basename $(dirname $BIN))"
        exit 1
    fi
done

# ── Backup existing models ───────────────────────────────────────────────────

BACKUP_DIR="$MODELS_DIR/backup_${TS}"
if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$BACKUP_DIR"
    log "Backing up $MODELS_DIR → $BACKUP_DIR"
    # Copy only .bqfp and .json files (fast, skip large corpus dirs)
    find "$MODELS_DIR" -maxdepth 1 -name "*.bqfp" -o -name "*.json" 2>/dev/null \
        | while read -r f; do cp "$f" "$BACKUP_DIR/" 2>/dev/null || true; done
    log "Backup complete"
else
    log "[DRY-RUN] would backup $MODELS_DIR → $BACKUP_DIR"
fi

# ── Determine corpus source ──────────────────────────────────────────────────

# Standard corpus map (family → dataset key for prep_corpus.py)
CORPUS_MAP_T04="ag_news"
CORPUS_MAP_T15="cnn_dm"
CORPUS_MAP_T16="cnn_dm"
CORPUS_MAP_T08="cnn_dm"
CORPUS_MAP_T14="cnn_dm"

corpus_for_family() {
    local FAM="$1"
    case "$FAM" in
        T04) echo "$CORPUS_MAP_T04" ;;
        T15) echo "$CORPUS_MAP_T15" ;;
        T16) echo "$CORPUS_MAP_T16" ;;
        T08) echo "$CORPUS_MAP_T08" ;;
        T14) echo "$CORPUS_MAP_T14" ;;
        *)   echo "cnn_dm" ;;  # default for domain families
    esac
}

recipe_for_family() {
    local FAM="$1"
    case "$FAM" in
        T04) echo "T04-C" ;;
        T15) echo "T15-C" ;;
        T16) echo "T16-C" ;;
        T08) echo "T08-C" ;;
        T14) echo "T14-C" ;;
        *)   echo "T04-C" ;;
    esac
}

# ── Rebuild each family ──────────────────────────────────────────────────────

mkdir -p "$OUT_ROOT"

IFS=',' read -ra FAM_LIST <<< "$FAMILIES"
REBUILT=0
FAILED=0

for FAM in "${FAM_LIST[@]}"; do
    FAM="${FAM// /}"   # trim whitespace
    log ""
    log "── Rebuilding $FAM ──────────────────────────"

    CORPUS_KEY="$(corpus_for_family "$FAM")"
    RECIPE="$(recipe_for_family "$FAM")"
    FAM_OUT="$OUT_ROOT/$FAM-$N"
    CORPUS_DIR="$FAM_OUT/corpus"
    RUN_DIR="$FAM_OUT/run"
    MODEL_ONNX="$RUN_DIR/train/model.onnx"
    BQFP_OUT="$MODELS_DIR/$FAM.bqfp"
    FRAG_OUT="$MODELS_DIR/$FAM-frag.bqfp"

    mkdir -p "$FAM_OUT" "$CORPUS_DIR" "$RUN_DIR"

    # ── Corpus prep ───────────────────────────────────────────────────
    if [[ $FROM_FAILURES -eq 1 ]]; then
        # Extract failure corpus from memory
        log "  corpus: extracting failure examples from memory"
        if ! dry "python3 $REPO_ROOT/scripts/auto_evolve.py extract-corpus --family $FAM --out $CORPUS_DIR --memory-dir $MEMORY_DIR"; then
            python3 "$REPO_ROOT/scripts/auto_evolve.py" extract-corpus \
                --family "$FAM" \
                --out "$CORPUS_DIR" \
                --memory-dir "$MEMORY_DIR" >> "$LOG_FILE" 2>&1 || true
        fi
    else
        # Standard corpus prep
        log "  corpus: $CORPUS_KEY → $CORPUS_DIR (n=$N)"
        if ! dry "prep_corpus.py $CORPUS_KEY --n $N --out $CORPUS_DIR"; then
            python3 "$REPO_ROOT/scripts/prep_corpus.py" \
                --dataset "$CORPUS_KEY" \
                --n "$N" \
                --out "$CORPUS_DIR" >> "$LOG_FILE" 2>&1 || {
                log "  corpus prep failed for $FAM — skipping"
                FAILED=$((FAILED + 1))
                continue
            }
        fi
    fi

    # Count corpus files
    N_CORPUS=0
    if [[ -d "$CORPUS_DIR" ]]; then
        N_CORPUS="$(find "$CORPUS_DIR" -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')"
    fi
    log "  corpus ready: $N_CORPUS texts"

    if [[ $N_CORPUS -eq 0 ]] && [[ $DRY_RUN -eq 0 ]]; then
        log "  ✗ no corpus texts found — skipping $FAM"
        FAILED=$((FAILED + 1))
        continue
    fi

    # ── Collapse train ────────────────────────────────────────────────
    log "  collapse: bonfyre-run $RECIPE"
    if ! dry "$BONFYRE_RUN $RECIPE $CORPUS_DIR --out $RUN_DIR"; then
        "$BONFYRE_RUN" "$RECIPE" "$CORPUS_DIR" --out "$RUN_DIR" >> "$LOG_FILE" 2>&1 || {
            log "  ✗ collapse failed for $FAM"
            FAILED=$((FAILED + 1))
            continue
        }
    fi

    if [[ ! -f "$MODEL_ONNX" ]] && [[ $DRY_RUN -eq 0 ]]; then
        log "  ✗ model.onnx not created at $MODEL_ONNX — skipping $FAM"
        FAILED=$((FAILED + 1))
        continue
    fi

    # ── Quantize ──────────────────────────────────────────────────────
    log "  quant: $MODEL_ONNX → $BQFP_OUT"
    if [[ -x "$BONFYRE_QUANT" ]]; then
        run "$BONFYRE_QUANT" "$MODEL_ONNX" "$BQFP_OUT"
    fi

    # ── Fragment extraction ───────────────────────────────────────────
    if [[ -x "$BONFYRE_LAYER" ]] && ([[ -f "$BQFP_OUT" ]] || [[ $DRY_RUN -eq 1 ]]); then
        log "  fragment: $BQFP_OUT → $FRAG_OUT"
        run "$BONFYRE_LAYER" "$BQFP_OUT" --frag "$FRAG_OUT"
    fi

    # ── Align to anchor ───────────────────────────────────────────────
    if [[ $SKIP_ALIGN -eq 0 ]] && [[ -x "$BONFYRE_FPQX" ]]; then
        ANCHOR_BQFP="$MODELS_DIR/$ANCHOR.bqfp"
        if [[ "$FAM" != "$ANCHOR" ]] && ([[ -f "$ANCHOR_BQFP" ]] || [[ $DRY_RUN -eq 1 ]]); then
            ALIGN_DIR="$MODELS_DIR/align-$FAM-$ANCHOR"
            log "  align: $FAM → $ANCHOR ($ALIGN_DIR)"
            if ! dry "$BONFYRE_FPQX align $BQFP_OUT $ANCHOR_BQFP --out $ALIGN_DIR"; then
                mkdir -p "$ALIGN_DIR"
                "$BONFYRE_FPQX" align "$BQFP_OUT" "$ANCHOR_BQFP" \
                    --out "$ALIGN_DIR" >> "$LOG_FILE" 2>&1 || \
                    log "  ⚠  alignment failed (non-fatal)"
            fi
        fi
    fi

    log "  ✓ $FAM rebuilt"
    REBUILT=$((REBUILT + 1))
done

# ── Regenerate frontier map ─────────────────────────────────────────────────

if [[ $SKIP_FRONTIER -eq 0 ]] && [[ $REBUILT -gt 0 ]]; then
    log ""
    log "── Regenerating frontier map ─────────────────"
    if ! dry "python3 $REPO_ROOT/scripts/frontier_map.py $MODELS_DIR"; then
        python3 "$REPO_ROOT/scripts/frontier_map.py" "$MODELS_DIR" >> "$LOG_FILE" 2>&1 \
            || log "⚠  frontier map regeneration failed (non-fatal)"
    fi
fi

# ── Update routing weights (resets biases now that models are fresh) ────────

if [[ $REBUILT -gt 0 ]] && [[ $DRY_RUN -eq 0 ]]; then
    log "── Resetting routing adjustments ─────────────"
    python3 "$REPO_ROOT/scripts/routing_adjust.py" \
        --memory-dir "$MEMORY_DIR" \
        --models-dir "$MODELS_DIR" >> "$LOG_FILE" 2>&1 || \
        log "⚠  routing_adjust failed (non-fatal)"
fi

# ── Write completion record ─────────────────────────────────────────────────

log ""
log "Rebuild complete: rebuilt=$REBUILT  failed=$FAILED"
log "Log: $LOG_FILE"

if [[ $DRY_RUN -eq 0 ]]; then
    DONE_TS="$(date -u +%Y%m%dT%H%M%SZ)"
    python3 - <<EOF
import json, os
path = "$MEMORY_DIR/graph/evolution_log.json"
entry = {
    "rebuilt_at": "$DONE_TS",
    "event": "rebuild",
    "families": "$FAMILIES",
    "rebuilt": $REBUILT,
    "failed": $FAILED,
    "from_failures": bool($FROM_FAILURES),
}
try:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
except Exception:
    pass
print("$LOG_PREFIX Completion record written")
EOF
fi
