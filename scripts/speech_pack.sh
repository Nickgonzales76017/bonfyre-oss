#!/usr/bin/env bash
# scripts/speech_pack.sh — end-to-end speech family pipeline
#
# Analogous to domain_pack.sh but for audio corpora.
# Runs: audio-corpus-prep → collapse (T30-C / T31-C) → BQFP → fragment →
#       align → frontier, producing a promotion-ready ASR transform family.
#
# Usage:
#   bash scripts/speech_pack.sh --domain librispeech_clean --family S01 [options]
#
# Required:
#   --domain DOMAIN     audio corpus key in prep_corpus_audio.py
#                       (librispeech_clean | librispeech_other | commonvoice_en |
#                        gigaspeech_s)
#   --family FAMILY_ID  new family ID (e.g. S01, S02, S03)
#
# Optional:
#   --n N               corpus size in utterances (default: 500)
#   --recipe CODE       bonfyre-run recipe (default: T30-C)
#   --whisper-model M   Whisper model for synth_teacher_asr (default: base)
#   --whisper-device D  Torch device for Whisper (default: cpu)
#   --out-root DIR      root output directory (default: /tmp/bonfyre-speech)
#   --models-dir DIR    BQFP/align output (default: /tmp/bonfyre-families)
#   --anchor-family ID  family to align against in frontier (default: T04)
#   --skip-corpus       skip audio corpus prep if dir already has .wav files
#   --skip-collapse     skip bonfyre-run collapse if model.onnx already exists
#   --skip-quant        skip BQFP quantization if .bqfp already exists
#   --skip-fragment     skip fragment extraction
#   --skip-align        skip FPQx alignment
#   --skip-frontier     skip frontier map regeneration
#
# Output layout:
#   <out-root>/<domain>-<n>/
#     corpus/               .wav + .txt + .label utterances
#     run/                  bonfyre-run output (model.onnx, metrics.json)
#   <models-dir>/
#     <FAMILY_ID>.bqfp
#     <FAMILY_ID>-frag.bqfp
#     align-<FAMILY_ID>-<anchor>/
#
# Environment:
#   WHISPER_MODEL     default whisper model (overridden by --whisper-model)
#   WHISPER_DEVICE    default whisper device (overridden by --whisper-device)
#   HF_TOKEN          HuggingFace token for gated datasets (gigaspeech_s)
#
# Requirements:
#   On PATH: bonfyre-run, bonfyre-quant, bonfyre-layer, bonfyre-fpqx, bonfyre-sli
#   Python: pip install datasets soundfile numpy openai-whisper \
#               sentence-transformers torch onnx onnxruntime scikit-learn
#
# macOS bash 3.2 compatible.

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── defaults ───────────────────────────────────────────────────────────────
DOMAIN=""
FAMILY=""
N=500
RECIPE="T30-C"
WHISPER_MODEL="${WHISPER_MODEL:-base}"
WHISPER_DEVICE="${WHISPER_DEVICE:-cpu}"
OUT_ROOT="/tmp/bonfyre-speech"
MODELS_DIR="/tmp/bonfyre-families"
ANCHOR_FAMILY="T04"
SKIP_CORPUS=0
SKIP_COLLAPSE=0
SKIP_QUANT=0
SKIP_FRAGMENT=0
SKIP_ALIGN=0
SKIP_FRONTIER=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)          DOMAIN="$2";         shift 2 ;;
        --family)          FAMILY="$2";         shift 2 ;;
        --n)               N="$2";              shift 2 ;;
        --recipe)          RECIPE="$2";         shift 2 ;;
        --whisper-model)   WHISPER_MODEL="$2";  shift 2 ;;
        --whisper-device)  WHISPER_DEVICE="$2"; shift 2 ;;
        --out-root)        OUT_ROOT="$2";       shift 2 ;;
        --models-dir)      MODELS_DIR="$2";     shift 2 ;;
        --anchor-family)   ANCHOR_FAMILY="$2";  shift 2 ;;
        --skip-corpus)     SKIP_CORPUS=1;       shift   ;;
        --skip-collapse)   SKIP_COLLAPSE=1;     shift   ;;
        --skip-quant)      SKIP_QUANT=1;        shift   ;;
        --skip-fragment)   SKIP_FRAGMENT=1;     shift   ;;
        --skip-align)      SKIP_ALIGN=1;        shift   ;;
        --skip-frontier)   SKIP_FRONTIER=1;     shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$DOMAIN" || -z "$FAMILY" ]]; then
    echo "Usage: speech_pack.sh --domain DOMAIN --family FAMILY_ID [options]"
    echo "  Domains: librispeech_clean | librispeech_other | commonvoice_en | gigaspeech_s"
    echo "  Example: speech_pack.sh --domain librispeech_clean --family S01"
    exit 1
fi

export WHISPER_MODEL WHISPER_DEVICE

# ── tool paths ────────────────────────────────────────────────────────────────
BF_RUN="bonfyre-run"
QUANT_BIN="$REPO_ROOT/cmd/BonfyreQuant/bonfyre-quant"
LAYER_BIN="$REPO_ROOT/cmd/BonfyreLayer/bonfyre-layer"
FPQX_BIN="$REPO_ROOT/cmd/BonfyreFPQX/bonfyre-fpqx"
SLI_BIN="$REPO_ROOT/cmd/BonfyreSLI/bonfyre-sli"

for tool in "$QUANT_BIN" "$LAYER_BIN" "$FPQX_BIN" "$SLI_BIN"; do
    if [[ ! -x "$tool" ]]; then
        echo "[speech_pack] ERROR: missing binary: $tool"
        echo "  Run: make -C $REPO_ROOT first"
        exit 1
    fi
done

if ! command -v "$BF_RUN" &>/dev/null; then
    echo "[speech_pack] ERROR: bonfyre-run not on PATH"
    exit 1
fi

# ── layout ────────────────────────────────────────────────────────────────────
PACK_DIR="$OUT_ROOT/${DOMAIN}-${N}"
CORPUS_DIR="$PACK_DIR/corpus"
RUN_DIR="$PACK_DIR/run"
ONNX_PATH="$RUN_DIR/train/model.onnx"
FRAG_DIR="$PACK_DIR/fragment"
FRAG_ONNX="$FRAG_DIR/layer_fragment.onnx"
FRAG_BQFP="$MODELS_DIR/${FAMILY}-frag.bqfp"
FAMILY_BQFP="$MODELS_DIR/${FAMILY}.bqfp"
ANCHOR_BQFP="$MODELS_DIR/${ANCHOR_FAMILY}.bqfp"
ALIGN_DIR="$MODELS_DIR/align-${FAMILY}-${ANCHOR_FAMILY}"
METRICS_OUT="$PACK_DIR/speech_pack_metrics.json"

mkdir -p "$PACK_DIR" "$CORPUS_DIR" "$MODELS_DIR"

echo "==================================================================="
echo " Bonfyre speech_pack.sh"
echo "  domain         : $DOMAIN"
echo "  family         : $FAMILY  (anchor: $ANCHOR_FAMILY)"
echo "  corpus n       : $N utterances"
echo "  recipe         : $RECIPE"
echo "  whisper model  : $WHISPER_MODEL on $WHISPER_DEVICE"
echo "  out-root       : $OUT_ROOT"
echo "  models-dir     : $MODELS_DIR"
echo "==================================================================="
echo ""

PACK_START=$(date +%s)
PASS=0
FAIL=0

# ── Step 1: Audio corpus prep ─────────────────────────────────────────────────
echo "── Step 1: Audio corpus prep ($DOMAIN × $N utterances) ─────────────────"
if [[ $SKIP_CORPUS -eq 1 ]]; then
    WAV_COUNT=$(find "$CORPUS_DIR" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$WAV_COUNT" -gt 0 ]]; then
        echo "  (skip) $CORPUS_DIR already has $WAV_COUNT .wav files"
    else
        echo "  WARNING: --skip-corpus set but no .wav files found; running prep anyway"
        SKIP_CORPUS=0
    fi
fi

if [[ $SKIP_CORPUS -eq 0 ]]; then
    python3 "$REPO_ROOT/scripts/prep_corpus_audio.py" \
        --dataset "$DOMAIN" \
        --out     "$CORPUS_DIR" \
        --n       "$N" \
        --write-labels \
        || { echo "  ERROR: audio corpus prep failed"; exit 1; }
fi

N_WAV=$(find "$CORPUS_DIR" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')
N_TXT=$(find "$CORPUS_DIR" -maxdepth 1 -name '*.txt' | wc -l | tr -d ' ')
echo "  $N_WAV .wav files, $N_TXT .txt files → $CORPUS_DIR"
echo ""

# ── Step 2: Collapse + student training ───────────────────────────────────────
echo "── Step 2: Collapse (bonfyre-run $RECIPE) ─────────────────────────────"
if [[ $SKIP_COLLAPSE -eq 1 ]] && [[ -f "$ONNX_PATH" ]]; then
    echo "  (skip) model.onnx already exists: $ONNX_PATH"
else
    mkdir -p "$RUN_DIR"
    (cd "$REPO_ROOT" && "$BF_RUN" "$RECIPE" "$CORPUS_DIR" --out "$RUN_DIR") \
        || { echo "  WARNING: bonfyre-run exited non-zero (checking for model.onnx)"; }
fi

if [[ -f "$ONNX_PATH" ]]; then
    SZ_KB=$(du -k "$ONNX_PATH" | cut -f1)
    echo "  model.onnx: ${SZ_KB}KB → $ONNX_PATH"
    PASS=$((PASS+1))
else
    echo "  ERROR: no model.onnx produced"
    FAIL=$((FAIL+1))
    exit 1
fi
echo ""

# ── Step 3: Quantize ONNX → BQFP ─────────────────────────────────────────────
echo "── Step 3: Quantize → ${FAMILY}.bqfp ──────────────────────────────────"
if [[ $SKIP_QUANT -eq 1 ]] && [[ -f "$FAMILY_BQFP" ]]; then
    echo "  (skip) already exists: $FAMILY_BQFP"
else
    "$QUANT_BIN" compress "$ONNX_PATH" --out "$FAMILY_BQFP" \
        || { echo "  ERROR: quant compress failed"; FAIL=$((FAIL+1)); exit 1; }
fi
SZ_BQFP=$(du -k "$FAMILY_BQFP" | cut -f1)
echo "  ${FAMILY}.bqfp: ${SZ_BQFP}KB"
PASS=$((PASS+1))
echo ""

# ── Step 4: Fragment extraction (layers 0:2) ──────────────────────────────────
echo "── Step 4: Fragment extraction (layers 0:2) ─────────────────────────────"
if [[ $SKIP_FRAGMENT -eq 1 ]] && [[ -f "$FRAG_BQFP" ]]; then
    echo "  (skip) fragment already exists: $FRAG_BQFP"
else
    mkdir -p "$FRAG_DIR"
    if "$LAYER_BIN" pull-layer "$ONNX_PATH" --range 0:2 --out "$FRAG_DIR" 2>&1; then
        SZ_FRAG=$(du -k "$FRAG_ONNX" 2>/dev/null | cut -f1 || echo "?")
        echo "  layer_fragment.onnx: ${SZ_FRAG}KB"
        "$QUANT_BIN" compress "$FRAG_ONNX" --out "$FRAG_BQFP" \
            || { echo "  WARNING: fragment quant failed (non-fatal)"; }
        if [[ -f "$FRAG_BQFP" ]]; then
            SZ_FRAG_BQFP=$(du -k "$FRAG_BQFP" | cut -f1)
            echo "  ${FAMILY}-frag.bqfp: ${SZ_FRAG_BQFP}KB"
            PASS=$((PASS+1))
        fi
    else
        echo "  WARNING: bonfyre-layer pull-layer failed (fragment skipped — non-fatal)"
    fi
fi
if [[ -f "$FRAG_BQFP" ]]; then
    echo "  fragment ready: $FRAG_BQFP"
fi
echo ""

# ── Step 5: FPQx alignment vs anchor ─────────────────────────────────────────
echo "── Step 5: FPQx alignment ($FAMILY ↔ $ANCHOR_FAMILY) ─────────────────"
if [[ $SKIP_ALIGN -eq 1 ]] && [[ -d "$ALIGN_DIR" ]]; then
    echo "  (skip) align dir already exists: $ALIGN_DIR"
elif [[ ! -f "$ANCHOR_BQFP" ]]; then
    echo "  WARNING: anchor $ANCHOR_BQFP not found — skipping alignment"
    echo "    (run gen_fpqx_alignments.sh first if anchor is missing)"
    FAIL=$((FAIL+1))
else
    mkdir -p "$ALIGN_DIR"
    "$FPQX_BIN" align "$FAMILY_BQFP" "$ANCHOR_BQFP" \
        --out "$ALIGN_DIR" --n-anchors 256 \
        || { echo "  ERROR: fpqx align failed"; FAIL=$((FAIL+1)); }
fi
if [[ -d "$ALIGN_DIR" ]]; then
    ALIGN_JSON="$ALIGN_DIR/fpqx_alignment.json"
    if [[ -f "$ALIGN_JSON" ]]; then
        COS=$(python3 -c "import json; d=json.load(open('$ALIGN_JSON')); print(d.get('cosine_mean','?'))" 2>/dev/null || echo "?")
        echo "  cosine_mean=${COS}"
        PASS=$((PASS+1))
    fi
fi
echo ""

# ── Step 6: ASR corpus stats (speech-specific metrics) ───────────────────────
echo "── Step 6: ASR corpus stats ─────────────────────────────────────────────"
python3 - <<PYEOF
import json, os, glob

corpus_dir = "$CORPUS_DIR"
run_dir    = "$RUN_DIR"

# Collect per-utterance duration from _corpus_stats.json
corpus_stats_path = os.path.join(corpus_dir, "_corpus_stats.json")
corpus_stats = {}
if os.path.exists(corpus_stats_path):
    corpus_stats = json.load(open(corpus_stats_path))

# Collect ASR confidence distribution from synth_teacher output
tag_dir = os.path.join(run_dir, "tag")
asr_stats_path = os.path.join(tag_dir if os.path.exists(tag_dir) else corpus_dir,
                               "_asr_corpus_stats.json")
asr_stats = {}
if os.path.exists(asr_stats_path):
    asr_stats = json.load(open(asr_stats_path))

# Count ground-truth .txt files (token count estimate for avg_utterance_len)
txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
total_tokens = 0
for tf in txt_files:
    total_tokens += len(open(tf).read().split())
avg_utterance_tokens = total_tokens / max(len(txt_files), 1)

speech_stats = {
    "n_utterances":        corpus_stats.get("n_written", len(txt_files)),
    "avg_utterance_tokens": round(avg_utterance_tokens, 1),
    "asr_clear_rate":      asr_stats.get("clear_rate", None),
    "asr_ambiguous_rate":  asr_stats.get("ambiguous_rate", None),
    "asr_noisy_rate":      asr_stats.get("noisy_rate", None),
    "whisper_model":       asr_stats.get("whisper_model", "$WHISPER_MODEL"),
}

print("  n_utterances          :", speech_stats["n_utterances"])
print("  avg_utterance_tokens  :", speech_stats["avg_utterance_tokens"])
if asr_stats:
    print(f"  asr_clear_rate        : {speech_stats['asr_clear_rate']:.2%}")
    print(f"  asr_ambiguous_rate    : {speech_stats['asr_ambiguous_rate']:.2%}")
    print(f"  asr_noisy_rate        : {speech_stats['asr_noisy_rate']:.2%}")

speech_stats_path = os.path.join("$PACK_DIR", "speech_stats.json")
with open(speech_stats_path, "w") as f:
    json.dump(speech_stats, f, indent=2)
print("  stats → " + speech_stats_path)
PYEOF
echo ""

# ── Step 7: Frontier map — register speech family ────────────────────────────
echo "── Step 7: Frontier map update ─────────────────────────────────────────"
if [[ $SKIP_FRONTIER -eq 1 ]]; then
    echo "  (skip)"
else
    METRICS_JSON="$RUN_DIR/train/metrics.json"
    F1="0.0"
    N_PARAMS="0"
    if [[ -f "$METRICS_JSON" ]]; then
        F1=$(python3 -c "import json; d=json.load(open('$METRICS_JSON')); print(d.get('f1_vs_consensus',0.0))" 2>/dev/null || echo "0.0")
        N_PARAMS=$(python3 -c "import json; d=json.load(open('$METRICS_JSON')); print(d.get('n_params',0))" 2>/dev/null || echo "0")
    fi

    DOMAIN_META="$MODELS_DIR/domain_families.json"
    python3 - <<PYEOF
import json, os, time

path = "$DOMAIN_META"
try:
    data = json.load(open(path))
except Exception:
    data = {}

data["$FAMILY"] = {
    "domain":       "$DOMAIN",
    "modality":     "speech",
    "recipe":       "$RECIPE",
    "f1":           float("$F1") if "$F1" != "0.0" else None,
    "geometry":     "asr",
    "task":         "asr-confidence",
    "corpus":       "$DOMAIN",
    "params":       int("$N_PARAMS") if "$N_PARAMS" not in ("0", "?") else 0,
    "tier":         "candidate",
    "onnx_path":    "$ONNX_PATH",
    "bqfp_path":    "$FAMILY_BQFP",
    "frag_bqfp":    "$FRAG_BQFP" if os.path.exists("$FRAG_BQFP") else None,
    "added_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

with open(path, "w") as f:
    json.dump(data, f, indent=2)

print(f"  registered $FAMILY in {path}")
PYEOF

    if python3 "$REPO_ROOT/scripts/frontier_map.py" "$MODELS_DIR" 2>/dev/null; then
        echo "  frontier.json regenerated"
        PASS=$((PASS+1))
    else
        echo "  (frontier_map.py skipped — domain_families.json updated)"
    fi
fi
echo ""

# ── Step 8: Write speech pack metrics summary ─────────────────────────────────
PACK_END=$(date +%s)
PACK_S=$((PACK_END - PACK_START))

python3 - <<PYEOF
import json, os, time

metrics = {
    "schema":        "bonfyre-speech-pack-v1",
    "domain":        "$DOMAIN",
    "family":        "$FAMILY",
    "recipe":        "$RECIPE",
    "whisper_model": "$WHISPER_MODEL",
    "n_corpus":      "$N_WAV",
    "onnx_path":     "$ONNX_PATH",
    "bqfp_path":     "$FAMILY_BQFP",
    "frag_bqfp":     "$FRAG_BQFP" if os.path.exists("$FRAG_BQFP") else None,
    "align_dir":     "$ALIGN_DIR" if os.path.exists("$ALIGN_DIR") else None,
    "anchor_family": "$ANCHOR_FAMILY",
    "pass":          $PASS,
    "fail":          $FAIL,
    "wall_s":        $PACK_S,
    "finished_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

with open("$METRICS_OUT", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"  metrics → $METRICS_OUT")
PYEOF

echo "==================================================================="
echo " speech_pack complete: $FAMILY ($DOMAIN)"
echo "  PASS=$PASS  FAIL=$FAIL  wall=${PACK_S}s"
echo "==================================================================="
echo ""
echo "Next steps:"
echo "  • Register S01 head in demo.py FAMILY_HEADS"
echo "  • Test: python3 scripts/demo.py --speech-in <audio.wav>"
echo "  • Promote: add $FAMILY to ESCALATION_CHAIN (T04 → S01 for audio inputs)"
