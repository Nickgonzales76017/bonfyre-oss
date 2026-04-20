#!/usr/bin/env python3
"""
scripts/demo.py — Bonfyre end-to-end demo

Takes one or more text inputs, runs the full transform mesh pipeline:

  1. Embed text     — MiniLM-L6-v2 → 384-dim float32 vectors
  2. Corpus stats   — compute avg_doc_len, vocab_size for routing
  3. Fragment run   — T04-frag pre-process (cheap, sub-model)
  4. Route          — pick full family (T04 / T15 / T16) from stats
  5. Chain + loop   — auto-run with FPQx alignment, N iterations
  6. Read output    — per-iteration artifact, convergence curve
  7. Classify       — argmax over logit dims for topic labels

Usage:
    python3 scripts/demo.py "Your text here"
    python3 scripts/demo.py --text-file docs/sample.txt
    python3 scripts/demo.py "Short doc" "Another doc" "A third one"
    python3 scripts/demo.py --loop 5 --models-dir /tmp/bonfyre-families "text..."

Output is printed to stdout in a readable summary.
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLI_BIN     = os.path.join(REPO_ROOT, "cmd", "BonfyreSLI", "bonfyre-sli")
MODEL_BIN   = os.path.join(REPO_ROOT, "cmd", "BonfyreModel", "bonfyre-model")
FPQX_BIN    = os.path.join(REPO_ROOT, "cmd", "BonfyreFPQX", "bonfyre-fpqx")

# ── Topic labels (T04 task: 4-class topic-map, ag_news labels) ───────────
TOPIC_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
CHUNK_LABELS = {0: "continuous", 1: "boundary"}

# ── Per-family ONNX heads + escalation chain ─────────────────────────────
FAMILY_HEADS = {
    "T04": {
        "path": "/tmp/bonfyre-72/runs/T04-C-ag_news-1000/train/model.onnx",
        "n_class": 4,
        "labels": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    },
    "T15": {
        "path": "/tmp/bonfyre-72/runs/T15-C-cnn_dm-1000/train/model.onnx",
        "n_class": 6,
        "labels": {0: "Politics", 1: "Tech", 2: "Business", 3: "Health", 4: "World", 5: "Sports"},
    },
    # ── Speech ASR families (populated once speech_pack.sh has run) ──────
    # S01: short-form clean ASR (T30-C collapse on librispeech_clean)
    "S01": {
        "path": "/tmp/bonfyre-speech/librispeech_clean-500/run/train/model.onnx",
        "n_class": 3,
        "labels": {0: "ambiguous", 1: "clear", 2: "noisy"},
    },
    # S02: noisy/diverse ASR (T31-C collapse on librispeech_other/commonvoice_en)
    "S02": {
        "path": "/tmp/bonfyre-speech/librispeech_other-500/run/train/model.onnx",
        "n_class": 3,
        "labels": {0: "ambiguous", 1: "clear", 2: "noisy"},
    },
}
ESCALATION_CHAIN = {"T04": "T15", "T15": "T16", "T16": None,
                    "S01": "S02", "S02": None}

# ── Speech escalation: ASR family chain ───────────────────────────────────
ASR_ESCALATION_CHAIN = {"S01": "S02", "S02": None}

# ── bonfyre-transcribe binary path ───────────────────────────────────────
TRANSCRIBE_BIN = os.path.join(REPO_ROOT, "cmd", "BonfyreTranscribe", "bonfyre-transcribe")

# ── I/O helpers ───────────────────────────────────────────────────────────

def write_vecs(path, vecs):
    """Write list-of-lists as BQFP-SLI input binary: [n, dim, float32...]"""
    n = len(vecs)
    d = len(vecs[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        for row in vecs:
            f.write(struct.pack(f"<{d}f", *row))


def read_vecs(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        vecs = []
        for _ in range(n):
            vecs.append(list(struct.unpack(f"<{d}f", f.read(d * 4))))
    return vecs, d


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 1e-12 and nb > 1e-12 else 0.0


def argmax(v):
    return max(range(len(v)), key=lambda i: v[i])


# ── Corpus stats ──────────────────────────────────────────────────────────

def compute_stats(texts):
    n_docs    = len(texts)
    avg_len   = sum(len(t) for t in texts) / max(n_docs, 1)
    vocab     = set()
    for t in texts:
        vocab.update(t.lower().split())
    return {"n_docs": n_docs, "avg_doc_len": round(avg_len, 1), "vocab_size": len(vocab)}


# ── Embed texts with MiniLM ───────────────────────────────────────────────

def embed_texts(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embs.tolist()


# ── Route via bonfyre-model ───────────────────────────────────────────────

def route(stats_path, frontier_path=None, from_family=None):
    cmd = [MODEL_BIN, "route", stats_path]
    if frontier_path and os.path.exists(frontier_path) and from_family:
        cmd += ["--frontier", frontier_path, "--from", from_family]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout.strip()
    family = "T04"
    cosine_bias = None
    for tok in out.split():
        if tok.startswith("family="):      family      = tok[7:]
        if tok.startswith("cosine_bias="): cosine_bias = float(tok[12:])
    return family, cosine_bias


# ── Run bonfyre-sli run ───────────────────────────────────────────────────

def sli_run(in_path, model_path, out_path):
    result = subprocess.run(
        [SLI_BIN, "run", "--in", in_path, "--model", model_path, "--out", out_path],
        capture_output=True, text=True
    )
    return result.returncode == 0


# ── Run bonfyre-sli auto-run ──────────────────────────────────────────────

def sli_auto_run(in_path, stats_path, out_dir, models_dir, loop, chain, fpqx, thresh):
    result = subprocess.run(
        [SLI_BIN, "auto-run",
         "--in",         in_path,
         "--stats",      stats_path,
         "--out",        out_dir,
         "--loop",       str(loop),
         "--chain",      chain,
         "--fpqx",       fpqx,
         "--thresh",     str(thresh),
         "--models-dir", models_dir],
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


# ── Read final SLI output vectors ────────────────────────────────────────

def read_last_iter(out_dir, loop):
    """Find the last iter directory and read its vectors."""
    for i in range(loop, 0, -1):
        vec_path = os.path.join(out_dir, f"iter-{i}", "vectors.bin")
        art_path = os.path.join(out_dir, f"iter-{i}", "artifact.json")
        if os.path.exists(vec_path):
            vecs, d = read_vecs(vec_path)
            art = json.load(open(art_path)) if os.path.exists(art_path) else {}
            return i, vecs, art
    return 0, [], {}


# ── Collect per-iter delta from SLI log ──────────────────────────────────

def parse_iter_log(log):
    lines = []
    for line in log.splitlines():
        if "iter " in line and "route →" in line:
            import re
            m = re.search(r"iter\s+(\d+)/\d+:\s+route\s+→\s+(\w+)\s+\(delta=([^)]+)\)", line)
            if m:
                it = int(m.group(1))
                fam = m.group(2)
                delta_s = m.group(3)
                delta = float(delta_s) if delta_s != "n/a" else None
                lines.append((it, fam, delta))
    return lines


# ── Classify via ONNX head (true logits) ─────────────────────────────────

def classify_onnx(embs, onnx_path):
    """
    Run true ONNX forward pass on 384-dim embeddings.
    Returns list of (label, confidence) or None if unavailable.
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        return None
    if not os.path.exists(onnx_path):
        return None
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    data = np.array(embs, dtype=np.float32)
    logits = sess.run(None, {input_name: data})[0]  # [n, n_classes]
    # softmax
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs  = exp_l / exp_l.sum(axis=1, keepdims=True)
    results = []
    for p in probs:
        idx = int(p.argmax())
        results.append((TOPIC_LABELS.get(idx, f"class_{idx}"), float(p[idx])))
    return results


def classify_onnx_family(embs, family):
    """
    Family-aware ONNX classify: uses FAMILY_HEADS dict for path and label map.
    Falls back to T04 head if family not in FAMILY_HEADS.
    """
    head = FAMILY_HEADS.get(family)
    if head is None:
        fallback = FAMILY_HEADS.get("T04")
        if fallback is None:
            return None
        head = fallback
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        return None
    if not os.path.exists(head["path"]):
        return None
    sess = ort.InferenceSession(head["path"], providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    data = np.array(embs, dtype=np.float32)
    logits = sess.run(None, {input_name: data})[0]
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    label_map = head["labels"]
    results = []
    for p in probs:
        idx = int(p.argmax())
        results.append((label_map.get(idx, f"class_{idx}"), float(p[idx])))
    return results


# ── Speech helpers ────────────────────────────────────────────────────────

def _transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file to text.

    Primary path: bonfyre-transcribe C binary (whisper.h backend).
    Fallback path: openai-whisper Python package (if binary unavailable).

    Returns dict with keys:
      transcript       (str)  full transcript text
      avg_logprob      (float) weighted avg_logprob across segments
      num_segments     (int)
      first_segment_ms (int)  end time of first segment in ms
      whisper_model    (str)
      source           (str)  "bonfyre-transcribe" | "openai-whisper"
    """
    import math, time

    if not os.path.exists(audio_path):
        print(f"[speech] ERROR: audio file not found: {audio_path}")
        return {}

    # Primary: bonfyre-transcribe C binary
    if os.path.exists(TRANSCRIBE_BIN):
        out_json = audio_path + ".transcription.json"
        t0 = time.monotonic()
        result = subprocess.run(
            [TRANSCRIBE_BIN, audio_path, "--out-json", out_json],
            capture_output=True, text=True
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if result.returncode == 0 and os.path.exists(out_json):
            try:
                data = json.load(open(out_json))
                segments = data.get("segments", [])
                total_tokens = sum(len(s.get("tokens", [])) or 1 for s in segments)
                w_lp = sum(
                    s.get("avg_logprob", -1.0) * (len(s.get("tokens", [])) or 1)
                    for s in segments
                )
                avg_lp = w_lp / max(total_tokens, 1) if segments else -1.0
                first_ms = int(segments[0].get("end", 0.0) * 1000) if segments else 0
                return {
                    "transcript":       data.get("text", "").strip(),
                    "avg_logprob":      round(avg_lp, 4),
                    "num_segments":     len(segments),
                    "first_segment_ms": first_ms,
                    "whisper_model":    data.get("model", "unknown"),
                    "source":           "bonfyre-transcribe",
                    "elapsed_ms":       elapsed_ms,
                }
            except Exception as exc:
                print(f"[speech] WARNING: could not parse bonfyre-transcribe JSON: {exc}")

    # Fallback: openai-whisper Python
    try:
        import whisper, math as _math
    except ImportError:
        print("[speech] ERROR: bonfyre-transcribe not found and openai-whisper not installed")
        print("  Install: pip install openai-whisper  or  build bonfyre-transcribe")
        return {}

    model_name = os.environ.get("WHISPER_MODEL", "base")
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    print(f"[speech] fallback: loading whisper '{model_name}' on {device} …")
    t0 = time.monotonic()
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(audio_path, language="en", fp16=False, verbose=False)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    segments = result.get("segments", [])
    total_tokens = sum(len(s.get("tokens", [])) or 1 for s in segments)
    w_lp = sum(
        s.get("avg_logprob", -1.0) * (len(s.get("tokens", [])) or 1)
        for s in segments
    )
    avg_lp = w_lp / max(total_tokens, 1) if segments else -1.0
    first_ms = int(segments[0].get("end", 0.0) * 1000) if segments else 0

    return {
        "transcript":       result.get("text", "").strip(),
        "avg_logprob":      round(avg_lp, 4),
        "num_segments":     len(segments),
        "first_segment_ms": first_ms,
        "whisper_model":    model_name,
        "source":           "openai-whisper",
        "elapsed_ms":       elapsed_ms,
    }


def _compute_wer(hyp: str, ref: str) -> float:
    """
    Compute Word Error Rate (WER) = (S + D + I) / N.
    Pure-Python Levenshtein on word tokens (no deps required).
    """
    def tokenize(s):
        return s.lower().split()

    h = tokenize(hyp)
    r = tokenize(ref)
    if not r:
        return 0.0 if not h else 1.0

    # Dynamic programming edit distance
    d = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(len(h) + 1):
        d[i][0] = i
    for j in range(len(r) + 1):
        d[0][j] = j
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            if h[i - 1] == r[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return round(d[len(h)][len(r)] / len(r), 4)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Bonfyre end-to-end demo")
    ap.add_argument("texts", nargs="*", help="Text inputs to process")
    ap.add_argument("--text-file", help="File with one text per line")
    ap.add_argument("--loop",       type=int, default=5)
    ap.add_argument("--thresh",     type=float, default=0.0)
    ap.add_argument("--models-dir", default="/tmp/bonfyre-families")
    ap.add_argument("--chain",      default="fragment:auto")
    ap.add_argument("--fpqx",       default="auto")
    ap.add_argument("--no-fragment", action="store_true",
                    help="Skip fragment pre-process step")
    ap.add_argument("--onnx-model",  default=None,
                    help="Path to ONNX classifier head (default: T04 ag_news-1000)")
    ap.add_argument("--conf-exit",    type=float, default=None,
                    help="Stop looping early when ALL inputs reach this ONNX confidence (0-1)")
    ap.add_argument("--frag-exit",    type=float, default=None,
                    help="Exit after fragment preflight if ALL inputs reach this confidence (0-1)")
    ap.add_argument("--escalate",     action="store_true",
                    help="Enable mid-loop family escalation on confidence drop (requires --conf-exit)")
    ap.add_argument("--escalate-drop", type=float, default=0.05,
                    help="Avg-confidence drop threshold that triggers escalation (default: 0.05)")
    ap.add_argument("--metrics-out",  default=None,
                    help="Write run metrics JSON to this path (appends if file exists as array)")
    ap.add_argument("--speech-in",    default=None,
                    help="Path to audio file (.wav/.mp3/.flac). Transcribes via "
                         "bonfyre-transcribe (or python whisper fallback), then runs "
                         "the full text adaptive pipeline on the transcript.")
    ap.add_argument("--speech-gt",    default=None,
                    help="Ground-truth transcript text or .txt file for WER computation "
                         "(only used with --speech-in)")
    ap.add_argument("--segment-route", action="store_true",
                    help="With --speech-in: run the ASR confidence gate on each Whisper "
                         "segment independently, report per-segment escalation, then "
                         "process the full transcript through the text pipeline.")
    ap.add_argument("--memory-dir", default=None,
                    help="Write run metrics to Bonfyre transform memory (SQLite). "
                         "Creates <memory_dir>/memory.db if absent. "
                         "When set, every run is automatically ingested for "
                         "self-evolution (failure detection, routing adjustment).")
    ap.add_argument("--auto-evolve", action="store_true",
                    help="After recording to memory, run one auto_evolve cycle: "
                         "detect failures, adjust routing, and discover paths. "
                         "Requires --memory-dir.")
    args = ap.parse_args()

    texts = list(args.texts)
    if args.text_file:
        with open(args.text_file) as f:
            texts += [ln.rstrip() for ln in f if ln.strip()]

    # ── Speech pre-processing: audio → transcript ─────────────────────────
    _speech_meta = None
    if args.speech_in:
        _speech_meta = _transcribe_audio(args.speech_in)
        if _speech_meta and _speech_meta.get("transcript"):
            texts = [_speech_meta["transcript"]]
            print(f"[speech] transcribed {args.speech_in}")
            print(f"         transcript : {_speech_meta['transcript'][:80]}")
            print(f"         avg_logprob: {_speech_meta.get('avg_logprob', 'n/a')}")
            print(f"         first_seg_ms: {_speech_meta.get('first_segment_ms', 'n/a')}ms")

            # ── Per-segment ASR gate (--segment-route) ───────────────
            if args.segment_route and _speech_meta.get("segments"):
                print()
                print("[speech/segment-route] per-segment ASR confidence gate:")
                print(f"  {'seg':<4} {'logprob':>8} {'family':>5} {'label':<10}  text")
                print("  " + "─" * 64)
                _seg_escalated = 0
                _seg_clear     = 0
                for _si, _seg in enumerate(_speech_meta["segments"]):
                    _st = _seg.get("text", "").strip()
                    if not _st:
                        continue
                    _slp = _seg.get("avg_logprob", -0.5)
                    # Embed segment text and check S01/S02 ONNX (graceful skip if unavailable)
                    try:
                        from sentence_transformers import SentenceTransformer as _ST
                        _seg_emb = _ST("all-MiniLM-L6-v2").encode(
                            [_st], normalize_embeddings=True)[0].tolist()
                    except Exception:
                        _seg_emb = None
                    _s01_path = FAMILY_HEADS.get("S01", {}).get("path", "")
                    _s01_lbls = FAMILY_HEADS.get("S01", {}).get("labels", {})
                    _s02_path = FAMILY_HEADS.get("S02", {}).get("path", "")
                    _s02_lbls = FAMILY_HEADS.get("S02", {}).get("labels", {})
                    _asr_fam, _asr_lbl, _asr_esc = "S01", "unknown", False
                    if _seg_emb:
                        _r1 = classify_onnx_family([_seg_emb], "S01")
                        if _r1:
                            _asr_lbl = _r1[0][0]
                            if _asr_lbl != "clear":
                                _r2 = classify_onnx_family([_seg_emb], "S02")
                                if _r2:
                                    _asr_fam, _asr_lbl, _asr_esc = "S02", _r2[0][0], True
                                    _seg_escalated += 1
                    if _asr_lbl == "clear":
                        _seg_clear += 1
                    _esc_flag = " ←ESC" if _asr_esc else "     "
                    print(f"  {_si:<4} {_slp:>8.3f} {_asr_fam:>5} "
                          f"{_asr_lbl:<10}{_esc_flag}  '{_st[:40]}'")
                print(f"\n  clear={_seg_clear}/{len(_speech_meta['segments'])}  "
                      f"escalated={_seg_escalated}")
            print()
        else:
            print(f"[speech] ERROR: could not transcribe {args.speech_in}")
            sys.exit(1)

    if not texts:
        texts = [
            "Apple reports record quarterly revenue driven by iPhone sales.",
            "Scientists discover new exoplanet in the habitable zone.",
            "World leaders gather for climate summit negotiations.",
            "Champions League final draws largest TV audience in years.",
        ]
        print("(no input given — using 4 built-in demo texts)\n")

    models_dir = args.models_dir
    frag_bqfp  = os.path.join(models_dir, "T04-frag.bqfp")
    use_frag   = not args.no_fragment and os.path.exists(frag_bqfp)
    frontier_path = os.path.join(models_dir, "frontier.json")
    onnx_path = args.onnx_model or "/tmp/bonfyre-72/runs/T04-C-ag_news-1000/train/model.onnx"

    # Metrics event accumulator — populated throughout the run
    _metrics = {
        "schema":            "bonfyre-metrics-v1",
        "n_inputs":          0,
        "input_texts":       [],   # stored for failure corpus extraction
        "routed_family":     None,
        "fragment_exit":     False,
        "early_exit_iter":   None,
        "full_loop":         False,
        "escalations":       [],   # list of {from, to, iter, conf_before, conf_after}
        "iterations_ran":    0,
        "avg_confidence":    None,
        "min_confidence":    None,
        "avg_latency_ms":    None,
        "labels":            [],
        "confidences":       [],
        # ── speech-specific metrics (null for text-only runs) ─────────
        "speech_in":                args.speech_in,
        "first_transcript_ms":      _speech_meta.get("first_segment_ms") if _speech_meta else None,
        "asr_avg_logprob":          _speech_meta.get("avg_logprob") if _speech_meta else None,
        "asr_num_segments":         _speech_meta.get("num_segments") if _speech_meta else None,
        "wer":                      None,   # filled below if --speech-gt provided
        "fragment_only_asr_rate":   None,   # filled if frag-exit triggered
        "args": {
            "loop":          args.loop,
            "chain":         args.chain,
            "frag_exit":     args.frag_exit,
            "conf_exit":     args.conf_exit,
            "escalate":      args.escalate,
            "escalate_drop": args.escalate_drop,
        },
    }

    # Compute WER if ground-truth transcript provided alongside --speech-in
    if _speech_meta and args.speech_gt:
        gt_text = args.speech_gt
        if os.path.isfile(gt_text):
            gt_text = open(gt_text, encoding="utf-8").read()
        _metrics["wer"] = _compute_wer(
            _speech_meta.get("transcript", ""), gt_text)

    def _emit_metrics(final_labels=None, final_confs=None):
        """Write metrics to args.metrics_out (JSON array, append-safe)."""
        import time
        if not args.metrics_out:
            return
        if final_labels:
            _metrics["labels"] = final_labels
        if final_confs:
            _metrics["confidences"] = [round(c, 4) if c else None for c in final_confs]
            valids = [c for c in final_confs if c is not None]
            if valids:
                _metrics["avg_confidence"] = round(sum(valids)/len(valids), 4)
                _metrics["min_confidence"] = round(min(valids), 4)
        _metrics["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        path = args.metrics_out
        # Append-safe: maintain a JSON array of runs
        if path:
            try:
                existing = json.load(open(path))
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []
            existing.append(_metrics)
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"  metrics → {path}  ({len(existing)} run(s) recorded)")

        # ── Memory recording (--memory-dir) ──────────────────────────
        if args.memory_dir:
            try:
                import sys as _sys
                _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if _pkg_root not in _sys.path:
                    _sys.path.insert(0, _pkg_root)
                from scripts.bonfyre_memory import BonfyreMemory as _BM
                _mem = _BM(args.memory_dir)
                _rid = _mem.record_run(_metrics)
                print(f"  memory  → {args.memory_dir}  (run_id={_rid})")
                # Auto-evolve (--auto-evolve)
                if args.auto_evolve:
                    from scripts.auto_evolve import evolve as _evolve
                    _ev = _evolve(
                        memory_dir=args.memory_dir,
                        models_dir=args.models_dir,
                        dry_run=False,
                        skip_discover=True,   # skip during hot path
                    )
                    if _ev.get("new_families"):
                        for _nf in _ev["new_families"]:
                            print(f"  ★ NEW FAMILY: {_nf['family_id']}  "
                                  f"(from {_nf['from_pattern']} on {_nf['from_family']})")
                    if _ev.get("routing_adjusted"):
                        print(f"  routing weights updated → "
                              f"{args.memory_dir}/graph/frontier_adjusted.json")
            except Exception as _mem_exc:
                print(f"  [memory] WARNING: {_mem_exc}")

    print("=" * 72)
    print(" BONFYRE  end-to-end demo")
    print(f"  inputs  : {len(texts)} text(s)")
    if args.speech_in:
        print(f"  speech  : {args.speech_in}"
              f"  (model={os.environ.get('WHISPER_MODEL', 'base')})")
    print(f"  loop    : {args.loop} iterations")
    print(f"  chain   : {args.chain}")
    print(f"  fpqx    : {args.fpqx}")
    print(f"  fragment: {'yes (T04-frag)' if use_frag else 'no'}")
    if args.frag_exit is not None:
        print(f"  frag-exit: {args.frag_exit:.2f}  (exit after fragment if all inputs ≥ threshold)")
    if args.conf_exit is not None:
        print(f"  conf-exit: {args.conf_exit:.2f}  (stop looping when all inputs ≥ threshold)")
    if args.escalate:
        print(f"  escalate : drop ≥ {args.escalate_drop:.2f} triggers T04→T15→T16")
    print("=" * 72)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── 1. Embed ───────────────────────────────────────────────────
        print("── 1. Embedding texts via MiniLM-L6-v2 ─────────────────────────")
        embs = embed_texts(texts)
        print(f"     {len(embs)} vectors × {len(embs[0])} dim")
        raw_path = os.path.join(tmpdir, "embeddings.bin")
        write_vecs(raw_path, embs)
        print()

        # ── 2. Corpus stats + route ────────────────────────────────────
        print("── 2. Corpus stats + routing ────────────────────────────────────")
        stats = compute_stats(texts)
        stats_path = os.path.join(tmpdir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        # fragment origin is T04 (T04-frag preflight) — use as frontier context
        from_fam = "T04" if use_frag else None
        routed_family, cosine_bias = route(stats_path, frontier_path, from_fam)
        _metrics["n_inputs"]      = len(texts)
        _metrics["input_texts"]   = [t[:200] for t in texts]  # cap length for storage
        _metrics["routed_family"] = routed_family
        print(f"     n_docs={stats['n_docs']}  avg_doc_len={stats['avg_doc_len']}"
              f"  vocab={stats['vocab_size']}")
        bias_str = f"  cosine_bias={cosine_bias:.4f}" if cosine_bias else ""
        print(f"     → routed to family: {routed_family}{bias_str}")
        print()

        # ── 3. Fragment pre-process (with optional exit gate) ─────────────
        active_chain = args.chain

        if use_frag and args.frag_exit is not None:
            print("── 3. Fragment pre-process + gate check ─────────────────────────────")
            frag_out = os.path.join(tmpdir, "frag_out.bin")
            ok = sli_run(raw_path, frag_bqfp, frag_out)
            if ok:
                frag_vecs, _ = read_vecs(frag_out)
                cos_vals = [cosine(embs[i], frag_vecs[i]) for i in range(len(embs))]
                print(f"     {len(embs)} vectors transformed"
                      f"  (raw→frag cosine: {sum(cos_vals)/len(cos_vals):.4f})")
                frag_onnx = classify_onnx_family(frag_vecs, routed_family)
                if frag_onnx:
                    min_conf = min(r[1] for r in frag_onnx)
                    avg_conf = sum(r[1] for r in frag_onnx) / len(frag_onnx)
                    print(f"     frag-gate: conf_avg={avg_conf*100:.1f}%"
                          f"  conf_min={min_conf*100:.1f}%"
                          f"  threshold={args.frag_exit*100:.0f}%", end="")
                    if min_conf >= args.frag_exit:
                        print("  ← FRAGMENT EXIT\n")
                        _metrics["fragment_exit"] = True
                        _metrics["iterations_ran"] = 0
                        if _speech_meta:
                            _metrics["fragment_only_asr_rate"] = 1.0
                        lbls = [r[0] for r in frag_onnx]
                        cfs  = [r[1] for r in frag_onnx]
                        _emit_metrics(lbls, cfs)
                        print("=" * 72)
                        print(" RESULTS  (fragment-only, 0 loop iterations)")
                        print("=" * 72)
                        print(f"  {'#':<4}  {'label':<12}  {'conf':>5}  input text")
                        print("  " + "─" * 70)
                        for i, text in enumerate(texts):
                            excerpt = (text[:48] + "…") if len(text) > 51 else text
                            lbl, cf = frag_onnx[i]
                            print(f"  {i+1:<4}  {lbl:<12}  {cf*100:5.1f}%  {excerpt}")
                        print()
                        print("=" * 72)
                        return
                    else:
                        print()
                in_path = frag_out
                active_chain = "auto" if args.chain == "fragment:auto" else args.chain
            else:
                print("     (fragment run failed — using raw embeddings)")
                in_path = raw_path
            print()
        elif use_frag and args.chain == "fragment:auto":
            print("── 3. Fragment pre-process ──────────── (built into chain; handled by SLI)\n")
            in_path = raw_path
        elif use_frag:
            print("── 3. Fragment pre-process (T04-frag, sub-model first-hop) ─────────")
            frag_out = os.path.join(tmpdir, "frag_out.bin")
            ok = sli_run(raw_path, frag_bqfp, frag_out)
            if ok:
                frag_vecs, _ = read_vecs(frag_out)
                cos_vals = [cosine(embs[i], frag_vecs[i]) for i in range(len(embs))]
                avg_cos = sum(cos_vals) / len(cos_vals)
                print(f"     {len(embs)} vectors transformed  (raw→frag cosine: {avg_cos:.4f})")
            else:
                frag_out = raw_path
                print("     (fragment run failed — using raw embeddings)")
            in_path = frag_out
            print()
        else:
            in_path = raw_path
            print("── 3. Fragment pre-process ──────────── (skipped)\n")

        # ── 4. Auto-run: route → align → loop (with optional early exit) ────
        print(f"── 4. auto-run  chain={active_chain}  fpqx={args.fpqx}"
              f"  loop={args.loop}  thresh={args.thresh} ─────")
        auto_out = os.path.join(tmpdir, "auto")

        if args.conf_exit is not None:
            # ── Early-exit path: run 1 iter at a time, classify + escalate ──
            cur_in         = in_path
            cur_chain      = active_chain
            current_family = routed_family
            prev_conf_avg  = None
            all_iters      = []
            exit_iter      = None
            for step in range(1, args.loop + 1):
                step_out = os.path.join(tmpdir, f"auto-step{step}")
                log_step = sli_auto_run(
                    cur_in, stats_path, step_out,
                    models_dir, 1, cur_chain, args.fpqx, args.thresh
                )
                step_iters = parse_iter_log(log_step)
                if step == 1 and "preflight:" in log_step and cur_chain != "auto":
                    print(f"     preflight  family={routed_family}-frag  (fragment applied)")
                for it, fam, delta in step_iters:
                    abs_iter = step
                    delta_s  = f"{delta:.4f}" if delta is not None else "n/a"
                    _, cur_vecs, _ = read_last_iter(step_out, 1)
                    if cur_vecs:
                        onnx_r = classify_onnx_family(cur_vecs, current_family)
                        if onnx_r:
                            min_c = min(r[1] for r in onnx_r)
                            avg_c = sum(r[1] for r in onnx_r) / len(onnx_r)
                            print(f"     iter {abs_iter:2d}  family={fam}  delta={delta_s}"
                                  f"  conf_avg={avg_c*100:.1f}%  conf_min={min_c*100:.1f}%",
                                  end="")
                            # Check escalation before exit
                            esc_msg = ""
                            if args.escalate and prev_conf_avg is not None:
                                drop = prev_conf_avg - avg_c
                                if drop >= args.escalate_drop:
                                    next_fam = ESCALATION_CHAIN.get(current_family)
                                    if next_fam:
                                        esc_msg = (
                                            f"  → ESCALATE {current_family}→{next_fam}"
                                            f" (conf {prev_conf_avg*100:.1f}%→{avg_c*100:.1f}%)"
                                        )
                                        _metrics["escalations"].append({
                                            "from": current_family,
                                            "to": next_fam,
                                            "iter": abs_iter,
                                            "conf_before": round(prev_conf_avg, 4),
                                            "conf_after":  round(avg_c, 4),
                                        })
                                        current_family = next_fam
                                        cur_chain = "auto"
                            if esc_msg:
                                print(esc_msg)
                            elif min_c >= args.conf_exit and exit_iter is None:
                                print(f"  ← EXIT (all ≥ {args.conf_exit*100:.0f}%)")
                                exit_iter = abs_iter
                            else:
                                print()
                            prev_conf_avg = avg_c
                        else:
                            print(f"     iter {abs_iter:2d}  family={fam}  delta={delta_s}")
                    all_iters.append((step, fam, delta))
                _, nxt_vecs, _ = read_last_iter(step_out, 1)
                if not nxt_vecs:
                    break
                nxt_path = os.path.join(tmpdir, f"iter{step}_out.bin")
                write_vecs(nxt_path, nxt_vecs)
                cur_in = nxt_path
                if exit_iter is not None:
                    break
            auto_out = step_out
            iters    = all_iters
            _metrics["iterations_ran"] = exit_iter if exit_iter is not None else args.loop
            if exit_iter is not None:
                _metrics["early_exit_iter"] = exit_iter
                saved = args.loop - exit_iter
                print(f"\n     early exit at iter {exit_iter}  "
                      f"({saved} loop iteration(s) saved)")
            else:
                _metrics["full_loop"] = True
                print(f"\n     no early exit — ran all {args.loop} iterations")
        else:
            # ── Standard path (full loop) ─────────────────────────────────────
            log = sli_auto_run(
                in_path, stats_path, auto_out,
                models_dir, args.loop, active_chain, args.fpqx, args.thresh
            )
            iters = parse_iter_log(log)
            _metrics["iterations_ran"] = len(iters)
            _metrics["full_loop"] = True
            has_preflight = "preflight:" in log
            if has_preflight:
                fam = routed_family
                print(f"     preflight  family={fam}-frag  (fragment applied before iter 1)")
            for it, fam, delta in iters:
                delta_s = f"{delta:.4f}" if delta is not None else "n/a"
                print(f"     iter {it:2d}  family={fam}  delta={delta_s}")
        print()

        # ── 5. Read final output + convergence ────────────────────────
        final_iter, final_vecs, final_art = read_last_iter(auto_out, args.loop)
        if not final_vecs:
            print("  ERROR: no output vectors found")
            sys.exit(1)

        families_seen = list(dict.fromkeys(fam for _, fam, _ in iters))
        final_family = iters[-1][1] if iters else routed_family
        deltas = [d for _, _, d in iters if d is not None]
        delta_trend = " → ".join(f"{d:.3f}" for d in deltas[:3]) + \
                      (" → … → " + f"{deltas[-1]:.3f}" if len(deltas) > 3 else "")

        # ── 6. Classify (ONNX head on raw 384-dim embeddings) ────────────────
        onnx_results = classify_onnx(embs, onnx_path)
        if onnx_results:
            labels    = [r[0] for r in onnx_results]
            confs     = [r[1] for r in onnx_results]
            cls_source = "ONNX head"
        else:
            # proxy fallback: argmax over first 4 tile dims
            labels = [TOPIC_LABELS.get(argmax(v[:4]), "?") for v in final_vecs[:len(texts)]]
            confs  = [None] * len(labels)
            cls_source = "proxy (tile argmax)"

        # ── 7. Per-iter cosine from raw input ─────────────────────────
        raw_vecs, _ = read_vecs(raw_path)
        final_cos_vals = [cosine(raw_vecs[i], final_vecs[i]) for i in range(len(raw_vecs))]
        avg_final_cos  = sum(final_cos_vals) / len(final_cos_vals)

        # ── Summary ───────────────────────────────────────────────────
        _emit_metrics(labels, confs)
        print("=" * 72)
        print(" RESULTS")
        print("=" * 72)
        print(f"  families traversed : {' → '.join(families_seen)}")
        print(f"  final family       : {final_family}")
        print(f"  iterations ran     : {final_iter}")
        print(f"  delta curve        : {delta_trend}")
        print(f"  raw→final cosine   : {avg_final_cos:.4f}"
              f"  (transform displacement from origin)")
        print(f"  classifier         : {cls_source}")
        print()
        conf_hdr = "  conf" if onnx_results else ""
        print(f"  {'#':<4}  {'label':<12}{conf_hdr}  {'input text'}")
        print("  " + "─" * 70)
        for i, (text, label) in enumerate(zip(texts, labels)):
            excerpt = (text[:48] + "…") if len(text) > 51 else text
            conf_s = f"  {confs[i]*100:5.1f}%" if confs[i] is not None else ""
            print(f"  {i+1:<4}  {label:<12}{conf_s}  {excerpt}")
        print()
        print("=" * 72)
        print(" fragment  → full-model → aligned chain → looped convergence")
        print(" routing   → chaining   → stabilized output")
        print("=" * 72)


if __name__ == "__main__":
    main()
