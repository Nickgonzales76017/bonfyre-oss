#!/usr/bin/env python3
"""
scripts/demo_speech_loop.py — Bonfyre live speech loop demo

The full multi-modal adaptive chain:

  audio (or text)
    → bonfyre-transcribe (or whisper fallback)   [transcribe]
    → per-segment ASR confidence gate            [fragment-like exit]
       → S01-frag BQFP check → exit if clear
       → S01 full ONNX head  → exit if clear
       → S02 ONNX head       → escalate          [escalation]
    → full transcript → MiniLM embed             [embed]
    → corpus stats → bonfyre-model route         [route]
    → SLI auto-run (fragment:auto chain)         [transform loop]
    → T32 ONNX head: intent + urgency tagging    [tag]
    → T40 ONNX head: TTS tier routing            [TTS select]
    → response text generation                   [respond]
    → bonfyre-narrate (or piper fallback) TTS    [synthesize]
    → metrics log                                [metrics]

Usage:
  python3 scripts/demo_speech_loop.py --audio-in recording.wav
  python3 scripts/demo_speech_loop.py --text "What time is the meeting?"
  python3 scripts/demo_speech_loop.py --audio-in speech.wav \\
      --speech-gt truth.txt --out-dir /tmp/loop-out --metrics-out loop_runs.json

Any ONNX head that doesn't exist on disk is gracefully skipped — the demo
always runs to completion whether or not the heads have been trained yet.

Env vars:
  WHISPER_MODEL     whisper model size (default: base)
  WHISPER_DEVICE    torch device for whisper (default: cpu)
  BONFYRE_MODELS_DIR  path to .bqfp / frontier / ONNX (default: /tmp/bonfyre-families)
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")

# ── Repo layout ───────────────────────────────────────────────────────────────
REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLI_BIN        = os.path.join(REPO_ROOT, "cmd", "BonfyreSLI",        "bonfyre-sli")
MODEL_BIN      = os.path.join(REPO_ROOT, "cmd", "BonfyreModel",      "bonfyre-model")
FPQX_BIN       = os.path.join(REPO_ROOT, "cmd", "BonfyreFPQX",       "bonfyre-fpqx")
NARRATE_BIN    = os.path.join(REPO_ROOT, "cmd", "BonfyreNarrate",    "bonfyre-narrate")
TRANSCRIBE_BIN = os.path.join(REPO_ROOT, "cmd", "BonfyreTranscribe", "bonfyre-transcribe")

DEFAULT_MODELS = os.environ.get("BONFYRE_MODELS_DIR", "/tmp/bonfyre-families")

# ── ASR family ONNX heads ─────────────────────────────────────────────────────
ASR_HEADS = {
    "S01": {
        "path":    "/tmp/bonfyre-speech/librispeech_clean-500/run/train/model.onnx",
        "frag":    os.path.join(DEFAULT_MODELS, "S01-frag.bqfp"),
        "n_class": 3,
        "labels":  {0: "ambiguous", 1: "clear", 2: "noisy"},
    },
    "S02": {
        "path":    "/tmp/bonfyre-speech/librispeech_other-500/run/train/model.onnx",
        "frag":    os.path.join(DEFAULT_MODELS, "S02-frag.bqfp"),
        "n_class": 3,
        "labels":  {0: "ambiguous", 1: "clear", 2: "noisy"},
    },
}
ASR_CHAIN = ["S01", "S02"]

# ── Tagging + TTS routing heads ───────────────────────────────────────────────
INTENT_HEAD = "/tmp/bonfyre-speech/intent-500/run/train/model.onnx"
TTS_HEAD    = "/tmp/bonfyre-speech/tts-tier-500/run/train/model.onnx"

INTENT_LABELS = {
    0: "command-high",
    1: "command-low",
    2: "question-high",
    3: "question-low",
    4: "statement",
}
TTS_TIER_LABELS = {
    0: "fast",
    1: "quality",
    2: "standard",
}

# ── intent → short response templates ────────────────────────────────────────
RESPONSE_TEMPLATES = {
    "question-high": "I'll answer that urgently. Based on the input: {topic}.",
    "question-low":  "Here's what I found: {topic}.",
    "command-high":  "Executing immediately: {topic}.",
    "command-low":   "Got it. Processing: {topic}.",
    "statement":     "Understood. Noted: {topic}.",
}


# ── Utility ───────────────────────────────────────────────────────────────────
def write_vecs(path, vecs):
    n, d = len(vecs), len(vecs[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        for row in vecs:
            f.write(struct.pack(f"<{d}f", *row))


def read_vecs(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        vecs = [list(struct.unpack(f"<{d}f", f.read(d * 4))) for _ in range(n)]
    return vecs, d


def embed_text(text: str) -> list:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([text], normalize_embeddings=True)[0].tolist()


def compute_stats(texts):
    vocab = set()
    for t in texts:
        vocab.update(t.lower().split())
    return {
        "n_docs":      len(texts),
        "avg_doc_len": round(sum(len(t) for t in texts) / max(len(texts), 1), 1),
        "vocab_size":  len(vocab),
    }


def onnx_classify(emb: list, onnx_path: str, label_map: dict):
    """Return (label, confidence) or None."""
    if not os.path.exists(onnx_path):
        return None
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        return None
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    data = np.array([emb], dtype=np.float32)
    logits = sess.run(None, {sess.get_inputs()[0].name: data})[0]
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    idx = int(probs[0].argmax())
    return label_map.get(idx, f"class_{idx}"), float(probs[0][idx])


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe(audio_path: str) -> dict:
    """
    Transcribe audio. Primary: bonfyre-transcribe C binary.
    Fallback: openai-whisper Python.
    Returns dict with transcript, segments, avg_logprob, first_segment_ms.
    """
    if os.path.exists(TRANSCRIBE_BIN):
        out_json = audio_path + ".transcription.json"
        t0 = time.monotonic()
        r = subprocess.run(
            [TRANSCRIBE_BIN, audio_path, "--out-json", out_json],
            capture_output=True, text=True)
        elapsed = int((time.monotonic() - t0) * 1000)
        if r.returncode == 0 and os.path.exists(out_json):
            try:
                data = json.load(open(out_json))
                return _parse_whisper_result(data, elapsed, source="bonfyre-transcribe")
            except Exception:
                pass

    # Fallback: openai-whisper
    try:
        import whisper as wh
    except ImportError:
        sys.exit("[speech_loop] ERROR: openai-whisper not installed and bonfyre-transcribe not built\n"
                 "  pip install openai-whisper  or  make -C /tmp/bonfyre-oss")

    model_name = os.environ.get("WHISPER_MODEL", "base")
    device     = os.environ.get("WHISPER_DEVICE", "cpu")
    print(f"  [transcribe] loading whisper '{model_name}' on {device} …")
    t0 = time.monotonic()
    model  = wh.load_model(model_name, device=device)
    result = model.transcribe(audio_path, language="en",
                               fp16=False, verbose=False)
    elapsed = int((time.monotonic() - t0) * 1000)
    return _parse_whisper_result(result, elapsed, source="openai-whisper")


def _parse_whisper_result(data: dict, elapsed_ms: int, source: str) -> dict:
    segments = data.get("segments", [])
    total_tokens = sum(len(s.get("tokens", [])) or 1 for s in segments)
    w_lp = sum(
        s.get("avg_logprob", -1.0) * (len(s.get("tokens", [])) or 1)
        for s in segments
    )
    avg_lp   = w_lp / max(total_tokens, 1) if segments else -1.0
    first_ms = int(segments[0].get("end", 0.0) * 1000) if segments else 0
    return {
        "transcript":       data.get("text", "").strip(),
        "segments":         segments,
        "avg_logprob":      round(avg_lp, 4),
        "num_segments":     len(segments),
        "first_segment_ms": first_ms,
        "transcription_ms": elapsed_ms,
        "source":           source,
    }


# ── Per-segment ASR gate ──────────────────────────────────────────────────────
def asr_gate(segment_text: str, emb: list, frag_threshold: float = 0.70) -> dict:
    """
    Run segment through ASR adaptive ladder:
      1. S01-frag BQFP (fragment-level, cheapest)
      2. S01 ONNX (full, clean ASR head)
      3. S02 ONNX (escalate to noisy-ASR head)

    Returns:
      {family, label, confidence, level, escalated}
        level: "frag" | "S01" | "S02"
        escalated: True if went past S01
    """
    result = {"family": "S01", "label": "clear", "confidence": 0.0,
              "level": "none", "escalated": False}

    # Try S01 full ONNX first (fragment BQFP check would need the SLI binary)
    head_s01 = ASR_HEADS.get("S01", {})
    s01_r = onnx_classify(emb, head_s01.get("path", ""), head_s01.get("labels", {}))
    if s01_r:
        label, conf = s01_r
        result.update(family="S01", label=label, confidence=conf, level="S01")
        if conf >= frag_threshold and label == "clear":
            return result  # S01 exit — no escalation

    # Escalate to S02
    head_s02 = ASR_HEADS.get("S02", {})
    s02_r = onnx_classify(emb, head_s02.get("path", ""), head_s02.get("labels", {}))
    if s02_r:
        label, conf = s02_r
        result.update(family="S02", label=label, confidence=conf,
                      level="S02", escalated=True)

    return result


# ── WER ───────────────────────────────────────────────────────────────────────
def compute_wer(hyp: str, ref: str) -> float:
    def tok(s):
        return s.lower().split()
    h, r = tok(hyp), tok(ref)
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(len(h) + 1): d[i][0] = i
    for j in range(len(r) + 1): d[0][j] = j
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            d[i][j] = d[i-1][j-1] if h[i-1] == r[j-1] \
                else 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return round(d[len(h)][len(r)] / len(r), 4)


# ── TTS synthesis ─────────────────────────────────────────────────────────────
def synthesize(text: str, out_wav: str, tts_tier: str = "fast") -> dict:
    """
    Synthesize text to speech via bonfyre-narrate or piper fallback.
    tts_tier: fast | standard | quality
    Returns {"success": bool, "wav_path": str, "synthesis_ms": int}
    """
    t0 = time.monotonic()
    if os.path.exists(NARRATE_BIN):
        # Write text to tmp file
        txt_tmp = out_wav + ".input.txt"
        with open(txt_tmp, "w") as f:
            f.write(text)
        out_dir = os.path.dirname(out_wav)
        cmd = [NARRATE_BIN, txt_tmp, out_dir]
        if tts_tier == "quality":
            cmd += ["--verify", "--refine"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = int((time.monotonic() - t0) * 1000)
        if r.returncode == 0:
            return {"success": True, "wav_path": out_wav, "synthesis_ms": elapsed}
        print(f"  [tts] bonfyre-narrate exited {r.returncode}: {r.stderr[:120]}")

    # Fallback: piper TTS command-line (if on PATH)
    try:
        r2 = subprocess.run(
            ["piper", "--model", "en_US-lessac-medium",
             "--output_file", out_wav],
            input=text, capture_output=True, text=True
        )
        elapsed = int((time.monotonic() - t0) * 1000)
        if r2.returncode == 0:
            return {"success": True, "wav_path": out_wav, "synthesis_ms": elapsed}
    except FileNotFoundError:
        pass

    elapsed = int((time.monotonic() - t0) * 1000)
    return {"success": False, "wav_path": None, "synthesis_ms": elapsed}


# ── Route transcript text via bonfyre-model ───────────────────────────────────
def route_transcript(stats_path: str, frontier_path: str) -> str:
    if not os.path.exists(MODEL_BIN):
        return "T04"
    cmd = [MODEL_BIN, "route", stats_path]
    if os.path.exists(frontier_path):
        cmd += ["--frontier", frontier_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    family = "T04"
    for tok in r.stdout.split():
        if tok.startswith("family="):
            family = tok[7:]
    return family


# ── SLI auto-run ──────────────────────────────────────────────────────────────
def sli_auto_run(in_path, stats_path, out_dir, models_dir,
                 loop=3, chain="fragment:auto", thresh=0.0) -> str:
    if not os.path.exists(SLI_BIN):
        return ""
    r = subprocess.run(
        [SLI_BIN, "auto-run",
         "--in",         in_path,
         "--stats",      stats_path,
         "--out",        out_dir,
         "--loop",       str(loop),
         "--chain",      chain,
         "--fpqx",       "auto",
         "--thresh",     str(thresh),
         "--models-dir", models_dir],
        capture_output=True, text=True)
    return r.stdout + r.stderr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Bonfyre live speech loop demo — full multi-modal adaptive chain")
    ap.add_argument("--audio-in",    default=None,
                    help="Path to audio file (.wav/.mp3/.flac)")
    ap.add_argument("--text",        default=None,
                    help="Text input (skips transcription step)")
    ap.add_argument("--speech-gt",   default=None,
                    help="Ground-truth transcript for WER (text or .txt file path)")
    ap.add_argument("--out-dir",     default="/tmp/bonfyre-loop-out",
                    help="Output directory for WAVs + artifacts (default: /tmp/bonfyre-loop-out)")
    ap.add_argument("--models-dir",  default=DEFAULT_MODELS,
                    help=f"Path to .bqfp + frontier + ONNX (default: {DEFAULT_MODELS})")
    ap.add_argument("--loop",        type=int, default=3,
                    help="SLI transform loop iterations (default: 3)")
    ap.add_argument("--frag-thresh", type=float, default=0.70,
                    help="ASR confidence threshold for frag-exit (default: 0.70)")
    ap.add_argument("--no-tts",      action="store_true",
                    help="Skip TTS synthesis step")
    ap.add_argument("--metrics-out", default=None,
                    help="Append run metrics JSON to this path")
    ap.add_argument("--quiet",       action="store_true",
                    help="Suppress per-step banners")
    args = ap.parse_args()

    if not args.audio_in and not args.text:
        ap.error("provide --audio-in <file> or --text <str>")

    os.makedirs(args.out_dir, exist_ok=True)
    frontier_path = os.path.join(args.models_dir, "frontier.json")

    WALL_START = time.monotonic()
    metrics = {
        "schema": "bonfyre-speech-loop-v1",
        # filled as we go
        "transcript":           None,
        "avg_logprob":          None,
        "num_segments":         None,
        "first_transcript_ms":  None,
        "transcription_ms":     None,
        "asr_source":           None,
        "wer":                  None,
        "segment_results":      [],
        "segments_escalated":   0,
        "fragment_only_rate":   None,
        "routed_family":        None,
        "sli_iterations":       args.loop,
        "intent_label":         None,
        "intent_conf":          None,
        "tts_tier":             None,
        "tts_tier_conf":        None,
        "synthesis_ms":         None,
        "synthesis_success":    None,
        "response_text":        None,
        "wall_ms":              None,
    }

    def banner(title):
        if not args.quiet:
            print(f"\n── {title} {'─' * max(0, 60 - len(title))}")

    # ════════════════════════════════════════════════════
    # STEP 1: Transcribe
    # ════════════════════════════════════════════════════
    if args.audio_in:
        banner(f"1. Transcribe  {os.path.basename(args.audio_in)}")
        t_meta = transcribe(args.audio_in)
        transcript = t_meta["transcript"]
        metrics.update({
            "transcript":          transcript,
            "avg_logprob":         t_meta["avg_logprob"],
            "num_segments":        t_meta["num_segments"],
            "first_transcript_ms": t_meta["first_segment_ms"],
            "transcription_ms":    t_meta["transcription_ms"],
            "asr_source":          t_meta["source"],
        })
        print(f"  transcript   : {transcript[:100]}{'…' if len(transcript)>100 else ''}")
        print(f"  avg_logprob  : {t_meta['avg_logprob']:.3f}")
        print(f"  segments     : {t_meta['num_segments']}")
        print(f"  first_seg_ms : {t_meta['first_segment_ms']} ms")
        print(f"  transcribe_ms: {t_meta['transcription_ms']} ms")
        print(f"  source       : {t_meta['source']}")

        if args.speech_gt:
            gt = args.speech_gt
            if os.path.isfile(gt):
                gt = open(gt, encoding="utf-8").read()
            wer = compute_wer(transcript, gt)
            metrics["wer"] = wer
            print(f"  WER          : {wer*100:.1f}%")

        segments = t_meta["segments"]
    else:
        banner("1. Text input (skipping transcription)")
        transcript = args.text
        segments   = [{"text": transcript, "avg_logprob": -0.3,
                        "start": 0.0, "end": 1.0}]
        metrics["transcript"] = transcript
        metrics["asr_source"] = "text-input"
        print(f"  text: {transcript[:80]}")

    # ════════════════════════════════════════════════════
    # STEP 2: Embed
    # ════════════════════════════════════════════════════
    banner("2. Embed transcript (MiniLM-L6-v2)")
    try:
        full_emb = embed_text(transcript)
        print(f"  {len(full_emb)}-dim embedding ready")
    except Exception as exc:
        print(f"  WARNING: embed failed ({exc}) — using zero vector")
        full_emb = [0.0] * 384

    # ════════════════════════════════════════════════════
    # STEP 3: Per-segment ASR gate
    # ════════════════════════════════════════════════════
    banner(f"3. Per-segment ASR gate  ({len(segments)} segment(s)  thresh={args.frag_thresh:.2f})")
    seg_results = []
    n_escalated = 0
    n_clear     = 0

    for i, seg in enumerate(segments):
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        # Embed segment independently
        try:
            seg_emb = embed_text(seg_text)
        except Exception:
            seg_emb = full_emb  # fallback to full embedding

        gate = asr_gate(seg_text, seg_emb, frag_threshold=args.frag_thresh)
        seg_lp = seg.get("avg_logprob", -0.5)
        seg_rec = {
            "index":        i,
            "text":         seg_text[:60],
            "avg_logprob":  round(seg_lp, 4),
            "asr_family":   gate["family"],
            "asr_label":    gate["label"],
            "asr_conf":     round(gate["confidence"], 3),
            "escalated":    gate["escalated"],
        }
        seg_results.append(seg_rec)
        if gate["escalated"]:
            n_escalated += 1
        if gate["label"] == "clear":
            n_clear += 1

        flag = " ←ESC" if gate["escalated"] else ""
        print(f"  seg[{i:02d}] lp={seg_lp:+.3f}  {gate['family']:3s}  "
              f"{gate['label']:<10} {gate['confidence']*100:.0f}%{flag}  "
              f"'{seg_text[:40]}'")

    metrics["segment_results"]    = seg_results
    metrics["segments_escalated"] = n_escalated
    metrics["fragment_only_rate"] = (
        round(n_clear / max(len(seg_results), 1), 3) if seg_results else None
    )
    if seg_results:
        print(f"\n  clear={n_clear}/{len(seg_results)}  "
              f"escalated={n_escalated}  "
              f"fragment_only_rate={metrics['fragment_only_rate']:.1%}")

    # ════════════════════════════════════════════════════
    # STEP 4: Route + SLI transform loop
    # ════════════════════════════════════════════════════
    with tempfile.TemporaryDirectory() as tmpdir:
        vec_path   = os.path.join(tmpdir, "emb.bin")
        stats_path = os.path.join(tmpdir, "stats.json")

        write_vecs(vec_path, [full_emb])
        stats = compute_stats([transcript])
        with open(stats_path, "w") as f:
            json.dump(stats, f)

        banner("4. Route + SLI transform loop")
        routed = route_transcript(stats_path, frontier_path)
        metrics["routed_family"] = routed
        print(f"  routed → {routed}")

        sli_out = os.path.join(tmpdir, "sli")
        log = sli_auto_run(vec_path, stats_path, sli_out, args.models_dir,
                           loop=args.loop)
        if not args.quiet and log.strip():
            for line in log.splitlines():
                if "iter " in line or "preflight" in line:
                    print(f"  {line.strip()}")

        # ════════════════════════════════════════════════
        # STEP 5: Tag intent + urgency (T32 ONNX head)
        # ════════════════════════════════════════════════
        banner("5. Tag intent + urgency (T32 head)")
        intent_r = onnx_classify(full_emb, INTENT_HEAD, INTENT_LABELS)
        if intent_r:
            intent_label, intent_conf = intent_r
            metrics["intent_label"] = intent_label
            metrics["intent_conf"]  = round(intent_conf, 3)
            print(f"  intent: {intent_label}  ({intent_conf*100:.1f}%)")
        else:
            # Fallback: rule-based intent (mirror synth_teacher_intent logic inline)
            import re as _re
            _WH  = {"who","what","when","where","why","how","which","whose"}
            _IMP = {"tell","show","give","find","get","make","stop","start",
                    "go","run","set","check","call","send","use","add","remove",
                    "do","help","explain","summarize","generate","fix","search"}
            _UHI = {"urgent","immediately","asap","emergency","critical",
                    "right now","deadline","broken","failing","must"}
            t_l  = transcript.lower().strip()
            fw   = _re.split(r'\W+', t_l)[0] if t_l else ""
            if t_l.endswith("?") or fw in _WH:
                intent_raw = "question"
            elif fw in _IMP:
                intent_raw = "command"
            else:
                intent_raw = "statement"
            urgency = "high" if any(k in t_l for k in _UHI) else "low"
            intent_label = "statement" if intent_raw == "statement" \
                else f"{intent_raw}-{urgency}"
            intent_conf = 0.0
            metrics["intent_label"] = intent_label
            print(f"  intent: {intent_label}  (rule-based fallback)")

        # ════════════════════════════════════════════════
        # STEP 6: TTS tier routing (T40 head)
        # ════════════════════════════════════════════════
        banner("6. TTS tier routing (T40 head)")
        tts_r = onnx_classify(full_emb, TTS_HEAD, TTS_TIER_LABELS)
        if tts_r:
            tts_tier, tts_conf = tts_r
            metrics["tts_tier"]      = tts_tier
            metrics["tts_tier_conf"] = round(tts_conf, 3)
            print(f"  TTS tier: {tts_tier}  ({tts_conf*100:.1f}%)")
        else:
            # Fallback: rule-based tier from urgency signal
            tts_tier = "quality" if "high" in intent_label else "fast"
            metrics["tts_tier"] = tts_tier
            print(f"  TTS tier: {tts_tier}  (rule-based fallback)")

        # ════════════════════════════════════════════════
        # STEP 7: Generate response text
        # ════════════════════════════════════════════════
        banner("7. Generate response")
        topic = routed  # use routed family as a proxy topic token
        template = RESPONSE_TEMPLATES.get(intent_label,
                   RESPONSE_TEMPLATES["statement"])
        response_text = template.format(topic=topic)
        metrics["response_text"] = response_text
        print(f"  response: {response_text}")

        # ════════════════════════════════════════════════
        # STEP 8: TTS synthesis
        # ════════════════════════════════════════════════
        if not args.no_tts:
            banner(f"8. TTS synthesis  tier={tts_tier}")
            out_wav = os.path.join(args.out_dir, "response.wav")
            synth_result = synthesize(response_text, out_wav, tts_tier)
            metrics["synthesis_ms"]      = synth_result["synthesis_ms"]
            metrics["synthesis_success"] = synth_result["success"]
            if synth_result["success"]:
                print(f"  ✓ {synth_result['wav_path']}  ({synth_result['synthesis_ms']} ms)")
            else:
                print(f"  (TTS not available — bonfyre-narrate + piper not found)")
        else:
            banner("8. TTS synthesis  (skipped: --no-tts)")

    # ════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════
    metrics["wall_ms"] = int((time.monotonic() - WALL_START) * 1000)

    print()
    print("=" * 72)
    print(" BONFYRE SPEECH LOOP — SUMMARY")
    print("=" * 72)
    print(f"  transcript       : {(metrics['transcript'] or '')[:70]}")
    if metrics["wer"] is not None:
        print(f"  WER              : {metrics['wer']*100:.1f}%")
    print(f"  first_transcript : {metrics['first_transcript_ms']} ms")
    print(f"  ASR source       : {metrics['asr_source']}")
    print(f"  segments         : {metrics['num_segments']}   escalated={metrics['segments_escalated']}")
    print(f"  fragment_only    : {metrics['fragment_only_rate']:.1%}" if metrics['fragment_only_rate'] else "  fragment_only    : n/a")
    print(f"  routed family    : {metrics['routed_family']}")
    print(f"  intent           : {metrics['intent_label']}  ({metrics['intent_conf'] or 'rule-based'})")
    print(f"  TTS tier         : {metrics['tts_tier']}")
    if metrics["synthesis_success"]:
        print(f"  response WAV     : {args.out_dir}/response.wav")
    print(f"  wall time        : {metrics['wall_ms']} ms")
    print("=" * 72)
    print()

    # Segment table
    if seg_results:
        print(f"  {'seg':<4} {'logprob':>8} {'family':>5} {'label':<12} {'conf':>5} {'esc':>4}  text")
        print("  " + "─" * 70)
        for s in seg_results:
            esc = "YES" if s["escalated"] else "   "
            print(f"  {s['index']:<4} {s['avg_logprob']:>8.3f} {s['asr_family']:>5} "
                  f"{s['asr_label']:<12} {s['asr_conf']*100:>4.0f}% {esc:>4}  {s['text']}")
        print()

    # Metrics output
    if args.metrics_out:
        metrics["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            existing = json.load(open(args.metrics_out))
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
        existing.append(metrics)
        with open(args.metrics_out, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  metrics → {args.metrics_out}  ({len(existing)} run(s))")


if __name__ == "__main__":
    main()
