#!/usr/bin/env python3
"""
synth_teacher_asr.py — per-doc ASR confidence tags for collapse_train.

Runs Whisper on every .wav (or .mp3/.flac) file in <audio-corpus-dir>,
computes segment-level avg_logprob, and writes a per-doc tags.json that
collapse_train.py can consume as the tag stage in T30-C.

ASR confidence labels (3-class):
  clear      avg_logprob > -0.5          high-confidence transcription
  ambiguous  -1.0 <= avg_logprob <= -0.5 borderline — worth a second pass
  noisy      avg_logprob < -1.0          low-confidence — escalate ASR family

The label is also written as the teacher signal for collapse (same schema as
synth_teacher_tags.py: {"tags": [{"label": "<str>", "score": <0-1>}]}).

Side-effects:
  • If no corresponding .txt file exists, writes the transcript to <stem>.txt
    so that synth_teacher_embed.py can embed it on the next stage.
  • Writes <stem>.asr.json with richer info (all segments, avg_logprob,
    first_segment_ms, num_segments, whisper_model).

Output format (tags.json per doc, matches synth_teacher_tags.py):
  {"tags": [{"label": "clear", "score": 0.92}]}

Usage:
  python3 scripts/synth_teacher_asr.py <audio-corpus-dir> <out-dir> \\
      [--model base] [--device cpu] [--min-dur 0.5] [--force-retranscribe]

Deps: pip install openai-whisper soundfile
"""
import argparse
import json
import math
import os
import sys
import time

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# Logprob thresholds → 3-class label
THRESH_CLEAR     = -0.5   # avg_logprob > THRESH_CLEAR    → "clear"
THRESH_AMBIGUOUS = -1.0   # > THRESH_AMBIGUOUS            → "ambiguous"
                           # <= THRESH_AMBIGUOUS           → "noisy"

# Probability-like score derived from logprobs (clamp 0..1)
def logprob_to_score(lp: float) -> float:
    """Map avg_logprob to [0,1] confidence score via exp(lp), clamped."""
    return min(1.0, max(0.0, math.exp(lp)))


def classify_logprob(avg_lp: float) -> str:
    if avg_lp > THRESH_CLEAR:
        return "clear"
    if avg_lp > THRESH_AMBIGUOUS:
        return "ambiguous"
    return "noisy"


def main():
    p = argparse.ArgumentParser(
        description="Synth ASR confidence teacher for collapse_train (T30-C)")
    p.add_argument("audio_dir",  help="Directory containing .wav/.mp3/.flac files")
    p.add_argument("out_dir",    help="Output directory for per-doc tags.json files")
    p.add_argument("--model",    default="base",
                   help="Whisper model name (tiny|base|small|medium|large, default: base)")
    p.add_argument("--device",   default="cpu",
                   help="Torch device (cpu|cuda, default: cpu)")
    p.add_argument("--min-dur",  type=float, default=0.5,
                   help="Skip audio files shorter than this many seconds (default: 0.5)")
    p.add_argument("--force-retranscribe", action="store_true",
                   help="Re-transcribe even if .asr.json already exists")
    p.add_argument("--lang",     default=None,
                   help="Whisper language hint (e.g. 'en'). Default: auto-detect")
    args = p.parse_args()

    try:
        import whisper
    except ImportError:
        sys.exit("[synth_asr] missing dep: pip install openai-whisper")

    audio_files = sorted(
        f for f in os.listdir(args.audio_dir)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    )
    if not audio_files:
        sys.exit(f"[synth_asr] no audio files found in {args.audio_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[synth_asr] loading whisper model '{args.model}' on {args.device} …")
    model = whisper.load_model(args.model, device=args.device)

    label_counts = {"clear": 0, "ambiguous": 0, "noisy": 0}
    written = 0
    skipped_short = 0

    for fname in audio_files:
        stem = os.path.splitext(fname)[0]
        audio_path = os.path.join(args.audio_dir, fname)
        asr_json_path = os.path.join(args.out_dir, f"{stem}.asr.json")
        tag_json_path = os.path.join(args.out_dir, f"{stem}.json")
        txt_path      = os.path.join(args.audio_dir, f"{stem}.txt")

        # Load cached ASR result if available
        if not args.force_retranscribe and os.path.exists(asr_json_path):
            cached = json.load(open(asr_json_path))
            avg_lp = cached.get("avg_logprob", -1.0)
        else:
            t0 = time.monotonic()
            try:
                result = model.transcribe(
                    audio_path,
                    language=args.lang,
                    verbose=False,
                    fp16=False,  # always fp32 on CPU for portability
                )
            except Exception as exc:
                print(f"  [WARN] {fname}: transcription failed: {exc}")
                continue

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            segments   = result.get("segments", [])

            if not segments:
                skipped_short += 1
                continue

            # Total audio duration via last segment end
            total_dur = max(s.get("end", 0.0) for s in segments)
            if total_dur < args.min_dur:
                skipped_short += 1
                continue

            # Aggregate avg_logprob across all segments (weighted by token count)
            total_tokens = sum(len(s.get("tokens", [])) or 1 for s in segments)
            w_lp = sum(
                s.get("avg_logprob", -1.0) * (len(s.get("tokens", [])) or 1)
                for s in segments
            )
            avg_lp = w_lp / max(total_tokens, 1)

            first_end_ms = int(segments[0].get("end", 0.0) * 1000) if segments else 0

            asr_rec = {
                "whisper_model":       args.model,
                "transcript":          result.get("text", "").strip(),
                "avg_logprob":         round(avg_lp, 4),
                "num_segments":        len(segments),
                "first_segment_ms":    first_end_ms,
                "total_duration_s":    round(total_dur, 2),
                "transcription_ms":    elapsed_ms,
                "language":            result.get("language", ""),
                "segments": [
                    {
                        "id":         s.get("id"),
                        "start":      s.get("start"),
                        "end":        s.get("end"),
                        "text":       s.get("text", "").strip(),
                        "avg_logprob": round(s.get("avg_logprob", -1.0), 4),
                        "no_speech_prob": round(s.get("no_speech_prob", 0.0), 4),
                    }
                    for s in segments
                ],
            }
            with open(asr_json_path, "w") as f:
                json.dump(asr_rec, f, indent=2)

            # Write transcript .txt if missing (for synth_teacher_embed.py)
            if not os.path.exists(txt_path):
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(asr_rec["transcript"] + "\n")

        # Write bonfyre-tag-format tags.json
        label = classify_logprob(avg_lp)
        score = logprob_to_score(avg_lp)
        tag_rec = {"tags": [{"label": label, "score": round(score, 4)}]}
        with open(tag_json_path, "w") as f:
            json.dump(tag_rec, f)

        label_counts[label] += 1
        written += 1
        if written % 50 == 0:
            print(f"  …{written} done")

    total = sum(label_counts.values())
    print(f"[synth_asr] done: {written} docs tagged  "
          f"(clear={label_counts['clear']}  "
          f"ambiguous={label_counts['ambiguous']}  "
          f"noisy={label_counts['noisy']})")
    if skipped_short:
        print(f"[synth_asr] skipped {skipped_short} files (too short or failed)")
    print(f"[synth_asr] tags written → {args.out_dir}/")

    # Write corpus-level stats for reporting
    stats_path = os.path.join(args.out_dir, "_asr_corpus_stats.json")
    stats = {
        "n_docs":         written,
        "n_skipped":      skipped_short,
        "label_counts":   label_counts,
        "clear_rate":     round(label_counts["clear"] / max(total, 1), 4),
        "ambiguous_rate": round(label_counts["ambiguous"] / max(total, 1), 4),
        "noisy_rate":     round(label_counts["noisy"] / max(total, 1), 4),
        "whisper_model":  args.model,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[synth_asr] corpus stats → {stats_path}")


if __name__ == "__main__":
    main()
