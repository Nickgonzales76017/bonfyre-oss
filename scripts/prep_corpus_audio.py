#!/usr/bin/env python3
"""
prep_corpus_audio.py — download an audio+transcript HuggingFace dataset
and write .wav/.txt/.label files for bonfyre-run speech recipes (T30-C etc.).

Analogous to prep_corpus.py but operates on audio datasets instead of text.

Usage:
  python3 scripts/prep_corpus_audio.py --dataset librispeech_clean \\
      --out /tmp/corpus/librispeech_clean --n 500
  python3 scripts/prep_corpus_audio.py --dataset commonvoice_en \\
      --out /tmp/corpus/commonvoice_en --n 200 --write-labels

Each output directory entry:
  <stem>.wav    — 16kHz mono PCM (resampled from dataset audio)
  <stem>.txt    — ground-truth transcript (lowercase, stripped)
  <stem>.label  — optional label (accent/gender/age if available and --write-labels)

Deps: pip install datasets soundfile numpy
      (soundfile requires libsndfile; on macOS: brew install libsndfile)
"""
import argparse
import io
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("[prep_audio] missing dep: pip install datasets")

try:
    import numpy as np
except ImportError:
    sys.exit("[prep_audio] missing dep: pip install numpy")

try:
    import soundfile as sf
except ImportError:
    sys.exit("[prep_audio] missing dep: pip install soundfile")


TARGET_SR = 16000  # Whisper expects 16kHz

AUDIO_CONFIGS = {
    # ── LibriSpeech clean 100h ─────────────────────────────────────────────────
    # Clean US English read speech. No accent/noise labels — ASR confidence
    # teachers will assign clear/ambiguous/noisy from Whisper logprobs.
    # URL: https://huggingface.co/datasets/openslr/librispeech_asr
    "librispeech_clean": {
        "path":             "openslr/librispeech_asr",
        "config":           "clean",
        "split":            "train.clean.100",
        "audio_field":      "audio",
        "transcript_field": "text",
        "label_field":      None,
        "label_names":      None,
        "min_dur_s":        0.5,
        "max_dur_s":        30.0,
    },
    # ── LibriSpeech other (noisy/diverse speakers) ─────────────────────────────
    # More challenging read speech — higher WER, useful for S02 (noisy ASR family).
    "librispeech_other": {
        "path":             "openslr/librispeech_asr",
        "config":           "other",
        "split":            "train.other.500",
        "audio_field":      "audio",
        "transcript_field": "text",
        "label_field":      None,
        "label_names":      None,
        "min_dur_s":        0.5,
        "max_dur_s":        30.0,
    },
    # ── Mozilla Common Voice 13 (English) ─────────────────────────────────────
    # Crowdsourced, accent-diverse. accent field → label if --write-labels.
    # URL: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0
    "commonvoice_en": {
        "path":             "mozilla-foundation/common_voice_13_0",
        "config":           "en",
        "split":            "train",
        "audio_field":      "audio",
        "transcript_field": "sentence",
        "label_field":      "accent",
        "label_names":      None,   # free-text accent description
        "min_dur_s":        0.5,
        "max_dur_s":        20.0,
    },
    # ── GigaSpeech small ──────────────────────────────────────────────────────
    # Diverse domains (podcast/YouTube/audiobook). domain-level label (P/Y/A/B/R).
    # Needs HF token for gated access: export HF_TOKEN=hf_...
    # URL: https://huggingface.co/datasets/speechcolab/gigaspeech
    "gigaspeech_s": {
        "path":             "speechcolab/gigaspeech",
        "config":           "s",
        "split":            "train",
        "audio_field":      "audio",
        "transcript_field": "text",
        "label_field":      "category",
        "label_names":      None,
        "min_dur_s":        0.5,
        "max_dur_s":        60.0,
    },
}


def resample(array, orig_sr: int, target_sr: int = TARGET_SR):
    """
    Simple linear resample using numpy.
    For production use scipy.signal.resample_poly; this avoids the dependency.
    """
    if orig_sr == target_sr:
        return array
    ratio = target_sr / orig_sr
    n_out = int(len(array) * ratio)
    indices = np.linspace(0, len(array) - 1, n_out)
    return np.interp(indices, np.arange(len(array)), array).astype(np.float32)


def write_wav(path: str, array, sr: int = TARGET_SR):
    """Write float32 mono array as 16-bit PCM WAV."""
    pcm = np.clip(array, -1.0, 1.0)
    sf.write(path, pcm, sr, subtype="PCM_16")


def main():
    p = argparse.ArgumentParser(
        description="Prepare audio corpus from HuggingFace for Bonfyre speech recipes")
    p.add_argument("--dataset",      required=True, choices=list(AUDIO_CONFIGS),
                   help="Dataset key: " + " | ".join(AUDIO_CONFIGS))
    p.add_argument("--out",          required=True,
                   help="Output directory for .wav + .txt (+ .label) files")
    p.add_argument("--n",            type=int, default=500,
                   help="Number of utterances to write (default 500)")
    p.add_argument("--write-labels", action="store_true",
                   help="Write .label files (accent/category/gender if available)")
    p.add_argument("--min-dur",      type=float, default=None,
                   help="Override min utterance duration in seconds")
    p.add_argument("--max-dur",      type=float, default=None,
                   help="Override max utterance duration in seconds")
    args = p.parse_args()

    cfg = AUDIO_CONFIGS[args.dataset]
    min_dur = args.min_dur if args.min_dur is not None else cfg["min_dur_s"]
    max_dur = args.max_dur if args.max_dur is not None else cfg["max_dur_s"]

    os.makedirs(args.out, exist_ok=True)

    print(f"[prep_audio] loading {cfg['path']} config={cfg['config']} "
          f"split={cfg['split']} …")

    load_kwargs = {
        "path":             cfg["path"],
        "name":             cfg["config"],
        "split":            cfg["split"],
        "streaming":        True,
        "trust_remote_code": True,
    }
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        load_kwargs["token"] = hf_token

    ds = load_dataset(**load_kwargs)

    written  = 0
    skipped  = 0

    for row in ds:
        if written >= args.n:
            break

        audio_obj = row.get(cfg["audio_field"])
        if audio_obj is None:
            skipped += 1
            continue

        # HF audio field is dict: {"array": np.ndarray, "sampling_rate": int, "path": str}
        array = audio_obj.get("array")
        orig_sr = audio_obj.get("sampling_rate", TARGET_SR)
        if array is None or len(array) == 0:
            skipped += 1
            continue

        # Duration check
        dur_s = len(array) / max(orig_sr, 1)
        if dur_s < min_dur or dur_s > max_dur:
            skipped += 1
            continue

        transcript = row.get(cfg["transcript_field"], "").strip()
        if not transcript:
            skipped += 1
            continue

        stem = f"{written:06d}"

        # Resample + write WAV
        audio_16k = resample(np.array(array, dtype=np.float32), orig_sr)
        wav_path = os.path.join(args.out, f"{stem}.wav")
        write_wav(wav_path, audio_16k)

        # Write transcript
        txt_path = os.path.join(args.out, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript.lower() + "\n")

        # Write label if requested and available
        if args.write_labels:
            lf = cfg.get("label_field")
            if lf and lf in row:
                raw_label = str(row[lf]).strip()
                if raw_label:
                    with open(os.path.join(args.out, f"{stem}.label"), "w") as f:
                        f.write(raw_label + "\n")

        written += 1
        if written % 100 == 0:
            print(f"  …{written}/{args.n} written")

    print(f"[prep_audio] done: {written} utterances written, "
          f"{skipped} skipped → {args.out}/")

    # Write corpus stats
    import json
    stats = {
        "dataset":     args.dataset,
        "hf_path":     cfg["path"],
        "split":       cfg["split"],
        "n_written":   written,
        "n_skipped":   skipped,
        "target_sr":   TARGET_SR,
        "min_dur_s":   min_dur,
        "max_dur_s":   max_dur,
    }
    stats_path = os.path.join(args.out, "_corpus_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[prep_audio] stats → {stats_path}")


if __name__ == "__main__":
    main()
