#!/usr/bin/env python3
"""
synth_teacher_sentiment.py — T08-C synthetic risk teacher.

Uses a lightweight HuggingFace sentiment pipeline to classify each doc as
high-risk (negative) or low-risk (positive/neutral), writing per-doc
risk JSON that collapse_train.py --task risk-score consumes via --risk-dir.

Output format (per doc):
  {"risk": "high"|"low", "score": float, "label": str}

Usage:
  python3 synth_teacher_sentiment.py <corpus_dir> <out_dir>
"""
import argparse
import json
import os
import sys


BATCH_SIZE = 32
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir")
    p.add_argument("out_dir")
    p.add_argument("--model", default=MODEL_NAME)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(args.corpus_dir) if f.endswith(".txt"))
    stems     = [f[:-4] for f in txt_files]

    if not stems:
        sys.exit(f"[synth_sentiment] no .txt files in {args.corpus_dir}")

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        sys.exit("[synth_sentiment] transformers required: pip install transformers")

    print(f"[synth_sentiment] loading {args.model} ...")
    classifier = hf_pipeline(
        "sentiment-analysis",
        model=args.model,
        truncation=True,
        max_length=512,
        device=-1,  # CPU
    )

    # Read all texts
    texts = []
    for stem in stems:
        path = os.path.join(args.corpus_dir, f"{stem}.txt")
        texts.append(open(path, encoding="utf-8").read()[:2000].strip())

    print(f"[synth_sentiment] classifying {len(texts)} docs in batches of {BATCH_SIZE} …")
    results = classifier(texts, batch_size=BATCH_SIZE)

    written = 0
    n_high  = 0
    for stem, res in zip(stems, results):
        label  = res["label"].lower()   # "negative" or "positive"
        score  = round(float(res["score"]), 4)
        risk   = "high" if label == "negative" else "low"
        n_high += 1 if risk == "high" else 0
        out = {"risk": risk, "score": score, "label": label}
        with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as f:
            json.dump(out, f)
        written += 1

    print(f"[synth_sentiment] wrote {written} risk files → {args.out_dir}/  "
          f"({n_high} high, {written - n_high} low)")


if __name__ == "__main__":
    main()
