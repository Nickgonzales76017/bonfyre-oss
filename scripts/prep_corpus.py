#!/usr/bin/env python3
"""
prep_corpus.py — download a HuggingFace dataset and write .txt files for bonfyre-run.

Usage
  python3 scripts/prep_corpus.py --dataset ag_news    --out /tmp/corpus/ag_news    --n 500
  python3 scripts/prep_corpus.py --dataset cnn_dm     --out /tmp/corpus/cnn_dm     --n 200
  python3 scripts/prep_corpus.py --dataset wiki       --out /tmp/corpus/wiki       --n 100

Each .txt file = one document (article body).
Filenames are zero-padded integers: 000000.txt … 000499.txt

Deps: pip install datasets
"""
import argparse
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("[prep_corpus] missing dep: pip install datasets")


CONFIGS = {
    "ag_news": {
        "path":   "ag_news",
        "split":  "train",
        "field":  "text",
        "config": None,
    },
    "cnn_dm": {
        "path":   "cnn_dailymail",
        "split":  "train",
        "field":  "article",
        "config": "3.0.0",
    },
    "wiki": {
        "path":   "wikipedia",
        "split":  "train",
        "field":  "text",
        "config": "20220301.en",
    },
}


def main():
    p = argparse.ArgumentParser(description="Prepare .txt corpus from HuggingFace dataset")
    p.add_argument("--dataset", required=True, choices=list(CONFIGS),
                   help="dataset key: ag_news | cnn_dm | wiki")
    p.add_argument("--out",     required=True, help="output directory for .txt files")
    p.add_argument("--n",       type=int, default=500,
                   help="number of documents to write (default 500)")
    p.add_argument("--min-len", type=int, default=80,
                   help="skip documents shorter than this many chars (default 80)")
    args = p.parse_args()

    cfg = CONFIGS[args.dataset]
    os.makedirs(args.out, exist_ok=True)

    print(f"[prep_corpus] loading {cfg['path']} ({cfg['split']}) …")
    ds = load_dataset(
        cfg["path"],
        cfg["config"],
        split=cfg["split"],
        streaming=True,          # don't download the whole thing
        trust_remote_code=True,
    )

    written = 0
    skipped = 0
    for row in ds:
        if written >= args.n:
            break
        text = row.get(cfg["field"], "").strip()
        if len(text) < args.min_len:
            skipped += 1
            continue
        fname = os.path.join(args.out, f"{written:06d}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        written += 1

    print(f"[prep_corpus] wrote {written} files → {args.out}/  (skipped {skipped} short docs)")


if __name__ == "__main__":
    main()
