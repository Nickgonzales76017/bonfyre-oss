#!/usr/bin/env python3
"""
synth_teacher_tags.py — write per-doc tags.json in bonfyre-tag output format.

Reads .label files written by prep_corpus.py alongside .txt files
and emits one <stem>.json per document in the expected format:
  {"tags": [{"label": "<str>", "score": 1.0}]}

Usage (inside a bonfyre-run stage or standalone):
  python3 scripts/synth_teacher_tags.py <corpus-dir> <out-dir>
"""
import json
import os
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: synth_teacher_tags.py <corpus-dir> <out-dir>")

    corpus_dir = sys.argv[1]
    out_dir    = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    label_files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".label"))
    if not label_files:
        sys.exit(f"[synth_tags] no .label files in {corpus_dir}\n"
                 f"  run prep_corpus.py --write-labels first")

    written = 0
    for fname in label_files:
        stem  = fname[:-6]  # strip .label
        label = open(os.path.join(corpus_dir, fname)).read().strip()
        if not label:
            continue
        out = {"tags": [{"label": label, "score": 1.0}]}
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(out, f)
        written += 1

    print(f"[synth_tags] wrote {written} tag files → {out_dir}/")


if __name__ == "__main__":
    main()
