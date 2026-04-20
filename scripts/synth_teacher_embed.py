#!/usr/bin/env python3
"""
synth_teacher_embed.py — write per-doc embedding JSON in bonfyre-embed output format.

Uses sentence-transformers directly (no ONNX runtime required) to produce
384-dim MiniLM embeddings, one <stem>.json per .txt file in corpus-dir.

Output format matches what bonfyre-embed --backend onnx writes:
  {"embedding": [...384 floats...], "dims": 384, "backend": "sentence-transformers"}

Usage (inside a bonfyre-run stage or standalone):
  python3 scripts/synth_teacher_embed.py <corpus-dir> <out-dir> [--model <name>]
"""
import argparse
import json
import os
import sys

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("[synth_embed] missing dep: pip install sentence-transformers")


BATCH_SIZE = 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir")
    p.add_argument("out_dir")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(args.corpus_dir) if f.endswith(".txt"))
    if not txt_files:
        sys.exit(f"[synth_embed] no .txt files in {args.corpus_dir}")

    print(f"[synth_embed] loading {args.model} …")
    model = SentenceTransformer(args.model)

    # Read all texts
    stems = [f[:-4] for f in txt_files]
    texts = [open(os.path.join(args.corpus_dir, f), encoding="utf-8").read().strip()
             for f in txt_files]

    print(f"[synth_embed] embedding {len(texts)} docs in batches of {BATCH_SIZE} …")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    written = 0
    for stem, vec in zip(stems, embeddings):
        out = {
            "embedding": vec.tolist(),
            "dims": len(vec),
            "backend": "sentence-transformers",
            "model": args.model,
        }
        with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as f:
            json.dump(out, f)
        written += 1

    print(f"[synth_embed] wrote {written} embedding files → {args.out_dir}/")


if __name__ == "__main__":
    main()
