#!/usr/bin/env python3
"""
synth_teacher_tags.py — write per-doc tags.json in bonfyre-tag output format.

Two modes:

1. Label mode (when .label files exist alongside .txt files):
   Reads ground-truth labels written by prep_corpus.py --write-labels.
   Precise — uses real topic annotations (ag_news: World/Sports/Business/Sci-Tech).

2. Cluster mode (fallback when no .label files exist, e.g. cnn_dm/wiki):
   Computes TF-IDF vectors (no model download required), runs k-means
   with K=8 clusters, assigns pseudo-labels "cluster-0" … "cluster-7".
   Weaker signal but keeps T04 runnable on any corpus.

Output format per doc:
  {"tags": [{"label": "<str>", "score": 1.0}]}

Usage:
  python3 scripts/synth_teacher_tags.py <corpus-dir> <out-dir> [--k N]
"""
import argparse
import json
import os
import sys


DEFAULT_K = 8


def label_mode(corpus_dir, out_dir, stems):
    written = 0
    for stem in stems:
        lpath = os.path.join(corpus_dir, f"{stem}.label")
        if not os.path.exists(lpath):
            continue
        label = open(lpath).read().strip()
        if not label:
            continue
        out = {"tags": [{"label": label, "score": 1.0}]}
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(out, f)
        written += 1
    print(f"[synth_tags] label mode: wrote {written} tag files → {out_dir}/")


def cluster_mode(corpus_dir, out_dir, stems, k):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        sys.exit("[synth_tags] cluster fallback needs scikit-learn: pip install scikit-learn")

    texts = []
    valid_stems = []
    for stem in stems:
        path = os.path.join(corpus_dir, f"{stem}.txt")
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read().strip()
        if text:
            texts.append(text)
            valid_stems.append(stem)

    if not texts:
        sys.exit(f"[synth_tags] no .txt files found in {corpus_dir}")

    k = min(k, len(texts))
    print(f"[synth_tags] cluster mode: TF-IDF + k-means (k={k}) on {len(texts)} docs …")

    vec   = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    X     = vec.fit_transform(texts)
    km    = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
    labels = km.fit_predict(X)

    written = 0
    for stem, cluster_id in zip(valid_stems, labels):
        out = {"tags": [{"label": f"cluster-{cluster_id}", "score": 1.0}]}
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(out, f)
        written += 1

    print(f"[synth_tags] cluster mode: wrote {written} tag files → {out_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir")
    p.add_argument("out_dir")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help=f"number of clusters for fallback mode (default {DEFAULT_K})")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    txt_files   = sorted(f for f in os.listdir(args.corpus_dir) if f.endswith(".txt"))
    label_files = sorted(f for f in os.listdir(args.corpus_dir) if f.endswith(".label"))
    stems       = [f[:-4] for f in txt_files]

    if not stems:
        sys.exit(f"[synth_tags] no .txt files in {args.corpus_dir}")

    if label_files:
        label_mode(args.corpus_dir, args.out_dir, stems)
    else:
        print(f"[synth_tags] no .label files found — falling back to cluster mode")
        cluster_mode(args.corpus_dir, args.out_dir, stems, args.k)


if __name__ == "__main__":
    main()
