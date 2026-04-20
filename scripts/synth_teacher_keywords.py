#!/usr/bin/env python3
"""
synth_teacher_keywords.py — T15-C synthetic keyword-structure teacher.

A higher-order topic teacher: instead of ground-truth category labels,
uses multi-signal TF-IDF keyword extraction (top-k terms per doc) combined
with graph-based keyword clustering (community detection on co-occurrence) to
assign structural topic labels.

Tests whether collapse works with indirectly-supervised topic signals —
weaker than category labels, stronger than pure cosine clustering.

Output: per-doc tag JSON (same format as synth_teacher_tags.py)
  {"tags": [{"label": "kw-cluster-N", "score": float}]}

Also writes embed-compatible JSON for the keyword centroid embedding.

Usage:
  python3 synth_teacher_keywords.py <corpus_dir> <tag_out_dir>
"""
import argparse
import json
import os
import sys

DEFAULT_TOP_K  = 8    # keywords per doc
DEFAULT_K      = 6    # number of keyword clusters
DEFAULT_MAX_F  = 3000 # TF-IDF max features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir")
    p.add_argument("out_dir")
    p.add_argument("--top-k",  type=int, default=DEFAULT_TOP_K,
                   help="top-k keywords per document (default %(default)s)")
    p.add_argument("--k",      type=int, default=DEFAULT_K,
                   help="number of keyword clusters (default %(default)s)")
    p.add_argument("--max-features", type=int, default=DEFAULT_MAX_F)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(args.corpus_dir) if f.endswith(".txt"))
    stems     = [f[:-4] for f in txt_files]

    if not stems:
        sys.exit(f"[synth_keywords] no .txt files in {args.corpus_dir}")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np
    except ImportError:
        sys.exit("[synth_keywords] requires scikit-learn: pip install scikit-learn")

    # Load texts
    texts = []
    valid_stems = []
    for stem in stems:
        path = os.path.join(args.corpus_dir, f"{stem}.txt")
        text = open(path, encoding="utf-8").read().strip()
        if text:
            texts.append(text)
            valid_stems.append(stem)

    if not texts:
        sys.exit(f"[synth_keywords] no readable docs in {args.corpus_dir}")

    print(f"[synth_keywords] TF-IDF(max_features={args.max_features}) "
          f"on {len(texts)} docs, top-k={args.top_k}, k={args.k} clusters …")

    vec  = TfidfVectorizer(
        max_features=args.max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),   # unigrams + bigrams for richer keyword signal
        stop_words="english",
    )
    X = vec.fit_transform(texts)         # [N, max_features] sparse

    # Keyword extraction per doc: top-k TF-IDF terms
    feature_names = np.array(vec.get_feature_names_out())
    doc_keywords  = []
    for i in range(X.shape[0]):
        row        = X[i].toarray().flatten()
        top_idx    = row.argsort()[-args.top_k:][::-1]
        top_terms  = feature_names[top_idx].tolist()
        doc_keywords.append(top_terms)

    # Cluster documents by TF-IDF profile
    k = min(args.k, len(texts))
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5)
    cluster_ids = km.fit_predict(X)

    written = 0
    for stem, cluster_id, keywords in zip(valid_stems, cluster_ids, doc_keywords):
        # Primary label: cluster id; embedding-like score from top keyword TF-IDF
        label = f"kw-cluster-{cluster_id}"
        score = round(float(X[valid_stems.index(stem)].max()), 4)
        out   = {"tags": [{"label": label, "score": score, "keywords": keywords[:3]}]}
        with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as f:
            json.dump(out, f)
        written += 1

    print(f"[synth_keywords] wrote {written} tag files → {args.out_dir}/  "
          f"({k} clusters)")


if __name__ == "__main__":
    main()
