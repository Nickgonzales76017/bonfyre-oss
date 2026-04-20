#!/usr/bin/env python3
"""
synth_teacher_readability.py — T17-C synthetic readability/complexity teacher.

No ML model downloads. Pure Python: computes per-document lexical and
structural complexity features, clusters into K buckets, outputs tag-dir
format compatible with collapse_train.py --task topic-map.

Features per document:
  - avg_sentence_len:  characters per sentence
  - avg_word_len:      characters per word
  - type_token_ratio:  unique tokens / total tokens (lexical richness)
  - punct_density:     punctuation chars / total chars
  - long_word_ratio:   words > 8 chars / total words
  - sentence_count:    log(1 + n_sentences)

Cluster with MiniBatchKMeans(k=5) → complexity-0 ... complexity-4.
Highest cluster index = most complex. Stable across corpora.

Output per document: {tags: [{label: "complexity-N", score: float}]}
(same format as synth_teacher_keywords.py → drop-in --tag-dir replacement)

Usage:
  python3 synth_teacher_readability.py <corpus_dir> <out_dir> [--k 5]
"""
import argparse
import json
import math
import os
import re
import string
import sys


def extract_features(text: str) -> list:
    """Return a 6-dim feature vector for one document."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        sentences = [text]

    words = re.findall(r'\b\w+\b', text.lower())
    total_words = max(len(words), 1)
    total_chars = max(len(text), 1)

    avg_sentence_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
    avg_word_len     = sum(len(w) for w in words) / total_words
    type_token_ratio = len(set(words)) / total_words
    punct_density    = sum(1 for c in text if c in string.punctuation) / total_chars
    long_word_ratio  = sum(1 for w in words if len(w) > 8) / total_words
    sentence_count   = math.log1p(len(sentences))

    return [
        avg_sentence_len,
        avg_word_len,
        type_token_ratio,
        punct_density,
        long_word_ratio,
        sentence_count,
    ]


def normalize_features(matrix: list) -> list:
    """Min-max normalize each feature column independently."""
    n_feats = len(matrix[0])
    mins = [min(row[i] for row in matrix) for i in range(n_feats)]
    maxs = [max(row[i] for row in matrix) for i in range(n_feats)]
    normed = []
    for row in matrix:
        normed.append([
            (row[i] - mins[i]) / max(maxs[i] - mins[i], 1e-9)
            for i in range(n_feats)
        ])
    return normed


def kmeans_batch(vectors: list, k: int, max_iter: int = 50, seed: int = 42) -> list:
    """Lightweight pure-Python k-means (no sklearn required)."""
    import random
    random.seed(seed)
    n = len(vectors)
    dim = len(vectors[0])

    # Init: k-means++ style — pick first centroid randomly, rest by distance
    centroids = [vectors[random.randint(0, n - 1)]]
    for _ in range(k - 1):
        dists = []
        for v in vectors:
            d = min(sum((v[j] - c[j]) ** 2 for j in range(dim)) for c in centroids)
            dists.append(d)
        total = sum(dists)
        r = random.random() * total
        cumul = 0.0
        for idx, d in enumerate(dists):
            cumul += d
            if cumul >= r:
                centroids.append(vectors[idx])
                break
        else:
            centroids.append(vectors[-1])

    labels = [0] * n
    for _ in range(max_iter):
        # Assign
        new_labels = []
        for v in vectors:
            best = min(range(k), key=lambda c: sum((v[j] - centroids[c][j]) ** 2 for j in range(dim)))
            new_labels.append(best)

        # Check convergence
        if new_labels == labels:
            break
        labels = new_labels

        # Update centroids
        sums   = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for idx, label in enumerate(labels):
            counts[label] += 1
            for j in range(dim):
                sums[label][j] += vectors[idx][j]
        for c in range(k):
            if counts[c] > 0:
                centroids[c] = [sums[c][j] / counts[c] for j in range(dim)]

    return labels


def rank_clusters_by_complexity(labels: list, features: list, k: int) -> dict:
    """Return mapping old_label → complexity_rank (0=simplest, k-1=most complex).

    Complexity proxy: mean of avg_sentence_len + avg_word_len + long_word_ratio
    (features 0, 1, 4).
    """
    cluster_scores = {}
    counts = {}
    for label, feat in zip(labels, features):
        s = feat[0] + feat[1] + feat[4]   # sentence_len + word_len + long_word_ratio
        cluster_scores[label] = cluster_scores.get(label, 0.0) + s
        counts[label] = counts.get(label, 0) + 1

    mean_scores = {c: cluster_scores[c] / counts[c] for c in cluster_scores}
    ranked = sorted(mean_scores, key=mean_scores.__getitem__)  # simplest first
    return {old: new for new, old in enumerate(ranked)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir")
    p.add_argument("out_dir")
    p.add_argument("--k", type=int, default=5, help="number of complexity buckets")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(args.corpus_dir) if f.endswith(".txt"))
    stems = [f[:-4] for f in txt_files]

    if not stems:
        sys.exit(f"[synth_readability] no .txt files in {args.corpus_dir}")

    print(f"[synth_readability] extracting features for {len(stems)} documents …")
    raw_features = []
    texts = []
    for stem in stems:
        path = os.path.join(args.corpus_dir, f"{stem}.txt")
        text = open(path, encoding="utf-8").read().strip()
        texts.append(text)
        raw_features.append(extract_features(text))

    normed = normalize_features(raw_features)

    print(f"[synth_readability] clustering into k={args.k} complexity buckets …")
    k = min(args.k, len(stems))
    labels = kmeans_batch(normed, k=k)
    rank_map = rank_clusters_by_complexity(labels, raw_features, k)

    # Compute per-doc distance to centroid as confidence score
    # (proxy: 1 - relative_distance)
    written = 0
    for stem, label, feat_orig, feat_norm in zip(stems, labels, raw_features, normed):
        complexity_rank = rank_map[label]
        tag_label = f"complexity-{complexity_rank}"

        # Score: use type_token_ratio as primary richness signal (0→1)
        score = round(feat_orig[2], 4)  # type_token_ratio

        out = {
            "tags": [
                {
                    "label": tag_label,
                    "score": score,
                    "features": {
                        "avg_sentence_len": round(feat_orig[0], 2),
                        "avg_word_len":     round(feat_orig[1], 3),
                        "type_token_ratio": round(feat_orig[2], 4),
                        "punct_density":    round(feat_orig[3], 4),
                        "long_word_ratio":  round(feat_orig[4], 4),
                        "sentence_count":   round(feat_orig[5], 3),
                    }
                }
            ]
        }

        out_path = os.path.join(args.out_dir, f"{stem}.json")
        with open(out_path, "w") as f:
            json.dump(out, f)
        written += 1

    label_dist = {}
    for label in labels:
        rank = rank_map[label]
        label_dist[f"complexity-{rank}"] = label_dist.get(f"complexity-{rank}", 0) + 1

    print(f"[synth_readability] wrote {written} tag files → {args.out_dir}")
    print(f"[synth_readability] distribution: {dict(sorted(label_dist.items()))}")


if __name__ == "__main__":
    main()
