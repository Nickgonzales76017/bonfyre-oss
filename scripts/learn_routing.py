#!/usr/bin/env python3
"""
scripts/learn_routing.py — learn routing weights from observed confidence

Problem:
  Current routing score = 0.7 * mean_f1 + 0.3 * cosine_mean (hardcoded)
  The right weights depend on which combination maximizes ONNX confidence
  of the SELECTED family, across multiple input distributions.

Method:
  1. Define a grid of alpha values (weight on f1; cosine weight = 1 - alpha)
  2. For each alpha: re-score all families, simulate which family would be selected
  3. Run ONNX head on both NEWS and ARTICLE corpora for that selection
  4. Compute avg(max_conf) across all inputs as the objective
  5. Pick alpha* that maximizes avg_conf
  6. Write routing_weights.json to models_dir

Output:
  routing_weights.json  — { "f1_weight": alpha*, "cosine_weight": 1-alpha* }
  Printed sweep table    — alpha vs selected_family vs avg_conf per corpus

Usage:
    python3 scripts/learn_routing.py
    python3 scripts/learn_routing.py --models-dir /tmp/bonfyre-families --alphas 11
    python3 scripts/learn_routing.py --write

Note:
  This does NOT retrain any model. It only learns the routing weight — the
  single hyperparameter that controls the F1-vs-cosine tradeoff. The mesh
  topology (frontier.json, geometry conditions) is fixed.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_BIN = os.path.join(REPO_ROOT, "cmd", "BonfyreModel", "bonfyre-model")

# ── Family metadata (mirrors compare.py) ─────────────────────────────────────
FAMILIES = {
    "T04": {
        "path":    "/tmp/bonfyre-72/runs/T04-C-ag_news-1000/train/model.onnx",
        "n_class": 4,
        "labels":  {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    },
    "T15": {
        "path":    "/tmp/bonfyre-72/runs/T15-C-cnn_dm-1000/train/model.onnx",
        "n_class": 6,
        "labels":  {0: "Politics", 1: "Tech", 2: "Business",
                    3: "Health", 4: "World", 5: "Sports"},
    },
}

# ── Family DB data (from bonfyre-model, replicated for offline scoring) ───────
# These are the fixed values — routing weights only change the blend ratio.
FAMILY_DATA = {
    "T04": {"f1": 0.914, "condition": "avg_doc_len <= 300"},
    "T15": {"f1": 0.911, "condition": None},
    "T16": {"f1": 0.931, "condition": "avg_doc_len > 500"},
    "T08": {"f1": 0.871, "condition": None},
    "T14": {"f1": 0.823, "condition": None},
}

# ── Corpora ───────────────────────────────────────────────────────────────────
NEWS_TEXTS = [
    "Apple reports record quarterly revenue driven by iPhone and services growth.",
    "Manchester City wins Champions League final in dramatic extra-time penalty shootout.",
    "Fed signals interest rate pause as inflation cools to near-target levels.",
    "New AI chip from NVIDIA promises threefold compute gain over prior generation.",
    "Senate confirms new Supreme Court justice in close party-line vote.",
    "SpaceX launches 60-satellite Starlink cluster in routine Falcon 9 mission.",
]

ARTICLE_TEXTS = [
    (
        "In a sweeping executive order signed Thursday, the administration outlined "
        "a broad new framework for regulating artificial intelligence across federal "
        "agencies, requiring safety audits, transparency disclosures, and mandatory "
        "impact assessments before deployment of any AI system in a government role."
    ),
    (
        "Climate scientists released a landmark report Wednesday documenting "
        "accelerating ice loss across the Antarctic shelf, warning that sea-level "
        "projections from prior models significantly underestimated the pace of "
        "melt driven by warming ocean currents."
    ),
    (
        "The Federal Reserve's latest minutes revealed a split among board members "
        "over the appropriate pace of rate adjustments, with several governors "
        "expressing concern that premature easing could reignite inflationary pressures."
    ),
    (
        "Researchers at three major university hospitals have published preliminary "
        "results from a multi-year longitudinal study linking chronic ultra-processed "
        "food consumption to measurably elevated markers of systemic inflammation."
    ),
]

CORPORA = {
    "news":    {"texts": NEWS_TEXTS,    "avg_doc_len": 75.0,  "expected_family": "T04"},
    "article": {"texts": ARTICLE_TEXTS, "avg_doc_len": 350.0, "expected_family": "T15"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def softmax(row):
    m = max(row)
    e = [math.exp(x - m) for x in row]
    s = sum(e)
    return [x / s for x in e]


def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return m.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def eval_condition(condition, avg_doc_len):
    """Evaluate a simple geometry condition string against avg_doc_len."""
    if not condition:
        return True
    import re
    m = re.match(r"avg_doc_len\s*([<>=!]+)\s*([\d.]+)", condition)
    if not m:
        return True
    op, thresh = m.group(1), float(m.group(2))
    if op == ">":  return avg_doc_len > thresh
    if op == ">=": return avg_doc_len >= thresh
    if op == "<":  return avg_doc_len < thresh
    if op == "<=": return avg_doc_len <= thresh
    return True


def frontier_cosine(frontier, fa, fb):
    """Look up cosine_mean for a pair from loaded frontier dict."""
    for pair in frontier.get("pairs", []):
        if (pair["family_a"] == fa and pair["family_b"] == fb) or \
           (pair["family_a"] == fb and pair["family_b"] == fa):
            return pair["cosine_mean"]
    return None


def select_family(alpha, avg_doc_len, frontier, from_family="T04"):
    """
    Simulate routing: pick family with highest (alpha*f1 + (1-alpha)*cosine_mean)
    among families whose geometry_condition passes.
    """
    best_fam   = None
    best_score = -1.0
    for fam, data in FAMILY_DATA.items():
        if not eval_condition(data["condition"], avg_doc_len):
            continue
        f1  = data["f1"]
        cos = frontier_cosine(frontier, from_family, fam)
        if cos is None:
            score = f1  # no cosine pair → pure f1
        else:
            score = alpha * f1 + (1.0 - alpha) * cos
        if score > best_score:
            best_score = score
            best_fam   = fam
    return best_fam


def avg_max_conf(embs, family):
    """Run ONNX head for family, return (avg, min) of max softmax prob."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        return 0.0, 0.0
    spec = FAMILIES.get(family)
    if not spec or not os.path.exists(spec["path"]):
        # family not in FAMILIES dict — no head available; penalize
        return 0.3, 0.3
    sess = ort.InferenceSession(spec["path"], providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    logits = sess.run(None, {iname: np.array(embs, dtype=np.float32)})[0]
    max_probs = [max(softmax(list(row))) for row in logits]
    return sum(max_probs) / len(max_probs), min(max_probs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Learn routing weights from observed confidence")
    ap.add_argument("--models-dir", default="/tmp/bonfyre-families")
    ap.add_argument("--alphas",     type=int, default=11,
                    help="Number of alpha values to sweep (default 11: 0.0 to 1.0 step 0.1)")
    ap.add_argument("--write",      action="store_true",
                    help="Write routing_weights.json to models-dir")
    ap.add_argument("--from-family", default="T04",
                    help="Origin family for cosine lookup (fragment preflight origin)")
    args = ap.parse_args()

    frontier_path = os.path.join(args.models_dir, "frontier.json")
    if not os.path.exists(frontier_path):
        print(f"ERROR: frontier.json not found at {frontier_path}")
        sys.exit(1)

    with open(frontier_path) as f:
        frontier = json.load(f)

    print("=" * 72)
    print(" BONFYRE  learn routing weights")
    print(f"  method : grid sweep over alpha (f1_weight) with {args.alphas} steps")
    print(f"  origin : --from {args.from_family}")
    print(f"  corpora: news ({len(NEWS_TEXTS)} texts) + article ({len(ARTICLE_TEXTS)} texts)")
    print("=" * 72)

    # ── Embed both corpora once ───────────────────────────────────────────────
    print("\n  Embedding texts …", end="", flush=True)
    corpus_embs = {
        name: embed(corpus["texts"])
        for name, corpus in CORPORA.items()
    }
    print(" done")

    # ── Sweep ────────────────────────────────────────────────────────────────
    alphas = [round(i / (args.alphas - 1), 2) for i in range(args.alphas)]
    results = []

    print()
    print(f"  {'alpha':>6}  {'selected (news)':>16}  {'selected (art)':>16}  "
          f"{'conf_news':>10}  {'conf_art':>10}  {'combined':>10}")
    print("  " + "─" * 76)

    for alpha in alphas:
        row = {"alpha": alpha}
        total_conf = 0.0
        for cname, corpus in CORPORA.items():
            fam = select_family(alpha, corpus["avg_doc_len"], frontier, args.from_family)
            avg_c, min_c = avg_max_conf(corpus_embs[cname], fam)
            row[cname] = {"family": fam, "avg_conf": avg_c, "min_conf": min_c}
            total_conf += avg_c
        row["combined"] = total_conf / len(CORPORA)
        results.append(row)

        news_fam = row["news"]["family"]
        art_fam  = row["article"]["family"]
        print(f"  {alpha:>6.2f}  {news_fam:>16}  {art_fam:>16}  "
              f"  {row['news']['avg_conf']*100:>7.1f}%  "
              f"{row['article']['avg_conf']*100:>7.1f}%  "
              f"{row['combined']*100:>7.1f}%")

    # ── Find best alpha ───────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["combined"])
    best_alpha = best["alpha"]
    best_cos_w = round(1.0 - best_alpha, 10)

    print()
    print("─" * 72)
    print(f"  best alpha  = {best_alpha:.2f}  (f1_weight)")
    print(f"  cos_weight  = {best_cos_w:.2f}  (cosine_mean weight)")
    print(f"  combined avg conf at best alpha: {best['combined']*100:.1f}%")

    # ── Compare best vs default (hardcoded 0.70/0.30) ────────────────────────
    default = next((r for r in results if abs(r["alpha"] - 0.70) < 0.01), None)
    if default:
        delta = (best["combined"] - default["combined"]) * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  vs default (0.70/0.30): {default['combined']*100:.1f}%  "
              f"→  learned gain: {sign}{delta:.1f}%")

    # ── Check if routing changed ──────────────────────────────────────────────
    default_news_fam = default["news"]["family"] if default else "?"
    default_art_fam  = default["article"]["family"] if default else "?"
    best_news_fam    = best["news"]["family"]
    best_art_fam     = best["article"]["family"]
    routing_changed  = (best_news_fam != default_news_fam) or (best_art_fam != default_art_fam)

    print()
    if routing_changed:
        print(f"  routing CHANGED vs default:")
        if best_news_fam != default_news_fam:
            print(f"    news:    {default_news_fam} → {best_news_fam}")
        if best_art_fam != default_art_fam:
            print(f"    article: {default_art_fam} → {best_art_fam}")
    else:
        print(f"  routing unchanged: news→{best_news_fam}  article→{best_art_fam}")
        print(f"  learned weights change the score but not the winner at these corpora")

    # ── Write output ──────────────────────────────────────────────────────────
    weights = {
        "schema":        "bonfyre-routing-weights-v1",
        "f1_weight":     best_alpha,
        "cosine_weight": round(best_cos_w, 10),
        "learned_from":  list(CORPORA.keys()),
        "corpora_n":     {k: len(v["texts"]) for k, v in CORPORA.items()},
        "best_combined_conf": round(best["combined"], 6),
        "default_combined_conf": round(default["combined"], 6) if default else None,
        "sweep_n":       len(alphas),
    }

    out_path = os.path.join(args.models_dir, "routing_weights.json")
    if args.write:
        with open(out_path, "w") as f:
            json.dump(weights, f, indent=2)
        print(f"\n  wrote: {out_path}")
    else:
        print(f"\n  (dry run — use --write to save to {out_path})")

    print()
    print("=" * 72)
    print(" interpretation")
    print("=" * 72)
    print(f"  alpha > 0.5: weighting model quality (F1) more than mesh alignment (cosine)")
    print(f"  alpha < 0.5: weighting mesh alignment more than raw model quality")
    print(f"  alpha = 1.0: pure F1 routing (ignore frontier topology)")
    print(f"  alpha = 0.0: pure cosine routing (ignore model quality, follow mesh edges)")
    print()
    print(f"  learned alpha={best_alpha:.2f} means:")
    if best_alpha > 0.7:
        print(f"  → F1 quality dominates; cosine provides gentle tie-breaking")
    elif best_alpha < 0.5:
        print(f"  → mesh topology dominates; routing follows strong frontier edges")
    else:
        print(f"  → balanced: F1 and cosine equally weighted in routing decisions")
    print("=" * 72)

    return weights


if __name__ == "__main__":
    main()
