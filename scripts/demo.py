#!/usr/bin/env python3
"""
scripts/demo.py — Bonfyre end-to-end demo

Takes one or more text inputs, runs the full transform mesh pipeline:

  1. Embed text     — MiniLM-L6-v2 → 384-dim float32 vectors
  2. Corpus stats   — compute avg_doc_len, vocab_size for routing
  3. Fragment run   — T04-frag pre-process (cheap, sub-model)
  4. Route          — pick full family (T04 / T15 / T16) from stats
  5. Chain + loop   — auto-run with FPQx alignment, N iterations
  6. Read output    — per-iteration artifact, convergence curve
  7. Classify       — argmax over logit dims for topic labels

Usage:
    python3 scripts/demo.py "Your text here"
    python3 scripts/demo.py --text-file docs/sample.txt
    python3 scripts/demo.py "Short doc" "Another doc" "A third one"
    python3 scripts/demo.py --loop 5 --models-dir /tmp/bonfyre-families "text..."

Output is printed to stdout in a readable summary.
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLI_BIN     = os.path.join(REPO_ROOT, "cmd", "BonfyreSLI", "bonfyre-sli")
MODEL_BIN   = os.path.join(REPO_ROOT, "cmd", "BonfyreModel", "bonfyre-model")
FPQX_BIN    = os.path.join(REPO_ROOT, "cmd", "BonfyreFPQX", "bonfyre-fpqx")

# ── Topic labels (T04 task: 4-class topic-map, ag_news labels) ───────────
TOPIC_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
CHUNK_LABELS = {0: "continuous", 1: "boundary"}

# ── I/O helpers ───────────────────────────────────────────────────────────

def write_vecs(path, vecs):
    """Write list-of-lists as BQFP-SLI input binary: [n, dim, float32...]"""
    n = len(vecs)
    d = len(vecs[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        for row in vecs:
            f.write(struct.pack(f"<{d}f", *row))


def read_vecs(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        vecs = []
        for _ in range(n):
            vecs.append(list(struct.unpack(f"<{d}f", f.read(d * 4))))
    return vecs, d


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 1e-12 and nb > 1e-12 else 0.0


def argmax(v):
    return max(range(len(v)), key=lambda i: v[i])


# ── Corpus stats ──────────────────────────────────────────────────────────

def compute_stats(texts):
    n_docs    = len(texts)
    avg_len   = sum(len(t) for t in texts) / max(n_docs, 1)
    vocab     = set()
    for t in texts:
        vocab.update(t.lower().split())
    return {"n_docs": n_docs, "avg_doc_len": round(avg_len, 1), "vocab_size": len(vocab)}


# ── Embed texts with MiniLM ───────────────────────────────────────────────

def embed_texts(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embs.tolist()


# ── Route via bonfyre-model ───────────────────────────────────────────────

def route(stats_path):
    result = subprocess.run(
        [MODEL_BIN, "route", stats_path],
        capture_output=True, text=True
    )
    out = result.stdout.strip()
    for tok in out.split():
        if tok.startswith("family="):
            return tok[7:]
    return "T04"  # safe default


# ── Run bonfyre-sli run ───────────────────────────────────────────────────

def sli_run(in_path, model_path, out_path):
    result = subprocess.run(
        [SLI_BIN, "run", "--in", in_path, "--model", model_path, "--out", out_path],
        capture_output=True, text=True
    )
    return result.returncode == 0


# ── Run bonfyre-sli auto-run ──────────────────────────────────────────────

def sli_auto_run(in_path, stats_path, out_dir, models_dir, loop, chain, fpqx, thresh):
    result = subprocess.run(
        [SLI_BIN, "auto-run",
         "--in",         in_path,
         "--stats",      stats_path,
         "--out",        out_dir,
         "--loop",       str(loop),
         "--chain",      chain,
         "--fpqx",       fpqx,
         "--thresh",     str(thresh),
         "--models-dir", models_dir],
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


# ── Read final SLI output vectors ────────────────────────────────────────

def read_last_iter(out_dir, loop):
    """Find the last iter directory and read its vectors."""
    for i in range(loop, 0, -1):
        vec_path = os.path.join(out_dir, f"iter-{i}", "vectors.bin")
        art_path = os.path.join(out_dir, f"iter-{i}", "artifact.json")
        if os.path.exists(vec_path):
            vecs, d = read_vecs(vec_path)
            art = json.load(open(art_path)) if os.path.exists(art_path) else {}
            return i, vecs, art
    return 0, [], {}


# ── Collect per-iter delta from SLI log ──────────────────────────────────

def parse_iter_log(log):
    lines = []
    for line in log.splitlines():
        if "iter " in line and "route →" in line:
            import re
            m = re.search(r"iter\s+(\d+)/\d+:\s+route\s+→\s+(\w+)\s+\(delta=([^)]+)\)", line)
            if m:
                it = int(m.group(1))
                fam = m.group(2)
                delta_s = m.group(3)
                delta = float(delta_s) if delta_s != "n/a" else None
                lines.append((it, fam, delta))
    return lines


# ── Classify from final vectors ───────────────────────────────────────────

def classify(vecs, family, n_originals):
    """
    Attempt a crude classification: reduce 16-dim SLI output to label.
    SLI operates on projected-down 16-dim tiles, not raw logits —
    we use the argmax of the first 4 dims as a proxy label, or
    chunk-boundary if family is T16.
    """
    results = []
    for i, v in enumerate(vecs[:n_originals]):
        if family == "T16":
            label_map = CHUNK_LABELS
            idx = int(v[0] > 0)
        else:
            label_map = TOPIC_LABELS
            idx = argmax(v[:4])
        results.append(label_map.get(idx, f"class_{idx}"))
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Bonfyre end-to-end demo")
    ap.add_argument("texts", nargs="*", help="Text inputs to process")
    ap.add_argument("--text-file", help="File with one text per line")
    ap.add_argument("--loop",       type=int, default=5)
    ap.add_argument("--thresh",     type=float, default=0.0)
    ap.add_argument("--models-dir", default="/tmp/bonfyre-families")
    ap.add_argument("--chain",      default="fragment:auto")
    ap.add_argument("--fpqx",       default="auto")
    ap.add_argument("--no-fragment", action="store_true",
                    help="Skip fragment pre-process step")
    args = ap.parse_args()

    texts = list(args.texts)
    if args.text_file:
        with open(args.text_file) as f:
            texts += [ln.rstrip() for ln in f if ln.strip()]
    if not texts:
        texts = [
            "Apple reports record quarterly revenue driven by iPhone sales.",
            "Scientists discover new exoplanet in the habitable zone.",
            "World leaders gather for climate summit negotiations.",
            "Champions League final draws largest TV audience in years.",
        ]
        print("(no input given — using 4 built-in demo texts)\n")

    models_dir = args.models_dir
    frag_bqfp  = os.path.join(models_dir, "T04-frag.bqfp")
    use_frag   = not args.no_fragment and os.path.exists(frag_bqfp)

    print("=" * 72)
    print(" BONFYRE  end-to-end demo")
    print(f"  inputs  : {len(texts)} text(s)")
    print(f"  loop    : {args.loop} iterations")
    print(f"  chain   : {args.chain}")
    print(f"  fpqx    : {args.fpqx}")
    print(f"  fragment: {'yes (T04-frag)' if use_frag else 'no'}")
    print("=" * 72)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:

        # ── 1. Embed ───────────────────────────────────────────────────
        print("── 1. Embedding texts via MiniLM-L6-v2 ─────────────────────────")
        embs = embed_texts(texts)
        print(f"     {len(embs)} vectors × {len(embs[0])} dim")
        raw_path = os.path.join(tmpdir, "embeddings.bin")
        write_vecs(raw_path, embs)
        print()

        # ── 2. Corpus stats + route ────────────────────────────────────
        print("── 2. Corpus stats + routing ────────────────────────────────────")
        stats = compute_stats(texts)
        stats_path = os.path.join(tmpdir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        routed_family = route(stats_path)
        print(f"     n_docs={stats['n_docs']}  avg_doc_len={stats['avg_doc_len']}"
              f"  vocab={stats['vocab_size']}")
        print(f"     → routed to family: {routed_family}")
        print()

        # ── 3. Fragment pre-process ─────────────────────────────────
        # When --chain fragment:auto is active, SLI handles the fragment
        # preflight internally (iter 1 applies <family>-frag if present).
        # We still surface it here for diagnostic visibility.
        if use_frag and args.chain == "fragment:auto":
            print("── 3. Fragment pre-process ──────────── (built into chain; handled by SLI)\n")
            in_path = raw_path
        elif use_frag:
            print("── 3. Fragment pre-process (T04-frag, sub-model first-hop) ─────")
            frag_out = os.path.join(tmpdir, "frag_out.bin")
            ok = sli_run(raw_path, frag_bqfp, frag_out)
            if ok:
                frag_vecs, _ = read_vecs(frag_out)
                cos_vals = [cosine(embs[i], frag_vecs[i]) for i in range(len(embs))]
                avg_cos = sum(cos_vals) / len(cos_vals)
                print(f"     {len(embs)} vectors transformed  (raw→frag cosine: {avg_cos:.4f})")
            else:
                frag_out = raw_path
                print("     (fragment run failed — using raw embeddings)")
            in_path = frag_out
            print()
        else:
            in_path = raw_path
            print("── 3. Fragment pre-process ──────────── (skipped)\n")

        # ── 4. Auto-run: route → align → loop ────────────────────────
        print(f"── 4. auto-run  chain={args.chain}  fpqx={args.fpqx}"
              f"  loop={args.loop}  thresh={args.thresh} ─────")
        auto_out = os.path.join(tmpdir, "auto")

        log = sli_auto_run(
            in_path, stats_path, auto_out,
            models_dir, args.loop, args.chain, args.fpqx, args.thresh
        )

        iters = parse_iter_log(log)
        # Also capture preflight line
        has_preflight = "preflight:" in log
        if has_preflight:
            fam = routed_family
            print(f"     preflight  family={fam}-frag  (fragment applied before iter 1)")
        for it, fam, delta in iters:
            delta_s = f"{delta:.4f}" if delta is not None else "n/a"
            print(f"     iter {it:2d}  family={fam}  delta={delta_s}")
        print()

        # ── 5. Read final output + convergence ────────────────────────
        final_iter, final_vecs, final_art = read_last_iter(auto_out, args.loop)
        if not final_vecs:
            print("  ERROR: no output vectors found")
            sys.exit(1)

        families_seen = list(dict.fromkeys(fam for _, fam, _ in iters))
        final_family = iters[-1][1] if iters else routed_family
        deltas = [d for _, _, d in iters if d is not None]
        delta_trend = " → ".join(f"{d:.3f}" for d in deltas[:3]) + \
                      (" → … → " + f"{deltas[-1]:.3f}" if len(deltas) > 3 else "")

        # ── 6. Classify ───────────────────────────────────────────────
        labels = classify(final_vecs, final_family, len(texts))

        # ── 7. Per-iter cosine from raw input ─────────────────────────
        raw_vecs, _ = read_vecs(raw_path)
        final_cos_vals = [cosine(raw_vecs[i], final_vecs[i]) for i in range(len(raw_vecs))]
        avg_final_cos  = sum(final_cos_vals) / len(final_cos_vals)

        # ── Summary ───────────────────────────────────────────────────
        print("=" * 72)
        print(" RESULTS")
        print("=" * 72)
        print(f"  families traversed : {' → '.join(families_seen)}")
        print(f"  final family       : {final_family}")
        print(f"  iterations ran     : {final_iter}")
        print(f"  delta curve        : {delta_trend}")
        print(f"  raw→final cosine   : {avg_final_cos:.4f}"
              f"  (transform displacement from origin)")
        print()
        print(f"  {'#':<4}  {'label':<12}  {'input text'}")
        print("  " + "─" * 65)
        for i, (text, label) in enumerate(zip(texts, labels)):
            excerpt = (text[:52] + "…") if len(text) > 55 else text
            print(f"  {i+1:<4}  {label:<12}  {excerpt}")
        print()
        print("=" * 72)
        print(" fragment  → full-model → aligned chain → looped convergence")
        print(" routing   → chaining   → stabilized output")
        print("=" * 72)


if __name__ == "__main__":
    main()
