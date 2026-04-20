#!/usr/bin/env python3
"""
scripts/bench_fragment.py — benchmark fragment:auto vs plain auto-run

Measures the direct cost/benefit of the T04-frag preflight:

  WITH    --chain fragment:auto   (cheap first hop before full transform)
  WITHOUT --chain auto            (full transform immediately each iter)

Metrics per run:
  latency_ms   wall-clock ms for auto-run call
  iters_run    number of iterations executed
  delta_1      delta at iter 1  (first full-transform step)
  delta_final  delta at final iter
  delta_drop   delta_final - delta_1  (convergence span)
  cos_final    raw-to-final cosine (transform displacement)
  stable_iter  first iter where delta > 0.95 (stability threshold)

Usage:
    python3 scripts/bench_fragment.py
    python3 scripts/bench_fragment.py --loop 8 --runs 3 --models-dir /tmp/bonfyre-families
    python3 scripts/bench_fragment.py --text-file docs/sample.txt --loop 6
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLI_BIN   = os.path.join(REPO_ROOT, "cmd", "BonfyreSLI", "bonfyre-sli")
MODEL_BIN = os.path.join(REPO_ROOT, "cmd", "BonfyreModel", "bonfyre-model")

DEMO_TEXTS = [
    "Apple reports record quarterly revenue driven by iPhone and services.",
    "Scientists identify new exoplanet in the habitable zone of a nearby star.",
    "World leaders convene in Geneva for climate summit negotiations.",
    "Manchester City wins Champions League in extra time penalty shootout.",
    "Federal Reserve signals interest rate pause amid cooling inflation data.",
    "New study links ultra-processed food consumption to increased cancer risk.",
    "Senate passes bipartisan infrastructure bill after weeks of debate.",
    "SpaceX Starship completes first full orbital flight test successfully.",
]


# ── I/O helpers ─────────────────────────────────────────────────────────────

def write_vecs(path, vecs):
    n, d = len(vecs), len(vecs[0])
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
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 1e-12 and nb > 1e-12 else 0.0


def embed_texts(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embs.tolist()


def compute_stats(texts):
    n_docs  = len(texts)
    avg_len = sum(len(t) for t in texts) / max(n_docs, 1)
    vocab   = set()
    for t in texts:
        vocab.update(t.lower().split())
    return {"n_docs": n_docs, "avg_doc_len": round(avg_len, 1), "vocab_size": len(vocab)}


# ── Parse bonfyre-sli auto-run log ──────────────────────────────────────────

def parse_log(log):
    """Return list of (iter_num, delta) tuples from auto-run log."""
    import re
    results = []
    for line in log.splitlines():
        m = re.search(r"iter\s+(\d+)/\d+:\s+route\s+→\s+\w+\s+\(delta=([^)]+)\)", line)
        if m:
            it = int(m.group(1))
            ds = m.group(2)
            delta = float(ds) if ds != "n/a" else None
            results.append((it, delta))
    return results


def first_stable_iter(iters, threshold=0.95):
    """Return the first iteration where delta >= threshold, or None."""
    for it, d in iters:
        if d is not None and d >= threshold:
            return it
    return None


# ── Single bench run ─────────────────────────────────────────────────────────

def bench_single(in_path, stats_path, out_dir, models_dir, loop, chain):
    t0 = time.perf_counter()
    result = subprocess.run(
        [SLI_BIN, "auto-run",
         "--in",         in_path,
         "--stats",      stats_path,
         "--out",        out_dir,
         "--loop",       str(loop),
         "--chain",      chain,
         "--fpqx",       "auto",
         "--thresh",     "0.0",
         "--models-dir", models_dir],
        capture_output=True, text=True
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    log = result.stdout + result.stderr
    iters = parse_log(log)
    return latency_ms, log, iters


def read_final_vecs(out_dir, loop):
    for i in range(loop, 0, -1):
        p = os.path.join(out_dir, f"iter-{i}", "vectors.bin")
        if os.path.exists(p):
            vecs, _ = read_vecs(p)
            return i, vecs
    return 0, []


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Benchmark fragment:auto vs auto-run")
    ap.add_argument("texts",        nargs="*", help="Text inputs (uses built-in 8 if omitted)")
    ap.add_argument("--text-file",  help="File with one text per line")
    ap.add_argument("--loop",       type=int, default=6)
    ap.add_argument("--runs",       type=int, default=3, help="Repeat each condition N times")
    ap.add_argument("--models-dir", default="/tmp/bonfyre-families")
    ap.add_argument("--stable-thresh", type=float, default=0.95,
                    help="Delta threshold for 'stable' labeling")
    args = ap.parse_args()

    texts = list(args.texts)
    if args.text_file:
        with open(args.text_file) as f:
            texts += [ln.rstrip() for ln in f if ln.strip()]
    if not texts:
        texts = DEMO_TEXTS

    models_dir = args.models_dir
    frag_bqfp  = os.path.join(models_dir, "T04-frag.bqfp")
    has_frag   = os.path.exists(frag_bqfp)

    print("=" * 72)
    print(" BONFYRE  fragment:auto  benchmark")
    print(f"  texts      : {len(texts)}")
    print(f"  loop       : {args.loop} iters each run")
    print(f"  runs       : {args.runs} × per condition")
    print(f"  models-dir : {models_dir}")
    print(f"  fragment   : {'found — T04-frag.bqfp' if has_frag else 'NOT FOUND'}")
    print("=" * 72)

    if not has_frag:
        print(f"\n  ERROR: {frag_bqfp} not found — cannot benchmark fragment path")
        sys.exit(1)

    print("\n  Embedding texts with MiniLM-L6-v2 …", end="", flush=True)
    embs = embed_texts(texts)
    print(f" {len(embs)} vectors × {len(embs[0])} dim")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path   = os.path.join(tmpdir, "embeddings.bin")
        stats_path = os.path.join(tmpdir, "stats.json")
        write_vecs(raw_path, embs)
        stats = compute_stats(texts)
        with open(stats_path, "w") as f:
            json.dump(stats, f)

        conditions = [
            ("fragment:auto",  "fragment:auto"),
            ("auto (no frag)", "auto"),
        ]

        all_results = {}
        for label, chain in conditions:
            print(f"\n  Running: chain={chain} × {args.runs} …", end="", flush=True)
            run_metrics = []
            for r in range(args.runs):
                out_dir = os.path.join(tmpdir, f"out-{chain.replace(':','-')}-{r}")
                lat, log, iters = bench_single(
                    raw_path, stats_path, out_dir, models_dir, args.loop, chain)
                has_preflight = "preflight:" in log
                final_iter, final_vecs = read_final_vecs(out_dir, args.loop)

                deltas = [d for _, d in iters if d is not None]
                delta_1     = deltas[0] if deltas else None
                delta_final = deltas[-1] if deltas else None
                stable_i    = first_stable_iter(iters, args.stable_thresh)

                avg_cos = None
                if final_vecs:
                    cos_vals = [cosine(embs[i], final_vecs[i]) for i in range(len(embs))]
                    avg_cos  = sum(cos_vals) / len(cos_vals)

                run_metrics.append({
                    "latency_ms":    lat,
                    "iters_run":     final_iter,
                    "delta_1":       delta_1,
                    "delta_final":   delta_final,
                    "cos_final":     avg_cos,
                    "stable_iter":   stable_i,
                    "has_preflight": has_preflight,
                })
                print(".", end="", flush=True)
            all_results[label] = run_metrics
            print()

        # ── Aggregate (mean over runs) ────────────────────────────────────
        def mean(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        agg = {}
        for label, runs in all_results.items():
            agg[label] = {
                "latency_ms":  mean([r["latency_ms"]  for r in runs]),
                "iters_run":   mean([r["iters_run"]   for r in runs]),
                "delta_1":     mean([r["delta_1"]     for r in runs]),
                "delta_final": mean([r["delta_final"] for r in runs]),
                "cos_final":   mean([r["cos_final"]   for r in runs]),
                "stable_iter": mean([r["stable_iter"] for r in runs]),
            }

        # ── Print comparison table ────────────────────────────────────────
        frag_m   = agg["fragment:auto"]
        nofrag_m = agg["auto (no frag)"]

        def fmt(v, fmt_s=".1f", suffix=""):
            return f"{v:{fmt_s}}{suffix}" if v is not None else "n/a"

        def delta_pct(a, b):
            """% difference: a vs b; positive = a is better (higher)."""
            if a is None or b is None or b == 0:
                return ""
            d = (a - b) / abs(b) * 100
            sign = "+" if d > 0 else ""
            return f"  ({sign}{d:.1f}%)"

        def delta_ms(a, b):
            if a is None or b is None:
                return ""
            d = a - b
            sign = "+" if d > 0 else ""
            return f"  ({sign}{d:.0f} ms)"

        print()
        print("=" * 72)
        print(" RESULTS  (mean over", args.runs, "runs each)")
        print("=" * 72)
        print(f"  {'metric':<26}  {'fragment:auto':>14}  {'auto (no frag)':>14}  {'Δ'}")
        print("  " + "─" * 68)

        lat_f  = frag_m["latency_ms"]
        lat_n  = nofrag_m["latency_ms"]
        print(f"  {'latency (ms)':<26}  {fmt(lat_f,'.0f'):>14}  "
              f"{fmt(lat_n,'.0f'):>14}  {delta_ms(lat_f, lat_n)}")

        d1_f = frag_m["delta_1"];   d1_n = nofrag_m["delta_1"]
        print(f"  {'delta at iter 1':<26}  {fmt(d1_f,'.4f'):>14}  "
              f"{fmt(d1_n,'.4f'):>14}  {delta_pct(d1_f, d1_n)}")

        df_f = frag_m["delta_final"]; df_n = nofrag_m["delta_final"]
        print(f"  {'delta at final iter':<26}  {fmt(df_f,'.4f'):>14}  "
              f"{fmt(df_n,'.4f'):>14}  {delta_pct(df_f, df_n)}")

        cos_f = frag_m["cos_final"]; cos_n = nofrag_m["cos_final"]
        print(f"  {'raw→final cosine':<26}  {fmt(cos_f,'.4f'):>14}  "
              f"{fmt(cos_n,'.4f'):>14}  {delta_pct(cos_f, cos_n)}")

        stab_f = frag_m["stable_iter"]; stab_n = nofrag_m["stable_iter"]
        stab_delta = ""
        if stab_f is not None and stab_n is not None:
            diff = stab_n - stab_f
            stab_delta = f"  ({'earlier' if diff > 0 else 'same'} by {abs(diff):.1f} iters)" if diff != 0 else "  (same)"
        print(f"  {'stable iter (δ≥{:.2f})'.format(args.stable_thresh):<26}  "
              f"{fmt(stab_f,'.1f'):>14}  {fmt(stab_n,'.1f'):>14}  {stab_delta}")

        print()
        print("  fragment:auto faster?  ", "yes" if lat_f < lat_n else "no",
              f"  ({abs(lat_f - lat_n):.0f} ms)" if lat_f is not None and lat_n is not None else "")
        print("  fragment raises delta_1?",
              "yes" if d1_f is not None and d1_n is not None and d1_f > d1_n else "no")
        print("  fragment improves stability?",
              "yes" if stab_f is not None and stab_n is not None and stab_f < stab_n else
              ("same" if stab_f == stab_n else "no"))
        print()
        print("=" * 72)
        print(" interpretation")
        print("=" * 72)
        if lat_f < lat_n:
            print(f"  fragment preflight is FASTER by {lat_n - lat_f:.0f} ms on {args.loop} iters ({len(texts)} texts)")
        else:
            print(f"  fragment preflight adds {lat_f - lat_n:.0f} ms startup cost")
        if d1_f is not None and d1_n is not None:
            if d1_f > d1_n:
                print(f"  fragment raises iter-1 delta from {d1_n:.4f} → {d1_f:.4f}:")
                print(f"  the pre-processed state starts closer to convergence")
            else:
                print(f"  fragment does not alter iter-1 delta ({d1_f:.4f} vs {d1_n:.4f})")
        if stab_f is not None and stab_n is not None and stab_f < stab_n:
            print(f"  stability reached {stab_n - stab_f:.1f} iters earlier with fragment")
        print("=" * 72)


if __name__ == "__main__":
    main()
