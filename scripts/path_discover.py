#!/usr/bin/env python3
"""
scripts/path_discover.py — Bonfyre chain path discovery.

Discovers new execution chains by:
  1. Enumerating known families from domain_families.json + FAMILY_REGISTRY
  2. Generating candidate chains (controlled combinatorial expansion)
  3. Benchmarking each candidate against the baseline (fragment:auto)
  4. Scoring: composite = avg_confidence / avg_iterations (efficiency)
  5. Storing winners in BonfyreMemory.paths table

WHAT IT EXPLORES
================
Given base families [T04, T15, T16, T08], it tries:
  - fragment hops interleaved: T04 → frag → T15 → frag → T16
  - cross-task shortcuts:     T15 → T08 → T16
  - skip-level chains:        T04 → T16 (bypassing T15)
  - domain family insertions: T04 → T20 (finance) → T15

It does NOT try all permutations (combinatorial explosion). Instead:
  - max chain length = 4 families
  - fragment hops can only be inserted between consecutive families
  - always starts from the family recommended by bonfyre-model route
  - max candidates per run = MAX_CANDIDATES (default 20)

SCORING
=======
  score = avg_confidence / max(avg_iterations, 1)

Higher score = same confidence in fewer iterations = more efficient.
A new chain is "discovered" if score > baseline + DISCOVERY_THRESHOLD.

USAGE
=====
    python3 scripts/path_discover.py [--memory-dir /tmp/bonfyre-memory]
                                      [--models-dir /tmp/bonfyre-families]
                                      [--texts "text1" "text2"]
                                      [--loop 5]
                                      [--max-candidates 20]
                                      [--dry-run]
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

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))

from scripts.bonfyre_memory import BonfyreMemory  # noqa: E402

REPO_ROOT = os.path.dirname(_SELF)
SLI_BIN   = os.path.join(REPO_ROOT, "cmd", "BonfyreSLI",   "bonfyre-sli")
MODEL_BIN = os.path.join(REPO_ROOT, "cmd", "BonfyreModel",  "bonfyre-model")

# ── Constants ─────────────────────────────────────────────────────────────

MAX_CANDIDATES       = 20
MAX_CHAIN_LENGTH     = 4
DISCOVERY_THRESHOLD  = 0.05   # new path must beat baseline score by this margin
DEFAULT_LOOP         = 5
STABILITY_THRESHOLD  = 0.95   # delta at which we call iteration "stable"

DEMO_TEXTS = [
    "Apple reports record quarterly revenue driven by iPhone sales.",
    "Scientists discover new exoplanet in the habitable zone.",
    "World leaders convene for climate summit negotiations.",
    "Federal Reserve signals interest rate pause amid cooling inflation.",
]

# ── Base families always available ────────────────────────────────────────

BASE_FAMILIES = ["T04", "T15", "T16", "T08", "T14"]


# ── I/O helpers ──────────────────────────────────────────────────────────

def write_vecs(path, vecs):
    n, d = len(vecs[0]) and len(vecs), len(vecs[0])
    with open(path, "wb") as fh:
        fh.write(struct.pack("<II", n, d))
        for row in vecs:
            fh.write(struct.pack(f"<{d}f", *row))


def embed_texts(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(texts, show_progress_bar=False,
                            normalize_embeddings=True)
        return embs.tolist()
    except ImportError:
        # Fallback: random unit vectors (tests structure without dependencies)
        import random
        d = 384
        vecs = []
        for _ in texts:
            v = [random.gauss(0, 1) for _ in range(d)]
            norm = math.sqrt(sum(x * x for x in v))
            vecs.append([x / norm for x in v])
        return vecs


def compute_stats(texts):
    avg_len = sum(len(t) for t in texts) / max(len(texts), 1)
    vocab = set()
    for t in texts:
        vocab.update(t.lower().split())
    return {"n_docs": len(texts), "avg_doc_len": round(avg_len, 1),
            "vocab_size": len(vocab)}


# ── Parse SLI auto-run log ────────────────────────────────────────────────

def parse_log(log: str) -> list:
    """Return list of (iter_num, family, delta) from bonfyre-sli output."""
    import re
    results = []
    for line in log.splitlines():
        m = re.search(
            r"iter\s+(\d+)/\d+:\s+route\s+→\s+(\w+)\s+\(delta=([^)]+)\)", line)
        if m:
            delta_s = m.group(3)
            results.append((
                int(m.group(1)),
                m.group(2),
                float(delta_s) if delta_s != "n/a" else None,
            ))
    return results


def first_stable_iter(iters, threshold=STABILITY_THRESHOLD):
    for it, _, d in iters:
        if d is not None and d >= threshold:
            return it
    return None


# ── Single chain benchmark ────────────────────────────────────────────────

def bench_chain(chain_str: str, texts: list, models_dir: str,
                n_loop: int = DEFAULT_LOOP) -> dict:
    """
    Run one chain against texts, return timing + confidence metrics.

    Returns:
        {chain, n_loop, avg_delta_final, min_delta, iters_stable,
         wall_ms, error}
    """
    if not os.path.exists(SLI_BIN):
        return {"chain": chain_str, "error": "bonfyre-sli not found"}

    try:
        embs = embed_texts(texts)
    except Exception as e:
        return {"chain": chain_str, "error": f"embed failed: {e}"}

    with tempfile.TemporaryDirectory() as td:
        in_path  = os.path.join(td, "input.bin")
        out_path = os.path.join(td, "output.bin")

        n, d = len(embs), len(embs[0])
        with open(in_path, "wb") as fh:
            fh.write(struct.pack("<II", n, d))
            for row in embs:
                fh.write(struct.pack(f"<{d}f", *row))

        stats = compute_stats(texts)
        stats_path = os.path.join(td, "stats.json")
        with open(stats_path, "w") as fh:
            json.dump(stats, fh)

        frontier = os.path.join(models_dir, "frontier.json")
        # Use adjusted frontier if it exists (prefer learned weights)
        adj_frontier = os.path.join(
            os.path.dirname(models_dir), "bonfyre-memory", "graph",
            "frontier_adjusted.json")
        if os.path.exists(adj_frontier):
            frontier = adj_frontier

        cmd = [
            SLI_BIN, "auto-run",
            "--in", in_path, "--out", out_path,
            "--models-dir", models_dir,
            "--chain", chain_str,
            "--n", str(n_loop),
        ]
        if os.path.exists(frontier):
            cmd += ["--frontier", frontier]

        t0 = time.monotonic()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            wall_ms = int((time.monotonic() - t0) * 1000)
            log = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return {"chain": chain_str, "error": "timeout"}
        except Exception as e:
            return {"chain": chain_str, "error": str(e)}

        iters = parse_log(log)
        if not iters:
            return {"chain": chain_str, "error": "no iter output", "log": log[:400]}

        deltas = [d for _, _, d in iters if d is not None]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
        min_delta = min(deltas) if deltas else 0.0
        stable_it = first_stable_iter(iters) or n_loop
        score = avg_delta / max(stable_it, 1)

        return {
            "chain":           chain_str,
            "n_loop":          n_loop,
            "avg_delta_final": round(avg_delta, 4),
            "min_delta":       round(min_delta, 4),
            "iters_stable":    stable_it,
            "wall_ms":         wall_ms,
            "score":           round(score, 4),
            "error":           None,
        }


# ── Candidate generation ──────────────────────────────────────────────────

def load_available_families(models_dir: str) -> list:
    """Load all known families: base + domain families from json."""
    families = list(BASE_FAMILIES)
    domain_json = os.path.join(models_dir, "domain_families.json")
    auto_json   = os.path.join(models_dir, "auto_families.json")  # from auto_evolve

    for path in (domain_json, auto_json):
        if os.path.exists(path):
            try:
                extra = json.load(open(path))
                for fid in extra:
                    if fid not in families:
                        families.append(fid)
            except Exception:
                pass
    return families


def generate_candidates(families: list,
                         max_chain: int = MAX_CHAIN_LENGTH,
                         max_total: int = MAX_CANDIDATES) -> list:
    """
    Generate candidate chain strings without full permutation explosion.

    Strategy:
      1. Single-family (baseline):    "fragment:auto"
      2. Two-family direct:           "T04:T15:auto", "T04:T16:auto"
      3. Three-family with fragment:  "T04:fragment:T15:fragment:T16:auto"
      4. Cross-task shortcuts:        "T15:T08:auto", "T08:T15:auto"
      5. Domain family insertions:    "T04:T20:T15:auto"

    Returns list of canonical chain strings.
    """
    candidates = set()

    # Baseline
    candidates.add("fragment:auto")

    primary    = [f for f in families if f in ("T04", "T15", "T16")]
    secondary  = [f for f in families if f in ("T08", "T14")]
    domain_fam = [f for f in families if f not in set(primary + secondary)]

    # Two-family direct chains
    for i, fa in enumerate(primary):
        for j, fb in enumerate(primary):
            if fa != fb:
                candidates.add(f"{fa}:{fb}:auto")
                candidates.add(f"fragment:{fa}:{fb}:auto")

    # Cross-task shortcuts (secondary families)
    for fa in primary:
        for fb in secondary:
            candidates.add(f"{fa}:{fb}:auto")
        for fc in primary:
            if fa != fc:
                for fb in secondary:
                    candidates.add(f"{fa}:{fb}:{fc}:auto")

    # Three-family with interleaved fragments
    for fa in primary[:2]:
        for fb in primary[1:]:
            if fa != fb:
                candidates.add(f"fragment:{fa}:fragment:{fb}:auto")

    # Domain family insertions
    for df in domain_fam[:3]:   # limit: don't explode
        for fa in primary[:2]:
            candidates.add(f"{fa}:{df}:auto")
            candidates.add(f"{df}:{fa}:auto")

    # Prune to max_total, preferring shorter chains
    sorted_cands = sorted(candidates, key=lambda c: (len(c.split(":")), c))
    return sorted_cands[:max_total]


# ── Discover ──────────────────────────────────────────────────────────────

def discover(memory_dir: str, models_dir: str,
             texts: list, n_loop: int = DEFAULT_LOOP,
             max_candidates: int = MAX_CANDIDATES,
             dry_run: bool = False) -> list:
    """
    Run path discovery: benchmark candidate chains, record winners.

    Returns list of winning path dicts (new discoveries only).
    """
    mem = BonfyreMemory(memory_dir)
    families = load_available_families(models_dir)
    candidates = generate_candidates(families, max_total=max_candidates)

    print(f"[path_discover] {len(candidates)} candidate chains  "
          f"(families: {len(families)}, texts: {len(texts)})")

    # Benchmark baseline first
    baseline = bench_chain("fragment:auto", texts, models_dir, n_loop)
    baseline_score = baseline.get("score", 0.0)
    print(f"  baseline   fragment:auto  score={baseline_score:.4f}  "
          f"avg_delta={baseline.get('avg_delta_final', 0):.4f}  "
          f"stable@{baseline.get('iters_stable', '?')}")

    if not dry_run:
        mem.record_path("fragment:auto",
                        avg_confidence=baseline.get("avg_delta_final", 0.0),
                        avg_iterations=baseline.get("iters_stable", n_loop))

    # Benchmark all candidates
    results = []
    discoveries = []

    for cand in candidates:
        if cand == "fragment:auto":
            continue
        r = bench_chain(cand, texts, models_dir, n_loop)
        if r.get("error"):
            continue
        results.append(r)
        improvement = r["score"] - baseline_score
        marker = ""
        if improvement > DISCOVERY_THRESHOLD:
            marker = "  ★ NEW"
            discoveries.append(r)
            if not dry_run:
                mem.record_path(
                    chain=cand,
                    avg_confidence=r["avg_delta_final"],
                    avg_iterations=r["iters_stable"],
                )
        print(f"  {cand:<35}  score={r['score']:.4f}  "
              f"Δ={improvement:+.4f}  "
              f"ms={r['wall_ms']}{marker}")

    # Sort by score descending
    results.sort(key=lambda x: -x["score"])

    # Write discovery report
    if not dry_run and discoveries:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report_path = os.path.join(memory_dir, "paths",
                                   f"{ts.replace(':', '-')}_discoveries.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({
                "discovered_at": ts,
                "baseline":      {"chain": "fragment:auto", "score": baseline_score},
                "threshold":     DISCOVERY_THRESHOLD,
                "n_candidates":  len(candidates),
                "discoveries":   discoveries,
            }, f, indent=2)
        print(f"\n  {len(discoveries)} new path(s) discovered → {report_path}")

    return discoveries


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Discover new execution chains in the Bonfyre transform graph")
    ap.add_argument("--memory-dir",     default="/tmp/bonfyre-memory")
    ap.add_argument("--models-dir",     default="/tmp/bonfyre-families")
    ap.add_argument("--texts",          nargs="+", default=None,
                    help="Override probe texts (default: 4 built-in)")
    ap.add_argument("--text-file",      default=None,
                    help="File with one text per line to use as probe corpus")
    ap.add_argument("--loop",           type=int, default=DEFAULT_LOOP)
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    ap.add_argument("--dry-run",        action="store_true",
                    help="Benchmark but do not write to memory")
    ap.add_argument("--json",           action="store_true")
    args = ap.parse_args()

    texts = args.texts or DEMO_TEXTS
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file) as fh:
            file_texts = [ln.strip() for ln in fh if ln.strip()]
        texts = file_texts or texts

    discoveries = discover(
        memory_dir=args.memory_dir,
        models_dir=args.models_dir,
        texts=texts,
        n_loop=args.loop,
        max_candidates=args.max_candidates,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(discoveries, indent=2))
        return

    print(f"\n[path_discover] {len(discoveries)} new path(s) discovered "
          f"(threshold={DISCOVERY_THRESHOLD})")
    if discoveries:
        top = sorted(discoveries, key=lambda x: -x["score"])
        for r in top[:5]:
            print(f"  ★ {r['chain']:<40}  score={r['score']:.4f}")
    if args.dry_run:
        print("  (dry-run: no files written)")


if __name__ == "__main__":
    main()
