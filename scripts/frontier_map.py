#!/usr/bin/env python3
"""
scripts/frontier_map.py
Generate the 5-family transform network frontier map.

Reads all 10 FPQx alignment manifests + runs eval for Procrustes preservation.
Outputs a markdown/text table + machine-readable frontier.json.

Usage:
    python3 scripts/frontier_map.py [models_dir]
"""

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bonfyre-families"
FPQX = os.path.join(REPO_ROOT, "cmd", "BonfyreFPQX", "bonfyre-fpqx")

# ── Family metadata ────────────────────────────────────────────────────────
FAMILIES = {
    "T04": {"f1": 0.914, "geometry": "global",      "task": "topic-map",      "corpus": "ag_news", "params": 99588,  "tier": "primary"},
    "T15": {"f1": 0.911, "geometry": "global",      "task": "topic-map",      "corpus": "cnn_dm",  "params": 100102, "tier": "primary"},
    "T16": {"f1": 0.931, "geometry": "conditional", "task": "chunk-boundary", "corpus": "cnn_dm",  "params": 98561,  "tier": "primary"},
    "T08": {"f1": 0.871, "geometry": "global",      "task": "topic-map",      "corpus": "cnn_dm",  "params": 24770,  "tier": "secondary"},
    "T14": {"f1": 0.823, "geometry": "global",      "task": "topic-map",      "corpus": "cnn_dm",  "params": 99331,  "tier": "secondary"},
}

# ── Pair use-case labels ───────────────────────────────────────────────────
USE_CASES = {
    ("T04", "T15"): "cross-corpus global bridge (same task, different training set)",
    ("T04", "T16"): "short→long-form geometry bridge (global → conditional)",
    ("T04", "T08"): "quality↔size tradeoff (full ↔ compact, 4× smaller)",
    ("T04", "T14"): "parallel global refinement (same geometry, lower baseline)",
    ("T08", "T14"): "compact↔full swap (low-F1 tier, size exploration)",
    ("T08", "T15"): "compact → quality boost to T15 global",
    ("T08", "T16"): "compact global → long-form conditional jump",
    ("T14", "T15"): "parallel global second-pass (T14→T15 quality upgrade)",
    ("T14", "T16"): "low-quality global → conditional long-form escalation",
    ("T15", "T16"): "same-corpus geometry upgrade (global → long-form, cnn_dm)",
}

PAIRS = list(USE_CASES.keys())


def run_eval(family_a, align_json):
    # Output: "  mean cosine(tile, A×tile): 0.9727  (1.0 = identity alignment)"
    bqfp = os.path.join(MODELS_DIR, f"{family_a}.bqfp")
    try:
        out = subprocess.check_output(
            [FPQX, "eval", bqfp, align_json], stderr=subprocess.DEVNULL
        ).decode()
        for line in out.splitlines():
            if "mean cosine" in line and ":" in line:
                val_str = line.split(":")[-1].strip().split()[0]
                return float(val_str)
    except Exception:
        pass
    return None


def find_align_json(a, b):
    """Return (actual_a, actual_b, path) trying both orderings."""
    for fa, fb in [(a, b), (b, a)]:
        p = os.path.join(MODELS_DIR, f"align-{fa}-{fb}", "fpqx_alignment.json")
        if os.path.exists(p):
            return fa, fb, p
    return a, b, None


print("=" * 80)
print(" bonfyre-fpqx transform network frontier map")
print(f" models_dir : {MODELS_DIR}")
print("=" * 80)
print()

rows = []

# ── Header ─────────────────────────────────────────────────────────────────
print(f"{'A':<4}  {'B':<4}  {'geom_A':<11}  {'geom_B':<12}  "
      f"{'anch':<5}  {'cos_mean':<8}  {'proc':<6}  use case")
print("─" * 100)

for a_orig, b_orig in PAIRS:
    a, b, align_json = find_align_json(a_orig, b_orig)

    if align_json is None:
        print(f"{a_orig:<4}  {b_orig:<4}  MISSING alignment JSON")
        continue

    with open(align_json) as f:
        manifest = json.load(f)

    cos_mean = manifest["cosine_mean"]
    n_anchors = manifest["n_anchors"]

    proc = run_eval(a, align_json)
    proc_str = f"{proc:.4f}" if proc is not None else "n/a"

    fa = FAMILIES[a]
    fb = FAMILIES[b]
    uc = USE_CASES.get((a_orig, b_orig), USE_CASES.get((b_orig, a_orig), "cross-family"))

    print(f"{a:<4}  {b:<4}  {fa['geometry']:<11}  {fb['geometry']:<12}  "
          f"{n_anchors:<5}  {cos_mean:<8.4f}  {proc_str:<6}  {uc}")

    rows.append({
        "family_a": a,
        "family_b": b,
        "f1_a": fa["f1"],
        "f1_b": fb["f1"],
        "geometry_a": fa["geometry"],
        "geometry_b": fb["geometry"],
        "task_a": fa["task"],
        "task_b": fb["task"],
        "params_a": fa["params"],
        "params_b": fb["params"],
        "n_anchors": n_anchors,
        "cosine_mean": round(cos_mean, 6),
        "procrustes_preservation": round(proc, 4) if proc else None,
        "use_case": uc,
    })

print()

# ── Write frontier.json ────────────────────────────────────────────────────
frontier = {
    "schema": "bonfyre-frontier-map-v1",
    "families": sorted(FAMILIES.keys()),
    "n_pairs": len(rows),
    "pairs": rows,
}
out_json = os.path.join(MODELS_DIR, "frontier.json")
with open(out_json, "w") as f:
    json.dump(frontier, f, indent=2)
print(f"frontier.json: {out_json}")
print()

# ── Per-family summary ──────────────────────────────────────────────────────
print(f"{'ID':<4}  {'mean_f1':<7}  {'tier':<9}  {'geometry':<11}  "
      f"{'task':<14}  {'params':<8}  corpus")
print("─" * 75)
for fid, fm in sorted(FAMILIES.items(), key=lambda x: -x[1]["f1"]):
    print(f"{fid:<4}  {fm['f1']:<7}  {fm['tier']:<9}  {fm['geometry']:<11}  "
          f"{fm['task']:<14}  {fm['params']:<8}  {fm['corpus']}")

print()
print("=" * 80)
print(" DONE: 5-family frontier map complete")
print("=" * 80)
