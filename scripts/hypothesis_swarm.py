#!/usr/bin/env python3
"""
scripts/hypothesis_swarm.py — Bonfyre Hypothesis Swarm Runner

Orchestrates running 10-50 lenses over the same corpus slice in parallel
(or serial iteration).  Each lens produces independent claims from its narrow
perspective.  All claims are persisted to claim_graph, then conflicts are
detected, clustered, and pressure zones identified.

HYPOTHESIS SWARM WORKFLOW:
==========================
1. Select corpus slice (doc_ids or text input)
2. Select lens set (all 10, or subset)
3. Run each lens → collect claims
4. Persist all claims to claim_graph
5. Detect conflicts (same subject+predicate, different object)
6. Cluster conflicts by type
7. Compute pressure zones (high conflict density per doc/span)
8. Emit:
   - claim_graph.json (all claims)
   - conflict_report.json (detected conflicts)
   - cluster_report.json (grouped conflict patterns)
   - pressure_zones.json (hot reprocessing targets)

USAGE (library):
    from scripts.hypothesis_swarm import run_swarm
    result = run_swarm(
        corpus={"doc1": "text...", "doc2": "text..."},
        lens_ids=["L01_deposition_parser", "L03_euphemism_detector"],
        memory_dir="/tmp/bonfyre-memory",
    )

USAGE (CLI):
    python3 scripts/hypothesis_swarm.py --docs /tmp/corpus/*.txt \
        --lenses all --memory-dir /tmp/bonfyre-memory --out /tmp/swarm_result.json

    python3 scripts/hypothesis_swarm.py --text "doc text" --doc-id 123 \
        --lenses L01,L03,L06 --memory-dir /tmp/bonfyre-memory

OUTPUT:
    {
      "n_lenses_ran": 10,
      "n_docs_processed": 5,
      "n_claims_total": 247,
      "n_conflicts_detected": 18,
      "n_clusters": 4,
      "elapsed_sec": 2.3,
      "pressure_zones": [...top 10 hot zones...],
      "cluster_summary": {...},
      "conflict_report_path": "/tmp/bonfyre-memory/swarm_conflicts.json",
    }
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))


def run_swarm(
    corpus: Dict[str, str],
    lens_ids: List[str] = None,
    memory_dir: str = "/tmp/bonfyre-memory",
    run_id: int = None,
    min_confidence: float = 0.25,
    output_dir: str = None,
) -> Dict:
    """
    Run hypothesis swarm over corpus.

    Args:
        corpus: dict of {doc_id: doc_text}
        lens_ids: list of lens IDs to run (default: all 10)
        memory_dir: path to Bonfyre memory dir (for claim_graph persistence)
        run_id: optional Bonfyre run_id to link claims to specific transform run
        min_confidence: minimum confidence threshold for conflict detection
        output_dir: where to write swarm outputs (default: memory_dir)

    Returns:
        Summary dict with claim counts, conflict counts, pressure zones
    """
    from scripts.lens_registry import LENS_FUNCTIONS, run_lens
    from scripts.claim_graph import ClaimGraph

    start_time = time.time()

    # Default to all lenses
    if lens_ids is None:
        lens_ids = list(LENS_FUNCTIONS.keys())

    # Validate lens IDs
    for lid in lens_ids:
        if lid not in LENS_FUNCTIONS:
            raise ValueError(f"Unknown lens: {lid}")

    # Initialize claim graph
    cg = ClaimGraph(memory_dir)

    # Run each lens on each doc
    all_claims = []
    lens_results = []

    for doc_id, doc_text in corpus.items():
        for lens_id in lens_ids:
            try:
                result = run_lens(lens_id, doc_text, doc_id)
                lens_results.append(result)
                for claim in result["claims"]:
                    claim["doc_id"] = doc_id
                    claim["lens"] = lens_id
                    all_claims.append(claim)
            except Exception as e:
                print(f"[hypothesis_swarm] ERROR running {lens_id} on {doc_id}: {e}",
                      file=sys.stderr)

    # Persist all claims to claim graph
    claim_ids = []
    for claim in all_claims:
        try:
            cid = cg.record_claim(claim, run_id=run_id)
            claim_ids.append(cid)
        except Exception as e:
            print(f"[hypothesis_swarm] WARN failed to persist claim: {e}", file=sys.stderr)

    # Detect conflicts
    conflicts = []
    for doc_id in corpus.keys():
        doc_conflicts = cg.detect_conflicts(doc_id=doc_id, min_confidence=min_confidence)
        conflicts.extend(doc_conflicts)

    # Persist conflicts
    conflict_ids = cg.persist_conflicts(conflicts)

    # Tag conflicts with persisted IDs for clustering
    for i, cid in enumerate(conflict_ids):
        if i < len(conflicts):
            conflicts[i]["_persisted_id"] = cid

    # Detect support links
    support_links = cg.detect_support_links(min_confidence=min_confidence)
    cg.persist_support_links(support_links)

    # Cluster conflicts
    clusters = cg.cluster_conflicts(conflicts)

    # Compute pressure zones
    pressure_zones = cg.get_pressure_zones(top_n=20)

    # Elapsed time
    elapsed_sec = round(time.time() - start_time, 3)

    # Output dir
    if output_dir is None:
        output_dir = memory_dir
    os.makedirs(output_dir, exist_ok=True)

    # Write outputs
    conflict_report_path = os.path.join(output_dir, "swarm_conflicts.json")
    with open(conflict_report_path, "w") as f:
        json.dump({
            "n_conflicts": len(conflicts),
            "conflicts": conflicts[:100],  # truncate for readability
        }, f, indent=2)

    cluster_report_path = os.path.join(output_dir, "swarm_clusters.json")
    with open(cluster_report_path, "w") as f:
        json.dump({
            "n_clusters": len(clusters),
            "clusters": clusters,
        }, f, indent=2)

    pressure_report_path = os.path.join(output_dir, "swarm_pressure_zones.json")
    with open(pressure_report_path, "w") as f:
        json.dump({
            "n_zones": len(pressure_zones),
            "zones": pressure_zones,
        }, f, indent=2)

    claim_report_path = os.path.join(output_dir, "swarm_claims.json")
    with open(claim_report_path, "w") as f:
        json.dump({
            "n_claims": len(all_claims),
            "claims": all_claims[:200],  # truncate
        }, f, indent=2)

    # Summary
    cluster_summary = {}
    for cl in clusters:
        ct = cl["cluster_type"]
        cluster_summary[ct] = cluster_summary.get(ct, 0) + 1

    return {
        "n_lenses_ran": len(lens_ids),
        "n_docs_processed": len(corpus),
        "n_claims_total": len(all_claims),
        "n_conflicts_detected": len(conflicts),
        "n_support_links": len(support_links),
        "n_clusters": len(clusters),
        "elapsed_sec": elapsed_sec,
        "pressure_zones": pressure_zones[:10],
        "cluster_summary": cluster_summary,
        "conflict_report_path": conflict_report_path,
        "cluster_report_path": cluster_report_path,
        "pressure_report_path": pressure_report_path,
        "claim_report_path": claim_report_path,
    }


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Bonfyre Hypothesis Swarm Runner")

    # Input: corpus from files or stdin
    inp_group = ap.add_mutually_exclusive_group(required=True)
    inp_group.add_argument("--docs", help="Glob pattern for docs (e.g. /tmp/*.txt)")
    inp_group.add_argument("--text", help="Single doc text (provide --doc-id)")

    ap.add_argument("--doc-id", help="Document ID for --text mode")
    ap.add_argument("--lenses", default="all",
                    help="Comma-separated lens IDs or 'all' (default: all)")
    ap.add_argument("--memory-dir", default="/tmp/bonfyre-memory",
                    help="Bonfyre memory directory")
    ap.add_argument("--run-id", type=int, help="Optional Bonfyre run_id to link claims")
    ap.add_argument("--min-confidence", type=float, default=0.25,
                    help="Min confidence for conflict detection")
    ap.add_argument("--out", help="Output directory for swarm results")

    args = ap.parse_args()

    # Build corpus
    corpus = {}
    if args.docs:
        paths = glob.glob(args.docs)
        if not paths:
            print(f"[hypothesis_swarm] ERROR: no files matched {args.docs!r}", file=sys.stderr)
            sys.exit(1)
        for path in paths:
            doc_id = os.path.basename(path)
            with open(path, "r") as f:
                corpus[doc_id] = f.read()
    else:  # --text
        if not args.doc_id:
            doc_id = "doc_stdin"
        else:
            doc_id = args.doc_id
        corpus[doc_id] = args.text

    # Parse lens IDs
    if args.lenses == "all":
        from scripts.lens_registry import LENS_FUNCTIONS
        lens_ids = list(LENS_FUNCTIONS.keys())
    else:
        lens_ids = [l.strip() for l in args.lenses.split(",") if l.strip()]

    # Run swarm
    result = run_swarm(
        corpus=corpus,
        lens_ids=lens_ids,
        memory_dir=args.memory_dir,
        run_id=args.run_id,
        min_confidence=args.min_confidence,
        output_dir=args.out,
    )

    # Print summary
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
