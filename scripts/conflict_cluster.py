#!/usr/bin/env python3
"""
scripts/conflict_cluster.py — Advanced conflict clustering + pressure analysis

While claim_graph.py provides basic conflict clustering, this module adds:
  1. Multi-attribute clustering (subject+predicate+temporal+doc grouping)
  2. Pressure score computation (contradiction_pressure, assumption_fragility)
  3. Hot-zone flagging (clusters needing lens expansion or reprocessing)
  4. Resolution tracking (marks clusters resolved by new lens passes)

WHAT IS CONFLICT PRESSURE?
===========================
Pressure = (conflict_count × avg_strength) / (support_count + 1)

High pressure → many disagreeing claims, low independent validation.
Low pressure → disagreements exist but supported by cross-validation.

HOT ZONE CRITERIA:
==================
A cluster becomes a hot zone when:
  1. pressure_score > 2.0             — high conflict density
  2. n_conflicts ≥ 3                  — recurrent pattern
  3. n_lenses ≥ 2                     — cross-lens disagreement (not single lens noise)
  4. assumption_fragility > 0.7       — claims rely on weak assumptions

USAGE (library):
    from scripts.conflict_cluster import cluster_advanced, flag_hot_zones
    clusters = cluster_advanced(claim_graph, min_conflicts=3)
    hot = flag_hot_zones(clusters, pressure_threshold=2.0)

USAGE (CLI):
    python3 scripts/conflict_cluster.py --memory-dir /tmp/bonfyre-memory \
        --min-conflicts 3 --out /tmp/clusters_advanced.json

    python3 scripts/conflict_cluster.py hot-zones --memory-dir /tmp/bonfyre-memory \
        --pressure-threshold 2.0
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))


def cluster_advanced(
    claim_graph,
    min_conflicts: int = 2,
    min_confidence: float = 0.25,
) -> List[Dict]:
    """
    Advanced conflict clustering with multi-attribute grouping.

    Groups conflicts by:
      - predicate type (entity_variant, timeline_anomaly, etc.)
      - dominant subject (most frequent entity in conflict)
      - doc cluster (conflicts spanning same doc set)

    Returns list of cluster dicts with pressure scores.
    """
    from scripts.claim_graph import ClaimGraph

    # Detect all conflicts from claim_graph
    conflicts = claim_graph.detect_conflicts(min_confidence=min_confidence)

    if len(conflicts) < min_conflicts:
        return []

    # Group by predicate type
    by_predicate = defaultdict(list)
    for c in conflicts:
        pred_type = c["conflict_type"].split(":")[0]  # normalize antonym: prefix
        by_predicate[pred_type].append(c)

    # Compute support counts per subject
    support_by_subject = {}
    sup_rows = claim_graph._db.execute("""
        SELECT c.subject, COUNT(*) as n
        FROM claims c
        JOIN support_links sl ON (sl.claim_a = c.id OR sl.claim_b = c.id)
        GROUP BY c.subject
    """).fetchall()
    for r in sup_rows:
        support_by_subject[r["subject"].lower().strip()] = r["n"]

    clusters = []

    for pred_type, group in by_predicate.items():
        if len(group) < min_conflicts:
            continue

        # Collect subjects, docs, lenses
        subjects = [c["subject"] for c in group]
        docs = list(set(
            c["claim_a"].get("doc_id", "") for c in group
        ) | set(
            c["claim_b"].get("doc_id", "") for c in group
        ))
        lenses = list(set(
            c["claim_a"].get("lens", "") for c in group
        ) | set(
            c["claim_b"].get("lens", "") for c in group
        ))

        # Dominant subject
        subject_counts = Counter(s.lower().strip() for s in subjects)
        dominant_subject, dom_count = subject_counts.most_common(1)[0] if subject_counts else ("unknown", 0)

        # Avg conflict strength
        avg_strength = sum(c["strength"] for c in group) / len(group)

        # Support count for involved subjects
        support_total = sum(support_by_subject.get(s.lower().strip(), 0) for s in set(subjects))

        # Pressure score
        pressure = round((len(group) * avg_strength) / max(support_total, 1), 4)

        # Assumption fragility (avg of claims' assumption count)
        assumption_counts = []
        for c in group:
            try:
                assumptions_a = json.loads(c["claim_a"].get("assumptions", "[]"))
                assumptions_b = json.loads(c["claim_b"].get("assumptions", "[]"))
                assumption_counts.append(len(assumptions_a) + len(assumptions_b))
            except Exception:
                assumption_counts.append(0)
        avg_assumptions = sum(assumption_counts) / max(len(assumption_counts), 1)
        # Fragility: higher assumption count → more fragile (linear scale)
        assumption_fragility = min(avg_assumptions / 5.0, 1.0)

        clusters.append({
            "cluster_type": pred_type,
            "n_conflicts": len(group),
            "n_lenses": len(lenses),
            "n_docs": len(docs),
            "dominant_subject": dominant_subject,
            "subjects": list(set(subjects))[:20],
            "docs": docs[:20],
            "lenses": lenses,
            "avg_strength": round(avg_strength, 4),
            "support_count": support_total,
            "pressure_score": pressure,
            "assumption_fragility": round(assumption_fragility, 4),
            "resolved": False,
        })

    # Sort by pressure descending
    clusters.sort(key=lambda x: -x["pressure_score"])
    return clusters


def flag_hot_zones(
    clusters: List[Dict],
    pressure_threshold: float = 2.0,
    min_conflicts: int = 3,
    min_lenses: int = 2,
    fragility_threshold: float = 0.7,
) -> List[Dict]:
    """
    Filter clusters to hot zones needing reprocessing.

    Hot zone criteria:
      - pressure_score > threshold
      - n_conflicts ≥ min_conflicts
      - n_lenses ≥ min_lenses (cross-lens disagreement)
      - assumption_fragility > fragility_threshold

    Returns list of hot zone dicts with recommended actions.
    """
    hot_zones = []

    for cl in clusters:
        is_hot = (
            cl["pressure_score"] > pressure_threshold and
            cl["n_conflicts"] >= min_conflicts and
            cl["n_lenses"] >= min_lenses and
            cl["assumption_fragility"] > fragility_threshold
        )

        if is_hot:
            # Recommend lens expansions
            recommended_lenses = _recommend_lenses_for_cluster(cl)

            hot_zones.append({
                **cl,
                "hot_zone": True,
                "recommended_lenses": recommended_lenses,
                "reprocess_docs": cl["docs"][:10],
                "reason": f"pressure={cl['pressure_score']:.2f}, "
                          f"fragility={cl['assumption_fragility']:.2f}, "
                          f"{cl['n_conflicts']} conflicts across {cl['n_lenses']} lenses",
            })

    return hot_zones


def _recommend_lenses_for_cluster(cluster: Dict) -> List[str]:
    """
    Heuristic lens recommendations based on cluster type.
    """
    LENS_MAP = {
        "entity_variant":      ["L02_alias_expansion", "L09_entity_consistency"],
        "timeline_anomaly":    ["L04_timeline_anomaly"],
        "speaker_role":        ["L01_deposition_parser"],
        "coercion_signal":     ["L06_coercion_language", "L03_euphemism_detector"],
        "redaction_found":     ["L05_redaction_shape"],
        "email_thread_depth":  ["L07_email_thread"],
        "ocr_candidate":       ["L08_ocr_restore"],
        "travel_anomaly":      ["L10_travel_anomaly"],
    }

    # Get cluster lenses already run
    existing = set(cluster.get("lenses", []))

    # Recommend lenses for this cluster type
    cluster_type = cluster["cluster_type"]
    candidates = LENS_MAP.get(cluster_type, [])

    # Filter out already-run lenses
    recommended = [lns for lns in candidates if lns not in existing]

    # If all recommended already ran, suggest "expand lens pool"
    if not recommended:
        recommended = ["expand_lens_pool"]

    return recommended


def mark_resolved(claim_graph, cluster_id: int):
    """
    Mark a cluster as resolved in claim_graph DB.
    """
    claim_graph._db.execute(
        "UPDATE conflict_clusters SET resolved=1 WHERE id=?",
        (cluster_id,)
    )
    claim_graph._db.commit()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Bonfyre conflict clustering + pressure analysis")
    sub = ap.add_subparsers(dest="cmd")

    # cluster-advanced
    p_cluster = sub.add_parser("cluster-advanced", help="Run advanced clustering")
    p_cluster.add_argument("--memory-dir", default="/tmp/bonfyre-memory")
    p_cluster.add_argument("--min-conflicts", type=int, default=2)
    p_cluster.add_argument("--min-confidence", type=float, default=0.25)
    p_cluster.add_argument("--out", help="Output JSON path")

    # hot-zones
    p_hot = sub.add_parser("hot-zones", help="Flag hot zones")
    p_hot.add_argument("--memory-dir", default="/tmp/bonfyre-memory")
    p_hot.add_argument("--pressure-threshold", type=float, default=2.0)
    p_hot.add_argument("--min-conflicts", type=int, default=3)
    p_hot.add_argument("--min-lenses", type=int, default=2)
    p_hot.add_argument("--fragility-threshold", type=float, default=0.7)
    p_hot.add_argument("--out", help="Output JSON path")

    args = ap.parse_args()

    if not args.cmd:
        ap.print_help()
        return

    from scripts.claim_graph import ClaimGraph
    cg = ClaimGraph(args.memory_dir)

    if args.cmd == "cluster-advanced":
        clusters = cluster_advanced(
            cg,
            min_conflicts=args.min_conflicts,
            min_confidence=args.min_confidence,
        )

        if args.out:
            with open(args.out, "w") as f:
                json.dump({
                    "n_clusters": len(clusters),
                    "clusters": clusters,
                }, f, indent=2)
            print(f"[conflict_cluster] {len(clusters)} cluster(s) → {args.out}")
        else:
            print(json.dumps({"n_clusters": len(clusters), "clusters": clusters}, indent=2))

    elif args.cmd == "hot-zones":
        # First cluster
        clusters = cluster_advanced(cg, min_conflicts=args.min_conflicts)

        # Then flag hot
        hot = flag_hot_zones(
            clusters,
            pressure_threshold=args.pressure_threshold,
            min_conflicts=args.min_conflicts,
            min_lenses=args.min_lenses,
            fragility_threshold=args.fragility_threshold,
        )

        if args.out:
            with open(args.out, "w") as f:
                json.dump({
                    "n_hot_zones": len(hot),
                    "hot_zones": hot,
                }, f, indent=2)
            print(f"[conflict_cluster] {len(hot)} hot zone(s) → {args.out}")
        else:
            print(f"{'type':<20}  {'pressure':>8}  {'fragility':>10}  {'n_conf':>6}  lenses  recs")
            print("─" * 90)
            for hz in hot:
                print(f"{hz['cluster_type']:<20}  "
                      f"{hz['pressure_score']:>8.4f}  "
                      f"{hz['assumption_fragility']:>10.4f}  "
                      f"{hz['n_conflicts']:>6}  "
                      f"{hz['n_lenses']:>2}  "
                      f"{', '.join(hz['recommended_lenses'][:3])}")


if __name__ == "__main__":
    main()
