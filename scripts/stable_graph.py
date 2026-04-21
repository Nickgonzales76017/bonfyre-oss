#!/usr/bin/env python3
"""
scripts/stable_graph.py — Extract stable/fragile/conflict graph layers

Separates the claim graph into three distinct layers:
  1. STABLE GRAPH   — high claim_strength, low conflict_density
  2. FRAGILE GRAPH  — high support but high conflict
  3. CONFLICT GRAPH — unresolved conflicts, high pressure

STABLE EDGES
============
Criteria:
  - claim_strength >= min_strength (default: 0.5)
  - conflict_density <= max_conflict_density (default: 0.3)
  - stability_score >= 0.3 (survives multiple runs)

FRAGILE EDGES
=============
Criteria:
  - support_count >= 2
  - conflict_density >= 0.5
  - high assumption_fragility

CONFLICT GRAPH
==============
All unresolved conflicts with pressure scores.

USAGE (library):
    from scripts.stable_graph import extract_graph_layers
    layers = extract_graph_layers(claim_graph, min_strength=0.5)
    stable = layers["stable_graph"]
    fragile = layers["fragile_graph"]
    conflict = layers["conflict_graph"]

USAGE (CLI):
    python3 scripts/stable_graph.py \
        --memory-dir /tmp/bonfyre-memory \
        --out-stable /tmp/stable_graph.json \
        --out-fragile /tmp/fragile_graph.json \
        --out-conflict /tmp/conflict_graph.json

OUTPUT FORMAT:
    Each graph is a list of nodes + edges:
    {
      "nodes": [
        {"id": "claim_123", "subject": "John Smith", "predicate": "arrived_on",
         "object": "2024-03-14", "strength": 0.85, ...},
        ...
      ],
      "edges": [
        {"from": "claim_123", "to": "claim_456", "type": "supports", "strength": 0.7},
        {"from": "claim_789", "to": "claim_123", "type": "conflicts", "strength": 0.6},
        ...
      ]
    }
"""

import json
import os
import sys
from typing import Dict, List

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))

from scripts.claim_graph import ClaimGraph


def extract_graph_layers(
    claim_graph: ClaimGraph,
    min_strength: float = 0.5,
    max_conflict_density: float = 0.3,
    min_support: int = 2,
    min_fragile_conflict_density: float = 0.5,
) -> Dict:
    """
    Extract stable/fragile/conflict graph layers.

    Returns:
        {
            "stable_graph": {...},
            "fragile_graph": {...},
            "conflict_graph": {...}
        }
    """

    # 1. STABLE GRAPH
    stable_claims = claim_graph.get_stable_edges(
        min_strength=min_strength,
        max_conflict_density=max_conflict_density,
    )

    stable_graph = _build_graph_structure(
        claims=stable_claims,
        claim_graph=claim_graph,
        graph_type="stable",
    )

    # 2. FRAGILE GRAPH
    fragile_claims = claim_graph.get_fragile_edges(
        min_support=min_support,
        min_conflict_density=min_fragile_conflict_density,
    )

    fragile_graph = _build_graph_structure(
        claims=fragile_claims,
        claim_graph=claim_graph,
        graph_type="fragile",
    )

    # 3. CONFLICT GRAPH
    conflict_graph = _build_conflict_graph(claim_graph)

    return {
        "stable_graph": stable_graph,
        "fragile_graph": fragile_graph,
        "conflict_graph": conflict_graph,
    }


def _build_graph_structure(
    claims: List[Dict],
    claim_graph: ClaimGraph,
    graph_type: str,
) -> Dict:
    """
    Build graph structure (nodes + edges) from claim list.
    """
    nodes = []
    edges = []

    claim_id_set = {c["id"] for c in claims}

    # Build nodes
    for claim in claims:
        nodes.append({
            "id": f"claim_{claim['id']}",
            "claim_id": claim["id"],
            "subject": claim["subject"],
            "predicate": claim["predicate"],
            "object": claim["object"],
            "doc_id": claim["doc_id"],
            "lens": claim["lens"],
            "family": claim.get("family"),
            "confidence": claim["confidence"],
            "claim_strength": claim.get("claim_strength", 0.0),
            "stability_score": claim.get("stability_score", 0.0),
            "support_count": claim.get("support_count", 0),
            "conflict_count": claim.get("conflict_count", 0),
            "assumptions": json.loads(claim.get("assumptions", "[]")),
            "graph_type": graph_type,
        })

    # Build edges (support + conflict links)
    for claim in claims:
        claim_id = claim["id"]

        # Support edges
        support_rows = claim_graph._db.execute("""
            SELECT claim_a, claim_b, strength FROM support_links
            WHERE (claim_a = ? OR claim_b = ?)
        """, (claim_id, claim_id)).fetchall()

        for row in support_rows:
            other_id = row["claim_b"] if row["claim_a"] == claim_id else row["claim_a"]
            if other_id in claim_id_set:
                edges.append({
                    "from": f"claim_{claim_id}",
                    "to": f"claim_{other_id}",
                    "type": "supports",
                    "strength": row["strength"],
                })

        # Conflict edges
        conflict_rows = claim_graph._db.execute("""
            SELECT claim_a, claim_b, conflict_type, strength FROM conflicts
            WHERE (claim_a = ? OR claim_b = ?)
        """, (claim_id, claim_id)).fetchall()

        for row in conflict_rows:
            other_id = row["claim_b"] if row["claim_a"] == claim_id else row["claim_a"]
            if other_id in claim_id_set:
                edges.append({
                    "from": f"claim_{claim_id}",
                    "to": f"claim_{other_id}",
                    "type": "conflicts",
                    "conflict_type": row["conflict_type"],
                    "strength": row["strength"],
                })

    return {
        "graph_type": graph_type,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _build_conflict_graph(claim_graph: ClaimGraph) -> Dict:
    """
    Build conflict-only graph from unresolved clusters.
    """
    # Get unresolved clusters
    clusters = claim_graph._db.execute("""
        SELECT * FROM conflict_clusters
        WHERE resolved = 0
        ORDER BY pressure_score DESC
    """).fetchall()

    nodes = []
    edges = []

    for cluster in clusters:
        cluster_id = cluster["id"]
        cluster_type = cluster["cluster_type"]

        # Get conflicts in this cluster
        conflict_ids = claim_graph._db.execute("""
            SELECT conflict_id FROM cluster_members WHERE cluster_id = ?
        """, (cluster_id,)).fetchall()

        for row in conflict_ids:
            conflict_id = row["conflict_id"]

            # Get conflict details
            conflict = claim_graph._db.execute("""
                SELECT * FROM conflicts WHERE id = ?
            """, (conflict_id,)).fetchone()

            if not conflict:
                continue

            # Get both claims
            claim_a = claim_graph._db.execute(
                "SELECT * FROM claims WHERE id = ?", (conflict["claim_a"],)
            ).fetchone()
            claim_b = claim_graph._db.execute(
                "SELECT * FROM claims WHERE id = ?", (conflict["claim_b"],)
            ).fetchone()

            if not claim_a or not claim_b:
                continue

            # Add nodes (if not already added)
            for claim in [claim_a, claim_b]:
                node_id = f"claim_{claim['id']}"
                if not any(n["id"] == node_id for n in nodes):
                    nodes.append({
                        "id": node_id,
                        "claim_id": claim["id"],
                        "subject": claim["subject"],
                        "predicate": claim["predicate"],
                        "object": claim["object"],
                        "doc_id": claim["doc_id"],
                        "lens": claim["lens"],
                        "confidence": claim["confidence"],
                        "graph_type": "conflict",
                        "cluster_type": cluster_type,
                    })

            # Add conflict edge
            edges.append({
                "from": f"claim_{claim_a['id']}",
                "to": f"claim_{claim_b['id']}",
                "type": "conflicts",
                "conflict_type": conflict["conflict_type"],
                "strength": conflict["strength"],
                "cluster_id": cluster_id,
                "cluster_type": cluster_type,
                "pressure_score": cluster["pressure_score"],
            })

    return {
        "graph_type": "conflict",
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_clusters": len(clusters),
        "nodes": nodes,
        "edges": edges,
    }


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Extract stable/fragile/conflict graph layers")
    ap.add_argument("--memory-dir", default="/tmp/bonfyre-memory",
                    help="Bonfyre memory directory")
    ap.add_argument("--min-strength", type=float, default=0.5,
                    help="Min claim_strength for stable edges")
    ap.add_argument("--max-conflict-density", type=float, default=0.3,
                    help="Max conflict_density for stable edges")
    ap.add_argument("--min-support", type=int, default=2,
                    help="Min support_count for fragile edges")
    ap.add_argument("--min-fragile-conflict-density", type=float, default=0.5,
                    help="Min conflict_density for fragile edges")
    ap.add_argument("--out-stable", help="Output path for stable_graph.json")
    ap.add_argument("--out-fragile", help="Output path for fragile_graph.json")
    ap.add_argument("--out-conflict", help="Output path for conflict_graph.json")
    ap.add_argument("--out-all", help="Output path for all layers combined")

    args = ap.parse_args()

    cg = ClaimGraph(args.memory_dir)

    # Compute scores first
    print("[stable_graph] Computing claim scores...")
    cg.compute_claim_scores()

    # Extract layers
    print("[stable_graph] Extracting graph layers...")
    layers = extract_graph_layers(
        cg,
        min_strength=args.min_strength,
        max_conflict_density=args.max_conflict_density,
        min_support=args.min_support,
        min_fragile_conflict_density=args.min_fragile_conflict_density,
    )

    # Print summary
    print(f"\n[stable_graph] Layer summary:")
    print(f"  STABLE:   {layers['stable_graph']['n_nodes']} nodes, "
          f"{layers['stable_graph']['n_edges']} edges")
    print(f"  FRAGILE:  {layers['fragile_graph']['n_nodes']} nodes, "
          f"{layers['fragile_graph']['n_edges']} edges")
    print(f"  CONFLICT: {layers['conflict_graph']['n_nodes']} nodes, "
          f"{layers['conflict_graph']['n_edges']} edges, "
          f"{layers['conflict_graph']['n_clusters']} clusters")

    # Write outputs
    if args.out_stable:
        with open(args.out_stable, "w") as f:
            json.dump(layers["stable_graph"], f, indent=2)
        print(f"[stable_graph] → {args.out_stable}")

    if args.out_fragile:
        with open(args.out_fragile, "w") as f:
            json.dump(layers["fragile_graph"], f, indent=2)
        print(f"[stable_graph] → {args.out_fragile}")

    if args.out_conflict:
        with open(args.out_conflict, "w") as f:
            json.dump(layers["conflict_graph"], f, indent=2)
        print(f"[stable_graph] → {args.out_conflict}")

    if args.out_all:
        with open(args.out_all, "w") as f:
            json.dump(layers, f, indent=2)
        print(f"[stable_graph] → {args.out_all}")

    if not any([args.out_stable, args.out_fragile, args.out_conflict, args.out_all]):
        print("\n[stable_graph] No output files specified — printing summary only")


if __name__ == "__main__":
    main()
