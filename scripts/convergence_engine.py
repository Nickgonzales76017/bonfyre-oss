#!/usr/bin/env python3
"""
scripts/convergence_engine.py — Bonfyre Structural Convergence Engine

Converts competing claims into stable structures through repeated independent pressure.

WHAT IS CONVERGENCE?
====================
Running the same corpus through multiple swarm passes, watching which claims:
  - consistently survive (stable edges)
  - consistently conflict (fragile edges)
  - resolve over iterations (converged clusters)

CONVERGENCE LOOP
================
For each conflict cluster:
  1. Re-run swarm ONLY on that cluster's docs/spans
  2. Increase lens diversity (add recommended lenses)
  3. Optionally escalate transform families
  4. Recompute claim strength
  5. Check if cluster stabilized (pressure < threshold)
  6. Record convergence history

Stop when:
  - All clusters stabilize (pressure < threshold)
  - OR max iterations reached
  - OR no new claims produced

USAGE (library):
    from scripts.convergence_engine import ConvergenceEngine
    engine = ConvergenceEngine(memory_dir="/tmp/bonfyre-memory")
    result = engine.run_convergence(
        corpus={"doc1": "text...", "doc2": "text..."},
        max_iterations=5,
        pressure_threshold=1.0
    )

USAGE (CLI):
    python3 scripts/convergence_engine.py \
        --docs '/tmp/corpus/*.txt' \
        --memory-dir /tmp/bonfyre-memory \
        --max-iterations 5 \
        --pressure-threshold 1.0

OUTPUT:
    {
      "converged": true,
      "iterations_ran": 3,
      "stable_edges": 87,
      "fragile_edges": 12,
      "resolved_clusters": 4,
      "pressure_decay": 0.65,
      "convergence_history": [...]
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

from scripts.claim_graph import ClaimGraph
from scripts.hypothesis_swarm import run_swarm
from scripts.conflict_cluster import cluster_advanced, flag_hot_zones


class ConvergenceEngine:
    """
    Orchestrates repeated swarm passes to resolve conflicts and stabilize claims.
    """

    def __init__(self, memory_dir: str = "/tmp/bonfyre-memory"):
        self.memory_dir = memory_dir
        self.claim_graph = ClaimGraph(memory_dir)

    def run_convergence(
        self,
        corpus: Dict[str, str],
        max_iterations: int = 5,
        pressure_threshold: float = 1.0,
        min_strength: float = 0.5,
        lens_ids: List[str] = None,
    ) -> Dict:
        """
        Execute convergence loop on corpus.

        Args:
            corpus: {doc_id: doc_text}
            max_iterations: max number of re-runs per cluster
            pressure_threshold: stop when cluster pressure < this
            min_strength: minimum claim_strength to consider stable
            lens_ids: initial lens set (default: all)

        Returns:
            Convergence report with stable/fragile edges, metrics
        """
        start_time = time.time()
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        report = {
            "converged": False,
            "iterations_ran": 0,
            "stable_edges": 0,
            "fragile_edges": 0,
            "resolved_clusters": 0,
            "pressure_decay": 0.0,
            "convergence_history": [],
            "started_at": ts,
        }

        # Initial swarm pass
        print(f"[convergence] Initial swarm pass ({len(corpus)} docs)...")
        initial_result = run_swarm(
            corpus=corpus,
            lens_ids=lens_ids,
            memory_dir=self.memory_dir,
            min_confidence=0.25,
        )

        print(f"[convergence]   {initial_result['n_claims_total']} claims, "
              f"{initial_result['n_conflicts_detected']} conflicts, "
              f"{initial_result['n_clusters']} clusters")

        # Compute initial claim scores
        print("[convergence] Computing initial claim scores...")
        self.claim_graph.compute_claim_scores()

        # Phase 14: Compute orthogonal pressure
        print("[convergence] Computing orthogonal pressure...")
        self.claim_graph.compute_orthogonal_pressure(corpus=corpus)
        self.claim_graph.recompute_final_strength()

        # Get convergence metrics before iteration
        metrics_before = self.claim_graph.get_convergence_metrics()

        # Identify hot zones
        clusters = cluster_advanced(
            self.claim_graph,
            min_conflicts=2,
            min_confidence=0.25,
        )
        hot_zones = flag_hot_zones(
            clusters,
            pressure_threshold=pressure_threshold,
            min_conflicts=2,
            min_lenses=1,
            fragility_threshold=0.5,
        )

        if not hot_zones:
            print("[convergence] No hot zones — already converged!")
            report["converged"] = True
            report["stable_edges"] = len(
                self.claim_graph.get_stable_edges(min_strength=min_strength)
            )
            report["fragile_edges"] = len(
                self.claim_graph.get_fragile_edges(min_support=2)
            )
            return report

        print(f"[convergence] {len(hot_zones)} hot zone(s) identified")

        # Convergence loop
        for iteration in range(1, max_iterations + 1):
            print(f"\n[convergence] ── Iteration {iteration}/{max_iterations} ──")

            resolved_this_iteration = 0
            total_pressure_before = sum(hz["pressure_score"] for hz in hot_zones)

            # Re-run swarm on hot zone docs
            hot_docs = set()
            for hz in hot_zones:
                hot_docs.update(hz.get("docs", []))

            hot_corpus = {doc_id: corpus[doc_id] for doc_id in hot_docs if doc_id in corpus}

            if not hot_corpus:
                print("[convergence] No hot docs in corpus — stopping")
                break

            # Expand lens set with recommendations
            expanded_lenses = self._expand_lens_set(hot_zones, lens_ids)

            print(f"[convergence] Re-running swarm on {len(hot_corpus)} hot doc(s) "
                  f"with {len(expanded_lenses)} lenses...")

            swarm_result = run_swarm(
                corpus=hot_corpus,
                lens_ids=expanded_lenses,
                memory_dir=self.memory_dir,
                min_confidence=0.25,
            )

            print(f"[convergence]   {swarm_result['n_claims_total']} new claims, "
                  f"{swarm_result['n_conflicts_detected']} conflicts")

            # Recompute claim scores
            self.claim_graph.compute_claim_scores()

            # Phase 14: Recompute orthogonal pressure
            self.claim_graph.compute_orthogonal_pressure(corpus=corpus)
            self.claim_graph.recompute_final_strength()

            # Re-cluster
            clusters_after = cluster_advanced(
                self.claim_graph,
                min_conflicts=2,
                min_confidence=0.25,
            )
            hot_zones_after = flag_hot_zones(
                clusters_after,
                pressure_threshold=pressure_threshold,
                min_conflicts=2,
                min_lenses=1,
                fragility_threshold=0.5,
            )

            total_pressure_after = sum(hz["pressure_score"] for hz in hot_zones_after)
            pressure_decay = (total_pressure_before - total_pressure_after) / max(total_pressure_before, 1)

            # Check resolution
            for hz in hot_zones:
                cluster_type = hz["cluster_type"]
                # Check if this cluster type still appears in hot_zones_after
                still_hot = any(
                    hz_after["cluster_type"] == cluster_type
                    for hz_after in hot_zones_after
                )
                if not still_hot:
                    resolved_this_iteration += 1
                    # Mark cluster as resolved in DB
                    try:
                        cluster_id = hz.get("cluster_id")
                        if cluster_id:
                            self.claim_graph._db.execute(
                                "UPDATE conflict_clusters SET resolved=1 WHERE id=?",
                                (cluster_id,)
                            )
                    except Exception:
                        pass

            self.claim_graph._db.commit()

            # Record convergence history
            self.claim_graph._db.execute("""
                INSERT INTO convergence_history
                    (iteration, n_claims_before, n_claims_after,
                     stable_edges, fragile_edges, pressure_decay, converged, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                iteration,
                initial_result["n_claims_total"] if iteration == 1 else swarm_result["n_claims_total"],
                swarm_result["n_claims_total"],
                len(self.claim_graph.get_stable_edges(min_strength=min_strength)),
                len(self.claim_graph.get_fragile_edges(min_support=2)),
                round(pressure_decay, 4),
                1 if not hot_zones_after else 0,
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ))
            self.claim_graph._db.commit()

            print(f"[convergence]   Pressure decay: {pressure_decay:.4f}")
            print(f"[convergence]   Resolved: {resolved_this_iteration} cluster(s)")
            print(f"[convergence]   Remaining hot zones: {len(hot_zones_after)}")

            report["iterations_ran"] = iteration
            report["resolved_clusters"] += resolved_this_iteration
            report["pressure_decay"] = pressure_decay
            report["convergence_history"].append({
                "iteration": iteration,
                "pressure_before": round(total_pressure_before, 4),
                "pressure_after": round(total_pressure_after, 4),
                "pressure_decay": round(pressure_decay, 4),
                "resolved": resolved_this_iteration,
                "hot_zones_remaining": len(hot_zones_after),
            })

            # Update hot_zones for next iteration
            hot_zones = hot_zones_after

            # Check convergence
            if not hot_zones:
                print("[convergence] ✓ CONVERGED — all clusters resolved!")
                report["converged"] = True
                break

            if pressure_decay < 0.01:
                print("[convergence] Pressure decay too low — stopping")
                break

        # Final metrics
        metrics_after = self.claim_graph.get_convergence_metrics()
        report["stable_edges"] = len(
            self.claim_graph.get_stable_edges(min_strength=min_strength)
        )
        report["fragile_edges"] = len(
            self.claim_graph.get_fragile_edges(min_support=2)
        )
        report["metrics_before"] = metrics_before
        report["metrics_after"] = metrics_after
        report["elapsed_sec"] = round(time.time() - start_time, 3)

        print(f"\n[convergence] Complete: {report['iterations_ran']} iteration(s), "
              f"{report['stable_edges']} stable edges, "
              f"{report['resolved_clusters']} cluster(s) resolved")

        return report

    def _expand_lens_set(self, hot_zones: List[Dict], base_lens_ids: List[str] = None) -> List[str]:
        """
        Expand lens set with recommendations from hot zones.
        """
        from scripts.lens_registry import LENS_FUNCTIONS

        if base_lens_ids is None:
            base_lens_ids = list(LENS_FUNCTIONS.keys())

        expanded = set(base_lens_ids)

        for hz in hot_zones:
            recommended = hz.get("recommended_lenses", [])
            for lns in recommended:
                if lns in LENS_FUNCTIONS:
                    expanded.add(lns)

        return list(expanded)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Bonfyre Structural Convergence Engine")
    ap.add_argument("--docs", required=True,
                    help="Glob pattern for docs (e.g. /tmp/*.txt)")
    ap.add_argument("--memory-dir", default="/tmp/bonfyre-memory",
                    help="Bonfyre memory directory")
    ap.add_argument("--max-iterations", type=int, default=5,
                    help="Max convergence iterations per cluster")
    ap.add_argument("--pressure-threshold", type=float, default=1.0,
                    help="Stop when cluster pressure < this")
    ap.add_argument("--min-strength", type=float, default=0.5,
                    help="Minimum claim_strength for stable edge")
    ap.add_argument("--lenses", help="Comma-separated lens IDs (default: all)")
    ap.add_argument("--out", help="Output JSON path")

    args = ap.parse_args()

    # Build corpus
    paths = glob.glob(args.docs)
    if not paths:
        print(f"[convergence] ERROR: no files matched {args.docs!r}", file=sys.stderr)
        sys.exit(1)

    corpus = {}
    for path in paths:
        doc_id = os.path.basename(path)
        with open(path, "r") as f:
            corpus[doc_id] = f.read()

    # Parse lenses
    lens_ids = None
    if args.lenses:
        if args.lenses != "all":
            lens_ids = [l.strip() for l in args.lenses.split(",") if l.strip()]

    # Run convergence
    engine = ConvergenceEngine(args.memory_dir)
    result = engine.run_convergence(
        corpus=corpus,
        max_iterations=args.max_iterations,
        pressure_threshold=args.pressure_threshold,
        min_strength=args.min_strength,
        lens_ids=lens_ids,
    )

    # Output
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[convergence] Result written to {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
