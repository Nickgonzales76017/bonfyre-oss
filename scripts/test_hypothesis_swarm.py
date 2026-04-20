#!/usr/bin/env python3
"""
Smoke test for Hypothesis Swarm (Phase 12).

Tests:
  1. claim_graph.py — record claims, detect conflicts, cluster
  2. lens_registry.py — run all 10 lenses on sample doc
  3. hypothesis_swarm.py — swarm over 3 docs, produce conflict report
  4. conflict_cluster.py — advanced clustering + hot-zone flagging
  5. auto_evolve.py lens generation — verify lens minting from hot zones

Expected output:
  ✓ claim_graph: N claims, M conflicts, K clusters
  ✓ lens_registry: 10 lenses, all ran without errors
  ✓ hypothesis_swarm: 3 docs, pressure zones identified
  ✓ conflict_cluster: hot zones flagged
  ✓ auto_evolve: L11+ lens generated from hot zone
"""

import json
import os
import sys
import tempfile

_SELF = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SELF)
sys.path.insert(0, REPO_ROOT)

from scripts.claim_graph import ClaimGraph
from scripts.lens_registry import run_all_lenses
from scripts.hypothesis_swarm import run_swarm
from scripts.conflict_cluster import cluster_advanced, flag_hot_zones


# ── Sample corpus ───────────────────────────────────────────────────────

SAMPLE_DOCS = {
    "deposition_001.txt": """
Q: Where were you on March 14, 2024?
A: I was in New York.
Q: And what about March 17?
A: I was in London.
BY MR. SMITH: Can you confirm you arrived on March 17?
A: Actually, I arrived on March 14.
Q: That contradicts your earlier statement. Were you in New York or London on the 14th?
A: I misspoke. I was in New York on March 14 and flew to London on March 17.
""",

    "email_thread_002.txt": """
Subject: Meeting rescheduled
From: john.doe@example.com
Date: 2024-03-10

Hi team, the meeting is moved to Friday.

---
Re: Meeting rescheduled
From: jane.smith@example.com
Date: 2024-03-08

Wait, I thought it was Thursday?

---
Fwd: Meeting rescheduled
From: admin@example.com
Date: 2024-03-12

Confirming Friday at 3pm.
""",

    "redacted_doc_003.txt": """
The defendant, John Smith, also known as J. Smith, traveled from New York to 
[REDACTED] on ██/██/2024. Witness reports state he was seen with ████████ at
the location. The massage therapist allegedly provided entertainment services
to minors during his visit to the island resort.
""",
}


# ── Test 1: claim_graph.py ──────────────────────────────────────────────

def test_claim_graph():
    print("\n[TEST 1] claim_graph.py")
    with tempfile.TemporaryDirectory() as tmp:
        cg = ClaimGraph(tmp)
        
        # Record sample claims
        claims = [
            {"doc_id": "doc1", "subject": "John", "predicate": "arrived_on",
             "object": "2024-03-14", "lens": "L04_timeline_anomaly",
             "confidence": 0.8, "assumptions": ["date_extraction"]},
            {"doc_id": "doc1", "subject": "John", "predicate": "arrived_on",
             "object": "2024-03-17", "lens": "L01_deposition_parser",
             "confidence": 0.7, "assumptions": ["Q/A_parsing"]},
            {"doc_id": "doc1", "subject": "John Smith", "predicate": "alias_of",
             "object": "J. Smith", "lens": "L02_alias_expansion",
             "confidence": 0.9, "assumptions": ["name_similarity"]},
        ]
        
        for c in claims:
            cg.record_claim(c)
        
        # Detect conflicts
        conflicts = cg.detect_conflicts(min_confidence=0.5)
        print(f"  ✓ {len(claims)} claims recorded")
        print(f"  ✓ {len(conflicts)} conflict(s) detected")
        
        # Cluster conflicts
        cg.persist_conflicts(conflicts)
        for i, c in enumerate(conflicts):
            conflicts[i]["_persisted_id"] = i + 1
        clusters = cg.cluster_conflicts(conflicts)
        print(f"  ✓ {len(clusters)} cluster(s) created")
        
        # Pressure zones
        zones = cg.get_pressure_zones(top_n=5)
        print(f"  ✓ {len(zones)} pressure zone(s)")
        
        assert len(conflicts) >= 1, "Expected at least 1 conflict from timeline discrepancy"
        print("  [PASS] claim_graph.py")


# ── Test 2: lens_registry.py ────────────────────────────────────────────

def test_lens_registry():
    print("\n[TEST 2] lens_registry.py")
    doc = SAMPLE_DOCS["deposition_001.txt"]
    results = run_all_lenses(doc, "deposition_001.txt")
    
    print(f"  ✓ {len(results)} lenses ran")
    
    total_claims = sum(len(r["claims"]) for r in results)
    print(f"  ✓ {total_claims} total claims produced")
    
    # Check specific lenses fired
    lens_ids = {r["lens_id"] for r in results}
    assert "L01_deposition_parser" in lens_ids
    assert "L04_timeline_anomaly" in lens_ids
    
    # Deposition parser should find Q/A structure
    depo_result = next(r for r in results if r["lens_id"] == "L01_deposition_parser")
    assert len(depo_result["claims"]) >= 3, "Expected Q/A claims from deposition"
    
    print("  [PASS] lens_registry.py")


# ── Test 3: hypothesis_swarm.py ─────────────────────────────────────────

def test_hypothesis_swarm():
    print("\n[TEST 3] hypothesis_swarm.py")
    with tempfile.TemporaryDirectory() as tmp:
        result = run_swarm(
            corpus=SAMPLE_DOCS,
            lens_ids=None,  # all lenses
            memory_dir=tmp,
            min_confidence=0.25,
            output_dir=tmp,
        )
        
        print(f"  ✓ {result['n_lenses_ran']} lenses ran")
        print(f"  ✓ {result['n_docs_processed']} docs processed")
        print(f"  ✓ {result['n_claims_total']} claims total")
        print(f"  ✓ {result['n_conflicts_detected']} conflicts detected")
        print(f"  ✓ {result['n_clusters']} clusters")
        
        assert result["n_claims_total"] > 0, "Expected claims from swarm"
        assert result["n_conflicts_detected"] > 0, "Expected conflicts (timeline anomaly)"
        
        # Check output files exist
        assert os.path.exists(result["conflict_report_path"])
        assert os.path.exists(result["cluster_report_path"])
        assert os.path.exists(result["pressure_report_path"])
        
        print("  [PASS] hypothesis_swarm.py")


# ── Test 4: conflict_cluster.py ─────────────────────────────────────────

def test_conflict_cluster():
    print("\n[TEST 4] conflict_cluster.py")
    with tempfile.TemporaryDirectory() as tmp:
        # Run swarm first to populate claim graph
        run_swarm(
            corpus=SAMPLE_DOCS,
            lens_ids=None,
            memory_dir=tmp,
            min_confidence=0.25,
        )
        
        # Advanced clustering
        cg = ClaimGraph(tmp)
        clusters = cluster_advanced(cg, min_conflicts=1, min_confidence=0.25)
        print(f"  ✓ {len(clusters)} advanced cluster(s)")
        
        # Flag hot zones
        hot_zones = flag_hot_zones(
            clusters, pressure_threshold=0.5, min_conflicts=1,
            min_lenses=1, fragility_threshold=0.3)
        
        print(f"  ✓ {len(hot_zones)} hot zone(s) flagged")
        
        if hot_zones:
            hz = hot_zones[0]
            print(f"    → {hz['cluster_type']}  pressure={hz['pressure_score']:.3f}  "
                  f"fragility={hz['assumption_fragility']:.3f}")
            assert "recommended_lenses" in hz
        
        print("  [PASS] conflict_cluster.py")


# ── Test 5: auto_evolve.py lens generation ──────────────────────────────

def test_auto_evolve_lens_generation():
    print("\n[TEST 5] auto_evolve.py lens generation")
    with tempfile.TemporaryDirectory() as tmp:
        # Populate claim graph with hot zones
        run_swarm(
            corpus=SAMPLE_DOCS,
            lens_ids=None,
            memory_dir=tmp,
            min_confidence=0.25,
        )
        
        # Manually trigger lens generation
        from scripts.auto_evolve import _generate_lenses_from_hot_zones
        cg = ClaimGraph(tmp)
        clusters = cluster_advanced(cg, min_conflicts=1, min_confidence=0.25)
        hot_zones = flag_hot_zones(
            clusters, pressure_threshold=0.5, min_conflicts=1,
            min_lenses=1, fragility_threshold=0.3)
        
        # Generate lenses (dry-run)
        new_lenses = _generate_lenses_from_hot_zones(hot_zones, tmp, dry_run=True)
        
        print(f"  ✓ {len(new_lenses)} new lens(es) would be generated")
        
        if new_lenses:
            lns = new_lenses[0]
            print(f"    → {lns['lens_id']}  for {lns['cluster_type']}")
            assert lns["lens_id"].startswith("L11") or lns["lens_id"].startswith("L12")
        
        print("  [PASS] auto_evolve.py lens generation")


# ── Run all tests ───────────────────────────────────────────────────────

def main():
    print("="*60)
    print("HYPOTHESIS SWARM SMOKE TEST (Phase 12)")
    print("="*60)
    
    try:
        test_claim_graph()
        test_lens_registry()
        test_hypothesis_swarm()
        test_conflict_cluster()
        test_auto_evolve_lens_generation()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
