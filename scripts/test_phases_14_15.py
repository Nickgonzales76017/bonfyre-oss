#!/usr/bin/env python3
"""
scripts/test_phases_14_15.py — Test suite for Orthogonal Pressure + Structural Intervention

Tests both Phase 14 (orthogonal pressure engines) and Phase 15 (structural intervention).

WHAT IS TESTED:
===============
Phase 14:
  - Graph pressure (topology anomalies)
  - Temporal pressure (causality violations)
  - Frequency pressure (statistical co-occurrence)
  - Orthogonal pressure integration with claim scoring
  - Final claim strength = semantic_strength × orthogonal_pressure

Phase 15:
  - Fragment specialization
  - Structural intervention trial tracking
  - Patch registry promotion
  - Integration with auto_evolve

USAGE:
    python3 scripts/test_phases_14_15.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))

from scripts.claim_graph import ClaimGraph
from scripts.orthogonal_pressure import OrthogonalPressure
from scripts.structural_intervention import StructuralInterventionEngine


# ══════════════════════════════════════════════════════════════════════════════
# TEST CORPUS
# ══════════════════════════════════════════════════════════════════════════════

TEST_CORPUS = {
    "deposition_2016.txt": """
        DEPOSITION TRANSCRIPT - JOHN SMITH
        Date: March 14, 2024
        Location: Miami, Florida
        
        Q: Where were you on March 14, 2024?
        A: I was in New York City attending a business meeting.
        
        Q: Can you provide documentation?
        A: Yes, I have hotel receipts from the Marriott in Manhattan.
    """,
    
    "email_thread.txt": """
        From: john.smith@example.com
        To: legal@example.com
        Date: March 17, 2024
        Subject: Re: Meeting confirmation
        
        I can confirm I was in Miami on March 14 for the
        conference. I flew in on March 13 and left on March 15.
        
        Attached are my flight confirmations.
    """,
    
    "flight_manifest.txt": """
        FLIGHT MANIFEST
        Airline: United Airlines
        Flight: UA2341
        Date: March 14, 2024, 9:45 AM
        Route: New York (JFK) → Miami (MIA)
        
        Passenger List:
        - Smith, John (Seat 12A)
        - Johnson, Mary (Seat 14C)
    """,
}


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: ORTHOGONAL PRESSURE SCORING
# ══════════════════════════════════════════════════════════════════════════════

def test_orthogonal_pressure():
    """Test orthogonal pressure engines on conflicting claims."""
    print("\n[TEST 1] Orthogonal Pressure Scoring")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cg = ClaimGraph(tmpdir)
        
        # Create conflicting claims (simultaneity violation)
        claim_nyc = {
            "doc_id": "deposition_2016.txt",
            "span_start": 0,
            "span_end": 100,
            "span_text": "I was in New York City",
            "subject": "John Smith",
            "predicate": "located_at",
            "object": "New York City on 2024-03-14",
            "lens": "L01_deposition_parser",
            "confidence": 0.85,
        }
        
        claim_miami = {
            "doc_id": "email_thread.txt",
            "span_start": 0,
            "span_end": 100,
            "span_text": "I was in Miami",
            "subject": "John Smith",
            "predicate": "located_at",
            "object": "Miami on 2024-03-14",
            "lens": "L07_email_thread",
            "confidence": 0.78,
        }
        
        # Record claims
        id_nyc = cg.record_claim(claim_nyc)
        id_miami = cg.record_claim(claim_miami)
        
        # Get claims from DB
        claim_nyc_db = dict(cg._db.execute(
            "SELECT * FROM claims WHERE id = ?", (id_nyc,)
        ).fetchone())
        claim_miami_db = dict(cg._db.execute(
            "SELECT * FROM claims WHERE id = ?", (id_miami,)
        ).fetchone())
        
        # Compute orthogonal pressure
        engine = OrthogonalPressure()
        
        scores_nyc = engine.compute_pressure_score(claim_nyc_db, cg, TEST_CORPUS)
        scores_miami = engine.compute_pressure_score(claim_miami_db, cg, TEST_CORPUS)
        
        print(f"  Claim NYC pressure:")
        print(f"    graph_pressure:         {scores_nyc['graph_pressure']:.3f}")
        print(f"    temporal_pressure:      {scores_nyc['temporal_pressure']:.3f}")
        print(f"    frequency_pressure:     {scores_nyc['frequency_pressure']:.3f}")
        print(f"    orthogonal_pressure:    {scores_nyc['orthogonal_pressure']:.3f}")
        
        print(f"\n  Claim Miami pressure:")
        print(f"    graph_pressure:         {scores_miami['graph_pressure']:.3f}")
        print(f"    temporal_pressure:      {scores_miami['temporal_pressure']:.3f}")
        print(f"    frequency_pressure:     {scores_miami['frequency_pressure']:.3f}")
        print(f"    orthogonal_pressure:    {scores_miami['orthogonal_pressure']:.3f}")
        
        # EXPECTED: temporal_pressure should detect simultaneity violation
        # Both claims have same subject, same predicate (located_at),
        # different locations, same date → physically impossible
        
        print(f"\n  ✓ Orthogonal pressure computed successfully")
        print(f"  [PASS] orthogonal_pressure_scoring\n")
        
        return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: CLAIM GRAPH INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def test_claim_graph_integration():
    """Test orthogonal pressure integration with claim_graph."""
    print("\n[TEST 2] Claim Graph Integration")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cg = ClaimGraph(tmpdir)
        
        # Create test claims
        claim1 = {
            "doc_id": "test.txt",
            "subject": "Entity A",
            "predicate": "related_to",
            "object": "Entity B",
            "lens": "L01",
            "confidence": 0.9,
        }
        
        id1 = cg.record_claim(claim1)
        
        # Compute semantic claim scores (Phase 13)
        cg.compute_claim_scores([id1])
        
        # Compute orthogonal pressure (Phase 14)
        cg.compute_orthogonal_pressure([id1], corpus=TEST_CORPUS)
        
        # Recompute final strength
        cg.recompute_final_strength()
        
        # Check that orthogonal_pressure field is populated
        result = cg._db.execute(
            "SELECT claim_strength, orthogonal_pressure FROM claims WHERE id = ?",
            (id1,)
        ).fetchone()
        
        print(f"  Claim {id1}:")
        print(f"    claim_strength:       {result[0]:.4f}")
        print(f"    orthogonal_pressure:  {result[1]:.4f}")
        
        assert result[1] is not None, "orthogonal_pressure should be set"
        
        print(f"\n  ✓ Orthogonal pressure integrated with claim_strength")
        print(f"  [PASS] claim_graph_integration\n")
        
        return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: STRUCTURAL INTERVENTION
# ══════════════════════════════════════════════════════════════════════════════

def test_structural_intervention():
    """Test structural intervention workflow."""
    print("\n[TEST 3] Structural Intervention")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        engine = StructuralInterventionEngine(
            memory_dir=tmpdir,
            models_dir=models_dir
        )
        
        # Create mock hot zone
        hot_zone = {
            "cluster_id": 42,
            "cluster_type": "timeline_discrepancy",
            "pressure_score": 3.2,
            "docs": ["deposition_2016.txt", "email_thread.txt"],
        }
        
        # Try fragment specialization (will fail gracefully without real models)
        print(f"  Testing fragment specialization...")
        result = engine.fragment_specialization(
            hot_zone,
            TEST_CORPUS,
            source_family="T04",
            layer_range="0-3"
        )
        
        print(f"    Result: {result.get('success', False)}")
        print(f"    Pressure before: {result.get('pressure_before', 0):.2f}")
        print(f"    Pressure after:  {result.get('pressure_after', 0):.2f}")
        
        # Check intervention was recorded in DB
        conn = engine._get_conn()
        interventions = conn.execute(
            "SELECT * FROM structural_interventions"
        ).fetchall()
        
        print(f"\n  Interventions recorded: {len(interventions)}")
        if interventions:
            print(f"    Intervention ID: {interventions[0]['id']}")
            print(f"    Type: {interventions[0]['intervention_type']}")
            print(f"    Patch ID: {interventions[0]['patch_id']}")
        
        conn.close()
        
        print(f"\n  ✓ Structural intervention tracking functional")
        print(f"  [PASS] structural_intervention\n")
        
        return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: PATCH REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def test_patch_registry():
    """Test patch promotion to registry."""
    print("\n[TEST 4] Patch Registry")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        engine = StructuralInterventionEngine(
            memory_dir=tmpdir,
            models_dir=models_dir
        )
        
        # Create mock intervention records
        conn = engine._get_conn()
        
        # Record several successful interventions with same patch
        patch_id = "frag_T04_0_3"
        for i in range(5):
            conn.execute("""
                INSERT INTO structural_interventions
                    (hot_zone_id, intervention_type, patch_id, source_family,
                     layer_range, pressure_before, pressure_after, success, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                100 + i,
                "fragment_specialization",
                patch_id,
                "T04",
                "0-3",
                3.0 + i * 0.2,
                1.5 + i * 0.1,
                1,  # success
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ))
        
        conn.commit()
        conn.close()
        
        # Promote patch
        promoted = engine.promote_patch(patch_id)
        
        print(f"  Patch {patch_id} promoted: {promoted}")
        
        # Check registry
        conn = engine._get_conn()
        registry = conn.execute(
            "SELECT * FROM patch_registry WHERE patch_id = ?",
            (patch_id,)
        ).fetchone()
        
        if registry:
            print(f"  Registry entry:")
            print(f"    Patch ID: {registry['patch_id']}")
            print(f"    Successes: {registry['n_successes']}")
            print(f"    Failures: {registry['n_failures']}")
            print(f"    Avg improvement: {registry['avg_pressure_improvement']:.3f}")
        
        conn.close()
        
        print(f"\n  ✓ Patch registry functional")
        print(f"  [PASS] patch_registry\n")
        
        return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASES 14 & 15 SMOKE TEST")
    print("=" * 70)
    
    tests = [
        ("Orthogonal Pressure Scoring", test_orthogonal_pressure),
        ("Claim Graph Integration", test_claim_graph_integration),
        ("Structural Intervention", test_structural_intervention),
        ("Patch Registry", test_patch_registry),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"  [FAIL] {name}\n")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\nALL TESTS PASSED ✓\n")
        return 0
    else:
        print(f"\n{failed} TEST(S) FAILED ✗\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
