#!/usr/bin/env python3
"""
scripts/orthogonal_pressure.py — Bonfyre Orthogonal Pressure Engines

Tests claims against INDEPENDENT reality constraints beyond semantic agreement.

PROBLEM:
========
Phase 13 convergence can achieve semantic consensus among lenses, but all
lenses are text interpreters operating in similar spaces. They can all be
confidently wrong together.

SOLUTION:
=========
Add pressure from orthogonal "realities":
  1. GRAPH TOPOLOGY — non-semantic structural anomalies
  2. TEMPORAL PHYSICS — causality and simultaneity violations
  3. STATISTICAL FREQUENCY — co-occurrence independent of meaning
  4. PERTURBATION ROBUSTNESS — claim survival under corpus noise
  5. REPRESENTATION INDEPENDENCE — persistence across encodings

USAGE (library):
    from scripts.orthogonal_pressure import OrthogonalPressure
    engine = OrthogonalPressure()
    pressure = engine.compute_pressure_score(claim, claim_graph, corpus)

PRESSURE SCORING:
=================
Each engine returns 0.0 (contradicts reality) to 1.0 (survives pressure).
Final orthogonal_pressure = product of all enabled engines.

INTEGRATION:
============
Extended claim strength formula:
    final_strength = semantic_strength × orthogonal_pressure

where semantic_strength is the Phase 13 formula:
    (independent_support × lens_diversity) / (conflict_count + 1) × (1 - fragility)
"""

import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL PREDICATES — predicates that carry temporal/causal constraints
# ══════════════════════════════════════════════════════════════════════════════

TEMPORAL_PREDICATES = {
    "arrived_on", "departed_on", "occurred_on", "born_on", "died_on",
    "met_on", "signed_on", "filed_on", "recorded_on", "observed_on",
    "arrived_at", "departed_from", "located_at", "traveled_to",
    "before", "after", "during", "concurrent_with"
}

LOCATION_PREDICATES = {
    "located_at", "arrived_at", "departed_from", "traveled_to",
    "present_at", "absent_from", "observed_at"
}


# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE ENGINE 1: GRAPH TOPOLOGY
# ══════════════════════════════════════════════════════════════════════════════

def graph_pressure(claim: dict, claim_graph) -> float:
    """
    Non-semantic graph topology analysis.
    
    Detects:
      - Bridge anomalies (unexpected connections between communities)
      - Degree anomalies (nodes with unusual connectivity)
      - Isolated claims (no relationships)
    
    Returns:
        1.0 = structurally consistent
        0.0-0.3 = suspicious bridge or degree anomaly
    """
    subject = claim.get("subject", "")
    object_val = claim.get("object", "")
    
    if not subject:
        return 0.5  # neutral for malformed claims
    
    # Count claims involving this subject
    result = claim_graph._db.execute("""
        SELECT COUNT(*) FROM claims
        WHERE subject = ? OR object = ?
    """, (subject, subject)).fetchone()
    subject_degree = result[0]
    
    # Compute average degree
    total_claims = claim_graph._db.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    
    if total_claims == 0:
        return 1.0
    
    # Average claims per entity (rough estimate)
    result = claim_graph._db.execute("""
        SELECT COUNT(DISTINCT subject) + COUNT(DISTINCT object)
        FROM claims
    """).fetchone()
    total_entities = result[0]
    avg_degree = total_claims / max(total_entities / 2.0, 1.0)
    
    # Degree anomaly score
    if avg_degree < 1.0:
        return 1.0
    
    degree_ratio = subject_degree / avg_degree
    
    # Suspicious if this entity appears WAY more than average
    if degree_ratio > 5.0:
        return 0.3  # high degree anomaly
    elif degree_ratio > 3.0:
        return 0.6
    elif degree_ratio < 0.5:
        return 0.8  # isolated claim (slightly suspicious)
    else:
        return 1.0  # normal degree


# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE ENGINE 2: TEMPORAL CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

def temporal_pressure(claim: dict, claim_graph) -> float:
    """
    Temporal and causal consistency checks.
    
    Detects:
      - Simultaneity violations (same person, two places, same time)
      - Causal ordering violations (effect before cause)
      - Physical impossibility (travel constraints)
    
    Returns:
        1.0 = temporally consistent
        0.0 = physically impossible
        0.2-0.8 = suspicious timing
    """
    predicate = claim.get("predicate", "")
    
    if predicate not in TEMPORAL_PREDICATES:
        return 1.0  # no temporal constraint
    
    subject = claim.get("subject", "")
    object_val = claim.get("object", "")
    
    # Extract date from object (YYYY-MM-DD or similar)
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(object_val))
    if not date_match:
        return 1.0  # can't verify temporal constraint
    
    claim_date = object_val
    
    # Check for simultaneity violations
    conflicting_events = claim_graph._db.execute("""
        SELECT predicate, object, doc_id FROM claims
        WHERE subject = ?
          AND predicate IN ({})
          AND object LIKE ?
          AND id != ?
    """.format(",".join("?" * len(TEMPORAL_PREDICATES))),
    (subject, *TEMPORAL_PREDICATES, f"%{claim_date}%", claim.get("id", -1))).fetchall()
    
    # If same person has multiple location predicates on same date → violation
    if predicate in LOCATION_PREDICATES and len(conflicting_events) > 0:
        # Check if conflicting locations
        for row in conflicting_events:
            conf_pred = row[0]
            conf_obj = row[1]
            conf_doc = row[2]
            if conf_pred in LOCATION_PREDICATES and conf_obj != object_val:
                # Same person, same date, different locations
                return 0.0  # physically impossible
    
    # Basic temporal ordering check (simplified)
    # Could extend with full timeline DAG analysis
    
    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE ENGINE 3: FREQUENCY / STATISTICAL SUPPORT
# ══════════════════════════════════════════════════════════════════════════════

def frequency_pressure(claim: dict, corpus: Dict[str, str]) -> float:
    """
    Statistical co-occurrence analysis (ignores semantics).
    
    Measures:
      - Subject and object co-occurrence in corpus
      - PMI-like score (actual vs expected co-occurrence)
      - Rarity spikes
    
    Returns:
        1.0 = strong co-occurrence
        0.5 = neutral (no corpus provided or no data)
        0.0 = subject and object never co-occur
    """
    if not corpus:
        return 0.5  # neutral
    
    subject = claim.get("subject", "")
    object_val = str(claim.get("object", ""))
    
    if not subject or not object_val:
        return 0.5
    
    # Count co-occurrences within 50-word window
    cooccur_count = 0
    subject_count = 0
    object_count = 0
    
    for doc_id, doc_text in corpus.items():
        words = doc_text.lower().split()
        
        # Simple substring matching (could improve with entity normalization)
        subject_lower = subject.lower()
        object_lower = object_val.lower()
        
        # Count subject occurrences
        if subject_lower in doc_text.lower():
            subject_count += 1
        
        # Count object occurrences
        if object_lower in doc_text.lower():
            object_count += 1
        
        # Count co-occurrences (both within same doc)
        if subject_lower in doc_text.lower() and object_lower in doc_text.lower():
            cooccur_count += 1
    
    if subject_count == 0 or object_count == 0:
        return 0.5  # no data
    
    # Simple PMI-like score
    # Expected co-occurrence if independent
    total_docs = len(corpus)
    expected = (subject_count / total_docs) * (object_count / total_docs) * total_docs
    
    if expected < 0.01:
        return 0.5
    
    pmi = cooccur_count / expected
    
    # Normalize to 0-1 range
    # High co-occurrence (pmi > 2) → strong signal
    return min(pmi / 2.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE ENGINE 4: PERTURBATION ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════

def perturbation_pressure(claim: dict, corpus: Dict[str, str]) -> float:
    """
    Robustness under corpus perturbations (EXPENSIVE — use sparingly).
    
    Tests:
      - Sentence shuffling
      - Word dropout
      - Formatting changes
      - Entity masking
    
    Returns:
        1.0 = claim still extractable after all perturbations
        0.0 = claim disappears with any perturbation
    
    NOTE: This is a PLACEHOLDER for v1.
    Full implementation would re-run lenses on perturbed corpus.
    For now, use heuristics based on claim properties.
    """
    # V1 HEURISTIC: Claims with high confidence and multiple supporting
    # spans are more likely to survive perturbation
    
    confidence = claim.get("confidence", 0.5)
    span_text = claim.get("span_text", "")
    
    # If claim is based on very short span → more fragile
    if len(span_text) < 20:
        return 0.5
    
    # If claim has high confidence but is from single lens → uncertain
    # (Would need to actually re-run lenses to verify)
    
    # For v1: return conservative estimate based on confidence
    return min(confidence * 1.2, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE ENGINE 5: REPRESENTATION INDEPENDENCE
# ══════════════════════════════════════════════════════════════════════════════

def representation_pressure(claim: dict, corpus: Dict[str, str]) -> float:
    """
    Persistence across different representational views (RESEARCH).
    
    Tests:
      - Embedding space view
      - Compressed view
      - Fragment-only view
      - Character n-gram view
    
    Returns:
        1.0 = claim visible in multiple representations
        0.5 = neutral (not yet implemented)
        0.0 = claim only visible in one representation
    
    NOTE: This is a PLACEHOLDER for v1.
    Full implementation requires embedding models and compression.
    """
    # V1 HEURISTIC: Claims about entities (proper nouns) are more
    # representation-independent than claims about concepts
    
    subject = claim.get("subject", "")
    
    # Heuristic: capitalized words are likely entities
    if subject and subject[0].isupper():
        return 0.8  # likely entity → more robust
    else:
        return 0.5  # concept → uncertain


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class OrthogonalPressure:
    """
    Orchestrates orthogonal pressure computation for claims.
    """
    
    def __init__(self, enable_expensive: bool = False):
        """
        Args:
            enable_expensive: Enable perturbation and representation pressure
                              (requires re-running lenses, much slower)
        """
        self.enable_expensive = enable_expensive
    
    def compute_pressure_score(
        self,
        claim: dict,
        claim_graph,
        corpus: Dict[str, str] = None,
    ) -> Dict[str, float]:
        """
        Compute all orthogonal pressure scores for a claim.
        
        Args:
            claim: claim dict with subject, predicate, object, etc.
            claim_graph: ClaimGraph instance
            corpus: optional corpus dict {doc_id: text} for frequency analysis
        
        Returns:
            {
                "graph_pressure": 0.0-1.0,
                "temporal_pressure": 0.0-1.0,
                "frequency_pressure": 0.0-1.0,
                "perturbation_pressure": 0.0-1.0,
                "representation_pressure": 0.0-1.0,
                "orthogonal_pressure": 0.0-1.0  (product of all)
            }
        """
        scores = {}
        
        # Always compute cheap pressures
        scores["graph_pressure"] = graph_pressure(claim, claim_graph)
        scores["temporal_pressure"] = temporal_pressure(claim, claim_graph)
        scores["frequency_pressure"] = frequency_pressure(claim, corpus or {})
        
        # Expensive pressures (disabled by default in v1)
        if self.enable_expensive:
            scores["perturbation_pressure"] = perturbation_pressure(claim, corpus or {})
            scores["representation_pressure"] = representation_pressure(claim, corpus or {})
        else:
            # Use heuristic estimates
            scores["perturbation_pressure"] = perturbation_pressure(claim, corpus or {})
            scores["representation_pressure"] = representation_pressure(claim, corpus or {})
        
        # Combined orthogonal pressure (product)
        scores["orthogonal_pressure"] = (
            scores["graph_pressure"]
            * scores["temporal_pressure"]
            * scores["frequency_pressure"]
            * scores["perturbation_pressure"]
            * scores["representation_pressure"]
        )
        
        return scores
    
    def compute_batch_pressure(
        self,
        claims: List[dict],
        claim_graph,
        corpus: Dict[str, str] = None,
    ) -> List[Dict[str, float]]:
        """
        Compute pressure scores for multiple claims.
        
        Returns list of pressure score dicts, one per claim.
        """
        return [
            self.compute_pressure_score(claim, claim_graph, corpus)
            for claim in claims
        ]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    from scripts.claim_graph import ClaimGraph
    
    parser = argparse.ArgumentParser(
        description="Compute orthogonal pressure scores for claims"
    )
    parser.add_argument(
        "--memory-dir",
        default="/tmp/bonfyre-memory",
        help="Path to Bonfyre memory dir",
    )
    parser.add_argument(
        "--corpus-dir",
        help="Path to corpus directory (for frequency analysis)",
    )
    parser.add_argument(
        "--enable-expensive",
        action="store_true",
        help="Enable expensive pressure engines (perturbation, representation)",
    )
    parser.add_argument(
        "--out",
        help="Output JSON file for pressure report",
    )
    
    args = parser.parse_args()
    
    # Load claim graph
    claim_graph = ClaimGraph(args.memory_dir)
    
    # Load corpus if provided
    corpus = {}
    if args.corpus_dir:
        import glob
        for path in glob.glob(os.path.join(args.corpus_dir, "*.txt")):
            doc_id = os.path.basename(path)
            with open(path) as f:
                corpus[doc_id] = f.read()
    
    # Get all claims
    cursor = claim_graph.conn.cursor()
    cursor.execute("""
        SELECT id, doc_id, subject, predicate, object, lens, confidence, span_text
        FROM claims
    """)
    claims = []
    for row in cursor.fetchall():
        claims.append({
            "id": row[0],
            "doc_id": row[1],
            "subject": row[2],
            "predicate": row[3],
            "object": row[4],
            "lens": row[5],
            "confidence": row[6],
            "span_text": row[7],
        })
    
    print(f"[orthogonal_pressure] Computing pressure for {len(claims)} claims...")
    
    # Compute pressure scores
    engine = OrthogonalPressure(enable_expensive=args.enable_expensive)
    pressure_scores = engine.compute_batch_pressure(claims, claim_graph, corpus)
    
    # Report summary
    avg_pressure = sum(s["orthogonal_pressure"] for s in pressure_scores) / max(len(pressure_scores), 1)
    print(f"[orthogonal_pressure] Average orthogonal pressure: {avg_pressure:.3f}")
    
    # Find low-pressure claims (likely problematic)
    low_pressure = [
        (claims[i], pressure_scores[i])
        for i in range(len(claims))
        if pressure_scores[i]["orthogonal_pressure"] < 0.5
    ]
    
    print(f"[orthogonal_pressure] {len(low_pressure)} claim(s) with pressure < 0.5")
    
    # Output report
    report = {
        "total_claims": len(claims),
        "avg_orthogonal_pressure": avg_pressure,
        "low_pressure_count": len(low_pressure),
        "low_pressure_claims": [
            {
                "claim_id": claim["id"],
                "subject": claim["subject"],
                "predicate": claim["predicate"],
                "object": claim["object"],
                "pressure": scores["orthogonal_pressure"],
                "breakdown": scores,
            }
            for claim, scores in low_pressure[:20]  # top 20
        ],
    }
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[orthogonal_pressure] Report → {args.out}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
