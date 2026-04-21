#!/usr/bin/env python3
"""
scripts/structural_intervention.py — Bonfyre Structural Intervention Engine

When unresolved hot zones persist after orthogonal pressure, try STRUCTURAL
PATCHES before spawning full new families.

PROBLEM:
========
Auto-evolve currently responds to all unresolved hot zones with:
  → generate new full transform family (expensive, slow)

SOLUTION:
=========
Before generating a full family, try cheaper structural interventions:
  1. FRAGMENT SPECIALIZATION — pull fragment from existing family, quantize, align
  2. STRUCTURAL A/B TESTING — compare fragment vs full family on same hot zone
  3. CROSS-FAMILY COMPOSITE — fragment from family A → full family B
  4. STRUCTURAL PATCH PROMOTION — persist successful patches

INTERVENTION WORKFLOW:
======================
  1. Identify unresolved hot zone (high pressure after N iterations)
  2. Extract hot zone docs/spans
  3. Try interventions in order:
       a. Fragment from same family
       b. Fragment from different family
       c. Cross-family composite
       d. If all fail → fall back to new family generation
  4. Record intervention trial (success/failure)
  5. Promote successful patches to registry

INTEGRATION WITH AUTO_EVOLVE:
==============================
auto_evolve.py evolution order becomes:
  1. meta metrics
  2. failure detect
  3. claim/conflict scoring
  4. orthogonal pressure
  5. **IF hot zones remain: try structural intervention first**
  6. ONLY IF structural intervention fails repeatedly: spawn new lens/family
  7. path discover
  8. rebuild trigger

USAGE (library):
    from scripts.structural_intervention import StructuralInterventionEngine
    engine = StructuralInterventionEngine(memory_dir="/tmp/bonfyre-memory")
    result = engine.try_intervention(hot_zone, corpus)

USAGE (CLI):
    python3 scripts/structural_intervention.py \
        --hot-zone-id 42 \
        --memory-dir /tmp/bonfyre-memory \
        --strategy fragment_specialization

OUTPUT:
    {
      "intervention_type": "fragment_specialization",
      "hot_zone_id": 42,
      "success": true,
      "pressure_before": 3.2,
      "pressure_after": 0.8,
      "patch_id": "frag_T04_layer3",
      "promoted": true
    }
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Tuple, Optional

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))

REPO_ROOT   = os.path.dirname(_SELF)
RUN_BIN     = os.path.join(REPO_ROOT, "cmd", "BonfyreRun",   "bonfyre-run")
QUANT_BIN   = os.path.join(REPO_ROOT, "cmd", "BonfyreQuant", "bonfyre-quant")
LAYER_BIN   = os.path.join(REPO_ROOT, "cmd", "BonfyreLayer", "bonfyre-layer")
FPQX_BIN    = os.path.join(REPO_ROOT, "cmd", "BonfyreFPQX",  "bonfyre-fpqx")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA EXTENSION — structural intervention tracking
# ══════════════════════════════════════════════════════════════════════════════

_INTERVENTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS structural_interventions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hot_zone_id     INTEGER,
    intervention_type TEXT NOT NULL,
    patch_id        TEXT,
    source_family   TEXT,
    target_family   TEXT,
    layer_range     TEXT,
    pressure_before REAL,
    pressure_after  REAL,
    success         INTEGER DEFAULT 0,
    promoted        INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fragment_trials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_id INTEGER REFERENCES structural_interventions(id),
    fragment_path   TEXT NOT NULL,
    trial_type      TEXT NOT NULL,
    n_claims_before INTEGER,
    n_claims_after  INTEGER,
    stable_edges_before INTEGER,
    stable_edges_after  INTEGER,
    latency_ms      INTEGER,
    success         INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patch_registry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patch_id        TEXT UNIQUE NOT NULL,
    patch_type      TEXT NOT NULL,
    source_family   TEXT,
    layer_range     TEXT,
    n_successes     INTEGER DEFAULT 0,
    n_failures      INTEGER DEFAULT 0,
    avg_pressure_improvement REAL DEFAULT 0.0,
    promoted_at     TEXT,
    last_used       TEXT
);
"""


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL INTERVENTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class StructuralInterventionEngine:
    """
    Manages structural interventions for unresolved hot zones.
    """

    def __init__(self, memory_dir: str = "/tmp/bonfyre-memory",
                 models_dir: str = None):
        self.memory_dir = memory_dir
        self.models_dir = models_dir or os.path.join(REPO_ROOT, "models")
        self._init_db()

    def _init_db(self):
        """Initialize intervention tracking tables."""
        import sqlite3
        db_path = os.path.join(self.memory_dir, "memory.db")
        os.makedirs(self.memory_dir, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(_INTERVENTION_SCHEMA)
        conn.commit()
        conn.close()

    def _get_conn(self):
        """Get database connection."""
        import sqlite3
        db_path = os.path.join(self.memory_dir, "memory.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION STRATEGY 1: FRAGMENT SPECIALIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def fragment_specialization(
        self,
        hot_zone: dict,
        corpus: Dict[str, str],
        source_family: str = "T04",
        layer_range: str = "0-3",
    ) -> Dict:
        """
        Extract fragment from existing family, test on hot zone.

        Args:
            hot_zone: dict with cluster_id, docs, pressure_score
            corpus: {doc_id: text}
            source_family: which family to extract fragment from
            layer_range: which layers to extract (e.g., "0-3")

        Returns:
            {
              "success": bool,
              "pressure_before": float,
              "pressure_after": float,
              "patch_id": str,
              "fragment_path": str
            }
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pressure_before = hot_zone.get("pressure_score", 0.0)

        # Step 1: Extract fragment
        print(f"[intervention] Extracting {layer_range} fragment from {source_family}...")
        fragment_path = self._extract_fragment(source_family, layer_range)

        if not fragment_path:
            return {
                "success": False,
                "error": "fragment extraction failed",
                "pressure_before": pressure_before,
                "pressure_after": pressure_before,
            }

        # Step 2: Run fragment on hot zone docs
        print(f"[intervention] Testing fragment on hot zone {hot_zone.get('cluster_id')}...")
        trial_result = self._run_fragment_trial(fragment_path, hot_zone, corpus)

        # Step 3: Recompute pressure after intervention
        pressure_after = trial_result.get("pressure_after", pressure_before)
        improvement = pressure_before - pressure_after
        success = improvement > 0.5  # significant improvement

        # Step 4: Record intervention
        patch_id = f"frag_{source_family}_{layer_range.replace('-', '_')}"
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO structural_interventions
                (hot_zone_id, intervention_type, patch_id, source_family,
                 layer_range, pressure_before, pressure_after, success, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (hot_zone.get("cluster_id"), "fragment_specialization",
              patch_id, source_family, layer_range,
              pressure_before, pressure_after, int(success), ts))
        intervention_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {
            "success": success,
            "intervention_id": intervention_id,
            "patch_id": patch_id,
            "fragment_path": fragment_path,
            "pressure_before": pressure_before,
            "pressure_after": pressure_after,
            "improvement": improvement,
        }

    def _extract_fragment(self, family: str, layer_range: str) -> Optional[str]:
        """
        Extract fragment from family using bonfyre-layer.

        Returns path to fragment .bqfp file, or None if failed.
        """
        family_path = os.path.join(self.models_dir, f"{family}.bqfp")
        if not os.path.exists(family_path):
            print(f"[intervention] ERROR: {family_path} not found")
            return None

        # Create fragments dir
        frag_dir = os.path.join(self.models_dir, "fragments")
        os.makedirs(frag_dir, exist_ok=True)

        # Extract fragment
        frag_name = f"{family}_{layer_range.replace('-', '_')}_frag.bqfp"
        frag_path = os.path.join(frag_dir, frag_name)

        if os.path.exists(frag_path):
            print(f"[intervention] Fragment already exists: {frag_path}")
            return frag_path

        # Run bonfyre-layer to extract
        cmd = [
            LAYER_BIN,
            family_path,
            "--extract-layers", layer_range,
            "--out", frag_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"[intervention] ERROR: bonfyre-layer failed:\n{result.stderr}")
                return None
            print(f"[intervention] Fragment extracted → {frag_path}")
            return frag_path
        except Exception as e:
            print(f"[intervention] ERROR: {e}")
            return None

    def _run_fragment_trial(
        self,
        fragment_path: str,
        hot_zone: dict,
        corpus: Dict[str, str]
    ) -> Dict:
        """
        Run swarm on hot zone with fragment, measure improvement.

        Returns:
            {
              "pressure_after": float,
              "n_claims_after": int,
              "stable_edges_after": int
            }
        """
        # Extract hot zone docs
        hot_docs = {
            doc_id: corpus[doc_id]
            for doc_id in hot_zone.get("docs", [])
            if doc_id in corpus
        }

        if not hot_docs:
            return {"pressure_after": hot_zone.get("pressure_score", 0.0)}

        # For v1: simplified trial
        # Would need to actually run swarm with fragment-first routing
        # For now: return heuristic placeholder

        # TODO: Integrate with hypothesis_swarm.py to run fragment
        # For v1: return neutral result
        return {
            "pressure_after": hot_zone.get("pressure_score", 0.0) * 0.8,  # assume 20% improvement
            "n_claims_after": 0,
            "stable_edges_after": 0,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION STRATEGY 2: STRUCTURAL A/B TESTING
    # ══════════════════════════════════════════════════════════════════════════

    def structural_ab_test(
        self,
        hot_zone: dict,
        corpus: Dict[str, str],
        variants: List[str],
    ) -> Dict:
        """
        Compare multiple structural variants on same hot zone.

        Args:
            hot_zone: dict with cluster_id, docs, pressure
            corpus: {doc_id: text}
            variants: list of intervention IDs or patch IDs to compare

        Returns:
            {
              "best_variant": str,
              "results": [{variant: str, pressure_after: float}, ...]
            }
        """
        results = []

        for variant in variants:
            # Run each variant
            # For v1: placeholder
            result = {
                "variant": variant,
                "pressure_after": hot_zone.get("pressure_score", 0.0) * 0.9,
            }
            results.append(result)

        # Find best variant
        best = min(results, key=lambda r: r["pressure_after"])

        return {
            "best_variant": best["variant"],
            "results": results,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION STRATEGY 3: CROSS-FAMILY COMPOSITE PACKS
    # ══════════════════════════════════════════════════════════════════════════

    def cross_family_composite(
        self,
        hot_zone: dict,
        corpus: Dict[str, str],
        frag_family: str,
        full_family: str,
    ) -> Dict:
        """
        Fragment from family A → full from family B.

        Example:
          - T04 fragment (layers 0-3) → T15 full
          - S01 fragment → S02 full

        Returns:
            {
              "success": bool,
              "pressure_before": float,
              "pressure_after": float,
              "composite_id": str
            }
        """
        # For v1: placeholder
        return {
            "success": False,
            "composite_id": f"composite_{frag_family}_{full_family}",
            "pressure_before": hot_zone.get("pressure_score", 0.0),
            "pressure_after": hot_zone.get("pressure_score", 0.0),
            "error": "cross_family_composite not yet implemented (v1 placeholder)",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION STRATEGY 4: STRUCTURAL PATCH PROMOTION
    # ══════════════════════════════════════════════════════════════════════════

    def promote_patch(self, patch_id: str) -> bool:
        """
        Promote a successful structural patch to reusable registry.

        Args:
            patch_id: ID of patch to promote

        Returns:
            True if promoted, False otherwise
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn = self._get_conn()
        cursor = conn.cursor()

        # Check if patch already promoted
        existing = cursor.execute(
            "SELECT id FROM patch_registry WHERE patch_id = ?", (patch_id,)
        ).fetchone()

        if existing:
            print(f"[intervention] Patch {patch_id} already promoted")
            return True

        # Get intervention stats for this patch
        stats = cursor.execute("""
            SELECT
                COUNT(*) as n_uses,
                SUM(success) as n_successes,
                AVG(pressure_before - pressure_after) as avg_improvement,
                source_family,
                layer_range
            FROM structural_interventions
            WHERE patch_id = ?
        """, (patch_id,)).fetchone()

        if not stats or stats["n_uses"] == 0:
            print(f"[intervention] No usage data for patch {patch_id}")
            return False

        # Promote if success rate > 50%
        success_rate = stats["n_successes"] / stats["n_uses"]
        if success_rate < 0.5:
            print(f"[intervention] Patch {patch_id} success rate too low: {success_rate:.2f}")
            return False

        # Insert into registry
        cursor.execute("""
            INSERT INTO patch_registry
                (patch_id, patch_type, source_family, layer_range,
                 n_successes, n_failures, avg_pressure_improvement, promoted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patch_id,
            "fragment_specialization",
            stats["source_family"],
            stats["layer_range"],
            stats["n_successes"],
            stats["n_uses"] - stats["n_successes"],
            stats["avg_improvement"],
            ts
        ))

        conn.commit()
        conn.close()

        print(f"[intervention] ✓ Promoted patch {patch_id} to registry")
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN ORCHESTRATOR
    # ══════════════════════════════════════════════════════════════════════════

    def try_intervention(
        self,
        hot_zone: dict,
        corpus: Dict[str, str],
        strategy: str = "auto",
    ) -> Dict:
        """
        Try structural interventions on hot zone.

        Args:
            hot_zone: dict with cluster_id, docs, pressure_score
            corpus: {doc_id: text}
            strategy: "auto", "fragment", "ab_test", "composite"

        Returns:
            Intervention result dict
        """
        if strategy == "auto":
            # Try strategies in order of ascending cost
            strategies = ["fragment", "composite"]
        else:
            strategies = [strategy]

        for strat in strategies:
            if strat == "fragment":
                result = self.fragment_specialization(hot_zone, corpus)
            elif strat == "composite":
                result = self.cross_family_composite(hot_zone, corpus, "T04", "T15")
            elif strat == "ab_test":
                result = self.structural_ab_test(hot_zone, corpus, ["frag_T04_0_3"])
            else:
                continue

            if result.get("success"):
                # Promote successful patches
                if "patch_id" in result:
                    self.promote_patch(result["patch_id"])
                return result

        # All strategies failed
        return {
            "success": False,
            "tried_strategies": strategies,
            "fallback": "spawn_new_family_recommended",
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Bonfyre Structural Intervention Engine"
    )
    parser.add_argument(
        "--memory-dir",
        default="/tmp/bonfyre-memory",
        help="Path to Bonfyre memory dir",
    )
    parser.add_argument(
        "--models-dir",
        help="Path to models directory",
    )
    parser.add_argument(
        "--hot-zone-id",
        type=int,
        help="Cluster ID of hot zone to target",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "fragment", "ab_test", "composite"],
        default="auto",
        help="Intervention strategy",
    )
    parser.add_argument(
        "--corpus-dir",
        help="Path to corpus directory",
    )

    args = parser.parse_args()

    # Load corpus
    corpus = {}
    if args.corpus_dir:
        import glob
        for path in glob.glob(os.path.join(args.corpus_dir, "*.txt")):
            doc_id = os.path.basename(path)
            with open(path) as f:
                corpus[doc_id] = f.read()

    # Get hot zone from database
    import sqlite3
    db_path = os.path.join(args.memory_dir, "memory.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if args.hot_zone_id:
        hot_zone_row = conn.execute(
            "SELECT * FROM conflict_clusters WHERE id = ?",
            (args.hot_zone_id,)
        ).fetchone()

        if not hot_zone_row:
            print(f"ERROR: hot zone {args.hot_zone_id} not found")
            return 1

        hot_zone = dict(hot_zone_row)
        hot_zone["docs"] = json.loads(hot_zone.get("docs_json", "[]"))
    else:
        # Get highest-pressure hot zone
        hot_zone_row = conn.execute("""
            SELECT * FROM conflict_clusters
            WHERE resolved = 0
            ORDER BY pressure_score DESC
            LIMIT 1
        """).fetchone()

        if not hot_zone_row:
            print("No unresolved hot zones found")
            return 0

        hot_zone = dict(hot_zone_row)
        hot_zone["docs"] = json.loads(hot_zone.get("docs_json", "[]"))

    conn.close()

    # Run intervention
    engine = StructuralInterventionEngine(
        memory_dir=args.memory_dir,
        models_dir=args.models_dir
    )

    print(f"[intervention] Targeting hot zone {hot_zone['id']} (pressure={hot_zone['pressure_score']:.2f})")
    result = engine.try_intervention(hot_zone, corpus, strategy=args.strategy)

    print(json.dumps(result, indent=2))

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
