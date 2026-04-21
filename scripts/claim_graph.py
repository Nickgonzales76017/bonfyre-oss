#!/usr/bin/env python3
"""
scripts/claim_graph.py — Bonfyre persistent claim graph layer.

Extends memory.db (BonfyreMemory) with five new tables:
  claims            — structured assertions produced by lens families
  conflicts         — claim-pair disagreements with type and strength
  conflict_clusters — grouped conflict patterns with pressure scores
  cluster_members   — many-to-many: cluster → conflict
  support_links     — claim-pair agreements (cross-lens or cross-doc validation)

WHAT IS A CLAIM?
================
A claim is one lens's structured assertion about a span of text:
  {subject, predicate, object, doc_id, span, lens, confidence, assumptions}

Claims are NOT facts.  They are *competing hypotheses* from *narrow, biased
micro-models*.  The graph pressure comes from disagreement, not consensus.

WHAT IS A CONFLICT?
===================
A conflict arises when two claims share the same (subject, predicate) pair
but assert incompatible objects — or when their predicates are antonyms.

Example:
  Claim A: (John Smith, arrived_on, 2024-03-14)  [lens: timeline_anomaly]
  Claim B: (John Smith, arrived_on, 2024-03-17)  [lens: email_thread]
  → conflict: timeline_discrepancy, strength=min(conf_A, conf_B)=0.61

WHAT IS A CLUSTER?
==================
A group of conflicts sharing the same predicate type (or entity).
Cluster pressure = conflict_count / max(support_count, 1).
High-pressure clusters become hot zones for reprocessing.

USAGE (library):
    from scripts.claim_graph import ClaimGraph
    cg = ClaimGraph("/tmp/bonfyre-memory")
    claim_id = cg.record_claim({...})
    conflicts = cg.detect_conflicts()
    clusters  = cg.cluster_conflicts(conflicts)
    hot_zones = cg.get_pressure_zones(top_n=10)

USAGE (CLI):
    python3 scripts/claim_graph.py summary --memory-dir /tmp/bonfyre-memory
    python3 scripts/claim_graph.py conflicts [--min-strength 0.4]
    python3 scripts/claim_graph.py hot-zones [--top 20]
    python3 scripts/claim_graph.py export /tmp/claim_export.json
"""

import json
import os
import sqlite3
import sys
import time

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))

# ── Claim Graph Schema (extends memory.db) ────────────────────────────────

_CLAIM_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS claims (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id              TEXT    NOT NULL,
    span_start          INTEGER DEFAULT 0,
    span_end            INTEGER DEFAULT 0,
    span_text           TEXT    DEFAULT '',
    subject             TEXT    NOT NULL,
    predicate           TEXT    NOT NULL,
    object              TEXT    NOT NULL,
    lens                TEXT    NOT NULL,
    family              TEXT,
    confidence          REAL    DEFAULT 0.5,
    assumptions         TEXT    DEFAULT '[]',
    claimed_at          TEXT    NOT NULL,
    run_id              INTEGER,
    -- Phase 13: Structural Convergence scoring fields
    support_count       INTEGER DEFAULT 0,
    independent_support INTEGER DEFAULT 0,
    conflict_count      INTEGER DEFAULT 0,
    assumption_fragility REAL   DEFAULT 0.0,
    lens_diversity      INTEGER DEFAULT 1,
    claim_strength      REAL    DEFAULT 0.0,
    stability_score     REAL    DEFAULT 0.0,
    last_scored         TEXT,
    -- Phase 14: Orthogonal Pressure scoring fields
    graph_pressure          REAL DEFAULT NULL,
    temporal_pressure       REAL DEFAULT NULL,
    frequency_pressure      REAL DEFAULT NULL,
    perturbation_pressure   REAL DEFAULT NULL,
    representation_pressure REAL DEFAULT NULL,
    orthogonal_pressure     REAL DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS conflicts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_a         INTEGER REFERENCES claims(id),
    claim_b         INTEGER REFERENCES claims(id),
    conflict_type   TEXT    NOT NULL,
    strength        REAL    DEFAULT 0.5,
    detected_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS conflict_clusters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_type    TEXT    NOT NULL,
    n_members       INTEGER DEFAULT 0,
    pressure_score  REAL    DEFAULT 0.0,
    subjects_json   TEXT    DEFAULT '[]',
    docs_json       TEXT    DEFAULT '[]',
    lens_json       TEXT    DEFAULT '[]',
    resolved        INTEGER DEFAULT 0,
    first_seen      TEXT,
    last_seen       TEXT
);

CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id  INTEGER REFERENCES conflict_clusters(id),
    conflict_id INTEGER REFERENCES conflicts(id),
    PRIMARY KEY (cluster_id, conflict_id)
);

CREATE TABLE IF NOT EXISTS support_links (
    claim_a     INTEGER REFERENCES claims(id),
    claim_b     INTEGER REFERENCES claims(id),
    strength    REAL    DEFAULT 0.5,
    linked_at   TEXT    NOT NULL,
    PRIMARY KEY (claim_a, claim_b)
);

CREATE TABLE IF NOT EXISTS independent_support_groups (
    group_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        INTEGER REFERENCES claims(id),
    support_claim_id INTEGER REFERENCES claims(id),
    is_independent  INTEGER DEFAULT 1,
    lens_diff       INTEGER DEFAULT 0,
    family_diff     INTEGER DEFAULT 0,
    assumption_diff INTEGER DEFAULT 0,
    created_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS convergence_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration       INTEGER NOT NULL,
    cluster_id      INTEGER,
    n_claims_before INTEGER,
    n_claims_after  INTEGER,
    stable_edges    INTEGER,
    fragile_edges   INTEGER,
    pressure_decay  REAL,
    converged       INTEGER DEFAULT 0,
    timestamp       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_doc        ON claims(doc_id);
CREATE INDEX IF NOT EXISTS idx_claims_subject    ON claims(subject);
CREATE INDEX IF NOT EXISTS idx_claims_predicate  ON claims(predicate);
CREATE INDEX IF NOT EXISTS idx_claims_lens       ON claims(lens);
CREATE INDEX IF NOT EXISTS idx_claims_strength   ON claims(claim_strength);
CREATE INDEX IF NOT EXISTS idx_claims_stability  ON claims(stability_score);
CREATE INDEX IF NOT EXISTS idx_conflicts_type    ON conflicts(conflict_type);
CREATE INDEX IF NOT EXISTS idx_support_groups    ON independent_support_groups(claim_id);
"""

# ── Conflict predicate pairs (antonyms / contradictions) ─────────────────

ANTONYM_PAIRS = [
    ("arrived_on",    "departed_on"),
    ("confirmed",     "denied"),
    ("present_at",    "absent_from"),
    ("sent_by",       "not_sent_by"),
    ("alias_of",      "distinct_from"),
    ("before",        "after"),
    ("supports",      "contradicts"),
    ("coercion_explicit", "coercion_implicit"),
    ("redacted",      "present"),
]

_ANTONYM_MAP = {}
for a, b in ANTONYM_PAIRS:
    _ANTONYM_MAP[a] = b
    _ANTONYM_MAP[b] = a


class ClaimGraph:
    """
    Persistent claim graph on top of Bonfyre transform memory.

    Opens (or extends) memory.db by adding claim-specific tables.
    Does not modify existing Bonfyre schema tables.
    """

    def __init__(self, memory_dir: str = "/tmp/bonfyre-memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        db_path = os.path.join(memory_dir, "memory.db")
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_CLAIM_SCHEMA)
        self._db.commit()

    # ── Core: record a claim ─────────────────────────────────────────

    def record_claim(self, claim: dict, run_id: int = None) -> int:
        """
        Persist one claim dict.  Returns claim_id.

        Required keys: doc_id, subject, predicate, object, lens
        Optional:      span_start, span_end, span_text, family,
                       confidence, assumptions
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        assumptions = claim.get("assumptions", [])
        if isinstance(assumptions, list):
            assumptions = json.dumps(assumptions)
        cur = self._db.execute("""
            INSERT INTO claims
                (doc_id, span_start, span_end, span_text, subject, predicate,
                 object, lens, family, confidence, assumptions, claimed_at, run_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            claim.get("doc_id", ""),
            claim.get("span_start", 0),
            claim.get("span_end", 0),
            claim.get("span_text", "")[:500],
            claim.get("subject", ""),
            claim.get("predicate", ""),
            claim.get("object", ""),
            claim.get("lens", ""),
            claim.get("family"),
            float(claim.get("confidence", 0.5)),
            assumptions,
            ts,
            run_id,
        ))
        self._db.commit()
        return cur.lastrowid

    def record_claims(self, claims: list, run_id: int = None) -> list:
        """Batch insert. Returns list of claim_ids."""
        return [self.record_claim(c, run_id) for c in claims]

    # ── Conflict detection ────────────────────────────────────────────

    def detect_conflicts(self, doc_id: str = None,
                         min_confidence: float = 0.25) -> list:
        """
        Scan claims for conflicts.  Two claims conflict when:
          A. Same (subject, predicate), different object, both conf > threshold
          B. Same subject, antonym predicates

        If doc_id given: only scan claims from that document.
        Returns list of conflict dicts (not yet persisted).
        """
        where = "WHERE c.confidence >= ?"
        params = [min_confidence]
        if doc_id:
            where += " AND c.doc_id = ?"
            params.append(doc_id)

        rows = self._db.execute(f"""
            SELECT id, doc_id, subject, predicate, object, lens, confidence
            FROM claims c
            {where}
            ORDER BY subject, predicate
        """, params).fetchall()

        from itertools import combinations

        # Group by (subject, predicate)
        groups: dict = {}
        for r in rows:
            key = (r["subject"].lower().strip(), r["predicate"])
            groups.setdefault(key, []).append(dict(r))

        conflicts = []
        seen_pairs = set()

        # Type A: same subject+predicate, different object
        for (subj, pred), members in groups.items():
            if len(members) < 2:
                continue
            for ca, cb in combinations(members, 2):
                if ca["object"].lower().strip() == cb["object"].lower().strip():
                    continue   # agreement, not conflict
                pair = (min(ca["id"], cb["id"]), max(ca["id"], cb["id"]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                strength = min(ca["confidence"], cb["confidence"])
                conflicts.append({
                    "claim_a_id":    ca["id"],
                    "claim_b_id":    cb["id"],
                    "claim_a":       ca,
                    "claim_b":       cb,
                    "conflict_type": pred,
                    "strength":      round(strength, 4),
                    "subject":       ca["subject"],
                    "predicate":     pred,
                    "object_a":      ca["object"],
                    "object_b":      cb["object"],
                })

        # Type B: antonym predicates on same subject
        by_subject: dict = {}
        for r in rows:
            by_subject.setdefault(r["subject"].lower().strip(), []).append(dict(r))

        for subj, members in by_subject.items():
            for ca in members:
                antonym = _ANTONYM_MAP.get(ca["predicate"])
                if not antonym:
                    continue
                for cb in members:
                    if cb["id"] == ca["id"] or cb["predicate"] != antonym:
                        continue
                    pair = (min(ca["id"], cb["id"]), max(ca["id"], cb["id"]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    strength = min(ca["confidence"], cb["confidence"])
                    conflicts.append({
                        "claim_a_id":    ca["id"],
                        "claim_b_id":    cb["id"],
                        "claim_a":       ca,
                        "claim_b":       cb,
                        "conflict_type": f"antonym:{ca['predicate']}↔{cb['predicate']}",
                        "strength":      round(strength, 4),
                        "subject":       ca["subject"],
                        "predicate":     f"{ca['predicate']}↔{cb['predicate']}",
                        "object_a":      ca["object"],
                        "object_b":      cb["object"],
                    })

        return conflicts

    def persist_conflicts(self, conflicts: list) -> list:
        """Insert conflict records, return list of conflict_ids."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ids = []
        for c in conflicts:
            cur = self._db.execute("""
                INSERT INTO conflicts (claim_a, claim_b, conflict_type, strength, detected_at)
                VALUES (?,?,?,?,?)
            """, (c["claim_a_id"], c["claim_b_id"],
                  c["conflict_type"], c["strength"], ts))
            ids.append(cur.lastrowid)
        self._db.commit()
        return ids

    # ── Support links (cross-validation) ─────────────────────────────

    def detect_support_links(self, min_confidence: float = 0.3) -> list:
        """
        Two claims *support* each other when same (subject, predicate, ~object)
        comes from different lenses or different documents.
        Returns list of support pair dicts.
        """
        rows = self._db.execute("""
            SELECT id, doc_id, subject, predicate, object, lens, confidence
            FROM claims WHERE confidence >= ?
        """, (min_confidence,)).fetchall()

        groups: dict = {}
        for r in rows:
            key = (r["subject"].lower().strip(), r["predicate"],
                   r["object"].lower().strip()[:30])
            groups.setdefault(key, []).append(dict(r))

        from itertools import combinations
        links = []
        seen = set()
        for (subj, pred, obj), members in groups.items():
            if len(members) < 2:
                continue
            for ca, cb in combinations(members, 2):
                if ca["lens"] == cb["lens"] and ca["doc_id"] == cb["doc_id"]:
                    continue   # same lens same doc = not independent support
                pair = (min(ca["id"], cb["id"]), max(ca["id"], cb["id"]))
                if pair in seen:
                    continue
                seen.add(pair)
                strength = (ca["confidence"] + cb["confidence"]) / 2
                links.append({
                    "claim_a_id": ca["id"],
                    "claim_b_id": cb["id"],
                    "claim_a":    ca,
                    "claim_b":    cb,
                    "subject":    ca["subject"],
                    "predicate":  pred,
                    "strength":   round(strength, 4),
                })
        return links

    def persist_support_links(self, links: list):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for lnk in links:
            try:
                self._db.execute("""
                    INSERT OR IGNORE INTO support_links (claim_a, claim_b, strength, linked_at)
                    VALUES (?,?,?,?)
                """, (lnk["claim_a_id"], lnk["claim_b_id"], lnk["strength"], ts))
            except Exception:
                pass
        self._db.commit()

    # ── Conflict clustering ───────────────────────────────────────────

    def cluster_conflicts(self, conflicts: list) -> list:
        """
        Group conflicts into clusters by (conflict_type, dominant subject).

        Each cluster:
          - conflict_type: e.g. "entity_variant", "timeline_anomaly"
          - subjects: entities most frequently in conflict
          - pressure_score: conflict_count / max(support_count_in_cluster, 1)
          - docs: source documents contributing to conflicts

        Returns list of cluster dicts, sorted by pressure_score descending.
        Persists clusters to conflict_clusters table.
        """
        if not conflicts:
            return []

        # Group by conflict_type
        by_type: dict = {}
        for c in conflicts:
            ct = c["conflict_type"].split(":")[0]   # normalize antonym: prefix
            by_type.setdefault(ct, []).append(c)

        # Support counts per subject for cross-checking
        support_by_subj = {}
        sup_rows = self._db.execute(
            "SELECT c.subject FROM claims c "
            "JOIN support_links sl ON (sl.claim_a = c.id OR sl.claim_b = c.id)"
        ).fetchall()
        for r in sup_rows:
            subj = r["subject"].lower().strip()
            support_by_subj[subj] = support_by_subj.get(subj, 0) + 1

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        clusters = []

        for ct, group in by_type.items():
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

            # Pressure = conflict count / (support from same subjects + 1)
            support_for_cluster = sum(
                support_by_subj.get(s.lower().strip(), 0) for s in set(subjects))
            pressure = round(len(group) / max(support_for_cluster, 1), 4)

            # Persist cluster
            cur = self._db.execute("""
                INSERT INTO conflict_clusters
                    (cluster_type, n_members, pressure_score, subjects_json,
                     docs_json, lens_json, first_seen, last_seen)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                ct, len(group), pressure,
                json.dumps(list(set(subjects))[:20]),
                json.dumps(docs[:20]),
                json.dumps(lenses[:20]),
                ts, ts,
            ))
            cluster_id = cur.lastrowid
            self._db.commit()

            # Persist cluster_members
            for ci, c in enumerate(group):
                conflict_id = c.get("_persisted_id")
                if conflict_id:
                    try:
                        self._db.execute(
                            "INSERT OR IGNORE INTO cluster_members VALUES (?,?)",
                            (cluster_id, conflict_id))
                    except Exception:
                        pass
            self._db.commit()

            clusters.append({
                "cluster_id":    cluster_id,
                "cluster_type":  ct,
                "n_conflicts":   len(group),
                "pressure_score": pressure,
                "subjects":      list(set(subjects))[:10],
                "docs":          docs[:10],
                "lenses":        lenses[:10],
                "resolved":      False,
            })

        clusters.sort(key=lambda x: -x["pressure_score"])
        return clusters

    # ── Hot zone extraction ───────────────────────────────────────────

    def get_pressure_zones(self, top_n: int = 20) -> list:
        """
        Identify document spans with highest conflict density.

        Returns list of hot zone dicts with:
          doc_id, span_start, span_end, conflict_count, pressure_score,
          dominant_cluster, recommended_lenses
        """
        # Get all claims involved in conflicts
        conflict_rows = self._db.execute("""
            SELECT cf.conflict_type, cf.strength,
                   ca.doc_id, ca.span_start, ca.span_end, ca.lens as lens_a,
                   cb.span_start as b_start, cb.span_end as b_end, cb.lens as lens_b
            FROM conflicts cf
            JOIN claims ca ON cf.claim_a = ca.id
            JOIN claims cb ON cf.claim_b = cb.id
            ORDER BY cf.strength DESC
        """).fetchall()

        # Accumulate per doc
        doc_conflicts: dict = {}
        for r in conflict_rows:
            doc = r["doc_id"]
            doc_conflicts.setdefault(doc, {
                "conflict_count": 0,
                "total_strength": 0.0,
                "span_min": r["span_start"],
                "span_max": r["span_end"],
                "conflict_types": [],
                "lenses_seen": set(),
            })
            d = doc_conflicts[doc]
            d["conflict_count"] += 1
            d["total_strength"] += r["strength"]
            d["span_min"] = min(d["span_min"], r["span_start"], r["b_start"])
            d["span_max"] = max(d["span_max"], r["span_end"], r["b_end"])
            d["conflict_types"].append(r["conflict_type"])
            d["lenses_seen"].add(r["lens_a"])
            d["lenses_seen"].add(r["lens_b"])

        zones = []
        for doc, info in doc_conflicts.items():
            cnt = info["conflict_count"]
            avg_str = info["total_strength"] / max(cnt, 1)
            pressure = round(avg_str * cnt, 4)

            # Dominant conflict type
            from collections import Counter
            dominant = Counter(info["conflict_types"]).most_common(1)
            dom_type = dominant[0][0] if dominant else "unknown"

            # Recommend missing lenses (lenses not seen for this doc)
            LENS_SUGGESTIONS = {
                "entity_variant":      ["alias_expansion", "entity_consistency"],
                "timeline_anomaly":    ["timeline_anomaly", "communication_cadence"],
                "speaker_role":        ["deposition_parser", "entity_role_swap"],
                "coercion_signal":     ["coercion_risk_phrase", "euphemism_detector"],
                "redacted":            ["redaction_boundary"],
                "email_thread":        ["email_thread", "communication_cadence"],
                "contradiction_found": ["contradiction_scan"],
            }
            recs = LENS_SUGGESTIONS.get(dom_type, [])
            recs = [r for r in recs if r not in info["lenses_seen"]]

            zones.append({
                "doc_id":             doc,
                "span_start":         info["span_min"],
                "span_end":           info["span_max"],
                "conflict_count":     cnt,
                "pressure_score":     pressure,
                "dominant_cluster":   dom_type,
                "recommended_lenses": recs,
            })

        zones.sort(key=lambda x: -x["pressure_score"])
        return zones[:top_n]

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> dict:
        n_claims     = self._db.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        n_conflicts  = self._db.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
        n_clusters   = self._db.execute("SELECT COUNT(*) FROM conflict_clusters").fetchone()[0]
        n_support    = self._db.execute("SELECT COUNT(*) FROM support_links").fetchone()[0]
        n_resolved   = self._db.execute(
            "SELECT COUNT(*) FROM conflict_clusters WHERE resolved=1").fetchone()[0]

        avg_pressure = self._db.execute(
            "SELECT AVG(pressure_score) FROM conflict_clusters WHERE n_members > 0"
        ).fetchone()[0] or 0.0

        lenses = self._db.execute(
            "SELECT lens, COUNT(*) as n FROM claims GROUP BY lens ORDER BY n DESC"
        ).fetchall()

        top_subjects = self._db.execute("""
            SELECT subject, COUNT(*) as n FROM claims
            GROUP BY subject ORDER BY n DESC LIMIT 5
        """).fetchall()

        return {
            "n_claims":         n_claims,
            "n_conflicts":      n_conflicts,
            "n_clusters":       n_clusters,
            "n_support_links":  n_support,
            "n_resolved":       n_resolved,
            "avg_pressure":     round(avg_pressure, 4),
            "lens_breakdown":   {r["lens"]: r["n"] for r in lenses},
            "top_subjects":     [(r["subject"], r["n"]) for r in top_subjects],
        }

    # ── Phase 13: Structural Convergence — Claim Scoring ─────────────

    def compute_claim_scores(self, claim_ids: list = None):
        """
        Compute claim strength for all claims (or subset).

        Formula:
            claim_strength = (independent_support × lens_diversity)
                            / (conflict_count + 1)
                            × (1 - assumption_fragility)
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # If no claim_ids given, score all claims
        if claim_ids is None:
            rows = self._db.execute("SELECT id FROM claims").fetchall()
            claim_ids = [r["id"] for r in rows]

        for claim_id in claim_ids:
            # Get claim
            claim = self._db.execute(
                "SELECT * FROM claims WHERE id = ?", (claim_id,)
            ).fetchone()
            if not claim:
                continue

            # 1. Count support links
            support_count = self._db.execute("""
                SELECT COUNT(*) FROM support_links
                WHERE claim_a = ? OR claim_b = ?
            """, (claim_id, claim_id)).fetchone()[0]

            # 2. Count independent support
            independent_support = self._count_independent_support(claim_id, claim)

            # 3. Count conflicts
            conflict_count = self._db.execute("""
                SELECT COUNT(*) FROM conflicts
                WHERE claim_a = ? OR claim_b = ?
            """, (claim_id, claim_id)).fetchone()[0]

            # 4. Compute assumption fragility
            assumptions = json.loads(claim["assumptions"] or "[]")
            assumption_fragility = min(len(assumptions) / 5.0, 1.0)

            # 5. Compute lens diversity (how many different lenses support this claim)
            lens_diversity = self._count_lens_diversity(claim_id, claim)

            # 6. Compute claim strength
            claim_strength = (
                (independent_support * lens_diversity)
                / (conflict_count + 1)
                * (1.0 - assumption_fragility)
            )

            # 7. Compute stability score (survives repeated runs)
            stability_score = self._compute_stability(claim_id, claim)

            # 8. Update claim record
            self._db.execute("""
                UPDATE claims SET
                    support_count = ?,
                    independent_support = ?,
                    conflict_count = ?,
                    assumption_fragility = ?,
                    lens_diversity = ?,
                    claim_strength = ?,
                    stability_score = ?,
                    last_scored = ?
                WHERE id = ?
            """, (
                support_count, independent_support, conflict_count,
                round(assumption_fragility, 4), lens_diversity,
                round(claim_strength, 4), round(stability_score, 4),
                ts, claim_id
            ))

        self._db.commit()

    def _count_independent_support(self, claim_id: int, claim: sqlite3.Row) -> int:
        """
        Count how many support links are truly independent.
        Independent = different lens OR different family OR different assumptions.
        """
        # Get all support links for this claim
        support_rows = self._db.execute("""
            SELECT c2.id, c2.lens, c2.family, c2.assumptions
            FROM support_links sl
            JOIN claims c2 ON (
                CASE WHEN sl.claim_a = ? THEN sl.claim_b ELSE sl.claim_a END = c2.id
            )
            WHERE sl.claim_a = ? OR sl.claim_b = ?
        """, (claim_id, claim_id, claim_id)).fetchall()

        independent_count = 0
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for sup in support_rows:
            lens_diff = 1 if sup["lens"] != claim["lens"] else 0
            family_diff = 1 if (sup["family"] or "") != (claim["family"] or "") else 0

            # Compare assumptions
            try:
                assump_a = set(json.loads(claim["assumptions"] or "[]"))
                assump_b = set(json.loads(sup["assumptions"] or "[]"))
                assump_diff = 1 if assump_a != assump_b else 0
            except Exception:
                assump_diff = 0

            is_independent = lens_diff or family_diff or assump_diff

            if is_independent:
                independent_count += 1

                # Record in independent_support_groups table
                try:
                    self._db.execute("""
                        INSERT OR IGNORE INTO independent_support_groups
                            (claim_id, support_claim_id, is_independent,
                             lens_diff, family_diff, assumption_diff, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (claim_id, sup["id"], 1, lens_diff, family_diff,
                          assump_diff, ts))
                except Exception:
                    pass

        return independent_count

    def _count_lens_diversity(self, claim_id: int, claim: sqlite3.Row) -> int:
        """
        Count how many different lenses support similar claims
        (same subject+predicate+object from different lenses).
        """
        # Find all claims with same subject, predicate, object (approx match)
        similar = self._db.execute("""
            SELECT DISTINCT lens FROM claims
            WHERE subject = ? AND predicate = ?
              AND LOWER(SUBSTR(object, 1, 30)) = LOWER(SUBSTR(?, 1, 30))
        """, (claim["subject"], claim["predicate"],
              claim["object"])).fetchall()

        return len(similar)

    def _compute_stability(self, claim_id: int, claim: sqlite3.Row) -> float:
        """
        Compute stability score: how consistently this claim appears
        across multiple runs with same doc_id + subject + predicate.
        """
        # Count how many different run_ids produced similar claims
        similar_runs = self._db.execute("""
            SELECT COUNT(DISTINCT run_id) FROM claims
            WHERE doc_id = ? AND subject = ? AND predicate = ?
              AND LOWER(SUBSTR(object, 1, 30)) = LOWER(SUBSTR(?, 1, 30))
              AND run_id IS NOT NULL
        """, (claim["doc_id"], claim["subject"], claim["predicate"],
              claim["object"])).fetchone()[0]

        # Stability = recurrence count normalized
        return min(similar_runs / 3.0, 1.0)

    def get_stable_edges(self, min_strength: float = 0.5,
                          max_conflict_density: float = 0.3) -> list:
        """
        Extract stable edges: high claim_strength, low conflict_density.

        Returns list of claim dicts sorted by strength descending.
        """
        rows = self._db.execute("""
            SELECT *, (CAST(conflict_count AS REAL) / (support_count + 1)) as conflict_density
            FROM claims
            WHERE claim_strength >= ?
              AND (CAST(conflict_count AS REAL) / (support_count + 1)) <= ?
            ORDER BY claim_strength DESC, stability_score DESC
        """, (min_strength, max_conflict_density)).fetchall()

        return [dict(r) for r in rows]

    def get_fragile_edges(self, min_support: int = 2,
                           min_conflict_density: float = 0.5) -> list:
        """
        Extract fragile edges: high support but high conflict.

        Returns list of claim dicts sorted by conflict_density descending.
        """
        rows = self._db.execute("""
            SELECT *, (CAST(conflict_count AS REAL) / (support_count + 1)) as conflict_density
            FROM claims
            WHERE support_count >= ?
              AND (CAST(conflict_count AS REAL) / (support_count + 1)) >= ?
            ORDER BY conflict_density DESC
        """, (min_support, min_conflict_density)).fetchall()

        return [dict(r) for r in rows]

    def get_convergence_metrics(self) -> dict:
        """
        Compute convergence metrics across all claims.

        Returns:
            stable_edge_ratio, fragile_edge_ratio, avg_claim_strength,
            conflict_resolution_rate, pressure_decay_rate
        """
        total_claims = self._db.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        if total_claims == 0:
            return {
                "stable_edge_ratio": 0.0,
                "fragile_edge_ratio": 0.0,
                "avg_claim_strength": 0.0,
                "conflict_resolution_rate": 0.0,
                "pressure_decay_rate": 0.0,
            }

        stable_edges = len(self.get_stable_edges(min_strength=0.5, max_conflict_density=0.3))
        fragile_edges = len(self.get_fragile_edges(min_support=2, min_conflict_density=0.5))

        avg_strength = self._db.execute(
            "SELECT AVG(claim_strength) FROM claims WHERE claim_strength > 0"
        ).fetchone()[0] or 0.0

        # Conflict resolution rate: resolved clusters / total clusters
        total_clusters = self._db.execute(
            "SELECT COUNT(*) FROM conflict_clusters"
        ).fetchone()[0]
        resolved_clusters = self._db.execute(
            "SELECT COUNT(*) FROM conflict_clusters WHERE resolved = 1"
        ).fetchone()[0]
        resolution_rate = resolved_clusters / max(total_clusters, 1)

        # Pressure decay: compare latest vs historical pressure scores
        decay = self._compute_pressure_decay()

        return {
            "stable_edge_ratio": round(stable_edges / total_claims, 4),
            "fragile_edge_ratio": round(fragile_edges / total_claims, 4),
            "avg_claim_strength": round(avg_strength, 4),
            "conflict_resolution_rate": round(resolution_rate, 4),
            "pressure_decay_rate": round(decay, 4),
        }

    def _compute_pressure_decay(self) -> float:
        """
        Compare pressure score trend over convergence history.
        Returns decay rate (positive = pressure decreasing).
        """
        history = self._db.execute("""
            SELECT pressure_decay FROM convergence_history
            ORDER BY iteration DESC LIMIT 5
        """).fetchall()

        if len(history) < 2:
            return 0.0

        decays = [h["pressure_decay"] for h in history if h["pressure_decay"] is not None]
        if not decays:
            return 0.0

        return sum(decays) / len(decays)

    # ── Phase 14: Orthogonal Pressure Integration ────────────────────

    def compute_orthogonal_pressure(self, claim_ids: list = None, corpus: dict = None):
        """
        Compute orthogonal pressure scores for claims.

        Args:
            claim_ids: list of claim IDs to score (default: all)
            corpus: dict of {doc_id: doc_text} for frequency analysis

        Updates orthogonal pressure fields in claims table:
          - graph_pressure
          - temporal_pressure
          - frequency_pressure
          - perturbation_pressure
          - representation_pressure
          - orthogonal_pressure (combined)
        """
        from scripts.orthogonal_pressure import OrthogonalPressure

        # If no claim_ids given, score all claims
        if claim_ids is None:
            rows = self._db.execute("SELECT id FROM claims").fetchall()
            claim_ids = [r["id"] for r in rows]

        engine = OrthogonalPressure(enable_expensive=False)

        for claim_id in claim_ids:
            # Get claim
            claim_row = self._db.execute(
                "SELECT * FROM claims WHERE id = ?", (claim_id,)
            ).fetchone()
            if not claim_row:
                continue

            claim = dict(claim_row)

            # Compute pressure scores
            scores = engine.compute_pressure_score(claim, self, corpus or {})

            # Update claim record
            self._db.execute("""
                UPDATE claims SET
                    graph_pressure = ?,
                    temporal_pressure = ?,
                    frequency_pressure = ?,
                    perturbation_pressure = ?,
                    representation_pressure = ?,
                    orthogonal_pressure = ?
                WHERE id = ?
            """, (
                round(scores["graph_pressure"], 4),
                round(scores["temporal_pressure"], 4),
                round(scores["frequency_pressure"], 4),
                round(scores["perturbation_pressure"], 4),
                round(scores["representation_pressure"], 4),
                round(scores["orthogonal_pressure"], 4),
                claim_id
            ))

        self._db.commit()

    def recompute_final_strength(self):
        """
        Recompute final claim strength as:
            final_strength = semantic_strength × orthogonal_pressure

        where semantic_strength is the Phase 13 formula.
        Only updates claims where orthogonal_pressure is not NULL.
        """
        self._db.execute("""
            UPDATE claims
            SET claim_strength = claim_strength * COALESCE(orthogonal_pressure, 1.0)
            WHERE orthogonal_pressure IS NOT NULL
        """)
        self._db.commit()

    def export_all(self, out_path: str):
        """Export full claim graph as JSON."""
        claims = [dict(r) for r in
                  self._db.execute("SELECT * FROM claims").fetchall()]
        conflicts = [dict(r) for r in
                     self._db.execute("SELECT * FROM conflicts").fetchall()]
        clusters = [dict(r) for r in
                    self._db.execute("SELECT * FROM conflict_clusters").fetchall()]
        support = [dict(r) for r in
                   self._db.execute("SELECT * FROM support_links").fetchall()]

        with open(out_path, "w") as f:
            json.dump({
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "claims":      claims,
                "conflicts":   conflicts,
                "clusters":    clusters,
                "support_links": support,
            }, f, indent=2)
        print(f"[claim_graph] exported {len(claims)} claims → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bonfyre claim graph inspector")
    sub = ap.add_subparsers(dest="cmd")

    for sc in ("summary", "conflicts", "hot-zones", "clusters"):
        p = sub.add_parser(sc)
        p.add_argument("--memory-dir", default="/tmp/bonfyre-memory")
        if sc == "conflicts":
            p.add_argument("--min-strength", type=float, default=0.3)
        if sc == "hot-zones":
            p.add_argument("--top", type=int, default=20)

    p_exp = sub.add_parser("export")
    p_exp.add_argument("out", help="Output path for JSON export")
    p_exp.add_argument("--memory-dir", default="/tmp/bonfyre-memory")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return

    cg = ClaimGraph(args.memory_dir)

    if args.cmd == "summary":
        s = cg.summary()
        print(json.dumps(s, indent=2))

    elif args.cmd == "conflicts":
        conflicts = cg.detect_conflicts(min_confidence=args.min_strength)
        print(f"  {len(conflicts)} conflict(s) detected:")
        for c in conflicts[:20]:
            print(f"  [{c['conflict_type']}]  "
                  f"{c['subject']!r}  "
                  f"{c['object_a']!r} ↔ {c['object_b']!r}  "
                  f"strength={c['strength']:.3f}")

    elif args.cmd == "hot-zones":
        zones = cg.get_pressure_zones(top_n=args.top)
        print(f"  {len(zones)} hot zone(s):")
        for z in zones:
            print(f"  [{z['dominant_cluster']}]  {z['doc_id']:<30}  "
                  f"conflicts={z['conflict_count']}  pressure={z['pressure_score']:.3f}  "
                  f"recs={z['recommended_lenses']}")

    elif args.cmd == "clusters":
        rows = cg._db.execute(
            "SELECT * FROM conflict_clusters ORDER BY pressure_score DESC LIMIT 20"
        ).fetchall()
        if not rows:
            print("(no clusters yet)")
            return
        print(f"  {'type':<20}  {'n':>4}  {'pressure':>8}  subjects")
        print("  " + "─" * 60)
        for r in rows:
            subj = ", ".join(json.loads(r["subjects_json"] or "[]")[:3])
            print(f"  {r['cluster_type']:<20}  {r['n_members']:>4}  "
                  f"{r['pressure_score']:>8.4f}  {subj[:40]}")

    elif args.cmd == "export":
        cg.export_all(args.out)


if __name__ == "__main__":
    main()
