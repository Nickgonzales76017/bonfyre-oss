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
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT    NOT NULL,
    span_start  INTEGER DEFAULT 0,
    span_end    INTEGER DEFAULT 0,
    span_text   TEXT    DEFAULT '',
    subject     TEXT    NOT NULL,
    predicate   TEXT    NOT NULL,
    object      TEXT    NOT NULL,
    lens        TEXT    NOT NULL,
    family      TEXT,
    confidence  REAL    DEFAULT 0.5,
    assumptions TEXT    DEFAULT '[]',
    claimed_at  TEXT    NOT NULL,
    run_id      INTEGER
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

CREATE INDEX IF NOT EXISTS idx_claims_doc        ON claims(doc_id);
CREATE INDEX IF NOT EXISTS idx_claims_subject    ON claims(subject);
CREATE INDEX IF NOT EXISTS idx_claims_predicate  ON claims(predicate);
CREATE INDEX IF NOT EXISTS idx_claims_lens       ON claims(lens);
CREATE INDEX IF NOT EXISTS idx_conflicts_type    ON conflicts(conflict_type);
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
