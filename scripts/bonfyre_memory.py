#!/usr/bin/env python3
"""
scripts/bonfyre_memory.py — Bonfyre transform memory layer.

SQLite-backed persistent memory for run history, routing decisions,
failure events, escalation events, discovered paths, fragment stats,
per-transition routing weights, and speech temporal memory.

This module is the foundation of Bonfyre's self-evolution system.
ALL other evolution scripts (failure_detect, routing_adjust, path_discover,
auto_evolve) read from and write to this store.

Usage — library:
    from scripts.bonfyre_memory import BonfyreMemory
    mem = BonfyreMemory("/tmp/bonfyre-memory")
    mem.record_run(metrics_dict)                    # ingest demo.py --metrics-out
    patterns = mem.get_failure_patterns(min_count=3)
    adj = mem.get_routing_adjustments()              # per-transition weights
    mem.update_routing_weight("T04", "T15", success=True)
    mem.record_speech_segment(...)                   # temporal speech memory

Usage — standalone:
    python3 scripts/bonfyre_memory.py ingest /tmp/runs.json
    python3 scripts/bonfyre_memory.py summary
    python3 scripts/bonfyre_memory.py export /tmp/memory_export.json
    python3 scripts/bonfyre_memory.py routing
    python3 scripts/bonfyre_memory.py failures [--min-count N]

Memory dir layout (created automatically):
    <memory_dir>/
        memory.db           — SQLite WAL database (all tables)
        runs/               — JSON sidecars of every ingested run
        decisions/          — routing decision snapshots
        failures/           — detected failure event logs
        escalations/        — escalation event logs
        paths/              — discovered chain records
        graph/              — frontier_weights.json, path_scores.json
        speech/             — per-audio temporal segment memory logs
"""

import json
import math
import os
import sqlite3
import sys
import time

# ── Schema ─────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at                  TEXT    NOT NULL,
    schema_ver              TEXT,
    n_inputs                INTEGER,
    routed_family           TEXT,
    fragment_exit           INTEGER,
    early_exit_iter         INTEGER,
    full_loop               INTEGER,
    iterations_ran          INTEGER,
    avg_confidence          REAL,
    min_confidence          REAL,
    escalation_count        INTEGER DEFAULT 0,
    loop_iterations         INTEGER,
    chain                   TEXT,
    speech_in               TEXT,
    asr_avg_logprob         REAL,
    asr_num_segments        INTEGER,
    wer                     REAL,
    fragment_only_asr_rate  REAL,
    raw_json                TEXT
);

CREATE TABLE IF NOT EXISTS escalations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES runs(id),
    run_at      TEXT,
    from_family TEXT,
    to_family   TEXT,
    at_iter     INTEGER,
    conf_before REAL,
    conf_after  REAL
);

CREATE TABLE IF NOT EXISTS routing_weights (
    from_family TEXT NOT NULL,
    to_family   TEXT NOT NULL,
    n_success   INTEGER DEFAULT 0,
    n_failure   INTEGER DEFAULT 0,
    n_escalated INTEGER DEFAULT 0,
    last_updated TEXT,
    PRIMARY KEY (from_family, to_family)
);

CREATE TABLE IF NOT EXISTS failures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    pattern     TEXT NOT NULL,
    family      TEXT,
    count       INTEGER DEFAULT 1,
    detail      TEXT
);

CREATE TABLE IF NOT EXISTS paths (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chain           TEXT NOT NULL UNIQUE,
    n_runs          INTEGER DEFAULT 0,
    avg_confidence  REAL,
    avg_iterations  REAL,
    best_confidence REAL,
    last_seen       TEXT,
    score           REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS fragment_stats (
    family          TEXT PRIMARY KEY,
    n_used          INTEGER DEFAULT 0,
    n_exit          INTEGER DEFAULT 0,
    n_promoted      INTEGER DEFAULT 0,
    exit_rate       REAL    DEFAULT 0.0,
    last_updated    TEXT
);

CREATE TABLE IF NOT EXISTS speech_segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_id    TEXT    NOT NULL,
    session_at  TEXT    NOT NULL,
    segment_idx INTEGER,
    seg_text    TEXT,
    avg_logprob REAL,
    asr_family  TEXT,
    escalated   INTEGER DEFAULT 0,
    label       TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_family    ON runs(routed_family);
CREATE INDEX IF NOT EXISTS idx_runs_at        ON runs(run_at);
CREATE INDEX IF NOT EXISTS idx_esc_families   ON escalations(from_family, to_family);
CREATE INDEX IF NOT EXISTS idx_speech_audio   ON speech_segments(audio_id);
"""


# ══════════════════════════════════════════════════════════════════════════════

class BonfyreMemory:
    """
    Persistent transform memory for Bonfyre self-evolution.

    Thread-safe via SQLite WAL mode.  All writes are immediate (no batching).
    All reads return plain Python dicts / lists (no ORM).
    """

    def __init__(self, memory_dir: str = "/tmp/bonfyre-memory"):
        self.memory_dir = memory_dir
        self._ensure_dirs()
        db_path = os.path.join(memory_dir, "memory.db")
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ── Directory setup ────────────────────────────────────────────────

    def _ensure_dirs(self):
        for sub in ("runs", "decisions", "failures", "escalations",
                    "paths", "graph", "speech"):
            os.makedirs(os.path.join(self.memory_dir, sub), exist_ok=True)

    # ── Core: ingest a demo.py metrics dict ───────────────────────────

    def record_run(self, metrics: dict) -> int:
        """
        Ingest one metrics dict produced by demo.py --metrics-out.
        Returns the new run row id.

        Also:
          - writes a JSON sidecar to runs/
          - updates routing_weights for every escalation in the run
          - updates fragment_stats for the routed family
          - upserts the chain into paths table
        """
        run_at = metrics.get("timestamp") or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        escs = metrics.get("escalations", [])

        # Insert into runs
        cur = self._db.execute("""
            INSERT INTO runs (
                run_at, schema_ver, n_inputs, routed_family,
                fragment_exit, early_exit_iter, full_loop,
                iterations_ran, avg_confidence, min_confidence,
                escalation_count, loop_iterations, chain,
                speech_in, asr_avg_logprob, asr_num_segments,
                wer, fragment_only_asr_rate, raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_at,
            metrics.get("schema"),
            metrics.get("n_inputs"),
            metrics.get("routed_family"),
            1 if metrics.get("fragment_exit") else 0,
            metrics.get("early_exit_iter"),
            1 if metrics.get("full_loop") else 0,
            metrics.get("iterations_ran"),
            metrics.get("avg_confidence"),
            metrics.get("min_confidence"),
            len(escs),
            metrics.get("args", {}).get("loop"),
            metrics.get("args", {}).get("chain"),
            metrics.get("speech_in"),
            metrics.get("asr_avg_logprob"),
            metrics.get("asr_num_segments"),
            metrics.get("wer"),
            metrics.get("fragment_only_asr_rate"),
            json.dumps(metrics),
        ))
        run_id = cur.lastrowid
        self._db.commit()

        # Insert escalation events
        for esc in escs:
            self._db.execute("""
                INSERT INTO escalations (run_id, run_at, from_family, to_family,
                                         at_iter, conf_before, conf_after)
                VALUES (?,?,?,?,?,?,?)
            """, (
                run_id, run_at,
                esc.get("from"), esc.get("to"),
                esc.get("iter"),
                esc.get("conf_before"),
                esc.get("conf_after"),
            ))
            # Update routing weight for this transition (failure — escalated)
            self.update_routing_weight(
                esc.get("from", "?"), esc.get("to", "?"),
                success=False, escalated=True)
        self._db.commit()

        # Update routing weight for the final family (success if no more escalations)
        family = metrics.get("routed_family")
        if family and not escs:
            self.update_routing_weight(family, family, success=True)

        # Update fragment stats
        if family:
            frag_exit = 1 if metrics.get("fragment_exit") else 0
            self._db.execute("""
                INSERT INTO fragment_stats (family, n_used, n_exit, last_updated)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(family) DO UPDATE SET
                    n_used       = n_used + 1,
                    n_exit       = n_exit + excluded.n_exit,
                    exit_rate    = CAST(n_exit + excluded.n_exit AS REAL) / (n_used + 1),
                    last_updated = excluded.last_updated
            """, (family, frag_exit, run_at))
            self._db.commit()

        # Upsert chain into paths table
        chain_str = metrics.get("args", {}).get("chain", "auto")
        if chain_str and metrics.get("avg_confidence") is not None:
            avg_c = metrics.get("avg_confidence", 0.0)
            n_it  = metrics.get("iterations_ran") or 1
            score = avg_c / max(n_it, 1)
            self._db.execute("""
                INSERT INTO paths (chain, n_runs, avg_confidence, avg_iterations,
                                   best_confidence, last_seen, score)
                VALUES (?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(chain) DO UPDATE SET
                    n_runs          = n_runs + 1,
                    avg_confidence  = (avg_confidence * n_runs + excluded.avg_confidence)
                                      / (n_runs + 1),
                    avg_iterations  = (avg_iterations * n_runs + excluded.avg_iterations)
                                      / (n_runs + 1),
                    best_confidence = MAX(best_confidence, excluded.best_confidence),
                    last_seen       = excluded.last_seen,
                    score           = (avg_confidence * n_runs + excluded.avg_confidence)
                                      / (n_runs + 1)
                                      / MAX(avg_iterations, 1)
            """, (chain_str, avg_c, float(n_it), avg_c, run_at, score))
            self._db.commit()

        # Write JSON sidecar
        sidecar = os.path.join(self.memory_dir, "runs",
                               f"{run_at.replace(':', '-')}_{run_id}.json")
        try:
            with open(sidecar, "w") as f:
                json.dump(metrics, f, indent=2)
        except OSError:
            pass

        return run_id

    # ── Routing weights ────────────────────────────────────────────────

    def update_routing_weight(self, from_family: str, to_family: str,
                               success: bool = True, escalated: bool = False):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._db.execute("""
            INSERT INTO routing_weights (from_family, to_family,
                n_success, n_failure, n_escalated, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(from_family, to_family) DO UPDATE SET
                n_success   = n_success   + excluded.n_success,
                n_failure   = n_failure   + excluded.n_failure,
                n_escalated = n_escalated + excluded.n_escalated,
                last_updated = excluded.last_updated
        """, (
            from_family, to_family,
            1 if success else 0,
            0 if success else 1,
            1 if escalated else 0,
            ts,
        ))
        self._db.commit()

    def get_routing_adjustments(self) -> dict:
        """
        Return per-transition weight dict suitable for routing_adjust.py.

        Output: {
          "T04→T15": {"success_rate": 0.7, "n_total": 10, "escalated_rate": 0.3},
          ...
        }
        """
        rows = self._db.execute(
            "SELECT from_family, to_family, n_success, n_failure, n_escalated "
            "FROM routing_weights"
        ).fetchall()
        result = {}
        for r in rows:
            total = r["n_success"] + r["n_failure"]
            if total == 0:
                continue
            key = f"{r['from_family']}→{r['to_family']}"
            result[key] = {
                "success_rate":   round(r["n_success"] / total, 4),
                "failure_rate":   round(r["n_failure"] / total, 4),
                "escalated_rate": round(r["n_escalated"] / max(total, 1), 4),
                "n_total":        total,
            }
        return result

    # ── Failure pattern queries ────────────────────────────────────────

    def get_failure_patterns(self, min_count: int = 3,
                              last_n_runs: int = 50) -> list:
        """
        Scan last N runs for three failure signatures:

        1. oscillation    — same A→B→A escalation in ≥ min_count run windows
        2. collapse       — avg_confidence trend is monotonically decreasing
                            across consecutive escalation runs
        3. repeat_esc     — same (from_family, to_family) pair in ≥ min_count runs
        4. fragment_fail  — fragment_exit=0 for family in ≥ min_count consecutive runs

        Returns list of dicts: [{pattern, family, count, detail}]
        """
        patterns = []

        # ── 1. Oscillation: A→B→A sequences ──────────────────────────
        rows = self._db.execute("""
            SELECT from_family, to_family, run_id
            FROM escalations
            ORDER BY id DESC
            LIMIT ?
        """, (last_n_runs * 4,)).fetchall()

        osc_counts = {}
        for i in range(len(rows) - 2):
            a, b, c = rows[i], rows[i + 1], rows[i + 2]
            if (a["from_family"] == c["to_family"] and
                    a["to_family"] == c["from_family"]):
                key = (a["from_family"], a["to_family"])
                osc_counts[key] = osc_counts.get(key, 0) + 1

        for (fa, fb), cnt in osc_counts.items():
            if cnt >= min_count:
                patterns.append({
                    "pattern": "oscillation",
                    "family":  fa,
                    "count":   cnt,
                    "detail":  {"from": fa, "to": fb, "sequence": f"{fa}→{fb}→{fa}"},
                })

        # ── 2. Confidence collapse: monotone drop across last N esc runs ──
        esc_runs = self._db.execute("""
            SELECT r.avg_confidence, r.run_at, r.routed_family
            FROM runs r
            WHERE r.escalation_count > 0
            ORDER BY r.run_at DESC
            LIMIT ?
        """, (last_n_runs,)).fetchall()

        by_family: dict = {}
        for row in esc_runs:
            fam = row["routed_family"] or "?"
            by_family.setdefault(fam, []).append(row["avg_confidence"] or 0.0)

        for fam, confs in by_family.items():
            if len(confs) < min_count:
                continue
            window = confs[:min_count]
            # Is the window monotonically decreasing?
            is_collapse = all(window[i] >= window[i + 1]
                              for i in range(len(window) - 1))
            if is_collapse:
                drop = round(window[0] - window[-1], 4)
                patterns.append({
                    "pattern": "collapse",
                    "family":  fam,
                    "count":   min_count,
                    "detail":  {"confidence_window": window, "drop": drop},
                })

        # ── 3. Repeated escalation: same (from→to) in ≥ min_count runs ──
        repeat = self._db.execute("""
            SELECT from_family, to_family, COUNT(*) as cnt
            FROM escalations
            GROUP BY from_family, to_family
            HAVING cnt >= ?
            ORDER BY cnt DESC
        """, (min_count,)).fetchall()

        for r in repeat:
            patterns.append({
                "pattern": "repeat_esc",
                "family":  r["from_family"],
                "count":   r["cnt"],
                "detail":  {"from": r["from_family"], "to": r["to_family"]},
            })

        # ── 4. Fragment failure: fragment_exit=0 streak ───────────────
        frag_rows = self._db.execute("""
            SELECT routed_family, fragment_exit
            FROM runs
            WHERE routed_family IS NOT NULL
            ORDER BY run_at DESC
            LIMIT ?
        """, (last_n_runs,)).fetchall()

        frag_streak: dict = {}
        for r in frag_rows:
            f = r["routed_family"]
            if r["fragment_exit"] == 0:
                frag_streak[f] = frag_streak.get(f, 0) + 1
            else:
                frag_streak[f] = 0

        for fam, streak in frag_streak.items():
            if streak >= min_count:
                patterns.append({
                    "pattern": "fragment_fail",
                    "family":  fam,
                    "count":   streak,
                    "detail":  {"failed_exits": streak, "family": fam},
                })

        return patterns

    # ── Path / chain queries ───────────────────────────────────────────

    def get_top_paths(self, n: int = 10) -> list:
        """Return top N chains by score (confidence_gain / iterations)."""
        rows = self._db.execute("""
            SELECT chain, n_runs, avg_confidence, avg_iterations,
                   best_confidence, score
            FROM paths
            WHERE n_runs > 0
            ORDER BY score DESC, best_confidence DESC
            LIMIT ?
        """, (n,)).fetchall()
        return [dict(r) for r in rows]

    def record_path(self, chain: str, avg_confidence: float,
                    avg_iterations: float):
        """Register a newly discovered chain (or update it if already known)."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        score = avg_confidence / max(avg_iterations, 1)
        self._db.execute("""
            INSERT INTO paths (chain, n_runs, avg_confidence, avg_iterations,
                               best_confidence, last_seen, score)
            VALUES (?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(chain) DO UPDATE SET
                n_runs          = n_runs + 1,
                avg_confidence  = (avg_confidence * n_runs + excluded.avg_confidence)
                                  / (n_runs + 1),
                avg_iterations  = (avg_iterations * n_runs + excluded.avg_iterations)
                                  / (n_runs + 1),
                best_confidence = MAX(best_confidence, excluded.best_confidence),
                last_seen       = excluded.last_seen,
                score           = (avg_confidence * n_runs + excluded.avg_confidence)
                                  / (n_runs + 1) / MAX(avg_iterations, 1)
        """, (chain, avg_confidence, avg_iterations, avg_confidence, ts, score))
        self._db.commit()

    # ── Speech temporal memory ─────────────────────────────────────────

    def record_speech_segment(self, audio_id: str, segment_idx: int,
                               seg_text: str, avg_logprob: float,
                               asr_family: str, escalated: bool,
                               label: str = ""):
        """Record one ASR segment for temporal memory."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._db.execute("""
            INSERT INTO speech_segments
                (audio_id, session_at, segment_idx, seg_text,
                 avg_logprob, asr_family, escalated, label)
            VALUES (?,?,?,?,?,?,?,?)
        """, (audio_id, ts, segment_idx, seg_text,
              avg_logprob, asr_family, 1 if escalated else 0, label))
        self._db.commit()

    def get_speaker_profile(self, audio_id: str) -> dict:
        """
        Return escalation profile for a known audio source.
        Used by temporal_memory.py to pre-select ASR family for new segments.

        Returns: {audio_id, n_segments, escalation_rate, preferred_family}
        """
        rows = self._db.execute("""
            SELECT asr_family, escalated,
                   AVG(avg_logprob) as mean_lp,
                   COUNT(*) as n
            FROM speech_segments
            WHERE audio_id = ?
            GROUP BY asr_family, escalated
        """, (audio_id,)).fetchall()

        if not rows:
            return {"audio_id": audio_id, "known": False}

        n_total = sum(r["n"] for r in rows)
        n_escalated = sum(r["n"] for r in rows if r["escalated"])
        esc_rate = round(n_escalated / max(n_total, 1), 4)

        # Preferred family = family with most non-escalated "clear" segments
        family_counts: dict = {}
        for r in rows:
            if not r["escalated"]:
                fam = r["asr_family"] or "S01"
                family_counts[fam] = family_counts.get(fam, 0) + r["n"]
        preferred = max(family_counts, key=family_counts.get) if family_counts else "S01"

        return {
            "audio_id":        audio_id,
            "known":           True,
            "n_segments":      n_total,
            "escalation_rate": esc_rate,
            "preferred_family": preferred,
            "skip_s01":        esc_rate > 0.5,  # start at S02 directly if noisy
        }

    # ── Failure recording ──────────────────────────────────────────────

    def record_failure(self, pattern: str, family: str,
                       count: int, detail: dict):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._db.execute("""
            INSERT INTO failures (detected_at, pattern, family, count, detail)
            VALUES (?,?,?,?,?)
        """, (ts, pattern, family, count, json.dumps(detail)))
        self._db.commit()

        # Write sidecar to failures/
        fname = os.path.join(
            self.memory_dir, "failures",
            f"{ts.replace(':', '-')}_{pattern}_{family}.json")
        try:
            with open(fname, "w") as fh:
                json.dump({"detected_at": ts, "pattern": pattern,
                           "family": family, "count": count, "detail": detail}, fh, indent=2)
        except OSError:
            pass

    # ── Meta summary ───────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return high-level memory summary dict."""
        runs_count = self._db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        esc_count  = self._db.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
        fail_count = self._db.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        path_count = self._db.execute("SELECT COUNT(*) FROM paths").fetchone()[0]
        seg_count  = self._db.execute("SELECT COUNT(*) FROM speech_segments").fetchone()[0]

        avg_conf = self._db.execute(
            "SELECT AVG(avg_confidence) FROM runs WHERE avg_confidence IS NOT NULL"
        ).fetchone()[0]

        top_family = self._db.execute("""
            SELECT routed_family, COUNT(*) as n
            FROM runs WHERE routed_family IS NOT NULL
            GROUP BY routed_family ORDER BY n DESC LIMIT 1
        """).fetchone()

        rw_rows = self._db.execute("SELECT * FROM routing_weights").fetchall()

        return {
            "total_runs":        runs_count,
            "total_escalations": esc_count,
            "total_failures":    fail_count,
            "total_paths":       path_count,
            "speech_segments":   seg_count,
            "avg_confidence":    round(avg_conf, 4) if avg_conf else None,
            "top_family":        dict(top_family) if top_family else None,
            "routing_weights":   {
                f"{r['from_family']}→{r['to_family']}": {
                    "n_success": r["n_success"],
                    "n_failure": r["n_failure"],
                    "n_escalated": r["n_escalated"],
                }
                for r in rw_rows
            },
        }

    def export_summary(self, out_path: str):
        """Write full summary JSON to out_path."""
        data = {
            "summary":          self.summary(),
            "top_paths":        self.get_top_paths(20),
            "failure_patterns": self.get_failure_patterns(min_count=2),
            "routing_adjustments": self.get_routing_adjustments(),
        }
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[memory] exported → {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Bonfyre memory layer CLI")
    sub = ap.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest a demo.py --metrics-out JSON")
    p_ingest.add_argument("path", help="Path to metrics JSON (array of run dicts)")
    p_ingest.add_argument("--memory-dir", default="/tmp/bonfyre-memory")

    p_summary = sub.add_parser("summary", help="Print memory summary")
    p_summary.add_argument("--memory-dir", default="/tmp/bonfyre-memory")

    p_export = sub.add_parser("export", help="Export full summary JSON")
    p_export.add_argument("out", help="Output path for summary JSON")
    p_export.add_argument("--memory-dir", default="/tmp/bonfyre-memory")

    p_routing = sub.add_parser("routing", help="Show routing weight table")
    p_routing.add_argument("--memory-dir", default="/tmp/bonfyre-memory")

    p_fail = sub.add_parser("failures", help="List detected failure patterns")
    p_fail.add_argument("--memory-dir", default="/tmp/bonfyre-memory")
    p_fail.add_argument("--min-count", type=int, default=3)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return

    mem = BonfyreMemory(args.memory_dir)

    if args.cmd == "ingest":
        try:
            data = json.load(open(args.path))
        except Exception as e:
            print(f"ERROR: {e}"); sys.exit(1)
        runs = data if isinstance(data, list) else [data]
        for i, run in enumerate(runs):
            rid = mem.record_run(run)
            print(f"  [{i+1}/{len(runs)}] ingested → run_id={rid}  "
                  f"family={run.get('routed_family')}  "
                  f"conf={run.get('avg_confidence')}")
        print(f"\nIngested {len(runs)} run(s) into {args.memory_dir}")

    elif args.cmd == "summary":
        s = mem.summary()
        print(json.dumps(s, indent=2))

    elif args.cmd == "export":
        mem.export_summary(args.out)

    elif args.cmd == "routing":
        adj = mem.get_routing_adjustments()
        if not adj:
            print("(no routing weight data yet)")
            return
        print(f"  {'transition':<15}  {'success':>8}  {'failure':>8}  "
              f"{'escalated':>9}  {'n_total':>7}")
        print("  " + "─" * 58)
        for key, v in sorted(adj.items()):
            print(f"  {key:<15}  {v['success_rate']:>8.3f}  "
                  f"{v['failure_rate']:>8.3f}  "
                  f"{v['escalated_rate']:>9.3f}  {v['n_total']:>7}")

    elif args.cmd == "failures":
        patterns = mem.get_failure_patterns(min_count=args.min_count)
        if not patterns:
            print(f"(no failure patterns with count >= {args.min_count})")
            return
        for p in patterns:
            print(f"  [{p['pattern']}]  family={p['family']}  "
                  f"count={p['count']}  detail={json.dumps(p['detail'])}")


if __name__ == "__main__":
    _cli()
