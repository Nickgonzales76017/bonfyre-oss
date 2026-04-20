#!/usr/bin/env python3
"""
scripts/failure_detect.py — Bonfyre failure pattern detector.

Reads the transform memory (SQLite) and emits structured failure events
for four failure modes:

    oscillation    — graph is trapped in A→B→A loop (ping-pong instability)
    collapse       — confidence monotonically decreasing across sequential runs
    repeat_esc     — same escalation transition fires in ≥ N consecutive runs
    fragment_fail  — fragment exit never fires for a family (fragment is dead weight)

When patterns are found they are:
  1. Written to memory.failures table (via BonfyreMemory.record_failure)
  2. Written as human-readable JSON to <memory_dir>/failures/

Caller (auto_evolve.py) decides what to do with each pattern.

Usage:
    python3 scripts/failure_detect.py [--memory-dir /tmp/bonfyre-memory]
                                       [--min-count N]
                                       [--last-runs N]
                                       [--dry-run]
"""

import argparse
import json
import os
import sys
import time

# Allow running as:  python3 scripts/failure_detect.py
_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))

from scripts.bonfyre_memory import BonfyreMemory  # noqa: E402


# ── Detector ────────────────────────────────────────────────────────────────

class FailureDetector:
    """
    Stateless detector: reads memory, emits pattern dicts.

    Design rules:
      - Pure query + heuristic, no ML, no external deps
      - Each pattern detector is independent and idempotent
      - min_count is the minimum occurrence threshold before a pattern fires
    """

    def __init__(self, mem: BonfyreMemory,
                 min_count: int = 3, last_runs: int = 50):
        self.mem = mem
        self.min_count = min_count
        self.last_runs = last_runs

    def detect_all(self) -> list:
        """
        Run all detectors.  Returns merged list of pattern dicts sorted by severity.

        Severity order: oscillation > collapse > repeat_esc > fragment_fail
        """
        patterns = self.mem.get_failure_patterns(
            min_count=self.min_count, last_n_runs=self.last_runs)

        # Enrich with severity score for auto_evolve.py
        severity_rank = {
            "oscillation":   4,
            "collapse":      3,
            "repeat_esc":    2,
            "fragment_fail": 1,
        }
        for p in patterns:
            p["severity"] = severity_rank.get(p["pattern"], 0)

        patterns.sort(key=lambda x: (-x["severity"], -x["count"]))
        return patterns

    def detect_oscillation(self) -> list:
        """
        Detect family oscillation: runs where the escalation sequence
        shows A→B followed shortly by B→A within a sliding window.
        """
        db = self.mem._db
        rows = db.execute("""
            SELECT e.from_family, e.to_family, e.run_id, r.run_at
            FROM escalations e
            JOIN runs r ON e.run_id = r.id
            ORDER BY e.id DESC
            LIMIT ?
        """, (self.last_runs * 4,)).fetchall()

        windows: dict = {}
        for i in range(len(rows) - 2):
            a, b, c = rows[i], rows[i + 1], rows[i + 2]
            if (a["from_family"] == c["to_family"] and
                    a["to_family"] == c["from_family"]):
                key = f"{a['from_family']}→{a['to_family']}→{a['from_family']}"
                windows[key] = windows.get(key, 0) + 1

        found = []
        for seq, cnt in windows.items():
            if cnt >= self.min_count:
                parts = seq.split("→")
                found.append({
                    "pattern":  "oscillation",
                    "family":   parts[0],
                    "count":    cnt,
                    "severity": 4,
                    "detail":   {"sequence": seq, "loop_count": cnt},
                    "recommendation": (
                        f"Generate specialized transform for the region where "
                        f"{parts[0]} and {parts[1]} disagree. "
                        f"Trigger: auto_evolve --from-failures"
                    ),
                })
        return found

    def detect_confidence_trend(self) -> list:
        """
        Detect monotone confidence degradation over last N escalation-bearing runs.
        Each family is evaluated independently.
        """
        db = self.mem._db
        family_runs = db.execute("""
            SELECT routed_family, avg_confidence, run_at
            FROM runs
            WHERE escalation_count > 0 AND avg_confidence IS NOT NULL
            ORDER BY run_at DESC
            LIMIT ?
        """, (self.last_runs,)).fetchall()

        by_family: dict = {}
        for r in family_runs:
            f = r["routed_family"] or "?"
            by_family.setdefault(f, []).append(r["avg_confidence"])

        found = []
        for fam, confs in by_family.items():
            if len(confs) < self.min_count:
                continue
            window = confs[:self.min_count]
            is_collapse = all(window[i] >= window[i + 1]
                              for i in range(len(window) - 1))
            if not is_collapse:
                continue
            drop = round(window[0] - window[-1], 4)
            found.append({
                "pattern":  "collapse",
                "family":   fam,
                "count":    len(window),
                "severity": 3,
                "detail":   {
                    "confidence_window": window,
                    "drop": drop,
                    "family": fam,
                },
                "recommendation": (
                    f"Family {fam} is losing confidence across runs. "
                    f"Consider re-collapsing with fresh corpus data, "
                    f"or demoting fragment and rebuilding from failures."
                ),
            })
        return found

    def detect_routing_dead_ends(self) -> list:
        """
        Identify transitions with very high escalation rates (> 80% of attempts).
        These are routing dead ends: the source family consistently needs help.
        """
        adj = self.mem.get_routing_adjustments()
        found = []
        for transition, stats in adj.items():
            if stats["n_total"] < self.min_count:
                continue
            if stats["escalated_rate"] > 0.8:
                parts = transition.split("→")
                from_f = parts[0] if len(parts) >= 1 else "?"
                to_f   = parts[1] if len(parts) >= 2 else "?"
                found.append({
                    "pattern":  "dead_end",
                    "family":   from_f,
                    "count":    stats["n_total"],
                    "severity": 2,
                    "detail":   {
                        "transition":     transition,
                        "escalated_rate": stats["escalated_rate"],
                        "n_total":        stats["n_total"],
                    },
                    "recommendation": (
                        f"Transition {transition} escalates {stats['escalated_rate']*100:.0f}% "
                        f"of the time. Weight this path DOWN in frontier_adjusted.json "
                        f"and consider generating a bridge transform."
                    ),
                })
        return found


# ── Report ───────────────────────────────────────────────────────────────────

def run_detection(memory_dir: str, min_count: int = 3,
                  last_runs: int = 50, dry_run: bool = False) -> list:
    """
    Run all detectors, persist new patterns, return full list.
    """
    mem = BonfyreMemory(memory_dir)
    det = FailureDetector(mem, min_count=min_count, last_runs=last_runs)

    # Merge: from memory queries + extended detectors
    base = det.detect_all()
    extended = det.detect_oscillation() + det.detect_confidence_trend() + \
               det.detect_routing_dead_ends()

    # Deduplicate by (pattern, family): keep highest count
    seen: dict = {}
    for p in base + extended:
        key = (p["pattern"], p.get("family", ""))
        if key not in seen or seen[key]["count"] < p["count"]:
            seen[key] = p

    all_patterns = sorted(seen.values(),
                           key=lambda x: (-x.get("severity", 0), -x["count"]))

    if not dry_run:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for p in all_patterns:
            mem.record_failure(
                pattern=p["pattern"],
                family=p.get("family", "?"),
                count=p["count"],
                detail=p.get("detail", {}),
            )
        # Write consolidated detection report
        report_path = os.path.join(
            memory_dir, "failures",
            f"{ts.replace(':', '-')}_detection_report.json")
        try:
            with open(report_path, "w") as fh:
                json.dump({
                    "detected_at": ts,
                    "min_count": min_count,
                    "last_runs": last_runs,
                    "n_patterns": len(all_patterns),
                    "patterns": all_patterns,
                }, fh, indent=2)
        except OSError:
            pass

    return all_patterns


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Detect failure patterns in Bonfyre transform memory")
    ap.add_argument("--memory-dir", default="/tmp/bonfyre-memory")
    ap.add_argument("--min-count",  type=int, default=3,
                    help="Minimum occurrences before pattern fires (default: 3)")
    ap.add_argument("--last-runs",  type=int, default=50,
                    help="Window size: look at last N runs (default: 50)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Detect but do not write to memory")
    ap.add_argument("--json",       action="store_true",
                    help="Output raw JSON instead of table")
    args = ap.parse_args()

    patterns = run_detection(
        memory_dir=args.memory_dir,
        min_count=args.min_count,
        last_runs=args.last_runs,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(patterns, indent=2))
        return

    if not patterns:
        print(f"[failure_detect] no patterns found "
              f"(min_count={args.min_count}, last {args.last_runs} runs)")
        return

    print(f"\n[failure_detect] {len(patterns)} pattern(s) detected "
          f"(min_count={args.min_count})")
    print()
    print(f"  {'pattern':<14}  {'family':<8}  {'count':>5}  {'sev':>3}  detail")
    print("  " + "─" * 72)
    for p in patterns:
        det = p.get("detail", {})
        det_str = json.dumps(det)[:60]
        print(f"  {p['pattern']:<14}  {p.get('family','?'):<8}  "
              f"{p['count']:>5}  {p.get('severity',0):>3}  {det_str}")
        if "recommendation" in p:
            print(f"  {'':>14}  → {p['recommendation'][:68]}")
    print()
    if not args.dry_run:
        print(f"  patterns written to {args.memory_dir}/failures/")


if __name__ == "__main__":
    main()
