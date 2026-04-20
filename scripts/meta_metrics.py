#!/usr/bin/env python3
"""
scripts/meta_metrics.py — Bonfyre graph-level efficiency metrics.

Reads BonfyreMemory and computes higher-order metrics that describe the
health and efficiency of the transform graph as a whole — not individual runs.

METRICS
=======
  graph_efficiency       avg_confidence / avg_path_length across all runs
                         Higher = better: same confidence in fewer hops
  avg_path_length        mean iterations_ran across all non-fragment-exit runs
  escalation_rate        escalation_count > 0 runs / total_runs
  fragment_success_rate  fragment_exit=1 runs / total_runs
  new_transform_usage    runs using auto-generated families / total_runs
  confidence_velocity    rate of change of avg_confidence over time (runs/day)
  discovery_rate         new paths discovered per N runs
  oscillation_score      oscillation events / total_runs (lower = better)
  routing_entropy        entropy of family distribution (high = diverse/exploratory)
  collapse_pressure      families with confirmed collapse / total_families

USAGE
=====
    python3 scripts/meta_metrics.py [--memory-dir /tmp/bonfyre-memory]
                                     [--json]
                                     [--watch N]     # re-run every N seconds

    # From auto_evolve.py:
    from scripts.meta_metrics import compute_meta_metrics
    m = compute_meta_metrics(mem)
    if m["escalation_rate"] > 0.4:
        trigger_rebuild()
"""

import argparse
import json
import math
import os
import sys
import time

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))

from scripts.bonfyre_memory import BonfyreMemory  # noqa: E402


# ── Metric thresholds (used by auto_evolve.py) ────────────────────────────

ESCALATION_RATE_WARN   = 0.30   # escalation in > 30% of runs → concern
ESCALATION_RATE_CRIT   = 0.50   # escalation in > 50% of runs → trigger rebuild
FRAGMENT_SUCCESS_MIN   = 0.20   # fragment should succeed at least 20% of runs
GRAPH_EFFICIENCY_WARN  = 0.15   # below this → graph is thrashing
VELOCITY_CONCERN       = -0.01  # avg_confidence dropping > 1% per run → concern


def compute_meta_metrics(mem: BonfyreMemory) -> dict:
    """
    Compute all meta-metrics from the transform memory.

    Returns a flat dict of metric_name → value with flags for anomalies.
    """
    db = mem._db

    # ── Base counts ───────────────────────────────────────────────────
    total_runs = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    if total_runs == 0:
        return {
            "total_runs": 0,
            "status": "insufficient_data",
            "message": "No runs in memory yet. Run demo.py with --memory-dir to start.",
        }

    esc_runs = db.execute(
        "SELECT COUNT(*) FROM runs WHERE escalation_count > 0").fetchone()[0]
    frag_runs = db.execute(
        "SELECT COUNT(*) FROM runs WHERE fragment_exit = 1").fetchone()[0]

    avg_conf_row = db.execute(
        "SELECT AVG(avg_confidence) FROM runs WHERE avg_confidence IS NOT NULL"
    ).fetchone()[0]
    avg_conf = avg_conf_row or 0.0

    avg_iters_row = db.execute(
        "SELECT AVG(iterations_ran) FROM runs "
        "WHERE iterations_ran IS NOT NULL AND fragment_exit = 0"
    ).fetchone()[0]
    avg_iters = max(avg_iters_row or 1.0, 1.0)

    # ── graph_efficiency ──────────────────────────────────────────────
    graph_efficiency = round(avg_conf / avg_iters, 4)

    # ── escalation_rate ───────────────────────────────────────────────
    escalation_rate = round(esc_runs / total_runs, 4)

    # ── fragment_success_rate ─────────────────────────────────────────
    fragment_success_rate = round(frag_runs / total_runs, 4)

    # ── confidence_velocity (per run, over last 20 runs) ─────────────
    recent = db.execute("""
        SELECT avg_confidence, run_at FROM runs
        WHERE avg_confidence IS NOT NULL
        ORDER BY run_at DESC LIMIT 20
    """).fetchall()

    velocity = 0.0
    if len(recent) >= 4:
        # Simple slope across recent window (index 0 = newest)
        cs = [r["avg_confidence"] for r in recent]
        n = len(cs)
        xs = list(range(n))
        x_m = sum(xs) / n
        y_m = sum(cs) / n
        num = sum((xs[i] - x_m) * (cs[i] - y_m) for i in range(n))
        den = sum((xs[i] - x_m) ** 2 for i in range(n))
        # Negative slope (index 0 = newest) → confidence is decreasing with time
        velocity = round(-num / den if den > 0 else 0.0, 5)

    # ── new_transform_usage (auto-generated families) ─────────────────
    # Auto families live in domain_families.json with metadata source="auto_evolve"
    auto_family_ids = _get_auto_family_ids(mem)
    if auto_family_ids:
        placeholder = ",".join("?" * len(auto_family_ids))
        auto_run_count = db.execute(
            f"SELECT COUNT(*) FROM runs WHERE routed_family IN ({placeholder})",
            auto_family_ids
        ).fetchone()[0]
    else:
        auto_run_count = 0
    new_transform_usage = round(auto_run_count / total_runs, 4)

    # ── discovery_rate (paths per run) ───────────────────────────────
    path_count = db.execute("SELECT COUNT(*) FROM paths").fetchone()[0]
    discovery_rate = round(path_count / total_runs, 4)

    # ── oscillation_score ────────────────────────────────────────────
    osc_events = db.execute("""
        SELECT COUNT(*) FROM failures WHERE pattern = 'oscillation'
    """).fetchone()[0]
    oscillation_score = round(osc_events / total_runs, 4)

    # ── routing_entropy (Shannon entropy of family distribution) ──────
    family_counts = db.execute("""
        SELECT routed_family, COUNT(*) as n
        FROM runs WHERE routed_family IS NOT NULL
        GROUP BY routed_family
    """).fetchall()
    routing_entropy = 0.0
    if family_counts:
        total_fam = sum(r["n"] for r in family_counts)
        for r in family_counts:
            p = r["n"] / total_fam
            if p > 0:
                routing_entropy -= p * math.log2(p)
        routing_entropy = round(routing_entropy, 4)

    # ── collapse_pressure ─────────────────────────────────────────────
    collapse_families = db.execute("""
        SELECT COUNT(DISTINCT family) FROM failures WHERE pattern = 'collapse'
    """).fetchone()[0]
    known_families = max(len(family_counts), 1)
    collapse_pressure = round(collapse_families / known_families, 4)

    # ── avg_path_length ───────────────────────────────────────────────
    avg_path_length = round(avg_iters, 2)

    # ── Anomaly flags ─────────────────────────────────────────────────
    flags = []
    if escalation_rate > ESCALATION_RATE_CRIT:
        flags.append({"flag": "escalation_critical",
                      "value": escalation_rate,
                      "threshold": ESCALATION_RATE_CRIT,
                      "action": "trigger_rebuild"})
    elif escalation_rate > ESCALATION_RATE_WARN:
        flags.append({"flag": "escalation_warning",
                      "value": escalation_rate,
                      "threshold": ESCALATION_RATE_WARN,
                      "action": "run_failure_detect"})

    if graph_efficiency < GRAPH_EFFICIENCY_WARN:
        flags.append({"flag": "efficiency_low",
                      "value": graph_efficiency,
                      "threshold": GRAPH_EFFICIENCY_WARN,
                      "action": "run_path_discover"})

    if fragment_success_rate < FRAGMENT_SUCCESS_MIN and total_runs > 10:
        flags.append({"flag": "fragment_underperforming",
                      "value": fragment_success_rate,
                      "threshold": FRAGMENT_SUCCESS_MIN,
                      "action": "rebuild_fragment"})

    if velocity < VELOCITY_CONCERN:
        flags.append({"flag": "confidence_declining",
                      "value": velocity,
                      "threshold": VELOCITY_CONCERN,
                      "action": "run_failure_detect"})

    # ── Final metric dict ─────────────────────────────────────────────
    return {
        "schema":                "bonfyre-meta-v1",
        "computed_at":           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runs":            total_runs,
        # Core graph metrics
        "graph_efficiency":      graph_efficiency,
        "avg_path_length":       avg_path_length,
        "escalation_rate":       escalation_rate,
        "fragment_success_rate": fragment_success_rate,
        "new_transform_usage":   new_transform_usage,
        # Velocity + discovery
        "confidence_velocity":   velocity,
        "discovery_rate":        discovery_rate,
        # Health indicators
        "oscillation_score":     oscillation_score,
        "routing_entropy":       routing_entropy,
        "collapse_pressure":     collapse_pressure,
        # Derived
        "avg_confidence":        round(avg_conf, 4),
        "n_auto_families":       len(auto_family_ids),
        # Anomaly flags (list of dicts with flag/value/threshold/action)
        "flags":                 flags,
        "status":                "healthy" if not flags else "degraded",
    }


def _get_auto_family_ids(mem: BonfyreMemory) -> list:
    """Return list of family IDs that were auto-generated by auto_evolve.py."""
    # Check auto_families.json (written by auto_evolve.py)
    auto_json = "/tmp/bonfyre-families/auto_families.json"
    if os.path.exists(auto_json):
        try:
            data = json.load(open(auto_json))
            return list(data.keys())
        except Exception:
            pass
    return []


# ── Report formatting ─────────────────────────────────────────────────────

def print_report(m: dict):
    print(f"\n{'='*60}")
    print(f" BONFYRE META-METRICS  ({m.get('computed_at', '')})")
    print(f"{'='*60}")
    print(f"  status           : {m.get('status', '?').upper()}")
    print(f"  total_runs       : {m.get('total_runs', 0)}")
    print()
    print(f"  graph_efficiency      : {m.get('graph_efficiency', 0):.4f}"
          f"  (conf / avg_path_len)")
    print(f"  avg_path_length       : {m.get('avg_path_length', 0):.2f}  iterations")
    print(f"  avg_confidence        : {m.get('avg_confidence', 0):.4f}")
    print(f"  escalation_rate       : {m.get('escalation_rate', 0):.4f}"
          f"  ({m.get('escalation_rate', 0)*100:.1f}% of runs)")
    print(f"  fragment_success_rate : {m.get('fragment_success_rate', 0):.4f}")
    print(f"  new_transform_usage   : {m.get('new_transform_usage', 0):.4f}"
          f"  (n_auto={m.get('n_auto_families', 0)})")
    print(f"  confidence_velocity   : {m.get('confidence_velocity', 0):+.5f}"
          f"  per run")
    print(f"  discovery_rate        : {m.get('discovery_rate', 0):.4f}"
          f"  paths/run")
    print(f"  oscillation_score     : {m.get('oscillation_score', 0):.4f}")
    print(f"  routing_entropy       : {m.get('routing_entropy', 0):.4f}  bits")
    print(f"  collapse_pressure     : {m.get('collapse_pressure', 0):.4f}")
    print()

    flags = m.get("flags", [])
    if flags:
        print(f"  ⚠  {len(flags)} anomaly flag(s):")
        for fl in flags:
            print(f"     [{fl['flag']}]  value={fl['value']}  "
                  f"threshold={fl['threshold']}  → {fl['action']}")
    else:
        print("  ✓  no anomalies detected")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Bonfyre graph-level meta-metrics")
    ap.add_argument("--memory-dir", default="/tmp/bonfyre-memory")
    ap.add_argument("--json",       action="store_true",
                    help="Output raw JSON")
    ap.add_argument("--watch",      type=int, default=0,
                    help="Re-run every N seconds (0 = run once)")
    ap.add_argument("--out",        default=None,
                    help="Write metrics JSON to this path")
    args = ap.parse_args()

    def run_once():
        mem = BonfyreMemory(args.memory_dir)
        m = compute_meta_metrics(mem)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(m, f, indent=2)
        if args.json:
            print(json.dumps(m, indent=2))
        else:
            print_report(m)
        return m

    if args.watch > 0:
        while True:
            run_once()
            time.sleep(args.watch)
    else:
        run_once()


if __name__ == "__main__":
    main()
