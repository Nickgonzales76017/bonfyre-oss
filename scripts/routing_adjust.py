#!/usr/bin/env python3
"""
scripts/routing_adjust.py — Dynamic routing weight adjuster.

Reads the BonfyreMemory routing_weights table and produces an
adjusted frontier JSON file that demo.py can use to bias routing.

HOW IT WORKS
============
Bonfyre's bonfyre-model route command reads `frontier.json` which contains
cosine-distance weights between family embedding centroids.  routing_adjust.py
computes a "confidence modifier" for each transition based on historical
success/failure/escalation rates, then writes:

    <memory_dir>/graph/frontier_adjusted.json

This is a modified copy of the original frontier.json with bias weights
applied to transitions that have a strong history.  The original frontier.json
is never modified (reversible).

ADJUSTMENT FORMULA
==================
For each transition (A → B):
  - success_rate = n_success / (n_success + n_failure)
  - escalated_rate = n_escalated / n_total
  - adjustment = success_rate * (1 - escalated_rate) * WEIGHT_SCALE
  - final_weight = base_cosine * (1 + adjustment - 0.5) * WEIGHT_SCALE

  Where WEIGHT_SCALE is bounded: no transition can be boosted > 2x or
  penalized below 0.0 (so the graph never becomes disconnected).

USAGE
=====
    python3 scripts/routing_adjust.py [--memory-dir /tmp/bonfyre-memory]
                                       [--models-dir /tmp/bonfyre-families]
                                       [--scale 0.3]
                                       [--dry-run]

    # In demo.py workflow:
    python3 scripts/routing_adjust.py
    python3 scripts/demo.py "text" --frontier /tmp/bonfyre-memory/graph/frontier_adjusted.json
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


# ── Constants ─────────────────────────────────────────────────────────────

WEIGHT_SCALE     = 0.3   # max fractional change per transition
MIN_TOTAL_RUNS   = 3     # don't adjust until at least this many runs
MAX_BOOST        = 2.0   # cap: never boost a weight more than 2x
MIN_WEIGHT       = 0.05  # floor: keep every edge non-zero (graph stays connected)


# ── Adjust ────────────────────────────────────────────────────────────────

def compute_adjustments(mem: BonfyreMemory) -> dict:
    """
    Return per-transition weight multipliers.

    {
      "T04→T15": 1.25,   # boost: this path has high success
      "T04→T16": 0.65,   # penalize: this path escalates too often
      ...
    }
    """
    adj = mem.get_routing_adjustments()
    multipliers = {}
    for transition, stats in adj.items():
        if stats["n_total"] < MIN_TOTAL_RUNS:
            continue
        sr = stats["success_rate"]
        er = stats["escalated_rate"]
        # success_rate contributes: above 0.5 → boost, below → penalize
        # escalated_rate contributes: high escalation → penalize
        raw = sr * (1.0 - er)
        # Center around 1.0: [0..1] → [1-WEIGHT_SCALE .. 1+WEIGHT_SCALE]
        m = 1.0 + (raw - 0.5) * 2.0 * WEIGHT_SCALE
        m = min(MAX_BOOST, max(MIN_WEIGHT, m))
        multipliers[transition] = round(m, 4)
    return multipliers


def apply_adjustments(frontier: dict, multipliers: dict) -> dict:
    """
    Apply per-transition multipliers to a frontier.json structure.

    frontier.json schema (from bonfyre-fpqx):
    {
      "families": ["T04", "T15", ...],
      "pairs": [
        {"a": "T04", "b": "T15", "cosine_mean": 0.924, "bias": 1.0},
        ...
      ]
    }
    """
    import copy
    adjusted = copy.deepcopy(frontier)

    pairs = adjusted.get("pairs", [])
    changed = []
    for pair in pairs:
        fa = pair.get("a", "")
        fb = pair.get("b", "")
        key_ab = f"{fa}→{fb}"
        key_ba = f"{fb}→{fa}"
        m = multipliers.get(key_ab) or multipliers.get(key_ba)
        if m is None:
            continue
        old_bias = pair.get("bias", 1.0)
        new_bias = round(old_bias * m, 4)
        new_bias = min(MAX_BOOST, max(MIN_WEIGHT, new_bias))
        pair["bias"] = new_bias
        pair["_memory_multiplier"] = m
        changed.append((key_ab, old_bias, new_bias))

    adjusted["_routing_adjust"] = {
        "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transitions_adjusted": len(changed),
        "weight_scale":  WEIGHT_SCALE,
    }
    return adjusted, changed


def rebuild_escalation_chain(mem: BonfyreMemory,
                              base_chain: dict) -> dict:
    """
    Return a routing chain dict updated with learned preferences.

    base_chain: {"T04": "T15", "T15": "T16", "T16": None, ...}

    Rules:
      - If a transition has escalated_rate > 0.8 AND n_total >= MIN_TOTAL_RUNS,
        skip one level in the chain (try the next escalation target instead).
      - If T04→T16 has better success_rate than T04→T15, promote T16 to first
        escalation target.
    """
    adj = mem.get_routing_adjustments()
    chain = dict(base_chain)

    # Collect all first-target pairs (A→B where chain[A]=B)
    for from_f, to_f in list(base_chain.items()):
        if to_f is None:
            continue
        key = f"{from_f}→{to_f}"
        stats = adj.get(key)
        if stats and stats["n_total"] >= MIN_TOTAL_RUNS:
            if stats["escalated_rate"] > 0.8:
                # Skip one level: find to_f's next target
                skip_to = base_chain.get(to_f)
                if skip_to is not None:
                    chain[from_f] = skip_to

    # Detect if any direct-to-deeper target outperforms the current first hop
    families = list(set(base_chain.keys()))
    for from_f in families:
        current_target = chain.get(from_f)
        skip_target = base_chain.get(current_target) if current_target else None
        if skip_target is None:
            continue
        key_curr = f"{from_f}→{current_target}"
        key_skip = f"{from_f}→{skip_target}"
        sc = adj.get(key_curr, {}).get("success_rate", 0.5)
        ss = adj.get(key_skip, {}).get("success_rate", 0.5)
        nc = adj.get(key_curr, {}).get("n_total", 0)
        ns = adj.get(key_skip, {}).get("n_total", 0)
        if nc >= MIN_TOTAL_RUNS and ns >= MIN_TOTAL_RUNS and ss > sc + 0.15:
            chain[from_f] = skip_target  # skip the underperforming intermediate

    return chain


# ── Main ──────────────────────────────────────────────────────────────────

def run_adjustment(memory_dir: str, models_dir: str,
                   scale: float = WEIGHT_SCALE,
                   dry_run: bool = False) -> dict:
    """
    Compute and write frontier_adjusted.json.

    Returns: {"multipliers": {...}, "changed": [...], "out_path": str}
    """
    mem = BonfyreMemory(memory_dir)
    adj = compute_adjustments(mem)

    if not adj:
        print("[routing_adjust] no data yet — nothing to adjust")
        return {"multipliers": {}, "changed": [], "out_path": None}

    # Load base frontier.json
    frontier_path = os.path.join(models_dir, "frontier.json")
    if not os.path.exists(frontier_path):
        # Build a minimal frontier stub from routing_weights data
        pairs = []
        for key in adj:
            parts = key.split("→")
            if len(parts) == 2:
                pairs.append({"a": parts[0], "b": parts[1],
                               "cosine_mean": 0.9, "bias": 1.0})
        frontier = {
            "generated_at": "stub",
            "families": list({p for k in adj for p in k.split("→")}),
            "pairs": pairs,
        }
    else:
        frontier = json.load(open(frontier_path))

    adjusted_frontier, changed = apply_adjustments(frontier, adj)

    # Update escalation chain
    base_chain = {"T04": "T15", "T15": "T16", "T16": None,
                  "S01": "S02", "S02": None}
    learned_chain = rebuild_escalation_chain(mem, base_chain)

    adjusted_frontier["_learned_escalation_chain"] = learned_chain

    out_path = os.path.join(memory_dir, "graph", "frontier_adjusted.json")
    chain_path = os.path.join(memory_dir, "graph", "escalation_chain.json")

    if not dry_run:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(adjusted_frontier, f, indent=2)
        with open(chain_path, "w") as f:
            json.dump({
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "base_chain":    base_chain,
                "learned_chain": learned_chain,
            }, f, indent=2)

    return {
        "multipliers": adj,
        "changed":     changed,
        "out_path":    out_path if not dry_run else None,
        "chain_path":  chain_path if not dry_run else None,
        "learned_chain": learned_chain,
    }


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Adjust Bonfyre routing weights from transform memory")
    ap.add_argument("--memory-dir",  default="/tmp/bonfyre-memory")
    ap.add_argument("--models-dir",  default="/tmp/bonfyre-families")
    ap.add_argument("--scale",       type=float, default=WEIGHT_SCALE,
                    help=f"Max fractional weight change (default: {WEIGHT_SCALE})")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Compute adjustments but do not write files")
    ap.add_argument("--json",        action="store_true",
                    help="Output raw JSON result")
    args = ap.parse_args()

    result = run_adjustment(
        memory_dir=args.memory_dir,
        models_dir=args.models_dir,
        scale=args.scale,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps({
            "multipliers": result["multipliers"],
            "changed_count": len(result.get("changed", [])),
            "learned_chain": result.get("learned_chain"),
        }, indent=2))
        return

    m = result["multipliers"]
    if not m:
        print("[routing_adjust] no routing history yet — nothing to adjust")
        return

    print(f"\n[routing_adjust] weight multipliers ({len(m)} transitions):")
    print(f"  {'transition':<16}  {'multiplier':>10}  direction")
    print("  " + "─" * 44)
    for key, mult in sorted(m.items(), key=lambda x: -abs(x[1] - 1.0)):
        direction = "▲ boost" if mult > 1.0 else ("▼ penalize" if mult < 1.0 else "= neutral")
        print(f"  {key:<16}  {mult:>10.4f}  {direction}")

    if result.get("changed"):
        print(f"\n  frontier_adjusted.json updated: {len(result['changed'])} pair(s) changed")
        print(f"  → {result['out_path']}")

    lc = result.get("learned_chain", {})
    if lc:
        print(f"\n[routing_adjust] learned escalation chain:")
        for k, v in lc.items():
            print(f"  {k} → {v}")
        if result.get("chain_path"):
            print(f"  → {result['chain_path']}")

    if args.dry_run:
        print("\n  (dry-run: no files written)")


if __name__ == "__main__":
    main()
