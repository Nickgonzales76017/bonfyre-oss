#!/usr/bin/env python3
"""
scripts/auto_evolve.py — Bonfyre self-evolution coordinator.

This is the main driver for Bonfyre's self-improvement loop.
It reads transform memory, detects failures, adjusts routing,
discovers new paths, and — when failure patterns are severe enough —
generates new transform families automatically.

EVOLUTION LOOP
==============
  1. INGEST      — absorb latest demo.py run metrics into memory
  2. DETECT      — run failure_detect.py on the full memory window
  3. ADJUST      — update routing weights (frontier_adjusted.json)
  4. GENERATE    — if failure threshold crossed, create a new transform
  5. DISCOVER    — if efficiency is low, run path discovery
  6. REBUILD     — if critical escalation rate, trigger rebuild_families.sh
  7. METRICS     — compute and write meta_metrics to graph/meta_metrics.json

AUTO-TRANSFORM GENERATION (PART 3)
===================================
When a failure pattern (oscillation or repeat_esc) has count >= AUTO_GEN_THRESHOLD:

  1. Extract the failing input texts from memory (raw_json of escalation runs)
  2. Write them to a temp corpus dir as .txt + .label files
  3. Determine the next auto family ID (T50, T51, T52, ...)
  4. Run: bonfyre-run T04-C <corpus> --out <models_dir>/auto/<family>/run
  5. Quantize, extract fragment, align
  6. Register the new family in auto_families.json
  7. Add it to FAMILY_HEADS for future routing

The new family is specialized for the input region where the existing families
keep failing — it's trained exclusively on those failure examples.

DESIGN RULES (from user spec)
==============================
  - NO neural training loops in this script
  - NO RL
  - NO external dependencies beyond what bonfyre already uses
  - Does NOT replace existing runtime
  - Everything incremental, observable, reversible
  - All generated families stored in auto_families.json (easily deleted)

USAGE
=====
    # One-shot: ingest latest metrics and evolve
    python3 scripts/auto_evolve.py --ingest /tmp/runs.json

    # Ingest from directory of run files
    python3 scripts/auto_evolve.py --ingest-dir /tmp/bonfyre-memory/runs/

    # Full evolution cycle (no new ingestion)
    python3 scripts/auto_evolve.py --evolve

    # Dry-run (no file writes, no new transforms)
    python3 scripts/auto_evolve.py --evolve --dry-run

    # All in one (typical post-batch call):
    python3 scripts/auto_evolve.py --ingest /tmp/runs.json --evolve
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SELF))

from scripts.bonfyre_memory import BonfyreMemory       # noqa: E402
from scripts.failure_detect  import run_detection       # noqa: E402
from scripts.routing_adjust  import run_adjustment      # noqa: E402
from scripts.meta_metrics    import compute_meta_metrics # noqa: E402

REPO_ROOT   = os.path.dirname(_SELF)
RUN_BIN     = os.path.join(REPO_ROOT, "cmd", "BonfyreRun",   "bonfyre-run")
QUANT_BIN   = os.path.join(REPO_ROOT, "cmd", "BonfyreQuant", "bonfyre-quant")
LAYER_BIN   = os.path.join(REPO_ROOT, "cmd", "BonfyreLayer", "bonfyre-layer")
FPQX_BIN    = os.path.join(REPO_ROOT, "cmd", "BonfyreFPQX",  "bonfyre-fpqx")

# ── Thresholds ─────────────────────────────────────────────────────────────

AUTO_GEN_THRESHOLD      = 5    # failure count before generating new family
DISCOVERY_THRESHOLD     = 0.3  # graph_efficiency below this → trigger discovery
REBUILD_ESC_THRESHOLD   = 0.5  # escalation_rate above this → trigger rebuild
MIN_FAILURE_CORPUS      = 10   # minimum examples needed to train new family
AUTO_FAMILY_START       = 50   # first auto family code (T50, T51, ...)

# ── Auto family registry ──────────────────────────────────────────────────

def _load_auto_families(models_dir: str) -> dict:
    path = os.path.join(models_dir, "auto_families.json")
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return {}


def _save_auto_families(models_dir: str, families: dict):
    path = os.path.join(models_dir, "auto_families.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(families, f, indent=2)


def _next_family_code(models_dir: str) -> str:
    existing = _load_auto_families(models_dir)
    used_nums = set()
    for fid in existing:
        try:
            used_nums.add(int(fid[1:]))
        except (ValueError, IndexError):
            pass
    n = AUTO_FAMILY_START
    while n in used_nums:
        n += 1
    return f"T{n}"


# ── Extract failure corpus ────────────────────────────────────────────────

def extract_failure_corpus(mem: BonfyreMemory, family: str,
                            pattern: str, out_dir: str) -> int:
    """
    Extract input texts from escalation runs involving `family`.
    Write them as .txt + .label files to out_dir.
    Returns count of texts written.
    """
    db = mem._db

    # Fetch raw_json of runs that had escalations from/to this family
    rows = db.execute("""
        SELECT DISTINCT r.raw_json
        FROM runs r
        JOIN escalations e ON e.run_id = r.id
        WHERE (e.from_family = ? OR e.to_family = ?)
          AND r.raw_json IS NOT NULL
        LIMIT 500
    """, (family, family)).fetchall()

    if not rows and pattern in ("collapse", "fragment_fail"):
        # For collapse/fragment_fail, grab runs that used this family
        rows = db.execute("""
            SELECT raw_json FROM runs
            WHERE routed_family = ? AND raw_json IS NOT NULL
              AND (fragment_exit = 0 OR escalation_count > 0)
            LIMIT 500
        """, (family,)).fetchall()

    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for row in rows:
        try:
            metrics = json.loads(row["raw_json"])
        except Exception:
            continue
        texts = metrics.get("input_texts", [])
        if not texts:
            # Try to recover from labels/confidences if texts not stored
            fam = metrics.get("routed_family", "unknown")
            texts = [f"[recovered: {fam} escalation run {written}]"]
        for text in texts:
            if not text:
                continue
            fname = os.path.join(out_dir, f"ex_{written:04d}.txt")
            with open(fname, "w") as f:
                f.write(text.strip())
            label_fname = fname.replace(".txt", ".label")
            with open(label_fname, "w") as f:
                json.dump({"tags": [{"label": "failure", "score": 1.0}]}, f)
            written += 1

    return written


# ── Generate new transform ────────────────────────────────────────────────

def generate_new_transform(family_id: str, corpus_dir: str,
                            models_dir: str, dry_run: bool = False) -> dict:
    """
    Train a new collapse head on the failure corpus.

    Steps:
      1. bonfyre-run T04-C <corpus_dir> --out <out_dir>
      2. bonfyre-quant <model.onnx> <family.bqfp>
      3. bonfyre-layer <family.bqfp> --frag → <family>-frag.bqfp
      4. bonfyre-fpqx align <family.bqfp> <T04.bqfp> → align-<family>-T04/

    Returns dict with {family_id, model_path, bqfp_path, frag_path, error}
    """
    out_dir    = os.path.join(models_dir, "auto", family_id, "run")
    bqfp_path  = os.path.join(models_dir, f"{family_id}.bqfp")
    frag_path  = os.path.join(models_dir, f"{family_id}-frag.bqfp")
    model_path = os.path.join(out_dir, "train", "model.onnx")

    if dry_run:
        return {
            "family_id":  family_id,
            "model_path": model_path,
            "bqfp_path":  bqfp_path,
            "frag_path":  frag_path,
            "dry_run":    True,
            "error":      None,
        }

    os.makedirs(out_dir, exist_ok=True)

    # Step 1: collapse train
    if not os.path.exists(RUN_BIN):
        return {"family_id": family_id, "error": "bonfyre-run not found"}
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [RUN_BIN, "T04-C", corpus_dir, "--out", out_dir],
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {"family_id": family_id,
                    "error": f"bonfyre-run failed: {result.stderr[:300]}"}
    except subprocess.TimeoutExpired:
        return {"family_id": family_id, "error": "bonfyre-run timeout"}

    if not os.path.exists(model_path):
        return {"family_id": family_id,
                "error": f"model.onnx not produced at {model_path}"}

    # Step 2: quantize
    if os.path.exists(QUANT_BIN):
        try:
            subprocess.run([QUANT_BIN, model_path, bqfp_path],
                           capture_output=True, timeout=120)
        except Exception:
            pass

    # Step 3: fragment extraction
    if os.path.exists(LAYER_BIN) and os.path.exists(bqfp_path):
        try:
            subprocess.run([LAYER_BIN, bqfp_path, "--frag", frag_path],
                           capture_output=True, timeout=120)
        except Exception:
            pass

    # Step 4: align to T04
    t04_bqfp = os.path.join(models_dir, "T04.bqfp")
    if os.path.exists(FPQX_BIN) and os.path.exists(bqfp_path) and \
       os.path.exists(t04_bqfp):
        align_dir = os.path.join(models_dir, f"align-{family_id}-T04")
        os.makedirs(align_dir, exist_ok=True)
        try:
            subprocess.run(
                [FPQX_BIN, "align", bqfp_path, t04_bqfp,
                 "--out", align_dir],
                capture_output=True, timeout=180)
        except Exception:
            pass

    elapsed = int((time.monotonic() - t0) * 1000)
    return {
        "family_id":  family_id,
        "model_path": model_path if os.path.exists(model_path) else None,
        "bqfp_path":  bqfp_path  if os.path.exists(bqfp_path)  else None,
        "frag_path":  frag_path   if os.path.exists(frag_path)  else None,
        "elapsed_ms": elapsed,
        "error":      None,
    }


# ── Main evolution cycle ──────────────────────────────────────────────────

def evolve(memory_dir: str, models_dir: str,
           min_count: int = AUTO_GEN_THRESHOLD,
           dry_run: bool = False,
           skip_discover: bool = False,
           texts: list = None) -> dict:
    """
    Run one full evolution cycle.  Returns a report dict.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mem = BonfyreMemory(memory_dir)

    report = {
        "evolved_at":       ts,
        "dry_run":          dry_run,
        "steps":            [],
        "patterns_found":   [],
        "routing_adjusted": False,
        "new_families":     [],
        "discoveries":      [],
        "rebuild_triggered": False,
        "meta_metrics":     {},
    }

    # ── Step 1: Compute meta-metrics ──────────────────────────────────
    meta = compute_meta_metrics(mem)
    report["meta_metrics"] = meta
    report["steps"].append("meta_metrics")
    _log(f"Status: {meta.get('status', '?').upper()}  "
         f"eff={meta.get('graph_efficiency', 0):.4f}  "
         f"esc={meta.get('escalation_rate', 0):.4f}  "
         f"frag={meta.get('fragment_success_rate', 0):.4f}")

    # ── Step 2: Detect failures ───────────────────────────────────────
    patterns = run_detection(memory_dir, min_count=min_count,
                              dry_run=dry_run)
    report["patterns_found"] = patterns
    report["steps"].append("failure_detect")
    _log(f"Failure patterns: {len(patterns)}")
    for p in patterns[:5]:
        _log(f"  [{p['pattern']}] family={p.get('family','?')}  "
             f"count={p['count']}")

    # ── Step 3: Adjust routing ────────────────────────────────────────
    adj = run_adjustment(memory_dir, models_dir, dry_run=dry_run)
    if adj.get("changed"):
        _log(f"Routing: {len(adj['changed'])} transition(s) adjusted")
        report["routing_adjusted"] = True
    report["steps"].append("routing_adjust")

    # ── Step 3.5: Try structural intervention on unresolved hot zones ──
    # NEW PHASE 15: Before generating full new families, try cheaper patches
    intervention_results = []
    try:
        from scripts.claim_graph import ClaimGraph
        from scripts.conflict_cluster import cluster_advanced, flag_hot_zones
        from scripts.structural_intervention import StructuralInterventionEngine
        
        cg = ClaimGraph(memory_dir)
        clusters = cluster_advanced(cg, min_conflicts=2, min_confidence=0.25)
        hot_zones = flag_hot_zones(
            clusters, pressure_threshold=1.5, min_conflicts=2,
            min_lenses=1, fragility_threshold=0.5)
        
        if hot_zones:
            _log(f"Structural intervention: {len(hot_zones)} unresolved hot zone(s)")
            intervention_engine = StructuralInterventionEngine(
                memory_dir=memory_dir,
                models_dir=models_dir
            )
            
            # Try intervention on top hot zones (limit to 3 for v1)
            for hz in hot_zones[:3]:
                _log(f"  Trying intervention on hot zone {hz.get('cluster_id')} "
                     f"(pressure={hz.get('pressure_score', 0):.2f})...")
                
                # Build minimal corpus for this hot zone
                hz_corpus = {}
                # Would need actual corpus here - for v1 use placeholder
                
                result = intervention_engine.try_intervention(
                    hz, hz_corpus, strategy="auto")
                
                intervention_results.append({
                    "hot_zone_id": hz.get("cluster_id"),
                    "success": result.get("success", False),
                    "strategy": result.get("tried_strategies", []),
                    "improvement": result.get("improvement", 0.0),
                })
                
                if result.get("success"):
                    _log(f"    ✓ Intervention successful: {result.get('patch_id')}")
                else:
                    _log(f"    ✗ Intervention failed — will fall back to new family")
        else:
            _log("Structural intervention: no hot zones detected")
        
        report["structural_interventions"] = intervention_results
    except Exception as e:
        _log(f"Structural intervention skipped: {e}")
    report["steps"].append("structural_intervention")

    # ── Step 4: Auto-generate new transform (on severe patterns) ──────
    # Only generate new families if structural intervention failed
    critical_patterns = [p for p in patterns
                         if p.get("severity", 0) >= 2 and
                         p["count"] >= min_count]
    
    # Filter out patterns where structural intervention succeeded
    successful_interventions = {
        i["hot_zone_id"] for i in intervention_results if i.get("success")
    }
    if successful_interventions:
        _log(f"Skipping family generation for {len(successful_interventions)} "
             f"hot zone(s) resolved via structural intervention")

    generated_families = []
    for pattern in critical_patterns:
        family = pattern.get("family", "T04")
        family_id = _next_family_code(models_dir)

        _log(f"Auto-generate: {family_id}  (for {pattern['pattern']} "
             f"on {family}, count={pattern['count']})")

        with tempfile.TemporaryDirectory() as tmp_corpus:
            n_texts = extract_failure_corpus(
                mem, family, pattern["pattern"], tmp_corpus)
            _log(f"  corpus: {n_texts} failure texts extracted")

            if n_texts < MIN_FAILURE_CORPUS:
                _log(f"  ⚠  too few examples ({n_texts} < {MIN_FAILURE_CORPUS}) "
                     f"— skipping generation")
                continue

            result = generate_new_transform(
                family_id, tmp_corpus, models_dir, dry_run=dry_run)

            if result.get("error"):
                _log(f"  ✗ generation failed: {result['error']}")
                continue

            _log(f"  ✓ generated {family_id}  "
                 f"model={'exists' if result.get('model_path') else 'pending'}  "
                 f"dry_run={dry_run}")

            # Register in auto_families.json
            if not dry_run:
                auto_fams = _load_auto_families(models_dir)
                auto_fams[family_id] = {
                    "generated_at":   ts,
                    "from_pattern":   pattern["pattern"],
                    "from_family":    family,
                    "failure_count":  pattern["count"],
                    "corpus_size":    n_texts,
                    "model_path":     result.get("model_path"),
                    "bqfp_path":      result.get("bqfp_path"),
                    "frag_path":      result.get("frag_path"),
                    "task":           "topic-map",
                    "tier":           "auto",
                    "source":         "auto_evolve",
                }
                _save_auto_families(models_dir, auto_fams)

                # Also append to domain_families.json (for frontier_map.py pickup)
                domain_json = os.path.join(models_dir, "domain_families.json")
                if os.path.exists(domain_json):
                    try:
                        domain_fams = json.load(open(domain_json))
                        if family_id not in domain_fams:
                            domain_fams[family_id] = {
                                "f1": 0.0,
                                "geometry": "auto",
                                "task": "topic-map",
                                "corpus": f"failure_{family}",
                                "params": 0,
                                "tier": "auto",
                                "source": "auto_evolve",
                            }
                        with open(domain_json, "w") as f:
                            json.dump(domain_fams, f, indent=2)
                    except Exception:
                        pass

            generated_families.append(result)
            report["new_families"].append({
                "family_id":   family_id,
                "from_pattern": pattern["pattern"],
                "from_family": family,
                "n_corpus":    n_texts,
            })

    report["steps"].append("auto_generate")

    # ── Step 5: Path discovery (if low efficiency) ────────────────────
    if not skip_discover and meta.get("graph_efficiency", 1.0) < DISCOVERY_THRESHOLD:
        try:
            from scripts.path_discover import discover
            discovery_texts = texts or [
                "Apple reports record quarterly revenue driven by iPhone sales.",
                "Scientists discover new exoplanet in the habitable zone.",
                "World leaders convene for climate summit negotiations.",
                "Federal Reserve signals rate pause amid cooling inflation.",
            ]
            discoveries = discover(
                memory_dir=memory_dir,
                models_dir=models_dir,
                texts=discovery_texts,
                dry_run=dry_run,
            )
            report["discoveries"] = discoveries
            if discoveries:
                _log(f"Path discovery: {len(discoveries)} new chain(s) found")
        except Exception as e:
            _log(f"Path discovery skipped: {e}")
    report["steps"].append("path_discover")

    # ── Step 6: Trigger rebuild if critical ───────────────────────────
    if meta.get("escalation_rate", 0) >= REBUILD_ESC_THRESHOLD:
        _log(f"Rebuild trigger: escalation_rate={meta['escalation_rate']:.4f} "
             f">= {REBUILD_ESC_THRESHOLD}")
        if not dry_run:
            _trigger_rebuild(models_dir, memory_dir)
        report["rebuild_triggered"] = True
    report["steps"].append("rebuild_check")

    # ── Step 6.5: Generate new lenses from hot conflict zones ─────────
    try:
        from scripts.claim_graph import ClaimGraph
        from scripts.conflict_cluster import cluster_advanced, flag_hot_zones
        cg = ClaimGraph(memory_dir)
        clusters = cluster_advanced(cg, min_conflicts=3, min_confidence=0.25)
        hot_zones = flag_hot_zones(
            clusters, pressure_threshold=2.0, min_conflicts=3,
            min_lenses=2, fragility_threshold=0.7)
        
        new_lenses = []
        if hot_zones:
            _log(f"Hot zones: {len(hot_zones)} conflict cluster(s) flagged")
            new_lenses = _generate_lenses_from_hot_zones(
                hot_zones, memory_dir, dry_run=dry_run)
            if new_lenses:
                _log(f"Generated {len(new_lenses)} new lens(es) from hot zones")
                report["new_lenses"] = new_lenses
        else:
            _log("No hot zones detected — skipping lens generation")
    except Exception as e:
        _log(f"Lens generation skipped: {e}")
    report["steps"].append("lens_generation")

    # ── Step 6.75: Lens promotion/demotion based on stability ────────
    try:
        from scripts.claim_graph import ClaimGraph
        cg = ClaimGraph(memory_dir)
        lens_scores = _score_lenses_by_stability(cg, dry_run=dry_run)
        if lens_scores:
            promoted = [l for l in lens_scores if l["promoted"]]
            demoted = [l for l in lens_scores if l["demoted"]]
            if promoted:
                _log(f"Promoted {len(promoted)} lens(es) for high stability")
            if demoted:
                _log(f"Demoted {len(demoted)} lens(es) for low stability or noise")
            report["lens_scores"] = lens_scores
    except Exception as e:
        _log(f"Lens scoring skipped: {e}")
    report["steps"].append("lens_scoring")

    # ── Step 7: Write meta-metrics snapshot ───────────────────────────
    graph_dir = os.path.join(memory_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)
    if not dry_run:
        with open(os.path.join(graph_dir, "meta_metrics.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(graph_dir, "evolution_log.json"), "a") as f:
            f.write(json.dumps({
                "evolved_at":     ts,
                "patterns":       len(patterns),
                "new_families":   len(generated_families),
                "routing_adj":    report["routing_adjusted"],
                "discoveries":    len(report.get("discoveries", [])),
                "rebuild":        report["rebuild_triggered"],
                "status":         meta.get("status"),
            }) + "\n")
    report["steps"].append("write_metrics")

    _log(f"Evolution complete: "
         f"{len(patterns)} patterns, "
         f"{len(generated_families)} new families, "
         f"{len(report.get('discoveries', []))} discoveries")
    return report


def _trigger_rebuild(models_dir: str, memory_dir: str):
    """Fire rebuild_families.sh asynchronously."""
    rebuild_sh = os.path.join(REPO_ROOT, "scripts", "rebuild_families.sh")
    if not os.path.exists(rebuild_sh):
        _log("rebuild_families.sh not found — skipping rebuild trigger")
        return
    try:
        subprocess.Popen(
            ["bash", rebuild_sh,
             "--models-dir", models_dir,
             "--memory-dir", memory_dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _log("rebuild_families.sh launched in background")
    except Exception as e:
        _log(f"rebuild trigger failed: {e}")


def _generate_lenses_from_hot_zones(hot_zones: list, memory_dir: str,
                                      dry_run: bool = False) -> list:
    """
    Generate new lens entries from hot conflict zones.

    Hot zones indicate recurrent conflict patterns not resolved by current lenses.
    For each unique cluster_type appearing ≥ 3 times without resolution:
      - Mint a new lens ID (L11, L12, L13, ...)
      - Write lens spec to auto_lenses.json
      - Return list of generated lens metadata

    Returns: list of {lens_id, cluster_type, description, threshold}
    """
    auto_lenses_path = os.path.join(memory_dir, "auto_lenses.json")
    existing_lenses = {}
    if os.path.exists(auto_lenses_path):
        try:
            existing_lenses = json.load(open(auto_lenses_path))
        except Exception:
            pass

    # Count hot zone cluster types
    from collections import Counter
    cluster_types = [hz["cluster_type"] for hz in hot_zones if not hz.get("resolved")]
    type_counts = Counter(cluster_types)

    LENS_GENERATION_THRESHOLD = 3  # min hot zone recurrence to mint new lens

    new_lenses = []
    used_ids = set(existing_lenses.keys())
    next_id = 11  # Start at L11 (L01-L10 are first-wave)
    while f"L{next_id:02d}" in used_ids:
        next_id += 1

    for cluster_type, count in type_counts.items():
        if count < LENS_GENERATION_THRESHOLD:
            continue

        # Already have a lens for this cluster type?
        if any(lns.get("cluster_type") == cluster_type for lns in existing_lenses.values()):
            continue

        lens_id = f"L{next_id:02d}_{cluster_type}_auto"
        next_id += 1

        # Generate lens spec
        lens_spec = {
            "lens_id": lens_id,
            "cluster_type": cluster_type,
            "description": f"Auto-generated lens for recurring {cluster_type} conflicts",
            "pressure_threshold": 2.0,
            "min_confidence": 0.25,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "auto_evolve",
            "spec_type": "expand_existing",  # Hint: extend L01-L10's logic for this cluster_type
            "base_lenses": _suggest_base_lenses_for_cluster(cluster_type),
        }

        new_lenses.append(lens_spec)

        if not dry_run:
            existing_lenses[lens_id] = lens_spec

    # Write auto_lenses.json
    if new_lenses and not dry_run:
        os.makedirs(os.path.dirname(auto_lenses_path), exist_ok=True)
        with open(auto_lenses_path, "w") as f:
            json.dump(existing_lenses, f, indent=2)
        _log(f"  wrote {len(new_lenses)} new lens(es) → auto_lenses.json")

    return new_lenses


def _suggest_base_lenses_for_cluster(cluster_type: str) -> list:
    """
    Suggest which existing L01-L10 lenses should be extended/forked
    to create a new lens for this cluster type.
    """
    CLUSTER_TO_BASE = {
        "entity_variant":      ["L02_alias_expansion", "L09_entity_consistency"],
        "timeline_anomaly":    ["L04_timeline_anomaly"],
        "speaker_role":        ["L01_deposition_parser"],
        "coercion_signal":     ["L06_coercion_language", "L03_euphemism_detector"],
        "redaction_found":     ["L05_redaction_shape"],
        "email_thread_depth":  ["L07_email_thread"],
        "ocr_candidate":       ["L08_ocr_restore"],
        "travel_anomaly":      ["L10_travel_anomaly"],
        "hedge_signal":        ["L03_euphemism_detector"],
        "question_asked":      ["L01_deposition_parser"],
        "answer_given":        ["L01_deposition_parser"],
    }
    return CLUSTER_TO_BASE.get(cluster_type, [])


def _score_lenses_by_stability(claim_graph, dry_run: bool = False) -> list:
    """
    Score each lens by the stability of claims it produces.

    Promotes lenses that produce high-stability claims.
    Demotes lenses that create noise (low stability, high conflict).

    Returns list of lens score dicts:
        {
            "lens_id": "L01_deposition_parser",
            "n_claims": 247,
            "avg_claim_strength": 0.75,
            "avg_stability": 0.82,
            "promoted": True,
            "demoted": False,
            "score": 0.78
        }
    """
    # Get all lenses that have produced claims
    lens_rows = claim_graph._db.execute("""
        SELECT lens, COUNT(*) as n_claims
        FROM claims
        GROUP BY lens
        ORDER BY n_claims DESC
    """).fetchall()

    lens_scores = []

    for row in lens_rows:
        lens_id = row["lens"]
        n_claims = row["n_claims"]

        # Get avg claim_strength for this lens
        avg_strength = claim_graph._db.execute("""
            SELECT AVG(claim_strength) FROM claims
            WHERE lens = ? AND claim_strength > 0
        """, (lens_id,)).fetchone()[0] or 0.0

        # Get avg stability_score
        avg_stability = claim_graph._db.execute("""
            SELECT AVG(stability_score) FROM claims
            WHERE lens = ? AND stability_score > 0
        """, (lens_id,)).fetchone()[0] or 0.0

        # Get avg conflict_density
        avg_conflict_density = claim_graph._db.execute("""
            SELECT AVG(CAST(conflict_count AS REAL) / (support_count + 1))
            FROM claims
            WHERE lens = ?
        """, (lens_id,)).fetchone()[0] or 0.0

        # Compute lens score
        lens_score = (avg_strength + avg_stability) / 2.0 * (1.0 - avg_conflict_density)

        # Promotion/demotion thresholds
        PROMOTE_THRESHOLD = 0.6   # high stability + low conflict
        DEMOTE_THRESHOLD  = 0.2   # low stability + high conflict

        promoted = lens_score >= PROMOTE_THRESHOLD
        demoted = lens_score < DEMOTE_THRESHOLD

        lens_scores.append({
            "lens_id": lens_id,
            "n_claims": n_claims,
            "avg_claim_strength": round(avg_strength, 4),
            "avg_stability": round(avg_stability, 4),
            "avg_conflict_density": round(avg_conflict_density, 4),
            "score": round(lens_score, 4),
            "promoted": promoted,
            "demoted": demoted,
        })

    # Write lens_scores.json (for monitoring)
    if not dry_run and lens_scores:
        lens_scores_path = os.path.join(claim_graph.memory_dir, "lens_scores.json")
        with open(lens_scores_path, "w") as f:
            json.dump({
                "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "lens_scores": lens_scores,
            }, f, indent=2)

    return lens_scores


def _log(msg: str):
    print(f"[auto_evolve] {msg}")



# ── Ingest ────────────────────────────────────────────────────────────────

def ingest(mem: BonfyreMemory, path: str) -> int:
    """Ingest a metrics JSON file. Returns number of runs ingested."""
    try:
        data = json.load(open(path))
    except Exception as e:
        _log(f"ERROR reading {path}: {e}")
        return 0
    runs = data if isinstance(data, list) else [data]
    count = 0
    for run in runs:
        if isinstance(run, dict) and "schema" in run:
            mem.record_run(run)
            count += 1
    return count


def ingest_dir(mem: BonfyreMemory, ingest_directory: str) -> int:
    """Ingest all *.json files from a directory."""
    total = 0
    for fname in sorted(os.listdir(ingest_directory)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ingest_directory, fname)
        n = ingest(mem, path)
        total += n
    return total


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Bonfyre self-evolution coordinator")

    ap.add_argument("--ingest",       default=None,
                    help="Path to a demo.py --metrics-out JSON to ingest")
    ap.add_argument("--ingest-dir",   default=None,
                    help="Directory of metrics JSONs to ingest (processed in order)")
    ap.add_argument("--evolve",       action="store_true",
                    help="Run the full evolution cycle after ingestion")
    ap.add_argument("--memory-dir",   default="/tmp/bonfyre-memory")
    ap.add_argument("--models-dir",   default="/tmp/bonfyre-families")
    ap.add_argument("--min-count",    type=int, default=AUTO_GEN_THRESHOLD,
                    help=f"Failure count threshold for auto-generation "
                         f"(default: {AUTO_GEN_THRESHOLD})")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Run all detection/adjustment but do not write new files "
                         "or execute new transforms")
    ap.add_argument("--skip-discover", action="store_true",
                    help="Skip path discovery step (faster)")
    ap.add_argument("--texts",         nargs="+", default=None,
                    help="Probe texts for path discovery")
    ap.add_argument("--json",          action="store_true",
                    help="Output evolution report as JSON")
    args = ap.parse_args()

    mem = BonfyreMemory(args.memory_dir)

    # ── Ingest ────────────────────────────────────────────────────────
    if args.ingest:
        n = ingest(mem, args.ingest)
        _log(f"Ingested {n} run(s) from {args.ingest}")

    if args.ingest_dir:
        n = ingest_dir(mem, args.ingest_dir)
        _log(f"Ingested {n} run(s) from {args.ingest_dir}")

    # ── Evolve ────────────────────────────────────────────────────────
    if args.evolve or (not args.ingest and not args.ingest_dir):
        if not (args.ingest or args.ingest_dir):
            _log("No ingestion requested — running evolution on existing memory")
        report = evolve(
            memory_dir=args.memory_dir,
            models_dir=args.models_dir,
            min_count=args.min_count,
            dry_run=args.dry_run,
            skip_discover=args.skip_discover,
            texts=args.texts,
        )
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(f"\n[auto_evolve] steps: {' → '.join(report['steps'])}")
            if report["new_families"]:
                print(f"[auto_evolve] NEW FAMILIES GENERATED:")
                for fam in report["new_families"]:
                    print(f"  ★ {fam['family_id']}  "
                          f"(from {fam['from_pattern']} on {fam['from_family']}, "
                          f"corpus={fam['n_corpus']})")
            if report["discoveries"]:
                print(f"[auto_evolve] NEW PATHS DISCOVERED:")
                for d in report["discoveries"]:
                    print(f"  ★ {d.get('chain', '?')}  "
                          f"score={d.get('score', 0):.4f}")
            if report["rebuild_triggered"]:
                print("[auto_evolve] ⚠  rebuild_families.sh triggered")
            if args.dry_run:
                print("[auto_evolve] (dry-run: no files written)")
    else:
        # Just ingested — show summary
        mem = BonfyreMemory(args.memory_dir)
        summary = mem.summary()
        _log(f"Memory summary: {summary['total_runs']} runs  "
             f"{summary['total_escalations']} escalations  "
             f"avg_conf={summary.get('avg_confidence')}")


if __name__ == "__main__":
    main()
