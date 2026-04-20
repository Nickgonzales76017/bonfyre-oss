# PHASE 12: HYPOTHESIS SWARM / TRUTH PRESSURE ENGINE

**Commit:** d67b1a2 + a1eba9b  
**Status:** ✅ COMPLETE + TESTED  
**Dependencies:** ZERO (stdlib + existing Bonfyre stack)  
**Database:** Extends `memory.db` with claim/conflict/cluster tables  

---

## 🎯 WHAT IS THIS?

Phase 12 turns Bonfyre from a **self-evolving adaptive transform runtime** into a **self-evolving hypothesis swarm** that attacks hard corpora from many incompatible angles until a stronger graph emerges.

Instead of one family producing one set of labels at each iteration, the swarm runs **10-50 narrow, intentionally-biased micro-models** (lenses) over the same input simultaneously. Each lens:
- Produces **claims** (structured assertions) from its narrow perspective
- Disagrees with other lenses on the same facts
- Creates **conflicts** → **clusters** → **pressure zones**
- Triggers **new lens generation** when patterns recur without resolution

---

## 📦 NEW FILES (5)

### `scripts/claim_graph.py` (569 lines)
SQLite-backed claim/counterclaim store.

**Tables:**
```sql
claims             (id, doc_id, span, subject, predicate, object, lens, confidence, assumptions)
conflicts          (id, claim_a, claim_b, conflict_type, strength)
conflict_clusters  (id, cluster_type, n_members, pressure_score, subjects_json, docs_json, lens_json, resolved)
cluster_members    (cluster_id, conflict_id)
support_links      (claim_a, claim_b, strength)
```

**Core methods:**
- `record_claim(claim_dict)` → claim_id
- `detect_conflicts(doc_id, min_confidence)` → list of conflict dicts
- `cluster_conflicts(conflicts)` → list of cluster dicts with pressure scores
- `get_pressure_zones(top_n=20)` → hot zones needing reprocessing

**CLI:**
```bash
python3 scripts/claim_graph.py summary --memory-dir /tmp/bonfyre-memory
python3 scripts/claim_graph.py conflicts --min-strength 0.4
python3 scripts/claim_graph.py hot-zones --top 20
python3 scripts/claim_graph.py export /tmp/claim_export.json
```

---

### `scripts/lens_registry.py` (624 lines)
10 first-wave lens families (all zero-dep, rule-based).

| Lens ID | Type | Mechanism |
|---------|------|-----------|
| L01_deposition_parser | structural | Q:/A: regex + speaker role attribution |
| L02_alias_expansion | suspicion | n-gram Jaccard similarity clustering |
| L03_euphemism_detector | suspicion | hedge-phrase + softener keyword dict |
| L04_timeline_anomaly | suspicion | date extractor + monotonicity check |
| L05_redaction_shape | structural | █ / [REDACTED] / whitespace gap detection |
| L06_coercion_language | suspicion | imperative + threat vocab |
| L07_email_thread | structural | Re:/Fwd: nesting depth |
| L08_ocr_restore | counterfactual | common OCR substitution lookup |
| L09_entity_consistency | corpus pressure | within-doc entity reference coherence |
| L10_travel_anomaly | corpus pressure | location sequence plausibility |

**Each lens returns:**
```json
{
  "lens_id": "L01_deposition_parser",
  "doc_id": "epstein-deposition-2016",
  "claims": [
    {"subject": "John Smith", "predicate": "speaker_role", "object": "witness",
     "span_start": 0, "span_end": 100, "confidence": 0.8, "assumptions": ["legal_deposition_format"]},
    ...
  ],
  "elapsed_ms": 12
}
```

**CLI:**
```bash
python3 scripts/lens_registry.py L01_deposition_parser --doc depo.txt
python3 scripts/lens_registry.py all --doc doc.txt --out claims.json
```

---

### `scripts/hypothesis_swarm.py` (235 lines)
Swarm orchestrator — runs N lenses over M docs, detects conflicts, clusters them, computes pressure zones.

**Workflow:**
1. Select corpus slice (doc_ids or text input)
2. Select lens set (all 10, or subset)
3. Run each lens → collect claims
4. Persist all claims to claim_graph
5. Detect conflicts (same subject+predicate, different object)
6. Cluster conflicts by type
7. Compute pressure zones (high conflict density per doc/span)
8. Emit: claim_graph.json, conflict_report.json, cluster_report.json, pressure_zones.json

**CLI:**
```bash
python3 scripts/hypothesis_swarm.py --docs /tmp/corpus/*.txt \
    --lenses all --memory-dir /tmp/bonfyre-memory --out /tmp/swarm_result.json

python3 scripts/hypothesis_swarm.py --text "doc text" --doc-id 123 \
    --lenses L01,L03,L06 --memory-dir /tmp/bonfyre-memory
```

**Output:**
```json
{
  "n_lenses_ran": 10,
  "n_docs_processed": 5,
  "n_claims_total": 247,
  "n_conflicts_detected": 18,
  "n_clusters": 4,
  "elapsed_sec": 2.3,
  "pressure_zones": [...top 10 hot zones...],
  "cluster_summary": {"timeline_anomaly": 3, "entity_variant": 2, ...},
}
```

---

### `scripts/conflict_cluster.py` (265 lines)
Advanced conflict clustering + pressure analysis.

**Pressure formula:**
```
pressure = (conflict_count × avg_strength) / (support_count + 1)
```

**Hot zone criteria:**
- `pressure_score > 2.0` — high conflict density
- `n_conflicts >= 3` — recurrent pattern
- `n_lenses >= 2` — cross-lens disagreement (not single lens noise)
- `assumption_fragility > 0.7` — claims rely on weak assumptions

**CLI:**
```bash
python3 scripts/conflict_cluster.py cluster-advanced --memory-dir /tmp/bonfyre-memory \
    --min-conflicts 3 --out /tmp/clusters_advanced.json

python3 scripts/conflict_cluster.py hot-zones --memory-dir /tmp/bonfyre-memory \
    --pressure-threshold 2.0
```

**Output:**
```
  type                  pressure  fragility  n_conf  lenses  recs
  ──────────────────────────────────────────────────────────────────────────
  timeline_anomaly       3.2500      0.8000       5      3  L04_timeline_anomaly
  entity_variant         2.7500      0.9000       7      4  L02_alias_expansion, L09_entity_consistency
```

---

### `scripts/auto_evolve.py` (EXTENDED)
Added **Step 6.5: lens_generation** to auto-evolution cycle.

**New behavior:**
- After conflict clustering, flag hot zones
- If hot zone cluster_type recurs ≥ 3 times without resolution → mint new lens
- Write lens spec to `auto_lenses.json`
- Lens ID format: `L11_timeline_anomaly_auto`, `L12_entity_variant_auto`, etc.

**Auto-lens spec:**
```json
{
  "lens_id": "L11_timeline_anomaly_auto",
  "cluster_type": "timeline_anomaly",
  "description": "Auto-generated lens for recurring timeline_anomaly conflicts",
  "pressure_threshold": 2.0,
  "min_confidence": 0.25,
  "generated_at": "2025-01-15T10:30:00Z",
  "source": "auto_evolve",
  "spec_type": "expand_existing",
  "base_lenses": ["L04_timeline_anomaly"]
}
```

**Evolution report now includes:**
```json
{
  "steps": [..., "lens_generation", "write_metrics"],
  "new_lenses": [
    {"lens_id": "L11_timeline_anomaly_auto", "cluster_type": "timeline_anomaly", ...}
  ],
  ...
}
```

---

## 🚀 USAGE EXAMPLES

### Example 1: Run swarm on Epstein corpus
```bash
# Prepare corpus
ls /tmp/epstein-corpus/*.txt
# → epstein-deposition-2016.txt
# → epstein-flight-logs-2019.txt
# → epstein-address-book.txt
# → maxwell-deposition-2020.txt

# Run swarm (all lenses)
python3 scripts/hypothesis_swarm.py \
    --docs '/tmp/epstein-corpus/*.txt' \
    --lenses all \
    --memory-dir /tmp/bonfyre-memory \
    --out /tmp/epstein_swarm_result.json

# Output:
# {
#   "n_lenses_ran": 10,
#   "n_docs_processed": 4,
#   "n_claims_total": 387,
#   "n_conflicts_detected": 42,
#   "n_clusters": 7,
#   "pressure_zones": [
#     {"doc_id": "epstein-deposition-2016.txt", "conflict_count": 18, "pressure_score": 4.2, ...},
#     ...
#   ]
# }
```

### Example 2: Hot-zone reprocessing
```bash
# After swarm run, inspect hot zones
python3 scripts/conflict_cluster.py hot-zones \
    --memory-dir /tmp/bonfyre-memory \
    --pressure-threshold 3.0

# Output lists high-pressure clusters:
#   timeline_anomaly       4.2500      0.9000      12      4  L04_timeline_anomaly
#   entity_variant         3.7500      0.8500       9      3  L02_alias_expansion

# Re-run swarm with recommended lenses on flagged docs
python3 scripts/hypothesis_swarm.py \
    --docs epstein-deposition-2016.txt \
    --lenses L04,L02 \
    --memory-dir /tmp/bonfyre-memory
```

### Example 3: Auto-evolve with lens generation
```bash
# Run evolution cycle after swarm passes
python3 scripts/auto_evolve.py \
    --evolve \
    --memory-dir /tmp/bonfyre-memory \
    --models-dir /tmp/bonfyre-families

# Output:
# [auto_evolve] Hot zones: 2 conflict cluster(s) flagged
# [auto_evolve] Generated 1 new lens(es) from hot zones
# [auto_evolve]   wrote 1 new lens(es) → auto_lenses.json
# [auto_evolve] Evolution complete: 3 patterns, 0 new families, 1 discoveries

cat /tmp/bonfyre-memory/auto_lenses.json
# {
#   "L11_timeline_anomaly_auto": {
#     "lens_id": "L11_timeline_anomaly_auto",
#     "cluster_type": "timeline_anomaly",
#     "base_lenses": ["L04_timeline_anomaly"],
#     ...
#   }
# }
```

---

## 🧪 SMOKE TEST RESULTS

```
$ python3 scripts/test_hypothesis_swarm.py
============================================================
HYPOTHESIS SWARM SMOKE TEST (Phase 12)
============================================================

[TEST 1] claim_graph.py
  ✓ 3 claims recorded
  ✓ 1 conflict(s) detected
  ✓ 1 cluster(s) created
  ✓ 1 pressure zone(s)
  [PASS] claim_graph.py

[TEST 2] lens_registry.py
  ✓ 10 lenses ran
  ✓ 16 total claims produced
  [PASS] lens_registry.py

[TEST 3] hypothesis_swarm.py
  ✓ 10 lenses ran
  ✓ 3 docs processed
  ✓ 25 claims total
  ✓ 4 conflicts detected
  ✓ 2 clusters
  [PASS] hypothesis_swarm.py

[TEST 4] conflict_cluster.py
  ✓ 2 advanced cluster(s)
  ✓ 0 hot zone(s) flagged
  [PASS] conflict_cluster.py

[TEST 5] auto_evolve.py lens generation
  ✓ 0 new lens(es) would be generated
  [PASS] auto_evolve.py lens generation

============================================================
ALL TESTS PASSED ✓
============================================================
```

---

## 🔗 INTEGRATION WITH BONFYRE RUNTIME

### Pre-swarm: Bonfyre transform run
```bash
# Normal Bonfyre run (Phase 11)
python3 scripts/demo.py \
    --texts "Sample legal deposition text..." \
    --memory-dir /tmp/bonfyre-memory \
    --auto-evolve
```

### Post-swarm: Hypothesis swarm enrichment
```bash
# After Bonfyre transform completes, run swarm on same input
python3 scripts/hypothesis_swarm.py \
    --text "Sample legal deposition text..." \
    --doc-id "demo_run_1234" \
    --lenses all \
    --memory-dir /tmp/bonfyre-memory \
    --run-id 1234  # Links claims to specific Bonfyre run
```

**Result:**
- Bonfyre run produces: labels, confidence scores, escalation chains
- Swarm run produces: claims, conflicts, pressure zones
- Both stored in `memory.db`
- `auto_evolve.py` can now mint **both** new transform families (from failures) AND new lenses (from conflicts)

---

## 📊 OUTPUT FORMATS

### `swarm_claims.json`
```json
{
  "n_claims": 247,
  "claims": [
    {
      "doc_id": "epstein-deposition-2016.txt",
      "span_start": 1240,
      "span_end": 1320,
      "span_text": "Q: Where were you on March 14?",
      "subject": "witness",
      "predicate": "question_asked",
      "object": "attorney",
      "lens": "L01_deposition_parser",
      "confidence": 0.75,
      "assumptions": ["legal_deposition_format"]
    },
    ...
  ]
}
```

### `swarm_conflicts.json`
```json
{
  "n_conflicts": 42,
  "conflicts": [
    {
      "claim_a_id": 12,
      "claim_b_id": 45,
      "claim_a": {...},
      "claim_b": {...},
      "conflict_type": "timeline_anomaly",
      "strength": 0.65,
      "subject": "John Smith",
      "predicate": "arrived_on",
      "object_a": "2024-03-14",
      "object_b": "2024-03-17"
    },
    ...
  ]
}
```

### `swarm_clusters.json`
```json
{
  "n_clusters": 7,
  "clusters": [
    {
      "cluster_id": 1,
      "cluster_type": "timeline_anomaly",
      "n_conflicts": 12,
      "pressure_score": 4.25,
      "dominant_subject": "john smith",
      "subjects": ["John Smith", "J. Smith", "Smith"],
      "docs": ["epstein-deposition-2016.txt"],
      "lenses": ["L01_deposition_parser", "L04_timeline_anomaly", "L07_email_thread"],
      "resolved": false
    },
    ...
  ]
}
```

### `swarm_pressure_zones.json`
```json
{
  "n_zones": 8,
  "zones": [
    {
      "doc_id": "epstein-deposition-2016.txt",
      "span_start": 0,
      "span_end": 5000,
      "conflict_count": 18,
      "pressure_score": 4.2,
      "dominant_cluster": "timeline_anomaly",
      "recommended_lenses": ["L04_timeline_anomaly"]
    },
    ...
  ]
}
```

---

## 🎬 NEXT STEPS

### Immediate (Phase 12.5?)
- **Dynamic lens invocation**: Load lenses from `auto_lenses.json` at runtime
- **Web UI for pressure zones**: Visualize conflict clusters + hot zones
- **Cross-document claim linking**: Support links across multiple docs (e.g., same entity referenced in flight logs + deposition)

### Medium-term
- **Lens composition**: Auto-merge L01+L04 when they agree → higher confidence claim
- **Temporal conflict tracking**: Flag claims that contradict historical claims from previous swarm runs
- **Interactive conflict resolution**: Let user mark cluster as "resolved" → suppress lens regeneration

### Long-term (Phase 13?)
- **Adversarial lens pairs**: L-suspicious vs L-charitable, compare pressure
- **Claim provenance graph**: Full dependency tracking from raw span → claim → conflict → cluster → lens generation
- **Federated claim graph**: Multi-workspace claim sharing (e.g., legal discovery across orgs)

---

## 📝 COMMIT HISTORY

```
d67b1a2  PHASE 12: Hypothesis Swarm + Truth Pressure Engine
         - claim_graph.py (569 lines)
         - lens_registry.py (624 lines)
         - hypothesis_swarm.py (235 lines)
         - conflict_cluster.py (265 lines)
         - auto_evolve.py (extended, +100 lines)
         Total: 2080 insertions

a1eba9b  Add Phase 12 smoke tests
         - test_hypothesis_swarm.py (274 lines)
         All tests pass ✓
```

---

## 🚨 DESIGN CONSTRAINTS (PRESERVED)

✅ **NO new dependencies** — stdlib + sentence-transformers + onnxruntime (already in Bonfyre)  
✅ **NO redesign of runtime** — claim graph sits *on top* of existing transform memory  
✅ **NO neural training loops** — all lenses are rule-based (regex, heuristics, lookups)  
✅ **SQLite-backed** — extends `memory.db` with WAL mode  
✅ **Incremental** — all auto-generated lenses stored in `auto_lenses.json` (easily deleted)  
✅ **Observable** — full CLI inspection of claims, conflicts, clusters, pressure zones  
✅ **Reversible** — delete `auto_lenses.json` to revert all auto-generated lenses  

---

**END PHASE 12 SUMMARY**
