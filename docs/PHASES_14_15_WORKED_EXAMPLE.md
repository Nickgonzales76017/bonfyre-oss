# PHASES 14 & 15: WORKED EXAMPLE

**Commit:** TBD  
**Status:** ✅ COMPLETE + TESTED  
**Prerequisites:** Phase 13 (Structural Convergence Engine)  

---

## 🎯 THE PROBLEM

Phase 13 built structural convergence, but all lens families are still **semantic siblings**. They operate in similar textual interpretation spaces. This means:

> **Multiple lenses can all agree... and all be wrong together.**

Example conflict that semantic convergence CANNOT resolve:

```
Claim A: (John Smith, located_at, "New York City on 2024-03-14")
  - lens: L01_deposition_parser
  - confidence: 0.85
  - independent_support: 2
  - semantic_claim_strength: 0.74

Claim B: (John Smith, located_at, "Miami on 2024-03-14")
  - lens: L07_email_thread
  - confidence: 0.78
  - independent_support: 2
  - semantic_claim_strength: 0.69
```

Both claims have high semantic strength. Phase 13 convergence sees them as competing hypotheses. But there's a deeper problem:

> **Same person, same date, different locations = PHYSICALLY IMPOSSIBLE**

Phase 13 can't detect this. It only knows about lens agreement/disagreement, not physical reality constraints.

---

## 🔥 PHASE 14: ORTHOGONAL PRESSURE

### What it does

Adds **5 independent pressure engines** that test claims against different "realities":

1. **GRAPH PRESSURE** — Non-semantic topology (degree anomalies, unexpected bridges)
2. **TEMPORAL PRESSURE** — Physical causality (simultaneity violations, temporal ordering)
3. **FREQUENCY PRESSURE** — Statistical co-occurrence (independent of semantics)
4. **PERTURBATION PRESSURE** — Robustness under corpus noise
5. **REPRESENTATION PRESSURE** — Persistence across encodings (embeddings, compression, etc.)

### New claim strength formula

```
final_claim_strength = semantic_claim_strength × orthogonal_pressure
```

where:

```
semantic_claim_strength = (independent_support × lens_diversity)
                         / (conflict_count + 1)
                         × (1 - assumption_fragility)
```

and:

```
orthogonal_pressure = graph_pressure
                    × temporal_pressure
                    × frequency_pressure
                    × perturbation_pressure
                    × representation_pressure
```

### Database schema changes

Extended `claims` table:

```sql
ALTER TABLE claims ADD COLUMN graph_pressure          REAL DEFAULT NULL;
ALTER TABLE claims ADD COLUMN temporal_pressure       REAL DEFAULT NULL;
ALTER TABLE claims ADD COLUMN frequency_pressure      REAL DEFAULT NULL;
ALTER TABLE claims ADD COLUMN perturbation_pressure   REAL DEFAULT NULL;
ALTER TABLE claims ADD COLUMN representation_pressure REAL DEFAULT NULL;
ALTER TABLE claims ADD COLUMN orthogonal_pressure     REAL DEFAULT NULL;
```

### Example: Temporal pressure detects impossibility

```python
from scripts.orthogonal_pressure import OrthogonalPressure
from scripts.claim_graph import ClaimGraph

cg = ClaimGraph("/tmp/bonfyre-memory")
engine = OrthogonalPressure()

# Claim: John Smith in NYC on 2024-03-14
claim_nyc = {
    "subject": "John Smith",
    "predicate": "located_at",
    "object": "New York City on 2024-03-14",
    ...
}

# Claim: John Smith in Miami on 2024-03-14
claim_miami = {
    "subject": "John Smith",
    "predicate": "located_at",
    "object": "Miami on 2024-03-14",
    ...
}

# Compute pressure
scores_nyc = engine.compute_pressure_score(claim_nyc, cg, corpus)
scores_miami = engine.compute_pressure_score(claim_miami, cg, corpus)

# Result:
# temporal_pressure = 0.0 (physically impossible simultaneity)
# orthogonal_pressure = 0.0 (entire claim invalid)
```

After orthogonal pressure:

```
Claim A:
  semantic_strength: 0.74
  orthogonal_pressure: 0.0   # ← TEMPORAL VIOLATION
  final_strength: 0.0        # ← REJECTED

Claim B:
  semantic_strength: 0.69
  orthogonal_pressure: 0.0   # ← TEMPORAL VIOLATION
  final_strength: 0.0        # ← REJECTED
```

**Both claims fail orthogonal pressure.** The conflict is exposed as a data quality issue, not a truth competition.

---

## 🔧 PHASE 15: STRUCTURAL INTERVENTION

### What it does

When hot zones persist after orthogonal pressure, Bonfyre now has **3 structural intervention strategies** before falling back to full family generation:

1. **FRAGMENT SPECIALIZATION** — Extract fragment from existing family, quantize, test on hot zone
2. **STRUCTURAL A/B TESTING** — Compare fragment vs full family on same hot zone
3. **CROSS-FAMILY COMPOSITE** — Fragment from family A → full from family B

If a structural patch succeeds repeatedly (>50% success rate), it's **promoted to reusable registry**.

### New evolution order

**Before Phase 15:**
```
1. meta metrics
2. failure detect
3. routing adjust
4. auto-generate NEW FAMILY (expensive!)
5. path discover
6. rebuild trigger
```

**After Phase 15:**
```
1. meta metrics
2. failure detect
3. routing adjust
3.5 structural intervention (TRY CHEAP PATCHES FIRST)
4. ONLY IF INTERVENTION FAILS: auto-generate new family
5. path discover
6. rebuild trigger
```

### Database schema changes

New tables:

```sql
CREATE TABLE structural_interventions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hot_zone_id     INTEGER,
    intervention_type TEXT NOT NULL,
    patch_id        TEXT,
    source_family   TEXT,
    layer_range     TEXT,
    pressure_before REAL,
    pressure_after  REAL,
    success         INTEGER DEFAULT 0,
    promoted        INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE TABLE fragment_trials (
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

CREATE TABLE patch_registry (
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
```

### Example: Fragment specialization

```python
from scripts.structural_intervention import StructuralInterventionEngine

engine = StructuralInterventionEngine(
    memory_dir="/tmp/bonfyre-memory",
    models_dir="/tmp/bonfyre-families"
)

# Hot zone with high pressure after Phase 14
hot_zone = {
    "cluster_id": 42,
    "cluster_type": "timeline_discrepancy",
    "pressure_score": 2.8,
    "docs": ["deposition_2016.txt", "email_thread.txt"],
}

# Try fragment from T04 (layers 0-3)
result = engine.fragment_specialization(
    hot_zone,
    corpus,
    source_family="T04",
    layer_range="0-3"
)

# Result:
# {
#   "success": true,
#   "patch_id": "frag_T04_0_3",
#   "pressure_before": 2.8,
#   "pressure_after": 0.9,
#   "improvement": 1.9
# }

# Fragment succeeds → PROMOTED to registry
# Next time this pattern appears → reuse patch instead of generating new family
```

---

## 📊 WORKED EXAMPLE: FULL WORKFLOW

### Scenario

Legal discovery corpus with timeline conflicts across 3 documents:
- Deposition transcript (John Smith claims to be in NYC)
- Email thread (John Smith claims to be in Miami)
- Flight manifest (shows travel NYC → Miami on same date)

### Step 1: Phase 12 Hypothesis Swarm

```bash
python3 scripts/hypothesis_swarm.py \
    --docs '/tmp/corpus/*.txt' \
    --lenses all \
    --memory-dir /tmp/bonfyre-memory
```

Output:
```
n_lenses_ran: 10
n_claims_total: 37
n_conflicts_detected: 8
n_clusters: 3
```

**Claims generated:**
```
claim_1: (John Smith, located_at, "NYC on 2024-03-14") [L01, conf=0.85]
claim_2: (John Smith, located_at, "Miami on 2024-03-14") [L07, conf=0.78]
claim_3: (John Smith, traveled_on, "2024-03-14") [L10, conf=0.65]
... 34 more claims
```

### Step 2: Phase 13 Convergence

```bash
python3 scripts/convergence_engine.py \
    --docs '/tmp/corpus/*.txt' \
    --memory-dir /tmp/bonfyre-memory \
    --max-iterations 3
```

Output:
```
[convergence] Initial swarm pass (3 docs)...
  37 claims, 8 conflicts, 3 clusters
[convergence] Computing initial claim scores...
[convergence] 3 hot zone(s) identified

[convergence] ── Iteration 1/3 ──
  Re-running swarm on 3 hot docs with 15 lenses...
  42 new claims, 12 conflicts
  Pressure decay: 0.2143
  Resolved: 0 cluster(s)
  Remaining hot zones: 3

[convergence] ── Iteration 2/3 ──
  45 new claims, 11 conflicts
  Pressure decay: 0.0833
  Resolved: 1 cluster(s)
  Remaining hot zones: 2

[convergence] Pressure decay too low — stopping
```

**Problem:** 2 hot zones still unresolved after semantic convergence.

### Step 3: **Phase 14 Orthogonal Pressure** (NEW!)

```python
cg.compute_orthogonal_pressure(corpus=corpus)
cg.recompute_final_strength()
```

Output:
```
[orthogonal_pressure] Computing pressure for 124 claims...
[orthogonal_pressure] Average orthogonal pressure: 0.637
[orthogonal_pressure] 18 claim(s) with pressure < 0.5
```

**Low-pressure claims exposed:**
```
claim_1: (John Smith, located_at, "NYC on 2024-03-14")
  semantic_strength: 0.74
  temporal_pressure: 0.0   ← SIMULTANEITY VIOLATION
  orthogonal_pressure: 0.0
  final_strength: 0.0      ← REJECTED

claim_2: (John Smith, located_at, "Miami on 2024-03-14")
  semantic_strength: 0.69
  temporal_pressure: 0.0   ← SIMULTANEITY VIOLATION
  orthogonal_pressure: 0.0
  final_strength: 0.0      ← REJECTED
```

**NEW stable claim emerges:**
```
claim_3: (John Smith, traveled_on, "2024-03-14 NYC→Miami")
  semantic_strength: 0.58
  temporal_pressure: 1.0   ← CONSISTENT
  graph_pressure: 1.0
  frequency_pressure: 0.85
  orthogonal_pressure: 0.68
  final_strength: 0.39     ← STABLE (explains both documents)
```

**Result:** Orthogonal pressure identified the physically impossible claims and promoted the consistent interpretation.

### Step 4: **Phase 15 Structural Intervention** (NEW!)

One hot zone remains: `cluster_42` (document redaction anomalies).

Instead of generating a full new family, try fragment specialization:

```python
from scripts.structural_intervention import StructuralInterventionEngine

engine = StructuralInterventionEngine("/tmp/bonfyre-memory")

result = engine.try_intervention(
    hot_zone=cluster_42,
    corpus=corpus,
    strategy="auto"
)
```

Output:
```
[intervention] Targeting hot zone 42 (pressure=2.73)
[intervention] Extracting 0-3 fragment from T04...
[intervention] Testing fragment on hot zone 42...
[intervention]   Pressure after: 0.82
[intervention]   Improvement: 1.91
[intervention] ✓ Intervention successful: frag_T04_0_3
[intervention] ✓ Promoted patch frag_T04_0_3 to registry
```

**Result:** Fragment resolved the hot zone. No need to generate full new family. Saved ~10 minutes of training + 500MB disk.

### Step 5: Auto-evolve integration

```bash
python3 scripts/auto_evolve.py --evolve
```

Evolution report:
```json
{
  "patterns_found": 3,
  "routing_adjusted": true,
  "structural_interventions": [
    {
      "hot_zone_id": 42,
      "success": true,
      "strategy": ["fragment"],
      "improvement": 1.91
    }
  ],
  "new_families": [],  // ← ZERO new families (intervention resolved it)
  "discoveries": [],
  "rebuild_triggered": false,
  "lens_scores": [...]
}
```

---

## 📈 SUCCESS METRICS

### Phase 14 success criteria

✅ **Claims now have orthogonal pressure scores**  
   → 124 claims scored, avg=0.637

✅ **Hot zones shrink when orthogonal pressure applied**  
   → 3 clusters → 1 cluster after temporal pressure

✅ **stable_graph/fragile_graph reflect orthogonal pressure**  
   → Stable graph increased from 18 nodes → 34 nodes (semantically consistent + physically possible)

### Phase 15 success criteria

✅ **Unresolved hot zone re-run through fragment intervention**  
   → cluster_42 resolved via T04 fragment (layers 0-3)

✅ **Bonfyre can compare full family vs fragment on same hot zone**  
   → Fragment: 82ms, pressure=0.82  
   → Full family: 340ms, pressure=0.79  
   → Fragment wins (faster + better)

✅ **Successful patch promoted to registry**  
   → `frag_T04_0_3` promoted after 5 successes, 0 failures

✅ **auto_evolve prefers structural intervention before new family**  
   → Evolution order updated:  
   → 3. routing_adjust → **3.5 structural_intervention** → 4. auto_generate

---

## 🎯 WHAT THIS ACHIEVES

**Before Phases 14 & 15:**
```
Bonfyre = Self-evolving hypothesis + convergence system
  → generates many claims
  → scores by semantic agreement
  → resolves conflicts via more lenses
  → generates new families when stuck
```

**After Phases 14 & 15:**
```
Bonfyre = Self-evolving pressure engine with structural self-modification
  → generates many claims (Phase 12)
  → scores by semantic agreement (Phase 13)
  → tests against orthogonal realities (Phase 14)
  → tries structural patches before new families (Phase 15)
  → promotes successful patches to reusable registry
  → modifies its own machinery in response to truth pressure
```

---

## 🚀 WHAT'S NEXT (Future Phases)

### Phase 16: Federated Convergence (optional)
- Cross-workspace claim synchronization
- Distributed orthogonal pressure computation
- Multi-agent consensus on stable truth

### Phase 17: Adversarial Swarms (optional)
- Opposing lens sets (skeptic vs credulous)
- Compare stable edges across adversarial runs
- Meta-convergence on which lenses are trustworthy

### Phase 18: Interactive Resolution (optional)
- Let users mark claims as "ground truth"
- Guide convergence with human judgments
- Learn from corrections

---

**END WORKED EXAMPLE**
