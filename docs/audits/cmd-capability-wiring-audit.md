# Bonfyre Cmd Capability + Wiring Audit

Generated: 2026-04-22T04:01:54Z

## Inventory Summary
- Cmd directories: 71
- Binaries with main.c: 71
- Binaries with explicit help/usage markers: 71

## Cmd → Target Mapping

| Cmd Dir | Target | Has main.c | Has help markers |
|---|---|---:|---:|
| BonfyreAPI | bonfyre-api | 1 | 1 |
| BonfyreAuth | bonfyre-auth | 1 | 1 |
| BonfyreBrief | bonfyre-brief | 1 | 1 |
| BonfyreCanon | bonfyre-canon | 1 | 1 |
| BonfyreCapability | bonfyre-capability | 1 | 1 |
| BonfyreCLI | bonfyre | 1 | 1 |
| BonfyreClips | bonfyre-clips | 1 | 1 |
| BonfyreCMS | bonfyre-cms | 1 | 1 |
| BonfyreCompete | bonfyre-compete | 1 | 1 |
| BonfyreCompress | bonfyre-compress | 1 | 1 |
| BonfyreControl | bonfyre-control | 1 | 1 |
| BonfyreDistribute | (no TARGET var) | 1 | 1 |
| BonfyreEconomy | bonfyre-economy | 1 | 1 |
| BonfyreEmbed | bonfyre-embed | 1 | 1 |
| BonfyreEmit | bonfyre-emit | 1 | 1 |
| BonfyreEntity | bonfyre-entity | 1 | 1 |
| BonfyreFinance | bonfyre-finance | 1 | 1 |
| BonfyreFlow | bonfyre-flow | 1 | 1 |
| BonfyreFPQ | (no TARGET var) | 1 | 1 |
| BonfyreFPQx | bonfyre-fpqx | 1 | 1 |
| BonfyreGate | bonfyre-gate | 1 | 1 |
| BonfyreGen | bonfyre-gen | 1 | 1 |
| BonfyreGraph | bonfyre-graph | 1 | 1 |
| BonfyreHash | bonfyre-hash | 1 | 1 |
| BonfyreIndex | bonfyre-index | 1 | 1 |
| BonfyreIngest | bonfyre-ingest | 1 | 1 |
| BonfyreKVCache | bonfyre-kvcache | 1 | 1 |
| BonfyreLayer | bonfyre-layer-c | 1 | 1 |
| BonfyreLearn | bonfyre-learn | 1 | 1 |
| BonfyreLedger | bonfyre-ledger | 1 | 1 |
| BonfyreMediaPrep | bonfyre-media-prep | 1 | 1 |
| BonfyreMeter | bonfyre-meter | 1 | 1 |
| BonfyreMFADict | bonfyre-mfa-dict | 1 | 1 |
| BonfyreModel | bonfyre-model | 1 | 1 |
| BonfyreMoQ | bonfyre-moq | 1 | 1 |
| BonfyreNarrate | bonfyre-narrate | 1 | 1 |
| BonfyreOffer | (no TARGET var) | 1 | 1 |
| BonfyreOrchestrate | (no TARGET var) | 1 | 1 |
| BonfyreOutreach | bonfyre-outreach | 1 | 1 |
| BonfyrePack | (no TARGET var) | 1 | 1 |
| BonfyreParagraph | bonfyre-paragraph | 1 | 1 |
| BonfyrePay | bonfyre-pay | 1 | 1 |
| BonfyrePipeline | bonfyre-pipeline | 1 | 1 |
| BonfyreProject | (no TARGET var) | 1 | 1 |
| BonfyreProof | (no TARGET var) | 1 | 1 |
| BonfyreProxy | bonfyre-proxy | 1 | 1 |
| BonfyreQuant | bonfyre-quant | 1 | 1 |
| BonfyreQuery | bonfyre-query | 1 | 1 |
| BonfyreQueue | (no TARGET var) | 1 | 1 |
| BonfyreRecipe | bonfyre-recipe | 1 | 1 |
| BonfyreRender | (no TARGET var) | 1 | 1 |
| BonfyreRepurpose | bonfyre-repurpose | 1 | 1 |
| BonfyreRun | bonfyre-run | 1 | 1 |
| BonfyreRuntime | (no TARGET var) | 1 | 1 |
| BonfyreSegment | bonfyre-segment | 1 | 1 |
| BonfyreSLI | bonfyre-sli | 1 | 1 |
| BonfyreSpace | bonfyre-space | 1 | 1 |
| BonfyreSpeechLoop | bonfyre-speechloop | 1 | 1 |
| BonfyreStitch | bonfyre-stitch | 1 | 1 |
| BonfyreSwarm | bonfyre-swarm | 1 | 1 |
| BonfyreSync | bonfyre-sync | 1 | 1 |
| BonfyreTag | bonfyre-tag | 1 | 1 |
| BonfyreTel | bonfyre-tel | 1 | 1 |
| BonfyreTier | bonfyre-tier | 1 | 1 |
| BonfyreTime | bonfyre-time | 1 | 1 |
| BonfyreTone | bonfyre-tone | 1 | 1 |
| BonfyreTranscribe | bonfyre-transcribe | 1 | 1 |
| BonfyreTranscriptClean | bonfyre-transcript-clean | 1 | 1 |
| BonfyreTranscriptFamily | (no TARGET var) | 1 | 1 |
| BonfyreVec | bonfyre-vec | 1 | 1 |
| BonfyreWeaviateIndex | bonfyre-weaviate-index | 1 | 1 |

## Cross-Binary Invocation Edges

Format: source -> target (count)

- BonfyreAPI ->  (10)
- BonfyreAuth ->  (18)
- BonfyreBrief ->  (1)
- BonfyreCanon ->  (11)
- BonfyreCapability ->  (31)
- BonfyreCLI ->  (52)
- BonfyreClips ->  (6)
- BonfyreCMS ->  (70)
- BonfyreCompete ->  (25)
- BonfyreCompress ->  (11)
- BonfyreControl ->  (51)
- BonfyreDistribute ->  (4)
- BonfyreEconomy ->  (20)
- BonfyreEmbed ->  (2)
- BonfyreEmit ->  (6)
- BonfyreEntity ->  (27)
- BonfyreFinance ->  (31)
- BonfyreFlow ->  (30)
- BonfyreFPQ ->  (20)
- BonfyreFPQx ->  (18)
- BonfyreGate ->  (8)
- BonfyreGen ->  (24)
- BonfyreGraph ->  (24)
- BonfyreHash ->  (10)
- BonfyreIndex ->  (18)
- BonfyreIngest ->  (2)
- BonfyreKVCache ->  (13)
- BonfyreLayer ->  (40)
- BonfyreLearn ->  (21)
- BonfyreLedger ->  (9)
- BonfyreMediaPrep ->  (6)
- BonfyreMeter ->  (9)
- BonfyreMFADict ->  (1)
- BonfyreModel ->  (62)
- BonfyreMoQ ->  (7)
- BonfyreNarrate ->  (15)
- BonfyreOffer ->  (1)
- BonfyreOrchestrate ->  (15)
- BonfyreOutreach ->  (16)
- BonfyrePack ->  (5)
- BonfyreParagraph ->  (1)
- BonfyrePay ->  (17)
- BonfyrePipeline ->  (28)
- BonfyreProject ->  (11)
- BonfyreProof ->  (3)
- BonfyreProxy ->  (22)
- BonfyreQuant ->  (17)
- BonfyreQuery ->  (11)
- BonfyreQueue ->  (21)
- BonfyreRecipe ->  (113)
- BonfyreRender ->  (9)
- BonfyreRepurpose ->  (9)
- BonfyreRun ->  (31)
- BonfyreRuntime ->  (50)
- BonfyreSegment ->  (8)
- BonfyreSLI ->  (36)
- BonfyreSpace ->  (23)
- BonfyreSpeechLoop ->  (15)
- BonfyreStitch ->  (20)
- BonfyreSwarm ->  (14)
- BonfyreSync ->  (4)
- BonfyreTag ->  (11)
- BonfyreTel ->  (42)
- BonfyreTier ->  (24)
- BonfyreTime ->  (28)
- BonfyreTone ->  (9)
- BonfyreTranscribe ->  (4)
- BonfyreTranscriptClean ->  (1)
- BonfyreTranscriptFamily ->  (8)
- BonfyreVec ->  (15)
- BonfyreWeaviateIndex ->  (2)

## High-Value Missing Wiring (Repo-Wide)

1. Runtime orchestration coverage is still thin versus cmd surface.
- Currently wired in runtime: queue, ledger, run/run-ledger, loop, parallel, pipeline, gen, swarm, control, conference, autowire.
- Missing wrappers for core ops binaries: model, recipe, run, stitch, proxy, sli, graph, index, finance, cms, capability, economy, flow, learn, compete, entity.

2. Control-plane feedback loops are not consistently connected.
- autowire runs control score/ops only post local/swarm launch; it does not persist control decision links to swarm dispatch IDs.
- bonfyre-run and bonfyre-runtime do not currently auto-invoke entropy-check before execution.

3. Model and swarm are partially connected but not fully closed-loop.
- bonfyre-model references bonfyre-swarm pull paths, but swarm does not expose pull/push model artifact endpoints.
- Missing consistent CAS hash handoff contract between model pull and swarm dispatch worker pathing.

4. Stitch op mapping includes legacy misroutes.
- In stitch op map, Clean maps to bonfyre-transcribe (likely should map to bonfyre-transcript-clean or bonfyre-paragraph depending intent).
- MetadataEmit maps to bonfyre-brief; likely wants bonfyre-emit in many paths.

5. Service entrypoints are fragmented.
- bonfyre-proxy provides OpenAI-compatible entrypoint but runtime does not expose proxy lifecycle commands.
- bonfyre-moq is present now, but no shared auth/session wiring with bonfyre-auth / bonfyre-pay / bonfyre-meter.

6. Build-target declaration is inconsistent across cmd tree.
- Multiple cmd Makefiles have no TARGET var, reducing discoverability/automation reliability (important for wasm-all/static tooling).

## Recommended Integration Pass Order

1. Runtime Surface Expansion Pass
- Add pass-through wrappers in BonfyreRuntime for: model, recipe, run, stitch, proxy, sli, graph, index.
- Add runtime service commands: proxy-serve, moq-relay, swarm-worker orchestration.

2. Control-Gated Execution Pass
- Wire entropy-check preflight and route/score hooks into bonfyre-run and runtime autowire for all modes.

3. Stitch Mapping Correctness Pass
- Fix op->binary mapping table and add map validation against existing cmd targets.

4. Model<->Swarm Contract Pass
- Add explicit swarm artifact pull API and model artifact exchange endpoint parity.

5. Packaging/Discoverability Pass
- Normalize TARGET in cmd Makefiles and emit machine-readable cmd capability index.

## Cross-Binary Reference Graph (Corrected)

Format: source -> referenced-binary (count, excluding self-name references).

- BonfyreAPI -> bonfyre-api (9)
- BonfyreAPI -> bonfyre-queue (1)
- BonfyreAuth -> bonfyre-auth (18)
- BonfyreBrief -> bonfyre-brief (1)
- BonfyreCanon -> bonfyre-canon (11)
- BonfyreCapability -> bonfyre-brief (1)
- BonfyreCapability -> bonfyre-capability (16)
- BonfyreCapability -> bonfyre-compete (1)
- BonfyreCapability -> bonfyre-control (1)
- BonfyreCapability -> bonfyre-distribute (1)
- BonfyreCapability -> bonfyre-economy (1)
- BonfyreCapability -> bonfyre-entity (1)
- BonfyreCapability -> bonfyre-flow (1)
- BonfyreCapability -> bonfyre-intake (1)
- BonfyreCapability -> bonfyre-learn (1)
- BonfyreCapability -> bonfyre-queue (1)
- BonfyreCapability -> bonfyre-space (1)
- BonfyreCapability -> bonfyre-tag (1)
- BonfyreCapability -> bonfyre-transcribe (2)
- BonfyreCapability -> bonfyre-translate (1)
- BonfyreCLI -> bonfyre-brief (2)
- BonfyreCLI -> bonfyre-compress (2)
- BonfyreCLI -> bonfyre-distribute (2)
- BonfyreCLI -> bonfyre-emit (2)
- BonfyreCLI -> bonfyre-gate (3)
- BonfyreCLI -> bonfyre-hash (3)
- BonfyreCLI -> bonfyre-index (2)
- BonfyreCLI -> bonfyre-ingest (2)
- BonfyreCLI -> bonfyre-ledger (2)
- BonfyreCLI -> bonfyre-media-prep (1)
- BonfyreCLI -> bonfyre-mediaprep (1)
- BonfyreCLI -> bonfyre-meter (2)
- BonfyreCLI -> bonfyre-narrate (2)
- BonfyreCLI -> bonfyre-offer (2)
- BonfyreCLI -> bonfyre-orchestrate (2)
- BonfyreCLI -> bonfyre-pack (2)
- BonfyreCLI -> bonfyre-paragraph (1)
- BonfyreCLI -> bonfyre-project (2)
- BonfyreCLI -> bonfyre-proof (2)
- BonfyreCLI -> bonfyre-queue (2)
- BonfyreCLI -> bonfyre-render (2)
- BonfyreCLI -> bonfyre-runtime (2)
- BonfyreCLI -> bonfyre-stitch (2)
- BonfyreCLI -> bonfyre-sync (2)
- BonfyreCLI -> bonfyre-transcribe (2)
- BonfyreCLI -> bonfyre-transcript-clean (1)
- BonfyreCLI -> bonfyre-transcript-family (2)
- BonfyreClips -> bonfyre-clips (6)
- BonfyreCMS -> bonfyre-cms (70)
- BonfyreCompete -> bonfyre-compete (21)
- BonfyreCompete -> bonfyre-control (2)
- BonfyreCompete -> bonfyre-economy (1)
- BonfyreCompete -> bonfyre-learn (1)
- BonfyreCompress -> bonfyre-compress (10)
- BonfyreCompress -> bonfyre-compress-test- (1)
- BonfyreControl -> bonfyre-control (40)
- BonfyreControl -> bonfyre-flow (1)
- BonfyreControl -> bonfyre-model (3)
- BonfyreControl -> bonfyre-run (6)
- BonfyreControl -> bonfyre-transcribe (1)
- BonfyreDistribute -> bonfyre-distribute (4)
- BonfyreEconomy -> bonfyre-control (2)
- BonfyreEconomy -> bonfyre-economy (17)
- BonfyreEconomy -> bonfyre-run (1)
- BonfyreEmbed -> bonfyre-embed (2)
- BonfyreEmit -> bonfyre-emit (6)
- BonfyreEntity -> bonfyre-entity (24)
- BonfyreEntity -> bonfyre-tag (1)
- BonfyreEntity -> bonfyre-time (1)
- BonfyreEntity -> bonfyre-transcribe (1)
- BonfyreFinance -> bonfyre-finance (31)
- BonfyreFlow -> bonfyre-control (1)
- BonfyreFlow -> bonfyre-flow (24)
- BonfyreFlow -> bonfyre-run (4)
- BonfyreFlow -> bonfyre-space (1)
- BonfyreFPQ -> bonfyre-fpq (20)
- BonfyreFPQx -> bonfyre-fpqx (18)
- BonfyreGate -> bonfyre-gate (8)
- BonfyreGen -> bonfyre-gen (21)
- BonfyreGen -> bonfyre-run (3)
- BonfyreGraph -> bonfyre-graph (24)
- BonfyreHash -> bonfyre-hash (10)
- BonfyreIndex -> bonfyre-index (18)
- BonfyreIngest -> bonfyre-ingest (2)
- BonfyreKVCache -> bonfyre-kvcache (10)
- BonfyreKVCache -> bonfyre-quant (3)
- BonfyreLayer -> bonfyre-layer (37)
- BonfyreLayer -> bonfyre-layer-target-v1 (1)
- BonfyreLayer -> bonfyre-model (2)
- BonfyreLearn -> bonfyre-compete (2)
- BonfyreLearn -> bonfyre-control (2)
- BonfyreLearn -> bonfyre-learn (17)
- BonfyreLedger -> bonfyre-ledger (9)
- BonfyreMediaPrep -> bonfyre-media-prep (6)
- BonfyreMeter -> bonfyre-meter (9)
- BonfyreMFADict -> bonfyre-mfa-dict (1)
- BonfyreModel -> bonfyre-fpq (2)
- BonfyreModel -> bonfyre-model (49)
- BonfyreModel -> bonfyre-oss (1)
- BonfyreModel -> bonfyre-quant (1)
- BonfyreModel -> bonfyre-run (3)
- BonfyreModel -> bonfyre-swarm (5)
- BonfyreModel -> bonfyre-topic-mapper-v1 (1)
- BonfyreMoQ -> bonfyre-moq (7)
- BonfyreNarrate -> bonfyre-narrate (8)
- BonfyreNarrate -> bonfyre-tone (7)
- BonfyreOffer -> bonfyre-offer (1)
- BonfyreOrchestrate -> bonfyre-api (1)
- BonfyreOrchestrate -> bonfyre-auth (1)
- BonfyreOrchestrate -> bonfyre-emit (1)
- BonfyreOrchestrate -> bonfyre-flow (1)
- BonfyreOrchestrate -> bonfyre-orchestrate (6)
- BonfyreOrchestrate -> bonfyre-orchestrate- (1)
- BonfyreOrchestrate -> bonfyre-queue (1)
- BonfyreOrchestrate -> bonfyre-render (1)
- BonfyreOrchestrate -> bonfyre-runtime (2)
- BonfyreOutreach -> bonfyre-outreach (16)
- BonfyrePack -> bonfyre-pack (2)
- BonfyrePack -> bonfyre-pack- (3)
- BonfyreParagraph -> bonfyre-paragraph (1)
- BonfyrePay -> bonfyre-meter (3)
- BonfyrePay -> bonfyre-pay (14)
- BonfyrePipeline -> bonfyre-brief (3)
- BonfyrePipeline -> bonfyre-media-prep (1)
- BonfyrePipeline -> bonfyre-offer (3)
- BonfyrePipeline -> bonfyre-pack (3)
- BonfyrePipeline -> bonfyre-pipeline (3)
- BonfyrePipeline -> bonfyre-proof (6)
- BonfyrePipeline -> bonfyre-tag (3)
- BonfyrePipeline -> bonfyre-transcribe (3)
- BonfyrePipeline -> bonfyre-transcript-clean (3)
- BonfyreProject -> bonfyre-cms (2)
- BonfyreProject -> bonfyre-index (2)
- BonfyreProject -> bonfyre-project (5)
- BonfyreProject -> bonfyre-stitch (2)
- BonfyreProof -> bonfyre-proof (3)
- BonfyreProxy -> bonfyre-brief (8)
- BonfyreProxy -> bonfyre-proxy (8)
- BonfyreProxy -> bonfyre-transcribe (6)
- BonfyreQuant -> bonfyre-quant (17)
- BonfyreQuery -> bonfyre-query (11)
- BonfyreQueue -> bonfyre-queue (21)
- BonfyreRecipe -> bonfyre-brief (6)
- BonfyreRecipe -> bonfyre-canon (1)
- BonfyreRecipe -> bonfyre-clips (1)
- BonfyreRecipe -> bonfyre-cms (1)
- BonfyreRecipe -> bonfyre-compress (1)
- BonfyreRecipe -> bonfyre-distribute (1)
- BonfyreRecipe -> bonfyre-embed (7)
- BonfyreRecipe -> bonfyre-emit (4)
- BonfyreRecipe -> bonfyre-finance (1)
- BonfyreRecipe -> bonfyre-gate (1)
- BonfyreRecipe -> bonfyre-graph (1)
- BonfyreRecipe -> bonfyre-hash (7)
- BonfyreRecipe -> bonfyre-index (2)
- BonfyreRecipe -> bonfyre-ingest (11)
- BonfyreRecipe -> bonfyre-ledger (5)
- BonfyreRecipe -> bonfyre-media-prep (2)
- BonfyreRecipe -> bonfyre-meter (3)
- BonfyreRecipe -> bonfyre-offer (2)
- BonfyreRecipe -> bonfyre-pack (4)
- BonfyreRecipe -> bonfyre-paragraph (1)
- BonfyreRecipe -> bonfyre-proof (4)
- BonfyreRecipe -> bonfyre-quant (3)
- BonfyreRecipe -> bonfyre-recipe (22)
- BonfyreRecipe -> bonfyre-render (2)
- BonfyreRecipe -> bonfyre-repurpose (2)
- BonfyreRecipe -> bonfyre-stitch (1)
- BonfyreRecipe -> bonfyre-sync (1)
- BonfyreRecipe -> bonfyre-tag (4)
- BonfyreRecipe -> bonfyre-tone (1)
- BonfyreRecipe -> bonfyre-transcribe (6)
- BonfyreRecipe -> bonfyre-transcript-clean (5)
- BonfyreRender -> bonfyre-brief (2)
- BonfyreRender -> bonfyre-narrate (2)
- BonfyreRender -> bonfyre-pack (2)
- BonfyreRender -> bonfyre-render (3)
- BonfyreRepurpose -> bonfyre-repurpose (9)
- BonfyreRun -> bonfyre-out (4)
- BonfyreRun -> bonfyre-recipe (8)
- BonfyreRun -> bonfyre-run (19)
- BonfyreRuntime -> bonfyre-clean (1)
- BonfyreRuntime -> bonfyre-control (2)
- BonfyreRuntime -> bonfyre-gen (4)
- BonfyreRuntime -> bonfyre-ingest (1)
- BonfyreRuntime -> bonfyre-ledger (2)
- BonfyreRuntime -> bonfyre-loop- (1)
- BonfyreRuntime -> bonfyre-moq (2)
- BonfyreRuntime -> bonfyre-pipeline (3)
- BonfyreRuntime -> bonfyre-queue (2)
- BonfyreRuntime -> bonfyre-runtime (26)
- BonfyreRuntime -> bonfyre-space (2)
- BonfyreRuntime -> bonfyre-swarm (3)
- BonfyreRuntime -> bonfyre-transcribe (1)
- BonfyreSegment -> bonfyre-segment (8)
- BonfyreSLI -> bonfyre-fpqx (1)
- BonfyreSLI -> bonfyre-model (16)
- BonfyreSLI -> bonfyre-quant (2)
- BonfyreSLI -> bonfyre-sli (17)
- BonfyreSpace -> bonfyre-compete (1)
- BonfyreSpace -> bonfyre-flow (1)
- BonfyreSpace -> bonfyre-run (1)
- BonfyreSpace -> bonfyre-space (20)
- BonfyreSpeechLoop -> bonfyre-brief (2)
- BonfyreSpeechLoop -> bonfyre-narrate (2)
- BonfyreSpeechLoop -> bonfyre-paragraph (2)
- BonfyreSpeechLoop -> bonfyre-speechloop (5)
- BonfyreSpeechLoop -> bonfyre-transcribe (2)
- BonfyreSpeechLoop -> bonfyre-transcript-clean (2)
- BonfyreStitch -> bonfyre-brief (2)
- BonfyreStitch -> bonfyre-compress (1)
- BonfyreStitch -> bonfyre-distribute (1)
- BonfyreStitch -> bonfyre-emit (1)
- BonfyreStitch -> bonfyre-ingest (1)
- BonfyreStitch -> bonfyre-narrate (1)
- BonfyreStitch -> bonfyre-offer (1)
- BonfyreStitch -> bonfyre-pack (1)
- BonfyreStitch -> bonfyre-proof (1)
- BonfyreStitch -> bonfyre-stitch (8)
- BonfyreStitch -> bonfyre-stitch-cache (1)
- BonfyreStitch -> bonfyre-transcribe (1)
- BonfyreSwarm -> bonfyre-control (1)
- BonfyreSwarm -> bonfyre-swarm (11)
- BonfyreSwarm -> bonfyre-transcribe (1)
- BonfyreSwarm -> bonfyre-worker- (1)
- BonfyreSync -> bonfyre-sync (4)
- BonfyreTag -> bonfyre-tag (11)
- BonfyreTel -> bonfyre-ingest (2)
- BonfyreTel -> bonfyre-pipeline (2)
- BonfyreTel -> bonfyre-sim (3)
- BonfyreTel -> bonfyre-tel (34)
- BonfyreTel -> bonfyre-verify (1)
- BonfyreTier -> bonfyre-control (2)
- BonfyreTier -> bonfyre-economy (2)
- BonfyreTier -> bonfyre-run (1)
- BonfyreTier -> bonfyre-tier (19)
- BonfyreTime -> bonfyre-entity (1)
- BonfyreTime -> bonfyre-model (1)
- BonfyreTime -> bonfyre-run (2)
- BonfyreTime -> bonfyre-time (24)
- BonfyreTone -> bonfyre-tone (9)
- BonfyreTranscribe -> bonfyre-media-prep (2)
- BonfyreTranscribe -> bonfyre-transcribe (2)
- BonfyreTranscriptClean -> bonfyre-transcript-clean (1)
- BonfyreTranscriptFamily -> bonfyre-paragraph (2)
- BonfyreTranscriptFamily -> bonfyre-transcribe (2)
- BonfyreTranscriptFamily -> bonfyre-transcript-clean (2)
- BonfyreTranscriptFamily -> bonfyre-transcript-family (2)
- BonfyreVec -> bonfyre-vec (15)
- BonfyreWeaviateIndex -> bonfyre-document (1)
- BonfyreWeaviateIndex -> bonfyre-weaviate-index (1)

## Binaries With No Outbound Cross-References


## Immediate Wiring Opportunities (Non-Cycle-9)

- Wire bonfyre-runtime wrappers for bonfyre-model, bonfyre-recipe, bonfyre-run, bonfyre-stitch, bonfyre-proxy, bonfyre-sli.
- Add runtime command bonfyre-runtime service proxy|moq|swarm-worker to standardize long-running process launch.
- Add bonfyre-run preflight hooks: bonfyre-control entropy-check + route before stage execution.
- Normalize stitch op mapping: Clean should target bonfyre-transcript-clean (or bonfyre-paragraph) rather than bonfyre-transcribe.
- Add Makefile TARGET for dirs missing TARGET to improve automation surfaces (static/wasm/capability index).
