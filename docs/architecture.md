# Bonfyre Architecture

## The One Abstraction

Bonfyre is **typed artifact families + composable operator graphs + projection-first outputs**.

Every binary in the system does one of three things:
1. **Produces** a `BfArtifact` manifest
2. **Transforms** one `BfArtifact` into another
3. **Serves** artifact state over HTTP or CLI

The artifact contract (`bonfyre.h`) is the universal interface. If you understand `BfArtifact`, you understand the whole system.

```c
typedef struct {
    char artifact_id[512];     // content-addressed (SHA-256 of canonical form)
    char artifact_type[128];   // "transcript", "brief", "proof", "offer", etc.
    char source_system[128];   // originating binary
    char created_at[128];      // ISO-8601 UTC
    char root_hash[128];       // SHA-256 of content
    char family_key[17];       // FNV-1a-64: groups structurally equivalent artifacts
    char canonical_key[17];    // FNV-1a-64: distinguishes signatures within a family
    int  atoms_count;          // sub-objects
    int  operators_count;      // transform nodes
    int  realizations_count;   // materialized outputs
    int  component_total;      // atoms + operators + realizations
} BfArtifact;
```

## Design Philosophy

1. **One binary, one job.** Each tool does one thing. Compose via pipes and files.
2. **SQLite everywhere.** Every binary that needs state uses SQLite. One folder backup.
3. **Zero cloud.** Everything runs on your machine. No external API calls unless you choose to.
4. **C11 for size and speed.** Every binary is < 70 KB. Startup is < 50 ms.
5. **Pure or stateful. Never both.** Every binary declares its behavioral class.
6. **Artifacts, not files.** Every output is a typed, hashed, family-grouped manifest.

## Layer Model

Binaries are classified by behavioral contract, not by marketing category.

```
┌──────────────────────────────────────────────────────────────────┐
│  SURFACE — product-facing, stateful services                     │
│  cms · api · auth · pipeline · cli · transcript-family · project │
├──────────────────────────────────────────────────────────────────┤
│  VALUE — monetization, metering, delivery                        │
│  offer · gate · meter · ledger · finance · outreach · pay        │
│  pack · distribute                                               │
├──────────────────────────────────────────────────────────────────┤
│  TRANSFORM — pure, cacheable, stateless                          │
│  media-prep · transcribe · transcript-clean · paragraph · brief  │
│  proof · embed · narrate · render · emit · mfa-dict              │
│  weaviate-index                                                  │
├──────────────────────────────────────────────────────────────────┤
│  SUBSTRATE — cold, formal, stable infrastructure                 │
│  ingest · hash · index · compress · stitch · graph               │
│  runtime · queue · sync                                          │
├──────────────────────────────────────────────────────────────────┤
│  LIBRARIES                                                       │
│  libbonfyre (runtime contract, operators, SHA-256, utilities)    │
│  liblambda-tensors (structural family compression)               │
└──────────────────────────────────────────────────────────────────┘
```

**Rules:**
- **Substrate** binaries never import product concepts. Cold infrastructure.
- **Transform** binaries are pure: same inputs → same outputs. Cacheable by `(operator, params, input_hash)`.
- **Surface** binaries own mutable state and serve HTTP or complex CLI interfaces.
- **Value** binaries handle money, metering, and delivery. Separate privilege boundary.

## Operator Model

Every binary declares a typed operator descriptor in `libbonfyre`:

| Flag | Meaning |
|---|---|
| `BF_OP_PURE` | Stateless: same inputs, same outputs |
| `BF_OP_STATEFUL` | Owns mutable state (SQLite, files) |
| `BF_OP_CACHEABLE` | Output can be cached by `(op, params, hash)` |
| `BF_OP_REVERSIBLE` | Output → input reconstruction is possible |
| `BF_OP_IDEMPOTENT` | Running twice = running once |
| `BF_OP_STREAMING` | Can process data incrementally |

A binary is **either** `PURE` **or** `STATEFUL`. Never both. Enforced by tests.

**Exactness classes:**

| Class | Meaning | Examples |
|---|---|---|
| `BF_EXACT_BYTE` | Byte-for-byte identical on replay | hash, compress, pack |
| `BF_EXACT_CANON` | Identical after canonicalization | index, transcript-clean |
| `BF_EXACT_LOSSY` | Derived but not perfectly reconstructable | transcribe, narrate |

## Dependency Graph

```
Binary              SQLite  zlib  Whisper  Piper  ONNX  Pandoc  Weaviate
──────              ──────  ────  ───────  ─────  ────  ──────  ────────
bonfyre-cms           ✓      ✓
bonfyre-api           ✓
bonfyre-auth          ✓
bonfyre-pipeline      ✓      ✓
bonfyre-index         ✓
bonfyre-meter         ✓
bonfyre-graph         ✓
bonfyre-queue         ✓
bonfyre-ledger        ✓
bonfyre-transcribe                  ✓
bonfyre-narrate                              ✓
bonfyre-embed                                       ✓
bonfyre-emit                                               ✓
bonfyre-weaviate-idx                                               ✓

All others: libc only (no external dependencies)
```

Required at build: C11 compiler + SQLite3 headers + zlib headers.
Optional at runtime: Whisper model, Piper TTS, ONNX runtime, pandoc, Weaviate.

## Family Model

Artifacts that share `(artifact_type, source_system)` belong to the same **family**.

Families are the natural unit of compression, indexing, caching, and storage:

```
family_key    = FNV-1a-64(normalize(type) + "|" + normalize(system))
canonical_key = FNV-1a-64(type + "|" + system + "|" + atom_count + "|" + op_count + "|" + real_count)
```

Lambda Tensors exploits structural similarity within families to achieve 13.5% of raw JSON size with random access.

## Artifact Lifecycle

```
Source → Ingest → Transform chain → Index → Realize
```

1. **Ingest:** `bonfyre-ingest` detects type, normalizes, stamps `BfArtifact` manifest
2. **Transform:** Pure operators in sequence. Each reads `BfArtifact`, produces `BfArtifact`
3. **Index:** `bonfyre-index` registers in SQLite, groups by `family_key`
4. **Realize:** Materialize outputs on demand (pack, emit, narrate, distribute)

**Key principle:** transforms produce artifacts; realizations produce outputs.

## Data Flow

```
audio.mp3
  → bonfyre-ingest     (manifest + type detection)
  → bonfyre-media-prep (16kHz mono WAV)
  → bonfyre-hash       (SHA-256 content address)
  → bonfyre-transcribe (Whisper speech-to-text)
  → bonfyre-transcript-clean (remove filler)
  → bonfyre-paragraph  (structure text)
  → bonfyre-brief      (summary + action items)
  → bonfyre-proof      (quality score)
  → bonfyre-offer      (pricing)
  → bonfyre-pack       (ZIP deliverable)
```

Or: `bonfyre-pipeline run` — all 10 steps in one process, 5–8 ms.

## Storage

All data lives in `~/.local/share/bonfyre/` by default:

```
~/.local/share/bonfyre/
├── cms.db          # BonfyreCMS schemas, content, tokens
├── jobs.db         # BonfyreAPI job tracking
├── meter.db        # Usage metering
├── index.db        # Artifact index
├── uploads/        # Uploaded files
└── artifacts/      # Pipeline outputs
```

## HTTP endpoints (bonfyre-api)

```
GET  /api/health           → {"status":"ok","version":"1.0.0"}
GET  /api/status           → job counts, upload counts, available binaries
POST /api/upload           → multipart file upload
POST /api/jobs             → submit pipeline job
GET  /api/jobs             → list all jobs
GET  /api/jobs/:id         → job detail
*    /api/binaries/:name/* → proxy to any bonfyre-* binary
GET  /*                    → static files (frontend SPA)
```

## Build System

```bash
make              # Build 2 libraries + 37 binaries
make lib          # Build liblambda-tensors + libbonfyre
make install      # Install to ~/.local
make clean        # Remove all build artifacts
make test         # Run all test suites
make sanitize     # Rebuild with ASan + UBSan for security testing
make help         # Show all targets
```

## Libraries

### libbonfyre (runtime contract)

The shared substrate. Defines the artifact contract, operator registry, SHA-256, FNV-1a, and common utilities that were previously duplicated across all 37 binaries.

- Header: `lib/libbonfyre/include/bonfyre.h`
- Static library: `lib/libbonfyre/libbonfyre.a`
- Tests: 20 tests covering artifact parsing, key computation, SHA-256 vectors, operator registry invariants

### liblambda-tensors (family compression)

Structural compression for families of similar JSON records.

- 5 encoding tiers: V1 (varint) → V2 (type-aware) → Interned → Huffman → Arithmetic
- Family string tables: cross-member deduplication
- Random access: O(1) per-field reads on compressed data
- 13.5% of raw JSON size at N=10,000 with Huffman

## Adding a New Binary

1. Create `cmd/BonfyreYourThing/src/main.c`
2. Create `cmd/BonfyreYourThing/Makefile` (copy any existing one)
3. Add an operator descriptor to `lib/libbonfyre/src/bf_operators.c`
4. Declare whether you are `BF_OP_PURE` or `BF_OP_STATEFUL` — pick one
5. Read/write `BfArtifact` manifests for all inputs and outputs
6. Implement at minimum: `status` command, `--help`, `--version`
7. `make` in the new directory, `make test` from root
8. Open a PR
