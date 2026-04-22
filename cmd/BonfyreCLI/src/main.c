/*
 * bonfyre — unified CLI dispatcher.
 *
 * Routes subcommands to their respective binaries:
 *   bonfyre <cmd> [args...]  →  bonfyre-<cmd> [args...]
 *
 * Binary search order per command:
 *   1. Same directory as this binary
 *   2. ../SiblingDir/bonfyre-<cmd>          (top-level Makefile output)
 *   3. ../SiblingDir/build/bonfyre-<cmd>    (Makefile build/ subdirectory)
 *   4. PATH
 *
 * Commands that route through bonfyre-runtime pass the original
 * subcommand token as the first argument so runtime can dispatch
 * internally.
 */
#include <limits.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ── Section constants ────────────────────────────────────────────── */
#define SEC_PIPELINE  "Pipeline"
#define SEC_AI        "AI / Models"
#define SEC_RECIPES   "Recipes & Runtime"
#define SEC_INFRA     "Infrastructure"
#define SEC_VALUE     "Value Capture"

typedef struct {
    const char *cmd;
    const char *binary;
    const char *sibling_dir;
    const char *section;
    const char *desc;
} Route;

typedef struct {
    int total;
    int available;
} RouteStats;

static const Route routes[] = {
    /* ── Pipeline ──────────────────────────────────────────────── */
    {"ingest",            "bonfyre-ingest",           "BonfyreIngest",          SEC_PIPELINE, "Universal asset intake"},
    {"mediaprep",         "bonfyre-media-prep",       "BonfyreMediaPrep",       SEC_PIPELINE, "Media normalisation"},
    {"transcribe",        "bonfyre-transcribe",       "BonfyreTranscribe",      SEC_PIPELINE, "Audio to text"},
    {"clean",             "bonfyre-transcript-clean", "BonfyreTranscriptClean", SEC_PIPELINE, "Transcript cleaning"},
    {"paragraph",         "bonfyre-paragraph",        "BonfyreParagraph",       SEC_PIPELINE, "Transcript paragraphizer"},
    {"brief",             "bonfyre-brief",            "BonfyreBrief",           SEC_PIPELINE, "Extract structured brief"},
    {"proof",             "bonfyre-proof",            "BonfyreProof",           SEC_PIPELINE, "Generate proof bundle"},
    {"offer",             "bonfyre-offer",            "BonfyreOffer",           SEC_PIPELINE, "Generate offer document"},
    {"narrate",           "bonfyre-narrate",          "BonfyreNarrate",         SEC_PIPELINE, "Text-to-speech narration"},
    {"pack",              "bonfyre-pack",             "BonfyrePack",            SEC_PIPELINE, "Package artifact family"},
    {"distribute",        "bonfyre-distribute",       "BonfyreDistribute",      SEC_PIPELINE, "Multi-channel distribution"},
    {"transcript-family", "bonfyre-transcript-family","BonfyreTranscriptFamily",SEC_PIPELINE, "Speech to cleaned transcript family"},
    {"render",            "bonfyre-render",           "BonfyreRender",          SEC_PIPELINE, "Universal artifact renderer"},
    {"repurpose",         "bonfyre-repurpose",        "BonfyreRepurpose",       SEC_PIPELINE, "Repurpose transcripts to new formats"},
    {"clips",             "bonfyre-clips",            "BonfyreClips",           SEC_PIPELINE, "Extract short clips from media"},
    {"frame-extract",     "bonfyre-frame-extract",    "BonfyreFrameExtract",    SEC_PIPELINE, "Extract frames from video"},
    {"scene-detect",      "bonfyre-scene-detect",     "BonfyreSceneDetect",     SEC_PIPELINE, "Scene boundary detection"},
    {"video-demux",       "bonfyre-video-demux",      "BonfyreVideoDemux",      SEC_PIPELINE, "Demux video streams to tracks"},
    {"detect-objects",    "bonfyre-detect-objects",   "BonfyreDetectObjects",   SEC_PIPELINE, "Object detection (vision pipeline stage)"},
    {"fragment",          "bonfyre-fragment",         "BonfyreFragment",        SEC_PIPELINE, "Fragment store — create / query / merge"},
    /* ── AI / Models ────────────────────────────────────────────── */
    {"model",             "bonfyre-model",            "BonfyreModel",           SEC_AI, "Model registry (list / pull / verify)"},
    {"embed",             "bonfyre-embed",            "BonfyreEmbed",           SEC_AI, "Generate text embeddings (ONNX)"},
    {"vec",               "bonfyre-vec",              "BonfyreVec",             SEC_AI, "Local vector search (FAISS)"},
    {"segment",           "bonfyre-segment",          "BonfyreSegment",         SEC_AI, "Speaker / VAD segmentation"},
    {"speech-loop",       "bonfyre-speechloop",       "BonfyreSpeechLoop",      SEC_AI, "Streaming RNNT + Whisper ASR loop"},
    {"mfa-dict",          "bonfyre-mfa-dict",         "BonfyreMFADict",         SEC_AI, "MFA forced-alignment dictionary"},
    {"tone",              "bonfyre-tone",             "BonfyreTone",            SEC_AI, "Tone / sentiment analysis"},
    {"tag",               "bonfyre-tag",              "BonfyreTag",             SEC_AI, "Auto-tagging pipeline"},
    {"entity",            "bonfyre-entity",           "BonfyreEntity",          SEC_AI, "Named-entity recognition"},
    {"canon",             "bonfyre-canon",            "BonfyreCanon",           SEC_AI, "Canonical form resolver"},
    {"gen",               "bonfyre-gen",              "BonfyreGen",             SEC_AI, "Natural language generation"},
    {"sli",               "bonfyre-sli",              "BonfyreSLI",             SEC_AI, "Structured layer inference (E8 lattice)"},
    {"quant",             "bonfyre-quant",            "BonfyreQuant",           SEC_AI, "BQFP model quantisation"},
    {"fpq",               "bonfyre-fpq",              "BonfyreFPQ",             SEC_AI, "Functional precision quantisation"},
    {"fpqx",              "bonfyre-fpqx",             "BonfyreFPQx",            SEC_AI, "FPQx extended quantisation"},
    {"layer",             "bonfyre-layer",            "BonfyreLayer",           SEC_AI, "Neural layer operations"},
    {"learn",             "bonfyre-learn",            "BonfyreLearn",           SEC_AI, "On-device fine-tuning / adapters"},
    {"weaviate",          "bonfyre-weaviate-index",   "BonfyreWeaviateIndex",   SEC_AI, "Weaviate vector index bridge"},
    /* ── Recipes & Runtime ──────────────────────────────────────── */
    {"recipe",            "bonfyre-recipe",           "BonfyreRecipe",          SEC_RECIPES, "Recipe registry (list / show / run / add)"},
    {"run",               "bonfyre-run",              "BonfyreRun",             SEC_RECIPES, "Execute a recipe by name or path"},
    {"flow",              "bonfyre-flow",             "BonfyreFlow",            SEC_RECIPES, "Coroutine-native pipeline flow graphs"},
    {"pipeline",          "bonfyre-pipeline",         "BonfyrePipeline",        SEC_RECIPES, "Streaming pipeline execution"},
    {"runtime",           "bonfyre-runtime",          "BonfyreRuntime",         SEC_RECIPES, "Replayable pipeline runtime"},
    {"orchestrate",       "bonfyre-orchestrate",      "BonfyreOrchestrate",     SEC_RECIPES, "Machine-only orchestration planner"},
    {"control",           "bonfyre-control",          "BonfyreControl",         SEC_RECIPES, "Control-plane command gateway"},
    {"swarm",             "bonfyre-swarm",            "BonfyreSwarm",           SEC_RECIPES, "Distributed worker swarm"},
    {"project",           "bonfyre-project",          "BonfyreProject",         SEC_RECIPES, "Content graph projection engine"},
    {"space",             "bonfyre-space",            "BonfyreSpace",           SEC_RECIPES, "Semantic space management"},
    {"proxy",             "bonfyre-proxy",            "BonfyreProxy",           SEC_RECIPES, "OpenAI-compatible API proxy"},
    {"doctor",            "bonfyre-runtime",          "BonfyreRuntime",         SEC_RECIPES, "Runtime dependency diagnostics"},
    {"capabilities",      "bonfyre-capability",       "BonfyreCapability",      SEC_RECIPES, "Capability discovery and matching registry"},
    /* ── Infrastructure ─────────────────────────────────────────── */
    {"hash",              "bonfyre-hash",             "BonfyreHash",            SEC_INFRA, "Content-addressing (SHA-256)"},
    {"index",             "bonfyre-index",            "BonfyreIndex",           SEC_INFRA, "Artifact family indexer"},
    {"compress",          "bonfyre-compress",         "BonfyreCompress",        SEC_INFRA, "Family-aware compression"},
    {"emit",              "bonfyre-emit",             "BonfyreEmit",            SEC_INFRA, "Multi-format output engine"},
    {"stitch",            "bonfyre-stitch",           "BonfyreStitch",          SEC_INFRA, "DAG materialiser"},
    {"queue",             "bonfyre-queue",            "BonfyreQueue",           SEC_INFRA, "Job queue management"},
    {"sync",              "bonfyre-sync",             "BonfyreSync",            SEC_INFRA, "Artifact synchronisation"},
    {"graph",             "bonfyre-graph",            "BonfyreGraph",           SEC_INFRA, "Merkle-DAG artifact graph (SQLite)"},
    {"query",             "bonfyre-query",            "BonfyreQuery",           SEC_INFRA, "Structured artifact query"},
    {"kvcache",           "bonfyre-kvcache",          "BonfyreKVCache",         SEC_INFRA, "KV-cache store"},
    {"auth",              "bonfyre-auth",             "BonfyreAuth",            SEC_INFRA, "Authentication and token management"},
    {"tel",               "bonfyre-tel",              "BonfyreTel",             SEC_INFRA, "Telemetry / observability"},
    {"moq",               "bonfyre-moq",              "BonfyreMoQ",             SEC_INFRA, "MoQ media-over-QUIC transport"},
    {"cms",               "bonfyre-cms",              "BonfyreCMS",             SEC_INFRA, "Content management store"},
    {"api",               "bonfyre-api",              "BonfyreAPI",             SEC_INFRA, "REST API server"},
    {"time",              "bonfyre-time",             "BonfyreTime",            SEC_INFRA, "Temporal metadata and scheduling"},
    /* ── Value Capture ───────────────────────────────────────────── */
    {"gate",              "bonfyre-gate",             "BonfyreGate",            SEC_VALUE, "License enforcement"},
    {"meter",             "bonfyre-meter",            "BonfyreMeter",           SEC_VALUE, "Usage metering and billing"},
    {"ledger",            "bonfyre-ledger",           "BonfyreLedger",          SEC_VALUE, "Value accounting"},
    {"economy",           "bonfyre-economy",          "BonfyreEconomy",         SEC_VALUE, "Economy / credits engine"},
    {"compete",           "bonfyre-compete",          "BonfyreCompete",         SEC_VALUE, "Competitive benchmarking"},
    {"pay",               "bonfyre-pay",              "BonfyrePay",             SEC_VALUE, "Payment processing"},
    {"finance",           "bonfyre-finance",          "BonfyreFinance",         SEC_VALUE, "Financial reporting"},
    {"tier",              "bonfyre-tier",             "BonfyreTier",            SEC_VALUE, "Feature / access tier management"},
    {"outreach",          "bonfyre-outreach",         "BonfyreOutreach",        SEC_VALUE, "Campaign outreach automation"},
    {NULL, NULL, NULL, NULL, NULL}
};

/* ── Binary resolution ────────────────────────────────────────────── */
static void get_self_dir(char *buf, size_t sz) {
    char self[PATH_MAX];
    memset(self, 0, sizeof(self));
#ifdef __APPLE__
    uint32_t bsz = sizeof(self);
    if (_NSGetExecutablePath(self, &bsz) != 0) self[0] = '\0';
#elif defined(__linux__)
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n > 0) self[n] = '\0'; else self[0] = '\0';
#else
    self[0] = '\0';
#endif
    if (self[0]) {
        char *last = strrchr(self, '/');
        if (last) { *last = '\0'; snprintf(buf, sz, "%s", self); return; }
    }
    buf[0] = '\0';
}

static void try_one(const char *path, char **argv) {
    if (access(path, X_OK) == 0) {
        argv[0] = (char *)path;
        execv(path, argv);
    }
}

static int resolve_binary_path(const char *binary, const char *sibling_dir, char *resolved, size_t resolved_size) {
    char self_dir[PATH_MAX];
    get_self_dir(self_dir, sizeof(self_dir));

    if (self_dir[0]) {
        char full[PATH_MAX];

        snprintf(full, sizeof(full), "%s/%s", self_dir, binary);
        if (access(full, X_OK) == 0) {
            snprintf(resolved, resolved_size, "%s", full);
            return 1;
        }

        if (sibling_dir && sibling_dir[0]) {
            snprintf(full, sizeof(full), "%s/../%s/%s", self_dir, sibling_dir, binary);
            if (access(full, X_OK) == 0) {
                snprintf(resolved, resolved_size, "%s", full);
                return 1;
            }

            snprintf(full, sizeof(full), "%s/../%s/build/%s", self_dir, sibling_dir, binary);
            if (access(full, X_OK) == 0) {
                snprintf(resolved, resolved_size, "%s", full);
                return 1;
            }
        }
    }

    const char *path_env = getenv("PATH");
    if (!path_env || !path_env[0]) return 0;

    char *path_copy = strdup(path_env);
    if (!path_copy) return 0;

    int found = 0;
    char *save = NULL;
    for (char *dir = strtok_r(path_copy, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", dir, binary);
        if (access(full, X_OK) == 0) {
            snprintf(resolved, resolved_size, "%s", full);
            found = 1;
            break;
        }
    }

    free(path_copy);
    return found;
}

static int contains_ci(const char *haystack, const char *needle) {
    if (!needle || !needle[0]) return 1;
    size_t needle_len = strlen(needle);
    if (needle_len == 0) return 1;

    for (size_t i = 0; haystack[i]; i++) {
        size_t j = 0;
        while (needle[j] && haystack[i + j] &&
               tolower((unsigned char)haystack[i + j]) == tolower((unsigned char)needle[j])) {
            j++;
        }
        if (j == needle_len) return 1;
    }
    return 0;
}

static int route_matches(const Route *r, const char *query) {
    return contains_ci(r->cmd, query)
        || contains_ci(r->desc, query)
        || contains_ci(r->section, query)
        || contains_ci(r->sibling_dir, query);
}

static RouteStats route_stats(const char *section, const char *query) {
    RouteStats stats = {0, 0};
    char resolved[PATH_MAX];

    for (const Route *r = routes; r->cmd; r++) {
        if (strcmp(r->section, section) != 0) continue;
        if (!route_matches(r, query)) continue;
        stats.total++;
        if (resolve_binary_path(r->binary, r->sibling_dir, resolved, sizeof(resolved)))
            stats.available++;
    }
    return stats;
}

static int try_exec(const char *binary, const char *sibling_dir, char **argv) {
    char self_dir[PATH_MAX];
    get_self_dir(self_dir, sizeof(self_dir));

    if (self_dir[0]) {
        char full[PATH_MAX];

        /* 1. Same directory as this binary */
        snprintf(full, sizeof(full), "%s/%s", self_dir, binary);
        try_one(full, argv);

        if (sibling_dir && sibling_dir[0]) {
            /* 2. ../SiblingDir/<binary>  (top-level build output) */
            snprintf(full, sizeof(full), "%s/../%s/%s", self_dir, sibling_dir, binary);
            try_one(full, argv);

            /* 3. ../SiblingDir/build/<binary>  (build/ subdirectory pattern) */
            snprintf(full, sizeof(full), "%s/../%s/build/%s", self_dir, sibling_dir, binary);
            try_one(full, argv);
        }
    }

    /* 4. Fall back to PATH */
    execvp(binary, argv);
    return -1;
}

/* ── Built-in: list ───────────────────────────────────────────────── */
static void cmd_list(const char *query) {
    static const char *sections[] = {
        SEC_PIPELINE,
        SEC_AI,
        SEC_RECIPES,
        SEC_INFRA,
        SEC_VALUE,
        NULL
    };
    int is_tty = isatty(STDOUT_FILENO);
    const char *bold = is_tty ? "\033[1m" : "";
    const char *dim = is_tty ? "\033[2m" : "";
    const char *green = is_tty ? "\033[32m" : "";
    const char *yellow = is_tty ? "\033[33m" : "";
    const char *reset = is_tty ? "\033[0m" : "";
    int total = 0;
    int available = 0;

    for (const Route *r = routes; r->cmd; r++) {
        char resolved[PATH_MAX];
        if (!route_matches(r, query)) continue;
        total++;
        if (resolve_binary_path(r->binary, r->sibling_dir, resolved, sizeof(resolved)))
            available++;
    }

    printf("%sbonfyre%s  command surface\n", bold, reset);
    if (query && query[0])
        printf("%sfilter%s    %s\n", dim, reset, query);
    printf("%ssummary%s   %d commands  %s%d ready%s  %s%d missing%s\n",
           dim, reset,
           total,
           green, available, reset,
           yellow, total - available, reset);
    printf("%snext%s      bonfyre doctor   bonfyre model list   bonfyre recipe list\n\n",
           dim, reset);

    for (int i = 0; sections[i]; i++) {
        RouteStats stats = route_stats(sections[i], query);
        if (stats.total == 0) continue;

        printf("%s%s%s  %s%d/%d ready%s\n",
               bold, sections[i], reset,
               dim, stats.available, stats.total, reset);

        for (const Route *r = routes; r->cmd; r++) {
            char resolved[PATH_MAX];
            int ready;

            if (strcmp(r->section, sections[i]) != 0) continue;
            if (!route_matches(r, query)) continue;

            ready = resolve_binary_path(r->binary, r->sibling_dir, resolved, sizeof(resolved));
            printf("  %s%-16s%s  %-20s %s\n",
                   ready ? green : yellow,
                   ready ? "ready" : "missing",
                   reset,
                   r->cmd,
                   r->desc);
        }
        printf("\n");
    }

    if (total == 0) {
        printf("No commands matched that filter. Try: bonfyre list model\n\n");
    }

    printf("Run 'bonfyre <command> --help' for command help, or 'bonfyre list <term>' to filter.\n");
}

/* ── Built-in: help ───────────────────────────────────────────────── */
static void cmd_help(void) {
    fprintf(stderr,
        "bonfyre -- adaptive artifact pipeline toolkit\n\n"
        "Usage: bonfyre <command> [args...]\n\n"
        "Built-ins:\n"
        "  list [term]   Show command surface, live readiness, and optional filtered search\n"
        "  version       Print version\n"
        "  help          Show this help\n\n"
        "Start here:\n"
        "  bonfyre doctor                 verify the installed surface\n"
        "  bonfyre list model            filter command surface by topic\n"
        "  bonfyre model list            inspect local model registry\n"
        "  bonfyre recipe list           inspect recipe registry\n\n"
        "Key workflows:\n"
        "  Pipeline:  ingest -> mediaprep -> transcribe -> clean -> paragraph\n"
        "             -> brief -> proof -> offer -> narrate -> pack -> distribute\n"
        "  Models:    model list / model pull <id> / model pull --recipe <code>\n"
        "  Recipes:   recipe list / recipe show <name> / run <recipe-name>\n"
        "  AI:        embed . vec . segment . sli . quant . fpq . fpqx . layer\n"
        "  Infra:     hash . index . compress . graph . queue . sync . tel\n"
        "  Value:     gate . meter . ledger . economy . compete\n\n"
        "Run 'bonfyre list' for the full command surface.\n"
    );
}

/* ── main ─────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 2) { cmd_help(); return 0; }
    const char *cmd = argv[1];

    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        cmd_help();
        return 0;
    }
    if (strcmp(cmd, "version") == 0 || strcmp(cmd, "--version") == 0) {
        printf("bonfyre 0.2.0\n");
        return 0;
    }
    if (strcmp(cmd, "list") == 0) {
        cmd_list(argc >= 3 ? argv[2] : NULL);
        return 0;
    }

    /* Route lookup */
    for (const Route *r = routes; r->cmd; r++) {
        if (strcmp(cmd, r->cmd) != 0) continue;

        /* Runtime-gateway commands: prepend the original subcommand token
         * so bonfyre-runtime can dispatch internally.
         */
        int via_runtime = (strcmp(r->binary, "bonfyre-runtime") == 0
                           && strcmp(r->cmd, "runtime") != 0);

        char **new_argv = malloc(sizeof(char *) * (size_t)(argc + 2));
        if (!new_argv) { perror("malloc"); return 1; }
        int j = 0;
        new_argv[j++] = (char *)r->binary;
        if (via_runtime)
            new_argv[j++] = (char *)r->cmd;
        for (int i = 2; i < argc; i++)
            new_argv[j++] = argv[i];
        new_argv[j] = NULL;

        try_exec(r->binary, r->sibling_dir, new_argv);
        fprintf(stderr, "bonfyre: '%s' is not installed or not in PATH\n", r->binary);
        fprintf(stderr, "  Build it: make -C cmd/%s\n", r->sibling_dir);
        free(new_argv);
        return 127;
    }

    /* Last-chance: delegate unknown commands to runtime for dynamic dispatch */
    {
        char **fb = malloc(sizeof(char *) * (size_t)(argc + 2));
        if (fb) {
            int j = 0;
            fb[j++] = "bonfyre-runtime";
            fb[j++] = argv[1];
            for (int i = 2; i < argc; i++) fb[j++] = argv[i];
            fb[j] = NULL;
            try_exec("bonfyre-runtime", "BonfyreRuntime", fb);
            free(fb);
        }
    }

    fprintf(stderr, "bonfyre: unknown command '%s'\n", cmd);
    fprintf(stderr, "Run 'bonfyre list' to see all commands.\n");
    return 1;
}
