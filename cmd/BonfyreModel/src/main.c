/*
 * BonfyreModel — model dependency manager for bonfyre pipelines.
 *
 * Content-addressed SQLite registry of AI models required by pipeline
 * recipes.  SHA-256 hash is the canonical model identity — source URLs
 * are hints, not truth.
 *
 * DB path: ~/.local/share/bonfyre/models.db  (override: $BONFYRE_MODEL_DB)
 * Cache:   ~/.cache/bonfyre/models/           (override: $BONFYRE_MODEL_CACHE)
 *
 * Commands:
 *   bonfyre-model list                    — list all registered models
 *   bonfyre-model show <id>               — print full model record
 *   bonfyre-model pull <id>               — ensure model is present in cache
 *   bonfyre-model pull --recipe <code>    — pull all models required by a recipe
 *   bonfyre-model add <file.json>         — register a model manifest from file
 *   bonfyre-model verify <id>             — re-check SHA-256 of cached file
 *   bonfyre-model path <id>               — print absolute cache path (for scripts)
 *   bonfyre-model rm <id>                 — remove from registry (does not delete cache)
 *   bonfyre-model rm --purge <id>         — remove from registry AND delete cache file
 *   bonfyre-model search <query>          — full-text search over names + descriptions
 *   bonfyre-model sources <id>            — list configured pull sources for a model
 *   bonfyre-model source add <id> <url>   — add a pull source URL for a model
 *   bonfyre-model source rm <id> <url>    — remove a pull source URL
 *   bonfyre-model ls-cache               — list cached files with sizes
 *   bonfyre-model status                  — registry + cache stats
 *   bonfyre-model help                    — this message
 *
 * Source URL schemes (evaluated in priority order per model):
 *   swarm://  — bonfyre-swarm local peer (fastest, $0)
 *   file://   — local path (already have it on disk)
 *   s3://     — S3-compatible object store
 *   hf://     — huggingface.co  (HTTPS GET, no SDK required)
 *   https://  — generic HTTPS download
 *
 * bonfyre-run integration:
 *   bonfyre-run reads model deps from recipe JSON and calls
 *   `bonfyre-model pull --recipe <code>` before DAG execution.
 *   Set BONFYRE_MODEL_SKIP_CHECK=1 to bypass (if models pre-staged).
 */

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <sqlite3.h>

#define VERSION        "1.0.0"
#define MAX_JSON       131072   /* 128 KB max manifest */
#define HASH_HEX       65
#define DB_ENV         "BONFYRE_MODEL_DB"
#define CACHE_ENV      "BONFYRE_MODEL_CACHE"
#define DB_SUBPATH     "/.local/share/bonfyre/models.db"
#define CACHE_SUBPATH  "/.cache/bonfyre/models"

/* ====================================================================
 * SHA-256  (FIPS 180-4, vendored — no external deps)
 * ==================================================================== */

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define RR32(x,n)   (((x)>>(n))|((x)<<(32-(n))))
#define S0(x)       (RR32(x,2)^RR32(x,13)^RR32(x,22))
#define S1(x)       (RR32(x,6)^RR32(x,11)^RR32(x,25))
#define s0(x)       (RR32(x,7)^RR32(x,18)^((x)>>3))
#define s1(x)       (RR32(x,17)^RR32(x,19)^((x)>>10))
#define CH(e,f,g)   (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c)  (((a)&(b))^((a)&(c))^((b)&(c)))

typedef struct { uint32_t h[8]; uint8_t buf[64]; uint64_t total; uint32_t used; } SHA256_CTX;

static void sha256_init(SHA256_CTX *c) {
    static const uint32_t H0[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    memcpy(c->h, H0, 32); c->total = 0; c->used = 0;
}
static void sha256_block(SHA256_CTX *c, const uint8_t *d) {
    uint32_t w[64], st[8], t1, t2;
    for(int i = 0; i < 16; i++)
        w[i] = ((uint32_t)d[i*4]<<24)|((uint32_t)d[i*4+1]<<16)|
               ((uint32_t)d[i*4+2]<<8)|d[i*4+3];
    for(int i = 16; i < 64; i++)
        w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];
    memcpy(st, c->h, 32);
    for(int i = 0; i < 64; i++) {
        t1 = st[7] + S1(st[4]) + CH(st[4],st[5],st[6]) + K256[i] + w[i];
        t2 = S0(st[0]) + MAJ(st[0],st[1],st[2]);
        st[7]=st[6]; st[6]=st[5]; st[5]=st[4]; st[4]=st[3]+t1;
        st[3]=st[2]; st[2]=st[1]; st[1]=st[0]; st[0]=t1+t2;
    }
    for(int i = 0; i < 8; i++) c->h[i] += st[i];
}
static void sha256_update(SHA256_CTX *c, const uint8_t *d, size_t len) {
    for(size_t i = 0; i < len; i++) {
        c->buf[c->used++] = d[i]; c->total++;
        if(c->used == 64) { sha256_block(c, c->buf); c->used = 0; }
    }
}
static void sha256_final(SHA256_CTX *c, uint8_t out[32]) {
    uint64_t bits = c->total * 8;
    uint8_t pad = 0x80;
    sha256_update(c, &pad, 1);
    while(c->used != 56) { uint8_t z=0; sha256_update(c,&z,1); }
    for(int i = 7; i >= 0; i--) {
        uint8_t b = (bits>>(i*8))&0xff;
        sha256_update(c,&b,1);
    }
    for(int i = 0; i < 8; i++) {
        out[i*4+0]=(c->h[i]>>24)&0xff; out[i*4+1]=(c->h[i]>>16)&0xff;
        out[i*4+2]=(c->h[i]>>8)&0xff;  out[i*4+3]=c->h[i]&0xff;
    }
}
static void sha256_hex(const uint8_t d[32], char out[65]) {
    static const char hex[]="0123456789abcdef";
    for(int i=0;i<32;i++){out[i*2]=hex[d[i]>>4];out[i*2+1]=hex[d[i]&0xf];}
    out[64]='\0';
}
static int sha256_file(const char *path, char hex[65]) {
    FILE *f = fopen(path, "rb");
    if(!f) return -1;
    SHA256_CTX c; sha256_init(&c);
    uint8_t buf[65536]; size_t n;
    while((n = fread(buf, 1, sizeof(buf), f)) > 0)
        sha256_update(&c, buf, n);
    fclose(f);
    uint8_t raw[32]; sha256_final(&c, raw);
    sha256_hex(raw, hex);
    return 0;
}

/* ====================================================================
 * Tiny JSON helpers (emit only — no parser needed for built-ins)
 * ==================================================================== */
static void json_escape(const char *s, char *out, size_t cap) {
    size_t j = 0;
    for(size_t i = 0; s[i] && j+4 < cap; i++) {
        unsigned char ch = (unsigned char)s[i];
        if(ch == '"')      { out[j++]='\\'; out[j++]='"'; }
        else if(ch == '\\') { out[j++]='\\'; out[j++]='\\'; }
        else if(ch == '\n') { out[j++]='\\'; out[j++]='n'; }
        else if(ch == '\r') { out[j++]='\\'; out[j++]='r'; }
        else if(ch == '\t') { out[j++]='\\'; out[j++]='t'; }
        else if(ch < 0x20)  { j += (size_t)snprintf(out+j, cap-j, "\\u%04x", ch); }
        else                 out[j++] = (char)ch;
    }
    out[j] = '\0';
}

/* ====================================================================
 * Path helpers
 * ==================================================================== */
static void get_db_path(char *buf, size_t n) {
    const char *e = getenv(DB_ENV);
    if(e) { snprintf(buf, n, "%s", e); return; }
    const char *h = getenv("HOME");
    if(!h) h = "/tmp";
    snprintf(buf, n, "%s%s", h, DB_SUBPATH);
}
static void get_cache_dir(char *buf, size_t n) {
    const char *e = getenv(CACHE_ENV);
    if(e) { snprintf(buf, n, "%s", e); return; }
    const char *h = getenv("HOME");
    if(!h) h = "/tmp";
    snprintf(buf, n, "%s%s", h, CACHE_SUBPATH);
}

static int mkdirs(const char *path) {
    char tmp[PATH_MAX]; size_t len;
    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if(tmp[len-1]=='/') tmp[--len]='\0';
    for(char *p = tmp+1; *p; p++) {
        if(*p=='/') { *p='\0'; mkdir(tmp, 0755); *p='/'; }
    }
    return mkdir(tmp, 0755);
}

/* ====================================================================
 * DB bootstrap
 * ==================================================================== */
static sqlite3 *db_open(const char *path) {
    /* ensure parent dir exists */
    char parent[PATH_MAX]; snprintf(parent, sizeof(parent), "%s", path);
    char *sl = strrchr(parent, '/');
    if(sl) { *sl='\0'; mkdirs(parent); }

    sqlite3 *db = NULL;
    if(sqlite3_open(path, &db) != SQLITE_OK) {
        fprintf(stderr, "error: cannot open model DB at %s: %s\n",
                path, sqlite3_errmsg(db));
        sqlite3_close(db); return NULL;
    }
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
    sqlite3_exec(db, "PRAGMA foreign_keys=ON;",  NULL, NULL, NULL);

    const char *schema =
        "CREATE TABLE IF NOT EXISTS models ("
        "  id          TEXT PRIMARY KEY,"
        "  name        TEXT NOT NULL,"
        "  description TEXT,"
        "  format      TEXT NOT NULL,"       /* gguf | safetensors | onnx | bin */
        "  sha256      TEXT UNIQUE,"
        "  size_mb     REAL,"
        "  fpq_sha256  TEXT,"
        "  fpq_size_mb REAL,"
        "  added_at    INTEGER NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS sources ("
        "  model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,"
        "  url      TEXT NOT NULL,"
        "  priority INTEGER NOT NULL DEFAULT 50,"
        "  PRIMARY KEY(model_id, url)"
        ");"
        "CREATE TABLE IF NOT EXISTS recipe_models ("
        "  recipe_code TEXT NOT NULL,"
        "  model_id    TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,"
        "  role        TEXT,"                 /* transcribe | embed | infer | score */
        "  PRIMARY KEY(recipe_code, model_id)"
        ");"
        "CREATE VIRTUAL TABLE IF NOT EXISTS models_fts USING fts5("
        "  id, name, description,"
        "  content=models, content_rowid=rowid"
        ");";
    char *err = NULL;
    if(sqlite3_exec(db, schema, NULL, NULL, &err) != SQLITE_OK) {
        fprintf(stderr, "error: schema: %s\n", err);
        sqlite3_free(err); sqlite3_close(db); return NULL;
    }
    return db;
}

/* ====================================================================
 * Built-in model registry
 * ==================================================================== */
typedef struct {
    const char *id;
    const char *name;
    const char *description;
    const char *format;
    double      size_mb;
    const char *sources[6];  /* NULL-terminated */
    const char *recipes[8];  /* recipe codes, NULL-terminated */
    const char *role;
} BuiltinModel;

static const BuiltinModel BUILTIN_MODELS[] = {
    {
        "whisper-large-v3",
        "Whisper Large v3",
        "OpenAI Whisper large-v3. 99-language ASR, 2.9 GB GGUF. Used by all A/M/V/R series recipes.",
        "gguf", 2941.0,
        { "hf://openai/whisper-large-v3-GGUF/whisper-large-v3-q5_k_m.gguf",
          "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_k_m.bin",
          NULL },
        { "A1","A2","A3","M1" }, "transcribe"
    },
    {
        "whisper-base",
        "Whisper Base",
        "OpenAI Whisper base model. 148 MB GGUF. Fast transcription for quick briefs (A1 default on low-memory systems).",
        "gguf", 148.0,
        { "hf://openai/whisper-base-GGUF/whisper-base-q5_k_m.gguf",
          "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_k_m.bin",
          NULL },
        { "A1", NULL }, "transcribe"
    },
    {
        "llama-3-8b-instruct",
        "Llama 3 8B Instruct",
        "Meta Llama 3 8B Instruct GGUF Q4_K_M. 4.7 GB. Brief extraction, repurpose, and summary stages.",
        "gguf", 4700.0,
        { "hf://meta-llama/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
          NULL },
        { "A2","A3","M1","P1","P2","V1","R1", NULL }, "infer"
    },
    {
        "llama-3-8b-instruct-fpq",
        "Llama 3 8B Instruct (FPQ compressed)",
        "bonfyre-fpq INT8 compressed Llama 3 8B Instruct. ~1.2 GB. Drop-in replacement; load with BonfyreFPQ proxy.",
        "bin", 1200.0,
        { "swarm://llama-3-8b-instruct-fpq", NULL },
        { "A2","A3","M1" }, "infer"
    },
    {
        "nomic-embed-text",
        "Nomic Embed Text v1.5",
        "Nomic AI embedding model. 274 MB GGUF. Used by BonfyreEmbed for vector indexing in A2/A3/R1.",
        "gguf", 274.0,
        { "hf://nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf",
          NULL },
        { "A2","A3","R1" }, "embed"
    },
    {
        "bge-reranker-base",
        "BGE Reranker Base",
        "BAAI BGE reranker base. 278 MB GGUF. FPQ scoring stage in A3 full pipeline.",
        "gguf", 278.0,
        { "hf://BAAI/bge-reranker-base-GGUF/bge-reranker-base-q5_k_m.gguf",
          NULL },
        { "A3" }, "score"
    },
    {
        "pyannote-speaker-segmentation",
        "Pyannote Speaker Segmentation 3.1",
        "Speaker diarisation model. 17 MB ONNX. BonfyreSegment stage. Required for A3/M1 multi-speaker.",
        "onnx", 17.0,
        { "hf://pyannote/speaker-segmentation-3.1/pytorch_model.bin",
          NULL },
        { "A3","M1" }, "score"
    },
    {
        "silero-vad",
        "Silero VAD",
        "Voice activity detection. 1.8 MB ONNX. BonfyreIngest stage gates silence removal before transcription.",
        "onnx", 1.8,
        { "hf://snakers4/silero-vad/silero_vad.onnx",
          NULL },
        { "A1","A2","A3","M1","V1", NULL }, "score"
    },
    {
        "whisper-medium",
        "Whisper Medium",
        "OpenAI Whisper medium. 769 MB GGUF. Balanced accuracy/speed; A2 archive default.",
        "gguf", 769.0,
        { "hf://openai/whisper-medium-GGUF/whisper-medium-q5_k_m.gguf",
          "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_k_m.bin",
          NULL },
        { "A2","V1" }, "transcribe"
    },
    {
        "llama-3-70b-instruct-fpq",
        "Llama 3 70B Instruct (FPQ compressed)",
        "bonfyre-fpq INT8 compressed Llama 3 70B Instruct. ~9.2 GB. A3 --tier pro inference stage.",
        "bin", 9200.0,
        { "swarm://llama-3-70b-instruct-fpq",
          "hf://meta-llama/Meta-Llama-3-70B-Instruct-GGUF/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf",
          NULL },
        { "A3" }, "infer"
    },
};
#define N_BUILTIN_MODELS (int)(sizeof(BUILTIN_MODELS)/sizeof(BUILTIN_MODELS[0]))

/* ====================================================================
 * Format helpers
 * ==================================================================== */
static void fmt_size(double mb, char *buf, size_t n) {
    if(mb >= 1024.0) snprintf(buf, n, "%.1f GB", mb/1024.0);
    else             snprintf(buf, n, "%.0f MB", mb);
}
static const char *scheme_label(const char *url) {
    if(!url) return "?";
    if(strncmp(url,"swarm://",8)==0) return "swarm";
    if(strncmp(url,"file://",7)==0)  return "local";
    if(strncmp(url,"s3://",5)==0)    return "s3";
    if(strncmp(url,"hf://",5)==0)    return "huggingface";
    return "https";
}

/* ====================================================================
 * Pull: source resolution
 * ==================================================================== */

/* Convert hf://owner/repo/filename → HTTPS URL */
static void hf_to_https(const char *hf, char *out, size_t n) {
    /* hf://owner/repo/file  →  https://huggingface.co/owner/repo/resolve/main/file */
    const char *p = hf + 5; /* skip "hf://" */
    char owner[128], repo[256], file[512];
    /* parse owner/repo/file — repo may contain slashes */
    const char *sl1 = strchr(p, '/');
    if(!sl1) { snprintf(out, n, ""); return; }
    snprintf(owner, sizeof(owner), "%.*s", (int)(sl1-p), p);
    p = sl1+1;
    const char *sl2 = strchr(p, '/');
    if(!sl2) { snprintf(out, n, ""); return; }
    snprintf(repo, sizeof(repo), "%.*s", (int)(sl2-p), p);
    snprintf(file, sizeof(file), "%s", sl2+1);
    snprintf(out, n, "https://huggingface.co/%s/%s/resolve/main/%s", owner, repo, file);
}

/* Invoke curl to download a URL to dest path.
 * No libcurl — just exec curl which is universally available on macOS/Linux.
 * Falls back to wget if curl not found. */
static int download_url(const char *url, const char *dest) {
    char https_url[2048];
    const char *actual_url = url;

    if(strncmp(url,"hf://",5)==0) {
        hf_to_https(url, https_url, sizeof(https_url));
        if(!https_url[0]) { fprintf(stderr,"error: malformed hf:// URL: %s\n",url); return -1; }
        actual_url = https_url;
    } else if(strncmp(url,"file://",7)==0) {
        /* local copy */
        const char *src = url+7;
        pid_t pid = fork();
        if(pid==0) { execlp("cp","cp",src,dest,(char*)NULL); _exit(127); }
        int st; waitpid(pid,&st,0);
        return WIFEXITED(st)&&WEXITSTATUS(st)==0 ? 0 : -1;
    } else if(strncmp(url,"swarm://",8)==0) {
        /* delegate to bonfyre-swarm */
        const char *hash = url+8;
        pid_t pid = fork();
        if(pid==0) {
            execlp("bonfyre-swarm","bonfyre-swarm","pull", hash, "--out", dest, (char*)NULL);
            _exit(127);
        }
        int st; waitpid(pid,&st,0);
        return WIFEXITED(st)&&WEXITSTATUS(st)==0 ? 0 : -1;
    } else if(strncmp(url,"s3://",5)==0) {
        /* delegate to aws cli or s5cmd */
        pid_t pid = fork();
        if(pid==0) {
            execlp("aws","aws","s3","cp",url,dest,(char*)NULL);
            _exit(127);
        }
        int st; waitpid(pid,&st,0);
        return WIFEXITED(st)&&WEXITSTATUS(st)==0 ? 0 : -1;
    }

    /* HTTPS: try curl, then wget */
    fprintf(stderr, "  → downloading from %s\n", actual_url);

    /* Check for curl */
    if(access("/usr/bin/curl",X_OK)==0 || access("/usr/local/bin/curl",X_OK)==0) {
        pid_t pid = fork();
        if(pid==0) {
            execlp("curl","curl","-fsSL","--progress-bar",
                   "-o", dest, actual_url, (char*)NULL);
            _exit(127);
        }
        int st; waitpid(pid,&st,0);
        if(WIFEXITED(st)&&WEXITSTATUS(st)==0) return 0;
        return -1;
    }
    /* Fallback: wget */
    pid_t pid = fork();
    if(pid==0) {
        execlp("wget","wget","-q","--show-progress","-O",dest,actual_url,(char*)NULL);
        _exit(127);
    }
    int st; waitpid(pid,&st,0);
    return WIFEXITED(st)&&WEXITSTATUS(st)==0 ? 0 : -1;
}

/* ====================================================================
 * Seed built-ins into DB
 * ==================================================================== */
static void seed_builtins(sqlite3 *db) {
    for(int i = 0; i < N_BUILTIN_MODELS; i++) {
        const BuiltinModel *m = &BUILTIN_MODELS[i];
        sqlite3_stmt *st;
        sqlite3_prepare_v2(db,
            "INSERT OR IGNORE INTO models(id,name,description,format,sha256,size_mb,added_at)"
            " VALUES(?,?,?,?,?,?,?)", -1, &st, NULL);
        sqlite3_bind_text(st,1,m->id,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,2,m->name,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,3,m->description,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,4,m->format,-1,SQLITE_STATIC);
        /* sha256 unknown until first pull — store NULL */
        sqlite3_bind_null(st,5);
        sqlite3_bind_double(st,6,m->size_mb);
        sqlite3_bind_int64(st,7,(sqlite3_int64)time(NULL));
        sqlite3_step(st); sqlite3_finalize(st);

        /* sources */
        for(int s = 0; m->sources[s]; s++) {
            sqlite3_prepare_v2(db,
                "INSERT OR IGNORE INTO sources(model_id,url,priority) VALUES(?,?,?)",
                -1, &st, NULL);
            sqlite3_bind_text(st,1,m->id,-1,SQLITE_STATIC);
            sqlite3_bind_text(st,2,m->sources[s],-1,SQLITE_STATIC);
            sqlite3_bind_int(st,3,s);
            sqlite3_step(st); sqlite3_finalize(st);
        }

        /* recipe associations */
        for(int r = 0; m->recipes[r]; r++) {
            sqlite3_prepare_v2(db,
                "INSERT OR IGNORE INTO recipe_models(recipe_code,model_id,role) VALUES(?,?,?)",
                -1, &st, NULL);
            sqlite3_bind_text(st,1,m->recipes[r],-1,SQLITE_STATIC);
            sqlite3_bind_text(st,2,m->id,-1,SQLITE_STATIC);
            sqlite3_bind_text(st,3,m->role,-1,SQLITE_STATIC);
            sqlite3_step(st); sqlite3_finalize(st);
        }

        /* FTS sync */
        sqlite3_prepare_v2(db,
            "INSERT OR REPLACE INTO models_fts(rowid,id,name,description)"
            " SELECT rowid,id,name,description FROM models WHERE id=?",
            -1, &st, NULL);
        sqlite3_bind_text(st,1,m->id,-1,SQLITE_STATIC);
        sqlite3_step(st); sqlite3_finalize(st);
    }
}

/* ====================================================================
 * Pull helpers
 * ==================================================================== */

/* Returns 1 if model file is already in cache and hash matches */
static int cache_hit(const char *model_id, const char *expected_sha256,
                     const char *cache_dir, char *hit_path, size_t hp_len) {
    /* Try <cache_dir>/<id>.gguf, <id>.bin, <id>.onnx, <id> */
    static const char *exts[] = { "gguf", "bin", "onnx", "safetensors", "", NULL };
    for(int i = 0; exts[i]; i++) {
        char path[PATH_MAX];
        if(exts[i][0])
            snprintf(path, sizeof(path), "%s/%s.%s", cache_dir, model_id, exts[i]);
        else
            snprintf(path, sizeof(path), "%s/%s", cache_dir, model_id);

        if(access(path, F_OK) != 0) continue;

        /* File exists. If sha256 is known and not "pending", verify */
        if(expected_sha256 && strcmp(expected_sha256,"pending")!=0) {
            char actual[65];
            if(sha256_file(path, actual) == 0 && strcmp(actual, expected_sha256)==0) {
                if(hit_path) snprintf(hit_path, hp_len, "%s", path);
                return 1;
            }
            /* hash mismatch — treat as miss */
            continue;
        }
        /* hash unknown — accept presence */
        if(hit_path) snprintf(hit_path, hp_len, "%s", path);
        return 1;
    }
    return 0;
}

static int pull_model(sqlite3 *db, const char *model_id,
                      const char *cache_dir, int verbose) {
    /* Look up model */
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT name, format, sha256, size_mb FROM models WHERE id=?",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,model_id,-1,SQLITE_STATIC);
    int rc = sqlite3_step(st);
    if(rc != SQLITE_ROW) {
        fprintf(stderr, "error: model '%s' not in registry. "
                "Run: bonfyre-model add <manifest.json>\n", model_id);
        sqlite3_finalize(st); return 1;
    }
    char name[256], format[32], sha256[65];
    double size_mb;
    snprintf(name,   sizeof(name),   "%s", (const char*)sqlite3_column_text(st,0));
    snprintf(format, sizeof(format), "%s", (const char*)sqlite3_column_text(st,1));
    snprintf(sha256, sizeof(sha256), "%s", sqlite3_column_text(st,2) ?
             (const char*)sqlite3_column_text(st,2) : "pending");
    size_mb = sqlite3_column_double(st,3);
    sqlite3_finalize(st);

    char sz[32]; fmt_size(size_mb, sz, sizeof(sz));

    /* Check cache */
    char hit[PATH_MAX];
    if(cache_hit(model_id, sha256, cache_dir, hit, sizeof(hit))) {
        printf("  ✓ %-30s  already cached  %s\n", model_id, hit);
        return 0;
    }

    printf("  ↓ %-30s  %s (%s)  ...\n", model_id, name, sz);
    fflush(stdout);

    /* Ensure cache dir exists */
    mkdirs(cache_dir);

    /* Get sources ordered by priority */
    sqlite3_prepare_v2(db,
        "SELECT url FROM sources WHERE model_id=? ORDER BY priority ASC",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,model_id,-1,SQLITE_STATIC);

    char dest[PATH_MAX];
    snprintf(dest, sizeof(dest), "%s/%s.%s", cache_dir, model_id, format);

    int pulled = 0;
    while(sqlite3_step(st) == SQLITE_ROW) {
        const char *url = (const char*)sqlite3_column_text(st,0);
        if(verbose) fprintf(stderr,"  trying source [%s]: %s\n", scheme_label(url), url);

        /* Skip swarm/s3/file sources if tool not available —
           check by trying the download and falling through on failure */
        int r = download_url(url, dest);
        if(r == 0) {
            pulled = 1;
            break;
        }
        fprintf(stderr,"  source failed, trying next...\n");
        unlink(dest); /* clean partial file */
    }
    sqlite3_finalize(st);

    if(!pulled) {
        fprintf(stderr,"error: all sources failed for '%s'.\n"
                "  Add a source: bonfyre-model source add %s <url>\n",
                model_id, model_id);
        return 1;
    }

    /* Verify SHA-256 if known */
    if(strcmp(sha256,"pending")!=0) {
        char actual[65];
        if(sha256_file(dest, actual) != 0) {
            fprintf(stderr,"error: cannot hash downloaded file %s\n", dest);
            return 1;
        }
        if(strcmp(actual, sha256) != 0) {
            fprintf(stderr,"error: SHA-256 mismatch for '%s'\n"
                    "  expected: %s\n  got:      %s\n"
                    "  Removing corrupt file.\n",
                    model_id, sha256, actual);
            unlink(dest);
            return 1;
        }
        printf("  ✓ %-30s  verified  %s\n", model_id, dest);
    } else {
        /* Record actual hash now that we have the file */
        char actual[65];
        if(sha256_file(dest, actual) == 0) {
            sqlite3_prepare_v2(db,
                "UPDATE models SET sha256=? WHERE id=?", -1, &st, NULL);
            sqlite3_bind_text(st,1,actual,-1,SQLITE_STATIC);
            sqlite3_bind_text(st,2,model_id,-1,SQLITE_STATIC);
            sqlite3_step(st); sqlite3_finalize(st);
        }
        printf("  ✓ %-30s  saved  %s\n", model_id, dest);
    }
    return 0;
}

/* ====================================================================
 * Command implementations
 * ==================================================================== */

static void cmd_help(void) {
    printf(
        "bonfyre-model %s — AI model dependency manager\n\n"
        "COMMANDS\n"
        "  list                       list all registered models\n"
        "  show <id>                  print full model record\n"
        "  pull <id>                  ensure model is in local cache\n"
        "  pull --recipe <code>       pull all models required by a recipe\n"
        "  add <manifest.json>        register a model from JSON manifest\n"
        "  verify <id>                re-check SHA-256 of cached file\n"
        "  path <id>                  print absolute cache path\n"
        "  rm <id>                    remove from registry\n"
        "  rm --purge <id>            remove from registry + delete cache\n"
        "  search <query>             full-text search over model names\n"
        "  sources <id>               list pull sources for a model\n"
        "  source add <id> <url>      add a pull source URL\n"
        "  source rm <id> <url>       remove a pull source URL\n"
        "  ls-cache                   list cached files with sizes\n"
        "  status                     registry + cache stats\n"
        "  help                       this message\n\n"
        "SOURCE SCHEMES (evaluated in priority order)\n"
        "  swarm://hash               bonfyre-swarm local peer\n"
        "  file:///abs/path           already on disk\n"
        "  s3://bucket/key            S3-compatible store (requires aws cli)\n"
        "  hf://owner/repo/file       huggingface.co (plain HTTPS, no SDK)\n"
        "  https://host/path          direct download\n\n"
        "ENVIRONMENT\n"
        "  BONFYRE_MODEL_DB           override model DB path\n"
        "  BONFYRE_MODEL_CACHE        override model cache dir\n"
        "  BONFYRE_MODEL_SKIP_CHECK   skip model checks in bonfyre-run (set to 1)\n\n"
        "EXAMPLES\n"
        "  bonfyre-model list\n"
        "  bonfyre-model pull whisper-large-v3\n"
        "  bonfyre-model pull --recipe A3\n"
        "  bonfyre-model source add whisper-large-v3 file:///data/models/whisper.gguf\n"
        "  bonfyre-model verify whisper-large-v3\n"
        "  bonfyre-model path whisper-large-v3\n",
        VERSION
    );
}

static int cmd_list(sqlite3 *db) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT m.id, m.name, m.format, m.size_mb, "
        "       (SELECT COUNT(*) FROM recipe_models r WHERE r.model_id=m.id) AS recipes "
        "FROM models m ORDER BY m.id",
        -1, &st, NULL);

    char cache_dir[PATH_MAX]; get_cache_dir(cache_dir, sizeof(cache_dir));

    printf("%-32s  %-8s  %-8s  %-7s  %s\n",
           "ID", "FORMAT", "SIZE", "RECIPES", "CACHED");
    printf("%-32s  %-8s  %-8s  %-7s  %s\n",
           "--------------------------------", "--------", "--------", "-------", "------");

    int count = 0;
    while(sqlite3_step(st) == SQLITE_ROW) {
        const char *id     = (const char*)sqlite3_column_text(st,0);
        const char *name   = (const char*)sqlite3_column_text(st,1);
        const char *fmt    = (const char*)sqlite3_column_text(st,2);
        double      sz     = sqlite3_column_double(st,3);
        int         nrec   = sqlite3_column_int(st,4);
        char        szs[16]; fmt_size(sz, szs, sizeof(szs));
        int         cached = cache_hit(id, NULL, cache_dir, NULL, 0);
        printf("%-32s  %-8s  %-8s  %-7d  %s\n",
               id, fmt, szs, nrec, cached ? "yes" : "-");
        (void)name;
        count++;
    }
    sqlite3_finalize(st);
    printf("\n%d model(s) registered.\n", count);
    return 0;
}

static int cmd_show(sqlite3 *db, const char *id) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT id,name,description,format,sha256,size_mb,fpq_sha256,fpq_size_mb,added_at"
        " FROM models WHERE id=?", -1, &st, NULL);
    sqlite3_bind_text(st,1,id,-1,SQLITE_STATIC);
    if(sqlite3_step(st) != SQLITE_ROW) {
        fprintf(stderr,"error: model '%s' not found\n", id);
        sqlite3_finalize(st); return 1;
    }

    char cache_dir[PATH_MAX]; get_cache_dir(cache_dir, sizeof(cache_dir));
    char hit[PATH_MAX]; int cached = cache_hit(id, NULL, cache_dir, hit, sizeof(hit));

    printf("{\n");
    printf("  \"id\": \"%s\",\n",           sqlite3_column_text(st,0));
    printf("  \"name\": \"%s\",\n",         sqlite3_column_text(st,1));
    const char *desc = (const char*)sqlite3_column_text(st,2);
    char edesc[512]; json_escape(desc?desc:"", edesc, sizeof(edesc));
    printf("  \"description\": \"%s\",\n",  edesc);
    printf("  \"format\": \"%s\",\n",       sqlite3_column_text(st,3));
    const char *sha_raw = (const char*)sqlite3_column_text(st,4);
    printf("  \"sha256\": \"%s\",\n", sha_raw ? sha_raw : "pending");
    double sz = sqlite3_column_double(st,5);
    char szs[16]; fmt_size(sz,szs,sizeof(szs));
    printf("  \"size\": \"%s\",\n",         szs);
    const char *fsz = (const char*)sqlite3_column_text(st,7);
    if(fsz) {
        char fszs[16]; fmt_size(atof(fsz),fszs,sizeof(fszs));
        printf("  \"fpq_sha256\": \"%s\",\n",  sqlite3_column_text(st,6));
        printf("  \"fpq_size\": \"%s\",\n",    fszs);
    }
    printf("  \"cached\": %s,\n",           cached ? "true" : "false");
    if(cached) { char ep[PATH_MAX]; json_escape(hit,ep,sizeof(ep));
                 printf("  \"cache_path\": \"%s\",\n", ep); }
    sqlite3_finalize(st);

    /* sources */
    sqlite3_prepare_v2(db,
        "SELECT url, priority FROM sources WHERE model_id=? ORDER BY priority",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,id,-1,SQLITE_STATIC);
    printf("  \"sources\": [\n");
    int first = 1;
    while(sqlite3_step(st)==SQLITE_ROW) {
        if(!first) printf(",\n"); first=0;
        char eu[1024]; json_escape((const char*)sqlite3_column_text(st,0),eu,sizeof(eu));
        printf("    { \"url\": \"%s\", \"scheme\": \"%s\" }",
               eu, scheme_label((const char*)sqlite3_column_text(st,0)));
    }
    printf("\n  ],\n");
    sqlite3_finalize(st);

    /* recipe associations */
    sqlite3_prepare_v2(db,
        "SELECT recipe_code, role FROM recipe_models WHERE model_id=? ORDER BY recipe_code",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,id,-1,SQLITE_STATIC);
    printf("  \"used_by\": [");
    first=1;
    while(sqlite3_step(st)==SQLITE_ROW)  {
        if(!first) printf(", "); first=0;
        printf("\"%s(%s)\"", sqlite3_column_text(st,0), sqlite3_column_text(st,1));
    }
    printf("]\n}\n");
    sqlite3_finalize(st);
    return 0;
}

static int cmd_pull(sqlite3 *db, const char *model_id) {
    char cache_dir[PATH_MAX]; get_cache_dir(cache_dir, sizeof(cache_dir));
    return pull_model(db, model_id, cache_dir, 1);
}

static int cmd_pull_recipe(sqlite3 *db, const char *recipe_code) {
    printf("Pulling models for recipe %s...\n", recipe_code);
    fflush(stdout);

    /* First seed builtins so recipe associations are present */
    seed_builtins(db);

    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT model_id, role FROM recipe_models WHERE recipe_code=? ORDER BY model_id",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,recipe_code,-1,SQLITE_STATIC);

    char cache_dir[PATH_MAX]; get_cache_dir(cache_dir, sizeof(cache_dir));
    int errors = 0, count = 0;
    while(sqlite3_step(st)==SQLITE_ROW) {
        const char *mid = (const char*)sqlite3_column_text(st,0);
        errors += pull_model(db, mid, cache_dir, 0);
        count++;
    }
    sqlite3_finalize(st);

    if(count == 0) {
        fprintf(stderr,"warning: no models registered for recipe '%s'.\n"
                "  Run: bonfyre-model list   to see all models.\n", recipe_code);
        return 1;
    }
    printf("\n%d model(s) checked for recipe %s. %d error(s).\n",
           count, recipe_code, errors);
    return errors ? 1 : 0;
}

static int cmd_verify(sqlite3 *db, const char *model_id) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT sha256, format FROM models WHERE id=?", -1, &st, NULL);
    sqlite3_bind_text(st,1,model_id,-1,SQLITE_STATIC);
    if(sqlite3_step(st) != SQLITE_ROW) {
        fprintf(stderr,"error: model '%s' not found\n", model_id);
        sqlite3_finalize(st); return 1;
    }
    char sha256[65], format[32];
    snprintf(sha256, sizeof(sha256), "%s", (const char*)sqlite3_column_text(st,0));
    snprintf(format, sizeof(format), "%s", (const char*)sqlite3_column_text(st,1));
    sqlite3_finalize(st);

    char cache_dir[PATH_MAX]; get_cache_dir(cache_dir, sizeof(cache_dir));
    char hit[PATH_MAX];
    if(!cache_hit(model_id, NULL, cache_dir, hit, sizeof(hit))) {
        fprintf(stderr,"error: '%s' not in cache. Run: bonfyre-model pull %s\n",
                model_id, model_id);
        return 1;
    }

    if(strcmp(sha256,"pending")==0) {
        printf("  pending  no expected hash stored — computing and storing...\n");
        char actual[65];
        if(sha256_file(hit, actual)!=0) {
            fprintf(stderr,"error: cannot hash file %s\n", hit); return 1;
        }
        sqlite3_prepare_v2(db,
            "UPDATE models SET sha256=? WHERE id=?", -1, &st, NULL);
        sqlite3_bind_text(st,1,actual,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,2,model_id,-1,SQLITE_STATIC);
        sqlite3_step(st); sqlite3_finalize(st);
        printf("  stored   %s  %s\n", actual, hit);
        return 0;
    }

    printf("  verifying  %s  ...\n", hit); fflush(stdout);
    char actual[65];
    if(sha256_file(hit, actual)!=0) {
        fprintf(stderr,"error: cannot hash file %s\n", hit); return 1;
    }
    if(strcmp(actual, sha256)==0) {
        printf("  ✓ ok      %s\n", sha256);
        return 0;
    }
    fprintf(stderr,"  ✗ FAIL    expected: %s\n               got:      %s\n",
            sha256, actual);
    return 1;
}

static int cmd_path(sqlite3 *db, const char *model_id) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT format FROM models WHERE id=?", -1, &st, NULL);
    sqlite3_bind_text(st,1,model_id,-1,SQLITE_STATIC);
    if(sqlite3_step(st) != SQLITE_ROW) {
        fprintf(stderr,"error: model '%s' not found\n", model_id);
        sqlite3_finalize(st); return 1;
    }
    sqlite3_finalize(st);

    char cache_dir[PATH_MAX]; get_cache_dir(cache_dir, sizeof(cache_dir));
    char hit[PATH_MAX];
    if(cache_hit(model_id, NULL, cache_dir, hit, sizeof(hit))) {
        printf("%s\n", hit);
        return 0;
    }
    fprintf(stderr,"error: '%s' not in cache. Run: bonfyre-model pull %s\n",
            model_id, model_id);
    return 1;
}

static int cmd_rm(sqlite3 *db, const char *model_id, int purge) {
    if(purge) {
        char cache_dir[PATH_MAX]; get_cache_dir(cache_dir, sizeof(cache_dir));
        char hit[PATH_MAX];
        if(cache_hit(model_id, NULL, cache_dir, hit, sizeof(hit))) {
            if(unlink(hit)==0) printf("  deleted cache file: %s\n", hit);
            else perror("unlink");
        }
    }
    char *err=NULL;
    char sql[512];
    snprintf(sql,sizeof(sql),"DELETE FROM models WHERE id='%s'", model_id);
    /* NOTE: Using parameterized query for safety */
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,"DELETE FROM models WHERE id=?",-1,&st,NULL);
    sqlite3_bind_text(st,1,model_id,-1,SQLITE_STATIC);
    sqlite3_step(st); sqlite3_finalize(st);
    int changed = sqlite3_changes(db);
    (void)err; (void)sql;
    if(changed==0) { fprintf(stderr,"model '%s' not found\n",model_id); return 1; }
    printf("  removed: %s%s\n", model_id, purge?" (cache purged)":"");
    return 0;
}

static int cmd_search(sqlite3 *db, const char *query) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT m.id, m.name, m.format, m.size_mb FROM models m "
        "JOIN models_fts f ON m.rowid = f.rowid "
        "WHERE models_fts MATCH ? ORDER BY rank",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,query,-1,SQLITE_STATIC);
    int count=0;
    while(sqlite3_step(st)==SQLITE_ROW) {
        const char *id  = (const char*)sqlite3_column_text(st,0);
        const char *nm  = (const char*)sqlite3_column_text(st,1);
        const char *fmt = (const char*)sqlite3_column_text(st,2);
        double sz       = sqlite3_column_double(st,3);
        char szs[16]; fmt_size(sz,szs,sizeof(szs));
        printf("%-32s  %-8s  %-8s  %s\n", id, fmt, szs, nm);
        count++;
    }
    sqlite3_finalize(st);
    if(count==0) printf("no results for '%s'\n", query);
    return 0;
}

static int cmd_sources(sqlite3 *db, const char *model_id) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT url, priority FROM sources WHERE model_id=? ORDER BY priority",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,model_id,-1,SQLITE_STATIC);
    int count=0;
    printf("Sources for '%s' (priority order):\n", model_id);
    while(sqlite3_step(st)==SQLITE_ROW) {
        printf("  [%d] [%-11s] %s\n",
               sqlite3_column_int(st,1),
               scheme_label((const char*)sqlite3_column_text(st,0)),
               sqlite3_column_text(st,0));
        count++;
    }
    sqlite3_finalize(st);
    if(count==0) printf("  (no sources registered)\n");
    return 0;
}

static int cmd_source_add(sqlite3 *db, const char *model_id, const char *url) {
    /* Get max priority */
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT COALESCE(MAX(priority),0)+10 FROM sources WHERE model_id=?",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,model_id,-1,SQLITE_STATIC);
    int pri = 50;
    if(sqlite3_step(st)==SQLITE_ROW) pri = sqlite3_column_int(st,0);
    sqlite3_finalize(st);

    sqlite3_prepare_v2(db,
        "INSERT OR REPLACE INTO sources(model_id,url,priority) VALUES(?,?,?)",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,model_id,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,2,url,-1,SQLITE_STATIC);
    sqlite3_bind_int(st,3,pri);
    sqlite3_step(st); sqlite3_finalize(st);
    printf("  added [%s] %s\n", scheme_label(url), url);
    return 0;
}

static int cmd_source_rm(sqlite3 *db, const char *model_id, const char *url) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "DELETE FROM sources WHERE model_id=? AND url=?", -1, &st, NULL);
    sqlite3_bind_text(st,1,model_id,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,2,url,-1,SQLITE_STATIC);
    sqlite3_step(st); sqlite3_finalize(st);
    int changed = sqlite3_changes(db);
    if(changed==0) { fprintf(stderr,"source not found\n"); return 1; }
    printf("  removed: %s\n", url);
    return 0;
}

static int cmd_add(sqlite3 *db, const char *json_path) {
    FILE *f = fopen(json_path, "r");
    if(!f) { perror(json_path); return 1; }
    char buf[MAX_JSON]; size_t n = fread(buf,1,sizeof(buf)-1,f); fclose(f);
    buf[n]='\0';

    /* Minimal JSON field extraction — no external parser needed */
    #define JFIELD(key, dest, dsz) do { \
        const char *_p = strstr(buf,"\"" key "\""); \
        dest[0]='\0'; \
        if(_p) { \
            _p = strchr(_p,':'); if(_p) { \
                _p++; while(*_p==' '||*_p=='\t') _p++; \
                if(*_p=='"') { _p++; size_t _i=0; \
                    while(*_p&&*_p!='"'&&_i<(dsz)-1) dest[_i++]=*_p++; \
                    dest[_i]='\0'; } \
            } \
        } \
    } while(0)

    char id[128], name[256], desc[512], fmt[32], sha256[65], size_s[32],
         fpq_sha[65], fpq_sz[32];
    JFIELD("id",          id,       sizeof(id));
    JFIELD("name",        name,     sizeof(name));
    JFIELD("description", desc,     sizeof(desc));
    JFIELD("format",      fmt,      sizeof(fmt));
    JFIELD("sha256",      sha256,   sizeof(sha256));
    JFIELD("size_mb",     size_s,   sizeof(size_s));
    JFIELD("fpq_sha256",  fpq_sha,  sizeof(fpq_sha));
    JFIELD("fpq_size_mb", fpq_sz,   sizeof(fpq_sz));
    #undef JFIELD

    if(!id[0] || !name[0] || !fmt[0]) {
        fprintf(stderr,"error: manifest must have id, name, format fields\n");
        return 1;
    }
    if(!sha256[0]) snprintf(sha256, sizeof(sha256), "pending");
    double size_mb  = size_s[0]  ? atof(size_s)  : 0.0;
    double fpq_size = fpq_sz[0]  ? atof(fpq_sz)  : 0.0;

    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "INSERT OR REPLACE INTO models(id,name,description,format,sha256,size_mb,"
        "fpq_sha256,fpq_size_mb,added_at) VALUES(?,?,?,?,?,?,?,?,?)",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,id,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,2,name,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,3,desc[0]?desc:NULL,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,4,fmt,-1,SQLITE_STATIC);
    /* use NULL for unknown sha256 so UNIQUE constraint allows multiple unknowns */
    if(sha256[0] && strcmp(sha256,"pending")!=0)
        sqlite3_bind_text(st,5,sha256,-1,SQLITE_STATIC);
    else
        sqlite3_bind_null(st,5);
    sqlite3_bind_double(st,6,size_mb);
    sqlite3_bind_text(st,7,fpq_sha[0]?fpq_sha:NULL,-1,SQLITE_STATIC);
    fpq_size>0 ? sqlite3_bind_double(st,8,fpq_size) : sqlite3_bind_null(st,8);
    sqlite3_bind_int64(st,9,(sqlite3_int64)time(NULL));
    int rc = sqlite3_step(st); sqlite3_finalize(st);

    if(rc != SQLITE_DONE) {
        fprintf(stderr,"error: DB insert failed: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    /* Parse and insert sources array: "sources": ["url1","url2"] */
    const char *src_start = strstr(buf,"\"sources\"");
    if(src_start) {
        src_start = strchr(src_start,'[');
        int pri=0;
        while(src_start && *src_start) {
            src_start = strchr(src_start,'"');
            if(!src_start) break;
            src_start++;
            const char *end = strchr(src_start,'"');
            if(!end) break;
            char url[1024];
            snprintf(url, sizeof(url), "%.*s", (int)(end-src_start), src_start);
            src_start = end+1;
            if(url[0]) {
                sqlite3_prepare_v2(db,
                    "INSERT OR IGNORE INTO sources(model_id,url,priority) VALUES(?,?,?)",
                    -1, &st, NULL);
                sqlite3_bind_text(st,1,id,-1,SQLITE_STATIC);
                sqlite3_bind_text(st,2,url,-1,SQLITE_STATIC);
                sqlite3_bind_int(st,3,pri++);
                sqlite3_step(st); sqlite3_finalize(st);
            }
            /* stop at ] */
            const char *cl = strchr(src_start,']');
            const char *nx = strchr(src_start,'"');
            if(!nx || (cl && cl<nx)) break;
        }
    }

    /* FTS update */
    sqlite3_prepare_v2(db,
        "INSERT OR REPLACE INTO models_fts(rowid,id,name,description)"
        " SELECT rowid,id,name,description FROM models WHERE id=?",
        -1, &st, NULL);
    sqlite3_bind_text(st,1,id,-1,SQLITE_STATIC);
    sqlite3_step(st); sqlite3_finalize(st);

    printf("  registered: %s  (%s)\n", id, name);
    return 0;
}

static int cmd_ls_cache(void) {
    char cache_dir[PATH_MAX]; get_cache_dir(cache_dir, sizeof(cache_dir));

    /* Use ls -lh via shell — no need to implement recursive walk here */
    char cmd[PATH_MAX+32];
    snprintf(cmd, sizeof(cmd), "ls -lh '%s' 2>/dev/null || echo '(cache empty or not found)'", cache_dir);
    printf("Cache: %s\n\n", cache_dir);
    system(cmd); /* intentionally using system here — read-only ls */
    return 0;
}

static int cmd_status(sqlite3 *db) {
    char db_path[PATH_MAX]; get_db_path(db_path, sizeof(db_path));
    char cache_dir[PATH_MAX]; get_cache_dir(cache_dir, sizeof(cache_dir));

    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,"SELECT COUNT(*) FROM models",-1,&st,NULL);
    sqlite3_step(st); int nmodel = sqlite3_column_int(st,0); sqlite3_finalize(st);
    sqlite3_prepare_v2(db,"SELECT COUNT(*) FROM sources",-1,&st,NULL);
    sqlite3_step(st); int nsrc   = sqlite3_column_int(st,0); sqlite3_finalize(st);
    sqlite3_prepare_v2(db,"SELECT COUNT(*) FROM recipe_models",-1,&st,NULL);
    sqlite3_step(st); int nrec   = sqlite3_column_int(st,0); sqlite3_finalize(st);

    /* Count cached files */
    int ncached=0; double total_mb=0;
    char cache_dir2[PATH_MAX]; get_cache_dir(cache_dir2, sizeof(cache_dir2));
    sqlite3_prepare_v2(db,"SELECT id,format,size_mb FROM models",-1,&st,NULL);
    while(sqlite3_step(st)==SQLITE_ROW) {
        const char *id  = (const char*)sqlite3_column_text(st,0);
        double sz = sqlite3_column_double(st,2);
        if(cache_hit(id,NULL,cache_dir2,NULL,0)) { ncached++; total_mb+=sz; }
    }
    sqlite3_finalize(st);

    char total_s[16]; fmt_size(total_mb, total_s, sizeof(total_s));

    printf("bonfyre-model %s\n\n", VERSION);
    printf("  DB:            %s\n", db_path);
    printf("  Cache:         %s\n", cache_dir);
    printf("  Models:        %d registered\n", nmodel);
    printf("  Sources:       %d configured\n", nsrc);
    printf("  Recipe links:  %d\n", nrec);
    printf("  Cached:        %d / %d  (%s on disk)\n",
           ncached, nmodel, total_s);
    return 0;
}

/* ====================================================================
 * main
 * ==================================================================== */
int main(int argc, char **argv) {
    if(argc < 2 || strcmp(argv[1],"help")==0 || strcmp(argv[1],"--help")==0) {
        cmd_help(); return 0;
    }

    char db_path[PATH_MAX]; get_db_path(db_path, sizeof(db_path));
    sqlite3 *db = db_open(db_path);
    if(!db) return 1;

    /* Always seed built-ins (INSERT OR IGNORE — idempotent) */
    seed_builtins(db);

    const char *cmd = argv[1];
    int ret = 0;

    if(strcmp(cmd,"list")==0) {
        ret = cmd_list(db);

    } else if(strcmp(cmd,"show")==0) {
        if(argc < 3) { fprintf(stderr,"usage: bonfyre-model show <id>\n"); ret=1; }
        else ret = cmd_show(db, argv[2]);

    } else if(strcmp(cmd,"pull")==0) {
        if(argc < 3) {
            fprintf(stderr,"usage: bonfyre-model pull <id> | --recipe <code>\n"); ret=1;
        } else if(strcmp(argv[2],"--recipe")==0) {
            if(argc < 4) { fprintf(stderr,"usage: bonfyre-model pull --recipe <code>\n"); ret=1; }
            else ret = cmd_pull_recipe(db, argv[3]);
        } else {
            ret = cmd_pull(db, argv[2]);
        }

    } else if(strcmp(cmd,"add")==0) {
        if(argc < 3) { fprintf(stderr,"usage: bonfyre-model add <manifest.json>\n"); ret=1; }
        else ret = cmd_add(db, argv[2]);

    } else if(strcmp(cmd,"verify")==0) {
        if(argc < 3) { fprintf(stderr,"usage: bonfyre-model verify <id>\n"); ret=1; }
        else ret = cmd_verify(db, argv[2]);

    } else if(strcmp(cmd,"path")==0) {
        if(argc < 3) { fprintf(stderr,"usage: bonfyre-model path <id>\n"); ret=1; }
        else ret = cmd_path(db, argv[2]);

    } else if(strcmp(cmd,"rm")==0) {
        int purge = 0; const char *id = NULL;
        for(int i=2;i<argc;i++) {
            if(strcmp(argv[i],"--purge")==0) purge=1;
            else id=argv[i];
        }
        if(!id) { fprintf(stderr,"usage: bonfyre-model rm [--purge] <id>\n"); ret=1; }
        else ret = cmd_rm(db, id, purge);

    } else if(strcmp(cmd,"search")==0) {
        if(argc < 3) { fprintf(stderr,"usage: bonfyre-model search <query>\n"); ret=1; }
        else ret = cmd_search(db, argv[2]);

    } else if(strcmp(cmd,"sources")==0) {
        if(argc < 3) { fprintf(stderr,"usage: bonfyre-model sources <id>\n"); ret=1; }
        else ret = cmd_sources(db, argv[2]);

    } else if(strcmp(cmd,"source")==0) {
        if(argc < 4) { fprintf(stderr,"usage: bonfyre-model source add|rm <id> <url>\n"); ret=1; }
        else if(strcmp(argv[2],"add")==0) {
            if(argc < 5) { fprintf(stderr,"usage: bonfyre-model source add <id> <url>\n"); ret=1; }
            else ret = cmd_source_add(db, argv[3], argv[4]);
        } else if(strcmp(argv[2],"rm")==0) {
            if(argc < 5) { fprintf(stderr,"usage: bonfyre-model source rm <id> <url>\n"); ret=1; }
            else ret = cmd_source_rm(db, argv[3], argv[4]);
        } else {
            fprintf(stderr,"unknown source subcommand: %s\n", argv[2]); ret=1;
        }

    } else if(strcmp(cmd,"ls-cache")==0) {
        ret = cmd_ls_cache();

    } else if(strcmp(cmd,"status")==0) {
        ret = cmd_status(db);

    } else if(strcmp(cmd,"version")==0 || strcmp(cmd,"--version")==0) {
        printf("bonfyre-model %s\n", VERSION);

    } else {
        fprintf(stderr,"unknown command: %s\n"
                "Run: bonfyre-model help\n", cmd);
        ret = 1;
    }

    sqlite3_close(db);
    return ret;
}
