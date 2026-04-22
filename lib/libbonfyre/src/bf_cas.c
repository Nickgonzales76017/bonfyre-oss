/*
 * bf_cas.c — Content-Addressable Result Store implementation
 *
 * Provides SHA-256-keyed "Instant Rerun" caching for Bonfyre pipeline runs.
 * If the (input_hash + recipe_level_hash) matches a previous run, the output
 * directory is symlinked from the cache instead of re-executing all binaries.
 */
#define _POSIX_C_SOURCE 200809L

#include "bf_cas.h"
#include "bonfyre.h"   /* BfSha256, bf_sha256_*, bf_ensure_dir */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ── sha256 convenience ─────────────────────────────────────────────── */

static void sha256_to_hex(const uint8_t digest[32], char hex[BF_CAS_HASH_LEN]) {
    static const char hx[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex[2*i]   = hx[digest[i] >> 4];
        hex[2*i+1] = hx[digest[i] & 0xf];
    }
    hex[64] = '\0';
}

static void hash_str(const char *s, char hex[BF_CAS_HASH_LEN]) {
    BfSha256 ctx; bf_sha256_init(&ctx);
    bf_sha256_update(&ctx, (const uint8_t *)s, strlen(s));
    uint8_t d[32]; bf_sha256_final(&ctx, d);
    sha256_to_hex(d, hex);
}

/* ── Init ────────────────────────────────────────────────────────────── */

int bf_cas_init(BfCasCtx *ctx) {
    const char *dir = getenv("BONFYRE_CAS_DIR");
    if (dir && dir[0]) {
        snprintf(ctx->root, sizeof(ctx->root), "%s", dir);
    } else {
        const char *home = getenv("HOME");
        if (!home) home = "/tmp";
        snprintf(ctx->root, sizeof(ctx->root),
                 "%s/.local/share/bonfyre/cas", home);
    }
    bf_ensure_dir(ctx->root);
    return 0;
}

/* ── Hashing helpers ─────────────────────────────────────────────────── */

int bf_cas_hash_file(const char *path, char hex_out[BF_CAS_HASH_LEN]) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    BfSha256 ctx; bf_sha256_init(&ctx);
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        bf_sha256_update(&ctx, buf, n);
    fclose(f);
    uint8_t d[32]; bf_sha256_final(&ctx, d);
    sha256_to_hex(d, hex_out);
    return 0;
}

int bf_cas_hash_levels(const char **const *levels, int n_levels,
                       char hex_out[BF_CAS_HASH_LEN]) {
    /* Hash each level independently: sort binary names, join with '|', SHA-256.
     * Then Merkle-chain the level hashes. */
    BfSha256 chain; bf_sha256_init(&chain);

    for (int L = 0; L < n_levels; L++) {
        /* Build a sorted concatenation of binary names for this level */
        /* Count entries */
        int count = 0;
        while (levels[L][count]) count++;

        /* Simple insertion-sort copy into local array (levels are small) */
        const char *sorted[256]; int sc = 0;
        for (int i = 0; i < count && sc < 256; i++) sorted[sc++] = levels[L][i];
        for (int i = 1; i < sc; i++) {
            const char *k = sorted[i]; int j = i - 1;
            while (j >= 0 && strcmp(sorted[j], k) > 0) { sorted[j+1] = sorted[j]; j--; }
            sorted[j+1] = k;
        }

        BfSha256 lvl; bf_sha256_init(&lvl);
        for (int i = 0; i < sc; i++) {
            bf_sha256_update(&lvl, (const uint8_t *)sorted[i], strlen(sorted[i]));
            bf_sha256_update(&lvl, (const uint8_t *)"|", 1);
        }
        uint8_t ld[32]; bf_sha256_final(&lvl, ld);

        /* Feed level digest into chain */
        bf_sha256_update(&chain, ld, 32);
    }

    uint8_t cd[32]; bf_sha256_final(&chain, cd);
    sha256_to_hex(cd, hex_out);
    return 0;
}

void bf_cas_run_hash(const char input_hex[BF_CAS_HASH_LEN],
                     const char recipe_hex[BF_CAS_HASH_LEN],
                     char run_hash_out[BF_CAS_HASH_LEN]) {
    BfSha256 ctx; bf_sha256_init(&ctx);
    bf_sha256_update(&ctx, (const uint8_t *)input_hex, 64);
    bf_sha256_update(&ctx, (const uint8_t *)":", 1);
    bf_sha256_update(&ctx, (const uint8_t *)recipe_hex, 64);
    uint8_t d[32]; bf_sha256_final(&ctx, d);
    sha256_to_hex(d, run_hash_out);
}

/* ── Internal path helpers ───────────────────────────────────────────── */

static void entry_dir(const BfCasCtx *ctx, const char run_hash[BF_CAS_HASH_LEN],
                      char *out, size_t sz) {
    /* Use first 16 hex chars as directory name */
    char prefix[17]; memcpy(prefix, run_hash, 16); prefix[16] = '\0';
    snprintf(out, sz, "%s/%s", ctx->root, prefix);
}

static void manifest_path(const BfCasCtx *ctx, const char run_hash[BF_CAS_HASH_LEN],
                           char *out, size_t sz) {
    char dir[4096]; entry_dir(ctx, run_hash, dir, sizeof(dir));
    snprintf(out, sz, "%s/run-manifest.json", dir);
}

/* ── Lookup ──────────────────────────────────────────────────────────── */

int bf_cas_lookup(BfCasCtx *ctx,
                  const char run_hash[BF_CAS_HASH_LEN],
                  const char *out_dir,
                  char *manifest_path_out, size_t manifest_sz) {
    char mpath[4096]; manifest_path(ctx, run_hash, mpath, sizeof(mpath));
    if (access(mpath, F_OK) != 0) return 0;    /* cache miss */

    /* Read manifest to find the cached result path */
    FILE *f = fopen(mpath, "r");
    if (!f) return -1;
    char line[4096]; char result_path[4096] = "";
    while (fgets(line, sizeof(line), f)) {
        /* Look for "result_dir": "..." */
        const char *p = strstr(line, "\"result_dir\"");
        if (p) {
            const char *q = strchr(p + 12, '"');
            if (q) { q++; const char *e = strchr(q, '"');
                if (e) { size_t n = (size_t)(e - q);
                    if (n < sizeof(result_path) - 1) {
                        memcpy(result_path, q, n); result_path[n] = '\0'; } } }
        }
    }
    fclose(f);

    if (!result_path[0]) return -1;

    /* Create symlink out_dir → result_path */
    if (out_dir && out_dir[0]) {
        /* Remove existing path if it exists */
        struct stat sb;
        if (lstat(out_dir, &sb) == 0) {
            if (S_ISLNK(sb.st_mode)) unlink(out_dir);
        }
        if (symlink(result_path, out_dir) != 0 && errno != EEXIST) return -1;
        fprintf(stderr, "[bf_cas] HIT %s → %s\n", out_dir, result_path);
    }

    if (manifest_path_out && manifest_sz > 0)
        snprintf(manifest_path_out, manifest_sz, "%s", mpath);

    return 1;
}

/* ── Store ───────────────────────────────────────────────────────────── */

static void mkdir_p(const char *path) {
    char tmp[4096]; snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

int bf_cas_store(BfCasCtx *ctx,
                 const char run_hash[BF_CAS_HASH_LEN],
                 const char input_hash[BF_CAS_HASH_LEN],
                 const char recipe_hash[BF_CAS_HASH_LEN],
                 const char *result_dir,
                 const char *recipe_name) {
    char edir[4096]; entry_dir(ctx, run_hash, edir, sizeof(edir));
    mkdir_p(edir);

    /* Write manifest */
    char mpath[4096]; snprintf(mpath, sizeof(mpath), "%s/run-manifest.json", edir);
    FILE *f = fopen(mpath, "w");
    if (!f) return -1;

    time_t now = time(NULL);
    struct tm *tm = gmtime(&now);
    char tsbuf[32]; strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%dT%H:%M:%SZ", tm);

    /* Resolve absolute path for result_dir */
    char abs_result[4096];
    if (!realpath(result_dir, abs_result))
        snprintf(abs_result, sizeof(abs_result), "%s", result_dir);

    fprintf(f,
        "{\n"
        "  \"cas_version\": \"1.0\",\n"
        "  \"run_hash\": \"%s\",\n"
        "  \"input_hash\": \"%s\",\n"
        "  \"recipe_hash\": \"%s\",\n"
        "  \"recipe_name\": \"%s\",\n"
        "  \"result_dir\": \"%s\",\n"
        "  \"stored_at\": \"%s\"\n"
        "}\n",
        run_hash, input_hash, recipe_hash,
        recipe_name ? recipe_name : "",
        abs_result, tsbuf);
    fclose(f);

    /* Symlink "result" → abs_result */
    char sympath[4096]; snprintf(sympath, sizeof(sympath), "%s/result", edir);
    unlink(sympath);
    symlink(abs_result, sympath);

    fprintf(stderr, "[bf_cas] STORE %s → %s\n", run_hash, abs_result);
    return 0;
}

/* ── Utilities ───────────────────────────────────────────────────────── */

int bf_cas_show(BfCasCtx *ctx, const char run_hash[BF_CAS_HASH_LEN]) {
    char mpath[4096]; manifest_path(ctx, run_hash, mpath, sizeof(mpath));
    FILE *f = fopen(mpath, "r");
    if (!f) { fprintf(stderr, "cas: not found: %s\n", run_hash); return 1; }
    char buf[8192]; size_t n = fread(buf, 1, sizeof(buf)-1, f); fclose(f);
    buf[n] = '\0'; fputs(buf, stdout); return 0;
}

int bf_cas_evict(BfCasCtx *ctx, const char run_hash[BF_CAS_HASH_LEN]) {
    char edir[4096]; entry_dir(ctx, run_hash, edir, sizeof(edir));
    char mpath[4096]; snprintf(mpath, sizeof(mpath), "%s/run-manifest.json", edir);
    if (access(mpath, F_OK) != 0) return 1;
    char sympath[4096]; snprintf(sympath, sizeof(sympath), "%s/result", edir);
    unlink(sympath); unlink(mpath); rmdir(edir);
    fprintf(stderr, "[bf_cas] EVICT %s\n", run_hash);
    return 0;
}
