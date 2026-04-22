/*
 * bf_cas.h — Content-Addressable Result Store
 *
 * Implements "Instant Reruns": if the SHA-256 of (input_file + recipe_level_hash)
 * matches a previously completed run, the result directory is symlinked from
 * the Bonfyre space cache instead of re-executing the binary wave.
 *
 * Design
 * ──────
 *   Cache root: $BONFYRE_CAS_DIR  or  ~/.local/share/bonfyre/cas/
 *
 *   Entry layout:
 *     <cas_root>/<hex16>/          — first 16 hex chars of run_hash
 *       run-manifest.json          — full metadata + source hash
 *       result -> <original_out>   — symlink to the actual output tree
 *
 *   run_hash = SHA-256(input_hash_hex + ":" + recipe_level_hash_hex)
 *   Both hashes are themselves SHA-256, computed by bf_cas_hash_file()
 *   and bf_cas_hash_levels() respectively.
 *
 * Merkle-level hashing
 * ─────────────────────
 *   A recipe is a sequence of levels, each level a sorted list of binary names.
 *   bf_cas_hash_levels(binaries[], n_levels[]) hashes each level independently
 *   then chains them: final_hash = SHA-256(L0_hash || L1_hash || ... || Lk_hash).
 *   This means two recipes that share a common prefix have matching level
 *   hashes up to the divergence point — enabling partial deduplication.
 *
 * Thread safety: none (single-writer design; reads are safe).
 */
#pragma once
#ifndef BF_CAS_H
#define BF_CAS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Hex string for a SHA-256 digest: 64 chars + NUL */
#define BF_CAS_HASH_LEN 65

typedef struct {
    char root[4096];            /* resolved cas root path */
} BfCasCtx;

/* ── Init ────────────────────────────────────────────────────────────── */

/* Initialise ctx, creating cache root if necessary.
 * Uses BONFYRE_CAS_DIR env var or ~/.local/share/bonfyre/cas/.
 * Returns 0 on success. */
int bf_cas_init(BfCasCtx *ctx);

/* ── Hashing helpers ─────────────────────────────────────────────────── */

/* Hash the byte-content of a file into hex_out (65 bytes). */
int bf_cas_hash_file(const char *path, char hex_out[BF_CAS_HASH_LEN]);

/* Hash a list of recipe level descriptors (sorted binary name arrays).
 *
 *   levels       — array of (char **) — each element is a NULL-terminated
 *                  sorted list of binary names for that level.
 *   n_levels     — number of levels.
 *   hex_out      — receives 64-char hex + NUL (Merkle-chain of level hashes).
 */
int bf_cas_hash_levels(const char **const *levels, int n_levels,
                       char hex_out[BF_CAS_HASH_LEN]);

/* Compute the run_hash from input_hash + recipe_hash.
 *   run_hash = SHA-256(input_hex ":" recipe_hex)
 */
void bf_cas_run_hash(const char input_hex[BF_CAS_HASH_LEN],
                     const char recipe_hex[BF_CAS_HASH_LEN],
                     char run_hash_out[BF_CAS_HASH_LEN]);

/* ── Cache lookup + store ─────────────────────────────────────────────── */

/* Look up run_hash in the cache.
 * If found: symlinks out_dir → cached result dir, writes manifest path
 *           to manifest_path_out (if non-NULL), returns 1.
 * If not found: returns 0.
 * On error: returns -1. */
int bf_cas_lookup(BfCasCtx *ctx,
                  const char run_hash[BF_CAS_HASH_LEN],
                  const char *out_dir,
                  char *manifest_path_out, size_t manifest_sz);

/* Record a completed run in the cache.
 *   run_hash     — 64-char hex
 *   input_hash   — 64-char hex (of the input file)
 *   recipe_hash  — 64-char hex (Merkle-chain of levels)
 *   result_dir   — the directory that was produced
 *   recipe_name  — human label (e.g. "A1", "A3")
 *
 * Creates <cas_root>/<hex16>/run-manifest.json and a "result" symlink.
 * Returns 0 on success. */
int bf_cas_store(BfCasCtx *ctx,
                 const char run_hash[BF_CAS_HASH_LEN],
                 const char input_hash[BF_CAS_HASH_LEN],
                 const char recipe_hash[BF_CAS_HASH_LEN],
                 const char *result_dir,
                 const char *recipe_name);

/* ── Utilities ───────────────────────────────────────────────────────── */

/* Print the cache manifest for a run_hash to stdout.
 * Returns 0 if found, 1 if not found. */
int bf_cas_show(BfCasCtx *ctx, const char run_hash[BF_CAS_HASH_LEN]);

/* Evict a single entry.  Returns 0 on success, 1 if not found. */
int bf_cas_evict(BfCasCtx *ctx, const char run_hash[BF_CAS_HASH_LEN]);

#ifdef __cplusplus
}
#endif
#endif /* BF_CAS_H */
