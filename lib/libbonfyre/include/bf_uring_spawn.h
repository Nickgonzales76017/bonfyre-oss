/*
 * bf_uring_spawn.h — io_uring_spawn level-parallel binary dispatch
 *
 * Linux 6.12+ stabilized IORING_OP_SPAWN, which allows pre-registered
 * process templates to be dispatched via a single io_uring SQE batch —
 * effectively zero-syscall process spawning after the initial template
 * registration.
 *
 * This header provides a two-tier implementation:
 *
 *   TIER 1 — io_uring path (Linux 6.12+, liburing ≥ 2.6)
 *     Template all binaries at startup via io_uring_register_templates().
 *     Dispatch a full level-parallel wave as a single sqe batch.
 *     Completion harvested with io_uring_wait_cqe_nr() — no poll loop.
 *
 *   TIER 2 — posix_spawn fallback (all other kernels)
 *     Uses posix_spawn() with POSIX_SPAWN_SETSIGMASK + file actions.
 *     Still faster than fork/exec for many short-lived processes because
 *     posix_spawn avoids COW page table duplication.
 *
 * API
 * ───
 *   BfSpawnCtx *bf_spawn_ctx_new(void)
 *     Allocate + initialize context.  Probes kernel for io_uring_spawn
 *     support and initialises the ring if available.
 *
 *   int bf_spawn_register(BfSpawnCtx *ctx, const char *path)
 *     Pre-register a binary template.  Returns a template_id ≥ 0.
 *     Must be called before bf_spawn_wave_submit.
 *     On fallback tier, stores path in a registry; returns index.
 *
 *   int bf_spawn_wave_submit(BfSpawnCtx *ctx, BfSpawnJob *jobs, int n)
 *     Submit N jobs as a single parallel wave.  Each job specifies:
 *       template_id, argv[], envp[], stdin_fd, stdout_fd, stderr_fd.
 *     Returns 0 on success (all jobs queued).
 *
 *   int bf_spawn_wave_wait(BfSpawnCtx *ctx, int *exit_codes, int n)
 *     Wait for all N jobs from the last wave.  Fills exit_codes[].
 *     Returns 0 if all exited successfully (0), else first non-zero code.
 *
 *   void bf_spawn_ctx_free(BfSpawnCtx *ctx)
 *     Tear down ring / release resources.
 *
 * Usage pattern (level-parallel recipe dispatch)
 * ───────────────────────────────────────────────
 *   BfSpawnCtx *ctx = bf_spawn_ctx_new();
 *
 *   // Register all binaries once at startup (cheap template bake)
 *   int tid_ingest     = bf_spawn_register(ctx, "/usr/bin/bonfyre-ingest");
 *   int tid_transcribe = bf_spawn_register(ctx, "/usr/bin/bonfyre-transcribe");
 *   ...
 *
 *   // Per-recipe: build the level-0 wave
 *   BfSpawnJob wave0[4] = {
 *       { .template_id = tid_ingest,  .argv = {...}, .stdout_fd = pipe_fds[0][1] },
 *       ...
 *   };
 *   bf_spawn_wave_submit(ctx, wave0, 4);
 *
 *   int codes[4];
 *   bf_spawn_wave_wait(ctx, codes, 4);   // harvests all 4 completions at once
 *
 *   // Advance to level-1 wave...
 */
#pragma once
#ifndef BF_URING_SPAWN_H
#define BF_URING_SPAWN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Job descriptor ─────────────────────────────────────────────────── */

#define BF_SPAWN_MAX_ARGV 64
#define BF_SPAWN_MAX_TEMPLATES 128

typedef struct {
    int          template_id;               /* from bf_spawn_register()   */
    char        *argv[BF_SPAWN_MAX_ARGV];   /* NULL-terminated            */
    char       **envp;                      /* NULL = inherit             */
    int          stdin_fd;                  /* -1 = /dev/null             */
    int          stdout_fd;                 /* -1 = inherit               */
    int          stderr_fd;                 /* -1 = inherit               */
    /* pipe-chain: if pipe_to_next is set, stdout is connected to the
     * stdin of the job at pipe_next_idx in the same wave — enabling
     * cross-binary pipelining within a single wave submission. */
    int          pipe_to_next;              /* 1 = chain stdout → next */
    int          pipe_next_idx;            /* index in current wave    */
} BfSpawnJob;

/* ── Opaque context ─────────────────────────────────────────────────── */

typedef struct BfSpawnCtx BfSpawnCtx;

/* ── API ─────────────────────────────────────────────────────────────── */

/* Allocate and initialise spawn context.
 * Probes Linux version; enables io_uring_spawn if kernel ≥ 6.12
 * and liburing ≥ 2.6 is present.  Never returns NULL. */
BfSpawnCtx *bf_spawn_ctx_new(void);

/* Pre-register a binary at path as a reusable template.
 * Returns template_id ≥ 0, or -1 on error. */
int bf_spawn_register(BfSpawnCtx *ctx, const char *path);

/* Submit N jobs as one parallel level wave.
 * All jobs start concurrently.  Returns 0 on success. */
int bf_spawn_wave_submit(BfSpawnCtx *ctx, BfSpawnJob *jobs, int n);

/* Wait for all N jobs in the last submitted wave.
 * exit_codes must be at least n ints.
 * Returns 0 if all processes exited with 0, else first non-zero code. */
int bf_spawn_wave_wait(BfSpawnCtx *ctx, int *exit_codes, int n);

/* Returns "io_uring" or "posix_spawn" — active backend. */
const char *bf_spawn_backend(BfSpawnCtx *ctx);

/* Release all resources. ctx is invalid after this call. */
void bf_spawn_ctx_free(BfSpawnCtx *ctx);

/* ── Kernel version probe ─────────────────────────────────────────────── */

/* Returns 1 if the running kernel supports io_uring_spawn (≥ 6.12). */
int bf_spawn_uring_available(void);

#ifdef __cplusplus
}
#endif
#endif /* BF_URING_SPAWN_H */
