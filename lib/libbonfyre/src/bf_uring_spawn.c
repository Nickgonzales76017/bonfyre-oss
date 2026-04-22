/*
 * bf_uring_spawn.c — io_uring_spawn level-parallel dispatch
 *                    with posix_spawn fallback
 *
 * Implementation notes
 * ────────────────────
 * io_uring path (Linux ≥ 6.12, IORING_OP_SPAWN):
 *   1. bf_spawn_ctx_new()      → io_uring_queue_init(256, &ring, 0)
 *   2. bf_spawn_register()     → io_uring_register_files() with O_PATH fd
 *                                returns index into fixed-file table
 *   3. bf_spawn_wave_submit()  → one sqe per job via
 *                                io_uring_get_sqe() + io_uring_prep_spawn()
 *                                then single io_uring_submit()
 *   4. bf_spawn_wave_wait()    → io_uring_wait_cqe_nr(ring, cqes, n)
 *
 * Because IORING_OP_SPAWN is not yet in the liburing 2.5 public headers
 * (expected in liburing 2.6, shipping with Ubuntu 26.04 / Fedora 40),
 * we guard the include behind BF_HAS_URING_SPAWN defined when
 * IORING_OP_SPAWN is present.  The fallback path is always compiled.
 *
 * posix_spawn path:
 *   Template registration is just storing the path in a table.
 *   Wave submit forks all N jobs with posix_spawn_file_actions for
 *   stdin/stdout/stderr redirect + pipe chaining.
 *   Wave wait calls waitpid() on all pids.
 *
 * Pipe chaining (cross-binary pipelining):
 *   If job[i].pipe_to_next == 1, an OS pipe is created.
 *   job[i].stdout_fd  ← write end
 *   job[i+1].stdin_fd ← read end
 *   Both ends are closed after the children are started.
 */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "bf_uring_spawn.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

/* environ is in unistd.h on Linux but needs an explicit extern on some targets */
extern char **environ;

/* ── Attempt to detect liburing 2.6+ SPAWN support ──────────────────── */

#if defined(__linux__)
#  if __has_include(<liburing.h>)
#    include <liburing.h>
#    if defined(IORING_OP_SPAWN)
#      define BF_HAS_URING_SPAWN 1
#    endif
#  endif
#endif

/* ── Kernel version probe ───────────────────────────────────────────── */

int bf_spawn_uring_available(void) {
#if defined(BF_HAS_URING_SPAWN)
    struct utsname u;
    if (uname(&u) != 0) return 0;
    /* Parse "6.12" or higher from release string "6.12.0-generic" */
    int major = 0, minor = 0;
    if (sscanf(u.release, "%d.%d", &major, &minor) == 2) {
        if (major > 6) return 1;
        if (major == 6 && minor >= 12) return 1;
    }
    return 0;
#else
    return 0;
#endif
}

/* ── Context structure ──────────────────────────────────────────────── */

#define MAX_WAVE 128

struct BfSpawnCtx {
    int   use_uring;
    /* template registry (both tiers) */
    char  template_paths[BF_SPAWN_MAX_TEMPLATES][4096];
    int   n_templates;
#if defined(BF_HAS_URING_SPAWN)
    struct io_uring ring;
#endif
    /* wave state (posix_spawn path + uring fallback collection) */
    pid_t wave_pids[MAX_WAVE];
    int   wave_n;
    /* pipe bookkeeping: pipe_fds[i] = {read, write} for chain i→i+1 */
    int   chain_fds[MAX_WAVE][2];
    int   chain_active[MAX_WAVE];
};

/* ── Context lifecycle ──────────────────────────────────────────────── */

BfSpawnCtx *bf_spawn_ctx_new(void) {
    BfSpawnCtx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { perror("bf_spawn_ctx_new"); abort(); }

    for (int i = 0; i < MAX_WAVE; i++) {
        ctx->chain_fds[i][0] = -1;
        ctx->chain_fds[i][1] = -1;
    }

#if defined(BF_HAS_URING_SPAWN)
    if (bf_spawn_uring_available()) {
        if (io_uring_queue_init(256, &ctx->ring, 0) == 0) {
            ctx->use_uring = 1;
            fprintf(stderr, "[bf_spawn] backend: io_uring_spawn (Linux ≥ 6.12)\n");
            return ctx;
        }
    }
#endif
    ctx->use_uring = 0;
    fprintf(stderr, "[bf_spawn] backend: posix_spawn\n");
    return ctx;
}

void bf_spawn_ctx_free(BfSpawnCtx *ctx) {
    if (!ctx) return;
#if defined(BF_HAS_URING_SPAWN)
    if (ctx->use_uring) io_uring_queue_exit(&ctx->ring);
#endif
    free(ctx);
}

const char *bf_spawn_backend(BfSpawnCtx *ctx) {
    return ctx->use_uring ? "io_uring" : "posix_spawn";
}

/* ── Template registration ──────────────────────────────────────────── */

int bf_spawn_register(BfSpawnCtx *ctx, const char *path) {
    if (ctx->n_templates >= BF_SPAWN_MAX_TEMPLATES) return -1;
    int id = ctx->n_templates++;
    snprintf(ctx->template_paths[id], 4096, "%s", path);

#if defined(BF_HAS_URING_SPAWN)
    if (ctx->use_uring) {
        /* Open with O_PATH for the fixed-file table */
        int fd = open(path, O_PATH | O_CLOEXEC);
        if (fd < 0) { ctx->n_templates--; return -1; }
        /* Register as fixed file at slot `id` */
        int fds[1] = { fd };
        if (io_uring_register_files_update(&ctx->ring, (unsigned)id, fds, 1) < 0) {
            close(fd); ctx->n_templates--; return -1;
        }
        close(fd);
    }
#endif
    return id;
}

/* ── Wave submit ────────────────────────────────────────────────────── */

static void close_chain_ends(BfSpawnCtx *ctx, int n) {
    for (int i = 0; i < n; i++) {
        if (ctx->chain_active[i]) {
            close(ctx->chain_fds[i][0]);
            close(ctx->chain_fds[i][1]);
            ctx->chain_fds[i][0] = -1;
            ctx->chain_fds[i][1] = -1;
            ctx->chain_active[i] = 0;
        }
    }
}

/* Create pipe for chain between job i and i+1, record in ctx. */
static int make_chain_pipe(BfSpawnCtx *ctx, int i, BfSpawnJob *jobs, int n) {
    if (!jobs[i].pipe_to_next) return 0;
    int next = jobs[i].pipe_next_idx;
    if (next < 0 || next >= n) return -1;
    if (pipe(ctx->chain_fds[i]) != 0) return -1;
    ctx->chain_active[i] = 1;
    /* Inject into job descriptors */
    jobs[i].stdout_fd   = ctx->chain_fds[i][1]; /* writer */
    jobs[next].stdin_fd = ctx->chain_fds[i][0]; /* reader */
    return 0;
}

#if defined(BF_HAS_URING_SPAWN)
static int submit_uring(BfSpawnCtx *ctx, BfSpawnJob *jobs, int n) {
    for (int i = 0; i < n; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ctx->ring);
        if (!sqe) return -1;

        struct io_uring_spawn_attr attr = {0};
        attr.flags    = IORING_SPAWN_FIXED_FILE;
        attr.file_idx = (unsigned)jobs[i].template_id;
        attr.argv     = (const char *const *)jobs[i].argv;
        attr.envp     = (const char *const *)jobs[i].envp;
        if (jobs[i].stdin_fd  >= 0) attr.stdin_fd  = jobs[i].stdin_fd;
        if (jobs[i].stdout_fd >= 0) attr.stdout_fd = jobs[i].stdout_fd;
        if (jobs[i].stderr_fd >= 0) attr.stderr_fd = jobs[i].stderr_fd;

        io_uring_prep_spawn(sqe, &attr);
        io_uring_sqe_set_data64(sqe, (uint64_t)i);
    }
    return io_uring_submit(&ctx->ring) == n ? 0 : -1;
}
#endif /* BF_HAS_URING_SPAWN */

static int submit_posix(BfSpawnCtx *ctx, BfSpawnJob *jobs, int n) {
    for (int i = 0; i < n; i++) {
        const char *path = ctx->template_paths[jobs[i].template_id];

        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);

        /* stdin */
        int in_fd = jobs[i].stdin_fd;
        if (in_fd < 0) {
            int null = open("/dev/null", O_RDONLY);
            if (null >= 0) {
                posix_spawn_file_actions_adddup2(&fa, null, STDIN_FILENO);
                posix_spawn_file_actions_addclose(&fa, null);
            }
        } else {
            posix_spawn_file_actions_adddup2(&fa, in_fd, STDIN_FILENO);
        }
        /* stdout */
        if (jobs[i].stdout_fd >= 0)
            posix_spawn_file_actions_adddup2(&fa, jobs[i].stdout_fd, STDOUT_FILENO);
        /* stderr */
        if (jobs[i].stderr_fd >= 0)
            posix_spawn_file_actions_adddup2(&fa, jobs[i].stderr_fd, STDERR_FILENO);

        char **envp = jobs[i].envp ? jobs[i].envp : environ;
        pid_t pid = 0;
        int rc = posix_spawn(&pid, path, &fa, NULL, jobs[i].argv, envp);
        posix_spawn_file_actions_destroy(&fa);

        if (rc != 0) {
            fprintf(stderr, "[bf_spawn] posix_spawn(%s): %s\n", path, strerror(rc));
            ctx->wave_pids[i] = -1;
            return -1;
        }
        ctx->wave_pids[i] = pid;
    }
    /* close pipe write-ends in parent after all children started */
    for (int i = 0; i < n; i++) {
        if (ctx->chain_active[i]) {
            close(ctx->chain_fds[i][1]); /* parent closes writer */
            ctx->chain_fds[i][1] = -1;
        }
    }
    return 0;
}

int bf_spawn_wave_submit(BfSpawnCtx *ctx, BfSpawnJob *jobs, int n) {
    if (n <= 0 || n > MAX_WAVE) return -1;
    ctx->wave_n = n;

    /* Pre-allocate pipe chains */
    for (int i = 0; i < n; i++) {
        ctx->chain_active[i] = 0;
        ctx->chain_fds[i][0] = -1;
        ctx->chain_fds[i][1] = -1;
    }
    for (int i = 0; i < n; i++) {
        if (jobs[i].pipe_to_next)
            make_chain_pipe(ctx, i, jobs, n);
    }

#if defined(BF_HAS_URING_SPAWN)
    if (ctx->use_uring) {
        int rc = submit_uring(ctx, jobs, n);
        close_chain_ends(ctx, n);
        return rc;
    }
#endif
    return submit_posix(ctx, jobs, n);
}

/* ── Wave wait ──────────────────────────────────────────────────────── */

#if defined(BF_HAS_URING_SPAWN)
static int wait_uring(BfSpawnCtx *ctx, int *codes, int n) {
    struct io_uring_cqe *cqes[MAX_WAVE];
    int rc = io_uring_wait_cqe_nr(&ctx->ring, cqes, (unsigned)n);
    if (rc < 0) return -1;
    int overall = 0;
    for (int i = 0; i < n; i++) {
        int idx = (int)(uint64_t)io_uring_cqe_get_data64(cqes[i]);
        int exit_code = cqes[i]->res;      /* IORING_OP_SPAWN: res = exit status */
        if (exit_code < 0) exit_code = 127;
        if (codes) codes[idx] = exit_code;
        if (exit_code != 0 && overall == 0) overall = exit_code;
        io_uring_cqe_seen(&ctx->ring, cqes[i]);
    }
    return overall;
}
#endif

static int wait_posix(BfSpawnCtx *ctx, int *codes, int n) {
    int overall = 0;
    for (int i = 0; i < n; i++) {
        if (ctx->wave_pids[i] <= 0) {
            if (codes) codes[i] = 127;
            overall = 127; continue;
        }
        int st = 0;
        waitpid(ctx->wave_pids[i], &st, 0);
        int code = WIFEXITED(st) ? WEXITSTATUS(st) : 1;
        if (codes) codes[i] = code;
        if (code != 0 && overall == 0) overall = code;
    }
    /* close any remaining pipe read-ends */
    for (int i = 0; i < n; i++) {
        if (ctx->chain_fds[i][0] >= 0) {
            close(ctx->chain_fds[i][0]);
            ctx->chain_fds[i][0] = -1;
        }
    }
    return overall;
}

int bf_spawn_wave_wait(BfSpawnCtx *ctx, int *exit_codes, int n) {
#if defined(BF_HAS_URING_SPAWN)
    if (ctx->use_uring) return wait_uring(ctx, exit_codes, n);
#endif
    return wait_posix(ctx, exit_codes, n);
}
