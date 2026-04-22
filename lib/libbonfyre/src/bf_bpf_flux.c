/*
 * bf_bpf_flux.c — BPF-Flux per-binary resource isolation
 *
 * cgroup-v2 resource envelopes for the Bonfyre binary suite.
 *
 * On Linux with cgroup-v2:
 *   Creates /sys/fs/cgroup/bonfyre/<binary_name>-<pid>/
 *   Writes memory.max, memory.high, cpu.weight.
 *   Reads memory.current, memory.peak, memory.events (throttled count),
 *   and cpu.pressure PSI for live monitoring.
 *
 * On non-Linux or cgroup-v1-only systems:
 *   All functions are no-ops (bf_flux_available() returns 0).
 *
 * BPF integration note
 *   Full BPF tracing (memory pressure tracepoints via cgroup_skb BPF
 *   programs) requires libbpf + compiled BPF objects.  That layer is
 *   intentionally left as a compile-in extension:
 *     #define BF_FLUX_WITH_BPF 1
 *   and link with libbpf + the pre-compiled bf_flux_prog.bpf.o skeleton.
 *   The base implementation here provides the cgroup isolation layer that
 *   the BPF hooks attach to, making it useful standalone.
 */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "bf_bpf_flux.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* ── Platform guard ─────────────────────────────────────────────────── */

#if !defined(__linux__)
/* Non-Linux stub implementations */
int bf_flux_available(void) { return 0; }
BfFluxEnvelope *bf_flux_envelope_create(const char *n, BfFluxTier t,
                                         const BfFluxLimits *l) {
    (void)n; (void)t; (void)l; return NULL;
}
int  bf_flux_enter(BfFluxEnvelope *e)              { (void)e; return 0; }
int  bf_flux_stat(BfFluxEnvelope *e, BfFluxStat *o){ (void)e; (void)o; return -1; }
void bf_flux_envelope_destroy(BfFluxEnvelope *e)   { (void)e; }

#else  /* Linux */

#define CGROUP_ROOT "/sys/fs/cgroup/bonfyre"

/* ── Internal structure ─────────────────────────────────────────────── */

struct BfFluxEnvelope {
    char   cgroup_path[512];
    size_t mem_max;
    size_t mem_high;
    int    cpu_weight;
    int    active;           /* 1 after cgroup directory created */
};

/* ── Availability ────────────────────────────────────────────────────── */

int bf_flux_available(void) {
    /* Check cgroup-v2 unified hierarchy via cgroup.controllers at root */
    return access("/sys/fs/cgroup/cgroup.controllers", F_OK) == 0;
}

/* ── File helpers ────────────────────────────────────────────────────── */

static int write_cg_file(const char *cgroup, const char *file,
                          const char *value) {
    char path[640];
    snprintf(path, sizeof(path), "%s/%s", cgroup, file);
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    ssize_t n = (ssize_t)strlen(value);
    ssize_t w = write(fd, value, (size_t)n);
    close(fd);
    return (w == n) ? 0 : -1;
}

static int read_cg_file(const char *cgroup, const char *file,
                         char *buf, size_t sz) {
    char path[640];
    snprintf(path, sizeof(path), "%s/%s", cgroup, file);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { buf[0] = '\0'; return -1; }
    ssize_t n = read(fd, buf, sz - 1);
    close(fd);
    if (n < 0) n = 0;
    buf[n] = '\0';
    /* trim trailing newline */
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';
    return 0;
}

static void mkdir_p_cg(const char *path) {
    char tmp[512]; snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

/* ── Tier defaults ───────────────────────────────────────────────────── */

static void tier_defaults(BfFluxTier tier, BfFluxLimits *out) {
    switch (tier) {
        case BF_FLUX_TIER_INSTANT:
            out->mem_max_bytes  = 64UL * 1024 * 1024;
            out->cpu_weight     = 100;
            break;
        case BF_FLUX_TIER_FAST:
            out->mem_max_bytes  = 256UL * 1024 * 1024;
            out->cpu_weight     = 50;
            break;
        case BF_FLUX_TIER_BATCH:
            out->mem_max_bytes  = 512UL * 1024 * 1024;
            out->cpu_weight     = 10;
            break;
        default:
            out->mem_max_bytes  = 256UL * 1024 * 1024;
            out->cpu_weight     = 50;
            break;
    }
    out->mem_high_bytes = 0; /* filled below: 80% of max */
}

/* ── Envelope lifecycle ─────────────────────────────────────────────── */

BfFluxEnvelope *bf_flux_envelope_create(const char *binary_name,
                                         BfFluxTier tier,
                                         const BfFluxLimits *limits) {
    if (!bf_flux_available()) return NULL;

    BfFluxEnvelope *env = calloc(1, sizeof(*env));
    if (!env) return NULL;

    /* Resolve limits */
    BfFluxLimits lim = {0};
    if (tier == BF_FLUX_TIER_CUSTOM && limits) {
        lim = *limits;
    } else {
        tier_defaults(tier, &lim);
    }
    if (lim.mem_high_bytes == 0)
        lim.mem_high_bytes = lim.mem_max_bytes * 4 / 5;  /* 80% */

    env->mem_max    = lim.mem_max_bytes;
    env->mem_high   = lim.mem_high_bytes;
    env->cpu_weight = lim.cpu_weight;

    /* Build cgroup path: /sys/fs/cgroup/bonfyre/<binary_name>-<pid> */
    snprintf(env->cgroup_path, sizeof(env->cgroup_path),
             CGROUP_ROOT "/%s-%d", binary_name ? binary_name : "unknown",
             (int)getpid());

    /* Create bonfyre parent cgroup if needed */
    mkdir_p_cg(CGROUP_ROOT);

    /* Enable subtree controllers in parent (needed for cgroup-v2) */
    write_cg_file(CGROUP_ROOT, "cgroup.subtree_control", "+memory +cpu");

    /* Create envelope cgroup directory */
    if (mkdir(env->cgroup_path, 0755) != 0 && errno != EEXIST) {
        free(env); return NULL;
    }
    env->active = 1;

    /* Write resource limits */
    char val[64];

    snprintf(val, sizeof(val), "%zu", lim.mem_max_bytes);
    write_cg_file(env->cgroup_path, "memory.max", val);

    snprintf(val, sizeof(val), "%zu", lim.mem_high_bytes);
    write_cg_file(env->cgroup_path, "memory.high", val);

    snprintf(val, sizeof(val), "%d", lim.cpu_weight);
    write_cg_file(env->cgroup_path, "cpu.weight", val);

    fprintf(stderr, "[bf_flux] created %s (mem_max=%zuMB cpu.weight=%d)\n",
            env->cgroup_path,
            lim.mem_max_bytes / (1024*1024), lim.cpu_weight);

    return env;
}

int bf_flux_enter(BfFluxEnvelope *env) {
    if (!env || !env->active) return 0;  /* no-op if cgroup unavailable */
    char val[32];
    snprintf(val, sizeof(val), "%d\n", (int)getpid());
    return write_cg_file(env->cgroup_path, "cgroup.procs", val);
}

/* ── Stat ────────────────────────────────────────────────────────────── */

int bf_flux_stat(BfFluxEnvelope *env, BfFluxStat *out) {
    if (!env || !env->active || !out) return -1;
    memset(out, 0, sizeof(*out));

    char buf[256];

    /* memory.current */
    if (read_cg_file(env->cgroup_path, "memory.current", buf, sizeof(buf)) == 0)
        out->mem_current = (size_t)strtoull(buf, NULL, 10);

    /* memory.peak (Linux ≥ 5.19) */
    if (read_cg_file(env->cgroup_path, "memory.peak", buf, sizeof(buf)) == 0)
        out->mem_max = (size_t)strtoull(buf, NULL, 10);

    /* memory.events: count "throttled N" */
    if (read_cg_file(env->cgroup_path, "memory.events", buf, sizeof(buf)) == 0) {
        const char *p = strstr(buf, "throttled");
        if (p) {
            p += 9;  /* skip "throttled" */
            while (*p == ' ' || *p == '\t') p++;
            long tv = strtol(p, NULL, 10);
            out->throttled = (tv > 0) ? 1 : 0;
        }
    }

    /* cpu.pressure PSI (some avg10) */
    if (read_cg_file(env->cgroup_path, "cpu.pressure", buf, sizeof(buf)) == 0) {
        /* "some avg10=X.XX avg60=… avg300=… total=…" */
        const char *p = strstr(buf, "avg10=");
        if (p) out->cpu_pressure = atof(p + 6);
    }

    return 0;
}

/* ── Destroy ─────────────────────────────────────────────────────────── */

void bf_flux_envelope_destroy(BfFluxEnvelope *env) {
    if (!env) return;
    if (env->active) {
        /* cgroup can only be removed when empty (after child exits) */
        if (rmdir(env->cgroup_path) != 0)
            fprintf(stderr, "[bf_flux] rmdir %s: %s (may still have procs)\n",
                    env->cgroup_path, strerror(errno));
    }
    free(env);
}

#endif /* __linux__ */
