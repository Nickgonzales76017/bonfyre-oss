/*
 * bf_bpf_flux.h — BPF-Flux per-binary resource isolation
 *
 * Wraps each bonfyre-run child process in a cgroup-v2 "envelope" that:
 *   • caps memory usage (hard limit → OOM-kill; soft limit → throttle)
 *   • caps CPU share weight so a heavy binary can't starve the control layer
 *   • optionally installs a BPF program (via bpf(2) + cgroup_skb / cgroup_sock
 *     or reuseport) that monitors cgroup memory pressure events and signals
 *     the parent to throttle before the hard limit is hit.
 *
 * Architecture
 * ────────────
 *                 bonfyre-run (parent)
 *                      │
 *             bf_flux_envelope_create()
 *                      │
 *              creates cgroup-v2 hierarchy:
 *          /sys/fs/cgroup/bonfyre/<pid>/
 *                      │
 *              fork() → child process moves into cgroup
 *                      │
 *          writes memory.max, memory.high, cpu.weight
 *                      │
 *          (if BPF available) loads bf_flux_prog.bpf.o
 *          attaches to cgroup for memory pressure tracing
 *                      │
 *          bf_flux_envelope_destroy() on child exit
 *
 * Tiers
 * ─────
 *   BF_FLUX_TIER_BATCH   — batch-tier defaults: 512MB max, cpu.weight=10
 *   BF_FLUX_TIER_FAST    — fast-tier defaults:   256MB max, cpu.weight=50
 *   BF_FLUX_TIER_INSTANT — instant-tier:          64MB max, cpu.weight=100
 *   BF_FLUX_TIER_CUSTOM  — caller provides limits explicitly
 *
 * Throttle vs. Kill
 *   memory.high (soft limit) = 80% of memory.max
 *   When the process crosses memory.high, cgroup-v2 adds an artificial
 *   delay to page fault handling — throttling the process without killing it.
 *   The BPF hook fires a tracepoint at the high-limit crossing, allowing
 *   the parent to log or react (e.g., promote to a lighter model).
 *
 * Portability
 *   cgroup-v2 required (Linux ≥ 4.5, enabled by default on 5.x+).
 *   BPF tracing is optional; graceful fallback to cgroup-only at runtime.
 *   On macOS / non-Linux: entire implementation is a no-op.
 *
 * Thread safety: each BfFluxEnvelope is independent; no shared state.
 */
#pragma once
#ifndef BF_BPF_FLUX_H
#define BF_BPF_FLUX_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Resource tiers ─────────────────────────────────────────────────── */

typedef enum {
    BF_FLUX_TIER_INSTANT = 0,  /*  64MB max, weight=100 — control layer    */
    BF_FLUX_TIER_FAST    = 1,  /* 256MB max, weight= 50 — light inference  */
    BF_FLUX_TIER_BATCH   = 2,  /* 512MB max, weight= 10 — heavy batch jobs */
    BF_FLUX_TIER_CUSTOM  = 3,  /* caller fills BfFluxLimits directly        */
} BfFluxTier;

typedef struct {
    size_t  mem_max_bytes;   /* hard memory limit (OOM-kill)              */
    size_t  mem_high_bytes;  /* soft limit (throttle; 0 = 80% of max)     */
    int     cpu_weight;      /* cgroup cpu.weight [1..10000]              */
} BfFluxLimits;

/* ── Envelope handle ─────────────────────────────────────────────────── */

typedef struct BfFluxEnvelope BfFluxEnvelope;

/* ── API ─────────────────────────────────────────────────────────────── */

/*
 * bf_flux_envelope_create()
 *
 * Allocate an envelope and create the cgroup hierarchy for a future child.
 * Call this BEFORE fork().  The child must call bf_flux_enter(env) after
 * fork() and before exec().
 *
 *   binary_name — human label used in the cgroup path (e.g. "bonfyre-transcribe")
 *   tier        — resource tier (or BF_FLUX_TIER_CUSTOM + fill limits)
 *   limits      — used only when tier == BF_FLUX_TIER_CUSTOM; may be NULL otherwise
 *
 * Returns NULL on error (e.g. cgroup-v2 not available).
 */
BfFluxEnvelope *bf_flux_envelope_create(const char *binary_name,
                                         BfFluxTier tier,
                                         const BfFluxLimits *limits);

/*
 * bf_flux_enter()
 *
 * Called by the child process after fork(), before exec().
 * Moves the calling process into the cgroup.
 * Returns 0 on success, -1 on error.
 */
int bf_flux_enter(BfFluxEnvelope *env);

/*
 * bf_flux_stat()
 *
 * Read current memory usage and CPU pressure for the envelope.
 * safe to call from the parent at any time while the child is live.
 */
typedef struct {
    size_t mem_current;    /* bytes currently used                     */
    size_t mem_max;        /* peak usage since cgroup creation         */
    double cpu_pressure;   /* PSI some-avg10 (%; 0.0 if unavailable)   */
    int    throttled;      /* 1 if memory.high was crossed this window */
} BfFluxStat;

int bf_flux_stat(BfFluxEnvelope *env, BfFluxStat *out);

/*
 * bf_flux_envelope_destroy()
 *
 * Remove the cgroup and release all resources.
 * Call after waitpid() on the child.
 */
void bf_flux_envelope_destroy(BfFluxEnvelope *env);

/* ── Availability probe ──────────────────────────────────────────────── */

/* Returns 1 if cgroup-v2 is mounted and writable at /sys/fs/cgroup. */
int bf_flux_available(void);

#ifdef __cplusplus
}
#endif
#endif /* BF_BPF_FLUX_H */
