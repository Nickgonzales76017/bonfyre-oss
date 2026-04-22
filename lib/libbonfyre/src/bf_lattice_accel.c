/*
 * bf_lattice_accel.c — SIMD-accelerated batch E8 lattice quantization
 *
 * Implements two quantization paths and a "frustrated boundary" heuristic:
 *
 *   STRICT path  — Conway-Sloane E8 with parity fix (exact nearest-point)
 *   RELAXED path — per-dimension snap, no parity enforcement (faster,
 *                  lower error on heavy-tailed distributions)
 *
 * SIMD acceleration
 *   AVX2:  8 groups processed per outer loop iteration (8×8 = 64 floats at once)
 *   NEON:  4 groups processed per outer loop iteration (4×8 = 32 floats)
 *   Scalar: 1 group at a time (fallback)
 *
 * "Frustrated boundary" heuristic
 *   A group is near a lattice boundary when the sum-of-nearest-integers is
 *   within `tol` of being exactly odd.  In that case the parity fix is
 *   ambiguous and may amplify error; we route those groups to the relaxed
 *   variant instead.
 *
 * Compile-time backend selection:
 *   -DBFL_FORCE_AVX2   — always use AVX2 (requires AVX2 + FMA hardware)
 *   -DBFL_FORCE_NEON   — always use NEON (requires AArch64)
 *   -DBFL_FORCE_SCALAR — always use scalar
 *   Otherwise: detected at compile time via __AVX2__ / __ARM_NEON macros.
 */

#include "bf_lattice_accel.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ── Backend selection ─────────────────────────────────────────────────── */

#if defined(BFL_FORCE_AVX2) || (!defined(BFL_FORCE_SCALAR) && !defined(BFL_FORCE_NEON) && defined(__AVX2__))
#  define BFL_BACKEND_AVX2 1
#  include <immintrin.h>
#elif defined(BFL_FORCE_NEON) || (!defined(BFL_FORCE_SCALAR) && !defined(BFL_FORCE_AVX2) && (defined(__ARM_NEON) || defined(__ARM_NEON__)))
#  define BFL_BACKEND_NEON 1
#  include <arm_neon.h>
#else
#  define BFL_BACKEND_SCALAR 1
#endif

const char *bf_lattice_accel_backend(void) {
#if defined(BFL_BACKEND_AVX2)
    return "avx2";
#elif defined(BFL_BACKEND_NEON)
    return "neon";
#else
    return "scalar";
#endif
}

/* ── Scalar E8 primitives (used in all paths for the parity fix) ─────── */

static inline float scalar_round(float x) {
    return (x >= 0.f) ? floorf(x + 0.5f) : ceilf(x - 0.5f);
}

/* Strict Conway-Sloane E8 snap: nearest D8 vs D8+½ coset, parity fix */
static void e8_snap_scalar(const float *x, float *out) {
    float d8[8], d8h[8];
    float sum_d8 = 0.f, sum_d8h = 0.f;

    for (int i = 0; i < 8; i++) {
        d8[i]  = scalar_round(x[i]);
        d8h[i] = floorf(x[i]) + 0.5f;
        float ed  = x[i] - d8[i];
        float edh = x[i] - d8h[i];
        sum_d8  += ed  * ed;
        sum_d8h += edh * edh;
    }

    float *snap = (sum_d8 <= sum_d8h) ? d8 : d8h;
    memcpy(out, snap, 8 * sizeof(float));

    /* Parity fix: sum of integer coordinates must be even */
    if ((snap == d8h)) return;  /* D8+½ always has half-integer sum — no fix needed */

    int isum = 0;
    for (int i = 0; i < 8; i++) isum += (int)out[i];
    if ((isum & 1) != 0) {
        /* flip the coord whose snap moved it least */
        float max_err = -1.f;
        int   worst   = 0;
        for (int i = 0; i < 8; i++) {
            float e = fabsf(x[i] - out[i]);
            if (e > max_err) { max_err = e; worst = i; }
        }
        out[worst] += (x[worst] > out[worst]) ? -1.f : 1.f;
    }
}

/* Relaxed E8 snap: per-dimension nearest integer, no parity enforcement */
static void e8_relax_scalar(const float *x, float *out) {
    for (int i = 0; i < 8; i++)
        out[i] = scalar_round(x[i]);
}

/* ── Boundary detection ─────────────────────────────────────────────────── */

/*
 * Returns 1 if the group is "frustrated" (near a lattice boundary).
 * We measure how odd the D8 integer sum is vs. 0.5 of fully-fractional;
 * if the fractional part of sum is within tol of 0 or 1, parity is ambiguous.
 */
static int is_frustrated(const float *x, float tol) {
    float sum = 0.f;
    for (int i = 0; i < 8; i++) {
        float r = scalar_round(x[i]);
        sum += fabsf(x[i] - r);
    }
    /* sum of absolute residuals — high when many dims are near 0.5 */
    float frac = sum - floorf(sum);  /* fractional part */
    return (frac < tol || frac > 1.f - tol);
}

/* ── Scalar batch ── (used when no SIMD or n < threshold) ─────────────── */

static int e8_batch_scalar(const float *in, float *out, size_t n) {
    for (size_t i = 0; i < n; i++)
        e8_snap_scalar(in + 8*i, out + 8*i);
    return 0;
}

/* ── NEON batch ─────────────────────────────────────────────────────────── */

#if defined(BFL_BACKEND_NEON)

/*
 * NEON: process 4 groups (32 floats) per iteration.
 * Each vld1q_f32 loads 4 floats; 2 loads = 8 floats = 1 group.
 * We run 4 groups in parallel → 8 vld1q_f32 calls per iteration.
 *
 * This path implements the D8 nearest-integer snap via vrndnq_f32
 * (round-to-nearest) and then falls back to scalar for the parity fix,
 * since parity is a cross-element reduction and not easily vectorised.
 */
static int e8_batch_neon(const float *in, float *out, size_t n) {
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        for (int g = 0; g < 4; g++) {
            const float *xi = in  + 8 * (i + g);
            float       *xo = out + 8 * (i + g);

            float32x4_t lo = vld1q_f32(xi);
            float32x4_t hi = vld1q_f32(xi + 4);

            /* round-to-nearest (vrndnq requires ARMv8-A) */
            float32x4_t rlo = vrndnq_f32(lo);
            float32x4_t rhi = vrndnq_f32(hi);

            vst1q_f32(xo,     rlo);
            vst1q_f32(xo + 4, rhi);

            /* parity fix (scalar) */
            int isum = 0;
            for (int k = 0; k < 8; k++) isum += (int)xo[k];
            if ((isum & 1) != 0) {
                float max_err = -1.f;
                int   worst   = 0;
                for (int k = 0; k < 8; k++) {
                    float e = fabsf(xi[k] - xo[k]);
                    if (e > max_err) { max_err = e; worst = k; }
                }
                xo[worst] += (xi[worst] > xo[worst]) ? -1.f : 1.f;
            }
        }
    }
    /* remainder */
    for (; i < n; i++)
        e8_snap_scalar(in + 8*i, out + 8*i);

    return 0;
}

#endif /* BFL_BACKEND_NEON */

/* ── AVX2 batch ─────────────────────────────────────────────────────────── */

#if defined(BFL_BACKEND_AVX2)

/*
 * AVX2: process 8 groups (64 floats) per iteration.
 * _mm256_round_ps gives round-to-nearest for the integer snap.
 * Parity fix done scalar per group (cross-lane reduction).
 */
static int e8_batch_avx2(const float *in, float *out, size_t n) {
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        for (int g = 0; g < 8; g++) {
            const float *xi = in  + 8 * (i + g);
            float       *xo = out + 8 * (i + g);

            __m256 v = _mm256_loadu_ps(xi);
            __m256 r = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_ps(xo, r);

            /* parity fix */
            int isum = 0;
            for (int k = 0; k < 8; k++) isum += (int)xo[k];
            if ((isum & 1) != 0) {
                float max_err = -1.f;
                int   worst   = 0;
                for (int k = 0; k < 8; k++) {
                    float e = fabsf(xi[k] - xo[k]);
                    if (e > max_err) { max_err = e; worst = k; }
                }
                xo[worst] += (xi[worst] > xo[worst]) ? -1.f : 1.f;
            }
        }
    }
    for (; i < n; i++)
        e8_snap_scalar(in + 8*i, out + 8*i);

    return 0;
}

#endif /* BFL_BACKEND_AVX2 */

/* ── Public API ─────────────────────────────────────────────────────────── */

int bf_e8_batch_simd(const float *in, float *out, size_t n) {
#if defined(BFL_BACKEND_NEON)
    return e8_batch_neon(in, out, n);
#elif defined(BFL_BACKEND_AVX2)
    return e8_batch_avx2(in, out, n);
#else
    return e8_batch_scalar(in, out, n);
#endif
}

int bf_e8_batch_simd_frustrated(const float *in, float *out, size_t n,
                                 float tol, int *relaxed_count_out) {
    int relaxed = 0;

    for (size_t i = 0; i < n; i++) {
        const float *xi = in  + 8 * i;
        float       *xo = out + 8 * i;

        if (tol > 0.f && is_frustrated(xi, tol)) {
            e8_relax_scalar(xi, xo);
            relaxed++;
        } else {
            e8_snap_scalar(xi, xo);
        }
    }

    if (relaxed_count_out) *relaxed_count_out = relaxed;
    return 0;
}
