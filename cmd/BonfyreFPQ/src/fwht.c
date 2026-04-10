/*
 * fwht.c — Fast Walsh-Hadamard Transform (Haar Rotation)
 *
 * The first step in FPQ: apply a random orthogonal rotation to make
 * the weight distribution data-oblivious. After FWHT + random signs,
 * coordinates follow a concentrated Beta distribution regardless of
 * the original data. The geometry is guaranteed by the Haar measure.
 *
 * This eliminates per-block scaling factors entirely.
 */
#include "fpq.h"
#include <math.h>
#include <string.h>

/* ── xorshift64 for reproducible random signs ── */
static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/*
 * In-place Fast Walsh-Hadamard Transform.
 * After this, the coordinates are decorrelated and follow a known
 * distribution — we've moved from Cartesian to Haar-rotated space.
 *
 * O(n log n) butterfly operations. n must be power of 2.
 */
void fpq_fwht(float *x, size_t n) {
    if (n <= 1) return;

    for (size_t len = 1; len < n; len <<= 1) {
        for (size_t i = 0; i < n; i += len << 1) {
            for (size_t j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }

    /* Normalize by 1/sqrt(n) to preserve energy */
    float scale = 1.0f / sqrtf((float)n);
    for (size_t i = 0; i < n; i++) {
        x[i] *= scale;
    }
}

/*
 * Inverse FWHT — identical butterfly, same normalization.
 * FWHT is its own inverse (up to scaling).
 */
void fpq_fwht_inverse(float *x, size_t n) {
    /* FWHT is self-inverse with same normalization */
    fpq_fwht(x, n);
}

/*
 * Apply random sign flips: x[i] *= random_sign(seed, i)
 *
 * Combined with FWHT, this gives us the full randomized Hadamard
 * transform — equivalent to a Haar-random rotation in the limit.
 * After this, ANY input data produces Beta-distributed coordinates.
 */
void fpq_random_signs(float *x, size_t n, uint64_t seed) {
    uint64_t state = seed ? seed : 0x5DEECE66DUL;
    for (size_t i = 0; i < n; i += 64) {
        uint64_t bits = xorshift64(&state);
        size_t end = (i + 64 < n) ? i + 64 : n;
        for (size_t j = i; j < end; j++) {
            if ((bits >> (j - i)) & 1) {
                x[j] = -x[j];
            }
        }
    }
}

/*
 * Undo random sign flips (same operation — signs are self-inverse).
 */
void fpq_random_signs_inverse(float *x, size_t n, uint64_t seed) {
    fpq_random_signs(x, n, seed);
}
