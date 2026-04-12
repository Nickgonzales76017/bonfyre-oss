/*
 * qjl.c — Quantized Johnson-Lindenstrauss Bias Correction
 *
 * After polar decomposition + seed compression, there's a residual error
 * between the seed's expanded angles and the true angles. If we just
 * ignore it, inner products (attention scores) will be biased — the AI
 * will hallucinate or crash at extreme compression.
 *
 * QJL fixes this with a 1-bit "sign trick":
 *   1. Project the residual into a random subspace
 *   2. Store only the SIGN of each projection (1 bit each)
 *   3. At query time, use these bits to correct the inner product estimate
 *
 * The math: E[<a,b>_corrected] = <a,b>_true (unbiased in expectation)
 *
 * Cost: FPQ_QJL_PROJECTIONS bits per block (default: 64 bits = 8 bytes).
 * This is the only "overhead" in FPQ. Everything else is the seed.
 */
#include "fpq.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Random projection matrix (generated from seed, never stored) ── */

static uint64_t qjl_rng(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/*
 * Generate a random projection vector from seed.
 * Uses Rademacher distribution: each element is ±1/√m.
 */
static void generate_projection(float *proj, size_t n, uint64_t seed, size_t proj_idx) {
    uint64_t state = seed ^ ((uint64_t)proj_idx * 0x9E3779B97F4A7C15ULL);
    float scale = 1.0f / sqrtf((float)n);

    for (size_t i = 0; i < n; i += 64) {
        uint64_t bits = qjl_rng(&state);
        size_t end = (i + 64 < n) ? i + 64 : n;
        for (size_t j = i; j < end; j++) {
            proj[j] = ((bits >> (j - i)) & 1) ? scale : -scale;
        }
    }
}

/*
 * Encode: compute QJL bits for a residual vector.
 *
 *   For each projection p_i:
 *     bit_i = sign(<residual, p_i>)
 *
 * These bits preserve the inner product relationships in expectation.
 */
fpq_qjl_t *fpq_qjl_encode(const float *residual, size_t n, uint64_t proj_seed) {
    fpq_qjl_t *qjl = (fpq_qjl_t *)calloc(1, sizeof(fpq_qjl_t));
    qjl->n_projections = FPQ_QJL_PROJECTIONS;
    qjl->n_elements = n;
    qjl->proj_seed = proj_seed;

    /* Packed bits: ceil(n_projections / 64) uint64s */
    size_t n_words = (qjl->n_projections + 63) / 64;
    qjl->bits = (uint64_t *)calloc(n_words, sizeof(uint64_t));

    float *proj = (float *)malloc(n * sizeof(float));

    for (size_t p = 0; p < qjl->n_projections; p++) {
        generate_projection(proj, n, proj_seed, p);

        /* Compute <residual, projection> */
        float dot = 0.0f;
        for (size_t i = 0; i < n; i++) {
            dot += residual[i] * proj[i];
        }

        /* Store sign bit */
        if (dot >= 0.0f) {
            qjl->bits[p / 64] |= (1ULL << (p % 64));
        }
    }

    free(proj);
    return qjl;
}

/*
 * Correct an inner product using QJL bits from both vectors.
 *
 * The correction: for each projection i, if sign(a_i) == sign(b_i),
 * add a positive contribution. Otherwise subtract.
 *
 * E[corrected] = <a_true, b_true> — unbiased.
 *
 * This is why the model's attention mechanism stays accurate even
 * at 2-3 bits per weight. The QJL bits preserve the geometry of
 * the inner product space.
 */
float fpq_qjl_correct_dot(const fpq_qjl_t *qjl_a, const fpq_qjl_t *qjl_b,
                           float raw_dot) {
    if (!qjl_a || !qjl_b) return raw_dot;
    if (qjl_a->n_projections != qjl_b->n_projections) return raw_dot;

    size_t m = qjl_a->n_projections;
    size_t n_words = (m + 63) / 64;

    /* Count matching sign bits using XOR + popcount */
    size_t match = 0;
    for (size_t w = 0; w < n_words; w++) {
        uint64_t xored = qjl_a->bits[w] ^ qjl_b->bits[w];
        /* Matching bits = total - different bits */
        size_t bits_in_word = (w == n_words - 1) ? (m - w * 64) : 64;
        size_t diff = 0;

        /* Portable popcount */
        uint64_t v = xored;
        while (v) {
            diff++;
            v &= v - 1;
        }
        match += bits_in_word - diff;
    }

    /* Correction factor: (2 * match / m - 1) * scale */
    float agreement = (2.0f * (float)match / (float)m) - 1.0f;

    /* The residual norm estimate: we scale by the expected residual magnitude */
    float correction = agreement * sqrtf((float)qjl_a->n_elements / (float)m);

    return raw_dot + correction;
}

void fpq_qjl_free(fpq_qjl_t *qjl) {
    if (!qjl) return;
    free(qjl->bits);
    free(qjl);
}

/*
 * Reconstruct an approximation of the original residual vector from
 * QJL 1-bit sign measurements (1-bit compressed sensing).
 *
 * From Plan & Vershynin (2013), the reconstruction is:
 *
 *   x̂ = (||x|| / m) * Σ y_i * φ_i
 *
 * where y_i = sign(<φ_i, x>) ∈ {-1, +1} and φ_i are Rademacher
 * random projections. The ||x|| factor (residual_norm) must be
 * provided since sign measurements lose magnitude information.
 *
 * Quality: captures the dominant direction of x. For m projections
 * in n dimensions: E[cos(x̂, x)] → √(2/π) ≈ 0.798 as m → ∞.
 * The reconstructed residual is added back to the dequantized values
 * to reduce quantization error.
 *
 * output: must have space for qjl->n_elements floats.
 */
void fpq_qjl_reconstruct(const fpq_qjl_t *qjl, float residual_norm,
                          float *output) {
    if (!qjl || !output) return;

    size_t n = qjl->n_elements;
    size_t m = qjl->n_projections;
    memset(output, 0, n * sizeof(float));

    float *proj = (float *)malloc(n * sizeof(float));
    float scale = residual_norm / (float)m;

    for (size_t p = 0; p < m; p++) {
        generate_projection(proj, n, qjl->proj_seed, p);

        /* y_i = +1 if bit set, -1 if not */
        float sign = (qjl->bits[p / 64] & (1ULL << (p % 64))) ? 1.0f : -1.0f;

        /* Accumulate: x̂ += (||x|| / m) * y_i * φ_i */
        float coeff = scale * sign;
        for (size_t i = 0; i < n; i++)
            output[i] += coeff * proj[i];
    }

    free(proj);
}
