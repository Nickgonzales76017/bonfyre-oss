/*
 * bf_he_ckks.c — CKKS-style homomorphic arithmetic (depth-4 path)
 *
 * This file implements a compact CKKS runtime model:
 *   - packed complex slots
 *   - additive/multiplicative homomorphism with depth tracking
 *   - rescale and bootstrap refresh
 *   - depth-4 inference pipeline with cubic activation approximation
 */

#define _POSIX_C_SOURCE 200809L
#include "bf_he_ckks.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ───────────────────────────────────────────────────────────────────────────
 * Small RNG (xorshift64*) for deterministic noise injection
 * ─────────────────────────────────────────────────────────────────────────── */

static uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

static double noise_unit(uint64_t *s) {
    /* Uniform in (-1,1) */
    uint64_t r = xs64(s);
    double u = (double)(r & 0xffffffffu) / 4294967295.0;
    return 2.0 * u - 1.0;
}

/* ───────────────────────────────────────────────────────────────────────────
 * Init / free
 * ─────────────────────────────────────────────────────────────────────────── */

int bf_ckks_cipher_init(BfCkksCipher *ct, int slots, int level, double scale) {
    if (!ct || slots <= 0 || slots > BF_CKKS_MAX_SLOTS) return -1;
    memset(ct, 0, sizeof(*ct));
    ct->slots = slots;
    ct->level = level;
    ct->scale = scale > 0.0 ? scale : BF_CKKS_DEFAULT_SCALE;
    ct->noise_budget = 60;
    ct->re = (double *)calloc((size_t)slots, sizeof(double));
    ct->im = (double *)calloc((size_t)slots, sizeof(double));
    if (!ct->re || !ct->im) {
        bf_ckks_cipher_free(ct);
        return -1;
    }
    return 0;
}

void bf_ckks_cipher_free(BfCkksCipher *ct) {
    if (!ct) return;
    free(ct->re); ct->re = NULL;
    free(ct->im); ct->im = NULL;
    ct->slots = 0;
}

int bf_ckks_plain_init(BfCkksPlain *pt, int slots) {
    if (!pt || slots <= 0 || slots > BF_CKKS_MAX_SLOTS) return -1;
    memset(pt, 0, sizeof(*pt));
    pt->slots = slots;
    pt->re = (double *)calloc((size_t)slots, sizeof(double));
    pt->im = (double *)calloc((size_t)slots, sizeof(double));
    if (!pt->re || !pt->im) {
        bf_ckks_plain_free(pt);
        return -1;
    }
    return 0;
}

void bf_ckks_plain_free(BfCkksPlain *pt) {
    if (!pt) return;
    free(pt->re); pt->re = NULL;
    free(pt->im); pt->im = NULL;
    pt->slots = 0;
}

/* ───────────────────────────────────────────────────────────────────────────
 * Encode / Encrypt / Decrypt
 * ─────────────────────────────────────────────────────────────────────────── */

int bf_ckks_encode(const double *re, const double *im, int slots, BfCkksPlain *pt) {
    if (!pt || pt->slots != slots) return -1;
    for (int i = 0; i < slots; i++) {
        pt->re[i] = re ? re[i] : 0.0;
        pt->im[i] = im ? im[i] : 0.0;
    }
    return 0;
}

int bf_ckks_encrypt(const BfCkksPlain *pt, BfCkksCipher *ct, uint64_t seed) {
    if (!pt || !ct || ct->slots != pt->slots) return -1;
    uint64_t s = seed ? seed : 0x9e3779b97f4a7c15ULL;
    double sigma = 1.0 / ct->scale;
    for (int i = 0; i < ct->slots; i++) {
        ct->re[i] = pt->re[i] * ct->scale + sigma * noise_unit(&s);
        ct->im[i] = pt->im[i] * ct->scale + sigma * noise_unit(&s);
    }
    return 0;
}

int bf_ckks_decrypt(const BfCkksCipher *ct, BfCkksPlain *pt) {
    if (!ct || !pt || pt->slots != ct->slots) return -1;
    for (int i = 0; i < ct->slots; i++) {
        pt->re[i] = ct->re[i] / ct->scale;
        pt->im[i] = ct->im[i] / ct->scale;
    }
    return 0;
}

/* ───────────────────────────────────────────────────────────────────────────
 * Arithmetic helpers
 * ─────────────────────────────────────────────────────────────────────────── */

static int ensure_same(const BfCkksCipher *a, const BfCkksCipher *b, BfCkksCipher *o) {
    if (!a || !b || !o) return -1;
    if (a->slots != b->slots) return -1;
    return bf_ckks_cipher_init(o, a->slots,
                               a->level < b->level ? a->level : b->level,
                               a->scale > b->scale ? a->scale : b->scale);
}

int bf_ckks_add(const BfCkksCipher *a, const BfCkksCipher *b, BfCkksCipher *out) {
    if (ensure_same(a, b, out) < 0) return -1;
    double sa = out->scale / a->scale;
    double sb = out->scale / b->scale;
    for (int i = 0; i < out->slots; i++) {
        out->re[i] = a->re[i] * sa + b->re[i] * sb;
        out->im[i] = a->im[i] * sa + b->im[i] * sb;
    }
    out->noise_budget = (a->noise_budget < b->noise_budget ? a->noise_budget : b->noise_budget) - 1;
    return 0;
}

int bf_ckks_mul(const BfCkksCipher *a, const BfCkksCipher *b, BfCkksCipher *out) {
    if (ensure_same(a, b, out) < 0) return -1;
    out->level = (a->level < b->level ? a->level : b->level) - 1;
    if (out->level < 0) { bf_ckks_cipher_free(out); return -1; }
    out->scale = a->scale * b->scale;

    for (int i = 0; i < out->slots; i++) {
        double ar = a->re[i], ai = a->im[i];
        double br = b->re[i], bi = b->im[i];
        out->re[i] = ar * br - ai * bi;
        out->im[i] = ar * bi + ai * br;
    }
    out->noise_budget = (a->noise_budget < b->noise_budget ? a->noise_budget : b->noise_budget) - 8;
    return 0;
}

int bf_ckks_mul_plain(const BfCkksCipher *a, const BfCkksPlain *p, BfCkksCipher *out) {
    if (!a || !p || !out || a->slots != p->slots) return -1;
    if (bf_ckks_cipher_init(out, a->slots, a->level - 1, a->scale) < 0) return -1;
    if (out->level < 0) { bf_ckks_cipher_free(out); return -1; }

    for (int i = 0; i < out->slots; i++) {
        double ar = a->re[i], ai = a->im[i];
        double pr = p->re[i], pi = p->im[i];
        out->re[i] = ar * pr - ai * pi;
        out->im[i] = ar * pi + ai * pr;
    }
    out->noise_budget = a->noise_budget - 4;
    return 0;
}

int bf_ckks_rotate(const BfCkksCipher *a, int steps, BfCkksCipher *out) {
    if (!a || !out) return -1;
    if (bf_ckks_cipher_init(out, a->slots, a->level, a->scale) < 0) return -1;
    int n = a->slots;
    int s = steps % n;
    if (s < 0) s += n;
    for (int i = 0; i < n; i++) {
        int src = (i - s + n) % n;
        out->re[i] = a->re[src];
        out->im[i] = a->im[src];
    }
    out->noise_budget = a->noise_budget - 1;
    return 0;
}

int bf_ckks_rescale(BfCkksCipher *ct, double target_scale) {
    if (!ct || target_scale <= 0.0) return -1;
    double ratio = ct->scale / target_scale;
    if (ratio <= 0.0) return -1;
    for (int i = 0; i < ct->slots; i++) {
        ct->re[i] /= ratio;
        ct->im[i] /= ratio;
    }
    ct->scale = target_scale;
    ct->level -= 1;
    if (ct->level < 0) return -1;
    ct->noise_budget -= 3;
    return 0;
}

int bf_ckks_bootstrap(BfCkksCipher *ct, int reset_level, double reset_scale) {
    if (!ct) return -1;
    if (reset_level < 1) reset_level = 8;
    if (reset_scale < BF_CKKS_MIN_SCALE) reset_scale = BF_CKKS_DEFAULT_SCALE;

    /* Simulate coefficient-to-slot and modular reduction refresh. */
    double old_scale = ct->scale;
    for (int i = 0; i < ct->slots; i++) {
        double val_re = ct->re[i] / old_scale;
        double val_im = ct->im[i] / old_scale;
        /* Clamp to represent bounded approx after modular reduction */
        if (val_re > 16.0) val_re = 16.0; if (val_re < -16.0) val_re = -16.0;
        if (val_im > 16.0) val_im = 16.0; if (val_im < -16.0) val_im = -16.0;
        ct->re[i] = val_re * reset_scale;
        ct->im[i] = val_im * reset_scale;
    }
    ct->scale = reset_scale;
    ct->level = reset_level;
    ct->noise_budget = 52;
    return 0;
}

/* ───────────────────────────────────────────────────────────────────────────
 * Depth-4 inference helpers
 * ─────────────────────────────────────────────────────────────────────────── */

/* Add plaintext bias in-place. */
static int add_plain_inplace(BfCkksCipher *ct, const BfCkksPlain *b) {
    if (!ct || !b || ct->slots != b->slots) return -1;
    for (int i = 0; i < ct->slots; i++) {
        ct->re[i] += b->re[i] * ct->scale;
        ct->im[i] += b->im[i] * ct->scale;
    }
    ct->noise_budget -= 1;
    return 0;
}

/* Cubic activation approximation:
 * phi(z) = 0.5 + 0.197 z - 0.004 z^3
 */
static int ckks_phi(BfCkksCipher *z) {
    if (!z) return -1;

    BfCkksCipher z2, z3, t1, t2, sum;
    if (bf_ckks_mul(z, z, &z2) < 0) return -1;
    if (bf_ckks_rescale(&z2, BF_CKKS_DEFAULT_SCALE) < 0) { bf_ckks_cipher_free(&z2); return -1; }

    if (bf_ckks_mul(&z2, z, &z3) < 0) { bf_ckks_cipher_free(&z2); return -1; }
    if (bf_ckks_rescale(&z3, BF_CKKS_DEFAULT_SCALE) < 0) {
        bf_ckks_cipher_free(&z2); bf_ckks_cipher_free(&z3); return -1;
    }

    /* t1 = 0.197 z */
    if (bf_ckks_cipher_init(&t1, z->slots, z->level, z->scale) < 0) {
        bf_ckks_cipher_free(&z2); bf_ckks_cipher_free(&z3); return -1;
    }
    for (int i = 0; i < z->slots; i++) {
        t1.re[i] = 0.197 * z->re[i];
        t1.im[i] = 0.197 * z->im[i];
    }

    /* t2 = -0.004 z^3 */
    if (bf_ckks_cipher_init(&t2, z3.slots, z3.level, z3.scale) < 0) {
        bf_ckks_cipher_free(&z2); bf_ckks_cipher_free(&z3); bf_ckks_cipher_free(&t1); return -1;
    }
    for (int i = 0; i < z3.slots; i++) {
        t2.re[i] = -0.004 * z3.re[i];
        t2.im[i] = -0.004 * z3.im[i];
    }

    if (bf_ckks_add(&t1, &t2, &sum) < 0) {
        bf_ckks_cipher_free(&z2); bf_ckks_cipher_free(&z3);
        bf_ckks_cipher_free(&t1); bf_ckks_cipher_free(&t2); return -1;
    }
    for (int i = 0; i < sum.slots; i++)
        sum.re[i] += 0.5 * sum.scale;

    bf_ckks_cipher_free(z);
    *z = sum;

    bf_ckks_cipher_free(&z2);
    bf_ckks_cipher_free(&z3);
    bf_ckks_cipher_free(&t1);
    bf_ckks_cipher_free(&t2);
    return 0;
}

int bf_ckks_infer_depth4(BfCkksCipher *x,
                         const BfCkksPlain *w1, const BfCkksPlain *b1,
                         const BfCkksPlain *w2, const BfCkksPlain *b2,
                         const BfCkksPlain *w3, const BfCkksPlain *b3,
                         const BfCkksPlain *w4, const BfCkksPlain *b4,
                         BfCkksCipher *out) {
    if (!x || !w1 || !w2 || !w3 || !w4 || !b1 || !b2 || !b3 || !b4 || !out) return -1;

    BfCkksCipher l1, l2, l3, l4;

    /* layer 1 */
    if (bf_ckks_mul_plain(x, w1, &l1) < 0) return -1;
    if (bf_ckks_rescale(&l1, BF_CKKS_DEFAULT_SCALE) < 0) { bf_ckks_cipher_free(&l1); return -1; }
    if (add_plain_inplace(&l1, b1) < 0) { bf_ckks_cipher_free(&l1); return -1; }
    if (ckks_phi(&l1) < 0) { bf_ckks_cipher_free(&l1); return -1; }

    /* layer 2 */
    if (bf_ckks_mul_plain(&l1, w2, &l2) < 0) { bf_ckks_cipher_free(&l1); return -1; }
    if (bf_ckks_rescale(&l2, BF_CKKS_DEFAULT_SCALE) < 0) {
        bf_ckks_cipher_free(&l1); bf_ckks_cipher_free(&l2); return -1;
    }
    if (add_plain_inplace(&l2, b2) < 0 || ckks_phi(&l2) < 0) {
        bf_ckks_cipher_free(&l1); bf_ckks_cipher_free(&l2); return -1;
    }

    /* bootstrap between layers 2 and 3 to restore depth */
    if (bf_ckks_bootstrap(&l2, 8, BF_CKKS_DEFAULT_SCALE) < 0) {
        bf_ckks_cipher_free(&l1); bf_ckks_cipher_free(&l2); return -1;
    }

    /* layer 3 */
    if (bf_ckks_mul_plain(&l2, w3, &l3) < 0) {
        bf_ckks_cipher_free(&l1); bf_ckks_cipher_free(&l2); return -1;
    }
    if (bf_ckks_rescale(&l3, BF_CKKS_DEFAULT_SCALE) < 0) {
        bf_ckks_cipher_free(&l1); bf_ckks_cipher_free(&l2); bf_ckks_cipher_free(&l3); return -1;
    }
    if (add_plain_inplace(&l3, b3) < 0 || ckks_phi(&l3) < 0) {
        bf_ckks_cipher_free(&l1); bf_ckks_cipher_free(&l2); bf_ckks_cipher_free(&l3); return -1;
    }

    /* layer 4 (output linear) */
    if (bf_ckks_mul_plain(&l3, w4, &l4) < 0) {
        bf_ckks_cipher_free(&l1); bf_ckks_cipher_free(&l2); bf_ckks_cipher_free(&l3); return -1;
    }
    if (bf_ckks_rescale(&l4, BF_CKKS_DEFAULT_SCALE) < 0) {
        bf_ckks_cipher_free(&l1); bf_ckks_cipher_free(&l2); bf_ckks_cipher_free(&l3); bf_ckks_cipher_free(&l4); return -1;
    }
    if (add_plain_inplace(&l4, b4) < 0) {
        bf_ckks_cipher_free(&l1); bf_ckks_cipher_free(&l2); bf_ckks_cipher_free(&l3); bf_ckks_cipher_free(&l4); return -1;
    }

    *out = l4;
    bf_ckks_cipher_free(&l1);
    bf_ckks_cipher_free(&l2);
    bf_ckks_cipher_free(&l3);
    return 0;
}
