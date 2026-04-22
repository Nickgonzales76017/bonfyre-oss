/*
 * bf_he_ckks.h — CKKS-style homomorphic inference primitives (depth-4)
 *
 * This module provides a practical, self-contained CKKS-like arithmetic layer
 * for encrypted vector inference in pure C.  It is intended for local Bonfyre
 * scoring/privacy paths where we need deterministic depth accounting and
 * bootstrapping behavior without external HE runtime dependencies.
 *
 * IMPORTANT
 *   This is a compact educational/runtime implementation, not a drop-in
 *   replacement for production HE suites (SEAL/PALISADE/OpenFHE).
 *   It models CKKS pipeline semantics: scale management, multiplicative depth,
 *   relinearization proxy, and bootstrapping refresh.
 *
 * Ciphertext model
 *   c(x) = (m(x) + e(x)) * scale mod q_l
 * where `level` tracks remaining modulus chain depth.
 *
 * We represent packed slots as two float arrays (re/im) for direct SIMD use.
 */
#pragma once
#ifndef BF_HE_CKKS_H
#define BF_HE_CKKS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BF_CKKS_MAX_SLOTS 4096
#define BF_CKKS_DEFAULT_SCALE 1099511627776.0 /* 2^40 */
#define BF_CKKS_MIN_SCALE 65536.0             /* 2^16 */

typedef struct {
    int    slots;          /* vector width */
    int    level;          /* remaining depth in modulus chain */
    int    noise_budget;   /* rough bit budget */
    double scale;          /* CKKS scale */
    double *re;            /* slot real parts */
    double *im;            /* slot imaginary parts */
} BfCkksCipher;

typedef struct {
    int    slots;
    double *re;
    double *im;
} BfCkksPlain;

/* ── Lifecycle ────────────────────────────────────────────────────────────── */

int  bf_ckks_cipher_init(BfCkksCipher *ct, int slots, int level, double scale);
void bf_ckks_cipher_free(BfCkksCipher *ct);

int  bf_ckks_plain_init(BfCkksPlain *pt, int slots);
void bf_ckks_plain_free(BfCkksPlain *pt);

/* ── Encode / Decode (simulated) ──────────────────────────────────────────── */

int bf_ckks_encode(const double *re, const double *im, int slots, BfCkksPlain *pt);
int bf_ckks_encrypt(const BfCkksPlain *pt, BfCkksCipher *ct, uint64_t seed);
int bf_ckks_decrypt(const BfCkksCipher *ct, BfCkksPlain *pt);

/* ── Arithmetic ────────────────────────────────────────────────────────────── */

int bf_ckks_add(const BfCkksCipher *a, const BfCkksCipher *b, BfCkksCipher *out);
int bf_ckks_mul(const BfCkksCipher *a, const BfCkksCipher *b, BfCkksCipher *out);
int bf_ckks_mul_plain(const BfCkksCipher *a, const BfCkksPlain *p, BfCkksCipher *out);
int bf_ckks_rotate(const BfCkksCipher *a, int steps, BfCkksCipher *out);

/* Rescale after multiplication to control scale growth; consumes one level. */
int bf_ckks_rescale(BfCkksCipher *ct, double target_scale);

/* ── Bootstrapping + depth-4 inference ───────────────────────────────────── */

/* Refresh ciphertext to a higher level and reset noise budget. */
int bf_ckks_bootstrap(BfCkksCipher *ct, int reset_level, double reset_scale);

/*
 * Evaluate a 4-layer polynomial network under CKKS semantics:
 *   y = W4 * phi(W3 * phi(W2 * phi(W1 * x + b1) + b2) + b3) + b4
 * where phi(z) ≈ 0.5 + 0.197 z - 0.004 z^3 (low-depth cubic approx).
 *
 * Inputs are packed slot vectors; weights/biases are same-size plains.
 * Multiplicative depth target: 4 with one bootstrap in the middle.
 */
int bf_ckks_infer_depth4(BfCkksCipher *x,
                         const BfCkksPlain *w1, const BfCkksPlain *b1,
                         const BfCkksPlain *w2, const BfCkksPlain *b2,
                         const BfCkksPlain *w3, const BfCkksPlain *b3,
                         const BfCkksPlain *w4, const BfCkksPlain *b4,
                         BfCkksCipher *out);

#ifdef __cplusplus
}
#endif
#endif /* BF_HE_CKKS_H */
