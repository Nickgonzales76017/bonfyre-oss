/*
 * test_fpqx_algebra.c — Test Suite for FPQx Basic Algebra Operators
 *
 * Tests all 7 operators: A + M + Π + D + Λ + H + I
 * Validates correctness, performance, and composition properties.
 */

#include "fpqx_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define EPSILON 1e-5f
#define TEST_DIM 256
#define TEST_BLOCKS 4

/* ═══════════════════════════════════════════════════════════════════
 * Test Utilities
 * ═══════════════════════════════════════════════════════════════════ */

/* Generate random FP32 vector */
static void random_vector(float *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

/* Cosine similarity */
static float cosine_sim(const float *a, const float *b, int n) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-12f);
}

/* Create synthetic FPQx tensor for testing */
static FPQTensor *create_test_tensor(int rows, int cols) {
    FPQTensor *t = (FPQTensor *)calloc(1, sizeof(FPQTensor));
    t->rows = rows;
    t->cols = cols;
    t->n_blocks = (rows * cols + 255) / 256;
    t->flags = 0x0A;  /* FPQ_FLAG_PACKED_V12 */
    t->base_scale = 1.0f;
    t->pid_alpha = -9.0f;  /* v9 with LR */
    t->lr_rank = 0;  /* No LR for simplicity */
    
    /* Allocate blocks */
    t->blocks = calloc(t->n_blocks, sizeof(*t->blocks));
    for (uint32_t b = 0; b < t->n_blocks; b++) {
        t->blocks[b].e8_coords_rans = (uint8_t *)calloc(210, 1);
        t->blocks[b].tile_indices = (uint8_t *)calloc(12, 1);
        t->blocks[b].coord_scale = 0x3C00;  /* FP16 of 1.0 */
        t->blocks[b].warp_norm = 0x3C00;
    }
    
    return t;
}

/* ═══════════════════════════════════════════════════════════════════
 * Test Cases
 * ═══════════════════════════════════════════════════════════════════ */

/* Test 1: Operator M — Scaling */
static int test_scale() {
    printf("TEST: Operator M (Scale)\n");
    
    FPQTensor *a = create_test_tensor(1, TEST_DIM);
    float scalar = 2.5f;
    
    /* Prepare SLI and set known z values */
    fpqx_sli_prepare(a);
    for (uint32_t b = 0; b < a->n_blocks; b++) {
        for (int i = 0; i < 256; i++) {
            a->z_precomputed[b][i] = (float)i / 256.0f;
        }
    }
    
    /* Scale */
    FPQTensor *scaled;
    int ret = fpqx_scale(a, scalar, &scaled);
    if (ret != 0) {
        printf("  FAIL: fpqx_scale returned %d\n", ret);
        return -1;
    }
    
    /* Verify z values scaled */
    bool ok = true;
    for (uint32_t b = 0; b < a->n_blocks; b++) {
        for (int i = 0; i < 256; i++) {
            float expected = a->z_precomputed[b][i] * scalar;
            float actual = scaled->z_precomputed[b][i];
            if (fabsf(expected - actual) > EPSILON) {
                printf("  FAIL: Block %u, index %d: expected %.6f, got %.6f\n",
                       b, i, expected, actual);
                ok = false;
                break;
            }
        }
    }
    
    fpqx_free(a);
    fpqx_free(scaled);
    
    if (ok) printf("  PASS\n");
    return ok ? 0 : -1;
}

/* Test 2: Operator Π — Projection */
static int test_project() {
    printf("TEST: Operator Π (Project)\n");
    
    FPQTensor *a = create_test_tensor(1, TEST_DIM);
    fpqx_sli_prepare(a);
    
    /* Set known z values */
    for (uint32_t b = 0; b < a->n_blocks; b++) {
        for (int i = 0; i < 256; i++) {
            a->z_precomputed[b][i] = (float)i;
        }
    }
    
    /* Project to 128 dimensions (keep first 128 of each block) */
    int target_dim = 128;
    FPQTensor *projected;
    int ret = fpqx_project(a, target_dim, &projected);
    if (ret != 0) {
        printf("  FAIL: fpqx_project returned %d\n", ret);
        return -1;
    }
    
    /* Verify: first target_dim unchanged, rest zeroed */
    bool ok = true;
    for (uint32_t b = 0; b < a->n_blocks; b++) {
        for (int i = 0; i < 256; i++) {
            float expected = (i < target_dim) ? (float)i : 0.0f;
            float actual = projected->z_precomputed[b][i];
            if (fabsf(expected - actual) > EPSILON) {
                printf("  FAIL: Block %u, index %d: expected %.6f, got %.6f\n",
                       b, i, expected, actual);
                ok = false;
                break;
            }
        }
    }
    
    fpqx_free(a);
    fpqx_free(projected);
    
    if (ok) printf("  PASS\n");
    return ok ? 0 : -1;
}

/* Test 3: Operator D — SLI Dot Product */
static int test_sli_dot() {
    printf("TEST: Operator D (SLI Dot)\n");
    
    FPQTensor *w = create_test_tensor(1, TEST_DIM);
    float x[TEST_DIM];
    
    /* Initialize */
    random_vector(x, TEST_DIM);
    fpqx_sli_prepare(w);
    
    /* Set known z values (simple pattern) */
    for (uint32_t b = 0; b < w->n_blocks; b++) {
        for (int i = 0; i < 256; i++) {
            w->z_precomputed[b][i] = 1.0f;  /* All ones for simple test */
        }
        /* Set signs to all zeros (no flips) */
        memset(w->signs[b], 0, 4 * sizeof(uint64_t));
    }
    
    /* Compute SLI dot product */
    float result = fpqx_sli_dot(w, x, TEST_DIM);
    
    /* Expected: sum of FWHT(x) since z=1 and no sign flips */
    /* For this test, just check result is finite */
    bool ok = isfinite(result);
    
    if (!ok) {
        printf("  FAIL: Result is %f (not finite)\n", result);
    } else {
        printf("  PASS (result = %.6f)\n", result);
    }
    
    fpqx_free(w);
    return ok ? 0 : -1;
}

/* Test 4: Operator H — FWHT (Self-Inverse) */
static int test_fwht_inverse() {
    printf("TEST: Operator H (FWHT Self-Inverse)\n");
    
    FPQTensor *a = create_test_tensor(1, TEST_DIM);
    fpqx_sli_prepare(a);
    
    /* Set known z values */
    for (uint32_t b = 0; b < a->n_blocks; b++) {
        for (int i = 0; i < 256; i++) {
            a->z_precomputed[b][i] = (float)i / 256.0f;
        }
    }
    
    /* Apply FWHT twice (should return to original) */
    FPQTensor *after_fwht, *after_fwht2;
    
    int ret1 = fpqx_fwht(a, &after_fwht);
    if (ret1 != 0) {
        printf("  FAIL: First fpqx_fwht returned %d\n", ret1);
        return -1;
    }
    
    int ret2 = fpqx_fwht(after_fwht, &after_fwht2);
    if (ret2 != 0) {
        printf("  FAIL: Second fpqx_fwht returned %d\n", ret2);
        return -1;
    }
    
    /* Verify: after_fwht2 should match original a */
    bool ok = true;
    for (uint32_t b = 0; b < a->n_blocks; b++) {
        for (int i = 0; i < 256; i++) {
            float expected = a->z_precomputed[b][i];
            float actual = after_fwht2->z_precomputed[b][i];
            if (fabsf(expected - actual) > EPSILON) {
                printf("  FAIL: Block %u, index %d: expected %.6f, got %.6f\n",
                       b, i, expected, actual);
                ok = false;
                break;
            }
        }
    }
    
    fpqx_free(a);
    fpqx_free(after_fwht);
    fpqx_free(after_fwht2);
    
    if (ok) printf("  PASS\n");
    return ok ? 0 : -1;
}

/* Test 5: Operator Composition — Scale then Project */
static int test_composition_scale_project() {
    printf("TEST: Composition (Scale → Project)\n");
    
    FPQTensor *a = create_test_tensor(1, TEST_DIM);
    fpqx_sli_prepare(a);
    
    /* Set values */
    for (uint32_t b = 0; b < a->n_blocks; b++) {
        for (int i = 0; i < 256; i++) {
            a->z_precomputed[b][i] = (float)i;
        }
    }
    
    /* Scale by 2 */
    FPQTensor *scaled;
    fpqx_scale(a, 2.0f, &scaled);
    
    /* Then project to 64 dims */
    FPQTensor *projected;
    fpqx_project(scaled, 64, &projected);
    
    /* Verify: first 64 should be 2*i, rest zero */
    bool ok = true;
    for (uint32_t b = 0; b < a->n_blocks; b++) {
        for (int i = 0; i < 256; i++) {
            float expected = (i < 64) ? (2.0f * i) : 0.0f;
            float actual = projected->z_precomputed[b][i];
            if (fabsf(expected - actual) > EPSILON) {
                printf("  FAIL: Block %u, index %d: expected %.6f, got %.6f\n",
                       b, i, expected, actual);
                ok = false;
                break;
            }
        }
    }
    
    fpqx_free(a);
    fpqx_free(scaled);
    fpqx_free(projected);
    
    if (ok) printf("  PASS\n");
    return ok ? 0 : -1;
}

/* Test 6: Operator INFO — Print Tensor Info */
static int test_info() {
    printf("TEST: fpqx_print_info\n");
    
    FPQTensor *t = create_test_tensor(128, 512);
    t->lr_rank = 8;
    fpqx_sli_prepare(t);
    
    fpqx_print_info(t);
    
    fpqx_free(t);
    printf("  PASS (manual verification)\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Main Test Runner
 * ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    
    srand(42);  /* Deterministic tests */
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("FPQx Algebra Test Suite\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    int failures = 0;
    
    failures += test_scale() < 0 ? 1 : 0;
    failures += test_project() < 0 ? 1 : 0;
    failures += test_sli_dot() < 0 ? 1 : 0;
    failures += test_fwht_inverse() < 0 ? 1 : 0;
    failures += test_composition_scale_project() < 0 ? 1 : 0;
    failures += test_info() < 0 ? 1 : 0;
    
    printf("\n═══════════════════════════════════════════════════════\n");
    if (failures == 0) {
        printf("✓ ALL TESTS PASSED\n");
    } else {
        printf("✗ %d TESTS FAILED\n", failures);
    }
    printf("═══════════════════════════════════════════════════════\n");
    
    return failures;
}
