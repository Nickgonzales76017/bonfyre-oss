/*
 * fpqx_algebra.c — FPQx Basic Algebra Operators Implementation
 *
 * Implements 7 primitive operators for computation in compressed domain.
 * Core innovation: No weight decompression required for inference.
 */

#include "fpqx_algebra.h"
#include "fpq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define HAVE_ACCELERATE 1
#else
#define HAVE_ACCELERATE 0
#endif

/* ═══════════════════════════════════════════════════════════════════
 * Internal Constants and Flags
 * ═══════════════════════════════════════════════════════════════════ */

#define FPQ_FLAG_E8_INT7       0x01
#define FPQ_FLAG_TILE_6BIT     0x02
#define FPQ_FLAG_FP8_SCALES    0x04
#define FPQ_FLAG_E8_ENTROPY    0x08
#define FPQ_FLAG_PACKED_V12    (FPQ_FLAG_E8_ENTROPY | FPQ_FLAG_TILE_6BIT)

#define FPQ_BLOCK_SIZE 256
#define PI 3.14159265358979323846f

static bool g_simd_avx2_enabled = false;
static bool g_simd_neon_enabled = false;

/* ═══════════════════════════════════════════════════════════════════
 * FWHT Implementation (Fast Walsh-Hadamard Transform)
 * ═══════════════════════════════════════════════════════════════════ */

static void fpqx_fwht_256(float *x) {
    /* In-place FWHT for n=256, with 1/√n normalization */
    int n = 256;
    float scale = 1.0f / sqrtf((float)n);
    
    /* Radix-2 decimation */
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    
    /* Normalize */
    for (int i = 0; i < n; i++) {
        x[i] *= scale;
    }
}

static void fpqx_fwht_inverse_256(float *x) {
    /* FWHT is self-inverse after scaling */
    fpqx_fwht_256(x);
}

/* ═══════════════════════════════════════════════════════════════════
 * E8 Lattice Utilities
 * ═══════════════════════════════════════════════════════════════════ */

static void fpqx_e8_decode_block(const uint8_t *e8_rans, float *coords) {
    /* Placeholder: decode rANS-compressed E8 coords to float
     * In production: call actual rANS decoder + INT7 unpacking
     * For now: stub with zeros
     */
    memset(coords, 0, 256 * sizeof(float));
    /* TODO: Implement rANS decode + unpack_int7 */
}

static void fpqx_e8_encode_block(const float *coords, uint8_t *e8_rans) {
    /* Placeholder: encode float coords to rANS E8
     * TODO: Implement snap to E8 + pack_int7 + rANS encode
     */
    memset(e8_rans, 0, 210);
}

/* ═══════════════════════════════════════════════════════════════════
 * Core Utilities
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_load(const char *path, FPQTensor ***tensors, int *n_tensors) {
    /* Placeholder: Load FPQ v12 file
     * TODO: Implement full .fpq parser (magic, headers, blocks)
     */
    fprintf(stderr, "fpqx_load: TODO - implement .fpq v12 parser\n");
    *tensors = NULL;
    *n_tensors = 0;
    return -1;
}

void fpqx_sli_prepare(FPQTensor *w) {
    if (!w || w->sli_prepared) return;
    
    /* Allocate z and signs if not already done */
    if (!w->z_precomputed) {
        w->z_precomputed = (float **)calloc(w->n_blocks, sizeof(float *));
        for (uint32_t b = 0; b < w->n_blocks; b++) {
            w->z_precomputed[b] = (float *)calloc(256, sizeof(float));
        }
    }
    
    if (!w->signs) {
        w->signs = (uint64_t **)calloc(w->n_blocks, sizeof(uint64_t *));
        for (uint32_t b = 0; b < w->n_blocks; b++) {
            w->signs[b] = (uint64_t *)calloc(4, sizeof(uint64_t));
            /* TODO: Read actual signs from .fpq file */
        }
    }
    
    /* For each block: decode E8+tile+QJL → FWHT → store as z */
    for (uint32_t b = 0; b < w->n_blocks; b++) {
        float z_prime[256];
        
        /* Step 1: Decode E8 coords */
        fpqx_e8_decode_block(w->blocks[b].e8_coords_rans, z_prime);
        
        /* Step 2: Scale by coord_scale */
        uint16_t scale_fp16 = w->blocks[b].coord_scale;
        float scale = *(float *)&scale_fp16;  /* TODO: proper FP16→FP32 */
        for (int i = 0; i < 256; i++) {
            z_prime[i] *= scale;
        }
        
        /* Step 3: Add tile corrections (RVQ)
         * TODO: Decode tile_indices and apply corrections */
        
        /* Step 4: FWHT-on-z optimization */
        fpqx_fwht_256(z_prime);
        
        /* Step 5: Store */
        memcpy(w->z_precomputed[b], z_prime, 256 * sizeof(float));
    }
    
    w->sli_prepared = true;
    w->fwht_on_z = true;
}

void fpqx_sli_prepare_all(FPQTensor **tensors, int n_tensors) {
    #pragma omp parallel for if(n_tensors > 4)
    for (int i = 0; i < n_tensors; i++) {
        fpqx_sli_prepare(tensors[i]);
    }
}

void fpqx_free(FPQTensor *w) {
    if (!w) return;
    
    if (w->lr_U) free(w->lr_U);
    if (w->lr_V) free(w->lr_V);
    if (w->lr_sigma) free(w->lr_sigma);
    
    if (w->blocks) {
        for (uint32_t b = 0; b < w->n_blocks; b++) {
            if (w->blocks[b].e8_coords_rans) free(w->blocks[b].e8_coords_rans);
            if (w->blocks[b].tile_indices) free(w->blocks[b].tile_indices);
        }
        free(w->blocks);
    }
    
    if (w->z_precomputed) {
        for (uint32_t b = 0; b < w->n_blocks; b++) {
            if (w->z_precomputed[b]) free(w->z_precomputed[b]);
        }
        free(w->z_precomputed);
    }
    
    if (w->signs) {
        for (uint32_t b = 0; b < w->n_blocks; b++) {
            if (w->signs[b]) free(w->signs[b]);
        }
        free(w->signs);
    }
    
    if (w->data_buffer) free(w->data_buffer);
    free(w);
}

void fpqx_free_all(FPQTensor **tensors, int n_tensors) {
    for (int i = 0; i < n_tensors; i++) {
        fpqx_free(tensors[i]);
    }
    free(tensors);
}

/* ═══════════════════════════════════════════════════════════════════
 * Operator A — Addition
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_add(const FPQTensor *a, const FPQTensor *b, FPQTensor **result) {
    if (!a || !b || !result) return -1;
    if (a->rows != b->rows || a->cols != b->cols) return -2;
    if (a->sli_prepared || b->sli_prepared) {
        fprintf(stderr, "fpqx_add: Cannot add SLI-prepared tensors (E8 data lost)\n");
        return -3;
    }
    
    /* Allocate result tensor */
    FPQTensor *r = (FPQTensor *)calloc(1, sizeof(FPQTensor));
    r->rows = a->rows;
    r->cols = a->cols;
    r->n_blocks = a->n_blocks;
    r->flags = a->flags;
    r->base_scale = (a->base_scale + b->base_scale) / 2.0f;
    r->pid_alpha = a->pid_alpha;
    
    /* Allocate blocks */
    r->blocks = calloc(r->n_blocks, sizeof(*r->blocks));
    
    /* For each block: decode, add, re-encode */
    for (uint32_t blk = 0; blk < r->n_blocks; blk++) {
        /* Allocate block memory */
        r->blocks[blk].e8_coords_rans = (uint8_t *)calloc(210, 1);
        r->blocks[blk].tile_indices = (uint8_t *)calloc(12, 1);
        
        float coords_a[256], coords_b[256], coords_sum[256];
        
        /* Decode */
        fpqx_e8_decode_block(a->blocks[blk].e8_coords_rans, coords_a);
        fpqx_e8_decode_block(b->blocks[blk].e8_coords_rans, coords_b);
        
        /* Add */
        for (int i = 0; i < 256; i++) {
            coords_sum[i] = coords_a[i] + coords_b[i];
        }
        
        /* Re-snap to E8 and encode */
        fpqx_e8_encode_block(coords_sum, r->blocks[blk].e8_coords_rans);
        
        /* Average scales */
        r->blocks[blk].coord_scale = (a->blocks[blk].coord_scale + b->blocks[blk].coord_scale) / 2;
        r->blocks[blk].warp_norm = (a->blocks[blk].warp_norm + b->blocks[blk].warp_norm) / 2;
        
        /* Copy tile indices from a (TODO: smarter merging) */
        r->blocks[blk].tile_indices = (uint8_t *)malloc(12);
        memcpy(r->blocks[blk].tile_indices, a->blocks[blk].tile_indices, 12);
    }
    
    *result = r;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Operator M — Multiplication by Scalar
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_scale(const FPQTensor *a, float scalar, FPQTensor **result) {
    if (!a || !result) return -1;
    
    /* Allocate result */
    FPQTensor *r = (FPQTensor *)malloc(sizeof(FPQTensor));
    memcpy(r, a, sizeof(FPQTensor));
    
    /* Duplicate blocks */
    r->blocks = calloc(r->n_blocks, sizeof(*r->blocks));
    for (uint32_t b = 0; b < r->n_blocks; b++) {
        /* E8 coords unchanged */
        r->blocks[b].e8_coords_rans = (uint8_t *)malloc(210);
        memcpy(r->blocks[b].e8_coords_rans, a->blocks[b].e8_coords_rans, 210);
        
        /* Tiles unchanged */
        r->blocks[b].tile_indices = (uint8_t *)malloc(12);
        memcpy(r->blocks[b].tile_indices, a->blocks[b].tile_indices, 12);
        
        /* SCALE the scales */
        float coord_scale_fp32 = *(float *)&a->blocks[b].coord_scale;  /* TODO: FP16 */
        coord_scale_fp32 *= scalar;
        r->blocks[b].coord_scale = *(uint16_t *)&coord_scale_fp32;
        
        float warp_norm_fp32 = *(float *)&a->blocks[b].warp_norm;
        warp_norm_fp32 *= scalar;
        r->blocks[b].warp_norm = *(uint16_t *)&warp_norm_fp32;
    }
    
    /* Scale base_scale */
    r->base_scale = a->base_scale * scalar;
    
    /* If SLI prepared, scale z arrays */
    if (a->sli_prepared) {
        r->z_precomputed = (float **)calloc(r->n_blocks, sizeof(float *));
        for (uint32_t b = 0; b < r->n_blocks; b++) {
            r->z_precomputed[b] = (float *)malloc(256 * sizeof(float));
            for (int i = 0; i < 256; i++) {
                r->z_precomputed[b][i] = a->z_precomputed[b][i] * scalar;
            }
        }
        
        /* Signs unchanged */
        r->signs = (uint64_t **)calloc(r->n_blocks, sizeof(uint64_t *));
        for (uint32_t b = 0; b < r->n_blocks; b++) {
            r->signs[b] = (uint64_t *)malloc(4 * sizeof(uint64_t));
            memcpy(r->signs[b], a->signs[b], 4 * sizeof(uint64_t));
        }
    }
    
    *result = r;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Operator Π — Projection
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_project(const FPQTensor *a, int target_dim, FPQTensor **result) {
    if (!a || !result) return -1;
    if (target_dim < 1 || target_dim > 256) return -2;
    if (!a->sli_prepared) {
        fprintf(stderr, "fpqx_project: Tensor must be SLI-prepared (operate on z)\n");
        return -3;
    }
    
    /* Allocate result */
    FPQTensor *r = (FPQTensor *)malloc(sizeof(FPQTensor));
    memcpy(r, a, sizeof(FPQTensor));
    
    /* Projection operates only on z, nullify blocks to avoid double-free */
    r->blocks = NULL;
    r->data_buffer = NULL;
    
    /* Duplicate z arrays and zero high-frequency components */
    r->z_precomputed = (float **)calloc(r->n_blocks, sizeof(float *));
    for (uint32_t b = 0; b < r->n_blocks; b++) {
        r->z_precomputed[b] = (float *)calloc(256, sizeof(float));
        
        /* Copy low-frequency */
        memcpy(r->z_precomputed[b], a->z_precomputed[b], target_dim * sizeof(float));
        
        /* Zero high-frequency (already zeroed by calloc) */
    }
    
    /* Copy signs */
    r->signs = (uint64_t **)calloc(r->n_blocks, sizeof(uint64_t *));
    for (uint32_t b = 0; b < r->n_blocks; b++) {
        r->signs[b] = (uint64_t *)malloc(4 * sizeof(uint64_t));
        memcpy(r->signs[b], a->signs[b], 4 * sizeof(uint64_t));
    }
    
    *result = r;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Operator D — Dot Product (SLI)
 * ═══════════════════════════════════════════════════════════════════ */

float fpqx_sli_dot(const FPQTensor *w, const float *x, int n) {
    if (!w || !x) return NAN;
    if (!w->sli_prepared) {
        fprintf(stderr, "fpqx_sli_dot: Tensor not SLI-prepared. Call fpqx_sli_prepare first.\n");
        return NAN;
    }
    if ((uint32_t)n != w->n_blocks * 256) {
        fprintf(stderr, "fpqx_sli_dot: Dimension mismatch (%d != %u)\n", n, w->n_blocks * 256);
        return NAN;
    }
    
    float score = 0.0f;
    
    /* Phase 0: Low-rank contribution (if present) */
    if (w->lr_rank > 0 && w->lr_U && w->lr_V) {
        /* score += σ_i * (u_i @ x) * v_i (simplified for 1D) */
        /* TODO: Full implementation requires knowing which row this is */
    }
    
    /* Phase 1: SLI over blocks */
    for (uint32_t b = 0; b < w->n_blocks; b++) {
        float x_block[256];
        memcpy(x_block, x + b * 256, 256 * sizeof(float));
        
        /* Apply sign flips */
        uint64_t *sign_mask = w->signs[b];
        for (int i = 0; i < 256; i++) {
            int word = i / 64;
            int bit = i % 64;
            bool flip = (sign_mask[word] >> bit) & 1;
            if (flip) x_block[i] = -x_block[i];
        }
        
        /* FWHT on x (activation side) */
        fpqx_fwht_256(x_block);
        
        /* Dot with precomputed z (FWHT already applied to z) */
        for (int i = 0; i < 256; i++) {
            score += w->z_precomputed[b][i] * x_block[i];
        }
    }
    
    return score;
}

void fpqx_sli_dot_batch(const FPQTensor *w, const float *X, 
                        float *y, int n, int batch_size) {
    #pragma omp parallel for if(batch_size > 4)
    for (int b = 0; b < batch_size; b++) {
        y[b] = fpqx_sli_dot(w, X + b * n, n);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Operator Λ — Low-Rank Decomposition
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_extract_lr(const FPQTensor *w, int rank, 
                    float **U, float **V, float **sigma) {
    if (!w || !U || !V) return -1;
    if (w->pid_alpha != -9.0f) {
        fprintf(stderr, "fpqx_extract_lr: No LR header (pid_alpha=%.1f, need -9.0)\n", w->pid_alpha);
        return -2;
    }
    if (rank > w->lr_rank) {
        fprintf(stderr, "fpqx_extract_lr: Requested rank %d > stored %u\n", rank, w->lr_rank);
        return -3;
    }
    
    /* Allocate and copy */
    *U = (float *)malloc(w->rows * rank * sizeof(float));
    *V = (float *)malloc(rank * w->cols * sizeof(float));
    if (sigma) *sigma = (float *)malloc(rank * sizeof(float));
    
    for (uint32_t r = 0; r < (uint32_t)rank; r++) {
        for (uint32_t i = 0; i < w->rows; i++) {
            (*U)[i * rank + r] = w->lr_U[i * w->lr_rank + r];
        }
    }
    
    for (uint32_t r = 0; r < (uint32_t)rank; r++) {
        for (uint32_t j = 0; j < w->cols; j++) {
            (*V)[r * w->cols + j] = w->lr_V[r * w->cols + j];
        }
    }
    
    if (sigma && w->lr_sigma) {
        memcpy(*sigma, w->lr_sigma, rank * sizeof(float));
    }
    
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Operator H — Hadamard Transform
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_fwht(const FPQTensor *a, FPQTensor **result) {
    if (!a || !result) return -1;
    if (!a->sli_prepared) {
        fprintf(stderr, "fpqx_fwht: Tensor must be SLI-prepared\n");
        return -2;
    }
    
    /* Allocate result */
    FPQTensor *r = (FPQTensor *)malloc(sizeof(FPQTensor));
    memcpy(r, a, sizeof(FPQTensor));
    
    /* FWHT operates only on z, nullify blocks to avoid double-free */
    r->blocks = NULL;
    r->data_buffer = NULL;
    
    /* Duplicate z arrays and apply FWHT */
    r->z_precomputed = (float **)calloc(r->n_blocks, sizeof(float *));
    for (uint32_t b = 0; b < r->n_blocks; b++) {
        r->z_precomputed[b] = (float *)malloc(256 * sizeof(float));
        memcpy(r->z_precomputed[b], a->z_precomputed[b], 256 * sizeof(float));
        
        /* Apply FWHT (self-inverse) */
        fpqx_fwht_256(r->z_precomputed[b]);
    }
    
    /* Toggle fwht_on_z flag (FWHT² = I) */
    r->fwht_on_z = !a->fwht_on_z;
    
    /* Copy signs */
    r->signs = (uint64_t **)calloc(r->n_blocks, sizeof(uint64_t *));
    for (uint32_t b = 0; b < r->n_blocks; b++) {
        r->signs[b] = (uint64_t *)malloc(4 * sizeof(uint64_t));
        memcpy(r->signs[b], a->signs[b], 4 * sizeof(uint64_t));
    }
    
    *result = r;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Operator I — Inference (Full Layer)
 * ═══════════════════════════════════════════════════════════════════ */

void fpqx_linear_layer(const FPQTensor *W, const float *x, float *y,
                       int out_dim, int in_dim) {
    /* W is [out_dim × in_dim] row-major */
    #pragma omp parallel for if(out_dim > 32)
    for (int o = 0; o < out_dim; o++) {
        /* Each output neuron = one SLI dot product */
        /* Assumes W is an array of out_dim tensors, each 1 × in_dim */
        /* TODO: Proper tensor layout for multi-row matrices */
        y[o] = fpqx_sli_dot(&W[o], x, in_dim);
    }
}

void fpqx_linear_layer_optimized(const FPQTensor *W, const float *x_fwht,
                                  float *y, int out_dim, int in_dim) {
    /* Assume x_fwht is already FWHTed in blocks of 256 */
    /* Optimized version: skip FWHT on x side */
    
    int n_blocks = in_dim / 256;
    
    #pragma omp parallel for if(out_dim > 32)
    for (int o = 0; o < out_dim; o++) {
        float score = 0.0f;
        
        for (int b = 0; b < n_blocks; b++) {
            /* Apply signs */
            float x_signed[256];
            const float *x_block = x_fwht + b * 256;
            uint64_t *signs = W[o].signs[b];
            
            for (int i = 0; i < 256; i++) {
                int word = i / 64;
                int bit = i % 64;
                bool flip = (signs[word] >> bit) & 1;
                x_signed[i] = flip ? -x_block[i] : x_block[i];
            }
            
            /* Dot with z */
            for (int i = 0; i < 256; i++) {
                score += W[o].z_precomputed[b][i] * x_signed[i];
            }
        }
        
        y[o] = score;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Validation and Diagnostics
 * ═══════════════════════════════════════════════════════════════════ */

void fpqx_print_info(const FPQTensor *w) {
    if (!w) return;
    
    printf("FPQTensor Info:\n");
    printf("  Dimensions: %u × %u\n", w->rows, w->cols);
    printf("  Blocks: %u (256 elem each)\n", w->n_blocks);
    printf("  Flags: 0x%02X\n", w->flags);
    printf("  Base scale: %.6f\n", w->base_scale);
    printf("  PID alpha: %.1f ", w->pid_alpha);
    if (w->pid_alpha == -9.0f) printf("(v9 with LR rank=%u)\n", w->lr_rank);
    else if (w->pid_alpha == -8.0f) printf("(v8)\n");
    else printf("(unknown)\n");
    
    printf("  SLI prepared: %s\n", w->sli_prepared ? "yes" : "no");
    if (w->sli_prepared) {
        printf("  FWHT-on-z: %s\n", w->fwht_on_z ? "yes" : "no");
    }
}

void fpqx_set_simd(bool enable_avx2, bool enable_neon) {
    g_simd_avx2_enabled = enable_avx2;
    g_simd_neon_enabled = enable_neon;
}

void fpqx_get_simd_status(bool *avx2_enabled, bool *neon_enabled) {
    if (avx2_enabled) *avx2_enabled = g_simd_avx2_enabled;
    if (neon_enabled) *neon_enabled = g_simd_neon_enabled;
}
