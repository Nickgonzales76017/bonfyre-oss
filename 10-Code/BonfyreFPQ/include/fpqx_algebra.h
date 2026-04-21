/*
 * fpqx_algebra.h — FPQx Basic Algebra Operators
 *
 * The 7 primitive operators for computation in compressed domain:
 *   A — Addition in E8 Lattice Space
 *   M — Multiplication by Scalar
 *   Π — Projection to Subspace
 *   D — Dot Product (SLI - Spectral Lattice Inference)
 *   Λ — Low-Rank Decomposition
 *   H — Hadamard Transform (FWHT)
 *   I — Inference (Full Layer)
 *
 * Core innovation: Compute directly on FPQ v12 compressed tensors
 * without decompression. 4.4× bandwidth reduction, 2.5× speed improvement.
 *
 * Usage:
 *   FPQTensor *W = fpqx_load("model.fpq");
 *   fpqx_sli_prepare(W);  // One-time FWHT-on-z optimization
 *   float y = fpqx_sli_dot(W, x, n);  // Inference without decompression
 */

#ifndef BONFYRE_FPQX_ALGEBRA_H
#define BONFYRE_FPQX_ALGEBRA_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════
 * FPQx Tensor Handle
 * Represents a compressed tensor in FPQ v12 format with SLI support
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Tensor metadata */
    uint32_t rows, cols;
    uint32_t n_blocks;      /* Number of 256-element blocks */
    uint8_t  flags;         /* FPQ_FLAG_PACKED_V12 etc */
    float    base_scale;
    float    pid_alpha;     /* -9.0f = v9 (LR), -8.0f = v8 */
    
    /* Low-rank header (if pid_alpha == -9.0f) */
    uint16_t lr_rank;       /* Ghost head rank (0-15) */
    float   *lr_U;          /* [rows × lr_rank] */
    float   *lr_V;          /* [lr_rank × cols] */
    float   *lr_sigma;      /* [lr_rank] singular values */
    
    /* Per-block compressed data */
    struct {
        uint8_t  *e8_coords_rans;  /* rANS entropy-coded E8 (~ 210 B) */
        uint8_t  *tile_indices;    /* 6-bit packed tiles (12 B) */
        uint16_t  coord_scale;     /* FP16 E8 scale */
        uint16_t  warp_norm;       /* FP16 μ-law warp norm */
    } *blocks;
    
    /* SLI-prepared data (in-place over E8 region after fpqx_sli_prepare) */
    float   **z_precomputed;   /* [n_blocks][256] precomputed FWHT(decode(E8+tile+QJL)) */
    uint64_t **signs;          /* [n_blocks][4] sign masks (256-bit each) */
    bool      sli_prepared;    /* true after fpqx_sli_prepare */
    bool      fwht_on_z;       /* true if FWHT applied to z (optimized) */
    
    /* Memory management */
    void     *data_buffer;     /* Original mmap/malloc buffer */
    size_t    data_size;       /* Size in bytes */
    
} FPQTensor;

/* ═══════════════════════════════════════════════════════════════════
 * Core Utilities
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * Load FPQ v12 model from .fpq file
 * Returns array of tensors (caller frees with fpqx_free_all)
 */
int fpqx_load(const char *path, FPQTensor ***tensors, int *n_tensors);

/*
 * Prepare tensor for SLI (once at load time)
 * - Decodes E8 + tile + QJL to z
 * - Applies FWHT to z (FWHT-on-z optimization)
 * - Stores z in-place, discarding E8 region (irreversible)
 */
void fpqx_sli_prepare(FPQTensor *w);
void fpqx_sli_prepare_all(FPQTensor **tensors, int n_tensors);

/*
 * Free single tensor or array
 */
void fpqx_free(FPQTensor *w);
void fpqx_free_all(FPQTensor **tensors, int n_tensors);

/*
 * Encode FP32 tensor to FPQx (for testing/validation)
 */
int fpqx_encode_fp32(const float *data, int rows, int cols, FPQTensor **result);

/*
 * Decode FPQx to FP32 (for validation only, not needed for inference)
 * NOTE: Fails if SLI prepared (E8 region overwritten by z)
 */
int fpqx_decode_to_fp32(const FPQTensor *w, float **data);

/* ═══════════════════════════════════════════════════════════════════
 * Operator A — Addition in E8 Lattice Space
 *
 * Add two FPQx tensors directly in compressed domain.
 * E8 lattice is closed under addition (Voronoi cell property).
 *
 * Use cases:
 *   - Residual connections: y = x + residual
 *   - Ensemble averaging: w_avg = (w1 + w2 + w3) / 3
 *   - Model merging: w' = α*w_base + (1-α)*w_finetuned
 *
 * Quality: Exact if sum stays in E8 bounds, near-lossless otherwise
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_add(const FPQTensor *a, const FPQTensor *b, FPQTensor **result);

/* ═══════════════════════════════════════════════════════════════════
 * Operator M — Multiplication by Scalar
 *
 * Scale FPQx tensor by constant factor.
 * Only scales change, E8 coords and tiles unchanged.
 *
 * Use cases:
 *   - Learning rate: w' = w - lr * grad
 *   - Weight decay: w' = (1 - decay) * w
 *   - Gradient scaling: grad' = scale * grad
 *
 * Quality: Exact (no quantization error)
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_scale(const FPQTensor *a, float scalar, FPQTensor **result);

/* ═══════════════════════════════════════════════════════════════════
 * Operator Π — Projection to Subspace
 *
 * Project to lower-dimensional spectral subspace.
 * FPQx tensors are in FWHT domain → projection = truncate high frequencies.
 *
 * Use cases:
 *   - Dimensionality reduction (low-pass filter)
 *   - Model pruning (drop high-frequency noise)
 *   - Feature selection (keep top-k spectral components)
 *
 * Quality: dim=256→exact, dim=128→0.999+, dim=64→0.99+ cosine
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_project(const FPQTensor *a, int target_dim, FPQTensor **result);

/* ═══════════════════════════════════════════════════════════════════
 * Operator D — Dot Product (SLI - Spectral Lattice Inference)
 *
 * Compute W @ x WITHOUT decompressing W.
 * Core equation: y = z^T @ FWHT(signs ⊙ x)
 * where z = FWHT(decode(E8, tile, QJL)) precomputed once.
 *
 * Performance: 4.4× bandwidth reduction, 2.5× faster, cosine 0.9999+
 *
 * Use cases:
 *   - ALL neural network forward passes
 *   - Semantic search (query @ document_embeddings)
 *   - Classification (input @ weight_matrix)
 *
 * Requirements: Must call fpqx_sli_prepare(w) once before use
 * ═══════════════════════════════════════════════════════════════════ */

float fpqx_sli_dot(const FPQTensor *w, const float *x, int n);

/*
 * Batched version: computes multiple dot products in parallel
 * y[i] = w @ x[i] for i = 0..batch_size-1
 */
void fpqx_sli_dot_batch(const FPQTensor *w, const float *X, 
                        float *y, int n, int batch_size);

/* ═══════════════════════════════════════════════════════════════════
 * Operator Λ — Low-Rank Decomposition
 *
 * Extract Ghost head (rank-1 to rank-15) without full decompression.
 * FPQ v9 stores low-rank structure in header.
 *
 * Use cases:
 *   - LoRA fine-tuning: W' = W + α*(B @ A)
 *   - Transfer learning: extract task-agnostic features
 *   - Memory-efficient adaptation: store only LR deltas
 *
 * Quality: rank=1 captures 0.4-1.9%, rank=15 captures 5-20% of error energy
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_extract_lr(const FPQTensor *w, int rank, 
                    float **U, float **V, float **sigma);

/* ═══════════════════════════════════════════════════════════════════
 * Operator H — Hadamard Transform (FWHT)
 *
 * Apply FWHT to FPQx tensor (or recognize it's already applied).
 * FPQx tensors are in FWHT domain by design. FWHT² = I (involution).
 *
 * Use cases:
 *   - Spectral analysis (switch between weight and frequency domains)
 *   - Frequency filtering
 *   - Signal processing
 *
 * Quality: Exact (orthogonal transform, no information loss)
 * ═══════════════════════════════════════════════════════════════════ */

int fpqx_fwht(const FPQTensor *a, FPQTensor **result);

/* ═══════════════════════════════════════════════════════════════════
 * Operator I — Inference (Full Layer)
 *
 * Compute entire linear layer output via batched SLI.
 * y = W @ x where W is [out_dim × in_dim] FPQx matrix.
 *
 * Optimized: Pre-apply FWHT to input blocks, amortize across outputs.
 *
 * Use cases:
 *   - Transformer attention (Q, K, V projections)
 *   - MLP layers (FFN up/down)
 *   - Classification heads
 *
 * Performance: 2.5× faster than dense matmul for m,n ≥ 512
 * ═══════════════════════════════════════════════════════════════════ */

void fpqx_linear_layer(const FPQTensor *W, const float *x, float *y,
                       int out_dim, int in_dim);

/*
 * Optimized version with pre-FWHTed input (amortizes FWHT cost)
 * Caller must pre-compute x_fwht = FWHT(x) in blocks of 256
 */
void fpqx_linear_layer_optimized(const FPQTensor *W, const float *x_fwht,
                                  float *y, int out_dim, int in_dim);

/* ═══════════════════════════════════════════════════════════════════
 * Validation and Diagnostics
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * Measure reconstruction quality
 * Returns cosine similarity between FPQx tensor and FP32 reference
 */
float fpqx_cosine_similarity(const FPQTensor *a, const float *b_fp32);

/*
 * Benchmark SLI performance
 * Runs n_queries random dot products, reports tok/s and bandwidth
 */
void fpqx_benchmark_sli(const FPQTensor *W, int n_queries);

/*
 * Print tensor metadata
 */
void fpqx_print_info(const FPQTensor *w);

/* ═══════════════════════════════════════════════════════════════════
 * SIMD Control (platform-specific optimizations)
 * ═══════════════════════════════════════════════════════════════════ */

void fpqx_set_simd(bool enable_avx2, bool enable_neon);

/*
 * Get current SIMD status
 */
void fpqx_get_simd_status(bool *avx2_enabled, bool *neon_enabled);

#ifdef __cplusplus
}
#endif

#endif /* BONFYRE_FPQX_ALGEBRA_H */
