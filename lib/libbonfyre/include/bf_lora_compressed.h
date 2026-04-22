/*
 * bf_lora_compressed.h — Compressed-domain LoRA fine-tuning adapter
 *
 * Stores LoRA A and B matrices in E8-quantized block format (BQFP).
 * Forward pass: decode A block → intermediates → encode to E8 → decode B →
 *               add to residual.
 * Backward pass: compute float gradient patch, snap to E8 via bf_e8_batch_simd,
 *                update adapter weights in compressed domain.
 *
 * This avoids full FP32 weight expansion: blocks are decoded only for the 8
 * dimensions active in the current minibatch, keeping peak memory linear in
 * rank rather than quadratic in model size.
 *
 * BQFP block format (48 bytes per 8 dimensions):
 *   [0]   uint32  magic = 0x42514650 ('BQFP')
 *   [4]   float   scale            (dequantization scalar)
 *   [8]   float   values[8]        (E8 lattice point, pre-scaled)
 *   [40]  uint32  family_id        (which weight group this belongs to)
 *   [44]  uint32  reserved
 *
 * File format (adapter on disk):
 *   magic 4B "LORA"
 *   version uint32
 *   rank uint32
 *   alpha float
 *   rows_A uint32  (= model_dim)
 *   cols_A uint32  (= rank)
 *   rows_B uint32  (= rank)
 *   cols_B uint32  (= out_dim)
 *   n_blocks_A uint32
 *   n_blocks_B uint32
 *   BfBqfpBlock[n_blocks_A]  (A matrix)
 *   BfBqfpBlock[n_blocks_B]  (B matrix)
 */
#pragma once
#ifndef BF_LORA_COMPRESSED_H
#define BF_LORA_COMPRESSED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── BQFP block ───────────────────────────────────────────────────────────── */

#define BF_BQFP_MAGIC    0x42514650u  /* 'BQFP' */
#define BF_BQFP_DIM      8            /* E8 lives in 8 dimensions */

typedef struct {
    uint32_t magic;
    float    scale;
    float    values[BF_BQFP_DIM];
    uint32_t family_id;
    uint32_t reserved;
} BfBqfpBlock;                        /* 48 bytes */

/* Decode a block to float[8]: out[i] = block->values[i] * block->scale */
static inline void bf_bqfp_decode(const BfBqfpBlock *block, float out[BF_BQFP_DIM]) {
    for (int i = 0; i < BF_BQFP_DIM; i++)
        out[i] = block->values[i] * block->scale;
}

/* Encode float[8] to a block: snap to nearest E8 point, store scale + values. */
int bf_bqfp_encode(const float in[BF_BQFP_DIM], uint32_t family_id,
                   BfBqfpBlock *out);

/* ── LoRA adapter ─────────────────────────────────────────────────────────── */

#define BF_LORA_MAGIC   "LORA"
#define BF_LORA_VERSION 1u

typedef struct {
    int         rank;          /* LoRA rank r                              */
    float       alpha;         /* scaling: effective_lr = alpha / rank     */
    int         model_dim;     /* input dimension (rows of A)              */
    int         out_dim;       /* output dimension (cols of B)             */
    int         n_blocks_A;    /* ceil(model_dim * rank / 8)               */
    int         n_blocks_B;    /* ceil(rank * out_dim / 8)                  */
    BfBqfpBlock *A;            /* heap-allocated n_blocks_A blocks          */
    BfBqfpBlock *B;            /* heap-allocated n_blocks_B blocks          */
} BfLoraAdapter;

/* ── API ──────────────────────────────────────────────────────────────────── */

/*
 * Allocate a LoRA adapter with zero-initialised A, unit-scaled B.
 *   model_dim  — input dimension
 *   rank       — LoRA rank (typically 4..64)
 *   out_dim    — output dimension
 *   alpha      — scaling factor (commonly equal to rank)
 * Returns 0 on success, -1 on alloc failure.
 */
int bf_lora_init(BfLoraAdapter *adapter, int model_dim, int rank, int out_dim,
                 float alpha);

/*
 * Forward pass:  out += (B * A * x) * (alpha / rank)
 *
 * x   — float[model_dim]
 * out — float[out_dim] (accumulate, not overwrite)
 * n   — batch size (process n independent x vectors sequentially)
 *
 * Blocks are decoded on-the-fly; no persistent FP32 expansion.
 */
void bf_lora_forward(const BfLoraAdapter *adapter,
                     const float *x, float *out, int n);

/*
 * Backward pass (SGD step):
 *   grad_out — float[out_dim * n]  (upstream gradient)
 *   x        — float[model_dim * n] (input that was used in forward)
 *   lr       — learning rate
 *   n        — batch size
 *
 * Updates A and B in-place via E8-snapped gradient step.
 * GPU-free; all arithmetic in float32 patch windows of size 8.
 */
void bf_lora_backward(BfLoraAdapter    *adapter,
                       const float      *x,
                       const float      *grad_out,
                       float             lr,
                       int               n);

/*
 * Save adapter to a binary file.
 * Returns 0 on success, -1 on error.
 */
int bf_lora_save(const BfLoraAdapter *adapter, const char *path);

/*
 * Load adapter from a binary file written by bf_lora_save.
 * Returns 0 on success, -1 on error (bad magic, truncated file, etc.).
 */
int bf_lora_load(BfLoraAdapter *adapter, const char *path);

/*
 * Free heap memory allocated by bf_lora_init / bf_lora_load.
 */
void bf_lora_free(BfLoraAdapter *adapter);

#ifdef __cplusplus
}
#endif
#endif /* BF_LORA_COMPRESSED_H */
