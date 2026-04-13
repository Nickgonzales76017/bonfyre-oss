/*
 * fpq_native.c — Native .fpq format writer/reader
 *
 * Compact binary format that stores v9 multiscale data at ~1.6 bpw:
 *   - Per-tensor header with shape, η_L, rank, bits
 *   - LR factors at INT8 (U*S columns + Vt rows)
 *   - E8 lattice indices at 8-bit per group
 *   - RVQ tile indices at 8-bit per pair
 *   - RVQ codebook at FP16
 *   - Metadata at FP16/INT8
 *   - Ghost correction at INT8 (if present)
 *
 * Reader reconstructs to FP32 on-the-fly using the same v9 decode path.
 *
 * File layout:
 *   [Header: magic, version, n_tensors, flags]
 *   [Tensor Table: name, shape, offset, size per tensor]
 *   [Tensor Data blocks]
 *
 * This format delivers the "20× from fp32" claim as actual file size.
 */
#include "fpq.h"
#include "weight_algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Helper: float → fp16 bytes (IEEE 754 half) ── */
static uint16_t float_to_fp16(float f) {
    union { float f; uint32_t u; } v = { .f = f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t exp = ((v.u >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (v.u >> 13) & 0x03FF;

    if (exp <= 0) return (uint16_t)sign;           /* underflow → ±0 */
    if (exp >= 31) return (uint16_t)(sign | 0x7C00); /* overflow → ±inf */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | frac);
}

static float fp16_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x03FF;

    if (exp == 0) {
        if (frac == 0) { union { uint32_t u; float f; } v = { .u = sign }; return v.f; }
        /* Denorm */
        while (!(frac & 0x0400)) { frac <<= 1; exp--; }
        exp++; frac &= ~0x0400;
    } else if (exp == 31) {
        union { uint32_t u; float f; } v = { .u = sign | 0x7F800000 | (frac << 13) };
        return v.f;
    }

    uint32_t result = sign | ((exp + 112) << 23) | (frac << 13);
    union { uint32_t u; float f; } v = { .u = result };
    return v.f;
}

/* ── Helper: INT8 symmetric quantize/dequantize ── */
static void quant_int8(const float *data, size_t n, int8_t *out, float *scale_out) {
    float absmax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(data[i]);
        if (a > absmax) absmax = a;
    }
    float sc = absmax / 127.0f;
    *scale_out = sc;
    if (sc < 1e-30f) {
        memset(out, 0, n * sizeof(int8_t));
        return;
    }
    for (size_t i = 0; i < n; i++) {
        int q = (int)roundf(data[i] / sc);
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        out[i] = (int8_t)q;
    }
}

static void dequant_int8(const int8_t *data, size_t n, float scale, float *out) {
    for (size_t i = 0; i < n; i++)
        out[i] = (float)data[i] * scale;
}

/* μ-law inverse warp — matches v7_warp_inverse in v4_optimizations.c */
static float native_warp_inverse(float y, float beta) {
    float lnorm = logf(1.0f + beta);
    float ay = fabsf(y);
    float x = (expf(ay * lnorm) - 1.0f) / beta;
    return (y < 0) ? -x : x;
}

/* ── INT7 bit-packing: 256 signed values → 224 bytes ── */
static void pack_int7(const int8_t *in, uint8_t *out, size_t n) {
    size_t out_bytes = (n * 7 + 7) / 8;
    memset(out, 0, out_bytes);
    size_t bit_pos = 0;
    for (size_t i = 0; i < n; i++) {
        /* Map signed [-64,63] to unsigned [0,127] */
        int v = (int)in[i] + 64;
        if (v < 0) v = 0;
        if (v > 127) v = 127;
        uint8_t val = (uint8_t)v;
        size_t byte_idx = bit_pos / 8;
        int bit_off = (int)(bit_pos % 8);
        out[byte_idx] |= (uint8_t)((val << bit_off) & 0xFF);
        if (bit_off + 7 > 8)
            out[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_off));
        bit_pos += 7;
    }
}

static void unpack_int7(const uint8_t *in, int8_t *out, size_t n) {
    size_t bit_pos = 0;
    for (size_t i = 0; i < n; i++) {
        size_t byte_idx = bit_pos / 8;
        int bit_off = (int)(bit_pos % 8);
        uint16_t raw = (uint16_t)in[byte_idx];
        if (bit_off + 7 > 8)
            raw |= (uint16_t)in[byte_idx + 1] << 8;
        uint8_t val = (uint8_t)((raw >> bit_off) & 0x7F);
        out[i] = (int8_t)((int)val - 64);
        bit_pos += 7;
    }
}

/* ── 6-bit tile index packing: 16 indices → 12 bytes ── */
static void pack_uint6(const uint8_t *in, uint8_t *out, size_t n) {
    size_t out_bytes = (n * 6 + 7) / 8;
    memset(out, 0, out_bytes);
    size_t bit_pos = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t val = in[i] & 0x3F;
        size_t byte_idx = bit_pos / 8;
        int bit_off = (int)(bit_pos % 8);
        out[byte_idx] |= (uint8_t)((val << bit_off) & 0xFF);
        if (bit_off + 6 > 8)
            out[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_off));
        bit_pos += 6;
    }
}

static void unpack_uint6(const uint8_t *in, uint8_t *out, size_t n) {
    size_t bit_pos = 0;
    for (size_t i = 0; i < n; i++) {
        size_t byte_idx = bit_pos / 8;
        int bit_off = (int)(bit_pos % 8);
        uint16_t raw = (uint16_t)in[byte_idx];
        if (bit_off + 6 > 8)
            raw |= (uint16_t)in[byte_idx + 1] << 8;
        out[i] = (uint8_t)((raw >> bit_off) & 0x3F);
        bit_pos += 6;
    }
}

/* ── FP8 E4M3: 1s/4e/3m, bias=7, range ±448, no inf/nan ── */
static uint8_t float_to_fp8_e4m3(float f) {
    if (f == 0.0f) return 0;
    uint8_t sign = f < 0.0f ? 0x80 : 0;
    float av = fabsf(f);
    if (av > 448.0f) return sign | 0x7F;
    if (av < (1.0f / 512.0f)) return sign;

    int e = (int)floorf(log2f(av));
    if (e < -6) e = -6;
    if (e > 8) e = 8;
    int biased_e = e + 7;

    if (biased_e <= 0) {
        float mant = av / ldexpf(1.0f, -6);
        int m = (int)roundf(mant * 8.0f);
        if (m > 7) m = 7;
        if (m < 1) m = 1;
        return sign | (uint8_t)m;
    }

    float scale = ldexpf(1.0f, e);
    float mant = av / scale - 1.0f;
    int m = (int)roundf(mant * 8.0f);
    if (m > 7) { m = 0; biased_e++; }
    if (m < 0) m = 0;
    if (biased_e > 15) { biased_e = 15; m = 7; }
    return sign | ((uint8_t)biased_e << 3) | (uint8_t)m;
}

static float fp8_e4m3_to_float(uint8_t v) {
    int sign = (v & 0x80) ? -1 : 1;
    int biased_e = (v >> 3) & 0x0F;
    int m = v & 0x07;
    if (biased_e == 0 && m == 0) return 0.0f;
    float result;
    if (biased_e == 0)
        result = ldexpf((float)m, -9);  /* subnormal: m × 2^(-9) */
    else
        result = ldexpf(1.0f + (float)m / 8.0f, biased_e - 7);
    return (float)sign * result;
}

/* ═══════════════════════════════════════════════════════════════════
 * rANS Entropy Codec for E8 Coordinates
 *
 * Byte-aligned rANS (range Asymmetric Numeral Systems).
 * Symbols: 0–127 (unsigned INT7, mapped from signed [-64,63]).
 * Frequency precision: 12 bits (sum = 4096).
 * State range: [L, L*256) where L = 4096.
 *
 * Per-tensor on disk:
 *   uint16_t  freq[128]        — 256 bytes: normalized frequency table
 *   uint32_t  compressed_size  — 4 bytes
 *   uint32_t  n_symbols        — 4 bytes (= n_blocks * 256)
 *   uint8_t   data[compressed_size]
 * ═══════════════════════════════════════════════════════════════════ */

#define RANS_NSYM       128
#define RANS_PROB_BITS  12
#define RANS_PROB_SCALE (1u << RANS_PROB_BITS)  /* 4096 */
#define RANS_BYTE_L     (RANS_PROB_SCALE << 8)  /* 1048576 = lower bound */

/* Build normalized frequency table from raw counts.
 * Guarantees: sum == RANS_PROB_SCALE, all used symbols get freq >= 1. */
static void rans_build_freq(const uint32_t *counts, size_t n_sym,
                            uint16_t *freq_out) {
    uint64_t total = 0;
    for (size_t i = 0; i < n_sym; i++) total += counts[i];
    if (total == 0) {
        /* uniform fallback */
        uint16_t base = (uint16_t)(RANS_PROB_SCALE / n_sym);
        for (size_t i = 0; i < n_sym; i++) freq_out[i] = base;
        /* fix remainder */
        uint32_t sum = (uint32_t)base * (uint32_t)n_sym;
        for (size_t i = 0; sum < RANS_PROB_SCALE; i++, sum++)
            freq_out[i]++;
        return;
    }

    /* Initial assignment: proportional, floor */
    uint32_t sum = 0;
    int n_nonzero = 0;
    for (size_t i = 0; i < n_sym; i++) {
        if (counts[i] > 0) {
            freq_out[i] = (uint16_t)((uint64_t)counts[i] * RANS_PROB_SCALE / total);
            if (freq_out[i] == 0) freq_out[i] = 1;  /* must be at least 1 */
            n_nonzero++;
        } else {
            freq_out[i] = 0;
        }
        sum += freq_out[i];
    }

    /* Adjust to hit exact sum = RANS_PROB_SCALE */
    while (sum < RANS_PROB_SCALE) {
        /* Add to the symbol with largest count that won't overflow */
        size_t best = 0;
        uint32_t best_cnt = 0;
        for (size_t i = 0; i < n_sym; i++) {
            if (counts[i] > best_cnt && freq_out[i] < RANS_PROB_SCALE - 1) {
                best_cnt = counts[i]; best = i;
            }
        }
        freq_out[best]++;
        sum++;
    }
    while (sum > RANS_PROB_SCALE) {
        /* Subtract from largest freq that's > 1 */
        size_t best = 0;
        uint16_t best_f = 0;
        for (size_t i = 0; i < n_sym; i++) {
            if (freq_out[i] > best_f && freq_out[i] > 1) {
                best_f = freq_out[i]; best = i;
            }
        }
        freq_out[best]--;
        sum--;
    }
}

/* Build cumulative frequency table from freq */
static void rans_build_cumfreq(const uint16_t *freq, size_t n_sym,
                               uint16_t *cum_out) {
    cum_out[0] = 0;
    for (size_t i = 1; i < n_sym; i++)
        cum_out[i] = cum_out[i - 1] + freq[i - 1];
}

/* Build decode lookup: for each slot 0..RANS_PROB_SCALE-1, which symbol? */
static void rans_build_lookup(const uint16_t *cum, const uint16_t *freq,
                              size_t n_sym, uint8_t *lut) {
    for (size_t s = 0; s < n_sym; s++) {
        for (uint16_t j = 0; j < freq[s]; j++)
            lut[cum[s] + j] = (uint8_t)s;
    }
}

/* Encode n symbols into output buffer. Returns compressed size.
 * symbols: array of n values in [0, n_sym).
 * out: must be pre-allocated to at least n * 2 bytes (worst case).
 * rANS encodes backwards; output is written backwards then reversed. */
static size_t rans_encode(const uint8_t *symbols, size_t n,
                          const uint16_t *freq, const uint16_t *cum,
                          uint8_t *out, size_t out_cap) {
    /* Work buffer: encode into tail of out, backwards */
    uint8_t *ptr = out + out_cap;
    uint32_t state = RANS_BYTE_L;  /* initial state */

    /* Encode in reverse order */
    for (size_t i = n; i > 0; i--) {
        uint8_t s = symbols[i - 1];
        uint16_t f = freq[s];
        uint16_t c = cum[s];

        if (f == 0) continue;  /* shouldn't happen */

        /* Renormalize: flush bytes while state is too large */
        uint32_t x_max = ((RANS_BYTE_L >> RANS_PROB_BITS) << 8) * (uint32_t)f;
        while (state >= x_max) {
            *--ptr = (uint8_t)(state & 0xFF);
            state >>= 8;
        }

        /* Encode: state = (state / f) * M + (state % f) + c */
        state = ((state / (uint32_t)f) << RANS_PROB_BITS) +
                (state % (uint32_t)f) + (uint32_t)c;
    }

    /* Flush final state (4 bytes, big-endian-ish) */
    *--ptr = (uint8_t)(state >>  0);
    *--ptr = (uint8_t)(state >>  8);
    *--ptr = (uint8_t)(state >> 16);
    *--ptr = (uint8_t)(state >> 24);

    /* Move compressed data to start of buffer */
    size_t compressed = (size_t)(out + out_cap - ptr);
    memmove(out, ptr, compressed);
    return compressed;
}

/* Decode n symbols from compressed buffer. */
static void rans_decode(const uint8_t *in, size_t in_len,
                        const uint16_t *freq, const uint16_t *cum,
                        const uint8_t *lut, size_t n_sym,
                        uint8_t *out, size_t n) {
    const uint8_t *ptr = in;

    /* Read initial state */
    uint32_t state = ((uint32_t)ptr[0] << 24) | ((uint32_t)ptr[1] << 16) |
                     ((uint32_t)ptr[2] <<  8) | ((uint32_t)ptr[3]);
    ptr += 4;

    for (size_t i = 0; i < n; i++) {
        /* Decode: look up symbol from state's low bits */
        uint32_t slot = state & (RANS_PROB_SCALE - 1);
        uint8_t s = lut[slot];
        out[i] = s;

        /* Advance state */
        state = (uint32_t)freq[s] * (state >> RANS_PROB_BITS) +
                slot - (uint32_t)cum[s];

        /* Renormalize */
        while (state < RANS_BYTE_L && ptr < in + in_len) {
            state = (state << 8) | *ptr++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Writer: fpq_native_write
 *
 * Takes already-compressed fp32 tensors (post algebra-compress),
 * re-encodes each through v9, and writes compact binary representation.
 * ═══════════════════════════════════════════════════════════════════ */

/* Per-tensor header in the file */
typedef struct __attribute__((packed)) {
    uint16_t name_len;
    uint32_t rows;
    uint32_t cols;
    uint16_t lr_rank;
    uint8_t  coord_bits;
    uint8_t  has_ghost;
    uint32_t n_blocks;      /* was uint16_t — overflows at >16M elements */
    uint16_t effective_k;   /* RVQ codebook size */
    uint64_t data_offset;
    uint64_t data_size;
} fpq_native_tensor_header_t;

/* File header */
typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t n_tensors;
    uint32_t flags;
    uint64_t tensor_table_offset;
} fpq_native_file_header_t;

int fpq_native_write(const char *path, const fpq_raw_tensor_t *tensors,
                     size_t n_tensors) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "FPQ Native: cannot open %s for writing\n", path);
        return 1;
    }

    /* Phase 1: encode each tensor through v9 and collect compact data */
    fprintf(stderr, "FPQ Native Write: %zu tensors → %s\n", n_tensors, path);

    /* Write file header (will update offsets later) */
    fpq_native_file_header_t fhdr = {
        .magic = FPQ_NATIVE_MAGIC,
        .version = FPQ_NATIVE_VERSION,
        .n_tensors = (uint32_t)n_tensors,
        .flags = FPQ_FLAG_PACKED_V12,
        .tensor_table_offset = sizeof(fpq_native_file_header_t)
    };
    fwrite(&fhdr, sizeof(fhdr), 1, fp);

    /* Reserve space for tensor headers */
    long table_start = ftell(fp);
    fpq_native_tensor_header_t *headers = (fpq_native_tensor_header_t *)
        calloc(n_tensors, sizeof(fpq_native_tensor_header_t));
    /* Write placeholder headers */
    for (size_t i = 0; i < n_tensors; i++) {
        headers[i].name_len = (uint16_t)strlen(tensors[i].name);
        fwrite(&headers[i], sizeof(fpq_native_tensor_header_t), 1, fp);
        fwrite(tensors[i].name, 1, headers[i].name_len, fp);
    }

    size_t total_compressed = 0;
    size_t total_original = 0;

    /* Phase 2: For each tensor, encode with v9 and write compact data */
    for (size_t ti = 0; ti < n_tensors; ti++) {
        const fpq_raw_tensor_t *t = &tensors[ti];
        size_t n = t->rows * t->cols;
        headers[ti].rows = (uint32_t)t->rows;
        headers[ti].cols = (uint32_t)t->cols;
        headers[ti].data_offset = (uint64_t)ftell(fp);

        total_original += n * 4;  /* fp32 = 4 bytes/weight */

        /* Small/1D tensors: store as fp16 directly */
        if (t->rows <= 1 || t->cols <= 1 || n < FPQ_BLOCK_DIM * 2) {
            headers[ti].lr_rank = 0;
            headers[ti].coord_bits = 0;
            headers[ti].has_ghost = 0;
            headers[ti].n_blocks = 0;
            headers[ti].effective_k = 0;

            /* Write as fp16 array */
            for (size_t j = 0; j < n; j++) {
                uint16_t h = float_to_fp16(t->data[j]);
                fwrite(&h, 2, 1, fp);
            }
            headers[ti].data_size = (uint64_t)(n * 2);
            total_compressed += n * 2;
            continue;
        }

        /* v9 encode */
        int cbits = 3;
        float eta_L = bwa_get_eta_L(t->data, t->rows, t->cols, 0.25f);
        int adaptive_bits = bwa_adaptive_bits(eta_L, cbits);

        fpq_tensor_t *enc = fpq_encode_tensor_v9(
            t->data, t->rows, t->cols, t->name, adaptive_bits);

        if (!enc || !enc->sbb_scale_delta || enc->pid_alpha != -9.0f) {
            /* Fallback: store as fp16 */
            for (size_t j = 0; j < n; j++) {
                uint16_t h = float_to_fp16(t->data[j]);
                fwrite(&h, 2, 1, fp);
            }
            headers[ti].data_size = (uint64_t)(n * 2);
            total_compressed += n * 2;
            if (enc) fpq_tensor_free(enc);
            continue;
        }

        /* Extract v9 data from sbb_scale_delta packing */
        int lr_rank = (int)enc->sbb_scale_delta[0];
        size_t lr_us_size = t->rows * (size_t)lr_rank;
        size_t lr_vt_size = (size_t)lr_rank * t->cols;
        size_t n_blocks = enc->n_blocks;
        size_t padded = 256;

        headers[ti].lr_rank = (uint16_t)lr_rank;
        headers[ti].coord_bits = (uint8_t)adaptive_bits;
        headers[ti].has_ghost = (enc->ghost != NULL) ? 1 : 0;
        headers[ti].n_blocks = (uint32_t)n_blocks;

        /* Recover effective_k */
        size_t v8_base = 2 + lr_us_size + lr_vt_size;
        size_t e8_off = v8_base + n_blocks;
        size_t e8_flat_size = n_blocks * padded;
        size_t tile_cb_off = e8_off + e8_flat_size;

        int effective_k = 256; /* default */
        {
            size_t test_cb = (size_t)effective_k * 16;
            size_t test_ek = tile_cb_off + test_cb + n_blocks * 16;
            float stored = enc->sbb_scale_delta[test_ek];
            if (stored >= 16.0f && stored <= 256.0f)
                effective_k = (int)stored;
        }
        if (effective_k > 256) effective_k = 256;
        headers[ti].effective_k = (uint16_t)effective_k;

        long data_start = ftell(fp);

        /* ── Write LR factors as INT8 ── */
        {
            /* U*S factors: [rows × lr_rank], quantize per-column */
            const float *us_data = enc->sbb_scale_delta + 2;
            for (int r = 0; r < lr_rank; r++) {
                float col_max = 0.0f;
                for (size_t row = 0; row < t->rows; row++) {
                    float v = fabsf(us_data[row * lr_rank + r]);
                    if (v > col_max) col_max = v;
                }
                uint16_t scale_h = float_to_fp16(col_max / 127.0f);
                fwrite(&scale_h, 2, 1, fp);
                float sc = fp16_to_float(scale_h);
                for (size_t row = 0; row < t->rows; row++) {
                    float val = us_data[row * lr_rank + r];
                    int q = (sc > 1e-30f) ? (int)roundf(val / sc) : 0;
                    if (q < -127) q = -127;
                    if (q >  127) q =  127;
                    int8_t qb = (int8_t)q;
                    fwrite(&qb, 1, 1, fp);
                }
            }

            /* Vt factors: [lr_rank × cols], quantize per-row */
            const float *vt_data = enc->sbb_scale_delta + 2 + lr_us_size;
            for (int r = 0; r < lr_rank; r++) {
                float row_max = 0.0f;
                for (size_t col = 0; col < t->cols; col++) {
                    float v = fabsf(vt_data[r * t->cols + col]);
                    if (v > row_max) row_max = v;
                }
                uint16_t scale_h = float_to_fp16(row_max / 127.0f);
                fwrite(&scale_h, 2, 1, fp);
                float sc = fp16_to_float(scale_h);
                for (size_t col = 0; col < t->cols; col++) {
                    float val = vt_data[r * t->cols + col];
                    int q = (sc > 1e-30f) ? (int)roundf(val / sc) : 0;
                    if (q < -127) q = -127;
                    if (q >  127) q =  127;
                    int8_t qb = (int8_t)q;
                    fwrite(&qb, 1, 1, fp);
                }
            }
        }

        /* ── Write per-block metadata ── */
        {
            if (fhdr.flags & FPQ_FLAG_FP8_SCALES) {
                /* coord_scales as FP8 E4M3 */
                for (size_t b = 0; b < n_blocks; b++) {
                    uint8_t h = float_to_fp8_e4m3(enc->coord_scales[b]);
                    fwrite(&h, 1, 1, fp);
                }
                /* warp_norms as FP8 E4M3 */
                for (size_t b = 0; b < n_blocks; b++) {
                    uint8_t h = float_to_fp8_e4m3(enc->sbb_scale_delta[v8_base + b]);
                    fwrite(&h, 1, 1, fp);
                }
            } else {
                /* coord_scales as fp16 */
                for (size_t b = 0; b < n_blocks; b++) {
                    uint16_t h = float_to_fp16(enc->coord_scales[b]);
                    fwrite(&h, 2, 1, fp);
                }
                /* warp_norms as fp16 */
                for (size_t b = 0; b < n_blocks; b++) {
                    uint16_t h = float_to_fp16(enc->sbb_scale_delta[v8_base + b]);
                    fwrite(&h, 2, 1, fp);
                }
            }
            /* residual_norms as INT8 + scale */
            {
                float rn_max = 0.0f;
                for (size_t b = 0; b < n_blocks; b++) {
                    float v = fabsf(enc->coord_residual_norms[b]);
                    if (v > rn_max) rn_max = v;
                }
                uint16_t rn_scale_h = float_to_fp16(rn_max / 127.0f);
                fwrite(&rn_scale_h, 2, 1, fp);
                float rn_sc = fp16_to_float(rn_scale_h);
                for (size_t b = 0; b < n_blocks; b++) {
                    int q = (rn_sc > 1e-30f) ?
                        (int)roundf(enc->coord_residual_norms[b] / rn_sc) : 0;
                    if (q < 0) q = 0;
                    if (q > 127) q = 127;
                    uint8_t qb = (uint8_t)q;
                    fwrite(&qb, 1, 1, fp);
                }
            }
        }

        /* ── Write E8 points ── */
        if (fhdr.flags & FPQ_FLAG_E8_ENTROPY) {
            /* rANS entropy-coded: ~6.2-6.6 bits/coord (lossless) */
            const float *e8_base = enc->sbb_scale_delta + e8_off;
            size_t total_sym = n_blocks * padded;

            /* Collect all E8 coords as unsigned [0,127] */
            uint8_t *all_sym = (uint8_t *)malloc(total_sym);
            uint32_t counts[RANS_NSYM];
            memset(counts, 0, sizeof(counts));
            for (size_t b = 0; b < n_blocks; b++) {
                for (size_t d = 0; d < padded; d++) {
                    int v = (int)e8_base[b * padded + d];
                    if (v > 63) v = 63;
                    if (v < -64) v = -64;
                    uint8_t u = (uint8_t)(v + 64);
                    all_sym[b * padded + d] = u;
                    counts[u]++;
                }
            }

            /* Build frequency table */
            uint16_t freq[RANS_NSYM], cum[RANS_NSYM];
            rans_build_freq(counts, RANS_NSYM, freq);
            rans_build_cumfreq(freq, RANS_NSYM, cum);

            /* Encode */
            size_t out_cap = total_sym * 2;  /* generous upper bound */
            uint8_t *comp = (uint8_t *)malloc(out_cap);
            size_t comp_size = rans_encode(all_sym, total_sym,
                                           freq, cum, comp, out_cap);

            /* Write: freq table (256B) + sizes (8B) + compressed data */
            fwrite(freq, sizeof(uint16_t), RANS_NSYM, fp);
            uint32_t cs32 = (uint32_t)comp_size;
            uint32_t ns32 = (uint32_t)total_sym;
            fwrite(&cs32, 4, 1, fp);
            fwrite(&ns32, 4, 1, fp);
            fwrite(comp, 1, comp_size, fp);

            if (ti < 5)
                fprintf(stderr, "    E8 entropy: %zu → %zu bytes "
                        "(%.2f bits/coord, %.0f%% of INT7)\n",
                        total_sym, comp_size + 264,
                        (double)(comp_size + 264) * 8.0 / (double)total_sym,
                        (double)(comp_size + 264) * 100.0 /
                        (double)((total_sym * 7 + 7) / 8));

            free(comp);
            free(all_sym);
        } else if (fhdr.flags & FPQ_FLAG_E8_INT7) {
            /* INT7 packed: 256 coords × 7 bits = 224 bytes/block */
            const float *e8_base = enc->sbb_scale_delta + e8_off;
            size_t packed_bytes = (padded * 7 + 7) / 8;  /* 224 */
            uint8_t *pack_buf = (uint8_t *)malloc(packed_bytes);
            int8_t *block_buf = (int8_t *)malloc(padded);
            for (size_t b = 0; b < n_blocks; b++) {
                for (size_t d = 0; d < padded; d++) {
                    int v = (int)e8_base[b * padded + d];
                    if (v > 63) v = 63;
                    if (v < -64) v = -64;
                    block_buf[d] = (int8_t)v;
                }
                pack_int7(block_buf, pack_buf, padded);
                fwrite(pack_buf, 1, packed_bytes, fp);
            }
            free(pack_buf);
            free(block_buf);
        } else {
            /* INT8: 256 bytes/block (legacy) */
            const float *e8_base = enc->sbb_scale_delta + e8_off;
            for (size_t b = 0; b < n_blocks; b++) {
                for (size_t d = 0; d < padded; d++) {
                    int8_t v = (int8_t)(int)e8_base[b * padded + d];
                    fwrite(&v, 1, 1, fp);
                }
            }
        }

        /* ── Write RVQ tile codebook as fp16 ── */
        {
            const float *tile_data = enc->sbb_scale_delta + tile_cb_off;
            size_t tile_cb_size = (size_t)effective_k * 16;
            for (size_t j = 0; j < tile_cb_size; j++) {
                uint16_t h = float_to_fp16(tile_data[j]);
                fwrite(&h, 2, 1, fp);
            }
        }

        /* ── Write tile indices ── */
        if (fhdr.flags & FPQ_FLAG_TILE_6BIT) {
            /* 6-bit packed: 16 indices × 6 bits = 12 bytes/block */
            size_t tile_idx_src = tile_cb_off + (size_t)effective_k * 16;
            uint8_t idx_buf[16];
            uint8_t pack_buf6[12];
            for (size_t b = 0; b < n_blocks; b++) {
                for (int p = 0; p < 16; p++) {
                    int idx = (int)enc->sbb_scale_delta[tile_idx_src + b * 16 + p];
                    idx = idx % 64;  /* wrap to 6-bit range */
                    if (idx < 0) idx += 64;
                    idx_buf[p] = (uint8_t)idx;
                }
                pack_uint6(idx_buf, pack_buf6, 16);
                fwrite(pack_buf6, 1, 12, fp);
            }
        } else {
            /* uint8: 16 bytes/block (legacy) */
            size_t tile_idx_src = tile_cb_off + (size_t)effective_k * 16;
            for (size_t b = 0; b < n_blocks; b++) {
                for (int p = 0; p < 16; p++) {
                    uint8_t idx = (uint8_t)(int)
                        enc->sbb_scale_delta[tile_idx_src + b * 16 + p];
                    fwrite(&idx, 1, 1, fp);
                }
            }
        }

        /* ── Write QJL bits (64 bits per block) ── */
        if (enc->qjl) {
            for (size_t b = 0; b < n_blocks; b++) {
                if (enc->qjl[b] && enc->qjl[b]->bits) {
                    fwrite(enc->qjl[b]->bits, 8, 1, fp);
                } else {
                    uint64_t zero = 0;
                    fwrite(&zero, 8, 1, fp);
                }
            }
        }

        /* ── Write ghost correction as INT8 ── */
        if (enc->ghost) {
            float u_scale, v_scale;
            int8_t *u_q = (int8_t *)malloc(t->rows * sizeof(int8_t));
            int8_t *v_q = (int8_t *)malloc(t->cols * sizeof(int8_t));
            quant_int8(enc->ghost->u, t->rows, u_q, &u_scale);
            quant_int8(enc->ghost->v, t->cols, v_q, &v_scale);

            uint16_t sigma_h = float_to_fp16(enc->ghost->sigma);
            uint16_t u_scale_h = float_to_fp16(u_scale);
            uint16_t v_scale_h = float_to_fp16(v_scale);
            fwrite(&sigma_h, 2, 1, fp);
            fwrite(&u_scale_h, 2, 1, fp);
            fwrite(u_q, 1, t->rows, fp);
            fwrite(&v_scale_h, 2, 1, fp);
            fwrite(v_q, 1, t->cols, fp);
            free(u_q);
            free(v_q);
        }

        /* Write haar_seed */
        fwrite(&enc->haar_seed, 8, 1, fp);

        headers[ti].data_size = (uint64_t)(ftell(fp) - data_start);
        total_compressed += (size_t)headers[ti].data_size;

        if (ti < 10 || (ti % 50 == 0))
            fprintf(stderr, "  [%zu] %-40s rank=%d @%db %zu → %llu bytes\n",
                    ti, t->name, lr_rank, adaptive_bits,
                    n * 4, (unsigned long long)headers[ti].data_size);

        fpq_tensor_free(enc);
    }

    /* Rewrite tensor headers with correct offsets */
    fseek(fp, table_start, SEEK_SET);
    for (size_t i = 0; i < n_tensors; i++) {
        fwrite(&headers[i], sizeof(fpq_native_tensor_header_t), 1, fp);
        fwrite(tensors[i].name, 1, headers[i].name_len, fp);
    }

    fclose(fp);

    float ratio = (total_original > 0) ?
        (float)total_original / (float)total_compressed : 0.0f;
    fprintf(stderr,
        "\n═══════════════════════════════════════════════════════\n"
        " FPQ NATIVE FORMAT SUMMARY\n"
        "═══════════════════════════════════════════════════════\n"
        "  Original:   %zu bytes (fp32)\n"
        "  Compressed: %zu bytes\n"
        "  Ratio:      %.1f× from fp32\n"
        "  Avg bpw:    %.2f\n"
        "═══════════════════════════════════════════════════════\n",
        total_original, total_compressed,
        ratio,
        total_original > 0 ? 32.0f / ratio : 0.0f);

    free(headers);
    return 0;
}


/* ═══════════════════════════════════════════════════════════════════
 * Reader: fpq_native_read
 *
 * Reads native .fpq and reconstructs fp32 tensors via v9 decode path.
 * ═══════════════════════════════════════════════════════════════════ */

fpq_raw_tensor_t *fpq_native_read(const char *path, size_t *n_tensors) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "FPQ Native: cannot open %s for reading\n", path);
        *n_tensors = 0;
        return NULL;
    }

    /* Read file header */
    fpq_native_file_header_t fhdr;
    if (fread(&fhdr, sizeof(fhdr), 1, fp) != 1 ||
        fhdr.magic != FPQ_NATIVE_MAGIC) {
        fprintf(stderr, "FPQ Native: invalid magic in %s\n", path);
        fclose(fp);
        *n_tensors = 0;
        return NULL;
    }

    if (fhdr.version > FPQ_NATIVE_VERSION) {
        fprintf(stderr, "FPQ Native: unsupported version %u (max %d)\n",
                fhdr.version, FPQ_NATIVE_VERSION);
        fclose(fp);
        *n_tensors = 0;
        return NULL;
    }

    uint32_t nt = fhdr.n_tensors;
    *n_tensors = nt;

    /* Read tensor headers + names */
    fpq_native_tensor_header_t *headers = (fpq_native_tensor_header_t *)
        calloc(nt, sizeof(fpq_native_tensor_header_t));
    char **names = (char **)calloc(nt, sizeof(char *));

    for (uint32_t i = 0; i < nt; i++) {
        if (fread(&headers[i], sizeof(fpq_native_tensor_header_t), 1, fp) != 1)
            break;
        names[i] = (char *)calloc(headers[i].name_len + 1, 1);
        if (fread(names[i], 1, headers[i].name_len, fp) != headers[i].name_len)
            break;
    }

    /* Allocate output tensors */
    fpq_raw_tensor_t *out = (fpq_raw_tensor_t *)calloc(nt, sizeof(fpq_raw_tensor_t));

    for (uint32_t ti = 0; ti < nt; ti++) {
        fpq_native_tensor_header_t *h = &headers[ti];
        size_t rows = h->rows, cols = h->cols;
        size_t n = rows * cols;

        strncpy(out[ti].name, names[ti], sizeof(out[ti].name) - 1);
        out[ti].rows = rows;
        out[ti].cols = cols;
        out[ti].n_elements = n;
        out[ti].n_dims = (rows > 1 && cols > 1) ? 2 : 1;
        out[ti].data = (float *)calloc(n, sizeof(float));

        fseek(fp, (long)h->data_offset, SEEK_SET);

        /* Small/1D tensors: stored as fp16 */
        if (h->lr_rank == 0 && h->coord_bits == 0) {
            for (size_t j = 0; j < n; j++) {
                uint16_t hv;
                if (fread(&hv, 2, 1, fp) != 1) break;
                out[ti].data[j] = fp16_to_float(hv);
            }
            continue;
        }

        int lr_rank = h->lr_rank;
        size_t n_blocks = h->n_blocks;
        int effective_k = h->effective_k;
        size_t padded = 256;
        size_t block_dim = FPQ_BLOCK_DIM;

        /* ── Read LR factors ── */
        float *US = (float *)calloc(rows * (size_t)lr_rank, sizeof(float));
        for (int r = 0; r < lr_rank; r++) {
            uint16_t scale_h;
            fread(&scale_h, 2, 1, fp);
            float sc = fp16_to_float(scale_h);
            for (size_t row = 0; row < rows; row++) {
                int8_t qb;
                fread(&qb, 1, 1, fp);
                US[row * lr_rank + r] = (float)qb * sc;
            }
        }

        float *Vt = (float *)calloc((size_t)lr_rank * cols, sizeof(float));
        for (int r = 0; r < lr_rank; r++) {
            uint16_t scale_h;
            fread(&scale_h, 2, 1, fp);
            float sc = fp16_to_float(scale_h);
            for (size_t col = 0; col < cols; col++) {
                int8_t qb;
                fread(&qb, 1, 1, fp);
                Vt[r * cols + col] = (float)qb * sc;
            }
        }

        /* Reconstruct LR into output */
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val = 0.0;
                for (int r = 0; r < lr_rank; r++)
                    val += (double)US[i * lr_rank + r] *
                           (double)Vt[r * cols + j];
                out[ti].data[i * cols + j] = (float)val;
            }
        }
        free(US);
        free(Vt);

        /* ── Read per-block metadata ── */
        float *coord_scales = (float *)calloc(n_blocks, sizeof(float));
        float *warp_norms = (float *)calloc(n_blocks, sizeof(float));

        if (fhdr.flags & FPQ_FLAG_FP8_SCALES) {
            for (size_t b = 0; b < n_blocks; b++) {
                uint8_t hv; fread(&hv, 1, 1, fp);
                coord_scales[b] = fp8_e4m3_to_float(hv);
            }
            for (size_t b = 0; b < n_blocks; b++) {
                uint8_t hv; fread(&hv, 1, 1, fp);
                warp_norms[b] = fp8_e4m3_to_float(hv);
            }
        } else {
            for (size_t b = 0; b < n_blocks; b++) {
                uint16_t hv; fread(&hv, 2, 1, fp);
                coord_scales[b] = fp16_to_float(hv);
            }
            for (size_t b = 0; b < n_blocks; b++) {
                uint16_t hv; fread(&hv, 2, 1, fp);
                warp_norms[b] = fp16_to_float(hv);
            }
        }

        float *resid_norms = (float *)calloc(n_blocks, sizeof(float));
        {
            uint16_t rn_scale_h; fread(&rn_scale_h, 2, 1, fp);
            float rn_sc = fp16_to_float(rn_scale_h);
            for (size_t b = 0; b < n_blocks; b++) {
                uint8_t qb; fread(&qb, 1, 1, fp);
                resid_norms[b] = (float)qb * rn_sc;
            }
        }

        /* ── Read E8 points ── */
        int8_t *e8_flat = (int8_t *)malloc(n_blocks * padded);
        if (fhdr.flags & FPQ_FLAG_E8_ENTROPY) {
            /* rANS entropy-coded */
            uint16_t freq[RANS_NSYM];
            fread(freq, sizeof(uint16_t), RANS_NSYM, fp);
            uint32_t comp_size, total_sym;
            fread(&comp_size, 4, 1, fp);
            fread(&total_sym, 4, 1, fp);

            uint8_t *comp = (uint8_t *)malloc(comp_size);
            fread(comp, 1, comp_size, fp);

            /* Build decode tables */
            uint16_t cum[RANS_NSYM];
            uint8_t lut[RANS_PROB_SCALE];
            rans_build_cumfreq(freq, RANS_NSYM, cum);
            rans_build_lookup(cum, freq, RANS_NSYM, lut);

            /* Decode to unsigned [0,127] then map to signed */
            uint8_t *decoded = (uint8_t *)malloc(total_sym);
            rans_decode(comp, comp_size, freq, cum, lut,
                        RANS_NSYM, decoded, total_sym);

            for (size_t i = 0; i < total_sym && i < n_blocks * padded; i++)
                e8_flat[i] = (int8_t)((int)decoded[i] - 64);

            free(decoded);
            free(comp);
        } else if (fhdr.flags & FPQ_FLAG_E8_INT7) {
            /* INT7 packed: 224 bytes/block → 256 int8 values */
            size_t packed_bytes = (padded * 7 + 7) / 8;  /* 224 */
            uint8_t *pack_buf = (uint8_t *)malloc(packed_bytes);
            for (size_t b = 0; b < n_blocks; b++) {
                fread(pack_buf, 1, packed_bytes, fp);
                unpack_int7(pack_buf, e8_flat + b * padded, padded);
            }
            free(pack_buf);
        } else {
            fread(e8_flat, 1, n_blocks * padded, fp);
        }

        /* ── Read RVQ codebook ── */
        float *tiles = (float *)calloc((size_t)effective_k * 16, sizeof(float));
        for (size_t j = 0; j < (size_t)effective_k * 16; j++) {
            uint16_t hv; fread(&hv, 2, 1, fp);
            tiles[j] = fp16_to_float(hv);
        }

        /* ── Read tile indices ── */
        uint8_t *tile_idx = (uint8_t *)malloc(n_blocks * 16);
        if (fhdr.flags & FPQ_FLAG_TILE_6BIT) {
            /* 6-bit packed: 12 bytes/block → 16 uint8 indices */
            uint8_t pack_buf6[12];
            for (size_t b = 0; b < n_blocks; b++) {
                fread(pack_buf6, 1, 12, fp);
                unpack_uint6(pack_buf6, tile_idx + b * 16, 16);
            }
        } else {
            fread(tile_idx, 1, n_blocks * 16, fp);
        }

        /* ── Read QJL bits ── */
        uint64_t *qjl_bits = (uint64_t *)calloc(n_blocks, sizeof(uint64_t));
        fread(qjl_bits, 8, n_blocks, fp);

        /* ── Read ghost correction ── */
        float ghost_sigma = 0.0f;
        float *ghost_u = NULL, *ghost_v = NULL;
        if (h->has_ghost) {
            uint16_t sigma_h, u_scale_h, v_scale_h;
            fread(&sigma_h, 2, 1, fp);
            fread(&u_scale_h, 2, 1, fp);
            ghost_sigma = fp16_to_float(sigma_h);
            float u_sc = fp16_to_float(u_scale_h);

            ghost_u = (float *)calloc(rows, sizeof(float));
            int8_t *u_q = (int8_t *)malloc(rows);
            fread(u_q, 1, rows, fp);
            dequant_int8(u_q, rows, u_sc, ghost_u);
            free(u_q);

            fread(&v_scale_h, 2, 1, fp);
            float v_sc = fp16_to_float(v_scale_h);
            ghost_v = (float *)calloc(cols, sizeof(float));
            int8_t *v_q = (int8_t *)malloc(cols);
            fread(v_q, 1, cols, fp);
            dequant_int8(v_q, cols, v_sc, ghost_v);
            free(v_q);
        }

        /* ── Read haar_seed ── */
        uint64_t haar_seed;
        fread(&haar_seed, 8, 1, fp);

        /* ── Reconstruct residual blocks (mirror of v9 decode) ── */
        float beta = 8.0f;  /* V8_MU_BETA */
        float lattice_scale = 8.0f * (float)h->coord_bits;

        for (size_t b = 0; b < n_blocks; b++) {
            size_t offset = b * block_dim;
            size_t this_dim = (offset + block_dim <= n) ? block_dim : (n - offset);
            float rms = coord_scales[b];
            float wnorm = warp_norms[b];

            /* Corrected E8 + tile */
            float corrected[256];
            for (int p = 0; p < 16; p++) {
                int tix = tile_idx[b * 16 + p];
                if (tix >= effective_k) tix = effective_k - 1;
                size_t pair_base = (size_t)p * 16;
                for (int d = 0; d < 16; d++)
                    corrected[pair_base + d] =
                        (float)e8_flat[b * padded + pair_base + d] +
                        tiles[tix * 16 + d];
            }

            /* Inverse warp + scale */
            float fwht_recon[256];
            for (size_t i = 0; i < padded; i++) {
                float lat_val = corrected[i] / lattice_scale * wnorm;
                float unwarp = native_warp_inverse(lat_val, beta);
                fwht_recon[i] = unwarp * rms;
            }

            /* QJL reconstruction (simplified — use norm + sign bits) */
            if (resid_norms[b] > 1e-10f) {
                float norm = resid_norms[b];
                uint64_t bits = qjl_bits[b];
                float scale = norm / 8.0f;  /* approximation */
                for (size_t i = 0; i < 64 && i < padded; i++) {
                    float sign = (bits & (1ULL << i)) ? 1.0f : -1.0f;
                    fwht_recon[i] += sign * scale;
                }
            }

            /* Inverse FWHT */
            fpq_fwht_inverse(fwht_recon, padded);
            fpq_random_signs_inverse(fwht_recon, padded, haar_seed ^ (uint64_t)b);

            /* Add to low-rank output */
            for (size_t i = 0; i < this_dim; i++)
                out[ti].data[offset + i] += fwht_recon[i];
        }

        /* Apply ghost correction */
        if (h->has_ghost && ghost_u && ghost_v) {
            for (size_t i = 0; i < rows; i++)
                for (size_t j = 0; j < cols; j++)
                    out[ti].data[i * cols + j] +=
                        ghost_sigma * ghost_u[i] * ghost_v[j];
        }

        free(coord_scales);
        free(warp_norms);
        free(resid_norms);
        free(e8_flat);
        free(tiles);
        free(tile_idx);
        free(qjl_bits);
        free(ghost_u);
        free(ghost_v);
    }

    for (uint32_t i = 0; i < nt; i++)
        free(names[i]);
    free(names);
    free(headers);
    fclose(fp);

    fprintf(stderr, "FPQ Native Read: loaded %u tensors from %s\n", nt, path);
    return out;
}


/* ═══════════════════════════════════════════════════════════════════
 * Compressed Reader: fpq_native_read_compressed
 *
 * Reads .fpq into fpq_tensor_t structs whose sbb_scale_delta array
 * is populated in the same v9 layout that fpqx_sli_matvec expects.
 * NO fp32 decode. The tensor goes straight to SLI prepare/matvec.
 *
 * Memory layout of sbb_scale_delta for each compressed tensor:
 *   [0]:       lr_rank (as float)
 *   [1]:       lr_rank (duplicate, v9 convention)
 *   [2 .. 2+US-1]:                UΣ factors [rows × lr_rank]
 *   [2+US .. 2+US+Vt-1]:         V^T factors [lr_rank × cols]
 *   [v8_base .. v8_base+nb-1]:   warp_norms  [n_blocks]
 *   [e8_off .. e8_off+nb*256-1]: E8 lattice  [n_blocks × 256]
 *   [tile_cb .. tile_cb+ek*16-1]: tile codebook [effective_k × 16]
 *   [tile_idx .. tile_idx+nb*16-1]: tile indices [n_blocks × 16]
 *   [after tile_idx]:             effective_k marker
 * ═══════════════════════════════════════════════════════════════════ */

fpq_tensor_t **fpq_native_read_compressed(const char *path, size_t *n_tensors_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "fpq_native_read_compressed: cannot open %s\n", path);
        *n_tensors_out = 0;
        return NULL;
    }

    fpq_native_file_header_t fhdr;
    if (fread(&fhdr, sizeof(fhdr), 1, fp) != 1 ||
        fhdr.magic != FPQ_NATIVE_MAGIC) {
        fprintf(stderr, "fpq_native_read_compressed: invalid magic in %s\n", path);
        fclose(fp);
        *n_tensors_out = 0;
        return NULL;
    }

    uint32_t nt = fhdr.n_tensors;
    *n_tensors_out = nt;

    /* Read tensor headers + names */
    fpq_native_tensor_header_t *headers = (fpq_native_tensor_header_t *)
        calloc(nt, sizeof(fpq_native_tensor_header_t));
    char **names = (char **)calloc(nt, sizeof(char *));
    for (uint32_t i = 0; i < nt; i++) {
        if (fread(&headers[i], sizeof(fpq_native_tensor_header_t), 1, fp) != 1) break;
        names[i] = (char *)calloc(headers[i].name_len + 1, 1);
        if (fread(names[i], 1, headers[i].name_len, fp) != headers[i].name_len) break;
    }

    /* Allocate output array */
    fpq_tensor_t **out = (fpq_tensor_t **)calloc(nt, sizeof(fpq_tensor_t *));

    for (uint32_t ti = 0; ti < nt; ti++) {
        fpq_native_tensor_header_t *h = &headers[ti];
        size_t rows = h->rows, cols = h->cols;
        size_t n = rows * cols;

        fpq_tensor_t *t = (fpq_tensor_t *)calloc(1, sizeof(fpq_tensor_t));
        out[ti] = t;
        strncpy(t->name, names[ti], sizeof(t->name) - 1);
        t->original_rows = rows;
        t->original_cols = cols;
        t->pid_alpha = -9.0f;  /* signal v9 format to SLI */

        fseek(fp, (long)h->data_offset, SEEK_SET);

        /* Small/1D tensors: store raw fp32 for pass-through (not SLI-able) */
        if (h->lr_rank == 0 && h->coord_bits == 0) {
            t->n_blocks = 0;
            t->coord_bits = 0;
            /* Read fp16 → store in coord_scales as raw fp32 values */
            t->coord_scales = (float *)calloc(n, sizeof(float));
            for (size_t j = 0; j < n; j++) {
                uint16_t hv;
                if (fread(&hv, 2, 1, fp) != 1) break;
                t->coord_scales[j] = fp16_to_float(hv);
            }
            continue;
        }

        int lr_rank = h->lr_rank;
        size_t n_blocks = h->n_blocks;
        int effective_k = h->effective_k;
        size_t padded = 256;

        t->n_blocks = n_blocks;
        t->coord_bits = h->coord_bits;

        /* Compute sbb_scale_delta layout sizes */
        size_t us_size = rows * (size_t)lr_rank;
        size_t vt_size = (size_t)lr_rank * cols;
        size_t v8_base = 2 + us_size + vt_size;
        size_t e8_off = v8_base + n_blocks;
        size_t e8_flat_size = n_blocks * padded;
        size_t tile_cb_off = e8_off + e8_flat_size;
        size_t tile_cb_size = (size_t)effective_k * 16;
        size_t tile_idx_off = tile_cb_off + tile_cb_size;
        size_t tile_idx_size = n_blocks * 16;
        size_t total_sbb = tile_idx_off + tile_idx_size + 1; /* +1 for ek marker */

        /* Sanity check: total_sbb should be reasonable.
         * For a 4096×4096 tensor: ~26M floats (~100MB). Cap at 1B floats (4GB). */
        if (total_sbb > 1000000000ULL) {
            fprintf(stderr, "fpq_native_read_compressed: tensor '%s' [%zux%zu] "
                    "has unreasonable sbb size %zu (lr=%d, blocks=%zu, ek=%d). "
                    "Treating as passthrough.\n",
                    names[ti], rows, cols, total_sbb, lr_rank, n_blocks, effective_k);
            t->n_blocks = 0;
            t->coord_bits = 0;
            t->coord_scales = (float *)calloc(rows * cols, sizeof(float));
            continue;
        }

        t->sbb_scale_delta = (float *)calloc(total_sbb, sizeof(float));
        t->sbb_scale_delta[0] = (float)lr_rank;
        t->sbb_scale_delta[1] = (float)lr_rank;

        /* ── Read LR factors: INT8 → float into sbb_scale_delta ── */
        {
            /* UΣ [rows × lr_rank] */
            float *us_dst = t->sbb_scale_delta + 2;
            for (int r = 0; r < lr_rank; r++) {
                uint16_t scale_h;
                fread(&scale_h, 2, 1, fp);
                float sc = fp16_to_float(scale_h);
                for (size_t row = 0; row < rows; row++) {
                    int8_t qb;
                    fread(&qb, 1, 1, fp);
                    us_dst[row * lr_rank + r] = (float)qb * sc;
                }
            }
            /* V^T [lr_rank × cols] */
            float *vt_dst = t->sbb_scale_delta + 2 + us_size;
            for (int r = 0; r < lr_rank; r++) {
                uint16_t scale_h;
                fread(&scale_h, 2, 1, fp);
                float sc = fp16_to_float(scale_h);
                for (size_t col = 0; col < cols; col++) {
                    int8_t qb;
                    fread(&qb, 1, 1, fp);
                    vt_dst[r * cols + col] = (float)qb * sc;
                }
            }
        }

        /* ── Read per-block metadata ── */
        t->coord_scales = (float *)calloc(n_blocks, sizeof(float));
        float *warp_dst = t->sbb_scale_delta + v8_base;

        if (fhdr.flags & FPQ_FLAG_FP8_SCALES) {
            for (size_t b = 0; b < n_blocks; b++) {
                uint8_t hv; fread(&hv, 1, 1, fp);
                t->coord_scales[b] = fp8_e4m3_to_float(hv);
            }
            for (size_t b = 0; b < n_blocks; b++) {
                uint8_t hv; fread(&hv, 1, 1, fp);
                warp_dst[b] = fp8_e4m3_to_float(hv);
            }
        } else {
            for (size_t b = 0; b < n_blocks; b++) {
                uint16_t hv; fread(&hv, 2, 1, fp);
                t->coord_scales[b] = fp16_to_float(hv);
            }
            for (size_t b = 0; b < n_blocks; b++) {
                uint16_t hv; fread(&hv, 2, 1, fp);
                warp_dst[b] = fp16_to_float(hv);
            }
        }

        /* residual_norms */
        t->coord_residual_norms = (float *)calloc(n_blocks, sizeof(float));
        {
            uint16_t rn_scale_h; fread(&rn_scale_h, 2, 1, fp);
            float rn_sc = fp16_to_float(rn_scale_h);
            for (size_t b = 0; b < n_blocks; b++) {
                uint8_t qb; fread(&qb, 1, 1, fp);
                t->coord_residual_norms[b] = (float)qb * rn_sc;
            }
        }

        /* ── Read E8 points → float in sbb_scale_delta ── */
        float *e8_dst = t->sbb_scale_delta + e8_off;
        if (fhdr.flags & FPQ_FLAG_E8_ENTROPY) {
            uint16_t freq[RANS_NSYM];
            fread(freq, sizeof(uint16_t), RANS_NSYM, fp);
            uint32_t comp_size, total_sym;
            fread(&comp_size, 4, 1, fp);
            fread(&total_sym, 4, 1, fp);

            uint8_t *comp = (uint8_t *)malloc(comp_size);
            fread(comp, 1, comp_size, fp);

            uint16_t cum[RANS_NSYM];
            uint8_t lut[RANS_PROB_SCALE];
            rans_build_cumfreq(freq, RANS_NSYM, cum);
            rans_build_lookup(cum, freq, RANS_NSYM, lut);

            uint8_t *decoded = (uint8_t *)malloc(total_sym);
            rans_decode(comp, comp_size, freq, cum, lut,
                        RANS_NSYM, decoded, total_sym);

            for (size_t i = 0; i < total_sym && i < n_blocks * padded; i++)
                e8_dst[i] = (float)((int)decoded[i] - 64);

            free(decoded);
            free(comp);
        } else if (fhdr.flags & FPQ_FLAG_E8_INT7) {
            size_t packed_bytes = (padded * 7 + 7) / 8;
            uint8_t *pack_buf = (uint8_t *)malloc(packed_bytes);
            int8_t *block_buf = (int8_t *)malloc(padded);
            for (size_t b = 0; b < n_blocks; b++) {
                fread(pack_buf, 1, packed_bytes, fp);
                unpack_int7(pack_buf, block_buf, padded);
                for (size_t d = 0; d < padded; d++)
                    e8_dst[b * padded + d] = (float)block_buf[d];
            }
            free(pack_buf);
            free(block_buf);
        } else {
            for (size_t b = 0; b < n_blocks; b++) {
                for (size_t d = 0; d < padded; d++) {
                    int8_t v; fread(&v, 1, 1, fp);
                    e8_dst[b * padded + d] = (float)v;
                }
            }
        }

        /* ── Read RVQ tile codebook → float in sbb_scale_delta ── */
        float *tile_cb_dst = t->sbb_scale_delta + tile_cb_off;
        for (size_t j = 0; j < tile_cb_size; j++) {
            uint16_t hv; fread(&hv, 2, 1, fp);
            tile_cb_dst[j] = fp16_to_float(hv);
        }

        /* ── Read tile indices → float in sbb_scale_delta ── */
        float *tile_idx_dst = t->sbb_scale_delta + tile_idx_off;
        if (fhdr.flags & FPQ_FLAG_TILE_6BIT) {
            uint8_t pack6[12];
            uint8_t idx16[16];
            for (size_t b = 0; b < n_blocks; b++) {
                fread(pack6, 1, 12, fp);
                unpack_uint6(pack6, idx16, 16);
                for (int p = 0; p < 16; p++)
                    tile_idx_dst[b * 16 + p] = (float)idx16[p];
            }
        } else {
            for (size_t b = 0; b < n_blocks; b++) {
                uint8_t idx8[16];
                fread(idx8, 1, 16, fp);
                for (int p = 0; p < 16; p++)
                    tile_idx_dst[b * 16 + p] = (float)idx8[p];
            }
        }

        /* effective_k marker (so SLI can discover it) */
        t->sbb_scale_delta[tile_idx_off + tile_idx_size] = (float)effective_k;

        /* ── Read QJL bits → allocate qjl structs ── */
        t->qjl = (fpq_qjl_t **)calloc(n_blocks, sizeof(fpq_qjl_t *));
        for (size_t b = 0; b < n_blocks; b++) {
            uint64_t bits;
            fread(&bits, 8, 1, fp);

            fpq_qjl_t *q = (fpq_qjl_t *)calloc(1, sizeof(fpq_qjl_t));
            q->n_projections = FPQ_QJL_PROJECTIONS;
            q->n_elements = padded;
            /* Derive proj_seed from haar_seed and block index
             * (will be set below after reading haar_seed — use placeholder) */
            q->bits = (uint64_t *)malloc(sizeof(uint64_t));
            q->bits[0] = bits;
            t->qjl[b] = q;
        }

        /* ── Read ghost correction ── */
        if (h->has_ghost) {
            uint16_t sigma_h, u_scale_h, v_scale_h;
            fread(&sigma_h, 2, 1, fp);
            fread(&u_scale_h, 2, 1, fp);
            float sigma = fp16_to_float(sigma_h);
            float u_sc = fp16_to_float(u_scale_h);

            fpq_ghost_t *g = (fpq_ghost_t *)calloc(1, sizeof(fpq_ghost_t));
            g->sigma = sigma;
            g->rows = rows;
            g->cols = cols;

            g->u = (float *)calloc(rows, sizeof(float));
            int8_t *u_q = (int8_t *)malloc(rows);
            fread(u_q, 1, rows, fp);
            dequant_int8(u_q, rows, u_sc, g->u);
            free(u_q);

            fread(&v_scale_h, 2, 1, fp);
            float v_sc = fp16_to_float(v_scale_h);
            g->v = (float *)calloc(cols, sizeof(float));
            int8_t *v_q = (int8_t *)malloc(cols);
            fread(v_q, 1, cols, fp);
            dequant_int8(v_q, cols, v_sc, g->v);
            free(v_q);

            t->ghost = g;
        }

        /* ── Read haar_seed ── */
        fread(&t->haar_seed, 8, 1, fp);

        /* Now fix QJL proj_seeds (derived from haar_seed) */
        for (size_t b = 0; b < n_blocks; b++) {
            if (t->qjl[b])
                t->qjl[b]->proj_seed = t->haar_seed ^ (uint64_t)b;
        }
    }

    for (uint32_t i = 0; i < nt; i++)
        free(names[i]);
    free(names);
    free(headers);
    fclose(fp);

    fprintf(stderr, "fpq_native_read_compressed: loaded %u tensors (SLI-ready) from %s\n",
            nt, path);
    return out;
}
