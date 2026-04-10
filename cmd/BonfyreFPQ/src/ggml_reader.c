/*
 * ggml_reader.c — Read GGML-format model files
 *
 * GGML is the format used by whisper.cpp / llama.cpp / ggml:
 *
 *   File layout:
 *     magic: 0x67676d6c ("ggml") or 0x67676a74 ("ggjt") etc.
 *     [hyperparams — model-specific]
 *     [vocab — model-specific]
 *     [tensors]:
 *       n_dims (uint32)
 *       name_len (uint32)
 *       type (uint32)  — GGML_TYPE_F32=0, F16=1, Q4_0=2, etc.
 *       dims[n_dims] (uint32 each)
 *       name[name_len] (char)
 *       [padding to 32-byte alignment]
 *       data[...]
 *
 * We read ALL tensors as f32 (dequantizing as needed) so FPQ can work
 * on the raw float values.
 *
 * Supported initial format: GGJT v3 (llama.cpp / whisper.cpp standard)
 */
#include "fpq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── GGML type enum (subset) ── */
enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
};

/* Type size in bytes per element for simple types */
static size_t ggml_type_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        default: return 0;   /* quantized types have block sizes */
    }
}

/* Block size for quantized types */
static size_t ggml_block_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return 32;
        case GGML_TYPE_Q4_1: return 32;
        case GGML_TYPE_Q5_0: return 32;
        case GGML_TYPE_Q5_1: return 32;
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q8_1: return 32;
        case GGML_TYPE_Q2_K: return 256;
        case GGML_TYPE_Q3_K: return 256;
        case GGML_TYPE_Q4_K: return 256;
        case GGML_TYPE_Q5_K: return 256;
        case GGML_TYPE_Q6_K: return 256;
        default: return 1;
    }
}

/* Bytes per block for quantized types */
static size_t ggml_type_block_bytes(uint32_t type) {
    switch (type) {
        /* Q4_0: 1 f16 scale + 32/2 nibbles = 2 + 16 = 18 bytes per 32 elements */
        case GGML_TYPE_Q4_0: return 18;
        case GGML_TYPE_Q4_1: return 20;   /* + f16 min */
        case GGML_TYPE_Q5_0: return 22;   /* + 4 byte high bits */
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;   /* f16 scale + 32 int8 */
        case GGML_TYPE_Q8_1: return 36;
        default: return 0;
    }
}

/* ── fp16 → fp32 conversion ── */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            union { uint32_t u; float f; } u = { sign };
            return u.f;
        }
        /* Denormalized */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        /* Inf/NaN */
        union { uint32_t u; float f; } u = { sign | 0x7F800000 | (mant << 13) };
        return u.f;
    }

    exp = exp + (127 - 15);
    mant = mant << 13;
    union { uint32_t u; float f; } u = { sign | (exp << 23) | mant };
    return u.f;
}

/* ── Q4_0 dequantization ── */
/* Block: 2 bytes (f16 scale) + 16 bytes (32 nibbles packed) = 18 bytes → 32 floats */
static void dequant_q4_0(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / 32;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * 18;
        uint16_t scale_fp16;
        memcpy(&scale_fp16, block, 2);
        float scale = fp16_to_fp32(scale_fp16);

        for (size_t i = 0; i < 16; i++) {
            uint8_t byte = block[2 + i];
            dst[b * 32 + i]      = ((float)(byte & 0xF) - 8.0f) * scale;
            dst[b * 32 + i + 16] = ((float)(byte >> 4)  - 8.0f) * scale;
        }
    }
}

/* ── Q8_0 dequantization ── */
/* Block: 2 bytes (f16 scale) + 32 bytes (32 int8) = 34 bytes → 32 floats */
static void dequant_q8_0(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / 32;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * 34;
        uint16_t scale_fp16;
        memcpy(&scale_fp16, block, 2);
        float scale = fp16_to_fp32(scale_fp16);

        for (size_t i = 0; i < 32; i++) {
            dst[b * 32 + i] = ((float)(int8_t)block[2 + i]) * scale;
        }
    }
}

/* ── Read raw tensor data and convert to f32 ── */
static float *read_tensor_data(FILE *f, uint32_t type, size_t n_elements) {
    float *data = (float *)malloc(n_elements * sizeof(float));
    if (!data) return NULL;

    if (type == GGML_TYPE_F32) {
        if (fread(data, sizeof(float), n_elements, f) != n_elements) {
            free(data);
            return NULL;
        }
    } else if (type == GGML_TYPE_F16) {
        uint16_t *buf = (uint16_t *)malloc(n_elements * sizeof(uint16_t));
        if (fread(buf, sizeof(uint16_t), n_elements, f) != n_elements) {
            free(buf);
            free(data);
            return NULL;
        }
        for (size_t i = 0; i < n_elements; i++) {
            data[i] = fp16_to_fp32(buf[i]);
        }
        free(buf);
    } else if (type == GGML_TYPE_Q4_0) {
        size_t n_blocks = n_elements / 32;
        size_t raw_bytes = n_blocks * 18;
        uint8_t *buf = (uint8_t *)malloc(raw_bytes);
        if (fread(buf, 1, raw_bytes, f) != raw_bytes) {
            free(buf);
            free(data);
            return NULL;
        }
        dequant_q4_0(buf, data, n_elements);
        free(buf);
    } else if (type == GGML_TYPE_Q8_0) {
        size_t n_blocks = n_elements / 32;
        size_t raw_bytes = n_blocks * 34;
        uint8_t *buf = (uint8_t *)malloc(raw_bytes);
        if (fread(buf, 1, raw_bytes, f) != raw_bytes) {
            free(buf);
            free(data);
            return NULL;
        }
        dequant_q8_0(buf, data, n_elements);
        free(buf);
    } else {
        /* Unsupported quantization — skip and zero-fill */
        fprintf(stderr, "  [WARN] Unsupported GGML type %u, skipping data\n", type);
        size_t bs = ggml_block_size(type);
        size_t bb = ggml_type_block_bytes(type);
        if (bs > 0 && bb > 0) {
            size_t n_blocks = (n_elements + bs - 1) / bs;
            fseek(f, (long)(n_blocks * bb), SEEK_CUR);
        }
        memset(data, 0, n_elements * sizeof(float));
    }

    return data;
}

/* ═══════════════════════════════════════════════════════════════════
 * GGUF v3 READER
 *
 * GGUF layout:
 *   magic:      "GGUF" (0x46554747)
 *   version:    uint32 (3)
 *   n_tensors:  uint64
 *   n_kv:       uint64
 *   [metadata KVs]  — key-value pairs (skip all)
 *   [tensor infos]  — name + n_dims + dims + type + offset per tensor
 *   [alignment padding]
 *   [tensor data]   — contiguous f16/q8/etc blocks
 *
 * GGUF string: uint64 len + char[len] (NOT null-terminated)
 * GGUF types: 0=uint8, 1=int8, 2=uint16, 3=int16, 4=uint32, 5=int32,
 *             6=float32, 7=bool, 8=string, 9=array, 10=uint64,
 *             11=int64, 12=float64
 * ═══════════════════════════════════════════════════════════════════ */

/* GGUF value type enum */
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* Read a GGUF string: uint64 len + char[len] */
static int gguf_read_string(FILE *f, char *buf, size_t buf_size) {
    uint64_t len;
    if (fread(&len, 8, 1, f) != 1) return -1;
    if (len >= buf_size) {
        /* String too long for buffer — read and discard */
        fseek(f, (long)len, SEEK_CUR);
        buf[0] = '\0';
        return 0;
    }
    if (len > 0 && fread(buf, 1, (size_t)len, f) != (size_t)len) return -1;
    buf[len] = '\0';
    return 0;
}

/* Skip a single GGUF value of given type */
static int gguf_skip_value(FILE *f, uint32_t vtype) {
    switch (vtype) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            fseek(f, 2, SEEK_CUR); break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_STRING: {
            uint64_t slen;
            if (fread(&slen, 8, 1, f) != 1) return -1;
            fseek(f, (long)slen, SEEK_CUR);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type;
            uint64_t n_elem;
            if (fread(&elem_type, 4, 1, f) != 1) return -1;
            if (fread(&n_elem, 8, 1, f) != 1) return -1;
            for (uint64_t i = 0; i < n_elem; i++) {
                if (gguf_skip_value(f, elem_type) != 0) return -1;
            }
            break;
        }
        default:
            fprintf(stderr, "GGUF: Unknown value type %u\n", vtype);
            return -1;
    }
    return 0;
}

/* Q4_K dequantization: 256-element superblocks */
/* Block layout (144 bytes per 256 elements):
 *   fp16 d (2) + fp16 dmin (2) + 12 bytes scales + 128 bytes qs */
static void dequant_q4_k(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / 256;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * 144;
        uint16_t d_fp16, dmin_fp16;
        memcpy(&d_fp16, block, 2);
        memcpy(&dmin_fp16, block + 2, 2);
        float d = fp16_to_fp32(d_fp16);
        float dmin = fp16_to_fp32(dmin_fp16);
        const uint8_t *scales = block + 4;
        const uint8_t *qs = block + 16;

        /* Decode 8 sub-blocks of 32 elements each */
        for (int j = 0; j < 256; j++) {
            int sub = j / 32;
            float sc, m;
            if (sub < 4) {
                sc = d * (float)(scales[sub] & 0x3F);
                m  = dmin * (float)(scales[sub + 4] & 0x3F);
            } else {
                sc = d * (float)(((scales[sub + 4] & 0xF) << 2) | ((scales[sub - 4] >> 6) & 3));
                m  = dmin * (float)(((scales[sub + 4] >> 4) << 2) | ((scales[sub] >> 6) & 3));
            }
            uint8_t byte = qs[j / 2];
            int nib = (j % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            dst[b * 256 + j] = sc * (float)nib - m;
        }
    }
}

/* Q6_K dequantization: 256-element superblocks */
/* Block layout (210 bytes per 256 elements):
 *   128 bytes ql + 64 bytes qh + 16 bytes scales + fp16 d (2) */
static void dequant_q6_k(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = n_elements / 256;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * 210;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *scales = (const int8_t *)(block + 192);
        uint16_t d_fp16;
        memcpy(&d_fp16, block + 208, 2);
        float d = fp16_to_fp32(d_fp16);

        for (int j = 0; j < 256; j++) {
            int sub = j / 16;
            int8_t sc = scales[sub];
            uint8_t q_lo = ql[j / 2];
            int lo = (j % 2 == 0) ? (q_lo & 0xF) : (q_lo >> 4);
            /* High bits from qh — 2 bits per element packed */
            int hi_byte_idx = j / 4;
            int hi_shift = (j % 4) * 2;
            int hi = (qh[hi_byte_idx] >> hi_shift) & 3;
            int q = lo | (hi << 4);
            dst[b * 256 + j] = d * sc * ((float)q - 32.0f);
        }
    }
}

/* Bytes per block for GGUF quantized types */
static size_t gguf_type_block_bytes(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:  return 0;  /* not block-quantized */
        case GGML_TYPE_F16:  return 0;
        case GGML_TYPE_Q4_0: return 18;   /* 32 elements */
        case GGML_TYPE_Q8_0: return 34;   /* 32 elements */
        case GGML_TYPE_Q4_K: return 144;  /* 256 elements */
        case GGML_TYPE_Q5_K: return 176;  /* 256 elements */
        case GGML_TYPE_Q6_K: return 210;  /* 256 elements */
        case GGML_TYPE_Q2_K: return 84;   /* 256 elements */
        case GGML_TYPE_Q3_K: return 110;  /* 256 elements */
        default: return 0;
    }
}

/* Compute raw byte size for a tensor given type + n_elements */
static size_t gguf_tensor_data_size(uint32_t type, size_t n_elements) {
    switch (type) {
        case GGML_TYPE_F32:  return n_elements * 4;
        case GGML_TYPE_F16:  return n_elements * 2;
        default: {
            size_t bs = ggml_block_size(type);
            size_t bb = gguf_type_block_bytes(type);
            if (bs == 0 || bb == 0) return 0;
            return (n_elements / bs) * bb;
        }
    }
}

static fpq_raw_tensor_t *fpq_gguf_read(FILE *f, const char *path, size_t *n_tensors) {
    /* Read GGUF header (magic already consumed) */
    uint32_t version;
    uint64_t tensor_count, n_kv;
    if (fread(&version, 4, 1, f) != 1) return NULL;
    if (fread(&tensor_count, 8, 1, f) != 1) return NULL;
    if (fread(&n_kv, 8, 1, f) != 1) return NULL;

    fprintf(stderr, "FPQ: GGUF v%u — %llu tensors, %llu metadata keys\n",
            version, (unsigned long long)tensor_count,
            (unsigned long long)n_kv);

    if (version < 2 || version > 3) {
        fprintf(stderr, "FPQ: Unsupported GGUF version %u\n", version);
        return NULL;
    }

    /* Skip all metadata KVs */
    for (uint64_t i = 0; i < n_kv; i++) {
        char key[512];
        if (gguf_read_string(f, key, sizeof(key)) != 0) {
            fprintf(stderr, "FPQ: Failed reading metadata key %llu\n",
                    (unsigned long long)i);
            return NULL;
        }
        uint32_t vtype;
        if (fread(&vtype, 4, 1, f) != 1) return NULL;
        if (gguf_skip_value(f, vtype) != 0) {
            fprintf(stderr, "FPQ: Failed skipping metadata value for '%s'\n", key);
            return NULL;
        }
    }

    /* Read tensor infos */
    typedef struct {
        char name[256];
        uint32_t n_dims;
        uint64_t dims[4];
        uint32_t type;
        uint64_t offset;  /* relative to start of data section */
    } gguf_tensor_info_t;

    gguf_tensor_info_t *infos = (gguf_tensor_info_t *)calloc(
        (size_t)tensor_count, sizeof(gguf_tensor_info_t));

    for (uint64_t i = 0; i < tensor_count; i++) {
        gguf_tensor_info_t *info = &infos[i];

        /* name (GGUF string) */
        if (gguf_read_string(f, info->name, sizeof(info->name)) != 0) goto fail;

        /* n_dims (uint32) */
        if (fread(&info->n_dims, 4, 1, f) != 1) goto fail;
        if (info->n_dims > 4) goto fail;

        /* dims (uint64 each) */
        for (uint32_t d = 0; d < info->n_dims; d++) {
            if (fread(&info->dims[d], 8, 1, f) != 1) goto fail;
        }

        /* type (uint32) */
        if (fread(&info->type, 4, 1, f) != 1) goto fail;

        /* offset (uint64, relative to data start) */
        if (fread(&info->offset, 8, 1, f) != 1) goto fail;
    }

    /* Data section starts at next alignment boundary (default 32 bytes) */
    long pos = ftell(f);
    long data_start = (pos + 31) & ~31L;

    /* Allocate output */
    fpq_raw_tensor_t *tensors = (fpq_raw_tensor_t *)calloc(
        (size_t)tensor_count, sizeof(fpq_raw_tensor_t));
    size_t count = 0;

    for (uint64_t i = 0; i < tensor_count; i++) {
        gguf_tensor_info_t *info = &infos[i];

        size_t n_elements = 1;
        for (uint32_t d = 0; d < info->n_dims; d++)
            n_elements *= (size_t)info->dims[d];

        if (n_elements > 500000000 || n_elements == 0) {
            fprintf(stderr, "  [SKIP] %s: %zu elements\n", info->name, n_elements);
            continue;
        }

        /* Seek to tensor data */
        long tensor_pos = data_start + (long)info->offset;
        fseek(f, tensor_pos, SEEK_SET);

        /* Read and dequantize */
        float *data = NULL;

        if (info->type == GGML_TYPE_F32) {
            data = (float *)malloc(n_elements * sizeof(float));
            if (fread(data, sizeof(float), n_elements, f) != n_elements) {
                free(data); data = NULL;
            }
        } else if (info->type == GGML_TYPE_F16) {
            uint16_t *buf = (uint16_t *)malloc(n_elements * sizeof(uint16_t));
            data = (float *)malloc(n_elements * sizeof(float));
            if (fread(buf, sizeof(uint16_t), n_elements, f) != n_elements) {
                free(buf); free(data); data = NULL;
            } else {
                for (size_t j = 0; j < n_elements; j++)
                    data[j] = fp16_to_fp32(buf[j]);
                free(buf);
            }
        } else if (info->type == GGML_TYPE_Q4_0) {
            size_t raw_bytes = (n_elements / 32) * 18;
            uint8_t *buf = (uint8_t *)malloc(raw_bytes);
            data = (float *)malloc(n_elements * sizeof(float));
            if (fread(buf, 1, raw_bytes, f) != raw_bytes) {
                free(buf); free(data); data = NULL;
            } else {
                dequant_q4_0(buf, data, n_elements);
                free(buf);
            }
        } else if (info->type == GGML_TYPE_Q8_0) {
            size_t raw_bytes = (n_elements / 32) * 34;
            uint8_t *buf = (uint8_t *)malloc(raw_bytes);
            data = (float *)malloc(n_elements * sizeof(float));
            if (fread(buf, 1, raw_bytes, f) != raw_bytes) {
                free(buf); free(data); data = NULL;
            } else {
                dequant_q8_0(buf, data, n_elements);
                free(buf);
            }
        } else if (info->type == GGML_TYPE_Q4_K) {
            size_t raw_bytes = (n_elements / 256) * 144;
            uint8_t *buf = (uint8_t *)malloc(raw_bytes);
            data = (float *)malloc(n_elements * sizeof(float));
            if (fread(buf, 1, raw_bytes, f) != raw_bytes) {
                free(buf); free(data); data = NULL;
            } else {
                dequant_q4_k(buf, data, n_elements);
                free(buf);
            }
        } else if (info->type == GGML_TYPE_Q6_K) {
            size_t raw_bytes = (n_elements / 256) * 210;
            uint8_t *buf = (uint8_t *)malloc(raw_bytes);
            data = (float *)malloc(n_elements * sizeof(float));
            if (fread(buf, 1, raw_bytes, f) != raw_bytes) {
                free(buf); free(data); data = NULL;
            } else {
                dequant_q6_k(buf, data, n_elements);
                free(buf);
            }
        } else {
            /* Unsupported type — zero-fill */
            fprintf(stderr, "  [WARN] %s: unsupported type %u, zero-filling\n",
                    info->name, info->type);
            data = (float *)calloc(n_elements, sizeof(float));
        }

        if (!data) {
            fprintf(stderr, "  [FAIL] %s: read error\n", info->name);
            continue;
        }

        strncpy(tensors[count].name, info->name, sizeof(tensors[count].name) - 1);
        tensors[count].n_dims = info->n_dims;
        tensors[count].rows = (info->n_dims >= 2) ? (size_t)info->dims[1] : 1;
        tensors[count].cols = (size_t)info->dims[0];
        tensors[count].ggml_type = info->type;
        tensors[count].n_elements = n_elements;
        tensors[count].data = data;

        fprintf(stderr, "  [%zu] %-50s %u-D [%llu",
                count, info->name, info->n_dims,
                (unsigned long long)info->dims[0]);
        for (uint32_t d = 1; d < info->n_dims; d++)
            fprintf(stderr, "×%llu", (unsigned long long)info->dims[d]);
        fprintf(stderr, "] type=%u n=%zu\n", info->type, n_elements);

        count++;
    }

    free(infos);
    *n_tensors = count;
    fprintf(stderr, "FPQ: Read %zu tensors from %s (GGUF v%u)\n", count, path, version);
    return tensors;

fail:
    free(infos);
    fprintf(stderr, "FPQ: GGUF parse error\n");
    *n_tensors = 0;
    return NULL;
}

/* ── Read GGML/GGJT model file ── */

/* GGML magic values */
#define GGML_MAGIC  0x67676D6C  /* "ggml" */
#define GGJT_MAGIC  0x67676A74  /* "ggjt" */
#define GGUF_MAGIC  0x46554747  /* "GGUF" */

fpq_raw_tensor_t *fpq_ggml_read(const char *path, size_t *n_tensors) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FPQ: Cannot open %s\n", path);
        return NULL;
    }

    /* Read magic */
    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    int is_ggjt = 0;
    uint32_t version = 0;

    if (magic == GGJT_MAGIC) {
        is_ggjt = 1;
        if (fread(&version, 4, 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        fprintf(stderr, "FPQ: GGJT v%u model\n", version);
    } else if (magic == GGML_MAGIC) {
        fprintf(stderr, "FPQ: GGML model\n");
    } else if (magic == GGUF_MAGIC) {
        /* ── GGUF v3 reader ── */
        fpq_raw_tensor_t *result = fpq_gguf_read(f, path, n_tensors);
        fclose(f);
        return result;
    } else {
        fprintf(stderr, "FPQ: Unknown magic 0x%08X\n", magic);
        fclose(f);
        return NULL;
    }

    /* Skip hyperparams + vocab
     * For whisper models: we skip until we find tensor headers.
     * For GGJT v3+: tensors immediately follow the header.
     *
     * Strategy: scan for tensor headers by looking for valid n_dims values.
     * A robust parser would know the model-specific header size,
     * but we want to be generic. For now, we accept a model_type hint.
     */

    /* For whisper GGJT v3: skip model-specific header
     * Whisper header: n_vocab(4) + n_audio_ctx(4) + n_audio_state(4) +
     *                 n_audio_head(4) + n_audio_layer(4) +
     *                 n_text_ctx(4) + n_text_state(4) +
     *                 n_text_head(4) + n_text_layer(4) +
     *                 n_mels(4) + ftype(4) = 44 bytes
     *
     * Then: n_vocab tokens (each: token_len(4) + token_bytes[token_len])
     *
     * For Gemma/llama GGJT: similar pattern
     *
     * Generic approach: read model header size from first few fields. */

    /* Try typical whisper header: read n_vocab */
    long header_start = ftell(f);
    uint32_t n_vocab;
    if (fread(&n_vocab, 4, 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    /* Heuristic: valid vocab sizes are 1000-200000 */
    if (n_vocab > 200000) {
        /* Not a whisper model; try to skip to tensors by scanning */
        fseek(f, header_start, SEEK_SET);
        n_vocab = 0;
    }

    /* Skip remaining hyperparams (10 uint32 for whisper = 40 bytes after n_vocab) */
    if (n_vocab > 0 && n_vocab < 200000) {
        fseek(f, 40, SEEK_CUR);

        /* Skip vocab tokens */
        for (uint32_t i = 0; i < n_vocab; i++) {
            uint32_t token_len;
            if (fread(&token_len, 4, 1, f) != 1) break;
            if (token_len > 10000) {
                /* Sanity check failed; this isn't the vocab */
                fprintf(stderr, "FPQ: Vocab parse error at token %u, len=%u\n", i, token_len);
                break;
            }
            fseek(f, (long)token_len, SEEK_CUR);
        }
    }

    /* Now read tensors */
    size_t capacity = 256;
    size_t count = 0;
    fpq_raw_tensor_t *tensors = (fpq_raw_tensor_t *)calloc(capacity, sizeof(fpq_raw_tensor_t));

    while (!feof(f)) {
        uint32_t n_dims;
        if (fread(&n_dims, 4, 1, f) != 1) break;

        /* Sanity check: n_dims should be 1-4 */
        if (n_dims < 1 || n_dims > 4) {
            /* Try advancing 1 byte and retrying (alignment scan) */
            fseek(f, -3, SEEK_CUR);
            continue;
        }

        uint32_t name_len;
        if (fread(&name_len, 4, 1, f) != 1) break;
        if (name_len > 256) {
            fseek(f, -7, SEEK_CUR);
            continue;
        }

        uint32_t ftype;
        if (fread(&ftype, 4, 1, f) != 1) break;
        if (ftype > 20) {
            fseek(f, -11, SEEK_CUR);
            continue;
        }

        /* Read dimensions */
        uint32_t dims[4] = {1, 1, 1, 1};
        for (uint32_t d = 0; d < n_dims; d++) {
            if (fread(&dims[d], 4, 1, f) != 1) goto done;
        }

        /* Read name */
        char name[257] = {0};
        if (name_len > 0) {
            if (fread(name, 1, name_len, f) != name_len) goto done;
        }

        /* Align to 32 bytes for GGJT */
        if (is_ggjt) {
            long pos = ftell(f);
            long aligned = (pos + 31) & ~31L;
            if (aligned > pos) fseek(f, aligned - pos, SEEK_CUR);
        }

        /* Total elements */
        size_t n_elements = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            n_elements *= dims[d];
        }

        /* Sanity check elements */
        if (n_elements > 500000000) {
            fprintf(stderr, "FPQ: Suspiciously large tensor '%s': %zu elements, skipping\n",
                    name, n_elements);
            continue;
        }

        /* Read and dequantize */
        float *data = read_tensor_data(f, ftype, n_elements);
        if (!data) {
            fprintf(stderr, "FPQ: Failed to read tensor '%s'\n", name);
            continue;
        }

        /* Store */
        if (count >= capacity) {
            capacity *= 2;
            tensors = (fpq_raw_tensor_t *)realloc(tensors, capacity * sizeof(fpq_raw_tensor_t));
        }

        strncpy(tensors[count].name, name, sizeof(tensors[count].name) - 1);
        tensors[count].n_dims = n_dims;
        tensors[count].rows = (n_dims >= 2) ? dims[1] : 1;
        tensors[count].cols = dims[0];
        tensors[count].ggml_type = ftype;
        tensors[count].data = data;
        tensors[count].n_elements = n_elements;

        fprintf(stderr, "  [%zu] %-40s %u-D [%u", count, name, n_dims, dims[0]);
        for (uint32_t d = 1; d < n_dims; d++) fprintf(stderr, "×%u", dims[d]);
        fprintf(stderr, "] type=%u n=%zu\n", ftype, n_elements);

        count++;
    }

done:
    fclose(f);
    *n_tensors = count;

    fprintf(stderr, "FPQ: Read %zu tensors from %s\n", count, path);
    return tensors;
}

/* ── Cleanup ── */

void fpq_raw_tensor_free(fpq_raw_tensor_t *tensors, size_t n) {
    if (!tensors) return;
    for (size_t i = 0; i < n; i++) {
        free(tensors[i].data);
    }
    free(tensors);
}
