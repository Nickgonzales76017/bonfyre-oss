/*
 * serialize.c — FPQ File Format Serialization
 *
 * THE .FPQ FILE FORMAT:
 *
 * This is radically different from GGUF/GGML/safetensors.
 * Those formats store WEIGHTS. This stores PROGRAMS.
 *
 * Layout:
 *   Header:
 *     magic:     "FPQ!" (4 bytes)
 *     version:   uint32 (currently 1)
 *     n_tensors: uint32
 *     flags:     uint32 (reserved)
 *
 *   Per tensor:
 *     name_len:       uint32
 *     name:           char[name_len]
 *     original_rows:  uint64
 *     original_cols:  uint64
 *     n_blocks:       uint64
 *     haar_seed:      uint64
 *     total_bits:     uint64
 *     total_nodes:    uint64
 *
 *     Per block:
 *       radius:     float32
 *       tree_size:  uint32
 *       tree_nodes: [serialized combinator tree]
 *       n_elements: uint32
 *       n_proj:     uint32
 *       qjl_bits:   uint64[packed_words]
 *       proj_seed:  uint64
 *
 *   Combinator tree node (serialized):
 *     op:         uint8  (FPQ_OP_*)
 *     param[0]:   float32
 *     param[1]:   float32
 *     iparam:     int32
 *     has_left:   uint8
 *     [left tree if has_left]
 *     has_right:  uint8
 *     [right tree if has_right]
 *
 * The beauty: a 2B-parameter model might compress from 4GB fp32 to
 * perhaps 50-200MB of seed programs + QJL bits.
 */
#include "fpq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FPQ_FILE_MAGIC  0x21515046  /* "FPQ!" little-endian */
#define FPQ_FILE_VERSION 3

/* ── Write helpers ── */

static int write_u8(FILE *f, uint8_t v)  { return fwrite(&v, 1, 1, f) == 1; }
static int write_i32(FILE *f, int32_t v)  { return fwrite(&v, 4, 1, f) == 1; }
static int write_u32(FILE *f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
static int write_u64(FILE *f, uint64_t v) { return fwrite(&v, 8, 1, f) == 1; }
static int write_f32(FILE *f, float v)    { return fwrite(&v, 4, 1, f) == 1; }

/* ── Read helpers ── */

static int read_u8(FILE *f, uint8_t *v)  { return fread(v, 1, 1, f) == 1; }
static int read_i32(FILE *f, int32_t *v)  { return fread(v, 4, 1, f) == 1; }
static int read_u32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1; }
static int read_u64(FILE *f, uint64_t *v) { return fread(v, 8, 1, f) == 1; }
static int read_f32(FILE *f, float *v)    { return fread(v, 4, 1, f) == 1; }

/* ── Serialize a combinator tree (recursive) ── */

static int write_node(FILE *f, const fpq_node_t *node) {
    if (!node) {
        write_u8(f, 0xFF);  /* Null sentinel */
        return 1;
    }

    write_u8(f, (uint8_t)node->op);
    write_f32(f, node->param[0]);
    write_f32(f, node->param[1]);
    write_i32(f, (int32_t)node->iparam);

    /* Left child */
    write_u8(f, node->left ? 1 : 0);
    if (node->left) {
        if (!write_node(f, node->left)) return 0;
    }

    /* Right child */
    write_u8(f, node->right ? 1 : 0);
    if (node->right) {
        if (!write_node(f, node->right)) return 0;
    }

    return 1;
}

static fpq_node_t *read_node(FILE *f) {
    uint8_t op;
    if (!read_u8(f, &op)) return NULL;
    if (op == 0xFF) return NULL;  /* Null sentinel */

    fpq_node_t *node = fpq_node_alloc((fpq_op_t)op);
    read_f32(f, &node->param[0]);
    read_f32(f, &node->param[1]);
    int32_t ip;
    read_i32(f, &ip);
    node->iparam = (int)ip;

    uint8_t has_left;
    read_u8(f, &has_left);
    if (has_left) node->left = read_node(f);

    uint8_t has_right;
    read_u8(f, &has_right);
    if (has_right) node->right = read_node(f);

    return node;
}

/* ── Serialize a seed ── */

static int write_seed(FILE *f, const fpq_seed_t *seed) {
    write_u32(f, (uint32_t)seed->target_dim);
    write_u32(f, (uint32_t)seed->tree_size);
    write_f32(f, seed->distortion);
    return write_node(f, seed->root);
}

static fpq_seed_t *read_seed(FILE *f) {
    fpq_seed_t *seed = (fpq_seed_t *)calloc(1, sizeof(fpq_seed_t));
    uint32_t target_dim, tree_size;
    read_u32(f, &target_dim);
    read_u32(f, &tree_size);
    read_f32(f, &seed->distortion);
    seed->target_dim = target_dim;
    seed->tree_size = tree_size;
    seed->root = read_node(f);
    return seed;
}

/* ── Serialize QJL ── */

static int write_qjl(FILE *f, const fpq_qjl_t *qjl) {
    write_u32(f, (uint32_t)qjl->n_elements);
    write_u32(f, (uint32_t)qjl->n_projections);
    write_u64(f, qjl->proj_seed);

    /* bits are already packed as uint64_t words:
     * ceil(n_projections / 64) words */
    size_t n_words = (qjl->n_projections + 63) / 64;
    for (size_t w = 0; w < n_words; w++) {
        write_u64(f, qjl->bits[w]);
    }

    return 1;
}

static fpq_qjl_t *read_qjl(FILE *f) {
    fpq_qjl_t *qjl = (fpq_qjl_t *)calloc(1, sizeof(fpq_qjl_t));
    uint32_t n_elements, n_proj;
    read_u32(f, &n_elements);
    read_u32(f, &n_proj);
    read_u64(f, &qjl->proj_seed);
    qjl->n_elements = n_elements;
    qjl->n_projections = n_proj;

    size_t n_words = (n_proj + 63) / 64;
    qjl->bits = (uint64_t *)calloc(n_words, sizeof(uint64_t));
    for (size_t w = 0; w < n_words; w++) {
        read_u64(f, &qjl->bits[w]);
    }

    return qjl;
}

/* ── Save complete FPQ model ── */

int fpq_save(const char *path, fpq_tensor_t **tensors, size_t n_tensors) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "FPQ: Cannot open %s for writing\n", path);
        return -1;
    }

    /* Header */
    write_u32(f, FPQ_FILE_MAGIC);
    write_u32(f, FPQ_FILE_VERSION);
    write_u32(f, (uint32_t)n_tensors);
    write_u32(f, 0); /* flags — reserved */

    for (size_t t = 0; t < n_tensors; t++) {
        const fpq_tensor_t *tensor = tensors[t];

        /* Tensor header */
        uint32_t name_len = (uint32_t)strlen(tensor->name);
        write_u32(f, name_len);
        fwrite(tensor->name, 1, name_len, f);
        write_u64(f, tensor->original_rows);
        write_u64(f, tensor->original_cols);
        write_u64(f, tensor->n_blocks);
        write_u64(f, tensor->haar_seed);
        write_u64(f, tensor->total_bits);
        write_u64(f, tensor->total_seed_nodes);

        /* v3: mode and optional base seed */
        write_u8(f, tensor->mode);
        write_u8(f, tensor->base_seed ? 1 : 0);
        if (tensor->base_seed) {
            write_seed(f, tensor->base_seed);
        }

        /* Per-block data */
        for (size_t b = 0; b < tensor->n_blocks; b++) {
            write_f32(f, tensor->radii[b]);
            write_seed(f, tensor->seeds[b]);
            write_qjl(f, tensor->qjl[b]);
        }
    }

    long file_size = ftell(f);
    fclose(f);

    fprintf(stderr, "FPQ: Saved %zu tensors to %s (%.2f MB)\n",
            n_tensors, path, (float)file_size / (1024.0f * 1024.0f));
    return 0;
}

/* ── Load complete FPQ model ── */

fpq_tensor_t **fpq_load(const char *path, size_t *n_tensors) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FPQ: Cannot open %s for reading\n", path);
        return NULL;
    }

    /* Header */
    uint32_t magic, version, n_tens, flags;
    read_u32(f, &magic);
    if (magic != FPQ_FILE_MAGIC) {
        fprintf(stderr, "FPQ: Invalid magic 0x%08X (expected FPQ!)\n", magic);
        fclose(f);
        return NULL;
    }
    read_u32(f, &version);
    if (version > FPQ_FILE_VERSION) {
        fprintf(stderr, "FPQ: Version %u not supported (max %u)\n", version, FPQ_FILE_VERSION);
        fclose(f);
        return NULL;
    }
    read_u32(f, &n_tens);
    read_u32(f, &flags);

    fpq_tensor_t **tensors = (fpq_tensor_t **)calloc(n_tens, sizeof(fpq_tensor_t *));

    for (uint32_t t = 0; t < n_tens; t++) {
        fpq_tensor_t *tensor = (fpq_tensor_t *)calloc(1, sizeof(fpq_tensor_t));

        /* Tensor header */
        uint32_t name_len;
        read_u32(f, &name_len);
        if (name_len > sizeof(tensor->name) - 1) name_len = sizeof(tensor->name) - 1;
        if (fread(tensor->name, 1, name_len, f) != name_len) {
            free(tensor);
            break;
        }
        read_u64(f, (uint64_t *)&tensor->original_rows);
        read_u64(f, (uint64_t *)&tensor->original_cols);
        read_u64(f, (uint64_t *)&tensor->n_blocks);
        read_u64(f, &tensor->haar_seed);
        read_u64(f, (uint64_t *)&tensor->total_bits);
        read_u64(f, (uint64_t *)&tensor->total_seed_nodes);

        /* v3: mode and optional base seed */
        if (version >= 3) {
            uint8_t mode_byte;
            read_u8(f, &mode_byte);
            tensor->mode = mode_byte;
            uint8_t has_base;
            read_u8(f, &has_base);
            if (has_base) {
                tensor->base_seed = read_seed(f);
            }
        }

        tensor->seeds = (fpq_seed_t **)calloc(tensor->n_blocks, sizeof(fpq_seed_t *));
        tensor->qjl   = (fpq_qjl_t **)calloc(tensor->n_blocks, sizeof(fpq_qjl_t *));
        tensor->radii  = (float *)calloc(tensor->n_blocks, sizeof(float));

        for (size_t b = 0; b < tensor->n_blocks; b++) {
            read_f32(f, &tensor->radii[b]);
            tensor->seeds[b] = read_seed(f);
            tensor->qjl[b]   = read_qjl(f);
        }

        tensors[t] = tensor;
    }

    fclose(f);
    *n_tensors = n_tens;

    fprintf(stderr, "FPQ: Loaded %u tensors from %s\n", n_tens, path);
    return tensors;
}
