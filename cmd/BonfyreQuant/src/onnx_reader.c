/*
 * onnx_reader.c — Minimal ONNX protobuf reader for bonfyre-quant
 *
 * Reads float32 initializer tensors from ONNX model files without
 * any external protobuf library. Handles:
 *   - raw_data  (field 9, bytes): dense float32 binary blob
 *   - float_data (field 4, repeated float32, packed/unpacked)
 *
 * Returns an array of OnnxTensor (caller frees with onnx_tensors_free).
 *
 * ONNX proto structure (fields used):
 *   ModelProto { graph (7: GraphProto) }
 *   GraphProto  { initializer (5: TensorProto[]) }
 *   TensorProto { dims(1,int64[]), data_type(2,int32),
 *                 float_data(4,float[]), name(5,string), raw_data(9,bytes) }
 */
#include "onnx_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Protobuf wire types ── */
#define WT_VARINT    0
#define WT_64BIT     1
#define WT_LEN       2  /* length-delimited */
#define WT_32BIT     5

/* ── Varint decoder ── */
static uint64_t pb_read_varint(const uint8_t *buf, size_t len,
                                size_t *pos) {
    uint64_t val = 0;
    int shift = 0;
    while (*pos < len) {
        uint8_t b = buf[(*pos)++];
        val |= (uint64_t)(b & 0x7F) << shift;
        shift += 7;
        if (!(b & 0x80)) break;
    }
    return val;
}

/* Skip a field of given wire type (for unknown fields) */
static void pb_skip(const uint8_t *buf, size_t len, size_t *pos, int wtype) {
    switch (wtype) {
    case WT_VARINT:
        pb_read_varint(buf, len, pos);
        break;
    case WT_64BIT:
        *pos += 8;
        break;
    case WT_LEN: {
        uint64_t sz = pb_read_varint(buf, len, pos);
        *pos += (size_t)sz;
        break;
    }
    case WT_32BIT:
        *pos += 4;
        break;
    default:
        *pos = len; /* bail */
    }
}

/* ── Parse one TensorProto message ── */
static OnnxTensor parse_tensor(const uint8_t *buf, size_t len) {
    OnnxTensor t;
    memset(&t, 0, sizeof(t));

    size_t pos = 0;
    size_t n_elements = 1;
    int    got_dims = 0;
    int    data_type = 0;

    while (pos < len) {
        uint64_t tag = pb_read_varint(buf, len, &pos);
        if (pos >= len && tag == 0) break;
        int field  = (int)(tag >> 3);
        int wtype  = (int)(tag & 0x7);

        if (field == 1 && wtype == WT_LEN) {
            /* dims — packed int64 repeated */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            size_t end = pos + (size_t)sz;
            n_elements = 1;
            got_dims = 1;
            while (pos < end) {
                int64_t d = (int64_t)pb_read_varint(buf, end, &pos);
                if (d > 0) n_elements *= (size_t)d;
            }
            pos = end;
        } else if (field == 1 && wtype == WT_VARINT) {
            /* dims — unpacked single value */
            int64_t d = (int64_t)pb_read_varint(buf, len, &pos);
            if (!got_dims) { n_elements = 1; got_dims = 1; }
            if (d > 0) n_elements *= (size_t)d;
        } else if (field == 2 && wtype == WT_VARINT) {
            /* data_type */
            data_type = (int)pb_read_varint(buf, len, &pos);
        } else if (field == 4 && wtype == WT_LEN) {
            /* float_data — packed floats (repeated float, proto3 wire) */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            size_t n = (size_t)sz / 4;
            if (n > 0 && !t.data) {
                t.data = (float *)malloc(n * sizeof(float));
                if (t.data) {
                    memcpy(t.data, buf + pos, n * sizeof(float));
                    t.n_elements = n;
                }
            }
            pos += (size_t)sz;
        } else if (field == 4 && wtype == WT_32BIT) {
            /* float_data — single unpacked float */
            if (pos + 4 <= len) {
                float v;
                memcpy(&v, buf + pos, 4);
                pos += 4;
                /* grow array */
                float *tmp = (float *)realloc(t.data,
                                (t.n_elements + 1) * sizeof(float));
                if (tmp) { t.data = tmp; t.data[t.n_elements++] = v; }
            }
        } else if (field == 5 && wtype == WT_LEN) {
            /* name */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            size_t copy = sz < (size_t)(ONNX_NAME_MAX - 1) ?
                          (size_t)sz : (size_t)(ONNX_NAME_MAX - 1);
            memcpy(t.name, buf + pos, copy);
            t.name[copy] = '\0';
            pos += (size_t)sz;
        } else if (field == 9 && wtype == WT_LEN) {
            /* raw_data — packed float32 binary */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            size_t n = (size_t)sz / 4;
            if (n > 0 && !t.data) {
                t.data = (float *)malloc(n * sizeof(float));
                if (t.data) {
                    memcpy(t.data, buf + pos, n * sizeof(float));
                    t.n_elements = n;
                }
            }
            pos += (size_t)sz;
        } else {
            pb_skip(buf, len, &pos, wtype);
        }
    }

    /* If dims said 0 elements (e.g. scalar), use actual parsed count */
    if (got_dims && n_elements == 0) n_elements = t.n_elements;
    if (t.n_elements == 0 && n_elements > 0) t.n_elements = n_elements;

    /* Only keep if FLOAT (data_type=1) or data_type unset */
    if (data_type != 0 && data_type != 1) {
        free(t.data);
        t.data = NULL;
        t.n_elements = 0;
    }
    return t;
}

/* ── Parse GraphProto: collect initializer TensorProtos ── */
static void parse_graph(const uint8_t *buf, size_t len,
                         OnnxTensor **out, size_t *n_out) {
    size_t pos = 0;
    while (pos < len) {
        uint64_t tag = pb_read_varint(buf, len, &pos);
        if (pos >= len && tag == 0) break;
        int field = (int)(tag >> 3);
        int wtype = (int)(tag & 0x7);

        if (field == 5 && wtype == WT_LEN) {
            /* initializer TensorProto */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            OnnxTensor t = parse_tensor(buf + pos, (size_t)sz);
            pos += (size_t)sz;

            if (t.data && t.n_elements > 0) {
                OnnxTensor *tmp = (OnnxTensor *)realloc(*out,
                    (*n_out + 1) * sizeof(OnnxTensor));
                if (tmp) {
                    *out = tmp;
                    (*out)[*n_out] = t;
                    (*n_out)++;
                } else {
                    free(t.data);
                }
            }
        } else {
            pb_skip(buf, len, &pos, wtype);
        }
    }
}

/* ── Public: onnx_read ── */
OnnxTensor *onnx_read(const char *path, size_t *n_out) {
    *n_out = 0;

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    rewind(f);
    if (fsz <= 0) { fclose(f); return NULL; }

    uint8_t *raw = (uint8_t *)malloc((size_t)fsz);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, (size_t)fsz, f) != (size_t)fsz) {
        free(raw); fclose(f); return NULL;
    }
    fclose(f);

    OnnxTensor *tensors = NULL;
    size_t pos = 0;
    size_t len = (size_t)fsz;

    while (pos < len) {
        uint64_t tag = pb_read_varint(raw, len, &pos);
        if (pos >= len && tag == 0) break;
        int field = (int)(tag >> 3);
        int wtype = (int)(tag & 0x7);

        if (field == 7 && wtype == WT_LEN) {
            /* graph: GraphProto */
            uint64_t sz = pb_read_varint(raw, len, &pos);
            parse_graph(raw + pos, (size_t)sz, &tensors, n_out);
            pos += (size_t)sz;
        } else {
            pb_skip(raw, len, &pos, wtype);
        }
    }

    free(raw);
    return tensors;
}

void onnx_tensors_free(OnnxTensor *tensors, size_t n) {
    if (!tensors) return;
    for (size_t i = 0; i < n; i++) free(tensors[i].data);
    free(tensors);
}
