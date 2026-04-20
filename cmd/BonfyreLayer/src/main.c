/*
 * bonfyre-layer — Layer-aware ONNX model inspection and extraction (C port)
 *
 * Part of the Bonfyre layer-aware infrastructure (Track B).
 * Operates on ONNX protobuf directly — no Python or onnx library required.
 *
 * Usage:
 *   bonfyre-layer inspect      <model.onnx>
 *   bonfyre-layer layers       <model.onnx>
 *   bonfyre-layer pull-layer   <model.onnx> --range START:END --out DIR
 *   bonfyre-layer pull-head    <model.onnx> --name PREFIX --out DIR
 *   bonfyre-layer pack-transform <name> <part1.onnx> [part2.onnx ...] --out DIR
 *   bonfyre-layer schema
 *   bonfyre-layer --help
 *
 * Layer Artifact (written to DIR/artifact.json):
 *   type, source_model, layer_spec, node_range, n_params,
 *   sha256, format = "onnx"
 *
 * Protobuf fields (ONNX subset used):
 *   ModelProto:  graph(7), ir_version(1), opset_import(8)
 *   GraphProto:  node(1), initializer(5), value_info(13), input(11), output(12)
 *   NodeProto:   input(1), output(2), name(3), op_type(4), attribute(5)
 *   TensorProto: dims(1), data_type(2), float_data(4), name(5), raw_data(9)
 */
#include <bonfyre.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define VERSION "1.0.0"
#define MAX_NODES 8192
#define MAX_NAME  512
#define MAX_PATH  4096

/* ═══════════════════════════════════════════════════════════════════
 * Minimal protobuf primitives (read + write)
 * ═══════════════════════════════════════════════════════════════════ */

#define WT_VARINT 0
#define WT_64BIT  1
#define WT_LEN    2
#define WT_32BIT  5

static uint64_t pb_read_varint(const uint8_t *buf, size_t len, size_t *pos) {
    uint64_t val = 0; int shift = 0;
    while (*pos < len) {
        uint8_t b = buf[(*pos)++];
        val |= (uint64_t)(b & 0x7F) << shift;
        shift += 7;
        if (!(b & 0x80)) break;
    }
    return val;
}

static void pb_skip(const uint8_t *buf, size_t len, size_t *pos, int wtype) {
    switch (wtype) {
    case WT_VARINT: pb_read_varint(buf, len, pos); break;
    case WT_64BIT:  *pos += 8; break;
    case WT_LEN: { uint64_t sz = pb_read_varint(buf, len, pos); *pos += (size_t)sz; break; }
    case WT_32BIT:  *pos += 4; break;
    default: *pos = len;
    }
}

/* Dynamic write buffer */
typedef struct { uint8_t *data; size_t len; size_t cap; } WBuf;

static int wbuf_grow(WBuf *b, size_t need) {
    if (b->len + need <= b->cap) return 1;
    size_t nc = (b->cap * 2 > b->len + need) ? b->cap * 2 : b->len + need + 256;
    uint8_t *p = (uint8_t *)realloc(b->data, nc);
    if (!p) return 0;
    b->data = p; b->cap = nc;
    return 1;
}

static void wbuf_init(WBuf *b) { b->data=NULL; b->len=0; b->cap=0; }
static void wbuf_free(WBuf *b) { free(b->data); wbuf_init(b); }

static void wbuf_write_varint(WBuf *b, uint64_t v) {
    uint8_t tmp[10]; int n=0;
    do { tmp[n++] = (uint8_t)((v & 0x7F) | (v > 127 ? 0x80 : 0)); v >>= 7; } while(v);
    if (!wbuf_grow(b, (size_t)n)) return;
    memcpy(b->data + b->len, tmp, (size_t)n); b->len += (size_t)n;
}

static void wbuf_write_tag(WBuf *b, int field, int wtype) {
    wbuf_write_varint(b, (uint64_t)(field << 3 | wtype));
}

static void wbuf_write_bytes(WBuf *b, int field, const uint8_t *data, size_t n) {
    wbuf_write_tag(b, field, WT_LEN);
    wbuf_write_varint(b, (uint64_t)n);
    if (!wbuf_grow(b, n)) return;
    memcpy(b->data + b->len, data, n); b->len += n;
}

static void wbuf_write_string(WBuf *b, int field, const char *s) {
    wbuf_write_bytes(b, field, (const uint8_t *)s, strlen(s));
}

static void wbuf_write_i64(WBuf *b, int field, int64_t v) {
    wbuf_write_tag(b, field, WT_VARINT);
    wbuf_write_varint(b, (uint64_t)v);
}

/* Write file from WBuf */
static int wbuf_save(const WBuf *b, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return 1; }
    int ok = (fwrite(b->data, 1, b->len, f) == b->len);
    fclose(f);
    return ok ? 0 : 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * ONNX graph reader — captures nodes + initializers with raw bytes
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    char     name[MAX_NAME];
    char     op_type[64];
    char     inputs[8][MAX_NAME];
    int      n_inputs;
    char     outputs[4][MAX_NAME];
    int      n_outputs;
    /* raw bytes of this NodeProto (for verbatim copy to new model) */
    uint8_t *raw;
    size_t   raw_len;
} BfLayerNode;

typedef struct {
    char    name[MAX_NAME];
    size_t  n_elements;
    /* raw bytes of this TensorProto */
    uint8_t *raw;
    size_t   raw_len;
} BfInitTensor;

typedef struct {
    BfLayerNode   *nodes;
    size_t         n_nodes;
    BfInitTensor  *inits;
    size_t         n_inits;
    int64_t        ir_version;
    int64_t        opset_version;
    /* raw ir_version & opset_import bytes (for copy to new model) */
    uint8_t        opset_raw[64];
    size_t         opset_raw_len;
} BfLayerGraph;

static void bf_graph_free(BfLayerGraph *g) {
    for (size_t i = 0; i < g->n_nodes; i++) free(g->nodes[i].raw);
    free(g->nodes);
    for (size_t i = 0; i < g->n_inits; i++) free(g->inits[i].raw);
    free(g->inits);
}

static void parse_node(const uint8_t *buf, size_t len, BfLayerNode *nd) {
    nd->n_inputs = 0; nd->n_outputs = 0;
    nd->name[0] = '\0'; nd->op_type[0] = '\0';
    size_t pos = 0;
    while (pos < len) {
        uint64_t tag = pb_read_varint(buf, len, &pos);
        int field = (int)(tag >> 3), wtype = (int)(tag & 7);
        if (field == 1 && wtype == WT_LEN) { /* input */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            if (nd->n_inputs < 8) {
                size_t cp = sz < MAX_NAME-1 ? (size_t)sz : MAX_NAME-1;
                memcpy(nd->inputs[nd->n_inputs], buf+pos, cp);
                nd->inputs[nd->n_inputs][cp] = '\0';
                nd->n_inputs++;
            }
            pos += (size_t)sz;
        } else if (field == 2 && wtype == WT_LEN) { /* output */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            if (nd->n_outputs < 4) {
                size_t cp = sz < MAX_NAME-1 ? (size_t)sz : MAX_NAME-1;
                memcpy(nd->outputs[nd->n_outputs], buf+pos, cp);
                nd->outputs[nd->n_outputs][cp] = '\0';
                nd->n_outputs++;
            }
            pos += (size_t)sz;
        } else if (field == 3 && wtype == WT_LEN) { /* name */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            size_t cp = sz < MAX_NAME-1 ? (size_t)sz : MAX_NAME-1;
            memcpy(nd->name, buf+pos, cp); nd->name[cp] = '\0';
            pos += (size_t)sz;
        } else if (field == 4 && wtype == WT_LEN) { /* op_type */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            size_t cp = sz < 63 ? (size_t)sz : 63;
            memcpy(nd->op_type, buf+pos, cp); nd->op_type[cp] = '\0';
            pos += (size_t)sz;
        } else {
            pb_skip(buf, len, &pos, wtype);
        }
    }
}

static size_t parse_tensor_nelems(const uint8_t *buf, size_t len) {
    size_t pos = 0, n = 1; int got = 0;
    while (pos < len) {
        uint64_t tag = pb_read_varint(buf, len, &pos);
        int field = (int)(tag >> 3), wtype = (int)(tag & 7);
        if (field == 1 && wtype == WT_LEN) { /* dims — packed */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            size_t end = pos + (size_t)sz; n = 1; got = 1;
            while (pos < end) { int64_t d = (int64_t)pb_read_varint(buf, end, &pos); if(d>0) n *= (size_t)d; }
            pos = end;
        } else if (field == 1 && wtype == WT_VARINT) { /* dims — unpacked */
            int64_t d = (int64_t)pb_read_varint(buf, len, &pos);
            if (!got) { n = 1; got = 1; } if (d > 0) n *= (size_t)d;
        } else { pb_skip(buf, len, &pos, wtype); }
    }
    return got ? n : 0;
}

static void parse_init_name(const uint8_t *buf, size_t len, char *name, size_t nsz) {
    size_t pos = 0; name[0] = '\0';
    while (pos < len) {
        uint64_t tag = pb_read_varint(buf, len, &pos);
        int field = (int)(tag >> 3), wtype = (int)(tag & 7);
        if (field == 5 && wtype == WT_LEN) {
            uint64_t sz = pb_read_varint(buf, len, &pos);
            size_t cp = sz < nsz-1 ? (size_t)sz : nsz-1;
            memcpy(name, buf+pos, cp); name[cp] = '\0';
            return;
        } else { pb_skip(buf, len, &pos, wtype); }
    }
}

static void parse_graph(const uint8_t *buf, size_t len, BfLayerGraph *g) {
    g->nodes  = (BfLayerNode *)calloc(MAX_NODES, sizeof(BfLayerNode));
    g->inits  = (BfInitTensor *)calloc(MAX_NODES, sizeof(BfInitTensor));
    g->n_nodes = 0; g->n_inits = 0;

    size_t pos = 0;
    while (pos < len) {
        uint64_t tag = pb_read_varint(buf, len, &pos);
        int field = (int)(tag >> 3), wtype = (int)(tag & 7);
        if (field == 1 && wtype == WT_LEN) { /* node */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            if (g->n_nodes < MAX_NODES) {
                BfLayerNode *nd = &g->nodes[g->n_nodes++];
                nd->raw = (uint8_t *)malloc((size_t)sz);
                if (nd->raw) { memcpy(nd->raw, buf+pos, (size_t)sz); nd->raw_len = (size_t)sz; }
                parse_node(buf+pos, (size_t)sz, nd);
                /* fill default name */
                if (!nd->name[0]) snprintf(nd->name, sizeof(nd->name), "%s_%zu", nd->op_type, g->n_nodes-1);
            }
            pos += (size_t)sz;
        } else if (field == 5 && wtype == WT_LEN) { /* initializer */
            uint64_t sz = pb_read_varint(buf, len, &pos);
            if (g->n_inits < MAX_NODES) {
                BfInitTensor *it = &g->inits[g->n_inits++];
                it->raw = (uint8_t *)malloc((size_t)sz);
                if (it->raw) { memcpy(it->raw, buf+pos, (size_t)sz); it->raw_len = (size_t)sz; }
                it->n_elements = parse_tensor_nelems(buf+pos, (size_t)sz);
                parse_init_name(buf+pos, (size_t)sz, it->name, sizeof(it->name));
            }
            pos += (size_t)sz;
        } else {
            pb_skip(buf, len, &pos, wtype);
        }
    }
}

static int read_model_file(const char *path, uint8_t **out, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); rewind(f);
    if (fsz <= 0) { fclose(f); return 1; }
    *sz = (size_t)fsz;
    *out = (uint8_t *)malloc(*sz);
    if (!*out) { fclose(f); return 1; }
    if (fread(*out, 1, *sz, f) != *sz) { free(*out); fclose(f); return 1; }
    fclose(f);
    return 0;
}

static int load_graph(const char *path, BfLayerGraph *g) {
    uint8_t *raw = NULL; size_t rawsz = 0;
    if (read_model_file(path, &raw, &rawsz)) return 1;

    g->ir_version = 0; g->opset_version = 0;
    g->opset_raw_len = 0;
    g->nodes = NULL; g->n_nodes = 0;
    g->inits = NULL; g->n_inits = 0;

    size_t pos = 0;
    while (pos < rawsz) {
        size_t tag_pos = pos;
        uint64_t tag = pb_read_varint(raw, rawsz, &pos);
        int field = (int)(tag >> 3), wtype = (int)(tag & 7);
        if (field == 1 && wtype == WT_VARINT) { /* ir_version */
            g->ir_version = (int64_t)pb_read_varint(raw, rawsz, &pos);
        } else if (field == 7 && wtype == WT_LEN) { /* graph */
            uint64_t sz = pb_read_varint(raw, rawsz, &pos);
            parse_graph(raw + pos, (size_t)sz, g);
            pos += (size_t)sz;
        } else if (field == 8 && wtype == WT_LEN) { /* opset_import */
            uint64_t sz = pb_read_varint(raw, rawsz, &pos);
            /* capture raw opset blob for forwarding */
            size_t blob_sz = (size_t)(pos - tag_pos) - 1 + (size_t)sz;
            (void)blob_sz;
            /* parse out version */
            size_t op_pos = 0;
            while (op_pos < (size_t)sz) {
                uint64_t t2 = pb_read_varint(raw+pos, (size_t)sz, &op_pos);
                int f2 = (int)(t2 >> 3), w2 = (int)(t2 & 7);
                if (f2 == 2 && w2 == WT_VARINT)
                    g->opset_version = (int64_t)pb_read_varint(raw+pos, (size_t)sz, &op_pos);
                else pb_skip(raw+pos, (size_t)sz, &op_pos, w2);
            }
            pos += (size_t)sz;
        } else {
            pb_skip(raw, rawsz, &pos, wtype);
        }
    }
    free(raw);
    return (g->nodes != NULL) ? 0 : 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * SHA-256 (for artifact.json hash)
 * ═══════════════════════════════════════════════════════════════════ */

static void sha256_file_hex(const char *path, char *out64) {
    /* Use bf_sha256_file from libbonfyre if available, else "unknown" */
    out64[0] = '\0';
    FILE *f = fopen(path, "rb");
    if (!f) { strcpy(out64, "unknown"); return; }
    /* Simple SHA-256 via OpenSSL CLI or system sha256sum if available,
     * otherwise delegate to libbonfyre's bf_sha256_hex */
    fclose(f);

    /* Use bf_sha256_hex if libbonfyre exposes it */
    FILE *cmd = NULL;
    char buf[256];
    char cmdbuf[MAX_PATH + 64];
    /* macOS: shasum -a 256; Linux: sha256sum */
    snprintf(cmdbuf, sizeof(cmdbuf), "shasum -a 256 '%s' 2>/dev/null || sha256sum '%s' 2>/dev/null", path, path);
    cmd = popen(cmdbuf, "r");
    if (cmd && fgets(buf, sizeof(buf), cmd)) {
        /* output: "<hash>  filename" */
        size_t i = 0;
        while (i < 64 && buf[i] && buf[i] != ' ' && buf[i] != '\t' && buf[i] != '\n') {
            out64[i] = buf[i]; i++;
        }
        out64[i] = '\0';
    } else {
        strcpy(out64, "unknown");
    }
    if (cmd) pclose(cmd);
}

/* ═══════════════════════════════════════════════════════════════════
 * mkdir -p helper
 * ═══════════════════════════════════════════════════════════════════ */

static void mkdirp(const char *path) {
    char tmp[MAX_PATH];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

/* ═══════════════════════════════════════════════════════════════════
 * Write extracted ONNX model (node slice + required initializers)
 * ═══════════════════════════════════════════════════════════════════ */

static int write_onnx_slice(const char *path,
                             const BfLayerGraph *g,
                             size_t node_start, size_t node_end) {
    /* Collect names consumed by selected nodes and not produced by them */
    /* Mark all outputs produced by selected nodes */
    char produced[MAX_NODES][MAX_NAME]; size_t n_produced = 0;
    for (size_t i = node_start; i < node_end && i < g->n_nodes; i++) {
        for (int o = 0; o < g->nodes[i].n_outputs; o++) {
            if (g->nodes[i].outputs[o][0] && n_produced < MAX_NODES)
                strncpy(produced[n_produced++], g->nodes[i].outputs[o], MAX_NAME-1);
        }
    }
    /* Collect initializers needed */
    WBuf graph_buf; wbuf_init(&graph_buf);
    /* nodes */
    for (size_t i = node_start; i < node_end && i < g->n_nodes; i++) {
        const BfLayerNode *nd = &g->nodes[i];
        if (nd->raw && nd->raw_len)
            wbuf_write_bytes(&graph_buf, 1, nd->raw, nd->raw_len);
    }
    /* initializers that are consumed but not produced internally */
    for (size_t i = node_start; i < node_end && i < g->n_nodes; i++) {
        const BfLayerNode *nd = &g->nodes[i];
        for (int inp = 0; inp < nd->n_inputs; inp++) {
            const char *iname = nd->inputs[inp];
            if (!iname[0]) continue;
            /* Check if it's a graph initializer */
            int already_produced = 0;
            for (size_t p = 0; p < n_produced && !already_produced; p++)
                if (strcmp(produced[p], iname) == 0) already_produced = 1;
            if (already_produced) continue;
            for (size_t ki = 0; ki < g->n_inits; ki++) {
                if (strcmp(g->inits[ki].name, iname) == 0 && g->inits[ki].raw) {
                    wbuf_write_bytes(&graph_buf, 5, g->inits[ki].raw, g->inits[ki].raw_len);
                    break;
                }
            }
        }
    }
    /* Wrap graph in ModelProto */
    WBuf model_buf; wbuf_init(&model_buf);
    wbuf_write_i64(&model_buf, 1, 8); /* ir_version = 8 */
    /* opset_import { version=17 } */
    WBuf opset_buf; wbuf_init(&opset_buf);
    wbuf_write_string(&opset_buf, 1, ""); /* domain = "" */
    wbuf_write_i64(&opset_buf, 2, 17);   /* version = 17 */
    wbuf_write_bytes(&model_buf, 8, opset_buf.data, opset_buf.len);
    wbuf_free(&opset_buf);
    wbuf_write_bytes(&model_buf, 7, graph_buf.data, graph_buf.len);
    wbuf_free(&graph_buf);

    int rc = wbuf_save(&model_buf, path);
    wbuf_free(&model_buf);
    return rc;
}

/* ═══════════════════════════════════════════════════════════════════
 * Write artifact.json
 * ═══════════════════════════════════════════════════════════════════ */

static void write_artifact(const char *artifact_path, const char *onnx_path,
                            const char *artifact_type,
                            const char *source_model, const char *layer_spec_type,
                            size_t node_start, size_t node_end, size_t n_nodes,
                            const char *name_or_range, size_t n_params) {
    char sha[65]; sha256_file_hex(onnx_path, sha);
    char ts[32]; bf_iso_timestamp(ts, sizeof(ts));

    FILE *f = fopen(artifact_path, "w");
    if (!f) { perror(artifact_path); return; }

    fprintf(f,
        "{\n"
        "  \"type\": \"%s\",\n"
        "  \"version\": \"1.0.0\",\n"
        "  \"source_model\": \"%s\",\n"
        "  \"layer_spec\": {\n"
        "    \"type\": \"%s\",\n"
        "    \"value\": \"%s\"\n"
        "  },\n"
        "  \"node_range\": [%zu, %zu],\n"
        "  \"n_nodes\": %zu,\n"
        "  \"n_params\": %zu,\n"
        "  \"sha256\": \"%s\",\n"
        "  \"fpq_sha256\": null,\n"
        "  \"format\": \"onnx\",\n"
        "  \"created_at\": \"%s\"\n"
        "}\n",
        artifact_type, source_model,
        layer_spec_type, name_or_range,
        node_start, node_end,
        n_nodes, n_params, sha, ts);
    fclose(f);
}

/* Count params in selected node range (sum up initializer n_elements for used inits) */
static size_t count_range_params(const BfLayerGraph *g, size_t start, size_t end) {
    size_t total = 0;
    /* For each initializer name consumed by nodes[start:end], sum elements */
    for (size_t i = start; i < end && i < g->n_nodes; i++) {
        for (int inp = 0; inp < g->nodes[i].n_inputs; inp++) {
            const char *iname = g->nodes[i].inputs[inp];
            for (size_t ki = 0; ki < g->n_inits; ki++) {
                if (strcmp(g->inits[ki].name, iname) == 0) {
                    total += g->inits[ki].n_elements;
                    break;
                }
            }
        }
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════
 * Commands
 * ═══════════════════════════════════════════════════════════════════ */

static int cmd_inspect(const char *model_path) {
    BfLayerGraph g;
    if (load_graph(model_path, &g)) {
        fprintf(stderr, "bonfyre-layer: failed to load %s\n", model_path);
        return 1;
    }
    /* Count total params */
    size_t total_params = 0;
    for (size_t i = 0; i < g.n_inits; i++) total_params += g.inits[i].n_elements;

    /* Op distribution */
    char ops[64][32]; int op_counts[64]; int n_ops = 0;
    for (size_t i = 0; i < g.n_nodes; i++) {
        int found = 0;
        for (int j = 0; j < n_ops; j++) {
            if (strcmp(ops[j], g.nodes[i].op_type) == 0) { op_counts[j]++; found = 1; break; }
        }
        if (!found && n_ops < 64) {
            strncpy(ops[n_ops], g.nodes[i].op_type, 31);
            op_counts[n_ops] = 1; n_ops++;
        }
    }

    printf("Model:      %s\n", model_path);
    printf("  IR ver:   %lld\n", (long long)g.ir_version);
    printf("  Opset:    %lld\n", (long long)g.opset_version);
    printf("  Nodes:    %zu\n", g.n_nodes);
    printf("  Inits:    %zu\n", g.n_inits);
    printf("  Params:   %zu\n\n", total_params);
    printf("Op distribution:\n");
    for (int i = 0; i < n_ops; i++)
        printf("  %-20s %d\n", ops[i], op_counts[i]);
    printf("\nLayer pull range: 0:%zu  (all nodes)\n", g.n_nodes);
    printf("  Example: bonfyre-layer pull-layer %s --range 0:%zu --out /tmp/layer\n",
           model_path, g.n_nodes / 2);

    bf_graph_free(&g);
    return 0;
}

static int cmd_layers(const char *model_path) {
    BfLayerGraph g;
    if (load_graph(model_path, &g)) return 1;

    printf("%-5s %-18s %-35s %s\n", "IDX", "OP", "NAME", "OUTPUTS");
    printf("%-5s %-18s %-35s %s\n", "---", "--", "----", "-------");
    for (size_t i = 0; i < g.n_nodes; i++) {
        const BfLayerNode *nd = &g.nodes[i];
        char out_str[128] = "";
        for (int o = 0; o < nd->n_outputs && o < 2; o++) {
            if (o > 0) strncat(out_str, ", ", sizeof(out_str)-1-strlen(out_str));
            strncat(out_str, nd->outputs[o], sizeof(out_str)-1-strlen(out_str));
        }
        if (nd->n_outputs > 2) strncat(out_str, " ...", sizeof(out_str)-1-strlen(out_str));
        printf("%-5zu %-18.18s %-35.35s %s\n", i, nd->op_type, nd->name, out_str);
    }
    bf_graph_free(&g);
    return 0;
}

static int cmd_pull_layer(const char *model_path, const char *range_str,
                           const char *out_dir) {
    BfLayerGraph g;
    if (load_graph(model_path, &g)) return 1;

    size_t start = 0, end = 0;
    if (sscanf(range_str, "%zu:%zu", &start, &end) != 2) {
        fprintf(stderr, "bonfyre-layer: --range must be START:END, got: %s\n", range_str);
        bf_graph_free(&g); return 1;
    }
    if (end > g.n_nodes || start >= end) {
        fprintf(stderr, "bonfyre-layer: range %zu:%zu out of bounds (0:%zu)\n",
                start, end, g.n_nodes);
        bf_graph_free(&g); return 1;
    }

    mkdirp(out_dir);
    char onnx_path[MAX_PATH], artifact_path[MAX_PATH];
    snprintf(onnx_path,     sizeof(onnx_path),     "%s/layer_fragment.onnx", out_dir);
    snprintf(artifact_path, sizeof(artifact_path), "%s/artifact.json",       out_dir);

    if (write_onnx_slice(onnx_path, &g, start, end)) {
        fprintf(stderr, "bonfyre-layer: failed to write %s\n", onnx_path);
        bf_graph_free(&g); return 1;
    }

    size_t n_params = count_range_params(&g, start, end);
    write_artifact(artifact_path, onnx_path, "layer_fragment",
                   model_path, "layer_range",
                   start, end, end - start,
                   range_str, n_params);

    printf("[bonfyre-layer] pulled layer range %s\n", range_str);
    printf("  nodes:    %zu\n", end - start);
    printf("  params:   %zu\n", n_params);
    printf("  onnx:     %s\n", onnx_path);
    printf("  artifact: %s\n\n", artifact_path);
    printf("  Next: python3 scripts/finalize_artifact.py %s\n", out_dir);
    printf("        bonfyre-model add %s\n", artifact_path);

    bf_graph_free(&g);
    return 0;
}

static int cmd_pull_head(const char *model_path, const char *name_prefix,
                          const char *out_dir) {
    BfLayerGraph g;
    if (load_graph(model_path, &g)) return 1;

    /* Find nodes matching name prefix or op_type prefix */
    size_t first = SIZE_MAX, last = 0; int found = 0;
    for (size_t i = 0; i < g.n_nodes; i++) {
        int match = (strncmp(g.nodes[i].name, name_prefix, strlen(name_prefix)) == 0 ||
                     strncmp(g.nodes[i].op_type, name_prefix, strlen(name_prefix)) == 0);
        if (match) {
            if (i < first) first = i;
            if (i > last)  last  = i;
            found++;
        }
    }
    if (!found) {
        fprintf(stderr, "bonfyre-layer: no nodes matching '%s'\n", name_prefix);
        printf("Available nodes (first 20):\n");
        for (size_t i = 0; i < g.n_nodes && i < 20; i++)
            printf("  [%3zu] %-16s %s\n", i, g.nodes[i].op_type, g.nodes[i].name);
        bf_graph_free(&g); return 1;
    }

    char range_str[64];
    snprintf(range_str, sizeof(range_str), "%zu:%zu", first, last + 1);
    printf("[bonfyre-layer] matched %d nodes for '%s' → range %s\n",
           found, name_prefix, range_str);

    bf_graph_free(&g);
    return cmd_pull_layer(model_path, range_str, out_dir);
}

static int cmd_pack_transform(const char *name, char **parts, int n_parts,
                               const char *out_dir) {
    if (n_parts == 0) {
        fprintf(stderr, "bonfyre-layer pack-transform: no parts specified\n");
        return 1;
    }
    mkdirp(out_dir);

    /* Load all parts and concatenate their node+initializer raw bytes */
    WBuf graph_buf; wbuf_init(&graph_buf);
    size_t total_params = 0, total_nodes = 0;

    for (int pi = 0; pi < n_parts; pi++) {
        BfLayerGraph g;
        if (load_graph(parts[pi], &g)) {
            fprintf(stderr, "bonfyre-layer: failed to load part %s\n", parts[pi]);
            wbuf_free(&graph_buf); return 1;
        }
        /* Append all nodes */
        for (size_t i = 0; i < g.n_nodes; i++) {
            if (g.nodes[i].raw)
                wbuf_write_bytes(&graph_buf, 1, g.nodes[i].raw, g.nodes[i].raw_len);
        }
        /* Append all initializers */
        for (size_t i = 0; i < g.n_inits; i++) {
            total_params += g.inits[i].n_elements;
            if (g.inits[i].raw)
                wbuf_write_bytes(&graph_buf, 5, g.inits[i].raw, g.inits[i].raw_len);
        }
        total_nodes += g.n_nodes;
        printf("[bonfyre-layer] part %d: %s (%zu nodes)\n", pi+1, parts[pi], g.n_nodes);
        bf_graph_free(&g);
    }

    /* Wrap in ModelProto */
    WBuf model_buf; wbuf_init(&model_buf);
    wbuf_write_i64(&model_buf, 1, 8);
    WBuf opset_buf; wbuf_init(&opset_buf);
    wbuf_write_string(&opset_buf, 1, "");
    wbuf_write_i64(&opset_buf, 2, 17);
    wbuf_write_bytes(&model_buf, 8, opset_buf.data, opset_buf.len);
    wbuf_free(&opset_buf);
    wbuf_write_bytes(&model_buf, 7, graph_buf.data, graph_buf.len);
    wbuf_free(&graph_buf);

    char onnx_path[MAX_PATH], artifact_path[MAX_PATH];
    snprintf(onnx_path,     sizeof(onnx_path),     "%s/transform.onnx", out_dir);
    snprintf(artifact_path, sizeof(artifact_path), "%s/artifact.json",  out_dir);

    if (wbuf_save(&model_buf, onnx_path)) {
        wbuf_free(&model_buf); return 1;
    }
    wbuf_free(&model_buf);

    char sha[65]; sha256_file_hex(onnx_path, sha);
    char ts[32];  bf_iso_timestamp(ts, sizeof(ts));
    FILE *af = fopen(artifact_path, "w");
    if (af) {
        fprintf(af,
            "{\n"
            "  \"type\": \"transform_fragment\",\n"
            "  \"version\": \"1.0.0\",\n"
            "  \"name\": \"%s\",\n"
            "  \"n_parts\": %d,\n"
            "  \"n_nodes\": %zu,\n"
            "  \"n_params\": %zu,\n"
            "  \"sha256\": \"%s\",\n"
            "  \"fpq_sha256\": null,\n"
            "  \"format\": \"onnx\",\n"
            "  \"created_at\": \"%s\"\n"
            "}\n",
            name, n_parts, total_nodes, total_params, sha, ts);
        fclose(af);
    }

    printf("[bonfyre-layer] packed transform '%s'\n", name);
    printf("  parts:    %d\n", n_parts);
    printf("  nodes:    %zu\n", total_nodes);
    printf("  params:   %zu\n", total_params);
    printf("  sha256:   %.16s…\n", sha);
    printf("  onnx:     %s\n", onnx_path);
    printf("  artifact: %s\n\n", artifact_path);
    printf("  Next: python3 scripts/finalize_artifact.py %s\n", out_dir);
    printf("        bonfyre-model add %s\n", artifact_path);
    return 0;
}

static const char *SCHEMA_JSON =
    "{\n"
    "  \"$schema\": \"bonfyre-layer-target-v1\",\n"
    "  \"description\": \"Recipe target schema for layer-aware Bonfyre transforms.\",\n"
    "  \"variants\": {\n"
    "    \"full_model\":   { \"example\": { \"model\": \"hf://org/repo/model.onnx\" } },\n"
    "    \"layer_range\":  { \"example\": { \"model\": \"hf://org/repo\","
                            " \"target\": { \"type\": \"layer_range\", \"range\": \"0:3\" } } },\n"
    "    \"head\":         { \"example\": { \"model\": \"hf://org/repo\","
                            " \"target\": { \"type\": \"head\", \"name\": \"classifier\" } } },\n"
    "    \"slice\":        { \"example\": { \"model\": \"hf://org/repo\","
                            " \"target\": { \"type\": \"slice\", \"spec\": \"encoder.0-3,proj.out\" } } },\n"
    "    \"fragment\":     { \"example\": { \"model\": \"bonfyre://transform/name\","
                            " \"target\": { \"type\": \"fragment\" } } }\n"
    "  }\n"
    "}\n";

/* ═══════════════════════════════════════════════════════════════════
 * CLI dispatch
 * ═══════════════════════════════════════════════════════════════════ */

static void usage(void) {
    fprintf(stderr,
        "bonfyre-layer v" VERSION " — Layer-aware ONNX inspection and extraction\n\n"
        "Usage:\n"
        "  bonfyre-layer inspect      <model.onnx>\n"
        "  bonfyre-layer layers       <model.onnx>\n"
        "  bonfyre-layer pull-layer   <model.onnx> --range START:END --out DIR\n"
        "  bonfyre-layer pull-head    <model.onnx> --name PREFIX --out DIR\n"
        "  bonfyre-layer pack-transform <name> <p1.onnx> [p2.onnx ...] --out DIR\n"
        "  bonfyre-layer schema\n"
    );
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    const char *cmd = argv[1];

    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) { usage(); return 0; }
    if (strcmp(cmd, "--version") == 0) {
        printf("bonfyre-layer v" VERSION "\n"); return 0;
    }

    if (strcmp(cmd, "schema") == 0) { puts(SCHEMA_JSON); return 0; }

    if (strcmp(cmd, "inspect") == 0) {
        if (argc < 3) { fprintf(stderr,"usage: bonfyre-layer inspect <model.onnx>\n"); return 1; }
        return cmd_inspect(argv[2]);
    }

    if (strcmp(cmd, "layers") == 0) {
        if (argc < 3) { fprintf(stderr,"usage: bonfyre-layer layers <model.onnx>\n"); return 1; }
        return cmd_layers(argv[2]);
    }

    if (strcmp(cmd, "pull-layer") == 0) {
        if (argc < 3) { fprintf(stderr,"usage: bonfyre-layer pull-layer <model.onnx> --range S:E --out DIR\n"); return 1; }
        const char *model = argv[2];
        const char *range = bf_arg_value(argc, argv, "--range");
        const char *out   = bf_arg_value(argc, argv, "--out");
        if (!range || !out) {
            fprintf(stderr,"bonfyre-layer pull-layer: --range and --out required\n"); return 1;
        }
        return cmd_pull_layer(model, range, out);
    }

    if (strcmp(cmd, "pull-head") == 0) {
        if (argc < 3) { fprintf(stderr,"usage: bonfyre-layer pull-head <model.onnx> --name PREFIX --out DIR\n"); return 1; }
        const char *model = argv[2];
        const char *name  = bf_arg_value(argc, argv, "--name");
        const char *out   = bf_arg_value(argc, argv, "--out");
        if (!name || !out) {
            fprintf(stderr,"bonfyre-layer pull-head: --name and --out required\n"); return 1;
        }
        return cmd_pull_head(model, name, out);
    }

    if (strcmp(cmd, "pack-transform") == 0) {
        if (argc < 4) {
            fprintf(stderr,"usage: bonfyre-layer pack-transform <name> <p1.onnx> ... --out DIR\n");
            return 1;
        }
        const char *name  = argv[2];
        const char *out   = bf_arg_value(argc, argv, "--out");
        if (!out) { fprintf(stderr,"bonfyre-layer pack-transform: --out required\n"); return 1; }
        /* Collect parts: argv[3..] except --out and its value */
        char *parts[64]; int n_parts = 0;
        for (int i = 3; i < argc && n_parts < 64; i++) {
            if (strcmp(argv[i], "--out") == 0) { i++; continue; }
            if (argv[i][0] == '-') continue;
            parts[n_parts++] = argv[i];
        }
        return cmd_pack_transform(name, parts, n_parts, out);
    }

    fprintf(stderr, "bonfyre-layer: unknown command '%s'\n", cmd);
    usage(); return 1;
}
