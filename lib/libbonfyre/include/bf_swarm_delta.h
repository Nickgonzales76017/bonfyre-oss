/*
 * bf_swarm_delta.h — federated transform graph delta gossip protocol
 *
 * Each Bonfyre node maintains a "stable graph" — a table of
 * (family_id → geometry, mean_f1, failure_rate) that describes the
 * best-known execution strategy for each artifact family.
 *
 * When a node promotes a new winner, it broadcasts a BfSwarmDelta to all
 * peers.  Peers apply it if the incoming mean_f1 exceeds local by a
 * configurable improvement threshold.
 *
 * Wire format (4-byte aligned, little-endian):
 *   [0]  uint32  magic      = 0x42465344  ('BFSD')
 *   [4]  uint32  version    = 1
 *   [8]  uint64  timestamp  (UNIX seconds, UTC)
 *   [16] uint32  node_count — number of BfSwarmDeltaEntry records
 *   [20] uint32  origin_len — length of origin_host string
 *   [24] char[]  origin_host (origin_len bytes, not NUL-terminated in wire)
 *   [24+origin_len] BfSwarmDeltaEntry[node_count]
 *
 * BfSwarmDeltaEntry (96 bytes):
 *   [0]  uint32  family_id
 *   [4]  char[64] geometry    (new best geometry descriptor)
 *   [68] float   mean_f1
 *   [72] float   failure_rate
 *   [76] uint32  sample_count
 *   [80] uint32  reserved
 *   [84..95 padding to 96]
 */
#pragma once
#ifndef BF_SWARM_DELTA_H
#define BF_SWARM_DELTA_H

#include <stdint.h>
#include <stddef.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BF_SWARM_DELTA_MAGIC    0x42465344u  /* 'BFSD' */
#define BF_SWARM_DELTA_VERSION  1u
#define BF_SWARM_DELTA_MAX_ENTRIES 256
#define BF_SWARM_DELTA_DEFAULT_PORT 9321

/* Improvement threshold: accept incoming update only if
 * remote_mean_f1 > local_mean_f1 + BF_SWARM_IMPROVEMENT_THRESHOLD */
#define BF_SWARM_IMPROVEMENT_THRESHOLD 0.02f

/* ── Entry ────────────────────────────────────────────────────────────────── */

typedef struct __attribute__((packed)) {
    uint32_t family_id;      /* FNV-1a-64 truncated to 32 bits */
    char     geometry[64];   /* geometry descriptor string      */
    float    mean_f1;        /* cumulative mean F1 score        */
    float    failure_rate;   /* fraction of runs that failed    */
    uint32_t sample_count;   /* number of runs this covers      */
    uint32_t reserved;
    /* 8 bytes padding to reach 96 */
    uint8_t  _pad[8];
} BfSwarmDeltaEntry;          /* = 96 bytes                      */

/* ── Delta ────────────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t          magic;
    uint32_t          version;
    uint64_t          timestamp;
    char              origin_host[256];
    int               n_entries;
    BfSwarmDeltaEntry entries[BF_SWARM_DELTA_MAX_ENTRIES];
} BfSwarmDelta;

/* ── Local graph node (one row in the stable graph) ──────────────────────── */

typedef struct {
    uint32_t family_id;
    char     geometry[64];
    float    mean_f1;
    float    failure_rate;
    uint32_t sample_count;
} BfLocalGraphNode;

#define BF_LOCAL_GRAPH_MAX 1024

typedef struct {
    BfLocalGraphNode nodes[BF_LOCAL_GRAPH_MAX];
    int              n_nodes;
} BfLocalGraph;

/* ── Callbacks ────────────────────────────────────────────────────────────── */

/* Called by bf_swarm_delta_serve() when a delta arrives and is accepted.
 * The delta has already been merged into local_graph before the callback fires. */
typedef void (*BfDeltaAcceptCb)(const BfSwarmDelta *delta, int n_accepted,
                                 void *userdata);

/* ── API ──────────────────────────────────────────────────────────────────── */

/* Initialise an empty local graph. */
void bf_local_graph_init(BfLocalGraph *g);

/* Upsert a node in the local graph.
 * If family_id exists: updates geometry/mean_f1/failure_rate/sample_count.
 * If not: appends a new node (up to BF_LOCAL_GRAPH_MAX).
 * Returns 0 on success, -1 if graph is full. */
int bf_local_graph_upsert(BfLocalGraph *g, const BfLocalGraphNode *node);

/* Look up a node by family_id.  Returns pointer into graph or NULL. */
BfLocalGraphNode *bf_local_graph_lookup(BfLocalGraph *g, uint32_t family_id);

/* ── Delta construction ───────────────────────────────────────────────────── */

/* Build a delta from dirty (recently promoted) nodes in the local graph.
 *   since_sample_count: include nodes whose sample_count > since_sample_count
 *   A delta with n_entries == 0 should not be broadcast. */
void bf_swarm_delta_from_graph(const BfLocalGraph  *g,
                                uint32_t             since_sample_count,
                                const char          *origin_host,
                                BfSwarmDelta        *out);

/* ── Serialisation ────────────────────────────────────────────────────────── */

/* Serialise delta to buf.  Returns bytes written or -1 if buf too small. */
int bf_swarm_delta_serialize(const BfSwarmDelta *delta, uint8_t *buf, size_t buflen);

/* Deserialise from buf.  Returns 0 on success, -1 on error (bad magic/version). */
int bf_swarm_delta_deserialize(const uint8_t *buf, size_t len, BfSwarmDelta *out);

/* ── Network ──────────────────────────────────────────────────────────────── */

/* Broadcast delta to all peers in ~./local/share/bonfyre/swarm-peers.txt.
 * Uses TCP.  Logs successes and failures to stderr.
 * Returns number of peers successfully notified. */
int bf_swarm_delta_broadcast(const BfSwarmDelta *delta);

/* UDP server: listen on port, deserialise arriving deltas, merge into graph,
 * call accept_cb for each accepted delta.  Blocks until SIGINT/SIGTERM.
 * Returns -1 on socket error, 0 if stopped cleanly. */
int bf_swarm_delta_serve(int port, BfLocalGraph *graph,
                          BfDeltaAcceptCb accept_cb, void *userdata);

/* Merge an incoming delta into the local graph.
 * For each entry: if incoming mean_f1 > local + threshold, accept and update.
 * Returns number of entries accepted. */
int bf_swarm_delta_merge(BfLocalGraph       *graph,
                          const BfSwarmDelta *delta,
                          float               improvement_threshold);

#ifdef __cplusplus
}
#endif
#endif /* BF_SWARM_DELTA_H */
