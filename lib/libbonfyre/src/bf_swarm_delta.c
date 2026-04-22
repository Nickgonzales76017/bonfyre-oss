/*
 * bf_swarm_delta.c — federated transform graph delta gossip
 *
 * Implements the BfSwarmDelta wire protocol: structured binary deltas
 * describing newly-promoted stable graph entries, broadcast over TCP to
 * all configured swarm peers and merged using an F1-improvement gate.
 *
 * Network topology: every node is both a client (broadcasts after promote)
 * and a server (UDP listener for incoming deltas).  The gossip is purely
 * push-based: each node sends to ALL known peers after a local promotion.
 * Anti-entropy full-graph sync is out of scope (planned for Cycle 10).
 */

#define _POSIX_C_SOURCE 200809L
#include "bf_swarm_delta.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ───────────────────────────────────────────────────────────────────────────
 * Little-endian I/O helpers
 * ─────────────────────────────────────────────────────────────────────────── */

static void put_u32le(uint8_t *p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v >> 8) & 0xff;
    p[2] = (v >> 16) & 0xff; p[3] = (v >> 24) & 0xff;
}
static void put_u64le(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) { p[i] = v & 0xff; v >>= 8; }
}
static uint32_t get_u32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t get_u64le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static void put_f32le(uint8_t *p, float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    put_u32le(p, bits);
}
static float get_f32le(const uint8_t *p) {
    uint32_t bits = get_u32le(p);
    float f; memcpy(&f, &bits, 4);
    return f;
}

/* ───────────────────────────────────────────────────────────────────────────
 * Local graph
 * ─────────────────────────────────────────────────────────────────────────── */

void bf_local_graph_init(BfLocalGraph *g) {
    if (!g) return;
    memset(g, 0, sizeof(*g));
}

int bf_local_graph_upsert(BfLocalGraph *g, const BfLocalGraphNode *node) {
    if (!g || !node) return -1;
    /* search for existing */
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].family_id == node->family_id) {
            g->nodes[i] = *node;
            return 0;
        }
    }
    if (g->n_nodes >= BF_LOCAL_GRAPH_MAX) return -1;
    g->nodes[g->n_nodes++] = *node;
    return 0;
}

BfLocalGraphNode *bf_local_graph_lookup(BfLocalGraph *g, uint32_t family_id) {
    if (!g) return NULL;
    for (int i = 0; i < g->n_nodes; i++)
        if (g->nodes[i].family_id == family_id) return &g->nodes[i];
    return NULL;
}

/* ───────────────────────────────────────────────────────────────────────────
 * Delta construction
 * ─────────────────────────────────────────────────────────────────────────── */

void bf_swarm_delta_from_graph(const BfLocalGraph  *g,
                                uint32_t             since_sample_count,
                                const char          *origin_host,
                                BfSwarmDelta        *out) {
    if (!g || !out) return;
    memset(out, 0, sizeof(*out));
    out->magic     = BF_SWARM_DELTA_MAGIC;
    out->version   = BF_SWARM_DELTA_VERSION;
    out->timestamp = (uint64_t)time(NULL);
    if (origin_host)
        strncpy(out->origin_host, origin_host, sizeof(out->origin_host) - 1);

    for (int i = 0; i < g->n_nodes && out->n_entries < BF_SWARM_DELTA_MAX_ENTRIES; i++) {
        const BfLocalGraphNode *n = &g->nodes[i];
        if (n->sample_count <= since_sample_count) continue;
        BfSwarmDeltaEntry *e = &out->entries[out->n_entries++];
        e->family_id    = n->family_id;
        strncpy(e->geometry, n->geometry, sizeof(e->geometry) - 1);
        e->mean_f1      = n->mean_f1;
        e->failure_rate = n->failure_rate;
        e->sample_count = n->sample_count;
    }
}

/* ───────────────────────────────────────────────────────────────────────────
 * Serialisation
 * ─────────────────────────────────────────────────────────────────────────── */

/*
 * Wire layout:
 *   0:  4B magic
 *   4:  4B version
 *   8:  8B timestamp (LE uint64)
 *   16: 4B node_count
 *   20: 4B origin_len
 *   24: origin_len bytes origin_host
 *   24+origin_len: node_count × 96-byte BfSwarmDeltaEntry
 */
#define ENTRY_WIRE_SIZE 96

int bf_swarm_delta_serialize(const BfSwarmDelta *delta, uint8_t *buf, size_t buflen) {
    if (!delta || !buf) return -1;
    uint32_t origin_len = (uint32_t)strlen(delta->origin_host);
    size_t need = 24 + origin_len + (size_t)delta->n_entries * ENTRY_WIRE_SIZE;
    if (buflen < need) return -1;

    uint8_t *p = buf;
    put_u32le(p, delta->magic);     p += 4;
    put_u32le(p, delta->version);   p += 4;
    put_u64le(p, delta->timestamp); p += 8;
    put_u32le(p, (uint32_t)delta->n_entries); p += 4;
    put_u32le(p, origin_len);       p += 4;
    memcpy(p, delta->origin_host, origin_len); p += origin_len;

    for (int i = 0; i < delta->n_entries; i++) {
        const BfSwarmDeltaEntry *e = &delta->entries[i];
        uint8_t eb[ENTRY_WIRE_SIZE]; memset(eb, 0, sizeof(eb));
        put_u32le(eb,    e->family_id);
        memcpy(eb + 4,   e->geometry, 64);
        put_f32le(eb+68, e->mean_f1);
        put_f32le(eb+72, e->failure_rate);
        put_u32le(eb+76, e->sample_count);
        memcpy(p, eb, ENTRY_WIRE_SIZE);
        p += ENTRY_WIRE_SIZE;
    }
    return (int)(p - buf);
}

int bf_swarm_delta_deserialize(const uint8_t *buf, size_t len, BfSwarmDelta *out) {
    if (!buf || !out || len < 24) return -1;
    const uint8_t *p = buf;

    uint32_t magic = get_u32le(p); p += 4;
    if (magic != BF_SWARM_DELTA_MAGIC) return -1;

    uint32_t version = get_u32le(p); p += 4;
    if (version != BF_SWARM_DELTA_VERSION) return -1;

    memset(out, 0, sizeof(*out));
    out->magic   = magic;
    out->version = version;
    out->timestamp = get_u64le(p); p += 8;

    uint32_t n_entries  = get_u32le(p); p += 4;
    uint32_t origin_len = get_u32le(p); p += 4;

    if (n_entries > BF_SWARM_DELTA_MAX_ENTRIES) return -1;
    if ((size_t)(p - buf) + origin_len + n_entries * ENTRY_WIRE_SIZE > len) return -1;

    if (origin_len >= sizeof(out->origin_host)) return -1;
    memcpy(out->origin_host, p, origin_len); p += origin_len;
    out->origin_host[origin_len] = '\0';

    out->n_entries = (int)n_entries;
    for (int i = 0; i < out->n_entries; i++) {
        BfSwarmDeltaEntry *e = &out->entries[i];
        const uint8_t *ep = p + (size_t)i * ENTRY_WIRE_SIZE;
        e->family_id    = get_u32le(ep);
        memcpy(e->geometry, ep + 4, 64);
        e->geometry[63] = '\0';
        e->mean_f1      = get_f32le(ep + 68);
        e->failure_rate = get_f32le(ep + 72);
        e->sample_count = get_u32le(ep + 76);
    }
    return 0;
}

/* ───────────────────────────────────────────────────────────────────────────
 * Merge
 * ─────────────────────────────────────────────────────────────────────────── */

int bf_swarm_delta_merge(BfLocalGraph       *graph,
                          const BfSwarmDelta *delta,
                          float               improvement_threshold) {
    if (!graph || !delta) return 0;
    int accepted = 0;
    for (int i = 0; i < delta->n_entries; i++) {
        const BfSwarmDeltaEntry *e = &delta->entries[i];
        BfLocalGraphNode *local = bf_local_graph_lookup(graph, e->family_id);
        float local_f1 = local ? local->mean_f1 : 0.0f;
        if (e->mean_f1 > local_f1 + improvement_threshold) {
            BfLocalGraphNode updated;
            updated.family_id    = e->family_id;
            strncpy(updated.geometry, e->geometry, sizeof(updated.geometry) - 1);
            updated.geometry[sizeof(updated.geometry)-1] = '\0';
            updated.mean_f1      = e->mean_f1;
            updated.failure_rate = e->failure_rate;
            updated.sample_count = e->sample_count;
            bf_local_graph_upsert(graph, &updated);
            accepted++;
            fprintf(stderr,
                "[bf_swarm_delta] accepted family=%08x  f1 %.3f -> %.3f  from %s\n",
                e->family_id, local_f1, e->mean_f1, delta->origin_host);
        }
    }
    return accepted;
}

/* ───────────────────────────────────────────────────────────────────────────
 * Broadcast (TCP, length-prefixed)
 * ─────────────────────────────────────────────────────────────────────────── */

static void peers_path(char *buf, size_t len) {
    const char *e = getenv("BONFYRE_SWARM_PEERS");
    if (e) { snprintf(buf, len, "%s", e); return; }
    const char *home = getenv("HOME"); if (!home) home = "/tmp";
    snprintf(buf, len, "%s/.local/share/bonfyre/swarm-peers.txt", home);
}

int bf_swarm_delta_broadcast(const BfSwarmDelta *delta) {
    if (!delta || delta->n_entries == 0) return 0;

    /* Serialise */
    static uint8_t wire[65536];
    int wlen = bf_swarm_delta_serialize(delta, wire, sizeof(wire));
    if (wlen <= 0) { fprintf(stderr, "bf_swarm_delta: serialise failed\n"); return 0; }

    /* Load peer list */
    char path[4096];
    peers_path(path, sizeof(path));
    FILE *pf = fopen(path, "r");
    if (!pf) {
        fprintf(stderr, "bf_swarm_delta: no peers file (%s)\n", path);
        return 0;
    }

    int sent = 0;
    char line[512];
    while (fgets(line, sizeof(line), pf)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        char host[256]; int port = BF_SWARM_DELTA_DEFAULT_PORT;
        char *colon = strrchr(p, ':');
        if (colon) {
            *colon = '\0';
            int pp = atoi(colon + 1);
            if (pp > 0 && pp < 65536) port = pp;
        }
        size_t hlen = strlen(p);
        while (hlen > 0 && (p[hlen-1] == '\r' || p[hlen-1] == '\n' ||
                             p[hlen-1] == ' '  || p[hlen-1] == '\t'))
            p[--hlen] = '\0';
        snprintf(host, sizeof(host), "%s", p);
        if (!host[0]) continue;

        struct addrinfo hints, *res;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family   = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        char portstr[16]; snprintf(portstr, sizeof(portstr), "%d", port);
        if (getaddrinfo(host, portstr, &hints, &res) != 0) {
            fprintf(stderr, "bf_swarm_delta: getaddrinfo failed for %s\n", host);
            continue;
        }
        int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd < 0) { freeaddrinfo(res); continue; }
        struct timeval tv = { 3, 0 };
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
            close(fd); freeaddrinfo(res); continue;
        }
        freeaddrinfo(res);

        /* Send length-prefixed wire packet */
        uint32_t nlen = htonl((uint32_t)wlen);
        write(fd, &nlen, 4);
        write(fd, wire, (size_t)wlen);
        close(fd);
        sent++;
    }
    fclose(pf);
    return sent;
}

/* ───────────────────────────────────────────────────────────────────────────
 * UDP listener (serve)
 * ─────────────────────────────────────────────────────────────────────────── */

static volatile int g_swarm_stop = 0;
static void swarm_sig(int s) { (void)s; g_swarm_stop = 1; }

int bf_swarm_delta_serve(int port, BfLocalGraph *graph,
                          BfDeltaAcceptCb accept_cb, void *userdata) {
    if (!graph) return -1;
    if (port <= 0) port = BF_SWARM_DELTA_DEFAULT_PORT;

    int fd = socket(AF_INET6, SOCK_DGRAM, 0);
    if (fd < 0) fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) { perror("bf_swarm_delta: socket"); return -1; }

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in6 addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin6_family = AF_INET6;
    addr.sin6_port   = htons((uint16_t)port);
    addr.sin6_addr   = in6addr_any;
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        fd = socket(AF_INET, SOCK_DGRAM, 0);
        struct sockaddr_in a4; memset(&a4, 0, sizeof(a4));
        a4.sin_family = AF_INET; a4.sin_port = htons((uint16_t)port);
        a4.sin_addr.s_addr = INADDR_ANY;
        if (bind(fd, (struct sockaddr *)&a4, sizeof(a4)) < 0) {
            perror("bf_swarm_delta: bind"); close(fd); return -1;
        }
    }

    signal(SIGINT,  swarm_sig);
    signal(SIGTERM, swarm_sig);

    fprintf(stderr, "[bf_swarm_delta] listening UDP port %d\n", port);

    static uint8_t dgram[65536];
    while (!g_swarm_stop) {
        struct sockaddr_storage peer; socklen_t plen = sizeof(peer);
        ssize_t n = recvfrom(fd, dgram, sizeof(dgram), 0,
                             (struct sockaddr *)&peer, &plen);
        if (n <= 0) {
            if (errno == EINTR) continue;
            break;
        }

        BfSwarmDelta incoming;
        if (bf_swarm_delta_deserialize(dgram, (size_t)n, &incoming) < 0) {
            fprintf(stderr, "[bf_swarm_delta] bad packet (%zd bytes)\n", n);
            continue;
        }

        int accepted = bf_swarm_delta_merge(graph, &incoming,
                                             BF_SWARM_IMPROVEMENT_THRESHOLD);
        if (accepted > 0 && accept_cb)
            accept_cb(&incoming, accepted, userdata);
    }
    close(fd);
    return 0;
}
