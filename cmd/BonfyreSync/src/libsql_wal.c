/*
 * libsql_wal.c — LibSQL embedded-replica WAL frame sync implementation
 *
 * Reads the SQLite WAL file format directly (no SQLite API calls needed
 * for frame extraction).  Sends/receives frames over a plain TCP connection.
 * Applies received frames by appending to the replica's WAL file — SQLite's
 * WAL recovery logic handles consistency on the next open.
 *
 * SQLite WAL file layout reference:
 *   https://www.sqlite.org/walformat.html
 *   File header: 32 bytes
 *   Frame header: 24 bytes (LswalFrameHdr fields in big-endian)
 *   Page data:  page_size bytes (default 4096)
 *
 * This implementation is compatible with SQLite 3.x WAL-mode databases
 * and therefore with libSQL databases (which use the same WAL format for
 * the embedded-replica wire protocol).
 */
#define _POSIX_C_SOURCE 200809L

#include "libsql_wal.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

/* ── WAL file format constants ─────────────────────────────────────────── */

#define WAL_MAGIC_BE      0x377f0682u   /* big-endian WAL magic */
#define WAL_MAGIC_LE      0x377f0683u   /* little-endian WAL magic */
#define WAL_HDR_SIZE      32
#define WAL_FRAME_HDR_SZ  24

/* ── Internal helpers ───────────────────────────────────────────────────── */

static uint32_t read_u32_be(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] <<  8) |  (uint32_t)p[3];
}

static void write_u32_be(uint8_t *p, uint32_t v) {
    p[0] = (v >> 24) & 0xff;
    p[1] = (v >> 16) & 0xff;
    p[2] = (v >>  8) & 0xff;
    p[3] =  v        & 0xff;
}

static int write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return -1;
        p += w; n -= (size_t)w;
    }
    return 0;
}

static int read_all(int fd, void *buf, size_t n) {
    uint8_t *p = buf;
    while (n > 0) {
        ssize_t r = read(fd, p, n);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}

/* ── WAL file open + parse ──────────────────────────────────────────────── */

typedef struct {
    FILE       *f;
    uint32_t    page_size;
    uint32_t    n_frames;   /* total frames in WAL */
    /* salt, ckpt etc. from WAL header */
} WalReader;

static int wal_open(const char *db_path, WalReader *wr) {
    char wal_path[4096];
    snprintf(wal_path, sizeof(wal_path), "%s-wal", db_path);

    wr->f = fopen(wal_path, "rb");
    if (!wr->f) return LSWAL_ERR_IO;

    uint8_t hdr[WAL_HDR_SIZE];
    if (fread(hdr, 1, WAL_HDR_SIZE, wr->f) != WAL_HDR_SIZE) {
        fclose(wr->f); return LSWAL_ERR_IO;
    }

    uint32_t magic = read_u32_be(hdr);
    if (magic != WAL_MAGIC_BE && magic != WAL_MAGIC_LE) {
        fclose(wr->f); return LSWAL_ERR_MAGIC;
    }

    wr->page_size = read_u32_be(hdr + 8);
    if (wr->page_size < 512) wr->page_size = LSWAL_PAGE_SIZE;

    /* compute frame count from file size */
    fseek(wr->f, 0, SEEK_END);
    long fsz = ftell(wr->f);
    long frames_region = fsz - WAL_HDR_SIZE;
    long frame_sz = WAL_FRAME_HDR_SZ + (long)wr->page_size;
    wr->n_frames = (frames_region > 0 && frame_sz > 0)
                   ? (uint32_t)(frames_region / frame_sz) : 0;

    return LSWAL_OK;
}

static int wal_read_frame(WalReader *wr, uint32_t frame_no,
                          LswalFrameHdr *hdr_out, uint8_t *page_buf) {
    /* frame_no is 1-based */
    long frame_sz = WAL_FRAME_HDR_SZ + (long)wr->page_size;
    long offset = WAL_HDR_SIZE + (long)(frame_no - 1) * frame_sz;

    if (fseek(wr->f, offset, SEEK_SET) != 0) return LSWAL_ERR_IO;

    uint8_t raw_hdr[WAL_FRAME_HDR_SZ];
    if (fread(raw_hdr, 1, WAL_FRAME_HDR_SZ, wr->f) != WAL_FRAME_HDR_SZ)
        return LSWAL_ERR_IO;

    hdr_out->pgno       = read_u32_be(raw_hdr + 0);
    hdr_out->db_size    = read_u32_be(raw_hdr + 4);
    hdr_out->salt1      = read_u32_be(raw_hdr + 8);
    hdr_out->salt2      = read_u32_be(raw_hdr + 12);
    hdr_out->checksum1  = read_u32_be(raw_hdr + 16);
    hdr_out->checksum2  = read_u32_be(raw_hdr + 20);

    if (fread(page_buf, 1, wr->page_size, wr->f) != wr->page_size)
        return LSWAL_ERR_IO;

    return LSWAL_OK;
}

static void wal_close(WalReader *wr) {
    if (wr->f) { fclose(wr->f); wr->f = NULL; }
}

/* ── Wire: send frames to a connected fd ────────────────────────────────── */

int lswal_send_frames(int fd, const char *db_path, uint32_t last_frame) {
    WalReader wr = {0};
    int rc = wal_open(db_path, &wr);
    if (rc != LSWAL_OK) return rc;

    uint32_t first = last_frame + 1;
    uint32_t count = (wr.n_frames >= first) ? (wr.n_frames - first + 1) : 0;

    /* Write wire header: magic + frame_count */
    uint8_t wire_hdr[8];
    write_u32_be(wire_hdr, LSWAL_MAGIC);
    write_u32_be(wire_hdr + 4, count);
    if (write_all(fd, wire_hdr, 8) != 0) { wal_close(&wr); return LSWAL_ERR_IO; }

    uint8_t *page_buf = malloc(wr.page_size);
    if (!page_buf) { wal_close(&wr); return LSWAL_ERR_MEM; }

    for (uint32_t f = first; f <= wr.n_frames; f++) {
        LswalFrameHdr fhdr;
        rc = wal_read_frame(&wr, f, &fhdr, page_buf);
        if (rc != LSWAL_OK) break;

        /* Serialise frame header (big-endian) */
        uint8_t raw[WAL_FRAME_HDR_SZ];
        write_u32_be(raw + 0,  fhdr.pgno);
        write_u32_be(raw + 4,  fhdr.db_size);
        write_u32_be(raw + 8,  fhdr.salt1);
        write_u32_be(raw + 12, fhdr.salt2);
        write_u32_be(raw + 16, fhdr.checksum1);
        write_u32_be(raw + 20, fhdr.checksum2);

        if (write_all(fd, raw, WAL_FRAME_HDR_SZ) != 0 ||
            write_all(fd, page_buf, wr.page_size) != 0) {
            rc = LSWAL_ERR_IO; break;
        }
    }

    free(page_buf);
    wal_close(&wr);
    return rc;
}

/* ── Primary: serve loop ─────────────────────────────────────────────────── */

int lswal_serve(const char *db_path, uint16_t port) {
    signal(SIGPIPE, SIG_IGN);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return LSWAL_ERR_IO;

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(srv, 4) != 0) {
        close(srv); return LSWAL_ERR_IO;
    }

    fprintf(stderr, "[lswal] serving %s on port %u\n", db_path, port);

    for (;;) {
        struct sockaddr_in caddr;
        socklen_t clen = sizeof(caddr);
        int cfd = accept(srv, (struct sockaddr *)&caddr, &clen);
        if (cfd < 0) continue;

        /* Read last_frame from client (4 bytes big-endian) */
        uint8_t req[4] = {0};
        read_all(cfd, req, 4);
        uint32_t last = read_u32_be(req);

        lswal_send_frames(cfd, db_path, last);
        close(cfd);
    }
    close(srv);
    return LSWAL_OK;
}

/* ── Replica: pull frames from primary ──────────────────────────────────── */

int lswal_pull_frames(const char *host, uint16_t port, uint32_t last_frame,
                      lswal_frame_cb cb, void *userdata) {
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char portstr[8];
    snprintf(portstr, sizeof(portstr), "%u", port);

    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res)
        return LSWAL_ERR_IO;

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return LSWAL_ERR_IO; }

    if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        freeaddrinfo(res); close(fd); return LSWAL_ERR_IO;
    }
    freeaddrinfo(res);

    /* Send last_frame */
    uint8_t req[4];
    write_u32_be(req, last_frame);
    write_all(fd, req, 4);

    /* Read wire header */
    uint8_t wire_hdr[8];
    if (read_all(fd, wire_hdr, 8) != 0) { close(fd); return LSWAL_ERR_IO; }

    uint32_t magic  = read_u32_be(wire_hdr);
    uint32_t fcount = read_u32_be(wire_hdr + 4);
    if (magic != LSWAL_MAGIC) { close(fd); return LSWAL_ERR_MAGIC; }

    uint8_t *page_buf = malloc(LSWAL_PAGE_SIZE);
    if (!page_buf) { close(fd); return LSWAL_ERR_MEM; }

    int rc = LSWAL_OK;
    for (uint32_t f = 0; f < fcount; f++) {
        uint8_t raw[WAL_FRAME_HDR_SZ];
        if (read_all(fd, raw, WAL_FRAME_HDR_SZ) != 0 ||
            read_all(fd, page_buf, LSWAL_PAGE_SIZE) != 0) {
            rc = LSWAL_ERR_IO; break;
        }

        LswalFrameHdr hdr;
        hdr.pgno      = read_u32_be(raw + 0);
        hdr.db_size   = read_u32_be(raw + 4);
        hdr.salt1     = read_u32_be(raw + 8);
        hdr.salt2     = read_u32_be(raw + 12);
        hdr.checksum1 = read_u32_be(raw + 16);
        hdr.checksum2 = read_u32_be(raw + 20);

        if (cb && cb(&hdr, page_buf, userdata) != 0) { rc = LSWAL_ERR_PROTO; break; }
    }

    free(page_buf);
    close(fd);
    return rc;
}

/* ── High-level: replicate to local WAL file ────────────────────────────── */

typedef struct { FILE *wal_f; uint32_t n_applied; } ReplicaCtx;

static int apply_frame_cb(const LswalFrameHdr *hdr, const uint8_t *data, void *ud) {
    ReplicaCtx *ctx = ud;
    uint8_t raw[WAL_FRAME_HDR_SZ];
    write_u32_be(raw + 0,  hdr->pgno);
    write_u32_be(raw + 4,  hdr->db_size);
    write_u32_be(raw + 8,  hdr->salt1);
    write_u32_be(raw + 12, hdr->salt2);
    write_u32_be(raw + 16, hdr->checksum1);
    write_u32_be(raw + 20, hdr->checksum2);
    if (fwrite(raw, 1, WAL_FRAME_HDR_SZ, ctx->wal_f) != WAL_FRAME_HDR_SZ) return -1;
    if (fwrite(data, 1, LSWAL_PAGE_SIZE, ctx->wal_f) != LSWAL_PAGE_SIZE) return -1;
    ctx->n_applied++;
    return 0;
}

int lswal_replicate(const char *primary_host, uint16_t primary_port,
                    const char *db_path) {
    uint32_t last = lswal_last_frame(db_path);

    char wal_path[4096];
    snprintf(wal_path, sizeof(wal_path), "%s-wal", db_path);

    FILE *wal_f = fopen(wal_path, "ab");
    if (!wal_f) return LSWAL_ERR_IO;

    ReplicaCtx ctx = { wal_f, 0 };
    int rc = lswal_pull_frames(primary_host, primary_port, last,
                               apply_frame_cb, &ctx);

    fflush(wal_f);
    fclose(wal_f);

    if (rc == LSWAL_OK)
        fprintf(stderr, "[lswal] applied %u frame(s) from %s:%u\n",
                ctx.n_applied, primary_host, primary_port);

    return rc;
}

/* ── Utility ─────────────────────────────────────────────────────────────── */

uint32_t lswal_last_frame(const char *db_path) {
    WalReader wr = {0};
    if (wal_open(db_path, &wr) != LSWAL_OK) return 0;
    uint32_t n = wr.n_frames;
    wal_close(&wr);
    return n;
}

const char *lswal_strerror(int code) {
    switch (code) {
        case LSWAL_OK:        return "ok";
        case LSWAL_ERR_MAGIC: return "bad WAL magic";
        case LSWAL_ERR_IO:    return "I/O error";
        case LSWAL_ERR_MEM:   return "out of memory";
        case LSWAL_ERR_PROTO: return "protocol error";
        case LSWAL_ERR_NOWAL: return "no WAL file (not in WAL mode)";
        default:              return "unknown error";
    }
}
