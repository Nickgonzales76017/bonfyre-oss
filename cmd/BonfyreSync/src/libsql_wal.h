/*
 * libsql_wal.h — LibSQL embedded-replica WAL frame sync protocol
 *
 * Implements the "cloud-optional" sync layer for Bonfyre control databases.
 * A primary node broadcasts WAL frames over a local TCP/Unix socket; replicas
 * subscribe and apply frames locally — no cloud dependency, but cross-node
 * sync for fleet scenarios (dev workstation ↔ edge device ↔ CI server).
 *
 * Philosophy: the OS owns the primary database; this sync layer is additive.
 * If the socket is not available, bonfyre-control falls back to local-only
 * mode silently. Replication is best-effort, not required.
 *
 * Protocol wire format:
 *   [LSWAL magic 4B] [frame_count u32] n × {
 *     [pgno u32] [db_size u32] [salt1 u32] [salt2 u32]
 *     [checksum1 u32] [checksum2 u32] [page_data PAGE_SIZE B]
 *   }
 *
 * This mirrors SQLite's WAL frame header (PRAGMA journal_mode=WAL) so that
 * frames written here can be directly applied to a replica WAL file.
 *
 * Usage:
 *   Primary:  bonfyre-sync wal-serve --db control.db --port 7832
 *   Replica:  bonfyre-sync wal-pull  --primary 10.0.0.1:7832 --db control.db
 *
 * See libsql_wal.c for implementation.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>

#define LSWAL_MAGIC     0x4C53574C  /* "LSWL" big-endian */
#define LSWAL_PAGE_SIZE 4096
#define LSWAL_PORT_DEFAULT 7832

/* One WAL frame header: matches SQLite WAL-frame-header layout */
typedef struct {
    uint32_t pgno;        /* page number (1-based) */
    uint32_t db_size;     /* size of database file in pages after commit (0 = non-commit) */
    uint32_t salt1;       /* copy of WAL salt-1 */
    uint32_t salt2;       /* copy of WAL salt-2 */
    uint32_t checksum1;   /* cumulative checksum through this frame */
    uint32_t checksum2;
} LswalFrameHdr;

/* Result codes */
#define LSWAL_OK         0
#define LSWAL_ERR_MAGIC  1   /* bad magic on receive */
#define LSWAL_ERR_IO     2   /* socket / file I/O error */
#define LSWAL_ERR_MEM    3   /* allocation failure */
#define LSWAL_ERR_PROTO  4   /* protocol violation */
#define LSWAL_ERR_NOWAL  5   /* source database has no WAL (not in WAL mode) */

/* Callbacks for frame delivery */
typedef int (*lswal_frame_cb)(const LswalFrameHdr *hdr,
                              const uint8_t *page_data,
                              void *userdata);

/* ── Primary (serve) side ──────────────────────────────────────────────── */

/*
 * Open the WAL file associated with db_path, collect all frames committed
 * since last_frame (0 = all frames), send them over fd in LSWAL wire format.
 * Returns LSWAL_OK or an error code.
 */
int lswal_send_frames(int fd, const char *db_path, uint32_t last_frame);

/*
 * Serve loop: listen on port, accept connections, send frames on demand.
 * Runs until SIGINT/SIGTERM. Calls lswal_send_frames for each connection.
 * Returns only on error.
 */
int lswal_serve(const char *db_path, uint16_t port);

/* ── Replica (pull) side ───────────────────────────────────────────────── */

/*
 * Connect to primary at host:port.  Receive all frames since last_frame.
 * Calls cb for each frame received.  Closes the connection when done.
 * Returns LSWAL_OK or error code.
 */
int lswal_pull_frames(const char *host, uint16_t port, uint32_t last_frame,
                      lswal_frame_cb cb, void *userdata);

/*
 * High-level: pull frames from primary and apply them to the replica WAL file
 * (db_path + "-wal").  Handles file creation, frame validation, and checksum
 * verification.  Safe to call while db_path is open read-only on the replica.
 * Returns LSWAL_OK or error.
 */
int lswal_replicate(const char *primary_host, uint16_t primary_port,
                    const char *db_path);

/* ── Utility ───────────────────────────────────────────────────────────── */

/* Return the last committed frame number in the WAL for db_path (0 if none) */
uint32_t lswal_last_frame(const char *db_path);

/* Human-readable error string */
const char *lswal_strerror(int code);
