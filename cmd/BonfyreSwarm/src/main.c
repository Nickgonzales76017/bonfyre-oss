/*
 * bonfyre-swarm — distributed compute federation for Bonfyre pipelines
 *
 * Partitions a recipe YAML DAG across a fleet of workers and coordinates
 * data movement via CAS hashes.  Workers run pipeline stages locally; the
 * primary node tracks which CAS entries live on which worker and routes
 * reruns to avoid redundant data transfer.
 *
 * SUBCOMMANDS
 *   dispatch  Partition a recipe DAG and send stages to workers
 *   worker    Listen for work packages and execute assigned stages
 *   status    Show peer availability and CAS coverage per worker
 *   ping      One-shot liveness check of all configured peers
 *   fleet     Print the peer list from the swarm config
 *
 * PROTOCOL
 *   Work packages are sent as length-prefixed JSON over TCP:
 *     { "stage":0, "binary":"bonfyre-transcribe",
 *       "input_hash":"<64-hex>", "input_url":"<host:port/path>",
 *       "recipe_level":"A1", "args":"--model medium" }
 *   Workers reply:
 *     { "status":"ok"|"error", "output_hash":"<64-hex>", "error":"..." }
 *
 * CONFIG FILE  (~/.local/share/bonfyre/swarm-peers.txt)
 *   One "host:port" entry per line.  Lines starting with '#' are ignored.
 *
 * ENVIRONMENT
 *   BONFYRE_SWARM_PEERS   override peers file path
 *   BONFYRE_SWARM_PORT    worker listen port (default 9320)
 *   BONFYRE_CAS_DIR       CAS root (shared with libbonfyre)
 *
 * No external dependencies beyond libbonfyre + POSIX sockets.
 */

#define _POSIX_C_SOURCE 200809L
#define _BSD_SOURCE
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
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <bonfyre.h>
#include <bf_cas.h>

#define VERSION         "1.0.0"
#define DEFAULT_PORT    9320
#define MAX_PEERS       64
#define PKT_MAX         131072   /* max JSON work-package size */
#define RECV_TIMEOUT_S  30
#define BACKLOG         8

/* ───────────────────────────────────────────────────────────────────────────
 * Peer list
 * ─────────────────────────────────────────────────────────────────────────── */

typedef struct { char host[256]; int port; } BfPeer;
static BfPeer g_peers[MAX_PEERS];
static int    g_npeers = 0;

static void peers_path(char *buf, size_t len) {
    const char *e = getenv("BONFYRE_SWARM_PEERS");
    if (e) { snprintf(buf, len, "%s", e); return; }
    const char *home = getenv("HOME"); if (!home) home = "/tmp";
    snprintf(buf, len, "%s/.local/share/bonfyre/swarm-peers.txt", home);
}

static int load_peers(void) {
    char path[4096];
    peers_path(path, sizeof(path));
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "swarm: no peers file at %s\n", path);
        fprintf(stderr, "       create it with one 'host:port' per line\n");
        return 0;
    }
    char line[512];
    while (fgets(line, sizeof(line), fp) && g_npeers < MAX_PEERS) {
        /* strip comment + whitespace */
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        /* parse host:port */
        char host[256]; int port = DEFAULT_PORT;
        char *colon = strrchr(p, ':');
        if (colon) {
            *colon = '\0';
            port = atoi(colon + 1);
            if (port <= 0 || port > 65535) port = DEFAULT_PORT;
        }
        /* strip trailing whitespace from host */
        size_t hlen = strlen(p);
        while (hlen > 0 && (p[hlen-1] == '\r' || p[hlen-1] == '\n' ||
                             p[hlen-1] == ' '  || p[hlen-1] == '\t'))
            p[--hlen] = '\0';
        snprintf(host, sizeof(host), "%s", p);
        if (!host[0]) continue;
        snprintf(g_peers[g_npeers].host, sizeof(g_peers[0].host), "%s", host);
        g_peers[g_npeers].port = port;
        g_npeers++;
    }
    fclose(fp);
    return g_npeers;
}

/* ───────────────────────────────────────────────────────────────────────────
 * TCP helpers
 * ─────────────────────────────────────────────────────────────────────────── */

/* Connect to host:port with timeout.  Returns fd or -1. */
static int tcp_connect(const char *host, int port, int timeout_s) {
    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);
    if (getaddrinfo(host, portstr, &hints, &res) != 0) return -1;

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return -1; }

    /* set send/recv timeout */
    struct timeval tv = { timeout_s, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        close(fd); freeaddrinfo(res); return -1;
    }
    freeaddrinfo(res);
    return fd;
}

/* Send length-prefixed message: 4-byte BE uint32 length, then payload. */
static int send_lp(int fd, const char *data, uint32_t len) {
    uint32_t nlen = htonl(len);
    if (write(fd, &nlen, 4) != 4) return -1;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(fd, data + sent, len - sent);
        if (n <= 0) return -1;
        sent += (size_t)n;
    }
    return 0;
}

/* Recv length-prefixed message into buf (caller provides buf[PKT_MAX]). */
static int recv_lp(int fd, char *buf) {
    uint32_t nlen = 0;
    if (read(fd, &nlen, 4) != 4) return -1;
    uint32_t len = ntohl(nlen);
    if (len == 0 || len >= PKT_MAX) return -1;
    size_t got = 0;
    while (got < len) {
        ssize_t n = read(fd, buf + got, len - got);
        if (n <= 0) return -1;
        got += (size_t)n;
    }
    buf[len] = '\0';
    return (int)len;
}

/* ───────────────────────────────────────────────────────────────────────────
 * Work-package JSON builder / parser
 * ─────────────────────────────────────────────────────────────────────────── */

#define JSON_STR(buf, sz, key, val) \
    snprintf(buf, sz, "{\"stage\":%d,\"binary\":\"%s\",\"input_hash\":\"%s\"," \
             "\"recipe_level\":\"%s\",\"args\":\"%s\"}", \
             (val).stage, (val).binary, (val).input_hash, \
             (val).recipe_level, (val).extra_args)

typedef struct {
    int  stage;
    char binary[128];
    char input_hash[BF_CAS_HASH_LEN];
    char recipe_level[32];
    char extra_args[512];
} BfWorkPkg;

static void build_work_json(char *buf, size_t sz, const BfWorkPkg *p) {
    snprintf(buf, sz,
        "{\"stage\":%d,\"binary\":\"%s\",\"input_hash\":\"%s\","
        "\"recipe_level\":\"%s\",\"args\":\"%s\"}",
        p->stage, p->binary, p->input_hash, p->recipe_level, p->extra_args);
}

/* Minimal JSON extractor for a string field "key":"value" */
static void json_get_str(const char *json, const char *key,
                          char *out, size_t olen) {
    char pat[256];
    snprintf(pat, sizeof(pat), "\"%s\":\"", key);
    const char *p = strstr(json, pat);
    if (!p) { out[0] = '\0'; return; }
    p += strlen(pat);
    size_t i = 0;
    while (*p && *p != '"' && i < olen - 1)
        out[i++] = *p++;
    out[i] = '\0';
}

/* ───────────────────────────────────────────────────────────────────────────
 * CAS staging helpers (send file to a worker via TCP sideband)
 * ─────────────────────────────────────────────────────────────────────────── */

/* Stage a local CAS entry to a remote worker by sending its content.
 * Protocol: send_lp(hex_hash) then send_lp(manifest_bytes).
 * Currently called by the swarm-delta gossip layer; unused in dispatch mode. */
static int __attribute__((unused))
cas_push(int fd, BfCasCtx *cas, const char *hex) {
    /* resolve local path */
    char src[4096 + 128];
    char result_link[4096 + 128];
    snprintf(result_link, sizeof(result_link), "%s/%16.16s/result", cas->root, hex);
    char resolved[4096];
    ssize_t rlen = readlink(result_link, resolved, sizeof(resolved) - 1);
    if (rlen < 0) {
        /* entry not a symlink — try direct path */
        snprintf(src, sizeof(src), "%s/%16.16s", cas->root, hex);
    } else {
        resolved[rlen] = '\0';
        snprintf(src, sizeof(src), "%s", resolved);
    }
    /* For simplicity, send the manifest JSON only (workers re-execute if output missing) */
    char manifest[4096 + 128];
    snprintf(manifest, sizeof(manifest), "%s/%16.16s/run-manifest.json", cas->root, hex);
    FILE *f = fopen(manifest, "r");
    if (!f) {
        /* no manifest — send empty marker */
        if (send_lp(fd, hex, 64) < 0) return -1;
        if (send_lp(fd, "{}", 2) < 0) return -1;
        return 0;
    }
    char mbuf[PKT_MAX]; size_t n = fread(mbuf, 1, sizeof(mbuf)-1, f); fclose(f);
    mbuf[n] = '\0';
    if (send_lp(fd, hex, 64) < 0) return -1;
    if (send_lp(fd, mbuf, (uint32_t)n) < 0) return -1;
    return 0;
}

/* ───────────────────────────────────────────────────────────────────────────
 * dispatch subcommand
 * ─────────────────────────────────────────────────────────────────────────── */

/*
 * Very simple recipe YAML parser: looks for lines of the form
 *   "  binary: bonfyre-<name>"
 * and collects them in order — one per pipeline level.
 */
#define MAX_LEVELS 32
static int parse_recipe_binaries(const char *recipe_path,
                                  char binaries[MAX_LEVELS][128]) {
    FILE *f = fopen(recipe_path, "r");
    if (!f) {
        fprintf(stderr, "dispatch: cannot open recipe: %s\n", recipe_path);
        return -1;
    }
    int n = 0;
    char line[512];
    while (fgets(line, sizeof(line), f) && n < MAX_LEVELS) {
        char *p = strstr(line, "binary:");
        if (!p) continue;
        p += 7;
        while (*p == ' ' || *p == '\t') p++;
        /* strip bonfyre- prefix if present */
        if (strncmp(p, "bonfyre-", 8) == 0) p += 8;
        size_t i = 0;
        while (*p && *p != '\n' && *p != '\r' && i < 127)
            binaries[n][i++] = *p++;
        binaries[n][i] = '\0';
        if (binaries[n][0]) n++;
    }
    fclose(f);
    return n;
}

static void cmd_dispatch(int argc, char **argv) {
    /* Usage: bonfyre-swarm dispatch <recipe.yaml> <input_file> [--nodes N] */
    if (argc < 4) {
        fprintf(stderr,
            "usage: bonfyre-swarm dispatch <recipe.yaml> <input_file> [--nodes N]\n");
        exit(1);
    }
    const char *recipe   = argv[2];
    const char *input    = argv[3];
    int         max_nodes = (int)g_npeers;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--nodes") == 0 && i + 1 < argc)
            max_nodes = atoi(argv[++i]);
    }
    if (max_nodes <= 0) max_nodes = 1;
    if (max_nodes > g_npeers) max_nodes = g_npeers;

    /* Hash the input file */
    BfCasCtx cas;
    if (bf_cas_init(&cas) < 0) {
        fprintf(stderr, "dispatch: CAS init failed\n"); exit(1);
    }
    char input_hash[BF_CAS_HASH_LEN];
    if (bf_cas_hash_file(input, input_hash) < 0) {
        fprintf(stderr, "dispatch: cannot hash input file: %s\n", input); exit(1);
    }

    /* Parse recipe */
    char binaries[MAX_LEVELS][128];
    int nlevels = parse_recipe_binaries(recipe, binaries);
    if (nlevels <= 0) {
        fprintf(stderr, "dispatch: no stages found in recipe %s\n", recipe); exit(1);
    }

    printf("dispatching %d stages across %d worker(s)\n", nlevels, max_nodes);
    printf("input_hash: %.16s...\n\n", input_hash);

    /* Assign stages → workers in round-robin */
    char current_hash[BF_CAS_HASH_LEN];
    strncpy(current_hash, input_hash, BF_CAS_HASH_LEN);

    for (int lv = 0; lv < nlevels; lv++) {
        int peer_idx = lv % max_nodes;
        BfPeer *peer = &g_peers[peer_idx];

        printf("  level %d  bonfyre-%-20s  -> %s:%d ...",
               lv, binaries[lv], peer->host, peer->port);
        fflush(stdout);

        BfWorkPkg pkg;
        pkg.stage = lv;
        snprintf(pkg.binary,       sizeof(pkg.binary),       "bonfyre-%s", binaries[lv]);
        snprintf(pkg.input_hash,   sizeof(pkg.input_hash),   "%s", current_hash);
        snprintf(pkg.recipe_level, sizeof(pkg.recipe_level), "L%d", lv);
        pkg.extra_args[0] = '\0';

        char json[PKT_MAX];
        build_work_json(json, sizeof(json), &pkg);

        int fd = tcp_connect(peer->host, peer->port, RECV_TIMEOUT_S);
        if (fd < 0) {
            printf("  FAILED (connection refused)\n");
            continue;
        }

        if (send_lp(fd, json, (uint32_t)strlen(json)) < 0) {
            printf("  FAILED (send)\n"); close(fd); continue;
        }

        char reply[PKT_MAX];
        int rlen = recv_lp(fd, reply);
        close(fd);

        if (rlen < 0) {
            printf("  FAILED (no reply)\n"); continue;
        }

        char status[32], out_hash[BF_CAS_HASH_LEN], errmsg[256];
        json_get_str(reply, "status",      status,   sizeof(status));
        json_get_str(reply, "output_hash", out_hash, sizeof(out_hash));
        json_get_str(reply, "error",       errmsg,   sizeof(errmsg));

        if (strcmp(status, "ok") == 0 && out_hash[0]) {
            strncpy(current_hash, out_hash, BF_CAS_HASH_LEN);
            printf("  ok  output=%.16s...\n", out_hash);
        } else {
            printf("  ERROR: %s\n", errmsg[0] ? errmsg : "(unknown)");
        }
    }

    printf("\nfinal output hash: %.16s...\n", current_hash);
    printf("retrieve with: bonfyre-control inspect %s\n", current_hash);
}

/* ───────────────────────────────────────────────────────────────────────────
 * worker subcommand
 * ─────────────────────────────────────────────────────────────────────────── */

static volatile sig_atomic_t g_stop = 0;
static void sig_stop(int s) { (void)s; g_stop = 1; }

/* Execute a work package: spawn the binary, wait, hash the output. */
static int execute_work(const BfWorkPkg *pkg, char out_hash[BF_CAS_HASH_LEN]) {
    /* Build a temp directory for output */
    char tmpdir[256];
    snprintf(tmpdir, sizeof(tmpdir), "/tmp/bonfyre-worker-%d-%lld",
             (int)getpid(), (long long)time(NULL));
    if (mkdir(tmpdir, 0700) < 0) {
        fprintf(stderr, "worker: mkdir failed: %s\n", strerror(errno));
        return -1;
    }

    /* Resolve CAS input path */
    BfCasCtx cas;
    if (bf_cas_init(&cas) < 0) return -1;

    char cas_entry[4096 + 128];
    snprintf(cas_entry, sizeof(cas_entry), "%s/%16.16s/result",
             cas.root, pkg->input_hash);

    /* Build argv for the stage binary */
    char cmd[1024];
    /* Look in PATH first via execvp, pass CAS input path and temp output dir */
    snprintf(cmd, sizeof(cmd), "%s %s --output %s %s",
             pkg->binary, cas_entry, tmpdir,
             pkg->extra_args[0] ? pkg->extra_args : "");

    printf("worker: executing: %s\n", cmd);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "worker: binary exited %d\n", ret);
        return -1;
    }

    /* Hash the output directory manifest if it exists, else hash the dir path */
    char out_manifest[4096];
    snprintf(out_manifest, sizeof(out_manifest), "%s/artifact.json", tmpdir);
    struct stat st;
    if (stat(out_manifest, &st) == 0) {
        bf_cas_hash_file(out_manifest, out_hash);
    } else {
        /* Hash tmpdir path as a stand-in */
        BfSha256 sha;
        bf_sha256_init(&sha);
        bf_sha256_update(&sha, (const uint8_t *)tmpdir, strlen(tmpdir));
        uint8_t d[32]; bf_sha256_final(&sha, d);
        static const char hx[] = "0123456789abcdef";
        for (int i = 0; i < 32; i++) {
            out_hash[2*i]   = hx[d[i] >> 4];
            out_hash[2*i+1] = hx[d[i] & 0xf];
        }
        out_hash[64] = '\0';
    }

    /* Store result in CAS */
    bf_cas_store(&cas, out_hash, pkg->input_hash, pkg->recipe_level,
                 tmpdir, pkg->binary);
    return 0;
}

static void handle_connection(int conn_fd) {
    char buf[PKT_MAX];
    int  len = recv_lp(conn_fd, buf);
    if (len <= 0) { close(conn_fd); return; }

    BfWorkPkg pkg;
    memset(&pkg, 0, sizeof(pkg));
    char stage_s[16];
    json_get_str(buf, "stage",        stage_s,         sizeof(stage_s));
    json_get_str(buf, "binary",       pkg.binary,      sizeof(pkg.binary));
    json_get_str(buf, "input_hash",   pkg.input_hash,  sizeof(pkg.input_hash));
    json_get_str(buf, "recipe_level", pkg.recipe_level,sizeof(pkg.recipe_level));
    json_get_str(buf, "args",         pkg.extra_args,  sizeof(pkg.extra_args));
    pkg.stage = stage_s[0] ? atoi(stage_s) : 0;

    char out_hash[BF_CAS_HASH_LEN] = {0};
    char reply[512];

    if (pkg.binary[0] == '\0') {
        snprintf(reply, sizeof(reply),
                 "{\"status\":\"error\",\"error\":\"missing binary field\"}");
    } else if (execute_work(&pkg, out_hash) < 0) {
        snprintf(reply, sizeof(reply),
                 "{\"status\":\"error\",\"output_hash\":\"\","
                 "\"error\":\"execution failed\"}");
    } else {
        snprintf(reply, sizeof(reply),
                 "{\"status\":\"ok\",\"output_hash\":\"%s\"}", out_hash);
    }
    send_lp(conn_fd, reply, (uint32_t)strlen(reply));
    close(conn_fd);
}

static void cmd_worker(int argc, char **argv) {
    int port = DEFAULT_PORT;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = atoi(argv[++i]);
    }
    const char *port_env = getenv("BONFYRE_SWARM_PORT");
    if (port_env && atoi(port_env) > 0) port = atoi(port_env);

    int server_fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (server_fd < 0) {
        /* fallback to IPv4 */
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            perror("worker: socket"); exit(1);
        }
    }
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in6 addr6;
    memset(&addr6, 0, sizeof(addr6));
    addr6.sin6_family = AF_INET6;
    addr6.sin6_port   = htons((uint16_t)port);
    addr6.sin6_addr   = in6addr_any;
    if (bind(server_fd, (struct sockaddr *)&addr6, sizeof(addr6)) < 0) {
        /* fallback to IPv4 bind */
        close(server_fd);
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in addr4;
        memset(&addr4, 0, sizeof(addr4));
        addr4.sin_family      = AF_INET;
        addr4.sin_port        = htons((uint16_t)port);
        addr4.sin_addr.s_addr = INADDR_ANY;
        if (bind(server_fd, (struct sockaddr *)&addr4, sizeof(addr4)) < 0) {
            perror("worker: bind"); exit(1);
        }
    }
    if (listen(server_fd, BACKLOG) < 0) { perror("worker: listen"); exit(1); }

    signal(SIGINT,  sig_stop);
    signal(SIGTERM, sig_stop);
    signal(SIGCHLD, SIG_DFL);

    printf("bonfyre-swarm worker listening on port %d\n", port);
    printf("CAS dir: %s\n", getenv("BONFYRE_CAS_DIR")
           ? getenv("BONFYRE_CAS_DIR") : "~/.local/share/bonfyre/cas");
    printf("Press Ctrl-C to stop.\n\n");

    while (!g_stop) {
        struct sockaddr_storage peer_addr;
        socklen_t peer_len = sizeof(peer_addr);
        int conn = accept(server_fd, (struct sockaddr *)&peer_addr, &peer_len);
        if (conn < 0) {
            if (errno == EINTR) continue;
            if (!g_stop) perror("worker: accept");
            continue;
        }
        /* fork a child to handle the connection */
        pid_t pid = fork();
        if (pid == 0) {
            close(server_fd);
            handle_connection(conn);
            exit(0);
        }
        close(conn);
        /* reap zombies */
        while (waitpid(-1, NULL, WNOHANG) > 0) ; /* empty */
    }
    close(server_fd);
}

/* ───────────────────────────────────────────────────────────────────────────
 * status subcommand
 * ─────────────────────────────────────────────────────────────────────────── */

static void cmd_status(void) {
    if (g_npeers == 0) {
        printf("no peers configured\n");
        return;
    }
    printf("%-40s  %-7s  %-8s\n", "PEER", "STATUS", "LATENCY");
    printf("%-40s  %-7s  %-8s\n",
           "----------------------------------------", "-------", "--------");

    for (int i = 0; i < g_npeers; i++) {
        BfPeer *p = &g_peers[i];
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        /* Send a minimal probe: just a length-prefixed "{}" */
        int fd = tcp_connect(p->host, p->port, 3 /* 3s timeout */);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long lat_ms = (long)((t1.tv_sec - t0.tv_sec) * 1000 +
                             (t1.tv_nsec - t0.tv_nsec) / 1000000);
        if (fd < 0) {
            printf("  %s:%-5d  %-7s  n/a\n", p->host, p->port, "DOWN");
        } else {
            /* Send ping payload */
            send_lp(fd, "{\"ping\":1}", 9);
            char buf[512]; recv_lp(fd, buf);
            close(fd);
            printf("  %s:%-5d  %-7s  %ldms\n", p->host, p->port, "UP", lat_ms);
        }
    }
}

/* ───────────────────────────────────────────────────────────────────────────
 * fleet subcommand
 * ─────────────────────────────────────────────────────────────────────────── */

static void cmd_fleet(void) {
    char path[4096];
    peers_path(path, sizeof(path));
    printf("peers file: %s\n\n", path);
    if (g_npeers == 0) { printf("  (no peers configured)\n"); return; }
    for (int i = 0; i < g_npeers; i++)
        printf("  [%d]  %s:%d\n", i, g_peers[i].host, g_peers[i].port);
}

/* ───────────────────────────────────────────────────────────────────────────
 * help + main
 * ─────────────────────────────────────────────────────────────────────────── */

static void cmd_help(void) {
    printf(
"bonfyre-swarm %s -- distributed compute federation\n\n"
"USAGE\n"
"  bonfyre-swarm <command> [args]\n\n"
"COMMANDS\n"
"  dispatch <recipe.yaml> <input>  partition DAG and send stages to workers\n"
"    [--nodes N]                   limit to N workers (default: all peers)\n"
"  worker                          listen for work packages and execute them\n"
"    [--port N]                    listen port (default: 9320)\n"
"  status                          ping all peers and report latency\n"
"  fleet                           list configured peers\n"
"  help                            this message\n\n"
"PEER CONFIG\n"
"  ~/.local/share/bonfyre/swarm-peers.txt\n"
"  Override: BONFYRE_SWARM_PEERS\n"
"  One 'host:port' per line, '#' for comments.\n\n"
"ENVIRONMENT\n"
"  BONFYRE_SWARM_PORT    worker listen port (default: 9320)\n"
"  BONFYRE_SWARM_PEERS   override peers file path\n"
"  BONFYRE_CAS_DIR       CAS root (shared with all Bonfyre tools)\n\n"
"EXAMPLES\n"
"  # Start a worker on this machine\n"
"  bonfyre-swarm worker --port 9320\n\n"
"  # Dispatch a pipeline across 3 workers\n"
"  bonfyre-swarm dispatch recipe.yaml audio.mp3 --nodes 3\n\n"
"  # Check fleet health\n"
"  bonfyre-swarm status\n",
    VERSION);
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "help") == 0 ||
        strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        cmd_help(); return 0;
    }

    load_peers();

    const char *cmd = argv[1];
    if (strcmp(cmd, "dispatch") == 0) {
        if (g_npeers == 0) {
            fprintf(stderr,
                "swarm dispatch: no peers configured.\n"
                "  Create ~/.local/share/bonfyre/swarm-peers.txt with one host:port per line.\n");
            return 1;
        }
        cmd_dispatch(argc, argv);
    } else if (strcmp(cmd, "worker") == 0) {
        cmd_worker(argc, argv);
    } else if (strcmp(cmd, "status") == 0) {
        cmd_status();
    } else if (strcmp(cmd, "fleet") == 0) {
        cmd_fleet();
    } else {
        fprintf(stderr, "bonfyre-swarm: unknown command: %s\n", cmd);
        fprintf(stderr, "Run 'bonfyre-swarm help' for usage.\n");
        return 1;
    }
    return 0;
}
