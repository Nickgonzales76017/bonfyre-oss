/*
 * BonfyreTel — FreeSWITCH Event Socket telephony adapter.
 *
 * Connects to FreeSWITCH via Event Socket (plain TCP on localhost:8021),
 * listens for call/SMS events, and triggers Bonfyre pipeline binaries.
 *
 * No Twilio dependency. Works with any SIP trunk (Telnyx, Bandwidth, etc.)
 *
 * Usage:
 *   bonfyre-tel listen  [--host HOST] [--port PORT] [--password PW] [--db FILE]
 *   bonfyre-tel send-sms --from NUM --to NUM --body TEXT [--db FILE]
 *   bonfyre-tel send-mms --from NUM --to NUM --body TEXT --media FILE [--db FILE]
 *   bonfyre-tel call     --from NUM --to NUM [--record] [--db FILE]
 *   bonfyre-tel hangup   --uuid UUID
 *   bonfyre-tel status   [--db FILE]
 *   bonfyre-tel version
 */
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <sqlite3.h>

#define VERSION         "1.0.0"
#define DEFAULT_HOST    "127.0.0.1"
#define DEFAULT_PORT    8021
#define DEFAULT_PASS    "ClueCon"
#define BUF_SIZE        (64 * 1024)
#define MAX_HEADERS     128
#define RECV_TIMEOUT_S  1

static volatile int g_running = 1;

/* ── SQLite schema ─────────────────────────────────────────────────── */

static const char *SCHEMA_SQL =
    "CREATE TABLE IF NOT EXISTS calls ("
    "  id         INTEGER PRIMARY KEY AUTOINCREMENT,"
    "  uuid       TEXT NOT NULL,"
    "  direction  TEXT NOT NULL,"   /* inbound | outbound */
    "  caller     TEXT,"
    "  callee     TEXT,"
    "  started_at TEXT NOT NULL,"
    "  ended_at   TEXT,"
    "  duration_s REAL,"
    "  recording  TEXT,"           /* path to .wav */
    "  pipeline   TEXT,"           /* pipeline job id */
    "  status     TEXT DEFAULT 'active'"
    ");"
    "CREATE TABLE IF NOT EXISTS messages ("
    "  id          INTEGER PRIMARY KEY AUTOINCREMENT,"
    "  direction   TEXT NOT NULL,"
    "  from_number TEXT,"
    "  to_number   TEXT,"
    "  body        TEXT,"
    "  media_url   TEXT,"
    "  sent_at     TEXT NOT NULL,"
    "  status      TEXT DEFAULT 'sent'"
    ");";

/* ── Utility ───────────────────────────────────────────────────────── */

static void iso_timestamp(char *buf, size_t len) {
    time_t t = time(NULL);
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(buf, len, "%Y-%m-%dT%H:%M:%SZ", &tm);
}

static const char *arg_get(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return NULL;
}

static int arg_has(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], flag) == 0) return 1;
    return 0;
}

static const char *default_db(void) {
    static char path[1024];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(path, sizeof(path), "%s/.local/share/bonfyre", home);
    mkdir(path, 0755);
    snprintf(path, sizeof(path), "%s/.local/share/bonfyre/tel.db", home);
    return path;
}

static sqlite3 *open_db(const char *path) {
    sqlite3 *db = NULL;
    if (sqlite3_open(path, &db) != SQLITE_OK) {
        fprintf(stderr, "tel: cannot open %s: %s\n", path, sqlite3_errmsg(db));
        sqlite3_close(db);
        return NULL;
    }
    char *err = NULL;
    sqlite3_exec(db, SCHEMA_SQL, NULL, NULL, &err);
    if (err) { fprintf(stderr, "tel: schema: %s\n", err); sqlite3_free(err); }
    return db;
}

static void sigint_handler(int sig) {
    (void)sig;
    g_running = 0;
}

/* ── Fork + exec (same pattern as BonfyreMediaPrep) ───────────────── */

static int run_process(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
        execvp(argv[0], argv);
        perror("execvp");
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); return 1; }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 1;
}

/* Non-blocking fork — don't wait for child (pipeline may take minutes) */
static pid_t run_process_async(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return -1; }
    if (pid == 0) {
        setsid(); /* detach from parent session */
        execvp(argv[0], argv);
        perror("execvp");
        _exit(127);
    }
    return pid;
}

/* ── ESL connection ────────────────────────────────────────────────── */

typedef struct {
    int    fd;
    char   buf[BUF_SIZE];
    size_t buf_len;
} EslConn;

static int esl_connect(EslConn *c, const char *host, int port) {
    memset(c, 0, sizeof(*c));

    struct addrinfo hints = {.ai_family = AF_INET, .ai_socktype = SOCK_STREAM};
    struct addrinfo *res = NULL;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) {
        fprintf(stderr, "tel: cannot resolve %s:%d\n", host, port);
        return -1;
    }

    c->fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (c->fd < 0) {
        perror("socket");
        freeaddrinfo(res);
        return -1;
    }

    if (connect(c->fd, res->ai_addr, res->ai_addrlen) < 0) {
        fprintf(stderr, "tel: cannot connect to FreeSWITCH at %s:%d: %s\n",
                host, port, strerror(errno));
        close(c->fd);
        c->fd = -1;
        freeaddrinfo(res);
        return -1;
    }

    freeaddrinfo(res);

    /* Set receive timeout so event loop can check g_running */
    struct timeval tv = {.tv_sec = RECV_TIMEOUT_S, .tv_usec = 0};
    setsockopt(c->fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    return 0;
}

static void esl_close(EslConn *c) {
    if (c->fd >= 0) { close(c->fd); c->fd = -1; }
}

/* Send a raw ESL command (terminated with \n\n) */
static int esl_send(EslConn *c, const char *fmt, ...) {
    char cmd[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(cmd, sizeof(cmd) - 2, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof(cmd) - 2) return -1;
    /* Ensure double newline terminator */
    if (n > 0 && cmd[n - 1] != '\n') cmd[n++] = '\n';
    cmd[n++] = '\n';
    cmd[n] = '\0';

    ssize_t sent = 0;
    while (sent < n) {
        ssize_t w = write(c->fd, cmd + sent, (size_t)(n - sent));
        if (w < 0) {
            if (errno == EINTR) continue;
            perror("write");
            return -1;
        }
        sent += w;
    }
    return 0;
}

/* Read until we get a full ESL event (double newline delimited).
 * Returns number of bytes in event, or 0 on timeout, -1 on error. */
static int esl_recv_event(EslConn *c, char *out, size_t out_sz) {
    for (;;) {
        /* Check if we already have a complete event in buffer */
        char *end = strstr(c->buf, "\n\n");
        if (end) {
            size_t event_len = (size_t)(end - c->buf + 2);
            if (event_len >= out_sz) event_len = out_sz - 1;
            memcpy(out, c->buf, event_len);
            out[event_len] = '\0';
            /* Shift remaining data */
            size_t remain = c->buf_len - (size_t)(end - c->buf + 2);
            if (remain > 0) memmove(c->buf, end + 2, remain);
            c->buf_len = remain;
            c->buf[c->buf_len] = '\0';
            return (int)event_len;
        }

        if (c->buf_len >= BUF_SIZE - 1) {
            /* Buffer full without complete event — discard */
            c->buf_len = 0;
            c->buf[0] = '\0';
        }

        ssize_t n = recv(c->fd, c->buf + c->buf_len, BUF_SIZE - 1 - c->buf_len, 0);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return 0; /* timeout */
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1; /* connection closed */
        c->buf_len += (size_t)n;
        c->buf[c->buf_len] = '\0';
    }
}

/* ── ESL event parsing ─────────────────────────────────────────────── */

typedef struct {
    char key[128];
    char value[1024];
} EslHeader;

typedef struct {
    EslHeader headers[MAX_HEADERS];
    int       count;
} EslEvent;

static void esl_parse_event(const char *raw, EslEvent *evt) {
    evt->count = 0;
    const char *p = raw;
    while (*p && evt->count < MAX_HEADERS) {
        const char *colon = strchr(p, ':');
        const char *nl = strchr(p, '\n');
        if (!colon || (nl && nl < colon)) {
            if (nl) { p = nl + 1; continue; }
            break;
        }
        size_t klen = (size_t)(colon - p);
        if (klen >= sizeof(evt->headers[0].key)) klen = sizeof(evt->headers[0].key) - 1;
        memcpy(evt->headers[evt->count].key, p, klen);
        evt->headers[evt->count].key[klen] = '\0';

        const char *vstart = colon + 1;
        while (*vstart == ' ') vstart++;
        const char *vend = nl ? nl : vstart + strlen(vstart);
        size_t vlen = (size_t)(vend - vstart);
        if (vlen >= sizeof(evt->headers[0].value)) vlen = sizeof(evt->headers[0].value) - 1;
        memcpy(evt->headers[evt->count].value, vstart, vlen);
        evt->headers[evt->count].value[vlen] = '\0';

        evt->count++;
        p = nl ? nl + 1 : vend;
    }
}

static const char *esl_get(const EslEvent *evt, const char *key) {
    for (int i = 0; i < evt->count; i++)
        if (strcasecmp(evt->headers[i].key, key) == 0)
            return evt->headers[i].value;
    return NULL;
}

/* ── Event handlers ────────────────────────────────────────────────── */

static void handle_recording_done(sqlite3 *db, const EslEvent *evt) {
    const char *uuid       = esl_get(evt, "Unique-ID");
    const char *rec_path   = esl_get(evt, "variable_record_file_path");
    const char *caller     = esl_get(evt, "Caller-Caller-ID-Number");
    const char *callee     = esl_get(evt, "Caller-Destination-Number");
    const char *duration   = esl_get(evt, "variable_record_seconds");

    if (!rec_path || !uuid) return;

    char ts[32];
    iso_timestamp(ts, sizeof(ts));
    fprintf(stderr, "tel: [%s] recording done: %s (%s → %s, %ss)\n",
            ts, rec_path, caller ? caller : "?", callee ? callee : "?",
            duration ? duration : "?");

    /* Log to DB */
    if (db) {
        const char *sql = "INSERT INTO calls "
            "(uuid, direction, caller, callee, started_at, recording, status) "
            "VALUES (?1, 'inbound', ?2, ?3, ?4, ?5, 'recorded')";
        sqlite3_stmt *stmt = NULL;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, uuid, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, caller ? caller : "", -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 3, callee ? callee : "", -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 4, ts, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 5, rec_path, -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }

    /* Trigger Bonfyre pipeline asynchronously */
    fprintf(stderr, "tel: triggering pipeline for %s\n", rec_path);
    char *const argv[] = {
        "bonfyre-pipeline", "run", (char *)rec_path, NULL
    };
    run_process_async(argv);
}

static void handle_sms_recv(sqlite3 *db, const EslEvent *evt) {
    const char *from = esl_get(evt, "from");
    const char *to   = esl_get(evt, "to");
    const char *body = esl_get(evt, "body");

    if (!from || !to) return;

    char ts[32];
    iso_timestamp(ts, sizeof(ts));
    fprintf(stderr, "tel: [%s] SMS received: %s → %s: %.80s\n",
            ts, from, to, body ? body : "(empty)");

    /* Log to DB */
    if (db) {
        const char *sql = "INSERT INTO messages "
            "(direction, from_number, to_number, body, sent_at, status) "
            "VALUES ('inbound', ?1, ?2, ?3, ?4, 'received')";
        sqlite3_stmt *stmt = NULL;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, from, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, to, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 3, body ? body : "", -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 4, ts, -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }

    /* Trigger ingest — feed the message body to the pipeline */
    if (body && body[0]) {
        char *const argv[] = {
            "bonfyre-ingest", "--text", (char *)body,
            "--meta", "channel=sms",
            "--meta", (char *)(from ? from : "unknown"),
            NULL
        };
        run_process_async(argv);
    }
}

static void handle_channel_hangup(sqlite3 *db, const EslEvent *evt) {
    const char *uuid     = esl_get(evt, "Unique-ID");
    const char *duration = esl_get(evt, "variable_billsec");

    if (!uuid) return;

    char ts[32];
    iso_timestamp(ts, sizeof(ts));
    fprintf(stderr, "tel: [%s] hangup: %s (duration: %ss)\n",
            ts, uuid, duration ? duration : "?");

    if (db) {
        const char *sql =
            "UPDATE calls SET ended_at=?1, duration_s=?2, status='completed' "
            "WHERE uuid=?3 AND ended_at IS NULL";
        sqlite3_stmt *stmt = NULL;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, ts, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, duration ? duration : "0", -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 3, uuid, -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }
}

/* ── ESL authenticate + subscribe ──────────────────────────────────── */

static int esl_auth(EslConn *c, const char *password) {
    /* FreeSWITCH sends "Content-Type: auth/request" on connect */
    char buf[BUF_SIZE];
    int n = esl_recv_event(c, buf, sizeof(buf));
    if (n <= 0) {
        fprintf(stderr, "tel: no auth prompt from FreeSWITCH\n");
        return -1;
    }
    if (!strstr(buf, "auth/request")) {
        fprintf(stderr, "tel: unexpected greeting: %.80s\n", buf);
        return -1;
    }

    esl_send(c, "auth %s", password);

    n = esl_recv_event(c, buf, sizeof(buf));
    if (n <= 0 || !strstr(buf, "command/reply")) {
        fprintf(stderr, "tel: auth failed\n");
        return -1;
    }
    if (strstr(buf, "-ERR")) {
        fprintf(stderr, "tel: auth rejected (bad password?)\n");
        return -1;
    }

    fprintf(stderr, "tel: authenticated with FreeSWITCH\n");
    return 0;
}

static int esl_subscribe(EslConn *c) {
    /* Subscribe to events we care about */
    esl_send(c, "event plain CHANNEL_EXECUTE_COMPLETE CHANNEL_HANGUP_COMPLETE");

    char buf[BUF_SIZE];
    int n = esl_recv_event(c, buf, sizeof(buf));
    if (n <= 0 || strstr(buf, "-ERR")) {
        fprintf(stderr, "tel: event subscription failed\n");
        return -1;
    }

    /* Also subscribe to custom SMS events */
    esl_send(c, "event plain CUSTOM sms::recv");
    n = esl_recv_event(c, buf, sizeof(buf));
    /* sms::recv may not exist until SMS traffic arrives — not fatal */

    fprintf(stderr, "tel: subscribed to call + SMS events\n");
    return 0;
}

/* ── Commands ──────────────────────────────────────────────────────── */

static int cmd_listen(const char *host, int port, const char *password,
                      sqlite3 *db) {
    EslConn conn;

    fprintf(stderr, "tel: connecting to FreeSWITCH at %s:%d...\n", host, port);

    if (esl_connect(&conn, host, port) < 0) return 1;
    if (esl_auth(&conn, password) < 0) { esl_close(&conn); return 1; }
    if (esl_subscribe(&conn) < 0) { esl_close(&conn); return 1; }

    fprintf(stderr, "tel: listening for events (Ctrl-C to stop)...\n");

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);
    signal(SIGCHLD, SIG_IGN); /* auto-reap async children */

    char event_buf[BUF_SIZE];
    while (g_running) {
        int n = esl_recv_event(&conn, event_buf, sizeof(event_buf));
        if (n < 0) {
            fprintf(stderr, "tel: connection lost, reconnecting in 3s...\n");
            esl_close(&conn);
            sleep(3);
            if (esl_connect(&conn, host, port) < 0) continue;
            if (esl_auth(&conn, password) < 0) { esl_close(&conn); continue; }
            if (esl_subscribe(&conn) < 0) { esl_close(&conn); continue; }
            continue;
        }
        if (n == 0) continue; /* timeout — check g_running */

        EslEvent evt;
        esl_parse_event(event_buf, &evt);

        const char *event_name = esl_get(&evt, "Event-Name");
        if (!event_name) continue;

        if (strcmp(event_name, "CHANNEL_EXECUTE_COMPLETE") == 0) {
            const char *app = esl_get(&evt, "Application");
            if (app && strcmp(app, "record") == 0) {
                handle_recording_done(db, &evt);
            }
        } else if (strcmp(event_name, "CHANNEL_HANGUP_COMPLETE") == 0) {
            handle_channel_hangup(db, &evt);
        } else if (strcmp(event_name, "CUSTOM") == 0) {
            const char *subclass = esl_get(&evt, "Event-Subclass");
            if (subclass && strcmp(subclass, "sms::recv") == 0) {
                handle_sms_recv(db, &evt);
            }
        }
    }

    fprintf(stderr, "\ntel: shutting down\n");
    esl_close(&conn);
    return 0;
}

static int cmd_send_sms(const char *host, int port, const char *password,
                        const char *from, const char *to, const char *body,
                        sqlite3 *db) {
    if (!from || !to || !body) {
        fprintf(stderr, "tel: send-sms requires --from, --to, --body\n");
        return 1;
    }

    EslConn conn;
    if (esl_connect(&conn, host, port) < 0) return 1;
    if (esl_auth(&conn, password) < 0) { esl_close(&conn); return 1; }

    /* Use FreeSWITCH chat API to send SMS via SIP MESSAGE */
    esl_send(&conn,
        "api chat sip|%s|%s|%s|text/plain",
        from, to, body);

    char buf[BUF_SIZE];
    int n = esl_recv_event(&conn, buf, sizeof(buf));
    int ok = (n > 0 && !strstr(buf, "-ERR"));

    if (ok)
        fprintf(stderr, "tel: SMS sent: %s → %s\n", from, to);
    else
        fprintf(stderr, "tel: SMS send failed\n");

    /* Log */
    if (db) {
        char ts[32];
        iso_timestamp(ts, sizeof(ts));
        const char *sql = "INSERT INTO messages "
            "(direction, from_number, to_number, body, sent_at, status) "
            "VALUES ('outbound', ?1, ?2, ?3, ?4, ?5)";
        sqlite3_stmt *stmt = NULL;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, from, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, to, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 3, body, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 4, ts, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 5, ok ? "sent" : "failed", -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }

    esl_close(&conn);
    return ok ? 0 : 1;
}

static int cmd_send_mms(const char *host, int port, const char *password,
                        const char *from, const char *to, const char *body,
                        const char *media, sqlite3 *db) {
    (void)host; (void)port; (void)password; /* MMS uses carrier REST API */
    if (!from || !to || !media) {
        fprintf(stderr, "tel: send-mms requires --from, --to, --media\n");
        return 1;
    }

    /*
     * MMS via SIP is carrier-dependent. Most SIP trunks (Telnyx, Bandwidth)
     * expose MMS via their REST API. We shell out to curl for portability.
     * The BONFYRE_TEL_MMS_ENDPOINT env var should point to the carrier's
     * MMS endpoint (e.g. https://api.telnyx.com/v2/messages).
     */
    const char *endpoint = getenv("BONFYRE_TEL_MMS_ENDPOINT");
    const char *api_key  = getenv("BONFYRE_TEL_API_KEY");

    if (!endpoint || !api_key) {
        fprintf(stderr, "tel: MMS requires BONFYRE_TEL_MMS_ENDPOINT and "
                "BONFYRE_TEL_API_KEY environment variables\n");
        return 1;
    }

    /* Build JSON payload */
    char payload[4096];
    snprintf(payload, sizeof(payload),
        "{\"from\":\"%s\",\"to\":\"%s\",\"text\":\"%s\","
        "\"media_urls\":[\"%s\"]}",
        from, to, body ? body : "", media);

    char auth_header[256];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);

    char *const argv[] = {
        "curl", "-s", "-X", "POST",
        "-H", "Content-Type: application/json",
        "-H", auth_header,
        "-d", payload,
        (char *)endpoint,
        NULL
    };

    fprintf(stderr, "tel: sending MMS %s → %s (media: %s)\n", from, to, media);
    int rc = run_process(argv);

    if (db) {
        char ts[32];
        iso_timestamp(ts, sizeof(ts));
        const char *sql = "INSERT INTO messages "
            "(direction, from_number, to_number, body, media_url, sent_at, status) "
            "VALUES ('outbound', ?1, ?2, ?3, ?4, ?5, ?6)";
        sqlite3_stmt *stmt = NULL;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, from, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, to, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 3, body ? body : "", -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 4, media, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 5, ts, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 6, rc == 0 ? "sent" : "failed", -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }

    return rc;
}

static int cmd_call(const char *host, int port, const char *password,
                    const char *from, const char *to, int do_record,
                    sqlite3 *db) {
    if (!from || !to) {
        fprintf(stderr, "tel: call requires --from and --to\n");
        return 1;
    }

    EslConn conn;
    if (esl_connect(&conn, host, port) < 0) return 1;
    if (esl_auth(&conn, password) < 0) { esl_close(&conn); return 1; }

    /*
     * Originate a call via FreeSWITCH.
     * The dialstring uses the default SIP profile — carrier config lives
     * in FreeSWITCH sip_profiles/, not in this binary.
     */
    if (do_record) {
        esl_send(&conn,
            "api originate {origination_caller_id_number=%s,"
            "execute_on_answer='record_session $${recordings_dir}/%s_%s_${strftime(%%Y%%m%%d_%%H%%M%%S)}.wav'}"
            "sofia/external/%s@${default_provider} &park()",
            from, from, to, to);
    } else {
        esl_send(&conn,
            "api originate {origination_caller_id_number=%s}"
            "sofia/external/%s@${default_provider} &park()",
            from, to);
    }

    char buf[BUF_SIZE];
    int n = esl_recv_event(&conn, buf, sizeof(buf));
    int ok = (n > 0 && !strstr(buf, "-ERR"));

    if (ok)
        fprintf(stderr, "tel: call originated: %s → %s%s\n",
                from, to, do_record ? " (recording)" : "");
    else
        fprintf(stderr, "tel: call failed: %.200s\n", n > 0 ? buf : "(no response)");

    if (db) {
        char ts[32];
        iso_timestamp(ts, sizeof(ts));
        const char *sql = "INSERT INTO calls "
            "(uuid, direction, caller, callee, started_at, status) "
            "VALUES ('pending', 'outbound', ?1, ?2, ?3, ?4)";
        sqlite3_stmt *stmt = NULL;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, from, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, to, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 3, ts, -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 4, ok ? "active" : "failed", -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }

    esl_close(&conn);
    return ok ? 0 : 1;
}

static int cmd_hangup(const char *host, int port, const char *password,
                      const char *uuid) {
    if (!uuid) {
        fprintf(stderr, "tel: hangup requires --uuid\n");
        return 1;
    }

    EslConn conn;
    if (esl_connect(&conn, host, port) < 0) return 1;
    if (esl_auth(&conn, password) < 0) { esl_close(&conn); return 1; }

    esl_send(&conn, "api uuid_kill %s", uuid);

    char buf[BUF_SIZE];
    int n = esl_recv_event(&conn, buf, sizeof(buf));
    int ok = (n > 0 && !strstr(buf, "-ERR"));

    fprintf(stderr, "tel: hangup %s: %s\n", uuid, ok ? "OK" : "failed");
    esl_close(&conn);
    return ok ? 0 : 1;
}

static int cmd_status(sqlite3 *db) {
    if (!db) {
        fprintf(stderr, "tel: no database\n");
        return 1;
    }

    printf("=== BonfyreTel Status ===\n\n");

    /* Call summary */
    sqlite3_stmt *stmt = NULL;
    const char *sql = "SELECT status, COUNT(*) FROM calls GROUP BY status";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        printf("Calls:\n");
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            printf("  %-12s %d\n",
                   sqlite3_column_text(stmt, 0),
                   sqlite3_column_int(stmt, 1));
        }
        sqlite3_finalize(stmt);
    }

    /* Message summary */
    sql = "SELECT direction, COUNT(*) FROM messages GROUP BY direction";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        printf("\nMessages:\n");
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            printf("  %-12s %d\n",
                   sqlite3_column_text(stmt, 0),
                   sqlite3_column_int(stmt, 1));
        }
        sqlite3_finalize(stmt);
    }

    /* Recent calls */
    sql = "SELECT uuid, direction, caller, callee, started_at, status "
          "FROM calls ORDER BY id DESC LIMIT 5";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        printf("\nRecent calls:\n");
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            printf("  [%s] %s %s → %s (%s) %s\n",
                   sqlite3_column_text(stmt, 4),
                   sqlite3_column_text(stmt, 1),
                   sqlite3_column_text(stmt, 2),
                   sqlite3_column_text(stmt, 3),
                   sqlite3_column_text(stmt, 5),
                   sqlite3_column_text(stmt, 0));
        }
        sqlite3_finalize(stmt);
    }

    /* Recent messages */
    sql = "SELECT direction, from_number, to_number, "
          "SUBSTR(body, 1, 50), sent_at "
          "FROM messages ORDER BY id DESC LIMIT 5";
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        printf("\nRecent messages:\n");
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            printf("  [%s] %s %s → %s: %s\n",
                   sqlite3_column_text(stmt, 4),
                   sqlite3_column_text(stmt, 0),
                   sqlite3_column_text(stmt, 1),
                   sqlite3_column_text(stmt, 2),
                   sqlite3_column_text(stmt, 3));
        }
        sqlite3_finalize(stmt);
    }

    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────── */

static void usage(void) {
    fprintf(stderr,
        "BonfyreTel %s — FreeSWITCH telephony adapter\n"
        "\n"
        "Usage:\n"
        "  bonfyre-tel listen    [--host H] [--port P] [--password PW] [--db FILE]\n"
        "  bonfyre-tel send-sms  --from NUM --to NUM --body TEXT [--db FILE]\n"
        "  bonfyre-tel send-mms  --from NUM --to NUM --body TEXT --media FILE [--db FILE]\n"
        "  bonfyre-tel call      --from NUM --to NUM [--record] [--db FILE]\n"
        "  bonfyre-tel hangup    --uuid UUID [--host H] [--port P]\n"
        "  bonfyre-tel status    [--db FILE]\n"
        "  bonfyre-tel version\n"
        "\n"
        "Environment:\n"
        "  BONFYRE_TEL_MMS_ENDPOINT  Carrier MMS REST endpoint (for send-mms)\n"
        "  BONFYRE_TEL_API_KEY       Carrier API key (for send-mms)\n"
        "\n"
        "FreeSWITCH ESL defaults: 127.0.0.1:8021, password ClueCon\n",
        VERSION);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }

    /* Parse global options */
    const char *db_path  = arg_get(argc, argv, "--db");
    const char *host     = arg_get(argc, argv, "--host");
    const char *port_str = arg_get(argc, argv, "--port");
    const char *password = arg_get(argc, argv, "--password");
    const char *from     = arg_get(argc, argv, "--from");
    const char *to       = arg_get(argc, argv, "--to");
    const char *body     = arg_get(argc, argv, "--body");
    const char *media    = arg_get(argc, argv, "--media");
    const char *uuid     = arg_get(argc, argv, "--uuid");
    int record           = arg_has(argc, argv, "--record");

    if (!db_path)  db_path  = default_db();
    if (!host)     host     = DEFAULT_HOST;
    if (!password) password = DEFAULT_PASS;
    int port = port_str ? atoi(port_str) : DEFAULT_PORT;

    const char *cmd = argv[1];

    if (strcmp(cmd, "version") == 0) {
        printf("bonfyre-tel %s\n", VERSION);
        return 0;
    }

    sqlite3 *db = open_db(db_path);
    int rc = 1;

    if (strcmp(cmd, "listen") == 0) {
        rc = cmd_listen(host, port, password, db);
    } else if (strcmp(cmd, "send-sms") == 0) {
        rc = cmd_send_sms(host, port, password, from, to, body, db);
    } else if (strcmp(cmd, "send-mms") == 0) {
        rc = cmd_send_mms(host, port, password, from, to, body, media, db);
    } else if (strcmp(cmd, "call") == 0) {
        rc = cmd_call(host, port, password, from, to, record, db);
    } else if (strcmp(cmd, "hangup") == 0) {
        rc = cmd_hangup(host, port, password, uuid);
    } else if (strcmp(cmd, "status") == 0) {
        rc = cmd_status(db);
    } else {
        fprintf(stderr, "tel: unknown command: %s\n", cmd);
        usage();
        rc = 1;
    }

    if (db) sqlite3_close(db);
    return rc;
}
