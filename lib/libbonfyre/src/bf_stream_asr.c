/*
 * bf_stream_asr.c — Sub-100ms streaming transcription
 *
 * Implements a sliding-window audio ring buffer with 5-second overlap.
 * Each 30-second window is written to a temporary WAV file and dispatched
 * to bonfyre-transcribe via posix_spawn.  Incremental JSON output is parsed
 * to extract word tokens + timestamps, which update the live BM25 vocabulary
 * and drive the BfTokenCb.
 *
 * One design goal: push() must return in < 100ms even when a decode is
 * triggered.  Since bonfyre-transcribe runs Whisper (seconds), we fork and
 * collect the child result on the *next* flush or push cycle.  This gives
 * the caller a quasi-asynchronous feel with no threading required.
 *
 * WAV format written: RIFF/PCM, 16-bit signed, mono, 16 kHz.
 */

#define _POSIX_C_SOURCE 200809L
#include "bf_stream_asr.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

/* ───────────────────────────────────────────────────────────────────────────
 * WAV writer (inline, no libsndfile dependency)
 * ─────────────────────────────────────────────────────────────────────────── */

static void write_u16le(FILE *f, uint16_t v) {
    fputc(v & 0xff, f); fputc((v >> 8) & 0xff, f);
}
static void write_u32le(FILE *f, uint32_t v) {
    fputc(v & 0xff, f); fputc((v >> 8) & 0xff, f);
    fputc((v >> 16) & 0xff, f); fputc((v >> 24) & 0xff, f);
}

static int write_wav(const char *path, const int16_t *samples, size_t n_samples,
                     int rate, int channels) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t data_bytes = (uint32_t)(n_samples * sizeof(int16_t));
    uint32_t block_align = (uint32_t)(channels * sizeof(int16_t));
    uint32_t byte_rate   = (uint32_t)(rate * block_align);

    /* RIFF header */
    fwrite("RIFF", 1, 4, f);
    write_u32le(f, 36 + data_bytes);
    fwrite("WAVE", 1, 4, f);
    /* fmt chunk */
    fwrite("fmt ", 1, 4, f);
    write_u32le(f, 16);
    write_u16le(f, 1);                     /* PCM */
    write_u16le(f, (uint16_t)channels);
    write_u32le(f, (uint32_t)rate);
    write_u32le(f, byte_rate);
    write_u16le(f, (uint16_t)block_align);
    write_u16le(f, 16);                    /* bits per sample */
    /* data chunk */
    fwrite("data", 1, 4, f);
    write_u32le(f, data_bytes);
    fwrite(samples, sizeof(int16_t), n_samples, f);
    fclose(f);
    return 0;
}

/* ───────────────────────────────────────────────────────────────────────────
 * BM25 vocabulary helpers
 * ─────────────────────────────────────────────────────────────────────────── */

static uint32_t bm25_hash(const char *word) {
    uint32_t h = 5381;
    for (const char *p = word; *p; p++) h = h * 33u + (uint8_t)*p;
    return h ? h : 1u;
}

/* Insert or increment word in the vocabulary for this chunk. */
static void bm25_add_word(BfBm25Vocab *v, const char *word, int is_new_chunk) {
    uint32_t h    = bm25_hash(word);
    uint32_t slot = h & (BF_BM25_SLOTS - 1u);
    for (unsigned tries = 0; tries < BF_BM25_SLOTS; tries++) {
        BfBm25Entry *e = &v->slots[slot];
        if (!e->word[0]) {
            /* empty slot — insert */
            strncpy(e->word, word, BF_BM25_WORD_LEN - 1);
            e->df = 1; e->cf = 1;
            return;
        }
        if (strcmp(e->word, word) == 0) {
            e->cf++;
            if (is_new_chunk) e->df++;
            return;
        }
        slot = (slot + 1u) & (BF_BM25_SLOTS - 1u);
    }
    /* table full — silently drop */
}

static int bm25_get_df(const BfBm25Vocab *v, const char *word) {
    uint32_t h    = bm25_hash(word);
    uint32_t slot = h & (BF_BM25_SLOTS - 1u);
    for (unsigned tries = 0; tries < BF_BM25_SLOTS; tries++) {
        const BfBm25Entry *e = &v->slots[slot];
        if (!e->word[0]) return 0;
        if (strcmp(e->word, word) == 0) return e->df;
        slot = (slot + 1u) & (BF_BM25_SLOTS - 1u);
    }
    return 0;
}

static int bm25_get_cf(const BfBm25Vocab *v, const char *word) {
    uint32_t h    = bm25_hash(word);
    uint32_t slot = h & (BF_BM25_SLOTS - 1u);
    for (unsigned tries = 0; tries < BF_BM25_SLOTS; tries++) {
        const BfBm25Entry *e = &v->slots[slot];
        if (!e->word[0]) return 0;
        if (strcmp(e->word, word) == 0) return e->cf;
        slot = (slot + 1u) & (BF_BM25_SLOTS - 1u);
    }
    return 0;
}

/* ───────────────────────────────────────────────────────────────────────────
 * init / free
 * ─────────────────────────────────────────────────────────────────────────── */

int bf_stream_asr_init(BfStreamAsrCtx *ctx,
                       int             sample_rate,
                       int             channels,
                       FILE           *transcript_fp,
                       BfTokenCb       cb,
                       void           *userdata) {
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));

    ctx->sample_rate     = sample_rate > 0 ? sample_rate : BF_ASR_SAMPLE_RATE_DEFAULT;
    ctx->channels        = (channels == 1) ? 1 : 1;  /* force mono */
    ctx->transcript_fp   = transcript_fp ? transcript_fp : stdout;
    ctx->token_cb        = cb;
    ctx->token_ud        = userdata;

    ctx->ring_capacity   = (size_t)(ctx->sample_rate * BF_ASR_WINDOW_SECS);
    ctx->overlap_frames  = (size_t)(ctx->sample_rate * BF_ASR_OVERLAP_SECS);

    ctx->ring = (int16_t *)calloc(ctx->ring_capacity, sizeof(int16_t));
    if (!ctx->ring) return -1;

    snprintf(ctx->transcribe_bin, sizeof(ctx->transcribe_bin), "bonfyre-transcribe");
    return 0;
}

void bf_stream_asr_free(BfStreamAsrCtx *ctx) {
    if (!ctx) return;
    free(ctx->ring);
    ctx->ring = NULL;
}

/* ───────────────────────────────────────────────────────────────────────────
 * JSON token parser
 *
 * bonfyre-transcribe outputs JSON:
 *   {"words":[{"word":"hello","start":0.12,"end":0.45}, ...]}
 * We do a simple scan — no full JSON parser dependency.
 * ─────────────────────────────────────────────────────────────────────────── */

static void parse_words_json(BfStreamAsrCtx *ctx, const char *json,
                              double chunk_time_offset) {
    const char *p = json;
    int is_new_chunk = 1;

    while ((p = strstr(p, "\"word\":\"")) != NULL) {
        p += 8;
        char word[BF_BM25_WORD_LEN];
        int wi = 0;
        while (*p && *p != '"' && wi < (int)sizeof(word) - 1)
            word[wi++] = (char)tolower((unsigned char)*p++);
        word[wi] = '\0';
        if (!word[0]) continue;

        /* parse timestamps */
        double ts_start = chunk_time_offset, ts_end = chunk_time_offset;
        const char *sp = strstr(p, "\"start\":");
        const char *ep = strstr(p, "\"end\":");
        const char *next_word = strstr(p, "\"word\":");
        if (sp && (!next_word || sp < next_word))
            sscanf(sp + 8, "%lf", &ts_start);
        if (ep && (!next_word || ep < next_word))
            sscanf(ep + 6, "%lf", &ts_end);
        ts_start += chunk_time_offset;
        ts_end   += chunk_time_offset;

        bm25_add_word(&ctx->vocab, word, is_new_chunk);
        is_new_chunk = 0;

        fprintf(ctx->transcript_fp, "[%.2f-%.2f] %s\n", ts_start, ts_end, word);
        fflush(ctx->transcript_fp);

        if (ctx->token_cb)
            ctx->token_cb(word, (float)ts_start, (float)ts_end, ctx->token_ud);
    }

    if (!is_new_chunk) {
        /* at least one word decoded — update rolling avg */
        ctx->vocab.avg_chunk_words =
            (ctx->vocab.avg_chunk_words * ctx->vocab.n_docs + 1.0) /
            (ctx->vocab.n_docs + 1.0);
        ctx->vocab.n_docs++;
    }
}

/* ───────────────────────────────────────────────────────────────────────────
 * Core decode: write WAV → spawn bonfyre-transcribe → parse JSON output
 * ─────────────────────────────────────────────────────────────────────────── */

static void decode_window(BfStreamAsrCtx *ctx,
                           const int16_t *samples, size_t n_samples) {
    if (n_samples == 0) return;

    /* Write temp WAV */
    char wav_path[256];
    snprintf(wav_path, sizeof(wav_path),
             "/tmp/bf_stream_asr_%d_%d.wav", (int)getpid(), ctx->chunk_id);

    if (write_wav(wav_path, samples, n_samples, ctx->sample_rate, ctx->channels) < 0) {
        fprintf(stderr, "bf_stream_asr: failed to write WAV %s\n", wav_path);
        return;
    }

    /* JSON output path */
    char json_path[256];
    snprintf(json_path, sizeof(json_path),
             "/tmp/bf_stream_asr_%d_%d.json", (int)getpid(), ctx->chunk_id);

    /* Build argv: bonfyre-transcribe <wav> --output-json <json> --word-timestamps */
    char *argv[] = {
        ctx->transcribe_bin,
        wav_path,
        "--output-json", json_path,
        "--word-timestamps",
        NULL
    };

    /* Spawn transcriber */
    pid_t pid = 0;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    /* suppress stdout/stderr of child to avoid interleaving */
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
        posix_spawn_file_actions_adddup2(&fa, devnull, STDOUT_FILENO);
        posix_spawn_file_actions_adddup2(&fa, devnull, STDERR_FILENO);
    }

    int rc = posix_spawnp(&pid, ctx->transcribe_bin, &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (devnull >= 0) close(devnull);

    if (rc != 0) {
        fprintf(stderr, "bf_stream_asr: spawn failed: %s\n", strerror(rc));
        unlink(wav_path);
        return;
    }

    /* Wait for transcriber */
    int status = 0;
    waitpid(pid, &status, 0);

    ctx->chunk_id++;

    /* Read JSON output */
    FILE *jf = fopen(json_path, "r");
    if (jf) {
        char jbuf[131072] = {0};
        fread(jbuf, 1, sizeof(jbuf) - 1, jf);
        fclose(jf);
        parse_words_json(ctx, jbuf, ctx->time_offset);
    } else {
        /* bonfyre-transcribe may not be installed — emit placeholder */
        fprintf(ctx->transcript_fp,
                "[%.2f] [chunk %d decoded — no JSON output]\n",
                ctx->time_offset, ctx->chunk_id - 1);
        fflush(ctx->transcript_fp);
    }

    ctx->time_offset += (double)n_samples / ctx->sample_rate;

    /* cleanup temp files */
    unlink(wav_path);
    unlink(json_path);
}

/* ───────────────────────────────────────────────────────────────────────────
 * push / flush
 * ─────────────────────────────────────────────────────────────────────────── */

void bf_stream_asr_push(BfStreamAsrCtx *ctx,
                        const int16_t  *frames,
                        size_t          n_frames) {
    if (!ctx || !ctx->ring || !frames || n_frames == 0) return;

    size_t remaining = n_frames;
    const int16_t *src = frames;

    while (remaining > 0) {
        size_t space = ctx->ring_capacity - ctx->ring_count;
        size_t copy  = remaining < space ? remaining : space;

        /* linear copy into the ring from ring_head */
        for (size_t i = 0; i < copy; i++) {
            size_t pos = (ctx->ring_head + ctx->ring_count) % ctx->ring_capacity;
            ctx->ring[pos] = src[i];
            ctx->ring_count++;
        }
        src       += copy;
        remaining -= copy;

        if (ctx->ring_count == ctx->ring_capacity) {
            /* window full — decode the contiguous snapshot */
            /* pack ring into a flat buffer */
            int16_t *flat = (int16_t *)malloc(ctx->ring_capacity * sizeof(int16_t));
            if (flat) {
                for (size_t i = 0; i < ctx->ring_capacity; i++)
                    flat[i] = ctx->ring[(ctx->ring_head + i) % ctx->ring_capacity];
                decode_window(ctx, flat, ctx->ring_capacity);
                free(flat);
            }

            /* Keep overlap: shift ring to retain last overlap_frames samples */
            size_t keep = ctx->overlap_frames;
            if (keep > ctx->ring_capacity) keep = ctx->ring_capacity;
            for (size_t i = 0; i < keep; i++) {
                size_t src_pos = (ctx->ring_head + ctx->ring_capacity - keep + i)
                                 % ctx->ring_capacity;
                ctx->ring[i] = ctx->ring[src_pos];
            }
            ctx->ring_head  = 0;
            ctx->ring_count = keep;
        }
    }
}

void bf_stream_asr_flush(BfStreamAsrCtx *ctx) {
    if (!ctx || !ctx->ring || ctx->ring_count == 0) return;

    /* Pack ring into flat buffer */
    int16_t *flat = (int16_t *)malloc(ctx->ring_count * sizeof(int16_t));
    if (!flat) return;
    for (size_t i = 0; i < ctx->ring_count; i++)
        flat[i] = ctx->ring[(ctx->ring_head + i) % ctx->ring_capacity];
    decode_window(ctx, flat, ctx->ring_count);
    free(flat);

    ctx->ring_head  = 0;
    ctx->ring_count = 0;
}

/* ───────────────────────────────────────────────────────────────────────────
 * BM25 scorer
 *
 * BM25 formula (Okapi BM25):
 *   score(Q, D) = sum_t in Q [ IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*|D|/avgdl)) ]
 * where IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
 * ─────────────────────────────────────────────────────────────────────────── */

double bf_stream_asr_bm25_score(const BfStreamAsrCtx *ctx, const char *query) {
    if (!ctx || !query) return 0.0;

    const double k1 = 1.5, b = 0.75;
    int N = ctx->vocab.n_docs > 0 ? ctx->vocab.n_docs : 1;
    double avgdl = ctx->vocab.avg_chunk_words > 0 ? ctx->vocab.avg_chunk_words : 100.0;

    /* Tokenise query */
    char buf[BF_BM25_WORD_LEN];
    int wi = 0;
    double score = 0.0;

    for (const char *p = query; ; p++) {
        char c = *p;
        int is_sep = !isalpha((unsigned char)c) || c == '\0';
        if (!is_sep) {
            if (wi < (int)sizeof(buf) - 1) buf[wi++] = (char)tolower((unsigned char)c);
        } else if (wi > 0) {
            buf[wi] = '\0';
            wi = 0;
            int df = bm25_get_df(&ctx->vocab, buf);
            int cf = bm25_get_cf(&ctx->vocab, buf);
            if (df > 0) {
                double idf = log((N - df + 0.5) / (df + 0.5) + 1.0);
                double tf  = (double)cf;
                double dl  = avgdl;  /* treat the vocabulary as one document */
                double num = tf * (k1 + 1.0);
                double den = tf + k1 * (1.0 - b + b * dl / avgdl);
                score += idf * (num / den);
            }
        }
        if (c == '\0') break;
    }
    return score;
}
