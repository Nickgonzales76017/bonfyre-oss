/*
 * bf_stream_asr.h — Sub-100ms streaming transcription via ring buffer + BM25
 *
 * Maintains a sliding 30-second audio ring buffer with 5-second overlap.
 * When a window fills, calls bonfyre-transcribe on the chunk and
 * accumulates a running transcript with live BM25 vocabulary updates.
 *
 * Caller integrates through three calls:
 *   1. bf_stream_asr_init()   — allocate ring buffer + BM25 state
 *   2. bf_stream_asr_push()   — feed raw PCM frames (16-bit / mono)
 *   3. bf_stream_asr_flush()  — drain remaining partial window
 *   4. bf_stream_asr_free()   — release memory
 *
 * Incremental output is delivered through a token callback invoked on
 * each word decoded:
 *   void token_cb(const char *word, float ts_start, float ts_end, void *ud)
 *
 * Thread safety: push/flush must not be called concurrently.
 */
#pragma once
#ifndef BF_STREAM_ASR_H
#define BF_STREAM_ASR_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Token callback ──────────────────────────────────────────────────────── */

typedef void (*BfTokenCb)(const char *word,
                           float      ts_start,
                           float      ts_end,
                           void      *userdata);

/* ── BM25 mini-vocabulary (open-addressing hash, 4096 slots) ────────────── */

#define BF_BM25_SLOTS   4096
#define BF_BM25_WORD_LEN 64

typedef struct {
    char  word[BF_BM25_WORD_LEN];
    int   df;        /* document (chunk) frequency */
    int   cf;        /* corpus (total) frequency    */
} BfBm25Entry;

typedef struct {
    BfBm25Entry slots[BF_BM25_SLOTS];
    int         n_docs;              /* number of chunks decoded so far */
    double      avg_chunk_words;     /* rolling average words per chunk  */
} BfBm25Vocab;

/* ── Context struct ──────────────────────────────────────────────────────── */

#define BF_ASR_SAMPLE_RATE_DEFAULT 16000
#define BF_ASR_CHANNELS            1
#define BF_ASR_WINDOW_SECS         30
#define BF_ASR_OVERLAP_SECS        5

typedef struct {
    /* ring buffer */
    int16_t *ring;           /* PCM samples: [window_frames] */
    size_t   ring_capacity;  /* total samples the ring can hold */
    size_t   ring_head;      /* write index                      */
    size_t   ring_count;     /* valid samples currently stored    */
    int      sample_rate;
    int      channels;

    /* overlap: keep last overlap_frames samples for next window */
    size_t   overlap_frames;

    /* decode state */
    int      chunk_id;       /* monotonically increasing chunk number */
    double   time_offset;    /* seconds: start time of current window */

    /* live transcript */
    FILE    *transcript_fp;  /* NULL = stdout */

    /* BM25 vocabulary */
    BfBm25Vocab vocab;

    /* user token callback */
    BfTokenCb  token_cb;
    void      *token_ud;

    /* transcriber binary path (default: bonfyre-transcribe in PATH) */
    char       transcribe_bin[512];
} BfStreamAsrCtx;

/* ── API ─────────────────────────────────────────────────────────────────── */

/*
 * Initialise context.
 *   ctx            — caller-allocated struct (will be fully populated)
 *   sample_rate    — PCM sample rate in Hz (e.g. 16000)
 *   channels       — must be 1 (mono) for bonfyre-transcribe
 *   transcript_fp  — where to write word-level transcript lines (NULL=stdout)
 *   cb / userdata  — token callback (may be NULL)
 * Returns 0 on success, -1 on allocation failure.
 */
int bf_stream_asr_init(BfStreamAsrCtx *ctx,
                       int             sample_rate,
                       int             channels,
                       FILE           *transcript_fp,
                       BfTokenCb       cb,
                       void           *userdata);

/*
 * Push raw PCM frames into the ring buffer.
 * When the ring fills a 30-second window, bf_stream_asr_decode() is called
 * automatically.  May call the token callback synchronously.
 *   frames   — interleaved int16_t samples (channels × n_frames values)
 *   n_frames — number of sample frames (not samples)
 */
void bf_stream_asr_push(BfStreamAsrCtx *ctx,
                        const int16_t  *frames,
                        size_t          n_frames);

/*
 * Decode any remaining partial window.  Call once when the audio stream ends.
 * Must be called before bf_stream_asr_free().
 */
void bf_stream_asr_flush(BfStreamAsrCtx *ctx);

/*
 * BM25 score a query string against the accumulated vocabulary.
 * Returns a non-negative float; higher is more relevant.
 * Useful for retrieving the most relevant portion of a long transcript.
 */
double bf_stream_asr_bm25_score(const BfStreamAsrCtx *ctx, const char *query);

/*
 * Free all heap allocations.  Context is invalid after this call.
 */
void bf_stream_asr_free(BfStreamAsrCtx *ctx);

#ifdef __cplusplus
}
#endif
#endif /* BF_STREAM_ASR_H */
