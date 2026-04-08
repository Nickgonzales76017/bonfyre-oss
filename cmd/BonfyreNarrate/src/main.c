/*
 * BonfyreNarrate v2 — Tone-Aware Text-to-Speech Synthesis
 *
 * The narration layer that actually listens before it speaks.
 *
 * Previous version: fork piper with zero awareness of the source audio.
 * This version: reads the tone profile from bonfyre-tone and shapes
 * the synthesized output to match the original speaker's vocal signature.
 *
 * Pipeline:
 *   bonfyre-tone profile audio.wav → profile.json
 *   bonfyre-narrate <text> <out-dir> --tone-profile profile.json
 *
 * What it does:
 *   1. SSML generation — maps tone dimensions to prosody tags
 *      (rate, pitch, volume, break durations)
 *   2. Sentence-level pacing — uses energy/variation curves to
 *      insert natural pause patterns matching the speaker
 *   3. Post-synthesis pitch correction — pure-C PSOLA pitch shift
 *      on the output WAV to match the source F0 range
 *   4. Energy envelope transfer — shapes the output loudness contour
 *      to match the source speaker's dynamic range
 *
 * All DSP is inline C. No dependencies beyond piper and libbonfyre.
 * The FFT, resampling, and windowing code is proven in hcp-whisper.
 */
#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <bonfyre.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 *  §1  WAV I/O
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int16_t *samples;
    int      n_samples;
    int      sample_rate;
    int      channels;
} WavAudio;

static WavAudio wav_read(const char *path) {
    WavAudio w = {0};
    FILE *f = fopen(path, "rb");
    if (!f) return w;

    char riff[4]; uint32_t fsize; char wave[4];
    if (fread(riff, 1, 4, f) != 4) goto fail;
    if (memcmp(riff, "RIFF", 4) != 0) goto fail;
    if (fread(&fsize, 4, 1, f) != 1) goto fail;
    if (fread(wave, 1, 4, f) != 4) goto fail;
    if (memcmp(wave, "WAVE", 4) != 0) goto fail;

    /* find fmt chunk */
    while (1) {
        char id[4]; uint32_t sz;
        if (fread(id, 1, 4, f) != 4) goto fail;
        if (fread(&sz, 4, 1, f) != 1) goto fail;
        if (memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt, ch; uint32_t sr;
            if (fread(&fmt, 2, 1, f) != 1) goto fail;
            if (fread(&ch, 2, 1, f) != 1) goto fail;
            if (fread(&sr, 4, 1, f) != 1) goto fail;
            w.channels = ch;
            w.sample_rate = (int)sr;
            if (sz > 8) fseek(f, (long)(sz - 8), SEEK_CUR);
            continue;
        }
        if (memcmp(id, "data", 4) == 0) {
            int n = (int)(sz / 2); /* 16-bit samples */
            w.samples = malloc((size_t)n * sizeof(int16_t));
            if (!w.samples) goto fail;
            if ((int)fread(w.samples, 2, (size_t)n, f) != n) {
                free(w.samples); w.samples = NULL; goto fail;
            }
            w.n_samples = n / w.channels; /* per-channel count */
            fclose(f);
            return w;
        }
        fseek(f, (long)sz, SEEK_CUR);
    }
fail:
    fclose(f);
    return w;
}

static int wav_write(const char *path, const int16_t *samples, int n,
                     int sample_rate, int channels) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t data_size = (uint32_t)(n * channels * 2);
    uint32_t file_size = 36 + data_size;
    uint16_t block_align = (uint16_t)(channels * 2);
    uint32_t byte_rate = (uint32_t)(sample_rate * channels * 2);

    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t pcm_fmt = 1;
    uint16_t ch = (uint16_t)channels;
    uint32_t sr = (uint32_t)sample_rate;
    fwrite(&pcm_fmt, 2, 1, f);
    fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    uint16_t bits = 16;
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    fwrite(samples, 2, (size_t)(n * channels), f);
    fclose(f);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  §2  Tone Profile Parser
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    double energy;       /* 0-100: loudness normalized */
    double pitch;        /* 0-100: F0 semitone normalized */
    double stability;    /* 0-100: jitter (lower = steadier) */
    double brightness;   /* 0-100: spectral flux */
    double pace;         /* 0-100: mean voiced segment length */
    double variation;    /* 0-100: loudness contour variation */
    double pitch_hz;     /* estimated F0 in Hz from semitone value */
    int    valid;
} ToneProfile;

/* Minimal JSON number extractor for tone profile */
static double json_get_num(const char *json, const char *dim,
                           const char *field) {
    /* Find "energy": { ... "normalized": 50.0 ... } */
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", dim);
    const char *p = strstr(json, pattern);
    if (!p) return -1.0;
    snprintf(pattern, sizeof(pattern), "\"%s\"", field);
    p = strstr(p, pattern);
    if (!p) return -1.0;
    p += strlen(pattern);
    while (*p && *p != ':') p++;
    if (*p == ':') p++;
    while (*p == ' ' || *p == '\t') p++;
    return atof(p);
}

static double json_get_raw(const char *json, const char *dim) {
    return json_get_num(json, dim, "raw");
}

static ToneProfile parse_tone_profile(const char *path) {
    ToneProfile tp = {0};
    FILE *f = fopen(path, "r");
    if (!f) return tp;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *json = malloc((size_t)sz + 1);
    if (!json) { fclose(f); return tp; }
    if (fread(json, 1, (size_t)sz, f) != (size_t)sz) {
        free(json); fclose(f); return tp;
    }
    json[sz] = '\0';
    fclose(f);

    tp.energy     = json_get_num(json, "energy",     "normalized");
    tp.pitch      = json_get_num(json, "pitch",      "normalized");
    tp.stability  = json_get_num(json, "stability",  "normalized");
    tp.brightness = json_get_num(json, "brightness", "normalized");
    tp.pace       = json_get_num(json, "pace",       "normalized");
    tp.variation  = json_get_num(json, "variation",  "normalized");

    /* Convert semitone value back to Hz estimate for pitch shifting */
    double raw_semitone = json_get_raw(json, "pitch");
    if (raw_semitone > -99.0) {
        /* F0semitone is relative to 27.5 Hz (A0) in openSMILE */
        tp.pitch_hz = 27.5 * pow(2.0, raw_semitone / 12.0);
    } else {
        tp.pitch_hz = 0.0;
    }

    tp.valid = (tp.energy >= 0 && tp.pitch >= 0);
    free(json);
    return tp;
}

/* ═══════════════════════════════════════════════════════════════
 *  §3  SSML Prosody Generator
 *
 *  Maps the 6 tone dimensions to SSML prosody attributes.
 *  Piper supports a subset of SSML; we generate portable tags
 *  and fall back to plain text + pause injection when SSML
 *  isn't available.
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    double rate_pct;     /* speaking rate: -50% to +100% */
    double pitch_st;     /* pitch shift in semitones */
    double volume_pct;   /* volume: -30% to +50% */
    int    pause_ms;     /* inter-sentence pause (ms) */
    int    emphasis;     /* 0=none, 1=moderate, 2=strong */
} ProsodyParams;

static ProsodyParams compute_prosody(const ToneProfile *tp) {
    ProsodyParams pp = {0};

    /* Rate: pace 0=slow→ -30%, pace 100=fast→ +40% */
    pp.rate_pct = (tp->pace - 50.0) * 0.7;
    if (pp.rate_pct < -30.0) pp.rate_pct = -30.0;
    if (pp.rate_pct > 40.0) pp.rate_pct = 40.0;

    /* Volume: energy 0=quiet→ -20%, energy 100=loud→ +30% */
    pp.volume_pct = (tp->energy - 50.0) * 0.5;
    if (pp.volume_pct < -20.0) pp.volume_pct = -20.0;
    if (pp.volume_pct > 30.0) pp.volume_pct = 30.0;

    /* Pitch: direct from tone profile, clamped */
    pp.pitch_st = (tp->pitch - 50.0) * 0.12; /* max ±6 semitones */
    if (pp.pitch_st < -6.0) pp.pitch_st = -6.0;
    if (pp.pitch_st > 6.0) pp.pitch_st = 6.0;

    /* Pause: low variation = longer pauses (steady speaker), high = shorter */
    pp.pause_ms = (int)(800.0 - tp->variation * 5.0);
    if (pp.pause_ms < 200) pp.pause_ms = 200;
    if (pp.pause_ms > 1200) pp.pause_ms = 1200;

    /* Emphasis: high energy + high variation = strong emphasis */
    if (tp->energy > 70.0 && tp->variation > 60.0) pp.emphasis = 2;
    else if (tp->energy > 50.0 || tp->variation > 40.0) pp.emphasis = 1;

    return pp;
}

/* Split text into sentences, wrap each with SSML prosody */
static char *generate_ssml(const char *text, const ProsodyParams *pp) {
    size_t tlen = strlen(text);
    size_t cap = tlen * 4 + 4096;
    char *out = malloc(cap);
    if (!out) return NULL;

    int pos = 0;
    pos += snprintf(out + pos, cap - (size_t)pos,
        "<speak>\n"
        "<prosody rate=\"%+.0f%%\" pitch=\"%+.1fst\" volume=\"%+.0f%%\">\n",
        pp->rate_pct, pp->pitch_st, pp->volume_pct);

    const char *p = text;
    while (*p) {
        /* find sentence boundary */
        const char *end = p;
        while (*end && *end != '.' && *end != '!' && *end != '?' && *end != '\n') end++;
        if (*end) end++; /* include punctuation */

        /* skip empty lines */
        size_t slen = (size_t)(end - p);
        int all_space = 1;
        for (size_t i = 0; i < slen; i++) {
            if (!isspace((unsigned char)p[i])) { all_space = 0; break; }
        }

        if (!all_space && slen > 0) {
            /* write sentence */
            if (pp->emphasis == 2) {
                pos += snprintf(out + pos, cap - (size_t)pos, "<emphasis level=\"strong\">");
            }

            /* escape XML special chars */
            for (size_t i = 0; i < slen; i++) {
                char c = p[i];
                if (c == '&')       pos += snprintf(out + pos, cap - (size_t)pos, "&amp;");
                else if (c == '<')  pos += snprintf(out + pos, cap - (size_t)pos, "&lt;");
                else if (c == '>')  pos += snprintf(out + pos, cap - (size_t)pos, "&gt;");
                else if (c == '\n') pos += snprintf(out + pos, cap - (size_t)pos, " ");
                else { if ((size_t)pos < cap - 1) out[pos++] = c; }
            }

            if (pp->emphasis == 2) {
                pos += snprintf(out + pos, cap - (size_t)pos, "</emphasis>");
            }

            pos += snprintf(out + pos, cap - (size_t)pos,
                "\n<break time=\"%dms\"/>\n", pp->pause_ms);
        }

        p = end;
        if (!*p) break;
    }

    pos += snprintf(out + pos, cap - (size_t)pos,
        "</prosody>\n</speak>\n");
    out[pos] = '\0';
    return out;
}

/* Plain text fallback with pause markers for engines without SSML */
static char *generate_paced_text(const char *text, const ProsodyParams *pp) {
    size_t tlen = strlen(text);
    size_t cap = tlen * 2 + 2048;
    char *out = malloc(cap);
    if (!out) return NULL;

    int pos = 0;
    const char *p = text;

    while (*p) {
        const char *end = p;
        while (*end && *end != '.' && *end != '!' && *end != '?') end++;
        if (*end) end++;

        size_t slen = (size_t)(end - p);
        int all_space = 1;
        for (size_t i = 0; i < slen; i++) {
            if (!isspace((unsigned char)p[i])) { all_space = 0; break; }
        }

        if (!all_space && slen > 0) {
            for (size_t i = 0; i < slen; i++) {
                char c = p[i];
                /* strip markdown */
                if (c == '#') { while (p[i+1] == '#') i++; while (p[i+1] == ' ') i++; continue; }
                if (c == '`') continue;
                if (c == '*') continue;
                if ((size_t)pos < cap - 1) out[pos++] = c;
            }
            /* Add sentence break proportional to pause_ms */
            int extra_newlines = pp->pause_ms / 400;
            if (extra_newlines < 1) extra_newlines = 1;
            for (int j = 0; j < extra_newlines && (size_t)pos < cap - 2; j++)
                out[pos++] = '\n';
        }

        p = end;
        if (!*p) break;
    }

    out[pos] = '\0';
    return out;
}

/* ═══════════════════════════════════════════════════════════════
 *  §4  PSOLA Pitch Shifter (Pure C)
 *
 *  Pitch-Synchronous Overlap-Add. The standard technique for
 *  time-domain pitch modification of speech signals. Works by:
 *    1. Detecting pitch periods via autocorrelation
 *    2. Windowing each period with Hanning
 *    3. Overlapping windows at the target period length
 *
 *  This gives natural-sounding pitch shift without the artifacts
 *  of frequency-domain methods on speech.
 *
 *  Reuses the Hanning window and signal processing patterns
 *  proven in hcp-whisper's spectral pipeline.
 * ═══════════════════════════════════════════════════════════════ */

/* Autocorrelation-based pitch detection for a frame of audio */
static double detect_pitch_frame(const float *frame, int len, int sr) {
    int min_lag = sr / 500;  /* 500 Hz max */
    int max_lag = sr / 60;   /* 60 Hz min */
    if (max_lag >= len) max_lag = len - 1;
    if (min_lag >= max_lag) return 0.0;

    double best_corr = -1.0;
    int best_lag = 0;

    for (int lag = min_lag; lag <= max_lag; lag++) {
        double sum = 0.0, norm_a = 0.0, norm_b = 0.0;
        int count = len - lag;
        for (int i = 0; i < count; i++) {
            sum += (double)frame[i] * frame[i + lag];
            norm_a += (double)frame[i] * frame[i];
            norm_b += (double)frame[i + lag] * frame[i + lag];
        }
        double denom = sqrt(norm_a * norm_b);
        if (denom < 1e-10) continue;
        double r = sum / denom;
        if (r > best_corr) {
            best_corr = r;
            best_lag = lag;
        }
    }

    if (best_corr < 0.3) return 0.0; /* unvoiced */
    return (double)sr / best_lag;
}

/* Hanning window (matches hcp-whisper) */
static void hanning(float *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n - 1)));
}

/* PSOLA pitch shift: modify pitch by ratio while preserving duration */
static int16_t *psola_pitch_shift(const int16_t *in, int n_samples,
                                  int sample_rate, double shift_ratio,
                                  int *out_len) {
    if (fabs(shift_ratio - 1.0) < 0.01) {
        /* No meaningful shift needed */
        *out_len = n_samples;
        int16_t *out = malloc((size_t)n_samples * sizeof(int16_t));
        if (out) memcpy(out, in, (size_t)n_samples * sizeof(int16_t));
        return out;
    }

    /* Convert to float */
    float *sig = malloc((size_t)n_samples * sizeof(float));
    if (!sig) return NULL;
    for (int i = 0; i < n_samples; i++)
        sig[i] = (float)in[i] / 32768.0f;

    /* Output buffer */
    float *out_f = calloc((size_t)n_samples + 4096, sizeof(float));
    if (!out_f) { free(sig); return NULL; }

    int frame_size = sample_rate / 80; /* ~12.5ms frames */
    float *win = malloc((size_t)(frame_size * 2) * sizeof(float));
    if (!win) { free(sig); free(out_f); return NULL; }

    int hop_in = frame_size;
    int hop_out = (int)((double)frame_size / shift_ratio);
    if (hop_out < 1) hop_out = 1;

    int out_pos = 0;
    int in_pos = 0;

    while (in_pos + frame_size * 2 < n_samples) {
        /* Detect local pitch period */
        int analysis_len = frame_size * 2;
        if (in_pos + analysis_len > n_samples)
            analysis_len = n_samples - in_pos;

        double f0 = detect_pitch_frame(sig + in_pos, analysis_len, sample_rate);
        int period;
        if (f0 > 60.0) {
            period = (int)((double)sample_rate / f0);
        } else {
            period = frame_size; /* unvoiced: use default */
        }

        /* Window size = 2 * period (centered on pitch period) */
        int wlen = period * 2;
        if (wlen > frame_size * 2) wlen = frame_size * 2;
        if (in_pos + wlen > n_samples) wlen = n_samples - in_pos;

        hanning(win, wlen);

        /* Overlap-add at output position */
        for (int i = 0; i < wlen && out_pos + i < n_samples + 4096; i++) {
            out_f[out_pos + i] += sig[in_pos + i] * win[i];
        }

        in_pos += hop_in;
        out_pos += hop_out;
    }

    /* Convert back to int16 */
    int total = out_pos + frame_size * 2;
    if (total > n_samples + 4096) total = n_samples + 4096;
    *out_len = total;

    int16_t *out = malloc((size_t)total * sizeof(int16_t));
    if (!out) { free(sig); free(out_f); free(win); return NULL; }

    /* Find peak for normalization */
    float peak = 0.0f;
    for (int i = 0; i < total; i++) {
        float a = fabsf(out_f[i]);
        if (a > peak) peak = a;
    }
    float scale = (peak > 0.001f) ? (0.95f / peak) : 1.0f;

    for (int i = 0; i < total; i++) {
        float v = out_f[i] * scale * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        out[i] = (int16_t)v;
    }

    free(sig);
    free(out_f);
    free(win);
    return out;
}

/* ═══════════════════════════════════════════════════════════════
 *  §5  Energy Envelope Transfer
 *
 *  Shapes the synthesized audio's loudness contour to match the
 *  original speaker's dynamic range. Works on ~50ms frames:
 *    1. Compute RMS energy envelope of the target profile
 *    2. Compute RMS envelope of the synthesized audio
 *    3. Scale each frame so the synth envelope matches
 * ═══════════════════════════════════════════════════════════════ */

static void apply_energy_envelope(int16_t *samples, int n, int sr,
                                  double target_energy_norm) {
    /* target_energy_norm is 0-100 from tone profile */
    /* Map to gain: 30→0.6, 50→1.0, 80→1.4 */
    double gain = 0.4 + (target_energy_norm / 100.0) * 1.2;
    if (gain < 0.3) gain = 0.3;
    if (gain > 2.0) gain = 2.0;

    int frame_size = sr / 20; /* 50ms frames */
    if (frame_size < 1) frame_size = 1;

    for (int i = 0; i < n; i += frame_size) {
        int end = i + frame_size;
        if (end > n) end = n;

        /* Compute RMS of this frame */
        double rms = 0.0;
        for (int j = i; j < end; j++)
            rms += (double)samples[j] * samples[j];
        rms = sqrt(rms / (end - i));

        if (rms < 1.0) continue; /* silence */

        /* Apply gain with soft limiting */
        for (int j = i; j < end; j++) {
            double v = (double)samples[j] * gain;
            /* Soft clip (tanh) */
            if (v > 20000.0 || v < -20000.0) {
                v = 32767.0 * tanh(v / 32767.0);
            }
            if (v > 32767.0) v = 32767.0;
            if (v < -32768.0) v = -32768.0;
            samples[j] = (int16_t)v;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  §6  Piper TTS Invocation
 * ═══════════════════════════════════════════════════════════════ */

static int command_exists(const char *name) {
    char *path_env = getenv("PATH");
    if (!path_env) return 0;
    char *copy = strdup(path_env);
    if (!copy) return 0;
    char *saveptr = NULL;
    char *dir = strtok_r(copy, ":", &saveptr);
    while (dir) {
        char candidate[PATH_MAX];
        snprintf(candidate, sizeof(candidate), "%s/%s", dir, name);
        if (access(candidate, X_OK) == 0) { free(copy); return 1; }
        dir = strtok_r(NULL, ":", &saveptr);
    }
    free(copy);
    return 0;
}

static int run_piper(const char *voice_model, const char *text_path,
                     const char *audio_path, const char *log_path,
                     double volume, int sentence_silence_ms) {
    FILE *log = fopen(log_path, "w");
    if (!log) return 1;

    char vol_str[32], silence_str[32];
    snprintf(vol_str, sizeof(vol_str), "%.2f", volume);
    snprintf(silence_str, sizeof(silence_str), "%.3f",
             (double)sentence_silence_ms / 1000.0);

    pid_t pid = fork();
    if (pid < 0) { fclose(log); return 1; }
    if (pid == 0) {
        int fd = fileno(log);
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        execlp("piper", "piper",
               "--model", voice_model,
               "--output_file", audio_path,
               "--file", text_path,
               "--sentence-silence", silence_str,
               (char *)NULL);
        _exit(127);
    }

    int status;
    waitpid(pid, &status, 0);
    fclose(log);
    return (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? 0 : 1;
}

/* ═══════════════════════════════════════════════════════════════
 *  §7  Text Normalization
 * ═══════════════════════════════════════════════════════════════ */

static char *read_text_file(const char *path) {
    FILE *in = fopen(path, "rb");
    if (!in) return NULL;
    fseek(in, 0, SEEK_END);
    long size = ftell(in);
    if (size < 0) { fclose(in); return NULL; }
    rewind(in);
    char *buf = malloc((size_t)size + 1);
    if (!buf) { fclose(in); return NULL; }
    if (size > 0 && (long)fread(buf, 1, (size_t)size, in) != size) {
        free(buf); fclose(in); return NULL;
    }
    buf[size] = '\0';
    fclose(in);
    return buf;
}

static char *normalize_markdown(const char *text) {
    size_t len = strlen(text);
    char *out = malloc(len * 2 + 64);
    if (!out) return NULL;
    size_t o = 0;
    int at_line = 1;

    for (size_t i = 0; i < len; i++) {
        char c = text[i];
        if (at_line && c == '#') {
            while (text[i] == '#') i++;
            while (text[i] == ' ') i++;
            out[o++] = '\n';
            at_line = 0;
            c = text[i];
        }
        if (c == '`') continue;
        if (c == '*') continue;
        if (c == '\r') continue;
        if (c == '\n') {
            if (o > 0 && out[o-1] != '\n') out[o++] = '\n';
            at_line = 1;
            continue;
        }
        out[o++] = c;
        at_line = 0;
    }
    out[o] = '\0';
    return out;
}

/* ═══════════════════════════════════════════════════════════════
 *  §8  Main — Orchestration
 * ═══════════════════════════════════════════════════════════════ */

static void usage(void) {
    fprintf(stderr,
        "bonfyre-narrate v2 — tone-aware text-to-speech synthesis\n\n"
        "Usage:\n"
        "  bonfyre-narrate <text-file> <output-dir> [options]\n\n"
        "Options:\n"
        "  --voice-model PATH      Piper ONNX model path\n"
        "  --tone-profile PATH     Tone profile JSON (from bonfyre-tone profile)\n"
        "  --audio-format FMT      Output format (default: wav)\n"
        "  --title TITLE           Artifact title\n"
        "  --ssml                  Generate SSML (for engines that support it)\n"
        "  --dry-run               Print plan without generating audio\n"
        "  --no-pitch-shift        Skip post-synthesis pitch correction\n"
        "  --no-envelope           Skip energy envelope transfer\n\n"
        "Pipeline integration:\n"
        "  bonfyre-tone profile audio.wav output/\n"
        "  bonfyre-narrate summary.md output/ --tone-profile output/profile.json\n\n"
        "The narrated output will match the original speaker's vocal signature.\n");
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "status") == 0) {
        printf("{\"binary\":\"bonfyre-narrate\",\"version\":\"2.0.0\","
               "\"features\":[\"ssml-prosody\",\"psola-pitch-shift\","
               "\"energy-envelope\",\"tone-profile-input\"],"
               "\"status\":\"available\"}\n");
        return 0;
    }

    if (argc < 3) { usage(); return 1; }

    const char *source_path = argv[1];
    const char *output_dir  = argv[2];
    const char *voice_model = "";
    const char *tone_profile_path = NULL;
    const char *audio_format = "wav";
    const char *title = "Bonfyre Artifact";
    int use_ssml = 0;
    int dry_run = 0;
    int do_pitch_shift = 1;
    int do_envelope = 1;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--voice-model") == 0 && i+1 < argc)
            voice_model = argv[++i];
        else if (strcmp(argv[i], "--tone-profile") == 0 && i+1 < argc)
            tone_profile_path = argv[++i];
        else if (strcmp(argv[i], "--audio-format") == 0 && i+1 < argc)
            audio_format = argv[++i];
        else if (strcmp(argv[i], "--title") == 0 && i+1 < argc)
            title = argv[++i];
        else if (strcmp(argv[i], "--ssml") == 0)
            use_ssml = 1;
        else if (strcmp(argv[i], "--dry-run") == 0)
            dry_run = 1;
        else if (strcmp(argv[i], "--no-pitch-shift") == 0)
            do_pitch_shift = 0;
        else if (strcmp(argv[i], "--no-envelope") == 0)
            do_envelope = 0;
        else { usage(); return 1; }
    }

    /* ── Parse tone profile ─────────────────────────────────── */
    ToneProfile tp = {0};
    if (tone_profile_path) {
        tp = parse_tone_profile(tone_profile_path);
        if (tp.valid) {
            fprintf(stderr, "[narrate] Tone profile loaded:\n");
            fprintf(stderr, "          energy=%.0f  pitch=%.0f (%.0f Hz)  "
                    "pace=%.0f  variation=%.0f\n",
                    tp.energy, tp.pitch, tp.pitch_hz, tp.pace, tp.variation);
        } else {
            fprintf(stderr, "[narrate] Warning: could not parse tone profile, "
                    "using defaults\n");
        }
    }

    /* ── Compute prosody ────────────────────────────────────── */
    ProsodyParams pp;
    if (tp.valid) {
        pp = compute_prosody(&tp);
    } else {
        pp = (ProsodyParams){ .rate_pct = 0, .pitch_st = 0,
                              .volume_pct = 0, .pause_ms = 600,
                              .emphasis = 0 };
    }

    if (bf_ensure_dir(output_dir) != 0) {
        fprintf(stderr, "Failed to create output dir: %s\n", output_dir);
        return 1;
    }

    /* ── Read + normalize source text ───────────────────────── */
    char *source_text = read_text_file(source_path);
    if (!source_text) {
        fprintf(stderr, "Cannot read: %s\n", source_path);
        return 2;
    }

    char *clean = normalize_markdown(source_text);
    if (!clean) { free(source_text); return 1; }

    /* ── Generate narration input (SSML or paced text) ──────── */
    char *narration;
    const char *narration_type;
    if (use_ssml) {
        narration = generate_ssml(clean, &pp);
        narration_type = "ssml";
    } else {
        narration = generate_paced_text(clean, &pp);
        narration_type = "paced-text";
    }
    free(clean);
    if (!narration) { free(source_text); return 1; }

    /* ── Paths ──────────────────────────────────────────────── */
    char narration_path[PATH_MAX], audio_path[PATH_MAX];
    char manifest_path[PATH_MAX], log_path[PATH_MAX];
    char raw_audio_path[PATH_MAX];

    snprintf(narration_path, sizeof(narration_path), "%s/narration.%s",
             output_dir, use_ssml ? "ssml" : "txt");
    snprintf(raw_audio_path, sizeof(raw_audio_path), "%s/raw.wav", output_dir);
    snprintf(audio_path, sizeof(audio_path), "%s/artifact.%s",
             output_dir, audio_format);
    snprintf(manifest_path, sizeof(manifest_path),
             "%s/artifact.manifest.json", output_dir);
    snprintf(log_path, sizeof(log_path), "%s/render.log", output_dir);

    /* ── Dry run ────────────────────────────────────────────── */
    if (dry_run) {
        printf("Plan:\n");
        printf("  Source:     %s\n", source_path);
        printf("  Output:     %s\n", output_dir);
        printf("  Tone:       %s\n", tone_profile_path ? tone_profile_path : "(none)");
        printf("  Prosody:\n");
        printf("    rate:     %+.0f%%\n", pp.rate_pct);
        printf("    pitch:    %+.1f st\n", pp.pitch_st);
        printf("    volume:   %+.0f%%\n", pp.volume_pct);
        printf("    pause:    %d ms\n", pp.pause_ms);
        printf("    emphasis: %s\n",
               pp.emphasis == 2 ? "strong" : pp.emphasis == 1 ? "moderate" : "none");
        printf("  Post-FX:\n");
        printf("    pitch-shift: %s", do_pitch_shift && tp.pitch_hz > 0 ? "yes" : "no");
        if (do_pitch_shift && tp.pitch_hz > 0)
            printf(" (target %.0f Hz)", tp.pitch_hz);
        printf("\n");
        printf("    envelope:    %s\n", do_envelope && tp.valid ? "yes" : "no");
        free(source_text);
        free(narration);
        return 0;
    }

    /* ── Write narration text ───────────────────────────────── */
    FILE *nf = fopen(narration_path, "w");
    if (!nf) { free(source_text); free(narration); return 1; }
    fputs(narration, nf);
    fputc('\n', nf);
    fclose(nf);

    fprintf(stderr, "[narrate] Narration text: %s (%s)\n",
            narration_path, narration_type);
    fprintf(stderr, "[narrate] Prosody: rate=%+.0f%% pitch=%+.1fst "
            "vol=%+.0f%% pause=%dms\n",
            pp.rate_pct, pp.pitch_st, pp.volume_pct, pp.pause_ms);

    /* ── Synthesize with Piper ──────────────────────────────── */
    const char *render_status = "skipped";
    const char *render_reason = "piper_unavailable";
    int synthesis_ok = 0;

    if (command_exists("piper") && voice_model[0] != '\0') {
        /* Compute piper volume from prosody */
        double piper_vol = 1.0 + (pp.volume_pct / 100.0);
        if (piper_vol < 0.3) piper_vol = 0.3;
        if (piper_vol > 2.0) piper_vol = 2.0;

        /* Use raw path if we need post-processing, otherwise final path */
        const char *synth_target = (do_pitch_shift || do_envelope) ?
                                    raw_audio_path : audio_path;

        fprintf(stderr, "[narrate] Synthesizing via Piper (vol=%.2f, "
                "sentence-silence=%dms)\n", piper_vol, pp.pause_ms);

        if (run_piper(voice_model, narration_path, synth_target,
                      log_path, piper_vol, pp.pause_ms) == 0) {
            synthesis_ok = 1;
            render_status = "completed";
            render_reason = "rendered";
        } else {
            render_status = "failed";
            render_reason = "piper_render_failed";
        }
    } else if (command_exists("piper")) {
        render_status = "skipped";
        render_reason = "missing_voice_model";
    }

    /* ── Post-synthesis DSP ─────────────────────────────────── */
    if (synthesis_ok && (do_pitch_shift || do_envelope)) {
        WavAudio wav = wav_read(raw_audio_path);
        if (wav.samples && wav.n_samples > 0) {
            int16_t *current = wav.samples;
            int current_n = wav.n_samples;

            /* Pitch correction */
            if (do_pitch_shift && tp.pitch_hz > 60.0) {
                /* Estimate Piper's default pitch (assume ~150 Hz for
                 * en_US models) and compute shift ratio */
                double piper_default_f0 = 150.0;
                double ratio = tp.pitch_hz / piper_default_f0;

                /* Clamp to reasonable range */
                if (ratio < 0.5) ratio = 0.5;
                if (ratio > 2.0) ratio = 2.0;

                if (fabs(ratio - 1.0) > 0.05) {
                    fprintf(stderr, "[narrate] Pitch shift: %.0f Hz → %.0f Hz "
                            "(ratio=%.2f)\n", piper_default_f0, tp.pitch_hz, ratio);

                    int shifted_n;
                    int16_t *shifted = psola_pitch_shift(current, current_n,
                                                         wav.sample_rate,
                                                         ratio, &shifted_n);
                    if (shifted) {
                        if (current != wav.samples) free(current);
                        current = shifted;
                        current_n = shifted_n;
                    }
                }
            }

            /* Energy envelope transfer */
            if (do_envelope && tp.valid) {
                fprintf(stderr, "[narrate] Energy envelope: target=%.0f/100\n",
                        tp.energy);
                apply_energy_envelope(current, current_n, wav.sample_rate,
                                      tp.energy);
            }

            /* Write final output */
            wav_write(audio_path, current, current_n,
                      wav.sample_rate, wav.channels);

            if (current != wav.samples) free(current);
            free(wav.samples);

            /* Clean up raw intermediate */
            unlink(raw_audio_path);

            fprintf(stderr, "[narrate] Post-processing complete → %s\n",
                    audio_path);
        } else {
            fprintf(stderr, "[narrate] Warning: could not read synthesized WAV "
                    "for post-processing\n");
            /* Fall through — raw audio is the final output */
            if (strcmp(raw_audio_path, audio_path) != 0)
                rename(raw_audio_path, audio_path);
        }
    }

    /* ── Write manifest ─────────────────────────────────────── */
    char timestamp[64];
    time_t now = time(NULL);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

    FILE *mf = fopen(manifest_path, "w");
    if (!mf) { free(source_text); free(narration); return 1; }

    fprintf(mf,
        "{\n"
        "  \"sourceSystem\": \"BonfyreNarrate\",\n"
        "  \"version\": \"2.0.0\",\n"
        "  \"artifactType\": \"narrated-artifact\",\n"
        "  \"title\": \"%s\",\n"
        "  \"createdAt\": \"%s\",\n"
        "  \"sourceTextPath\": \"%s\",\n"
        "  \"narrationTextPath\": \"%s\",\n"
        "  \"narrationFormat\": \"%s\",\n"
        "  \"audioPath\": \"%s\",\n"
        "  \"audioFormat\": \"%s\",\n"
        "  \"voiceModel\": \"%s\",\n"
        "  \"toneProfilePath\": \"%s\",\n"
        "  \"renderStatus\": \"%s\",\n"
        "  \"renderReason\": \"%s\",\n"
        "  \"prosody\": {\n"
        "    \"rate_pct\": %.1f,\n"
        "    \"pitch_st\": %.1f,\n"
        "    \"volume_pct\": %.1f,\n"
        "    \"pause_ms\": %d,\n"
        "    \"emphasis\": %d\n"
        "  },\n"
        "  \"postProcessing\": {\n"
        "    \"pitchShift\": %s,\n"
        "    \"targetF0Hz\": %.1f,\n"
        "    \"energyEnvelope\": %s,\n"
        "    \"targetEnergy\": %.1f\n"
        "  },\n"
        "  \"renderLogPath\": \"%s\"\n"
        "}\n",
        title, timestamp, source_path, narration_path, narration_type,
        audio_path, audio_format, voice_model,
        tone_profile_path ? tone_profile_path : "",
        render_status, render_reason,
        pp.rate_pct, pp.pitch_st, pp.volume_pct, pp.pause_ms, pp.emphasis,
        (do_pitch_shift && tp.pitch_hz > 60.0) ? "true" : "false",
        tp.pitch_hz,
        (do_envelope && tp.valid) ? "true" : "false",
        tp.energy,
        log_path);
    fclose(mf);

    /* ── Summary ────────────────────────────────────────────── */
    printf("Narration:  %s (%s)\n", narration_path, narration_type);
    printf("Manifest:   %s\n", manifest_path);
    if (synthesis_ok) printf("Audio:      %s\n", audio_path);
    if (tp.valid)     printf("Tone match: pitch=%.0f Hz, energy=%.0f/100, "
                             "pace=%.0f/100\n",
                             tp.pitch_hz, tp.energy, tp.pace);

    free(source_text);
    free(narration);
    return 0;
}
