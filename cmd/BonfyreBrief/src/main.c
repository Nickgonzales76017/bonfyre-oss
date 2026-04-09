#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <bonfyre.h>

#define MAX_SENTENCES 1024
#define MAX_LINE 8192
#define MAX_BULLETS 6
#define MAX_VOCAB  4096

typedef struct {
    char *text;
    int score;
    int is_action;
} Sentence;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} TextBuffer;

/* ── TF-IDF vocabulary ── */
typedef struct {
    char word[64];
    int doc_freq;   /* number of sentences containing this word */
} VocabEntry;

static VocabEntry g_vocab[MAX_VOCAB];
static int g_vocab_count = 0;

static int ensure_dir(const char *path) { return bf_ensure_dir(path); }
static void iso_timestamp(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    strftime(buffer, size, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
}

static char *trim_copy(const char *src) {
    while (*src && isspace((unsigned char)*src)) src++;
    size_t len = strlen(src);
    while (len > 0 && isspace((unsigned char)src[len - 1])) len--;
    char *out = malloc(len + 1);
    if (!out) return NULL;
    memcpy(out, src, len);
    out[len] = '\0';
    return out;
}

static int textbuf_reserve(TextBuffer *buf, size_t extra) {
    size_t need = buf->len + extra + 1;
    if (need <= buf->cap) return 0;
    size_t next = buf->cap ? buf->cap * 2 : 1024;
    while (next < need) next *= 2;
    char *grown = realloc(buf->data, next);
    if (!grown) return 1;
    buf->data = grown;
    buf->cap = next;
    return 0;
}

static int textbuf_append_char(TextBuffer *buf, char ch) {
    if (textbuf_reserve(buf, 1) != 0) return 1;
    buf->data[buf->len++] = ch;
    buf->data[buf->len] = '\0';
    return 0;
}

static int textbuf_append_json_string(TextBuffer *buf, const char **cursor) {
    const char *p = *cursor;
    if (*p != '"') return 1;
    p++;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (!*p) break;
            switch (*p) {
                case 'n': if (textbuf_append_char(buf, ' ') != 0) return 1; break;
                case 'r': break;
                case 't': if (textbuf_append_char(buf, ' ') != 0) return 1; break;
                case '"': if (textbuf_append_char(buf, '"') != 0) return 1; break;
                case '\\': if (textbuf_append_char(buf, '\\') != 0) return 1; break;
                case '/': if (textbuf_append_char(buf, '/') != 0) return 1; break;
                case 'b': break;
                case 'f': break;
                case 'u':
                    /* Keep parser tiny: skip unicode escape payload. */
                    for (int i = 0; i < 4 && p[1]; i++) p++;
                    break;
                default:
                    if (textbuf_append_char(buf, *p) != 0) return 1;
                    break;
            }
            p++;
            continue;
        }
        if (textbuf_append_char(buf, *p) != 0) return 1;
        p++;
    }
    if (*p == '"') p++;
    *cursor = p;
    return 0;
}

static char *extract_segment_text_json(const char *json) {
    if (!json) return NULL;
    const char *segments = strstr(json, "\"segments\"");
    if (!segments) return NULL;
    TextBuffer buf = {0};
    const char *p = segments;
    int found = 0;
    while ((p = strstr(p, "\"text\"")) != NULL) {
        const char *cursor = p + strlen("\"text\"");
        while (*cursor && isspace((unsigned char)*cursor)) cursor++;
        if (*cursor != ':') {
            p = cursor;
            continue;
        }
        cursor++;
        while (*cursor && isspace((unsigned char)*cursor)) cursor++;
        if (*cursor != '"') {
            p = cursor;
            continue;
        }
        if (found && textbuf_append_char(&buf, ' ') != 0) {
            free(buf.data);
            return NULL;
        }
        if (textbuf_append_json_string(&buf, &cursor) != 0) {
            free(buf.data);
            return NULL;
        }
        found = 1;
        p = cursor;
    }
    if (!found) {
        free(buf.data);
        return NULL;
    }
    return buf.data;
}

static void lowercase_word(const char *src, char *dst, size_t dst_sz) {
    size_t i = 0;
    while (src[i] && i < dst_sz - 1) {
        dst[i] = (char)tolower((unsigned char)src[i]);
        i++;
    }
    dst[i] = '\0';
}

static int vocab_find_or_add(const char *word) {
    char lw[64];
    lowercase_word(word, lw, sizeof(lw));
    if (lw[0] == '\0' || strlen(lw) < 3) return -1;
    /* Skip stopwords */
    static const char *stops[] = {
        "the", "and", "that", "this", "with", "from", "have", "has", "had",
        "was", "were", "been", "are", "for", "not", "but", "what", "all",
        "can", "her", "his", "one", "our", "out", "you", "its", "they",
        "she", "him", "how", "about", "just", "into", "your", "some",
        "them", "than", "then", "now", "look", "only", "come", "could",
        "will", "would", "also", "back", "after", "use", "two", "way",
        "because", "any", "these", "give", "day", "most", "find",
        "here", "thing", "many", "well", "very", "when", "where", "which",
        "their", "said", "each", "tell", "does", "set", "three", "want",
        "did", "get", "make", "like", "going", "know", "really", NULL
    };
    for (int i = 0; stops[i]; i++)
        if (strcmp(lw, stops[i]) == 0) return -1;

    for (int i = 0; i < g_vocab_count; i++)
        if (strcmp(g_vocab[i].word, lw) == 0) return i;
    if (g_vocab_count >= MAX_VOCAB) return -1;
    int idx = g_vocab_count++;
    snprintf(g_vocab[idx].word, sizeof(g_vocab[idx].word), "%s", lw);
    g_vocab[idx].doc_freq = 0;
    return idx;
}

static void build_idf(Sentence *sentences, int count) {
    /* Count document frequency: how many sentences contain each word */
    for (int s = 0; s < count; s++) {
        /* Track which vocab entries we've already counted for this sentence */
        int seen[MAX_VOCAB] = {0};
        const char *p = sentences[s].text;
        while (*p) {
            while (*p && !isalpha((unsigned char)*p)) p++;
            if (!*p) break;
            const char *start = p;
            while (*p && isalpha((unsigned char)*p)) p++;
            size_t wlen = (size_t)(p - start);
            if (wlen >= 64) continue;
            char word[64];
            memcpy(word, start, wlen);
            word[wlen] = '\0';
            int idx = vocab_find_or_add(word);
            if (idx >= 0 && !seen[idx]) {
                g_vocab[idx].doc_freq++;
                seen[idx] = 1;
            }
        }
    }
}

static double tfidf_score(const char *text, int total_docs) {
    /* Compute TF-IDF sum for this sentence */
    double score = 0.0;
    int word_count = 0;
    const char *p = text;
    while (*p) {
        while (*p && !isalpha((unsigned char)*p)) p++;
        if (!*p) break;
        const char *start = p;
        while (*p && isalpha((unsigned char)*p)) p++;
        size_t wlen = (size_t)(p - start);
        if (wlen >= 64) continue;
        char word[64];
        memcpy(word, start, wlen);
        word[wlen] = '\0';
        char lw[64];
        lowercase_word(word, lw, sizeof(lw));
        for (int i = 0; i < g_vocab_count; i++) {
            if (strcmp(g_vocab[i].word, lw) == 0 && g_vocab[i].doc_freq > 0) {
                score += log((double)total_docs / (double)g_vocab[i].doc_freq);
                break;
            }
        }
        word_count++;
    }
    /* Normalize by word count to avoid length bias */
    return word_count > 0 ? score / word_count : 0.0;
}

static int contains_any(const char *text, const char *words[]) {
    for (int i = 0; words[i]; i++) {
        if (strstr(text, words[i])) return 1;
    }
    return 0;
}

static int sentence_score(const char *text, int total_docs) {
    static const char *summary_words[] = {
        "problem", "customer", "market", "revenue", "pricing", "workflow",
        "decision", "learned", "traction", "focus", "strategy", "validation",
        "founder", "operator", "pain", "channel", "segment", "growth",
        "insight", "opportunity", "challenge", "solution", "result",
        "metric", "data", "evidence", "pattern", "trend",
        "important", "key", "reason", "example", "research", "study",
        "explains", "discovered", "method", "approach", "benefit",
        "concept", "principle", "idea", "point", "argument", "claim",
        "experience", "lesson", "mistake", "advice", "recommend",
        "habit", "routine", "system", "process", "step", "rule", NULL
    };
    static const char *action_words[] = {
        "should", "need to", "must", "next", "plan", "test", "focus", "send",
        "build", "validate", "launch", "review", "write", "ship", "consider",
        "evaluate", "implement", "schedule", "prioritize", "follow up",
        "try", "start", "stop", "avoid", "practice", "remember", NULL
    };
    static const char *outro_words[] = {
        "thanks for watching", "see you", "subscribe", "like and subscribe",
        "leave a comment", "hit the bell", "next video", "bye",
        "thanks for listening", "see ya", "check out", "link in",
        "follow me", "follow us", "smash that", NULL
    };
    int score = 0;
    size_t len = strlen(text);

    /* Length bonuses */
    if (len > 30) score += 1;
    if (len > 80) score += 1;
    if (len > 200) score -= 1; /* penalize overly long sentences */

    /* Domain keyword bonuses */
    if (contains_any(text, summary_words)) score += 2;
    if (contains_any(text, action_words)) score += 2;
    if (strchr(text, '$') || strstr(text, "percent") || strstr(text, "%")) score += 2;

    /* Filler / low-quality penalties */
    if (strstr(text, "I think") || strstr(text, "i think")) score -= 2;
    if (strstr(text, "yeah") || strstr(text, "Yeah")) score -= 2;
    if (strstr(text, "you know") || strstr(text, "You know")) score -= 2;
    if (strstr(text, "um ") || strstr(text, "uh ") || strstr(text, "hmm")) score -= 1;

    /* Outro / boilerplate penalty */
    if (contains_any(text, outro_words)) score -= 5;

    /* TF-IDF bonus: sentences with rare/distinctive terms score higher */
    double idf = tfidf_score(text, total_docs);
    score += (int)(idf * 2.0); /* scale TF-IDF contribution */

    return score;
}

static int is_action_sentence(const char *text) {
    static const char *action_words[] = {
        "should", "need to", "must", "next", "plan", "test", "focus", "send",
        "build", "validate", "launch", "review", "write", "ship", NULL
    };
    return contains_any(text, action_words);
}

static int cmp_sentence_desc(const void *a, const void *b) {
    const Sentence *left = (const Sentence *)a;
    const Sentence *right = (const Sentence *)b;
    return right->score - left->score;
}

static int split_sentences(const char *text, Sentence *sentences, int max_sentences) {
    int count = 0;
    const char *start = text;
    for (const char *p = text; ; p++) {
        if (*p == '.' || *p == '!' || *p == '?' || *p == '\0') {
            size_t len = (size_t)(p - start + (*p ? 1 : 0));
            if (len > 1) {
                char buffer[MAX_LINE];
                if (len >= sizeof(buffer)) len = sizeof(buffer) - 1;
                memcpy(buffer, start, len);
                buffer[len] = '\0';
                char *trimmed = trim_copy(buffer);
                if (trimmed && trimmed[0] != '\0' && count < max_sentences) {
                    sentences[count].text = trimmed;
                    sentences[count].score = 0; /* scored after IDF built */
                    sentences[count].is_action = is_action_sentence(trimmed);
                    count++;
                } else if (trimmed) {
                    free(trimmed);
                }
            }
            if (*p == '\0') break;
            start = p + 1;
        }
    }
    return count;
}

static void write_brief(const char *path, const char *title, Sentence *sentences, int count) {
    FILE *out = fopen(path, "w");
    if (!out) return;

    Sentence ranked[MAX_SENTENCES];
    memcpy(ranked, sentences, sizeof(Sentence) * count);
    qsort(ranked, count, sizeof(Sentence), cmp_sentence_desc);

    fprintf(out, "# %s\n\n", title);
    fprintf(out, "## Summary\n");
    int summary_written = 0;
    for (int i = 0; i < count && summary_written < MAX_BULLETS; i++) {
        if (ranked[i].score < 2 || ranked[i].is_action) continue;
        fprintf(out, "- %s\n", ranked[i].text);
        summary_written++;
    }
    if (!summary_written) fprintf(out, "- No summary generated\n");

    fprintf(out, "\n## Action Items\n");
    int action_written = 0;
    for (int i = 0; i < count && action_written < MAX_BULLETS; i++) {
        if (!ranked[i].is_action) continue;
        fprintf(out, "- %s\n", ranked[i].text);
        action_written++;
    }
    if (!action_written) fprintf(out, "- No action items detected\n");

    fprintf(out, "\n## Deep Summary\n");
    fprintf(out, "- Key Details\n");
    for (int i = 0; i < count && i < 4; i++) {
        if (ranked[i].score < 1) continue;
        fprintf(out, "  - %s\n", ranked[i].text);
    }

    fprintf(out, "\n## Transcript\n");
    for (int i = 0; i < count; i++) {
        fprintf(out, "%s%s", sentences[i].text, (i + 1 < count) ? " " : "");
    }
    fprintf(out, "\n");
    fclose(out);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: bonfyre-brief <transcript-file> <output-dir> [--title TITLE]\n");
        return 1;
    }

    const char *transcript_path = argv[1];
    const char *output_dir = argv[2];
    const char *title = "Bonfyre Brief";

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--title") == 0 && i + 1 < argc) {
            title = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    if (ensure_dir(output_dir) != 0) {
        fprintf(stderr, "Failed to create output dir: %s\n", output_dir);
        return 1;
    }

    size_t input_len = 0;
    char *input_buffer = bf_read_file(transcript_path, &input_len);
    if (!input_buffer || input_len == 0) {
        free(input_buffer);
        fprintf(stderr, "Failed to read transcript: %s\n", transcript_path);
        return 1;
    }

    char *buffer = NULL;
    if (input_buffer[0] == '{' || input_buffer[0] == '[') {
        buffer = extract_segment_text_json(input_buffer);
    }
    if (!buffer) {
        buffer = input_buffer;
        input_buffer = NULL;
    }
    free(input_buffer);

    Sentence sentences[MAX_SENTENCES] = {0};
    int count = split_sentences(buffer, sentences, MAX_SENTENCES);

    /* Build TF-IDF vocabulary from all sentences, then score */
    g_vocab_count = 0;
    build_idf(sentences, count);
    for (int i = 0; i < count; i++) {
        sentences[i].score = sentence_score(sentences[i].text, count);
        /* Position penalty: first 10% and last 10% of sentences are
         * typically intro/outro fluff. Penalize them. */
        if (count > 10) {
            int tenth = count / 10;
            if (tenth < 1) tenth = 1;
            if (i < tenth || i >= count - tenth)
                sentences[i].score -= 2;
        }
    }

    char brief_path[PATH_MAX];
    char meta_path[PATH_MAX];
    snprintf(brief_path, sizeof(brief_path), "%s/brief.md", output_dir);
    snprintf(meta_path, sizeof(meta_path), "%s/brief-meta.json", output_dir);

    write_brief(brief_path, title, sentences, count);

    char timestamp[32];
    iso_timestamp(timestamp, sizeof(timestamp));
    FILE *meta = fopen(meta_path, "w");
    if (meta) {
        fprintf(meta,
                "{\n"
                "  \"source_system\": \"BonfyreBrief\",\n"
                "  \"created_at\": \"%s\",\n"
                "  \"transcript_path\": \"%s\",\n"
                "  \"brief_path\": \"%s\",\n"
                "  \"sentence_count\": %d\n"
                "}\n",
                timestamp,
                transcript_path,
                brief_path,
                count);
        fclose(meta);
    }

    /* --- Emit artifact.json (Bonfyre universal manifest) --- */
    char artifact_path[PATH_MAX];
    snprintf(artifact_path, sizeof(artifact_path), "%s/artifact.json", output_dir);
    FILE *af = fopen(artifact_path, "w");
    if (af) {
        fprintf(af,
                "{\n"
                "  \"schema_version\": \"1.0.0\",\n"
                "  \"artifact_id\": \"brief-%s\",\n"
                "  \"artifact_type\": \"brief\",\n"
                "  \"created_at\": \"%s\",\n"
                "  \"source_system\": \"BonfyreBrief\",\n"
                "  \"tags\": [\"brief\"],\n"
                "  \"root_hash\": \"\",\n"
                "  \"atoms\": [\n"
                "    {\n"
                "      \"atom_id\": \"source-transcript\",\n"
                "      \"content_hash\": \"\",\n"
                "      \"media_type\": \"text/plain\",\n"
                "      \"path\": \"%s\",\n"
                "      \"label\": \"Source transcript\"\n"
                "    }\n"
                "  ],\n"
                "  \"operators\": [\n"
                "    {\n"
                "      \"operator_id\": \"op-brief-extract\",\n"
                "      \"op\": \"BriefExtract\",\n"
                "      \"inputs\": [\"source-transcript\"],\n"
                "      \"output\": \"brief-md\",\n"
                "      \"params\": {\"top_sentences\": 6, \"top_actions\": 6},\n"
                "      \"version\": \"1.0.0\",\n"
                "      \"deterministic\": true\n"
                "    },\n"
                "    {\n"
                "      \"operator_id\": \"op-brief-meta\",\n"
                "      \"op\": \"MetadataEmit\",\n"
                "      \"inputs\": [\"brief-md\"],\n"
                "      \"output\": \"brief-meta\",\n"
                "      \"params\": {},\n"
                "      \"version\": \"1.0.0\",\n"
                "      \"deterministic\": true\n"
                "    }\n"
                "  ],\n"
                "  \"realizations\": [\n"
                "    {\n"
                "      \"realization_id\": \"brief-md\",\n"
                "      \"media_type\": \"text/markdown\",\n"
                "      \"path\": \"%s\",\n"
                "      \"pinned\": true,\n"
                "      \"produced_by\": \"op-brief-extract\",\n"
                "      \"label\": \"Brief markdown\"\n"
                "    },\n"
                "    {\n"
                "      \"realization_id\": \"brief-meta\",\n"
                "      \"media_type\": \"application/json\",\n"
                "      \"path\": \"%s\",\n"
                "      \"pinned\": true,\n"
                "      \"produced_by\": \"op-brief-meta\",\n"
                "      \"label\": \"Brief metadata\"\n"
                "    }\n"
                "  ],\n"
                "  \"realization_targets\": [\n"
                "    {\n"
                "      \"target_id\": \"narrated-brief\",\n"
                "      \"media_type\": \"audio/wav\",\n"
                "      \"op\": \"Narrate\",\n"
                "      \"params\": {\"profile\": \"operator_status\"},\n"
                "      \"description\": \"Spoken brief via BonfyreNarrate\"\n"
                "    }\n"
                "  ],\n"
                "  \"metadata\": {\n"
                "    \"title\": \"%s\",\n"
                "    \"sentence_count\": %d\n"
                "  }\n"
                "}\n",
                timestamp,     /* artifact_id suffix */
                timestamp,     /* created_at */
                transcript_path,
                brief_path,
                meta_path,
                title,
                count);
        fclose(af);
    }

    printf("Brief: %s\n", brief_path);
    printf("Meta: %s\n", meta_path);
    if (af) printf("Artifact: %s\n", artifact_path);

    for (int i = 0; i < count; i++) free(sentences[i].text);
    free(buffer);
    return 0;
}
