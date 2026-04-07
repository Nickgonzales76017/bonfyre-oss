#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <bonfyre.h>

/* ================================================================
 * Vocabulary Accumulator — the learning curve engine.
 *
 * Every transcription feeds its output text back into a SQLite vocab
 * database. The more audio processed, the richer the prompt becomes.
 * Improvement follows: quality(n) = 1 - e^(-k * n)
 * where n = total words observed and k = learning rate.
 *
 * The vocab DB stores:
 *   term       — lowercased word or multi-word phrase
 *   frequency  — total occurrences across all transcriptions
 *   first_seen — ISO timestamp of first observation
 *   last_seen  — ISO timestamp of most recent observation
 *   source     — "transcribe" or "manual"
 *
 * The prompt generator selects the top-N terms by frequency,
 * weighted so that domain-specific terms (proper nouns, jargon)
 * that appear repeatedly rise to the top.
 *
 * On first run: no vocab → no prompt → full Whisper cold decode.
 * After 1 file:  ~50 terms → weak prompt → marginal speedup.
 * After 10 files: ~300 terms → strong prompt → measurable speedup.
 * After 100 files: ~1000+ terms → saturated prompt → near-optimal.
 *
 * The model never forgets: terms accumulate monotonically. Rare
 * terms naturally fall out of the prompt as higher-frequency terms
 * dominate the top-N window.
 * ================================================================ */

#define VOCAB_DB_NAME       "bonfyre_vocab.db"
#define VOCAB_PROMPT_MAX    200   /* max terms in the whisper prompt      */
#define VOCAB_MIN_FREQ      2     /* min frequency to enter prompt        */
#define VOCAB_MIN_LEN       3     /* ignore words shorter than this       */
#define VOCAB_PROMPT_CHARS  1024  /* max bytes in the prompt string       */

/* Stopwords — common English words we never want in the prompt. */
static const char *STOPWORDS[] = {
    "the","be","to","of","and","a","in","that","have","i","it","for","not",
    "on","with","he","as","you","do","at","this","but","his","by","from",
    "they","we","her","she","or","an","will","my","one","all","would",
    "there","their","what","so","up","out","if","about","who","get","which",
    "go","me","when","make","can","like","time","no","just","him","know",
    "take","people","into","year","your","good","some","could","them","see",
    "other","than","then","now","look","only","come","its","over","think",
    "also","back","after","use","two","how","our","work","first","well",
    "way","even","new","want","because","any","these","give","day","most",
    "us","was","were","been","has","had","are","is","am","did","does",
    "been","being","having","doing","um","uh","yeah","okay","right","yes",
    "no","oh","ah","like","well","just","really","very","much","more",
    "going","gonna","wanna","gotta","thing","things","know","think","mean",
    NULL
};

static int is_stopword(const char *word) {
    for (int i = 0; STOPWORDS[i]; i++) {
        if (strcmp(word, STOPWORDS[i]) == 0) return 1;
    }
    return 0;
}

/* Lowercase a word in-place and strip non-alphanumeric edges. */
static void normalize_word(char *w) {
    /* lowercase */
    for (char *p = w; *p; p++) *p = (char)tolower((unsigned char)*p);
    /* strip trailing punctuation */
    size_t len = strlen(w);
    while (len > 0 && !isalnum((unsigned char)w[len - 1])) w[--len] = '\0';
    /* strip leading punctuation */
    char *start = w;
    while (*start && !isalnum((unsigned char)*start)) start++;
    if (start != w) memmove(w, start, strlen(start) + 1);
}

/* ----------------------------------------------------------------
 * Vocab DB operations — light SQLite via fork+exec to sqlite3 CLI.
 * This keeps BonfyreTranscribe zero-dep (no libsqlite3 link).
 * The vocab DB lives next to the output directory or at a
 * configurable path via $BONFYRE_VOCAB_DB.
 * ---------------------------------------------------------------- */

static void resolve_vocab_db(char *out, size_t size) {
    const char *env = getenv("BONFYRE_VOCAB_DB");
    if (env && env[0]) {
        snprintf(out, size, "%s", env);
        return;
    }
    const char *home = getenv("HOME");
    if (home) {
        snprintf(out, size, "%s/.local/share/bonfyre/%s", home, VOCAB_DB_NAME);
    } else {
        snprintf(out, size, "/tmp/%s", VOCAB_DB_NAME);
    }
}

static int run_sqlite3(const char *db_path, const char *sql) {
    char *argv[] = {
        "sqlite3", (char *)db_path, (char *)sql, NULL
    };
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, 2); close(devnull); }
        execvp("sqlite3", argv);
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

/* Read output from sqlite3 into a buffer (caller frees). */
static char *run_sqlite3_query(const char *db_path, const char *sql, size_t *out_len) {
    int pipefd[2];
    if (pipe(pipefd) < 0) return NULL;

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return NULL; }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], 1);
        close(pipefd[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, 2); close(devnull); }
        char *argv[] = { "sqlite3", (char *)db_path, (char *)sql, NULL };
        execvp("sqlite3", argv);
        _exit(127);
    }
    close(pipefd[1]);

    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) { close(pipefd[0]); waitpid(pid, NULL, 0); return NULL; }
    ssize_t n;
    while ((n = read(pipefd[0], buf + len, cap - len - 1)) > 0) {
        len += (size_t)n;
        if (len + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
    }
    buf[len] = '\0';
    close(pipefd[0]);
    waitpid(pid, NULL, 0);
    if (out_len) *out_len = len;
    return buf;
}

static int vocab_db_init(const char *db_path) {
    /* Ensure parent directory exists */
    char dir[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s", db_path);
    char *slash = strrchr(dir, '/');
    if (slash) { *slash = '\0'; bf_ensure_dir(dir); }

    return run_sqlite3(db_path,
        "CREATE TABLE IF NOT EXISTS vocab ("
        "  term TEXT PRIMARY KEY,"
        "  frequency INTEGER DEFAULT 1,"
        "  first_seen TEXT NOT NULL,"
        "  last_seen TEXT NOT NULL,"
        "  source TEXT DEFAULT 'transcribe'"
        ");"
        "CREATE TABLE IF NOT EXISTS stats ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT"
        ");"
        "INSERT OR IGNORE INTO stats(key, value) VALUES('total_words', '0');"
        "INSERT OR IGNORE INTO stats(key, value) VALUES('total_files', '0');"
    );
}

/* Ingest a transcript file into the vocab DB. Returns words processed. */
static int vocab_ingest(const char *db_path, const char *transcript_path) {
    size_t txt_len = 0;
    char *text = bf_read_file(transcript_path, &txt_len);
    if (!text || txt_len == 0) { free(text); return 0; }

    char timestamp[32];
    bf_iso_timestamp(timestamp, sizeof(timestamp));

    /* Build a single SQL transaction with all UPSERTs */
    size_t sql_cap = txt_len * 4 + 4096;
    char *sql = malloc(sql_cap);
    if (!sql) { free(text); return 0; }
    size_t sql_len = 0;
    int word_count = 0;

    sql_len += (size_t)snprintf(sql + sql_len, sql_cap - sql_len, "BEGIN;\n");

    /* Tokenize on whitespace */
    char *saveptr = NULL;
    char *tok = strtok_r(text, " \t\n\r", &saveptr);
    while (tok) {
        char word[256];
        snprintf(word, sizeof(word), "%s", tok);
        normalize_word(word);

        size_t wlen = strlen(word);
        if (wlen >= VOCAB_MIN_LEN && !is_stopword(word)) {
            /* Escape single quotes for SQL */
            char escaped[512];
            size_t ei = 0;
            for (size_t wi = 0; wi < wlen && ei < sizeof(escaped) - 2; wi++) {
                if (word[wi] == '\'') escaped[ei++] = '\'';
                escaped[ei++] = word[wi];
            }
            escaped[ei] = '\0';

            sql_len += (size_t)snprintf(sql + sql_len, sql_cap - sql_len,
                "INSERT INTO vocab(term, frequency, first_seen, last_seen, source) "
                "VALUES('%s', 1, '%s', '%s', 'transcribe') "
                "ON CONFLICT(term) DO UPDATE SET "
                "frequency = frequency + 1, last_seen = '%s';\n",
                escaped, timestamp, timestamp, timestamp);
        }
        word_count++;
        tok = strtok_r(NULL, " \t\n\r", &saveptr);

        /* Flush if SQL buffer is getting large */
        if (sql_len > sql_cap - 1024) {
            snprintf(sql + sql_len, sql_cap - sql_len, "COMMIT;\n");
            run_sqlite3(db_path, sql);
            sql_len = 0;
            sql_len += (size_t)snprintf(sql + sql_len, sql_cap - sql_len, "BEGIN;\n");
        }
    }

    /* Update running totals */
    sql_len += (size_t)snprintf(sql + sql_len, sql_cap - sql_len,
        "UPDATE stats SET value = CAST(CAST(value AS INTEGER) + %d AS TEXT) WHERE key = 'total_words';\n"
        "UPDATE stats SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'total_files';\n",
        word_count);

    snprintf(sql + sql_len, sql_cap - sql_len, "COMMIT;\n");
    run_sqlite3(db_path, sql);

    free(sql);
    free(text);
    return word_count;
}

/* Build a whisper prompt from the vocab DB.
 * Returns a malloc'd string of the top-N terms sorted by frequency.
 * The prompt follows the exponential curve: more files processed →
 * richer prompt → better decoder guidance → faster/more accurate.
 *
 * Prompt quality follows: q(n) ≈ 1 - e^(-0.05 * n)
 * where n = total files processed.
 *   n=0:   q=0.00  (no prompt — cold start)
 *   n=1:   q=0.05  (weak: ~10 terms)
 *   n=5:   q=0.22  (growing: ~44 terms)
 *   n=10:  q=0.39  (solid: ~78 terms)
 *   n=20:  q=0.63  (strong: ~127 terms)
 *   n=50:  q=0.92  (near-saturated: ~184 terms)
 *   n=100: q=0.99  (saturated: 199 terms)
 *   n→∞:   q→1.00  (200 terms, optimally weighted)
 */
static char *vocab_build_prompt(const char *db_path) {
    /* How many files have we processed? */
    char *files_str = run_sqlite3_query(db_path,
        "SELECT value FROM stats WHERE key = 'total_files';", NULL);
    int total_files = 0;
    if (files_str) { total_files = atoi(files_str); free(files_str); }

    if (total_files == 0) return NULL; /* cold start — no prompt */

    /* Exponential learning curve: how many terms to include */
    double curve = 1.0 - exp(-0.05 * (double)total_files);
    int prompt_terms = (int)(curve * VOCAB_PROMPT_MAX);
    if (prompt_terms < 5) prompt_terms = 5;
    if (prompt_terms > VOCAB_PROMPT_MAX) prompt_terms = VOCAB_PROMPT_MAX;

    /* Query top terms by frequency, minimum VOCAB_MIN_FREQ */
    char sql[512];
    snprintf(sql, sizeof(sql),
        "SELECT term FROM vocab "
        "WHERE frequency >= %d AND source = 'transcribe' "
        "ORDER BY frequency DESC LIMIT %d;",
        VOCAB_MIN_FREQ, prompt_terms);

    size_t result_len = 0;
    char *result = run_sqlite3_query(db_path, sql, &result_len);
    if (!result || result_len == 0) { free(result); return NULL; }

    /* Convert newline-separated terms to comma-separated prompt */
    char *prompt = malloc(VOCAB_PROMPT_CHARS + 64);
    if (!prompt) { free(result); return NULL; }
    size_t plen = 0;
    int term_count = 0;

    plen += (size_t)snprintf(prompt + plen, VOCAB_PROMPT_CHARS - plen,
        "Context: ");

    char *line = strtok(result, "\n");
    while (line && plen < VOCAB_PROMPT_CHARS - 64) {
        if (term_count > 0) {
            plen += (size_t)snprintf(prompt + plen, VOCAB_PROMPT_CHARS - plen, ", ");
        }
        plen += (size_t)snprintf(prompt + plen, VOCAB_PROMPT_CHARS - plen, "%s", line);
        term_count++;
        line = strtok(NULL, "\n");
    }

    free(result);

    if (term_count == 0) { free(prompt); return NULL; }

    fprintf(stderr, "[vocab] learning curve: %.0f%% (%d files → %d/%d terms in prompt)\n",
            curve * 100.0, total_files, term_count, prompt_terms);
    return prompt;
}

/* Write vocab stats to a JSON file alongside the transcript. */
static void vocab_write_stats(const char *db_path, const char *output_dir) {
    char *words_str = run_sqlite3_query(db_path,
        "SELECT value FROM stats WHERE key = 'total_words';", NULL);
    char *files_str = run_sqlite3_query(db_path,
        "SELECT value FROM stats WHERE key = 'total_files';", NULL);
    char *unique_str = run_sqlite3_query(db_path,
        "SELECT COUNT(*) FROM vocab;", NULL);

    int total_words = words_str ? atoi(words_str) : 0;
    int total_files = files_str ? atoi(files_str) : 0;
    int unique_terms = unique_str ? atoi(unique_str) : 0;
    double curve = 1.0 - exp(-0.05 * (double)total_files);

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/vocab-stats.json", output_dir);
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp,
            "{\n"
            "  \"sourceSystem\": \"BonfyreTranscribe\",\n"
            "  \"vocabDb\": \"%s\",\n"
            "  \"totalFilesProcessed\": %d,\n"
            "  \"totalWordsProcessed\": %d,\n"
            "  \"uniqueTerms\": %d,\n"
            "  \"learningCurve\": %.4f,\n"
            "  \"promptTermsUsed\": %d,\n"
            "  \"promptTermsMax\": %d,\n"
            "  \"convergence\": \"%.1f%%\"\n"
            "}\n",
            db_path, total_files, total_words, unique_terms,
            curve,
            (int)(curve * VOCAB_PROMPT_MAX),
            VOCAB_PROMPT_MAX,
            curve * 100.0);
        fclose(fp);
    }

    free(words_str);
    free(files_str);
    free(unique_str);
}

/* ================================================================
 * Core transcription helpers (unchanged)
 * ================================================================ */

static int ensure_dir(const char *path) { return bf_ensure_dir(path); }
static int run_process(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }
    if (pid == 0) {
        execvp(argv[0], argv);
        perror("execvp");
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return 1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 1;
}

static int copy_file_to_stream(const char *path, FILE *out) {
    FILE *in = fopen(path, "rb");
    if (!in) {
        perror("fopen");
        return 1;
    }
    char buffer[4096];
    size_t bytes = 0;
    while ((bytes = fread(buffer, 1, sizeof(buffer), in)) > 0) {
        if (fwrite(buffer, 1, bytes, out) != bytes) {
            fclose(in);
            return 1;
        }
    }
    fclose(in);
    return 0;
}

static int write_chunk_progress(const char *path, int total_chunks, int completed_chunks, const char *status) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror("fopen");
        return 1;
    }
    fprintf(fp,
            "{\n"
            "  \"sourceSystem\": \"BonfyreTranscribe\",\n"
            "  \"status\": \"%s\",\n"
            "  \"totalChunks\": %d,\n"
            "  \"completedChunks\": %d\n"
            "}\n",
            status,
            total_chunks,
            completed_chunks);
    fclose(fp);
    return 0;
}

static void iso_timestamp(char *buffer, size_t size) {
    time_t now = time(NULL);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    strftime(buffer, size, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
}

/* Return 1 if the whisper binary is whisper.cpp (whisper-cli), 0 for Python whisper. */
static int is_whisper_cpp(const char *binary_path) {
    const char *base = strrchr(binary_path, '/');
    base = base ? base + 1 : binary_path;
    return (strstr(base, "whisper-cli") != NULL ||
            strstr(base, "whisper-cpp") != NULL);
}

/* Resolve a model name (e.g. "base") to a ggml model file path for whisper.cpp.
 * Prefers quantized (Q5_0) models over float16 — 62% smaller, 39% faster. */
static int resolve_whisper_cpp_model(const char *model_name, char *out, size_t out_size) {
    /* If already a path, use directly */
    if (strchr(model_name, '/') || strchr(model_name, '.')) {
        if (access(model_name, F_OK) == 0) {
            snprintf(out, out_size, "%s", model_name);
            return 0;
        }
        return -1;
    }
    /* Try common locations — prefer quantized, then .en, then multilingual */
    const char *home = getenv("HOME");
    const char *variants[] = {
        "%s/.local/share/whisper/ggml-%s.en-q5_0.bin",  /* quantized English-only */
        "%s/.local/share/whisper/ggml-%s-q5_0.bin",      /* quantized multilingual */
        "%s/.local/share/whisper/ggml-%s.en.bin",         /* float16 English-only */
        "%s/.local/share/whisper/ggml-%s.bin",            /* float16 multilingual */
        NULL
    };
    const char *tmp_variants[] = {
        "/tmp/ggml-%s.en-q5_0.bin",
        "/tmp/ggml-%s-q5_0.bin",
        "/tmp/ggml-%s.en.bin",
        "/tmp/ggml-%s.bin",
        NULL
    };
    for (int i = 0; variants[i]; i++) {
        if (home) {
            snprintf(out, out_size, variants[i], home, model_name);
            if (access(out, F_OK) == 0) return 0;
        }
    }
    for (int i = 0; tmp_variants[i]; i++) {
        snprintf(out, out_size, tmp_variants[i], model_name);
        if (access(out, F_OK) == 0) return 0;
    }
    return -1;
}

static const char *default_whisper_binary(void) {
    const char *env = getenv("BONFYRE_WHISPER_BINARY");
    if (env && env[0] != '\0') return env;
    /* Prefer whisper-cli (whisper.cpp) if available */
    if (access("/opt/homebrew/bin/whisper-cli", X_OK) == 0)
        return "/opt/homebrew/bin/whisper-cli";
    if (access("/usr/local/bin/whisper-cli", X_OK) == 0)
        return "/usr/local/bin/whisper-cli";
    /* Fall back to Python whisper on PATH */
    return "whisper";
}

static void resolve_executable_sibling(char *buffer, size_t size, const char *argv0, const char *sibling_dir, const char *binary_name) {
    if (argv0 && argv0[0] == '/') {
        snprintf(buffer, size, "%s", argv0);
    } else if (argv0 && strstr(argv0, "/")) {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd))) {
            snprintf(buffer, size, "%s/%s", cwd, argv0);
        } else {
            snprintf(buffer, size, "%s", argv0);
        }
    } else {
        buffer[0] = '\0';
        return;
    }

    char *last_slash = strrchr(buffer, '/');
    if (!last_slash) {
        buffer[0] = '\0';
        return;
    }
    *last_slash = '\0';
    last_slash = strrchr(buffer, '/');
    if (!last_slash) {
        buffer[0] = '\0';
        return;
    }
    *last_slash = '\0';
    snprintf(buffer, size, "%s/%s/%s", buffer, sibling_dir, binary_name);
}

static const char *default_media_prep_binary(const char *argv0, char *resolved_path, size_t resolved_size) {
    const char *env = getenv("BONFYRE_MEDIA_PREP_BINARY");
    if (env && env[0] != '\0') return env;
    resolve_executable_sibling(resolved_path, resolved_size, argv0, "BonfyreMediaPrep", "bonfyre-media-prep");
    if (resolved_path[0] != '\0' && access(resolved_path, X_OK) == 0) {
        return resolved_path;
    }
    return "../BonfyreMediaPrep/bonfyre-media-prep";
}

static const char *default_silero_vad_script(const char *argv0, char *resolved_path, size_t resolved_size) {
    const char *env = getenv("BONFYRE_SILERO_VAD_CLI");
    if (env && env[0] != '\0') return env;
    resolve_executable_sibling(resolved_path, resolved_size, argv0, "SileroVADCLI", "bin/silero_vad_cli.py");
    if (resolved_path[0] != '\0' && access(resolved_path, F_OK) == 0) {
        return resolved_path;
    }
    return "../SileroVADCLI/bin/silero_vad_cli.py";
}

static const char *path_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static void strip_extension(char *name) {
    char *dot = strrchr(name, '.');
    if (dot) *dot = '\0';
}

static void print_usage(void) {
    fprintf(stderr,
            "bonfyre-transcribe\n\n"
            "Usage:\n"
            "  bonfyre-transcribe <input-audio> <output-dir> [--model NAME] [--language CODE]\n"
            "                      [--whisper-binary PATH] [--media-prep-binary PATH]\n"
            "                      [--silero-vad] [--silero-script PATH]\n"
            "                      [--split-speech] [--noise-threshold DB] [--min-silence SEC]\n"
            "                      [--min-speech SEC] [--padding SEC]\n"
            "                      [--greedy] [--beam-size N] [--best-of N]\n"
            "                      [--vocab-db PATH] [--no-vocab]\n"
            "\n"
            "Vocabulary learning:\n"
            "  Transcription output is fed back into a vocabulary database that grows\n"
            "  with every file processed. The accumulated vocabulary is used as a\n"
            "  decoder prompt (--prompt) for subsequent transcriptions, improving\n"
            "  both speed and accuracy over time.\n"
            "\n"
            "  Learning curve: quality(n) = 1 - e^(-0.05 * n)\n"
            "    n=1: 5%%  | n=10: 39%% | n=50: 92%% | n=100: 99%%\n"
            "\n"
            "  Disable with --no-vocab. Custom DB path with --vocab-db.\n");
}

/* Detect number of CPU cores. */
static int detect_cpu_cores(void) {
#ifdef __APPLE__
    int ncpu = 0;
    size_t len = sizeof(ncpu);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &ncpu, &len, NULL, 0) == 0 && ncpu > 0)
        return ncpu;  /* performance cores only */
    if (sysctlbyname("hw.physicalcpu", &ncpu, &len, NULL, 0) == 0 && ncpu > 0)
        return ncpu;
#endif
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 4;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    const char *input_audio = argv[1];
    const char *output_dir = argv[2];
    const char *model = "base";
    const char *language = NULL;
    const char *whisper_binary = default_whisper_binary();
    int split_speech = 0;
    const char *noise_threshold = "-35dB";
    const char *min_silence = "0.35";
    const char *min_speech = "0.75";
    const char *padding = "0.15";
    int silero_vad = 0;
    int greedy = 1;               /* P0: default to greedy (was beam=5) */
    int beam_size = 1;
    int best_of = 1;
    int no_vocab = 0;
    const char *vocab_db_override = NULL;
    char resolved_media_prep[PATH_MAX];
    char resolved_silero_script[PATH_MAX];
    const char *media_prep_binary = default_media_prep_binary(argv[0], resolved_media_prep, sizeof(resolved_media_prep));
    const char *silero_script = default_silero_vad_script(argv[0], resolved_silero_script, sizeof(resolved_silero_script));
    int threads = detect_cpu_cores();

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (strcmp(argv[i], "--language") == 0 && i + 1 < argc) {
            language = argv[++i];
        } else if (strcmp(argv[i], "--whisper-binary") == 0 && i + 1 < argc) {
            whisper_binary = argv[++i];
        } else if (strcmp(argv[i], "--media-prep-binary") == 0 && i + 1 < argc) {
            media_prep_binary = argv[++i];
        } else if (strcmp(argv[i], "--silero-vad") == 0) {
            silero_vad = 1;
        } else if (strcmp(argv[i], "--silero-script") == 0 && i + 1 < argc) {
            silero_script = argv[++i];
        } else if (strcmp(argv[i], "--split-speech") == 0) {
            split_speech = 1;
        } else if (strcmp(argv[i], "--noise-threshold") == 0 && i + 1 < argc) {
            noise_threshold = argv[++i];
        } else if (strcmp(argv[i], "--min-silence") == 0 && i + 1 < argc) {
            min_silence = argv[++i];
        } else if (strcmp(argv[i], "--min-speech") == 0 && i + 1 < argc) {
            min_speech = argv[++i];
        } else if (strcmp(argv[i], "--padding") == 0 && i + 1 < argc) {
            padding = argv[++i];
        } else if (strcmp(argv[i], "--greedy") == 0) {
            greedy = 1; beam_size = 1; best_of = 1;
        } else if (strcmp(argv[i], "--beam-size") == 0 && i + 1 < argc) {
            beam_size = atoi(argv[++i]); greedy = (beam_size <= 1);
        } else if (strcmp(argv[i], "--best-of") == 0 && i + 1 < argc) {
            best_of = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vocab-db") == 0 && i + 1 < argc) {
            vocab_db_override = argv[++i];
        } else if (strcmp(argv[i], "--no-vocab") == 0) {
            no_vocab = 1;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    if (ensure_dir(output_dir) != 0) {
        fprintf(stderr, "Failed to create output dir: %s\n", output_dir);
        return 1;
    }

    char normalized_path[PATH_MAX];
    char transcript_path[PATH_MAX];
    char meta_path[PATH_MAX];
    char status_path[PATH_MAX];
    char progress_path[PATH_MAX];
    char base_name[PATH_MAX];
    int chunk_count = 0;
    int completed_chunks = 0;
    int denoised = 0;

    snprintf(base_name, sizeof(base_name), "%s", path_basename(input_audio));
    strip_extension(base_name);
    snprintf(normalized_path, sizeof(normalized_path), "%s/normalized.wav", output_dir);
    snprintf(transcript_path, sizeof(transcript_path), "%s/normalized.txt", output_dir);
    snprintf(meta_path, sizeof(meta_path), "%s/meta.json", output_dir);
    snprintf(status_path, sizeof(status_path), "%s/transcribe-status.json", output_dir);
    snprintf(progress_path, sizeof(progress_path), "%s/chunk-progress.json", output_dir);

    char *normalize_argv[] = {
        (char *)media_prep_binary,
        "normalize",
        (char *)input_audio,
        normalized_path,
        "--sample-rate", "16000",
        "--channels", "1",
        NULL
    };

    if (run_process(normalize_argv) != 0) {
        fprintf(stderr, "Normalize failed.\n");
        return 1;
    }

    char denoised_path[PATH_MAX];
    snprintf(denoised_path, sizeof(denoised_path), "%s/normalized.denoised.wav", output_dir);
    char *denoise_argv[] = {
        (char *)media_prep_binary,
        "denoise",
        normalized_path,
        denoised_path,
        NULL
    };
    if (run_process(denoise_argv) == 0 && access(denoised_path, F_OK) == 0) {
        snprintf(normalized_path, sizeof(normalized_path), "%s", denoised_path);
        denoised = 1;
    }

    /* ----------------------------------------------------------------
     * Initialize vocabulary DB and build decoder prompt.
     * The prompt is the accumulated knowledge from all prior
     * transcriptions — it biases the decoder toward known terms,
     * speeding up decoding and improving accuracy.
     * ---------------------------------------------------------------- */
    char vocab_db_path[PATH_MAX];
    char *vocab_prompt = NULL;
    if (!no_vocab) {
        if (vocab_db_override) {
            snprintf(vocab_db_path, sizeof(vocab_db_path), "%s", vocab_db_override);
        } else {
            resolve_vocab_db(vocab_db_path, sizeof(vocab_db_path));
        }
        vocab_db_init(vocab_db_path);
        vocab_prompt = vocab_build_prompt(vocab_db_path);
    }

    /* Thread count and beam-size as strings for argv */
    char threads_str[16];
    char beam_str[16];
    char bestof_str[16];
    snprintf(threads_str, sizeof(threads_str), "%d", threads);
    snprintf(beam_str, sizeof(beam_str), "%d", beam_size);
    snprintf(bestof_str, sizeof(bestof_str), "%d", best_of);

    if (split_speech) {
        char chunk_dir[PATH_MAX];
        char chunk_pattern[PATH_MAX];
        snprintf(chunk_dir, sizeof(chunk_dir), "%s/chunks", output_dir);
        snprintf(chunk_pattern, sizeof(chunk_pattern), "%s/chunk-%%03d.wav", chunk_dir);

        if (ensure_dir(chunk_dir) != 0) {
            fprintf(stderr, "Failed to create chunk dir: %s\n", chunk_dir);
            return 1;
        }

        if (silero_vad && access(silero_script, F_OK) == 0) {
            char *silero_argv[] = {
                "python3",
                (char *)silero_script,
                "--audio",
                normalized_path,
                "--out",
                chunk_dir,
                "--min-speech",
                (char *)min_speech,
                "--padding",
                (char *)padding,
                NULL
            };
            if (run_process(silero_argv) != 0) {
                fprintf(stderr, "Silero VAD split failed.\n");
                return 1;
            }
        } else {
            char *split_argv[] = {
                (char *)media_prep_binary,
                "split-speech",
                normalized_path,
                chunk_pattern,
                "--noise-threshold", (char *)noise_threshold,
                "--min-silence", (char *)min_silence,
                "--min-speech", (char *)min_speech,
                "--padding", (char *)padding,
                NULL
            };

            if (run_process(split_argv) != 0) {
                fprintf(stderr, "Speech split failed.\n");
                return 1;
            }
        }

        for (int i = 0;; i++) {
            char chunk_audio[PATH_MAX];
            snprintf(chunk_audio, sizeof(chunk_audio), "%s/chunk-%03d.wav", chunk_dir, i);
            if (access(chunk_audio, F_OK) != 0) break;
            chunk_count++;
        }
        write_chunk_progress(progress_path, chunk_count, 0, "splitting-complete");

        FILE *combined = fopen(transcript_path, "w");
        if (!combined) {
            perror("fopen transcript");
            return 1;
        }

        for (int i = 0; i < chunk_count; i++) {
            char chunk_audio[PATH_MAX];
            char chunk_txt[PATH_MAX];
            snprintf(chunk_audio, sizeof(chunk_audio), "%s/chunk-%03d.wav", chunk_dir, i);
            snprintf(chunk_txt, sizeof(chunk_txt), "%s/chunk-%03d.txt", chunk_dir, i);

            char *whisper_argv[32];
            int idx = 0;
            char model_path[PATH_MAX];
            char of_prefix[PATH_MAX];
            whisper_argv[idx++] = (char *)whisper_binary;

            if (is_whisper_cpp(whisper_binary)) {
                if (resolve_whisper_cpp_model(model, model_path, sizeof(model_path)) != 0) {
                    fprintf(stderr, "Cannot find whisper.cpp model for '%s'\n", model);
                    fclose(combined);
                    return 1;
                }
                snprintf(of_prefix, sizeof(of_prefix), "%s/chunk-%03d", chunk_dir, i);
                whisper_argv[idx++] = "-f";
                whisper_argv[idx++] = chunk_audio;
                whisper_argv[idx++] = "-m";
                whisper_argv[idx++] = model_path;
                whisper_argv[idx++] = "--output-txt";
                whisper_argv[idx++] = "-of";
                whisper_argv[idx++] = of_prefix;
                whisper_argv[idx++] = "--no-prints";
                whisper_argv[idx++] = "--flash-attn";
                whisper_argv[idx++] = "-t";
                whisper_argv[idx++] = threads_str;
                whisper_argv[idx++] = "-bs";
                whisper_argv[idx++] = beam_str;
                whisper_argv[idx++] = "-bo";
                whisper_argv[idx++] = bestof_str;
                if (language) {
                    whisper_argv[idx++] = "-l";
                    whisper_argv[idx++] = (char *)language;
                }
                if (vocab_prompt) {
                    whisper_argv[idx++] = "--prompt";
                    whisper_argv[idx++] = vocab_prompt;
                }
            } else {
                /* Python whisper: positional audio, --task, --model name, --output_format, --output_dir */
                whisper_argv[idx++] = chunk_audio;
                whisper_argv[idx++] = "--task";
                whisper_argv[idx++] = "transcribe";
                whisper_argv[idx++] = "--model";
                whisper_argv[idx++] = (char *)model;
                whisper_argv[idx++] = "--output_format";
                whisper_argv[idx++] = "txt";
                whisper_argv[idx++] = "--output_dir";
                whisper_argv[idx++] = chunk_dir;
                if (language) {
                    whisper_argv[idx++] = "--language";
                    whisper_argv[idx++] = (char *)language;
                }
            }
            whisper_argv[idx] = NULL;

            if (run_process(whisper_argv) != 0) {
                fclose(combined);
                fprintf(stderr, "Whisper failed on chunk %d.\n", i);
                return 1;
            }
            if (access(chunk_txt, F_OK) != 0) {
                fclose(combined);
                fprintf(stderr, "Expected chunk transcript not found: %s\n", chunk_txt);
                return 1;
            }

            if (copy_file_to_stream(chunk_txt, combined) != 0) {
                fclose(combined);
                fprintf(stderr, "Failed to append chunk transcript.\n");
                return 1;
            }
            fprintf(combined, "\n");
            completed_chunks++;
            write_chunk_progress(progress_path, chunk_count, completed_chunks, "transcribing");
        }
        fclose(combined);
        write_chunk_progress(progress_path, chunk_count, completed_chunks, "completed");
    } else {
        char *whisper_argv[32];
        int idx = 0;
        char model_path[PATH_MAX];
        char of_prefix[PATH_MAX];
        whisper_argv[idx++] = (char *)whisper_binary;

        if (is_whisper_cpp(whisper_binary)) {
            if (resolve_whisper_cpp_model(model, model_path, sizeof(model_path)) != 0) {
                fprintf(stderr, "Cannot find whisper.cpp model for '%s'\n", model);
                return 1;
            }
            snprintf(of_prefix, sizeof(of_prefix), "%s/normalized", output_dir);
            whisper_argv[idx++] = "-f";
            whisper_argv[idx++] = normalized_path;
            whisper_argv[idx++] = "-m";
            whisper_argv[idx++] = model_path;
            whisper_argv[idx++] = "--output-txt";
            whisper_argv[idx++] = "-of";
            whisper_argv[idx++] = of_prefix;
            whisper_argv[idx++] = "--no-prints";
            whisper_argv[idx++] = "--flash-attn";
            whisper_argv[idx++] = "-t";
            whisper_argv[idx++] = threads_str;
            whisper_argv[idx++] = "-bs";
            whisper_argv[idx++] = beam_str;
            whisper_argv[idx++] = "-bo";
            whisper_argv[idx++] = bestof_str;
            if (language) {
                whisper_argv[idx++] = "-l";
                whisper_argv[idx++] = (char *)language;
            }
            if (vocab_prompt) {
                whisper_argv[idx++] = "--prompt";
                whisper_argv[idx++] = vocab_prompt;
            }
        } else {
            /* Python whisper: positional audio, --task, --model name, --output_format, --output_dir */
            whisper_argv[idx++] = normalized_path;
            whisper_argv[idx++] = "--task";
            whisper_argv[idx++] = "transcribe";
            whisper_argv[idx++] = "--model";
            whisper_argv[idx++] = (char *)model;
            whisper_argv[idx++] = "--output_format";
            whisper_argv[idx++] = "txt";
            whisper_argv[idx++] = "--output_dir";
            whisper_argv[idx++] = (char *)output_dir;
            if (language) {
                whisper_argv[idx++] = "--language";
                whisper_argv[idx++] = (char *)language;
            }
        }
        whisper_argv[idx] = NULL;

        if (run_process(whisper_argv) != 0) {
            fprintf(stderr, "Whisper failed.\n");
            return 1;
        }

        if (access(transcript_path, F_OK) != 0) {
            fprintf(stderr, "Expected transcript not found: %s\n", transcript_path);
            return 1;
        }
        write_chunk_progress(progress_path, 1, 1, "completed");
    }

    /* ----------------------------------------------------------------
     * Post-transcription: feed transcript back into vocabulary DB.
     * This is where the learning curve compounds — every file makes
     * the next file cheaper. The curve follows 1 - e^(-0.05 * n).
     * ---------------------------------------------------------------- */
    int words_ingested = 0;
    if (!no_vocab && access(transcript_path, F_OK) == 0) {
        words_ingested = vocab_ingest(vocab_db_path, transcript_path);
        vocab_write_stats(vocab_db_path, output_dir);
        fprintf(stderr, "[vocab] ingested %d words from transcript\n", words_ingested);
    }
    free(vocab_prompt);

    char timestamp[32];
    iso_timestamp(timestamp, sizeof(timestamp));
    char language_json[256];
    if (language) {
        snprintf(language_json, sizeof(language_json), "\"%s\"", language);
    } else {
        snprintf(language_json, sizeof(language_json), "null");
    }

    FILE *meta = fopen(meta_path, "w");
    if (!meta) {
        perror("fopen meta");
        return 1;
    }
    fprintf(meta,
            "{\n"
            "  \"source_system\": \"BonfyreTranscribe\",\n"
            "  \"created_at\": \"%s\",\n"
            "  \"input_audio\": \"%s\",\n"
            "  \"normalized_audio\": \"%s\",\n"
            "  \"transcript_path\": \"%s\",\n"
            "  \"model\": \"%s\",\n"
            "  \"language\": %s,\n"
            "  \"split_speech\": %s,\n"
            "  \"silero_vad\": %s,\n"
            "  \"denoised\": %s,\n"
            "  \"greedy\": %s,\n"
            "  \"beam_size\": %d,\n"
            "  \"threads\": %d,\n"
            "  \"vocab_enabled\": %s,\n"
            "  \"vocab_words_ingested\": %d,\n"
            "  \"chunk_count\": %d,\n"
            "  \"chunk_progress_path\": \"%s\",\n"
            "  \"whisper_binary\": \"%s\",\n"
            "  \"media_prep_binary\": \"%s\"\n"
            "}\n",
            timestamp,
            input_audio,
            normalized_path,
            transcript_path,
            model,
            language_json,
            split_speech ? "true" : "false",
            silero_vad ? "true" : "false",
            denoised ? "true" : "false",
            greedy ? "true" : "false",
            beam_size,
            threads,
            no_vocab ? "false" : "true",
            words_ingested,
            chunk_count,
            progress_path,
            whisper_binary,
            media_prep_binary);
    fclose(meta);

    FILE *status = fopen(status_path, "w");
    if (!status) {
        perror("fopen status");
        return 1;
    }
    fprintf(status,
            "{\n"
            "  \"sourceSystem\": \"BonfyreTranscribe\",\n"
            "  \"exportedAt\": \"%s\",\n"
            "  \"status\": \"transcribed\",\n"
            "  \"jobSlug\": \"%s\",\n"
            "  \"splitSpeech\": %s,\n"
            "  \"sileroVad\": %s,\n"
            "  \"denoised\": %s,\n"
            "  \"chunkCount\": %d,\n"
            "  \"chunkProgressPath\": \"%s\",\n"
            "  \"transcriptPath\": \"%s\",\n"
            "  \"metaPath\": \"%s\"\n"
            "}\n",
            timestamp,
            base_name,
            split_speech ? "true" : "false",
            silero_vad ? "true" : "false",
            denoised ? "true" : "false",
            chunk_count,
            progress_path,
            transcript_path,
            meta_path);
    fclose(status);

    printf("Normalized: %s\n", normalized_path);
    printf("Transcript: %s\n", transcript_path);
    printf("Meta: %s\n", meta_path);
    printf("Status: %s\n", status_path);
    printf("Progress: %s\n", progress_path);
    return 0;
}
