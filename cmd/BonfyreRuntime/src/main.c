#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void path_join(char *buffer, size_t size, const char *left, const char *right) {
    snprintf(buffer, size, "%s/%s", left, right);
}

static void resolve_executable_sibling(char *buffer, size_t size, const char *argv0, const char *sibling_dir, const char *binary_name) {
    if (argv0 && argv0[0] == '/') snprintf(buffer, size, "%s", argv0);
    else if (argv0 && strstr(argv0, "/")) {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd))) snprintf(buffer, size, "%s/%s", cwd, argv0);
        else snprintf(buffer, size, "%s", argv0);
    } else {
        buffer[0] = '\0';
        return;
    }
    char *last = strrchr(buffer, '/');
    if (!last) { buffer[0] = '\0'; return; }
    *last = '\0';
    last = strrchr(buffer, '/');
    if (!last) { buffer[0] = '\0'; return; }
    *last = '\0';
    snprintf(buffer, size, "%s/%s/%s", buffer, sibling_dir, binary_name);
}

static const char *default_binary(const char *env_name, const char *argv0, char *resolved, size_t resolved_size, const char *dir, const char *name, const char *fallback) {
    const char *env = getenv(env_name);
    if (env && env[0] != '\0') return env;
    resolve_executable_sibling(resolved, resolved_size, argv0, dir, name);
    if (resolved[0] != '\0' && access(resolved, X_OK) == 0) return resolved;
    return fallback;
}

static int run_command(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        execv(argv[0], argv);
        perror(argv[0]);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return 1;
    if (!WIFEXITED(status)) return 1;
    return WEXITSTATUS(status);
}

static int run_command_to_file(char *const argv[], const char *output_path) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        FILE *fp = fopen(output_path, "w");
        if (!fp) _exit(127);
        if (dup2(fileno(fp), STDOUT_FILENO) < 0) _exit(127);
        fclose(fp);
        execv(argv[0], argv);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return 1;
    if (!WIFEXITED(status)) return 1;
    return WEXITSTATUS(status);
}

static void print_usage(void) {
    fprintf(stderr,
            "bonfyre-runtime\n\n"
            "Usage:\n"
            "  bonfyre-runtime run <input> [pipeline args...]\n"
            "  bonfyre-runtime run-ledger <input> [pipeline args...]\n"
            "  bonfyre-runtime queue <queue args...>\n"
            "  bonfyre-runtime ledger <ledger args...>\n"
            "  bonfyre-runtime loop <N> <binary> [args...]\n"
            "  bonfyre-runtime parallel [-- cmd args...]...\n"
            "\n"
            "  loop:     runs <binary> N times; passes previous artifact.json as --in\n"
            "            to each subsequent iteration.\n"
            "  parallel: forks all '-- cmd args...' groups concurrently; waits for\n"
            "            all to finish; returns 0 only if every child exited 0.\n"
            "            Separate independent pipeline stages with '--'.\n");
}

static void print_usage(void);
static int  cmd_parallel(int argc, char **argv);

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    char queue_resolved[PATH_MAX];
    char pipeline_resolved[PATH_MAX];
    char ledger_resolved[PATH_MAX];
    const char *queue_bin = default_binary("BONFYRE_QUEUE_BINARY", argv[0], queue_resolved, sizeof(queue_resolved), "BonfyreQueue", "bonfyre-queue", "../BonfyreQueue/bonfyre-queue");
    const char *pipeline_bin = default_binary("BONFYRE_PIPELINE_BINARY", argv[0], pipeline_resolved, sizeof(pipeline_resolved), "BonfyrePipeline", "bonfyre-pipeline", "../BonfyrePipeline/bonfyre-pipeline");
    const char *ledger_bin = default_binary("BONFYRE_LEDGER_BINARY", argv[0], ledger_resolved, sizeof(ledger_resolved), "BonfyreLedger", "bonfyre-ledger", "../BonfyreLedger/bonfyre-ledger");

    if (strcmp(argv[1], "queue") == 0) {
        if (argc < 3) return 1;
        char **child = calloc((size_t)argc, sizeof(char *));
        if (!child) return 1;
        child[0] = (char *)queue_bin;
        for (int i = 2; i < argc; i++) child[i - 1] = argv[i];
        int rc = run_command(child);
        free(child);
        return rc;
    }

    if (strcmp(argv[1], "ledger") == 0) {
        if (argc < 3) return 1;
        char **child = calloc((size_t)argc, sizeof(char *));
        if (!child) return 1;
        child[0] = (char *)ledger_bin;
        for (int i = 2; i < argc; i++) child[i - 1] = argv[i];
        int rc = run_command(child);
        free(child);
        return rc;
    }

    if (strcmp(argv[1], "run") == 0 || strcmp(argv[1], "run-ledger") == 0) {
        if (argc < 3) {
            print_usage();
            return 1;
        }
        const int with_ledger = (strcmp(argv[1], "run-ledger") == 0);
        const char *input = argv[2];
        const char *out_dir = NULL;
        for (int i = 3; i < argc - 1; i++) {
            if (strcmp(argv[i], "--out") == 0) out_dir = argv[i + 1];
        }
        if (!out_dir) {
            fprintf(stderr, "run and run-ledger require --out DIR\n");
            return 1;
        }

        char **pipeline_argv = calloc((size_t)argc + 2, sizeof(char *));
        if (!pipeline_argv) return 1;
        int p = 0;
        pipeline_argv[p++] = (char *)pipeline_bin;
        pipeline_argv[p++] = "run";
        pipeline_argv[p++] = (char *)input;
        for (int i = 3; i < argc; i++) pipeline_argv[p++] = argv[i];
        pipeline_argv[p] = NULL;
        int rc = run_command(pipeline_argv);
        free(pipeline_argv);
        if (rc != 0 || !with_ledger) return rc;

        char artifact_path[PATH_MAX];
        char ledger_json[PATH_MAX];
        path_join(artifact_path, sizeof(artifact_path), out_dir, "artifact.json");
        path_join(ledger_json, sizeof(ledger_json), out_dir, "ledger-assessment.json");
        char *ledger_argv[] = {
            (char *)ledger_bin,
            "assess-json",
            artifact_path,
            NULL
        };
        return run_command_to_file(ledger_argv, ledger_json);
    }

    if (strcmp(argv[1], "loop") == 0) {
        if (argc < 4) {
            fprintf(stderr, "usage: bonfyre-runtime loop <N> <binary> [args...]\n");
            return 1;
        }
        int n_iters = atoi(argv[2]);
        if (n_iters <= 0 || n_iters > 1000) {
            fprintf(stderr, "loop: N must be 1..1000 (got %s)\n", argv[2]);
            return 1;
        }
        const char *binary = argv[3];

        /* Find --out DIR in user args, if any */
        const char *base_out = NULL;
        for (int i = 4; i < argc - 1; i++) {
            if (strcmp(argv[i], "--out") == 0) { base_out = argv[i + 1]; break; }
        }

        char prev_artifact[PATH_MAX];
        prev_artifact[0] = '\0';

        pid_t self_pid = getpid();

        for (int iter = 1; iter <= n_iters; iter++) {
            /* Build output dir for this iteration */
            char iter_out[PATH_MAX];
            if (base_out) {
                snprintf(iter_out, sizeof(iter_out), "%s-%d", base_out, iter);
            } else {
                snprintf(iter_out, sizeof(iter_out),
                         "/tmp/bonfyre-loop-%d-%d", (int)self_pid, iter);
            }

            /* mkdir -p iter_out */
            {
                char tmp[PATH_MAX];
                snprintf(tmp, sizeof(tmp), "%s", iter_out);
                for (char *p = tmp + 1; *p; p++) {
                    if (*p == '/') {
                        *p = '\0';
                        mkdir(tmp, 0755);
                        *p = '/';
                    }
                }
                mkdir(tmp, 0755);
            }

            /* Build child argv:
             *   binary [original args, with --out replaced by iter_out]
             *   [--in prev_artifact  if iter > 1]
             */
            int extra = (iter > 1) ? 2 : 0;  /* --in <path> */
            char **child = (char **)calloc((size_t)(argc - 4 + 3 + extra + 2),
                                           sizeof(char *));
            if (!child) return 1;
            int ci = 0;
            child[ci++] = (char *)binary;

            int skip_next = 0;
            for (int i = 4; i < argc; i++) {
                if (skip_next) { skip_next = 0; continue; }
                if (strcmp(argv[i], "--out") == 0) {
                    child[ci++] = "--out";
                    child[ci++] = iter_out;
                    skip_next = 1; /* skip original DIR */
                } else {
                    child[ci++] = argv[i];
                }
            }
            if (!base_out) {
                child[ci++] = "--out";
                child[ci++] = iter_out;
            }
            if (iter > 1 && prev_artifact[0]) {
                child[ci++] = "--in";
                child[ci++] = prev_artifact;
            }
            child[ci] = NULL;

            fprintf(stderr, "bonfyre-runtime loop [%d/%d]: %s\n",
                    iter, n_iters, binary);

            int rc = run_command(child);
            free(child);
            if (rc != 0) {
                fprintf(stderr, "bonfyre-runtime loop: iteration %d failed (rc=%d)\n",
                        iter, rc);
                return rc;
            }

            /* Next iteration's --in = this iteration's artifact.json */
            path_join(prev_artifact, sizeof(prev_artifact),
                      iter_out, "artifact.json");
            /* If no artifact.json was written, clear so --in is not passed */
            if (access(prev_artifact, F_OK) != 0) prev_artifact[0] = '\0';
        }

        fprintf(stderr, "bonfyre-runtime loop: completed %d iterations\n", n_iters);
        return 0;
    }

    if (strcmp(argv[1], "parallel") == 0) {
        return cmd_parallel(argc - 2, argv + 2);
    }

    print_usage();
    return 1;
}

/* ================================================================
 * parallel subcommand
 *
 * bonfyre-runtime parallel [-- binary arg...] [-- binary arg...] ...
 *
 * Parses the argv into groups delimited by "--".  Forks each group
 * simultaneously, then collects all exit codes.  Returns 0 only if
 * every child exited with status 0.  This lets independent pipeline
 * stages (transcription, hashing, ledger update) overlap their I/O
 * and CPU work.
 *
 * Max groups: 64 (enough for any realistic pipeline fan-out).
 * ================================================================ */
#define PAR_MAX_GROUPS 64

static int cmd_parallel(int argc, char **argv) {
    /* Collect group start indices in remaining argv (after "parallel") */
    int group_starts[PAR_MAX_GROUPS];
    int group_count = 0;

    int i = 0;
    while (i < argc) {
        if (strcmp(argv[i], "--") == 0) {
            i++;
            if (i < argc && group_count < PAR_MAX_GROUPS) {
                group_starts[group_count++] = i;
            }
            continue;
        }
        /* First argument without a leading "--" also starts an implicit group */
        if (group_count == 0 && group_count < PAR_MAX_GROUPS) {
            group_starts[group_count++] = i;
        }
        i++;
    }

    if (group_count == 0) {
        fprintf(stderr, "bonfyre-runtime parallel: no commands given\n");
        return 1;
    }

    pid_t pids[PAR_MAX_GROUPS];

    /* Compute extent of each group (from its start to the next "--" or end) */
    for (int g = 0; g < group_count; g++) {
        int start = group_starts[g];
        /* Find end: next "--" marker */
        int end = argc;
        for (int j = start; j < argc; j++) {
            if (strcmp(argv[j], "--") == 0) { end = j; break; }
        }
        int len = end - start;
        if (len <= 0) { pids[g] = -1; continue; }

        /* Build null-terminated argv for execv */
        char **cargv = calloc((size_t)(len + 1), sizeof(char *));
        if (!cargv) {
            fprintf(stderr, "bonfyre-runtime parallel: OOM\n");
            /* Kill already-forked children */
            for (int k = 0; k < g; k++) if (pids[k] > 0) kill(pids[k], SIGTERM);
            return 1;
        }
        for (int j = 0; j < len; j++) cargv[j] = argv[start + j];
        cargv[len] = NULL;

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            free(cargv);
            for (int k = 0; k < g; k++) if (pids[k] > 0) kill(pids[k], SIGTERM);
            return 1;
        }
        if (pid == 0) {
            execv(cargv[0], cargv);
            perror(cargv[0]);
            _exit(127);
        }
        free(cargv);
        pids[g] = pid;
    }

    /* Collect all children */
    int overall = 0;
    for (int g = 0; g < group_count; g++) {
        if (pids[g] <= 0) continue;
        int st = 0;
        waitpid(pids[g], &st, 0);
        int code = WIFEXITED(st) ? WEXITSTATUS(st) : 1;
        if (code != 0) {
            fprintf(stderr, "bonfyre-runtime parallel: group %d exited %d\n", g, code);
            overall = code;
        }
    }
    return overall;
}
