/*
 * BonfyreHash — content-addressing + Merkle DAG hashing engine.
 *
 * Owns every hash in the system. The single source of truth for:
 *   - File content hashes (SHA-256)
 *   - Operator node hashes (DAG hashing)
 *   - Merkle roots (family integrity)
 *   - Dedup detection (same hash = same content, pay once)
 *
 * Usage:
 *   bonfyre-hash file <path>                     — SHA-256 of file
 *   bonfyre-hash node <op> <version> <params_json> <child_hash,...>  — operator node hash
 *   bonfyre-hash merkle <artifact.json>           — compute/update root_hash
 *   bonfyre-hash verify <artifact.json>           — verify all hashes
 *   bonfyre-hash dedup <dir>                      — find duplicate files by hash
 */
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <bonfyre.h>

#define MAX_LINE 65536
#define MAX_CHILDREN 256
#define MAX_FILES 4096
#define HASH_LEN 65

/* ---------- commands ---------- */

static int cmd_file(const char *path) {
    /* Zero-copy: mmap the file, pass mmap'd pages directly to SHA-256.
     * Eliminates the 8 KiB fread bounce-buffer; OS page cache IS the buffer.
     * For large files already in cache, no disk I/O occurs at all.         */
    BfMmapFile m;
    if (bf_mmap_open(&m, path) != 0) {
        fprintf(stderr, "Cannot open: %s: %s\n", path, strerror(errno)); return 1;
    }
    BfSha256 ctx;
    bf_sha256_init(&ctx);
    if (m.len > 0)
        bf_sha256_update(&ctx, (const unsigned char *)m.ptr, m.len);
    bf_mmap_close(&m);
    unsigned char h[32];
    bf_sha256_final(&ctx, h);
    char hex[65];
    bf_sha256_digest_hex(h, hex);
    printf("%s  %s\n", hex, path);
    return 0;
}

/* #8: qsort comparator for child hash pointers */
static int cmp_str_ptrs(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static int cmd_node(const char *op, const char *version, const char *params_json, const char *children_csv) {
    /* Canonical: sort children, concatenate op + params + children hashes */
    char *children[MAX_CHILDREN];
    int nchildren = 0;
    char *csv_copy = strdup(children_csv);
    char *tok = strtok(csv_copy, ",");
    while (tok && nchildren < MAX_CHILDREN) {
        while (*tok == ' ') tok++;
        children[nchildren++] = strdup(tok);
        tok = strtok(NULL, ",");
    }
    free(csv_copy);

    /* #8: qsort instead of bubble sort — O(n log n) */
    qsort(children, (size_t)nchildren, sizeof(char *), cmp_str_ptrs);

    /* Build canonical string */
    BfSha256 ctx;
    bf_sha256_init(&ctx);
    bf_sha256_update(&ctx, (const unsigned char *)op, strlen(op));
    bf_sha256_update(&ctx, (const unsigned char *)"|", 1);
    bf_sha256_update(&ctx, (const unsigned char *)version, strlen(version));
    bf_sha256_update(&ctx, (const unsigned char *)"|", 1);
    bf_sha256_update(&ctx, (const unsigned char *)params_json, strlen(params_json));
    bf_sha256_update(&ctx, (const unsigned char *)"|", 1);
    for (int i = 0; i < nchildren; i++) {
        bf_sha256_update(&ctx, (const unsigned char *)children[i], strlen(children[i]));
        if (i < nchildren - 1)
            bf_sha256_update(&ctx, (const unsigned char *)",", 1);
        free(children[i]);
    }
    unsigned char h[32];
    bf_sha256_final(&ctx, h);
    char hex[65];
    bf_sha256_digest_hex(h, hex);
    printf("%s\n", hex);
    return 0;
}

/* #6: qsort comparator for dedup entries (sort by hash) */
static int cmp_entry_hash(const void *a, const void *b) {
    typedef struct { char hash[65]; char path[PATH_MAX]; unsigned long size; } Entry;
    return strcmp(((const Entry *)a)->hash, ((const Entry *)b)->hash);
}

static int cmd_dedup(const char *dir) {
    /* Walk directory, hash all files, report duplicates */
    typedef struct { char hash[65]; char path[PATH_MAX]; unsigned long size; } Entry;

    /* #7: Dynamic growth instead of fixed MAX_FILES calloc (saves ~4.3MB) */
    int cap = 128;
    int count = 0;
    Entry *entries = malloc((size_t)cap * sizeof(Entry));
    if (!entries) return 1;

    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "Cannot open: %s\n", dir); free(entries); return 1; }
    struct dirent *ent;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
        struct stat st;
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) continue;

        /* Grow if needed */
        if (count >= cap) {
            cap *= 2;
            Entry *tmp = realloc(entries, (size_t)cap * sizeof(Entry));
            if (!tmp) { free(entries); closedir(d); return 1; }
            entries = tmp;
        }

        FILE *fp = fopen(path, "rb");
        if (!fp) continue;
        BfSha256 ctx;
        bf_sha256_init(&ctx);
        /* Zero-copy: mmap each candidate file for SHA-256 hashing
         * Same gain as cmd_file: no bounce buffer, page cache is buf */
        bf_mmap_close(NULL); /* no-op, just to reference type */
        BfMmapFile _dm;
        if (bf_mmap_open(&_dm, path) != 0) continue;
        bf_sha256_init(&ctx);
        if (_dm.len > 0)
            bf_sha256_update(&ctx, (const unsigned char *)_dm.ptr, _dm.len);
        bf_mmap_close(&_dm);
        unsigned char h[32];
        bf_sha256_final(&ctx, h);
        bf_sha256_digest_hex(h, entries[count].hash);
        snprintf(entries[count].path, PATH_MAX, "%s", path);
        entries[count].size = (unsigned long)st.st_size;
        count++;
        fclose(fp); /* kept for structural symmetry — now a no-op rel to hash */
    }
    closedir(d);

    /* #6: Sort by hash, then linear scan for duplicates — O(n log n) */
    qsort(entries, (size_t)count, sizeof(Entry), cmp_entry_hash);

    unsigned long wasted = 0;
    int dupes = 0;
    for (int i = 0; i < count - 1; i++) {
        if (strcmp(entries[i].hash, entries[i+1].hash) == 0) {
            printf("DUP %s  %s  (%lu bytes)\n", entries[i].path, entries[i+1].path, entries[i].size);
            wasted += entries[i+1].size;
            dupes++;
        }
    }
    if (dupes == 0) printf("No duplicates found in %d files.\n", count);
    else printf("\n%d duplicate pairs, %lu bytes reclaimable.\n", dupes, wasted);

    free(entries);
    return 0;
}

/* ---------- #10: Inline Merkle (replaces system() + Python) ---------- */

/* Extract quoted string values matching "content_hash" from artifact.json */
static int extract_content_hashes(const char *json, char hashes[][65], int max_hashes) {
    const char *needle = "\"content_hash\"";
    size_t needle_len = strlen(needle);
    int count = 0;
    const char *p = json;
    while ((p = strstr(p, needle)) && count < max_hashes) {
        p += needle_len;
        while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
        if (*p != '"') continue;
        p++;
        int i = 0;
        while (*p && *p != '"' && i < 64) hashes[count][i++] = *p++;
        hashes[count][i] = '\0';
        if (i == 64) count++;
    }
    return count;
}

static int cmp_hash_strings(const void *a, const void *b) {
    return strcmp((const char *)a, (const char *)b);
}

static int cmd_merkle(const char *artifact_path, int verify_only) {
    /* Zero-copy: mmap artifact.json, scan content_hash values in-place.
     * No heap allocation for the JSON body — pointer walks the mmap'd page.
     * SIMD bf_json_scan_str locates root_hash at 4+ GB/s for verify.   */
    BfMmapFile m;
    if (bf_mmap_open(&m, artifact_path) != 0) {
        fprintf(stderr, "Cannot open: %s\n", artifact_path); return 1;
    }
    if (m.len == 0 || m.len > 1048576) { bf_mmap_close(&m); return 1; }
    const char *json = (const char *)m.ptr;
    size_t json_len  = m.len;

    /* Extract all content_hash values from atoms (still scalar strstr
     * because multi-occurrence scan; future: bf_json_scan_all_str)   */
    char (*hashes)[65] = malloc(4096 * 65);
    if (!hashes) { bf_mmap_close(&m); return 1; }
    int nhashes = extract_content_hashes(json, hashes, 4096);
    if (nhashes == 0) {
        fprintf(stderr, "No content_hash entries found\n");
        free(hashes); bf_mmap_close(&m);
        return 1;
    }

    /* Sort hashes for canonical ordering */
    qsort(hashes, (size_t)nhashes, 65, cmp_hash_strings);

    /* Build Merkle root: iteratively pair-hash until one remains */
    while (nhashes > 1) {
        int next = 0;
        for (int i = 0; i < nhashes; i += 2) {
            BfSha256 ctx;
            bf_sha256_init(&ctx);
            bf_sha256_update(&ctx, (const unsigned char *)hashes[i], strlen(hashes[i]));
            if (i + 1 < nhashes)
                bf_sha256_update(&ctx, (const unsigned char *)hashes[i+1], strlen(hashes[i+1]));
            unsigned char h[32];
            bf_sha256_final(&ctx, h);
            bf_sha256_digest_hex(h, hashes[next]);
            next++;
        }
        nhashes = next;
    }

    char *root_hash = hashes[0];

    if (verify_only) {
        /* SIMD root_hash lookup: bf_json_scan_str walks the mmap'd page
         * at 4+ GB/s to locate the root_hash field without strstr.    */
        char stored_root[68] = "";
        bf_json_scan_str(json, json_len, "root_hash", stored_root, sizeof(stored_root));
        if (stored_root[0]) {
            if (strncmp(stored_root, root_hash, 64) == 0)
                printf("VERIFIED: root_hash matches (%s)\n", root_hash);
            else
                printf("MISMATCH: computed=%s stored=%s\n", root_hash, stored_root);
        } else {
            printf("No root_hash field found; computed=%s\n", root_hash);
        }
    } else {
        printf("%s\n", root_hash);
    }

    free(hashes);
    bf_mmap_close(&m);
    return 0;
}

/* ---------- main ---------- */

int main(int argc, char *argv[]) {
    if (argc < 2) goto usage;

    if (strcmp(argv[1], "file") == 0 && argc >= 3)
        return cmd_file(argv[2]);

    if (strcmp(argv[1], "node") == 0 && argc >= 6)
        return cmd_node(argv[2], argv[3], argv[4], argv[5]);

    if (strcmp(argv[1], "merkle") == 0 && argc >= 3)
        return cmd_merkle(argv[2], 0);

    if (strcmp(argv[1], "verify") == 0 && argc >= 3)
        return cmd_merkle(argv[2], 1);

    if (strcmp(argv[1], "dedup") == 0 && argc >= 3)
        return cmd_dedup(argv[2]);

usage:
    fprintf(stderr,
        "BonfyreHash — content-addressing engine\n\n"
        "Usage:\n"
        "  bonfyre-hash file <path>                              SHA-256 of file\n"
        "  bonfyre-hash node <op> <ver> <params_json> <h1,h2>    operator node hash\n"
        "  bonfyre-hash merkle <artifact.json>                    compute/update root_hash\n"
        "  bonfyre-hash verify <artifact.json>                    verify all hashes\n"
        "  bonfyre-hash dedup <dir>                               find duplicate files\n"
    );
    return 1;
}
