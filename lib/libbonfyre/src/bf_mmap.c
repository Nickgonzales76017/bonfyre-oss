/*
 * bf_mmap.c — zero-copy mmap layer for Bonfyre hot paths.
 *
 * bf_lmdb reads are pointer casts, not memcpy.
 * bf_bfrec_mmap returns a pointer directly into the mmap'd .bfrec page.
 * Hot-path artifact reads are allocation-free.
 *
 * On POSIX, mmap(PROT_READ, MAP_PRIVATE) lets the OS page cache serve
 * as the buffer.  For sequential reads (SHA-256 hashing, artifact parsing)
 * this eliminates the user-space copy that read()/fread() would incur.
 *
 * For files already in the page cache (the common case for hot .bfrec
 * records read by many pipeline stages), no disk I/O occurs at all —
 * the mmap just installs the virtual mapping and SIMD/SHA code walks
 * the pages directly.
 */

#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE  /* madvise + MADV_SEQUENTIAL on macOS */
#include "bonfyre.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ================================================================
 * bf_mmap_open / bf_mmap_close
 * ================================================================ */
int bf_mmap_open(BfMmapFile *m, const char *path) {
    if (!m || !path) { errno = EINVAL; return -1; }
    m->ptr = NULL;
    m->len = 0;
    m->fd  = -1;

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }

    if (st.st_size == 0) {
        /* Empty file: valid, but mmap(len=0) is undefined.
         * Return a non-NULL sentinel so callers can distinguish from error. */
        m->ptr = (void *)""; /* static read-only byte — never written */
        m->len = 0;
        m->fd  = fd;
        return 0;
    }

    void *ptr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) { close(fd); return -1; }

    /* Advise sequential access for linear scan workloads (SHA-256, JSON scan) */
    madvise(ptr, (size_t)st.st_size, MADV_SEQUENTIAL);

    m->ptr = ptr;
    m->len = (size_t)st.st_size;
    m->fd  = fd;
    return 0;
}

void bf_mmap_close(BfMmapFile *m) {
    if (!m) return;
    /* Don't munmap the static sentinel (len == 0 && fd >= 0 → empty file) */
    if (m->ptr && m->len > 0)
        munmap(m->ptr, m->len);
    if (m->fd >= 0)
        close(m->fd);
    m->ptr = NULL;
    m->len = 0;
    m->fd  = -1;
}

/* ================================================================
 * bf_bfrec_mmap
 *
 * Zero-copy hot path for artifact binary cache records.
 *
 * .bfrec files contain exactly one BfBinaryRecord struct written by
 * save_manifest_binary().  Instead of fopen + fread + fclose (which
 * copies sizeof(BfBinaryRecord) ≈ 700 bytes through the kernel
 * buffer and a stack/heap copy), this function:
 *
 *   1. mmap()s the file (installs virtual mapping — no data copy)
 *   2. Validates magic bytes in the mapped page (pointer arithmetic)
 *   3. Returns a typed pointer DIRECTLY into the mmap'd page
 *
 * The struct fields are readable without any allocation or copy.
 * The caller MUST call bf_mmap_close(m) when done to release the
 * mapping; the returned pointer becomes invalid after that.
 *
 * Returns NULL if:
 *   - file cannot be opened / mapped
 *   - file size ≠ sizeof(BfBinaryRecord)  (corrupt or truncated)
 *   - magic bytes mismatch
 * ================================================================ */
const BfBinaryRecord *bf_bfrec_mmap(const char *path, BfMmapFile *m) {
    if (!path || !m) return NULL;
    if (bf_mmap_open(m, path) != 0) return NULL;

    /* Size guard: must be exactly one record */
    if (m->len != sizeof(BfBinaryRecord)) {
        bf_mmap_close(m);
        return NULL;
    }

    /* Magic check: pointer cast, zero copy */
    const BfBinaryRecord *rec = (const BfBinaryRecord *)m->ptr;
    if (strncmp(rec->magic, BF_BINARY_MAGIC, BF_MAGIC_LEN) != 0) {
        bf_mmap_close(m);
        return NULL;
    }
    return rec; /* caller walks struct fields directly in mmap'd page */
}
