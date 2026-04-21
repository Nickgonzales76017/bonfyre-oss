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
 *
 * bf_mmap_prefetch() can be called at pipeline startup to issue
 * MADV_WILLNEED for a set of known-needed paths, asking the kernel to
 * begin prefaulting pages while the process does other setup work.
 * Beyond that, caching is the OS's job; we don't re-implement it.
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
        m->ptr = (void *)"";
        m->len = 0;
        m->fd  = fd;
        return 0;
    }

    void *ptr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) { close(fd); return -1; }

    madvise(ptr, (size_t)st.st_size, MADV_SEQUENTIAL);

    m->ptr = ptr;
    m->len = (size_t)st.st_size;
    m->fd  = fd;
    return 0;
}

void bf_mmap_close(BfMmapFile *m) {
    if (!m) return;
    if (m->ptr && m->len > 0)
        munmap(m->ptr, m->len);
    if (m->fd >= 0)
        close(m->fd);
    m->ptr = NULL;
    m->len = 0;
    m->fd  = -1;
}

/* ================================================================
 * bf_bfrec_mmap — zero-copy .bfrec record read.
 * Caller must bf_mmap_close(m) when done.
 * ================================================================ */
const BfBinaryRecord *bf_bfrec_mmap(const char *path, BfMmapFile *m) {
    if (!path || !m) return NULL;
    if (bf_mmap_open(m, path) != 0) return NULL;

    if (m->len != sizeof(BfBinaryRecord)) {
        bf_mmap_close(m);
        return NULL;
    }

    const BfBinaryRecord *rec = (const BfBinaryRecord *)m->ptr;
    if (strncmp(rec->magic, BF_BINARY_MAGIC, BF_MAGIC_LEN) != 0) {
        bf_mmap_close(m);
        return NULL;
    }
    return rec;
}

/* ================================================================
 * bf_mmap_prefetch
 *
 * Issue MADV_WILLNEED for each path so the kernel begins prefaulting
 * pages asynchronously.  Call once at pipeline startup for the set of
 * files the pipeline will read; returns number of paths advised.
 * Non-fatal — failed opens are silently skipped.
 * ================================================================ */
int bf_mmap_prefetch(const char * const *paths, int n) {
    if (!paths || n <= 0) return 0;
    int ok = 0;
    for (int i = 0; i < n; i++) {
        if (!paths[i]) continue;
        int fd = open(paths[i], O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            void *ptr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (ptr != MAP_FAILED) {
                madvise(ptr, (size_t)st.st_size, MADV_WILLNEED);
                munmap(ptr, (size_t)st.st_size);
            }
        }
        close(fd);
        ok++;
    }
    return ok;
}


#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE  /* madvise + MADV_SEQUENTIAL on macOS */
