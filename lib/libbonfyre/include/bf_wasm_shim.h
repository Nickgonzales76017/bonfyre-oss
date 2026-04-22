/*
 * bf_wasm_shim.h — WebAssembly compatibility layer for Bonfyre
 *
 * Provides:
 *   1. BF_WASM_EXPORT macro — marks functions for Emscripten export
 *   2. BF_WASM_ALLOC / BF_WASM_FREE — allocation wrappers safe for WASM heap
 *   3. bonfyre_wasm_init()    — initialise in-memory SQLite + pipeline state
 *   4. bonfyre_wasm_run()     — base64-decode input, execute recipe, return JSON
 *   5. bonfyre_wasm_version() — return VERSION string
 *
 * Build with:
 *   emcc -O2 -s WASM=1 -s EXPORTED_FUNCTIONS='["_bonfyre_wasm_run",
 *         "_bonfyre_wasm_init","_bonfyre_wasm_version"]'
 *         -s EXTRA_EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' *         -D BF_WASM_BUILD=1
 *         src/bf_wasm_shim.c lib-a -o bonfyre-runtime.js
 *
 * Platform guards:
 *   BF_WASM_BUILD — defined by emcc builds; activates Emscripten-specific APIs.
 *   If not defined, all exported symbols are available as normal C functions
 *   so native builds can call the same API for testing.
 */
#pragma once
#ifndef BF_WASM_SHIM_H
#define BF_WASM_SHIM_H

#include <stddef.h>
#include <stdint.h>

/* ── Export macro ─────────────────────────────────────────────────────────── */

#ifdef BF_WASM_BUILD
  /* Emscripten: EMSCRIPTEN_KEEPALIVE prevents dead-code elimination */
  #include <emscripten.h>
  #define BF_WASM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
  /* Native build: use GCC visibility to match Emscripten semantics */
  #define BF_WASM_EXPORT __attribute__((visibility("default")))
#endif

/* ── Alloc ────────────────────────────────────────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate n bytes on the WASM/native heap.  Used by JS glue layer to
 * allocate buffers that C code will fill. */
BF_WASM_EXPORT void *bonfyre_wasm_alloc(size_t n);

/* Free a pointer returned by bonfyre_wasm_alloc or bonfyre_wasm_run. */
BF_WASM_EXPORT void  bonfyre_wasm_free(void *ptr);

/* ── Lifecycle ────────────────────────────────────────────────────────────── */

/*
 * Initialise the WASM runtime.  Call once before any bonfyre_wasm_run().
 *   wasm_data_dir — path prefix for runtime data in the WASM VFS
 *                   (use "/" or "/data" in Emscripten FS)
 * Returns 0 on success, -1 on error.
 */
BF_WASM_EXPORT int bonfyre_wasm_init(const char *wasm_data_dir);

/*
 * Run a pipeline entirely in-process.
 *
 *   recipe_yaml  — NUL-terminated recipe YAML (same format as bonfyre-run)
 *   input_b64    — base64-encoded input file bytes (audio, text, etc.)
 *   input_mime   — MIME type hint: "audio/wav", "text/plain", etc.
 *
 * Returns a heap-allocated NUL-terminated JSON string:
 *   {"status":"ok","output":"<b64>","artifact":{...}}
 *   {"status":"error","message":"..."}
 *
 * Caller must call bonfyre_wasm_free() on the returned pointer.
 * Thread safety: NOT thread-safe (WASM is single-threaded by default).
 */
BF_WASM_EXPORT char *bonfyre_wasm_run(const char *recipe_yaml,
                                       const char *input_b64,
                                       const char *input_mime);

/*
 * Return a static NUL-terminated version string, e.g. "1.0.0-wasm".
 */
BF_WASM_EXPORT const char *bonfyre_wasm_version(void);

/*
 * Return a static NUL-terminated JSON of all available capabilities
 * (same data as bonfyre-gen --list-capabilities but in JSON).
 */
BF_WASM_EXPORT const char *bonfyre_wasm_capabilities(void);

#ifdef __cplusplus
}
#endif
#endif /* BF_WASM_SHIM_H */
