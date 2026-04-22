/*
 * bf_lattice_accel.h — SIMD-accelerated batch E8 lattice quantization
 *
 * Public API for the NEON / AVX2 accelerated batch nearest-E8-point search.
 * The "frustrated-boundary" heuristic routes groups that lie very close to
 * a lattice boundary (dist to D8+½ coset ≈ dist to D8 coset within tol) to
 * the relaxed-parity variant so the parity fix never amplifies outlier error.
 */
#pragma once
#ifndef BF_LATTICE_ACCEL_H
#define BF_LATTICE_ACCEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * bf_e8_batch_simd()
 *
 * Snap N groups of 8 floats each to the nearest E8 lattice point.
 *
 *   in  — Nx8 floats (row-major, 8-aligned preferred)
 *   out — Nx8 floats, in-place results
 *   n   — number of 8D groups
 *
 * Selects NEON, AVX2, or scalar path at compile time.
 * Returns 0 on success.
 */
int bf_e8_batch_simd(const float *in, float *out, size_t n);

/*
 * bf_e8_batch_simd_frustrated()
 *
 * Same as above but applies the "frustrated boundary" heuristic:
 * groups where |dist_D8 - dist_D8half| < tol are quantized with the
 * relaxed-parity variant (no parity fix) to avoid parity-fix outlier
 * amplification.
 *
 *   in, out, n — same as bf_e8_batch_simd
 *   tol         — boundary tolerance (e.g. 0.05f). 0 = standard E8 for all.
 *
 * Returns number of groups that used the relaxed path (for diagnostics).
 */
int bf_e8_batch_simd_frustrated(const float *in, float *out, size_t n,
                                 float tol, int *relaxed_count_out);

/*
 * bf_lattice_accel_backend()
 *
 * Returns a static string describing the active SIMD backend:
 *   "avx2", "neon", or "scalar"
 */
const char *bf_lattice_accel_backend(void);

#ifdef __cplusplus
}
#endif
#endif /* BF_LATTICE_ACCEL_H */
