/*
 * e8_relax.h — Relaxed-Constraint E8 Lattice Quantizer (header)
 *
 * Relaxed variant of the Conway-Sloane E8 nearest-point algorithm.
 * Removes strict parity enforcement; optionally applies Hadamard rotation
 * to spread outlier energy before snapping.
 *
 * See e8_relax.c for full technical commentary.
 */
#pragma once
#include <stddef.h>

/* Single 8D vector — relaxed parity, no Hadamard (fastest) */
void  e8_relax_snap(const float *x, float *out);

/* Single 8D vector — relaxed parity + H8 pre/post rotation (best quality) */
void  e8_relax_snap_h(const float *x, float *out);

/* Batch: N groups of 8 floats; use_h=1 enables Hadamard path */
void  e8_relax_snap_batch(const float *groups, float *out, int n, int use_h);

/* Diagnostic: RMSE between n8 groups of 8 floats before and after snap */
float e8_relax_rmse(const float *x, const float *snapped, int n8);
