#!/usr/bin/env python3
"""
gen_bqfp_family.py — Generate a synthetic BQFP file for a named transform family.

Usage:
  python3 gen_bqfp_family.py <family> <output.bqfp> [--seed N] [--n-elements N]

The BQFP binary format (v1):
  Header (16 bytes):  magic(4) version(4) n_tensors(4) bits(4)
  Per tensor:
    name[256] n_elements[8] n_blocks[8] eff_k[4] pad[4]
    codebook[eff_k * 16 * 4]
    blocks[n_blocks] — each block = scale(4)+warp_norm(4)+e8i[256](1 each)+tile_idx[16](1 each)

We synthesize distinct weight distributions per family so that alignment
matrices computed via bonfyre-fpqx are geometrically meaningful.

Family distribution parameters:
  T04: Gaussian N(0, 0.8)  — broad, global
  T15: Gaussian N(0, 0.6)  — slightly tighter, global
  T16: Gaussian N(0.1, 0.5) + 0.3 * cos(k/16)  — biased, conditional (long-doc)
"""

import struct
import sys
import math
import random

BLOCK_DIM = 256
TILE_DIM  = 16
E8_PAIRS  = 16   # 256 / 16
BQFP_MAGIC   = 0x50464251  # "BQFP" LE
BQFP_VERSION = 1
BITS = 3

MU_BETA = 8.0


def mu_warp(x, beta=MU_BETA):
    return math.copysign(math.log1p(beta * abs(x)) / math.log1p(beta), x)


def mu_unwarp(y, beta=MU_BETA):
    return math.copysign((math.exp(abs(y) * math.log1p(beta)) - 1.0) / beta, y)


def fwht(a):
    """Fast Walsh-Hadamard Transform (in-place, unnormalized)."""
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j], a[j + h] = x + y, x - y
        h *= 2
    norm = math.sqrt(n)
    for i in range(n): a[i] /= norm


def e8_snap(v):
    """Snap a float to the nearest E8 lattice half-integer."""
    # E8 integer snap: round to nearest integer, then fix parity
    iv = [round(x) for x in v]
    s = sum(iv) % 2
    if s != 0:
        # Flip the element with largest fractional deviation
        min_idx = min(range(len(v)), key=lambda i: abs(v[i] - iv[i]))
        iv[min_idx] = iv[min_idx] + 1 if v[min_idx] > iv[min_idx] else iv[min_idx] - 1
    return iv


def encode_block(weights, haar_seed, lattice_scale):
    """Encode one BLOCK_DIM block. Returns (scale, warp_norm, e8i[256], tile_idx[16])."""
    # Random sign flip (mirrors bonfyre-quant xorshift64)
    buf = list(weights) + [0.0] * (BLOCK_DIM - len(weights))
    rng = haar_seed & 0xFFFFFFFFFFFFFFFF

    def xorshift64(s):
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= (s >> 7)  & 0xFFFFFFFFFFFFFFFF
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        return s

    signs = []
    for _ in range(BLOCK_DIM // 64):
        rng = xorshift64(rng)
        bits = rng
        for b in range(64):
            signs.append(-1.0 if (bits >> b) & 1 else 1.0)
    for i in range(BLOCK_DIM): buf[i] *= signs[i]

    fwht(buf)

    rms = math.sqrt(sum(x*x for x in buf) / BLOCK_DIM)
    if rms < 1e-30: rms = 1.0
    scale = rms

    warped = [mu_warp(x / rms) for x in buf]
    wn = math.sqrt(sum(x*x for x in warped) / BLOCK_DIM)
    if wn < 1e-30: wn = 1.0
    warp_norm = wn

    e8i = []
    for g in range(32):  # 32 groups of E8_DIM=8
        chunk = [warped[g*8 + d] / wn * lattice_scale for d in range(8)]
        snapped = e8_snap(chunk)
        for val in snapped:
            iv = max(-128, min(127, int(val * 2)))  # store ×2 for half-int precision
            e8i.append(iv)

    # Tile residuals: placeholder uniform codebook (tile_idx all 0 for synthetic)
    tile_idx = [0] * E8_PAIRS

    return scale, warp_norm, e8i, tile_idx


def gen_weights(n_elements, family, seed):
    """Generate weight array with family-specific distribution."""
    rng = random.Random(seed)
    gauss = rng.gauss

    weights = []
    if family == "T04":
        for _ in range(n_elements):
            weights.append(gauss(0.0, 0.8))
    elif family == "T15":
        for _ in range(n_elements):
            weights.append(gauss(0.0, 0.6))
    elif family == "T16":
        # Biased Gaussian + cosine modulation (conditional long-doc transform)
        for k in range(n_elements):
            bias = 0.3 * math.cos(k / 16.0)
            weights.append(gauss(0.1 + bias, 0.5))
    else:
        # Generic fallback
        hval = hash(family) % 1000
        std = 0.5 + hval / 2000.0
        for _ in range(n_elements):
            weights.append(gauss(0.0, std))
    return weights


def write_bqfp(path, family, n_elements=1024, seed=0):
    """Write a valid single-tensor BQFP file for the given family."""
    print(f"  generating {n_elements}-element {family} weight tensor (seed={seed})")
    weights = gen_weights(n_elements, family, seed)

    bits = BITS
    lattice_scale = 8.0 * bits
    n_blocks = (n_elements + BLOCK_DIM - 1) // BLOCK_DIM

    # Haar seed from family name (mirrors bonfyre-quant behavior)
    haar_seed = 0x12345678
    for ch in family:
        haar_seed = (haar_seed * 31 + ord(ch)) & 0xFFFFFFFFFFFFFFFF

    # Build a simple uniform codebook (eff_k = min(16, n_blocks))
    eff_k = min(16, n_blocks)
    codebook = [0.0] * (eff_k * TILE_DIM)
    # Seed codebook with distinct family-specific vectors
    cb_rng = random.Random(seed ^ 0xDEADBEEF)
    for i in range(eff_k * TILE_DIM):
        codebook[i] = cb_rng.gauss(0.0, 0.3)

    # Encode all blocks
    blocks = []
    for b in range(n_blocks):
        off = b * BLOCK_DIM
        chunk = weights[off: off + BLOCK_DIM]
        block_seed = (haar_seed ^ b) & 0xFFFFFFFFFFFFFFFF
        scale, warp_norm, e8i, tile_idx = encode_block(chunk, block_seed, lattice_scale)
        # Assign tile_idx by nearest codebook entry (simplified: sequential round-robin)
        for p in range(E8_PAIRS):
            tile_idx[p] = (b * E8_PAIRS + p) % eff_k
        blocks.append((scale, warp_norm, e8i, tile_idx))

    # Write file
    with open(path, "wb") as f:
        # File header
        f.write(struct.pack("<IIII", BQFP_MAGIC, BQFP_VERSION, 1, bits))

        # Tensor header
        name_bytes = family.encode("utf-8")[:255]
        name_buf = name_bytes + b"\x00" * (256 - len(name_bytes))
        f.write(name_buf)
        f.write(struct.pack("<QQ", n_elements, n_blocks))
        f.write(struct.pack("<II", eff_k, 0))  # eff_k, pad

        # Codebook
        for v in codebook:
            f.write(struct.pack("<f", v))

        # Blocks
        for (scale, warp_norm, e8i, tile_idx) in blocks:
            f.write(struct.pack("<ff", scale, warp_norm))
            f.write(bytes([v & 0xFF for v in e8i]))
            f.write(bytes(tile_idx))

    size = sum(1 for _ in open(path, "rb").read())
    with open(path, "rb") as f:
        size = len(f.read())
    print(f"  wrote {path}  ({size} bytes, {n_blocks} blocks, eff_k={eff_k})")
    return 0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    family = sys.argv[1]
    output = sys.argv[2]

    seed = 42
    n_elements = 1024
    for i, a in enumerate(sys.argv[3:], 3):
        if a == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
        if a == "--n-elements" and i + 1 < len(sys.argv):
            n_elements = int(sys.argv[i + 1])

    return write_bqfp(output, family, n_elements, seed)


if __name__ == "__main__":
    sys.exit(main())
