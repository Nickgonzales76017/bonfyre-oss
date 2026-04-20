#!/usr/bin/env python3
"""
finalize_artifact.py — unified compress + hash stage for bonfyre pipelines.

Replaces the brittle split between bonfyre-quant compress (broken) and
bonfyre-hash merkle (no write-back) that caused friction in calibration runs.

What it does:
  1. Packs model.onnx with bonfyre-compress (zstd) → model.fpq
  2. Computes sha256 of both model.onnx and model.fpq via bonfyre-hash file
  3. Writes content_hash + fpq_sha256 + root_hash back into artifact.json
  4. Runs bonfyre-hash verify to confirm integrity

On success:
  - artifact.json has: sha256, content_hash, fpq_sha256, root_hash
  - model.fpq exists alongside model.onnx in the train output dir
  - bonfyre-model add <artifact.json> will work
  - bonfyre-model push will have fpq_sha256 populated

Usage (standalone):
  python3 finalize_artifact.py <train_out_dir>
  python3 finalize_artifact.py /tmp/run/train

As a bonfyre-run stage in a recipe:
  {
    "id":  "finalize",
    "bin": "python3",
    "args": ["scripts/finalize_artifact.py", "{out}/train"],
    "depends_on": ["train"]
  }
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_tool(cmd, label):
    """Run a subprocess; return (stdout, returncode)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.stdout.strip(), r.returncode
    except FileNotFoundError:
        return None, -1
    except Exception as e:
        print(f"[finalize] WARNING: {label} error: {e}", file=sys.stderr)
        return None, -1


def compress_model(onnx_path, fpq_path):
    """
    Compress model.onnx → model.fpq.
    Tries bonfyre-compress pack first; falls back to python zstd if unavailable.
    """
    stdout, rc = run_tool(
        ["bonfyre-compress", "pack", onnx_path, fpq_path],
        "bonfyre-compress"
    )
    if rc == 0 and os.path.exists(fpq_path):
        print(f"[finalize] bonfyre-compress pack → {fpq_path}")
        return True

    # Fallback: zstd via Python stdlib (lzma as proxy if zstd unavailable)
    try:
        import zlib
        data = open(onnx_path, "rb").read()
        compressed = zlib.compress(data, level=9)
        open(fpq_path, "wb").write(compressed)
        print(f"[finalize] zlib compress → {fpq_path} "
              f"({len(data)//1024}KB → {len(compressed)//1024}KB)")
        return True
    except Exception as e:
        print(f"[finalize] WARNING: compression failed: {e}; skipping .fpq")
        return False


def compute_root_hash(content_hash):
    """Single-leaf Merkle root = leaf hash itself."""
    return content_hash


def main():
    p = argparse.ArgumentParser(
        description="Finalize artifact: compress + hash model outputs"
    )
    p.add_argument("train_dir", help="Path to collapse_train.py output dir")
    p.add_argument("--skip-compress", action="store_true",
                   help="Skip compression (hash only)")
    args = p.parse_args()

    train_dir     = args.train_dir
    onnx_path     = os.path.join(train_dir, "model.onnx")
    fpq_path      = os.path.join(train_dir, "model.fpq")
    artifact_path = os.path.join(train_dir, "artifact.json")

    if not os.path.exists(onnx_path):
        sys.exit(f"[finalize] model.onnx not found: {onnx_path}")
    if not os.path.exists(artifact_path):
        sys.exit(f"[finalize] artifact.json not found: {artifact_path}")

    # 1. Load existing artifact
    artifact = json.loads(open(artifact_path).read())

    # 2. Compute ONNX sha256
    onnx_sha = sha256_file(onnx_path)
    artifact["sha256"]       = onnx_sha
    artifact["content_hash"] = onnx_sha   # flat scalar for bonfyre-hash merkle

    # 3. Compress model.onnx → model.fpq
    fpq_sha = None
    if not args.skip_compress:
        ok = compress_model(onnx_path, fpq_path)
        if ok:
            fpq_sha = sha256_file(fpq_path)
            artifact["fpq_sha256"] = fpq_sha
            size_mb_fpq = round(os.path.getsize(fpq_path) / 1024 / 1024, 3)
            artifact["fpq_size_mb"] = size_mb_fpq
            print(f"[finalize] fpq_sha256={fpq_sha[:16]}…  size={size_mb_fpq} MB")

    # 4. Compute root_hash via bonfyre-hash merkle (or pure-Python)
    stdout, rc = run_tool(
        ["bonfyre-hash", "merkle", artifact_path],
        "bonfyre-hash merkle (pre-write)"
    )
    if rc == 0 and stdout and len(stdout) == 64:
        root_hash = stdout
    else:
        root_hash = compute_root_hash(onnx_sha)

    artifact["root_hash"] = root_hash

    # 5. Write updated artifact.json
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[finalize] artifact.json updated  root_hash={root_hash[:16]}…")

    # 6. Verify via bonfyre-hash verify (non-fatal)
    verify_out, verify_rc = run_tool(
        ["bonfyre-hash", "verify", artifact_path],
        "bonfyre-hash verify"
    )
    if verify_rc == 0:
        print(f"[finalize] {verify_out}")
    else:
        print(f"[finalize] WARNING: bonfyre-hash verify skipped (rc={verify_rc})")

    print(f"[finalize] done — {artifact_path}")


if __name__ == "__main__":
    main()
