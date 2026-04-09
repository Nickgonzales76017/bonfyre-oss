#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 BonfyreBrief [BonfyreTag ...]" >&2
  exit 64
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${BONFYRE_PREFIX:-$HOME/.local}"
BINDIR="${PREFIX}/bin"
LIBDIR="${PREFIX}/lib"
INCDIR="${PREFIX}/include"

COMMON_CFLAGS="${BONFYRE_PORTABLE_CFLAGS:--O3 -Wall -Wextra -std=c11 -D_GNU_SOURCE -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -D_XOPEN_SOURCE=700 -include limits.h -include unistd.h -include strings.h -include sys/time.h}"

mkdir -p "$BINDIR" "$LIBDIR" "$INCDIR"

echo "=== Building portable Bonfyre runtime ==="
echo "prefix: $PREFIX"
echo "targets: $*"

make -C "$ROOT/lib/liblambda-tensors" clean
make -C "$ROOT/lib/libbonfyre" clean
make -C "$ROOT/lib/liblambda-tensors" CC="${CC:-cc}" OPTFLAGS="$COMMON_CFLAGS"
make -C "$ROOT/lib/libbonfyre" CC="${CC:-cc}" OPTFLAGS="$COMMON_CFLAGS"

cp "$ROOT/lib/liblambda-tensors/liblambda-tensors.a" "$LIBDIR/" 2>/dev/null || true
cp "$ROOT/lib/liblambda-tensors/liblambda-tensors.so" "$LIBDIR/" 2>/dev/null || true
cp "$ROOT/lib/liblambda-tensors/include/lambda_tensors.h" "$INCDIR/" 2>/dev/null || true
cp "$ROOT/lib/libbonfyre/libbonfyre.a" "$LIBDIR/" 2>/dev/null || true
cp "$ROOT/lib/libbonfyre/include/bonfyre.h" "$INCDIR/" 2>/dev/null || true

for target in "$@"; do
  dir="$ROOT/cmd/$target"
  if [ ! -d "$dir" ]; then
    echo "missing target directory: $dir" >&2
    exit 65
  fi
  echo "--- $target ---"
  make -C "$dir" clean >/dev/null 2>&1 || true
  make -C "$dir" CC="${CC:-cc}" CFLAGS="$COMMON_CFLAGS"
  find "$dir" -maxdepth 1 -name 'bonfyre-*' -type f -perm -111 -exec cp {} "$BINDIR/" \;
done

echo "=== Portable runtime ready at $BINDIR ==="
