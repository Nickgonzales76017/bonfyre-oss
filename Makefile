# Bonfyre — top-level Makefile
# Builds all Bonfyre binaries + liblambda-tensors + libbonfyre runtime

PREFIX ?= $(HOME)/.local
BINDIR  = $(PREFIX)/bin
LIBDIR  = $(PREFIX)/lib
INCDIR  = $(PREFIX)/include

CC     ?= cc
CFLAGS ?= -O3 -march=native -flto=auto -ffunction-sections -fdata-sections -Wall -Wextra -std=c11

# Every directory under cmd/ with a Makefile
BINARIES := $(sort $(dir $(wildcard cmd/*/Makefile)))

.PHONY: all lib binaries clean install test help sanitize fuzz

all: lib binaries

# ── Libraries ────────────────────────────────────────────────
lib:
	@echo "=== Building liblambda-tensors ==="
	$(MAKE) -C lib/liblambda-tensors CC="$(CC)" OPTFLAGS="$(CFLAGS)"
	@echo "=== Building libbonfyre ==="
	$(MAKE) -C lib/libbonfyre CC="$(CC)" OPTFLAGS="$(CFLAGS)"

# ── Binaries ─────────────────────────────────────────────────
binaries: lib
	@total=0; ok=0; fail=0; \
	for dir in $(BINARIES); do \
		name=$$(basename $$dir); \
		printf "  [%2d] %-28s" $$((total+1)) "$$name"; \
		logfile=$$(mktemp); \
		if $(MAKE) -C $$dir CC="$(CC)" CFLAGS="$(CFLAGS)" > "$$logfile" 2>&1; then \
			echo "✓"; \
			ok=$$((ok+1)); \
		else \
			echo "✗"; \
			sed 's/^/      /' "$$logfile"; \
			fail=$$((fail+1)); \
		fi; \
		rm -f "$$logfile"; \
		total=$$((total+1)); \
	done; \
	echo ""; \
	echo "=== $$ok/$$total built ($$fail failed) ==="

# ── Install ──────────────────────────────────────────────────
install: all
	@mkdir -p $(BINDIR) $(LIBDIR) $(INCDIR)
	@echo "Installing to $(PREFIX)"
	@for dir in $(BINARIES); do \
		name=$$(basename $$dir); \
		find "$$dir" -maxdepth 1 -name 'bonfyre-*' -type f -perm +111 -exec cp {} $(BINDIR)/ \; 2>/dev/null; \
	done
	@cp lib/liblambda-tensors/liblambda-tensors.a $(LIBDIR)/ 2>/dev/null || true
	@cp lib/liblambda-tensors/liblambda-tensors.so $(LIBDIR)/ 2>/dev/null || true
	@cp lib/liblambda-tensors/include/lambda_tensors.h $(INCDIR)/ 2>/dev/null || true
	@cp lib/libbonfyre/libbonfyre.a $(LIBDIR)/ 2>/dev/null || true
	@cp lib/libbonfyre/include/bonfyre.h $(INCDIR)/ 2>/dev/null || true
	@echo "Done. Ensure $(BINDIR) is in your PATH."

# ── Clean ────────────────────────────────────────────────────
clean:
	$(MAKE) -C lib/liblambda-tensors clean
	$(MAKE) -C lib/libbonfyre clean
	@for dir in $(BINARIES); do \
		$(MAKE) -C $$dir clean 2>/dev/null || true; \
	done
	@echo "Clean."

# ── Test ─────────────────────────────────────────────────────
test: all
	@echo "=== Running tests ==="
	$(MAKE) -C lib/liblambda-tensors test || true
	$(MAKE) -C lib/libbonfyre test || true
	@pass=0; \
	for dir in $(BINARIES); do \
		for bin in "$$dir"/bonfyre-*; do \
			[ -x "$$bin" ] || continue; \
			if "$$bin" status > /dev/null 2>&1; then \
				echo "  ✓ $$(basename $$bin) status"; \
				pass=$$((pass+1)); \
			fi; \
		done; \
	done; \
	echo "=== $$pass binaries passed status check ==="

# ── Security hardening ───────────────────────────────────────
# Address Sanitizer: catches buffer overflows, use-after-free, leaks
sanitize:
	@echo "=== Building with AddressSanitizer + UndefinedBehaviorSanitizer ==="
	$(MAKE) -C lib/liblambda-tensors clean
	$(MAKE) -C lib/liblambda-tensors CC="$(CC)" OPTFLAGS="-g -fsanitize=address,undefined -fno-omit-frame-pointer -std=c11"
	$(MAKE) -C lib/libbonfyre clean
	$(MAKE) -C lib/libbonfyre CC="$(CC)" OPTFLAGS="-g -fsanitize=address,undefined -fno-omit-frame-pointer -std=c11"
	@for dir in $(BINARIES); do \
		$(MAKE) -C $$dir CC="$(CC)" CFLAGS="-g -fsanitize=address,undefined -fno-omit-frame-pointer -std=c11" \
			LDFLAGS="-fsanitize=address,undefined" 2>/dev/null || true; \
	done
	@echo "=== Sanitizer build done. Run binaries to detect memory errors. ==="

# ── Profile-Guided Optimization ──────────────────────────────
# Step 1: `make pgo-gen` → builds with profiling instrumentation
# Step 2: Run representative workloads on the instrumented binaries
# Step 3: `make pgo-use` → rebuilds using collected profile data
PGO_DIR = $(CURDIR)/pgo-data

pgo-gen: clean
	@echo "=== PGO: instrumented build ==="
	$(MAKE) all CFLAGS="$(CFLAGS) -fprofile-generate=$(PGO_DIR)"

pgo-use:
	@echo "=== PGO: optimized build from profile data ==="
	$(MAKE) clean
	$(MAKE) all CFLAGS="$(CFLAGS) -fprofile-use=$(PGO_DIR) -fprofile-correction"

pgo-clean:
	rm -rf $(PGO_DIR)

# ── Static binary build (musl or -static) ───────────────────
# Produces fully statically linked binaries (~400 KB each, zero runtime deps).
# Requires musl-gcc (install: apt install musl-tools) or will fall back to
# system cc with -static (works on Linux; partially on macOS with static libc).
#
# Usage:
#   make static                        # use musl-gcc if available
#   make static STATIC_CC=musl-gcc     # explicit musl path
#   make static STATIC_CC="cc -static" # force -static with system libc
#
STATIC_CC    ?= $(shell command -v musl-gcc 2>/dev/null || echo "$(CC)")
STATIC_FLAGS ?= -O2 -std=c11 -static -static-libgcc -lpthread

.PHONY: static
static:
	@echo "=== Static build (CC=$(STATIC_CC)) ==="
	$(MAKE) -C lib/liblambda-tensors clean
	$(MAKE) -C lib/liblambda-tensors CC="$(STATIC_CC)" OPTFLAGS="$(STATIC_FLAGS)"
	$(MAKE) -C lib/libbonfyre clean
	$(MAKE) -C lib/libbonfyre CC="$(STATIC_CC)" OPTFLAGS="$(STATIC_FLAGS)"
	@for dir in $(BINARIES); do \
		$(MAKE) -C $$dir \
			CC="$(STATIC_CC)" \
			CFLAGS="$(STATIC_FLAGS)" \
			LDFLAGS="-static" \
			2>/dev/null && echo "  [static] $$dir" || echo "  [skip]   $$dir"; \
	done
	@echo "=== Static build done ==="
	@echo "Strip with: find . -name 'bonfyre-*' -not -name '*.c' -exec strip {} +"

# ── WASM build via Emscripten ───────────────────────────────
# Requires Emscripten SDK: https://emscripten.org/docs/getting_started/
# Source: source /path/to/emsdk/emsdk_env.sh
#
# Usage:
#   make wasm                           # build bonfyre-runtime.{wasm,js}
#   make wasm WASM_OUT=site/assets/     # emit into site/assets/
#
WASM_CC  ?= emcc
WASM_OUT ?= site/assets
WASM_FLAGS = -O2 -std=c11 \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_bonfyre_wasm_run","_bonfyre_wasm_init","_bonfyre_wasm_version","_bonfyre_wasm_capabilities","_bonfyre_wasm_alloc","_bonfyre_wasm_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToUTF8","lengthBytesUTF8"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=BonfyreModule \
  -s NO_EXIT_RUNTIME=1 \
  -s ENVIRONMENT=web,worker \
  -DBF_WASM_BUILD=1 \
  -I lib/libbonfyre/include

WASM_SRCS = \
  lib/libbonfyre/src/bf_wasm_shim.c \
  lib/libbonfyre/src/bf_artifact.c \
  lib/libbonfyre/src/bf_common.c \
  lib/libbonfyre/src/bf_sha256.c \
  lib/libbonfyre/src/bf_operators.c

.PHONY: wasm wasm-check wasm-all
wasm-check:
	@command -v $(WASM_CC) >/dev/null 2>&1 || \
		(echo "ERROR: Emscripten not found.  Install: https://emscripten.org/docs/getting_started/" && exit 1)

wasm: wasm-check
	@echo "=== WASM build (emcc) ==="
	@mkdir -p $(WASM_OUT)
	$(WASM_CC) $(WASM_FLAGS) \
		$(WASM_SRCS) \
		-o $(WASM_OUT)/bonfyre-runtime.js
	@echo "  WASM output: $(WASM_OUT)/bonfyre-runtime.{wasm,js}"
	@echo "=== WASM build done ==="

wasm-all: wasm
	@echo "=== Generating browser wrappers for all Bonfyre binaries ==="
	@for dir in $(BINARIES); do \
		name=$$(basename $$dir | tr '[:upper:]' '[:lower:]' | sed 's/^bonfyre//'); \
		bin="bonfyre-$$name"; \
		out="$(WASM_OUT)/$$bin.js"; \
		printf "%s\n" "import BonfyreModuleFactory from './bonfyre-runtime.js';" > $$out; \
		printf "%s\n" "" >> $$out; \
		printf "%s\n" "export default async function runBonfyreBinary(recipeYaml, inputBase64, mime='application/octet-stream') {" >> $$out; \
		printf "%s\n" "  const Module = await BonfyreModuleFactory();" >> $$out; \
		printf "%s\n" "  const run = Module.cwrap('bonfyre_wasm_run', 'string', ['string','string','string']);" >> $$out; \
		printf "%s\n" "  const result = run(recipeYaml, inputBase64, mime);" >> $$out; \
		printf "%s\n" "  return JSON.parse(result);" >> $$out; \
		printf "%s\n" "}" >> $$out; \
		echo "  [wasm] $$out"; \
	done
	@echo "=== WASM wrappers ready in $(WASM_OUT) ==="

# ── Docker ────────────────────────────────────────────────────
.PHONY: docker docker-up docker-down
docker:
	docker build -t bonfyre .

docker-up: docker
	docker compose up -d

docker-down:
	docker compose down

# ── Help ─────────────────────────────────────────────────────
help:
	@echo "Bonfyre — Bonfyre binary fleet + 2 libraries, ~2.1 MB total"
	@echo ""
	@echo "  make           Build everything"
	@echo "  make lib       Build liblambda-tensors + libbonfyre"
	@echo "  make install   Install to PREFIX (default: ~/.local)"
	@echo "  make models    Download required ML models"
	@echo "  make clean     Remove all build artifacts"
	@echo "  make test      Run all test suites"
	@echo "  make sanitize  Rebuild with ASan + UBSan for testing"
	@echo "  make pgo-gen   Build with profiling instrumentation"
	@echo "  make pgo-use   Rebuild using collected profile data"
	@echo "  make pgo-clean Remove collected profile data"
	@echo "  make docker    Build Docker image"
	@echo "  make docker-up Start API + worker via compose"
	@echo "  make docker-down Stop compose stack"
	@echo "  make help      This message"

# ── Models ───────────────────────────────────────────────────
WHISPER_DIR  = $(HOME)/.local/share/whisper
MODEL_DIR    = $(HOME)/.bonfyre/models

.PHONY: models
models:
	@echo "=== Downloading models ==="
	@mkdir -p $(WHISPER_DIR) $(MODEL_DIR)
	@if [ ! -f $(WHISPER_DIR)/ggml-base.en.bin ]; then \
		echo "  ↓ whisper base.en (~140MB)..."; \
		curl -fSL -o $(WHISPER_DIR)/ggml-base.en.bin \
			"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"; \
		echo "  ✓ ggml-base.en.bin"; \
	else echo "  ✓ ggml-base.en.bin (exists)"; fi
	@if [ ! -f $(MODEL_DIR)/lid.176.bin ]; then \
		echo "  ↓ fastText lid.176 (~125MB)..."; \
		curl -fSL -o $(MODEL_DIR)/lid.176.bin \
			"https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"; \
		echo "  ✓ lid.176.bin"; \
	else echo "  ✓ lid.176.bin (exists)"; fi
	@if [ ! -f $(MODEL_DIR)/all-MiniLM-L6-v2.onnx ]; then \
		echo "  ↓ sentence-transformer ONNX (~22MB)..."; \
		curl -fSL -o $(MODEL_DIR)/all-MiniLM-L6-v2.onnx \
			"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"; \
		echo "  ✓ all-MiniLM-L6-v2.onnx"; \
	else echo "  ✓ all-MiniLM-L6-v2.onnx (exists)"; fi
	@echo "=== Models ready ==="
