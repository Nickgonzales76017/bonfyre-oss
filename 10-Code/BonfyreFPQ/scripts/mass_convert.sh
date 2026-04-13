#!/bin/bash
# mass_convert.sh — Convert a list of HF models to v12 .fpq
# Run on any cheap server with: bash mass_convert.sh
# Requirements: 
#   - bonfyre-fpq binary (make in BonfyreFPQ/)
#   - python3, huggingface-cli (pip install huggingface_hub)
#   - Disk: 2× largest model (download + convert, then cleanup)
#   - RAM: ~4× largest tensor (typically <8GB for <14B models)
#   - NO GPU needed — conversion is pure CPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERT="${SCRIPT_DIR}/batch_convert_hf.sh"

# Target models — sorted by size, smallest first for fast iteration
MODELS=(
    # Tier 1: Small models (<2B) — minutes each
    "google/gemma-2-2b"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "HuggingFaceTB/SmolLM2-1.7B"
    "microsoft/phi-1_5"
    "stabilityai/stablelm-2-1_6b"
    
    # Tier 2: 3B models — ~10 min each  
    "Qwen/Qwen2.5-3B"
    "Qwen/Qwen2.5-3B-Instruct"
    "microsoft/phi-3.5-mini-instruct"
    
    # Tier 3: 7-8B models — ~30 min each, need ~16GB RAM
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "mistralai/Mistral-7B-v0.3"
    "Qwen/Qwen2.5-7B"
    "google/gemma-2-9b"
    
    # Tier 4: Specialty models
    "openai/whisper-large-v3-turbo"
    "openai/whisper-large-v3"
    "Wan-AI/Wan2.1-T2V-1.3B"
    "facebook/musicgen-small"
    "parler-tts/parler-tts-mini-v1"
    
    # Tier 5: 14B+ models — need ~32GB RAM, hours each
    "Qwen/Qwen2.5-14B"
    "microsoft/phi-4"
    "meta-llama/Llama-3.1-70B"  # needs ~140GB disk, ~64GB RAM
)

echo "=== BonfyreFPQ Mass Conversion ==="
echo "  Models: ${#MODELS[@]}"
echo "  Script: $CONVERT"
echo ""

DONE=0
FAIL=0
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  [$((DONE+FAIL+1))/${#MODELS[@]}] $MODEL"
    echo "================================================================"
    
    if bash "$CONVERT" "$MODEL" --bits 3; then
        DONE=$((DONE+1))
        echo "  ✓ $MODEL"
    else
        FAIL=$((FAIL+1))
        echo "  ✗ $MODEL (FAILED)"
    fi
done

echo ""
echo "=== RESULTS ==="
echo "  Success: $DONE"
echo "  Failed:  $FAIL"
echo "  Total:   ${#MODELS[@]}"
