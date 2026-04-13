#!/usr/bin/env python3
"""
test_e2e_whisper.py — End-to-end Whisper transcription: original vs FPQ SLI.

Transcribes audio using original Whisper weights and FPQ SLI weights,
then compares the output text and intermediate logits.

Usage:
  python3 scripts/test_e2e_whisper.py \
    --fpq models/whisper-v3-turbo-v12/model.fpq \
    --hf openai/whisper-large-v3-turbo
"""
import argparse
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from fpq_bridge import FPQModel, FPQLinear


def manual_patch(model, fpq, verbose=True):
    """Patch nn.Linear layers with FPQ SLI."""
    fpq_names = set(fpq.tensor_names())
    n_patched = 0
    n_skipped = 0

    def _recurse(module, prefix):
        nonlocal n_patched, n_skipped
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear):
                tensor_name = f"{full}.weight"
                if tensor_name in fpq_names:
                    ti = fpq.tensor_info(tensor_name)
                    if ti and ti["rows"] == child.out_features and ti["cols"] == child.in_features:
                        bias_data = child.bias.data.clone().float() if child.bias is not None else None
                        fpq_lin = FPQLinear(fpq, tensor_name, bias_data)
                        setattr(module, name, fpq_lin)
                        n_patched += 1
                        tag = "SLI" if ti["has_sli"] else "DENSE"
                        if verbose:
                            print(f"  [{tag}] {full} ({ti['rows']}×{ti['cols']})")
                    else:
                        n_skipped += 1
                else:
                    n_skipped += 1
            else:
                _recurse(child, full)

    _recurse(model, "")

    # Load passthrough tensors
    state = model.state_dict()
    n_pass = 0
    for tname in fpq_names:
        ti = fpq.tensor_info(tname)
        if ti and not ti["has_sli"]:
            if tname in state:
                pt_data = fpq.get_passthrough(tname)
                if pt_data is not None:
                    t = torch.from_numpy(pt_data)
                    if t.shape == state[tname].shape:
                        state[tname] = t.to(state[tname].dtype)
                        n_pass += 1
                    elif t.numel() == state[tname].numel():
                        state[tname] = t.reshape(state[tname].shape).to(state[tname].dtype)
                        n_pass += 1

    if n_pass > 0:
        model.load_state_dict(state, strict=False)

    if verbose:
        print(f"\nPatched: {n_patched} layers, {n_pass} passthrough, {n_skipped} skipped")
    return n_patched, n_skipped, n_pass


def generate_test_audio(processor, duration_sec=3.0, sr=16000):
    """Generate synthetic audio (sine wave + noise) for testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    # Mix of frequencies to create speech-like audio
    audio = (0.3 * np.sin(2 * np.pi * 200 * t) +
             0.2 * np.sin(2 * np.pi * 400 * t) +
             0.1 * np.sin(2 * np.pi * 800 * t) +
             0.05 * np.random.randn(len(t)).astype(np.float32))
    return audio


def transcribe(model, processor, audio, device="cpu"):
    """Run Whisper transcription, return text + encoder logits."""
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device).float()

    with torch.no_grad():
        t0 = time.time()
        generated = model.generate(
            input_features,
            max_new_tokens=128,
            language="en",
            task="transcribe",
        )
        gen_time = time.time() - t0

    text = processor.batch_decode(generated, skip_special_tokens=True)[0]

    # Get encoder output for comparison
    with torch.no_grad():
        enc_out = model.model.encoder(input_features).last_hidden_state

    return text, enc_out.cpu().float(), gen_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpq", required=True, help="Path to .fpq file")
    parser.add_argument("--hf", required=True, help="HuggingFace model ID")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  BonfyreFPQ Whisper End-to-End Test")
    print("=" * 70)

    # Load processor
    processor = AutoProcessor.from_pretrained(args.hf)

    # Generate test audio
    audio = generate_test_audio(processor)
    print(f"Test audio: {len(audio)/16000:.1f}s synthetic")

    # ── Original ─────────────────────────────────────────────────
    print(f"\n[Phase 1] Loading original: {args.hf}")
    t0 = time.time()
    model_orig = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.hf, torch_dtype=torch.float32,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    text_orig, enc_orig, t_orig = transcribe(model_orig, processor, audio, args.device)
    print(f"  Text: '{text_orig}'")
    print(f"  Time: {t_orig:.1f}s")
    print(f"  Encoder output shape: {enc_orig.shape}")

    del model_orig
    import gc; gc.collect()

    # ── FPQ ──────────────────────────────────────────────────────
    print(f"\n[Phase 2] Loading FPQ model...")
    model_fpq = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.hf, torch_dtype=torch.float32,
    )

    fpq = FPQModel(args.fpq)
    print(f"  FPQ info: {fpq.info()}")
    manual_patch(model_fpq, fpq, verbose=True)

    text_fpq, enc_fpq, t_fpq = transcribe(model_fpq, processor, audio, args.device)
    print(f"  Text: '{text_fpq}'")
    print(f"  Time: {t_fpq:.1f}s")

    # ── Compare ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")

    enc_cos = torch.nn.functional.cosine_similarity(
        enc_orig.flatten().unsqueeze(0),
        enc_fpq.flatten().unsqueeze(0),
    ).item()
    enc_mse = ((enc_orig - enc_fpq) ** 2).mean().item()

    text_match = text_orig.strip() == text_fpq.strip()

    print(f"  Encoder cosine:  {enc_cos:.6f}")
    print(f"  Encoder MSE:     {enc_mse:.6e}")
    print(f"  Text match:      {'YES' if text_match else 'NO'}")
    print(f"  Original text:   '{text_orig}'")
    print(f"  FPQ text:        '{text_fpq}'")

    if args.out:
        results = {
            "model": args.hf,
            "fpq": args.fpq,
            "encoder_cosine": enc_cos,
            "encoder_mse": enc_mse,
            "text_match": text_match,
            "text_original": text_orig,
            "text_fpq": text_fpq,
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {args.out}")


if __name__ == "__main__":
    main()
