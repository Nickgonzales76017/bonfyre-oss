#!/usr/bin/env python3
"""
test_e2e_generation.py — End-to-end generation: original HF model vs FPQ SLI.

Loads the HF model normally, generates text, then patches with FPQ SLI
and generates with the same prompts/seeds. Compares:
  - Token-level agreement (top-1 match %)
  - Logit cosine similarity
  - Generated text quality

Usage:
  python3 scripts/test_e2e_generation.py \
    --fpq models/tinyllama-v12/model.fpq \
    --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
import argparse
import sys
import os
import time
import json

# Add parent dir so fpq_bridge can find libfpq
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fpq_bridge import FPQModel, FPQLinear

# ─── Prompts ─────────────────────────────────────────────────────────────────

PROMPTS = [
    "The capital of France is",
    "In quantum mechanics, the Heisenberg uncertainty principle states that",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"",
    "The three laws of thermodynamics are:",
    "Once upon a time in a small village,",
]


# ─── Manual patch (more control than patch_model) ───────────────────────────

def manual_patch(model, fpq, verbose=True):
    """
    Patch nn.Linear layers in a HuggingFace model with FPQ SLI.
    Also loads passthrough tensors (norms, embeddings).
    Returns (n_patched, n_skipped, n_passthrough).
    """
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
                        if verbose:
                            print(f"  [SKIP] {full} — shape mismatch or not found in .fpq")
                else:
                    n_skipped += 1
            else:
                _recurse(child, full)

    _recurse(model, "")

    # Load passthrough tensors into model state dict
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
        print(f"\nPatched: {n_patched} SLI, {n_pass} passthrough, {n_skipped} skipped")

    return n_patched, n_skipped, n_pass


# ─── Generate ───────────────────────────────────────────────────────────────

def generate_and_collect(model, tokenizer, prompts, max_new_tokens=50, device="cpu"):
    """Generate text and collect logits for each prompt."""
    results = []
    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            t0 = time.time()
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
            gen_time = time.time() - t0

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        n_gen = out.shape[1] - inputs["input_ids"].shape[1]
        tps = n_gen / gen_time if gen_time > 0 else 0

        # Get logits for the input prefix (for cosine comparison)
        with torch.no_grad():
            logits = model(**inputs).logits

        results.append({
            "prompt": prompt,
            "text": text,
            "completion": text[len(prompt):len(prompt) + 300],
            "n_tokens": n_gen,
            "tok_per_sec": tps,
            "gen_time": gen_time,
            "logits": logits.cpu().float(),
        })
    return results


def compare_results(orig_results, fpq_results):
    """Compare original vs FPQ generation results."""
    metrics = []
    for orig, fpq_r in zip(orig_results, fpq_results):
        # Logit cosine similarity
        o_flat = orig["logits"].flatten()
        f_flat = fpq_r["logits"].flatten()
        cos = torch.nn.functional.cosine_similarity(
            o_flat.unsqueeze(0), f_flat.unsqueeze(0)
        ).item()

        # MSE
        mse = ((o_flat - f_flat) ** 2).mean().item()

        # Top-1 token agreement
        o_top1 = orig["logits"].argmax(dim=-1)
        f_top1 = fpq_r["logits"].argmax(dim=-1)
        agree = (o_top1 == f_top1).float().mean().item()

        # Text match
        text_match = orig["completion"].strip() == fpq_r["completion"].strip()

        metrics.append({
            "prompt": orig["prompt"][:60],
            "cosine": cos,
            "mse": mse,
            "top1_agreement": agree,
            "text_match": text_match,
            "orig_completion": orig["completion"][:100],
            "fpq_completion": fpq_r["completion"][:100],
        })
    return metrics


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E2E generation: original vs FPQ SLI")
    parser.add_argument("--fpq", required=True, nargs="+",
                        help="Path(s) to .fpq model shard files")
    parser.add_argument("--hf", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--device", default="cpu",
                        help="Device for inference (cpu/cuda/mps)")
    parser.add_argument("--out", default=None, help="Save results JSON to this path")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Custom prompts (overrides defaults)")
    args = parser.parse_args()

    prompts = args.prompts or PROMPTS
    device = args.device

    print("=" * 70)
    print("  BonfyreFPQ End-to-End Generation Test")
    print("=" * 70)

    # ── Phase 1: Original model ──────────────────────────────────────────
    print(f"\n[Phase 1] Loading original model: {args.hf}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.hf)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_orig = AutoModelForCausalLM.from_pretrained(
        args.hf,
        torch_dtype=torch.float32,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model_orig = model_orig.float()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Params: {sum(p.numel() for p in model_orig.parameters())/1e6:.0f}M")

    print(f"\n[Phase 1] Generating with original weights...")
    orig_results = generate_and_collect(model_orig, tokenizer, prompts,
                                        max_new_tokens=args.max_tokens, device=device)
    for r in orig_results:
        print(f"\n  Prompt: {r['prompt'][:60]}...")
        print(f"  Output: {r['completion'][:100]}")
        print(f"  [{r['n_tokens']} tokens, {r['tok_per_sec']:.1f} tok/s]")

    # Free original model
    del model_orig
    import gc; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    # ── Phase 2: FPQ SLI model ───────────────────────────────────────────
    print(f"\n[Phase 2] Loading FPQ model + patching...")
    t0 = time.time()

    # Reload model skeleton with random weights
    model_fpq = AutoModelForCausalLM.from_pretrained(
        args.hf,
        torch_dtype=torch.float32,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model_fpq = model_fpq.float()

    # Load FPQ shards and patch
    for fpq_path in args.fpq:
        print(f"\n  Opening {fpq_path}...")
        fpq = FPQModel(fpq_path)
        print(f"  Info: {fpq.info()}")
        manual_patch(model_fpq, fpq, verbose=True)

    print(f"  Total patch time: {time.time()-t0:.1f}s")

    print(f"\n[Phase 2] Generating with FPQ SLI weights...")
    fpq_results = generate_and_collect(model_fpq, tokenizer, prompts,
                                        max_new_tokens=args.max_tokens, device=device)
    for r in fpq_results:
        print(f"\n  Prompt: {r['prompt'][:60]}...")
        print(f"  Output: {r['completion'][:100]}")
        print(f"  [{r['n_tokens']} tokens, {r['tok_per_sec']:.1f} tok/s]")

    # ── Phase 3: Compare ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")

    metrics = compare_results(orig_results, fpq_results)
    for m in metrics:
        status = "MATCH" if m["text_match"] else "DIFF"
        print(f"\n  [{status}] {m['prompt']}")
        print(f"    Logit cosine:     {m['cosine']:.6f}")
        print(f"    Logit MSE:        {m['mse']:.6e}")
        print(f"    Top-1 agreement:  {m['top1_agreement']*100:.1f}%")
        if not m["text_match"]:
            print(f"    ORIG: {m['orig_completion'][:80]}")
            print(f"    FPQ:  {m['fpq_completion'][:80]}")

    # Summary
    avg_cos = np.mean([m["cosine"] for m in metrics])
    avg_agree = np.mean([m["top1_agreement"] for m in metrics])
    n_match = sum(1 for m in metrics if m["text_match"])

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Mean logit cosine:     {avg_cos:.6f}")
    print(f"  Mean top-1 agreement:  {avg_agree*100:.1f}%")
    print(f"  Text exact matches:    {n_match}/{len(metrics)}")
    print(f"  Model: {args.hf}")
    print(f"  FPQ:   {', '.join(args.fpq)}")

    # Save results
    if args.out:
        save_data = {
            "model": args.hf,
            "fpq_files": args.fpq,
            "summary": {
                "mean_cosine": avg_cos,
                "mean_top1_agreement": avg_agree,
                "text_matches": n_match,
                "total_prompts": len(metrics),
            },
            "per_prompt": [{k: v for k, v in m.items()} for m in metrics],
        }
        with open(args.out, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Results saved to {args.out}")

    return avg_cos, avg_agree, n_match


if __name__ == "__main__":
    main()
