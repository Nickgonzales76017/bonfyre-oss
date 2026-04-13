#!/usr/bin/env python3
"""Compare inference: original Qwen 3B vs FPQ v12 decoded."""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

ORIGINAL = "/workspace/qwen25-3b"
DECODED  = "/workspace/qwen3b-decoded"

SEP = "=" * 60

prompts = [
    "The capital of France is",
    "In quantum mechanics, the Heisenberg uncertainty principle states that",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"",
    "The three laws of thermodynamics are:",
    "Once upon a time in a small village,",
]


def run_inference(model_path, label, tokenizer):
    print(f"\n{SEP}")
    print(f"  {label}: {model_path}")
    print(SEP)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")
    print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            t0 = time.time()
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=1.0,
            )
            gen_time = time.time() - t0

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        tokens_gen = out.shape[1] - inputs["input_ids"].shape[1]
        tps = tokens_gen / gen_time if gen_time > 0 else 0
        results.append((prompt, text, tokens_gen, tps))

        print(f"\n  Prompt: {prompt[:60]}...")
        completion = text[len(prompt) : len(prompt) + 200]
        print(f"  Output: {completion}")
        print(f"  [{tokens_gen} tokens, {tps:.1f} tok/s]")

    # Get logits for comparison
    test_text = "The quick brown fox jumps over the lazy dog and runs through the forest"
    inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits

    del model
    torch.cuda.empty_cache()

    return results, logits


# Load tokenizer once
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ORIGINAL)

# Run original
orig_results, orig_logits = run_inference(ORIGINAL, "ORIGINAL (BF16)", tokenizer)

# Run decoded v12
v12_results, v12_logits = run_inference(DECODED, "FPQ v12 DECODED (BF16)", tokenizer)

# Compare logits
print(f"\n{SEP}")
print("  LOGIT COMPARISON")
print(SEP)
cos = torch.nn.functional.cosine_similarity(
    orig_logits.float().flatten(),
    v12_logits.float().flatten(),
    dim=0,
)
mse = ((orig_logits.float() - v12_logits.float()) ** 2).mean()
print(f"  Cosine similarity: {cos.item():.6f}")
print(f"  MSE: {mse.item():.6e}")

# Top-1 token agreement
orig_top1 = orig_logits.argmax(dim=-1)
v12_top1 = v12_logits.argmax(dim=-1)
agreement = (orig_top1 == v12_top1).float().mean()
print(f"  Top-1 token agreement: {agreement.item() * 100:.1f}%")

# Compare generations
print(f"\n{SEP}")
print("  GENERATION COMPARISON")
print(SEP)
match_count = 0
for (p, orig_text, _, _), (_, v12_text, _, _) in zip(orig_results, v12_results):
    match = orig_text == v12_text
    if match:
        match_count += 1
    print(f"\n  Prompt: {p[:50]}...")
    print(f"  Match: {'YES' if match else 'NO'}")
    if not match:
        for i, (a, b) in enumerate(zip(orig_text, v12_text)):
            if a != b:
                print(f"  Diverges at char {i}:")
                print(f"    orig: ...{orig_text[max(0, i - 10) : i + 20]}...")
                print(f"    v12:  ...{v12_text[max(0, i - 10) : i + 20]}...")
                break

print(f"\n  Exact match: {match_count}/{len(prompts)} prompts")

# File sizes
import os

fpq_s1 = os.path.getsize("/workspace/qwen3b-v12/model-00001-of-00002.fpq")
fpq_s2 = os.path.getsize("/workspace/qwen3b-v12/model-00002-of-00002.fpq")
orig_s1 = os.path.getsize(f"{ORIGINAL}/model-00001-of-00002.safetensors")
orig_s2 = os.path.getsize(f"{ORIGINAL}/model-00002-of-00002.safetensors")
total_params = 3_085_898_752  # Qwen 2.5 3B

fpq_total = fpq_s1 + fpq_s2
orig_total = orig_s1 + orig_s2
bparam = (fpq_total * 8) / total_params

print(f"\n{SEP}")
print("  SUMMARY")
print(SEP)
print(f"  Original size:    {orig_total / 1e9:.2f} GB (BF16)")
print(f"  FPQ v12 size:     {fpq_total / 1e9:.2f} GB")
print(f"  Compression:      {orig_total / fpq_total:.2f}x")
print(f"  Bits/param:       {bparam:.3f}")
print(f"  Logit cosine:     {cos.item():.6f}")
print(f"  Logit MSE:        {mse.item():.6e}")
print(f"  Top-1 agreement:  {agreement.item() * 100:.1f}%")
print(f"  Exact gen match:  {match_count}/{len(prompts)}")
print(SEP)
