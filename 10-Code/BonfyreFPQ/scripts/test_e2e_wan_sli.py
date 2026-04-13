#!/usr/bin/env python3
"""
test_e2e_wan_sli.py — End-to-end SLI benchmark: Wan2.1-T2V-1.3B

Loads the original Wan DiT from safetensors, deep-copies it, patches the
copy with FPQ SLI (weights never leave compressed form), then runs both
through identical forward passes and compares outputs.

Usage:
  python3 scripts/test_e2e_wan_sli.py \
    --safetensors /workspace/models/wan/diffusion_pytorch_model.safetensors \
    --fpq /workspace/models/wan/wan2.1-t2v-1.3b.fpq \
    --out /workspace/results/wan_sli/out.json

Setup (RunPod):
  pip install torch diffusers transformers safetensors numpy huggingface_hub
  # Download models:
  huggingface-cli download NICKO/wan2.1-t2v-1.3b-v12-fpq --local-dir /workspace/models/wan
  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B diffusion_pytorch_model.safetensors \
      --local-dir /workspace/models/wan-orig
"""
import argparse
import copy
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Bridge lives beside this script
sys.path.insert(0, os.path.dirname(__file__))
from fpq_bridge import FPQModel, FPQLinear, patch_model

# ── Wan model config ───────────────────────────────────────────────────

WAN_1_3B_CONFIG = {
    "num_attention_heads": 12,
    "attention_head_dim": 128,   # 12 * 128 = 1536
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "ffn_dim": 8960,
    "num_layers": 30,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-06,
}

# ── Key remapping helpers ──────────────────────────────────────────────
#
# The .fpq stores weights under their ORIGINAL Wan safetensors names, e.g.:
#   blocks.0.self_attn.q.weight
#
# diffusers WanTransformer3DModel named_modules() yields paths like:
#   blocks.0.attn1.to_q
#
# remap_wan_to_diffusers() converts original Wan keys → diffusers keys.
# We invert that mapping to build the resolver used by patch_model().

def remap_wan_to_diffusers(k):
    """Original Wan safetensors key → diffusers WanTransformer3DModel key."""
    k = k.replace(".self_attn.", ".attn1.")
    k = k.replace(".cross_attn.", ".attn2.")
    k = re.sub(r"\.attn([12])\.q\.", r".attn\1.to_q.", k)
    k = re.sub(r"\.attn([12])\.k\.", r".attn\1.to_k.", k)
    k = re.sub(r"\.attn([12])\.v\.", r".attn\1.to_v.", k)
    k = re.sub(r"\.attn([12])\.o\.", r".attn\1.to_out.0.", k)
    k = re.sub(r"\.ffn\.0\.", ".ffn.net.0.proj.", k)
    k = re.sub(r"\.ffn\.2\.", ".ffn.net.2.", k)
    k = k.replace(".norm3.", ".norm2.")
    k = re.sub(r"(blocks\.\d+)\.modulation$", r"\1.scale_shift_table", k)
    k = k.replace("head.head.", "proj_out.")
    k = k.replace("text_embedding.0.", "condition_embedder.text_embedder.linear_1.")
    k = k.replace("text_embedding.2.", "condition_embedder.text_embedder.linear_2.")
    k = k.replace("time_embedding.0.", "condition_embedder.time_embedder.linear_1.")
    k = k.replace("time_embedding.2.", "condition_embedder.time_embedder.linear_2.")
    k = k.replace("time_projection.1.", "condition_embedder.time_proj.")
    return k


def build_wan_name_resolver(fpq_model):
    """
    Build a dict: diffusers_module_path → fpq_tensor_name.

    For each tensor in the .fpq, convert its name to the diffusers equivalent
    using remap_wan_to_diffusers. Strip ".weight" suffix to get the module path.
    Returns a callable suitable for patch_model(name_resolver=...).
    """
    resolver_map = {}
    for fpq_name in fpq_model.tensor_names():
        if not fpq_name.endswith(".weight"):
            continue
        # fpq_name: "blocks.0.self_attn.q.weight"
        # → diffusers state dict key: "blocks.0.attn1.to_q.weight"
        diffusers_key = remap_wan_to_diffusers(fpq_name)
        # module path = drop .weight suffix
        module_path = diffusers_key[: -len(".weight")]
        resolver_map[module_path] = fpq_name

    def resolver(module_path):
        return resolver_map.get(module_path)

    return resolver, resolver_map


# ── Model loading ──────────────────────────────────────────────────────

def load_wan_original(safetensors_path, device="cpu", dtype=torch.bfloat16):
    """Load WanTransformer3DModel from original safetensors weights."""
    from diffusers import WanTransformer3DModel
    from safetensors.torch import load_file

    print(f"  Creating WanTransformer3DModel skeleton from config...")
    model = WanTransformer3DModel(**WAN_1_3B_CONFIG)

    print(f"  Loading weights from {os.path.basename(safetensors_path)}...")
    state = load_file(safetensors_path)

    model_state = model.state_dict()
    loaded = 0
    unmapped = []
    for file_key, tensor in state.items():
        model_key = remap_wan_to_diffusers(file_key)
        if model_key in model_state:
            target = model_state[model_key]
            if tensor.shape != target.shape and tensor.numel() == target.numel():
                tensor = tensor.reshape(target.shape)
            model_state[model_key] = tensor.to(target.dtype)
            loaded += 1
        else:
            unmapped.append((file_key, model_key))

    model.load_state_dict(model_state)
    model = model.to(dtype).eval()

    print(f"  Loaded {loaded}/{len(model_state)} tensors "
          f"({len(unmapped)} unmapped{'  ← check remap_key' if unmapped else ''})")
    if unmapped:
        for fk, mk in unmapped[:4]:
            print(f"    {fk!r} → {mk!r}")
    return model


# ── Metrics ────────────────────────────────────────────────────────────

def compute_metrics(a, b, label=""):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    cos  = F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
    mse  = ((a_f - b_f) ** 2).mean().item()
    psnr = 10.0 * np.log10(4.0 / mse) if mse > 0 else float("inf")
    max_err  = (a_f - b_f).abs().max().item()
    mean_err = (a_f - b_f).abs().mean().item()
    rel_err  = ((a_f - b_f).norm() / (a_f.norm() + 1e-9)).item()
    return {
        "label": label,
        "cosine": cos,
        "mse": mse,
        "psnr_db": psnr,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "rel_error": rel_err,
    }


# ── Forward pass helper ────────────────────────────────────────────────

@torch.no_grad()
def forward_pass(model, hidden_states, encoder_hidden_states, timestep, device):
    model.to(device)
    t0 = time.time()
    out = model(
        hidden_states.to(device),
        encoder_hidden_states=encoder_hidden_states.to(device),
        timestep=timestep.to(device),
        return_dict=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elapsed = time.time() - t0
    if isinstance(out, tuple):
        out = out[0]
    out = out.cpu()
    model.to("cpu")
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out, elapsed


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wan2.1-T2V-1.3B SLI end-to-end benchmark"
    )
    parser.add_argument("--safetensors", required=True,
                        help="Path to diffusion_pytorch_model.safetensors (original Wan)")
    parser.add_argument("--fpq", required=True,
                        help="Path to wan2.1-t2v-1.3b.fpq (v12 FPQ compressed)")
    parser.add_argument("--out", default="/tmp/wan_sli_results.json",
                        help="Output JSON path")
    parser.add_argument("--timestep", type=float, default=500.0,
                        help="Main comparison timestep (0–999)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full timestep sweep [0,100,500,900,999]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Wan2.1-T2V-1.3B — SLI Bridge End-to-End Benchmark")
    print("=" * 70)
    print(f"  device: {device}  |  dtype: {dtype}  |  seed: {args.seed}")
    print(f"  safetensors: {args.safetensors}")
    print(f"  fpq: {args.fpq}")

    # ── Synthetic inputs (same as wan_dit_compare.py) ──
    batch, num_frames = 1, 1
    h_latent, w_latent = 60, 104   # 480/8, 832/8
    latent_channels = 16
    text_seq_len, text_dim = 64, 4096

    hidden_states = torch.randn(batch, latent_channels, num_frames, h_latent, w_latent,
                                dtype=dtype, device="cpu")
    encoder_hidden_states = torch.randn(batch, text_seq_len, text_dim,
                                        dtype=dtype, device="cpu")
    main_ts = torch.tensor([args.timestep], dtype=dtype, device="cpu")

    print(f"\n  inputs: latent {hidden_states.shape}  text {encoder_hidden_states.shape}")

    # ── Phase 1: Load original model ──
    print(f"\n{'─'*70}")
    print("[1/4] Loading original Wan model...")
    t_load0 = time.time()
    model_orig = load_wan_original(args.safetensors, device="cpu", dtype=dtype)
    print(f"  Load time: {time.time()-t_load0:.1f}s")

    # ── Phase 2: Load FPQ + build name resolver ──
    print(f"\n{'─'*70}")
    print("[2/4] Loading .fpq model + building name resolver...")
    t_fpq0 = time.time()
    fpq = FPQModel(args.fpq)
    info = fpq.info()
    print(f"  Tensors: {info['n_tensors']} total  "
          f"({info['n_sli']} SLI, {info['n_passthrough']} passthrough)")

    resolver, resolver_map = build_wan_name_resolver(fpq)
    print(f"  Name resolver: {len(resolver_map)} diffusers paths mapped")

    # ── Phase 3: Patch a copy of the original model ──
    print(f"\n{'─'*70}")
    print("[3/4] Patching copy with FPQ SLI...")
    model_sli = copy.deepcopy(model_orig)
    patched = patch_model(model_sli, fpq, prefix="", verbose=True,
                          name_resolver=resolver)
    print(f"\n  SLI layers: {len(patched)}")
    sample_keys = list(patched.keys())[:3]
    for k in sample_keys:
        print(f"    {k}  →  {patched[k]}")
    print(f"  Load time total: {time.time()-t_fpq0:.1f}s")

    if len(patched) == 0:
        print("\nERROR: No layers were patched — name resolver produced no matches.")
        print("Dumping first 10 resolver_map entries:")
        for k, v in list(resolver_map.items())[:10]:
            print(f"  {k!r} → {v!r}")
        print("\nDumping first 5 model named_modules (Linear only):")
        for n, m in model_sli.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f"  {n!r} ({m.in_features}→{m.out_features})")
                import itertools
                if sum(1 for _ in model_sli.named_modules() if isinstance(_, torch.nn.Linear)) > 5:
                    break
        sys.exit(1)

    # ── Phase 4: Forward passes ──
    print(f"\n{'─'*70}")
    print("[4/4] Running forward passes...")

    print(f"  [original]  timestep={args.timestep}")
    out_orig, t_orig = forward_pass(model_orig, hidden_states, encoder_hidden_states,
                                    main_ts, device)
    print(f"    done in {t_orig:.2f}s  →  shape {out_orig.shape}")

    print(f"  [SLI]       timestep={args.timestep}")
    out_sli, t_sli = forward_pass(model_sli, hidden_states, encoder_hidden_states,
                                  main_ts, device)
    print(f"    done in {t_sli:.2f}s  →  shape {out_sli.shape}")

    # ── Main metrics ──
    main_metrics = compute_metrics(out_orig, out_sli, label=f"t={args.timestep}")

    print(f"\n{'='*70}")
    print("  MAIN RESULT")
    print(f"{'='*70}")
    print(f"  Cosine similarity:   {main_metrics['cosine']:.8f}")
    print(f"  PSNR:                {main_metrics['psnr_db']:.2f} dB")
    print(f"  MSE:                 {main_metrics['mse']:.2e}")
    print(f"  Max abs error:       {main_metrics['max_abs_error']:.4f}")
    print(f"  Relative error:      {main_metrics['rel_error']*100:.3f}%")
    print(f"  SLI layers patched:  {len(patched)}")
    print(f"  Original time:       {t_orig:.2f}s")
    print(f"  SLI time:            {t_sli:.2f}s")

    # ── Per-channel cosine (first 4 channels) ──
    print(f"\n{'─'*70}")
    print("  PER-CHANNEL (first 4 output channels)")
    channel_metrics = []
    for c in range(min(4, out_orig.shape[1])):
        ch_a = out_orig[0, c].float().flatten()
        ch_b = out_sli[0, c].float().flatten()
        ch_cos = F.cosine_similarity(ch_a.unsqueeze(0), ch_b.unsqueeze(0)).item()
        ch_mse = ((ch_a - ch_b) ** 2).mean().item()
        print(f"  ch {c}: cos={ch_cos:.8f}  mse={ch_mse:.2e}")
        channel_metrics.append({"channel": c, "cosine": ch_cos, "mse": ch_mse})

    # ── Optional timestep sweep ──
    sweep_results = []
    if args.sweep:
        print(f"\n{'─'*70}")
        print("  TIMESTEP SWEEP")
        print(f"  {'Timestep':>10}  {'Cosine':>12}  {'PSNR':>10}  {'MSE':>12}")
        print(f"  {'─'*50}")
        for ts_val in [0.0, 100.0, 500.0, 900.0, 999.0]:
            ts = torch.tensor([ts_val], dtype=dtype, device="cpu")
            o_orig, _ = forward_pass(model_orig, hidden_states, encoder_hidden_states,
                                     ts, device)
            o_sli, _  = forward_pass(model_sli,  hidden_states, encoder_hidden_states,
                                     ts, device)
            m = compute_metrics(o_orig, o_sli, label=f"t={ts_val}")
            print(f"  {ts_val:10.1f}  {m['cosine']:12.8f}  {m['psnr_db']:10.2f}  {m['mse']:12.2e}")
            sweep_results.append(m)

    # ── Save results ──
    results = {
        "model": "Wan2.1-T2V-1.3B",
        "fpq_path": args.fpq,
        "safetensors_path": args.safetensors,
        "device": device,
        "dtype": str(dtype),
        "seed": args.seed,
        "fpq_info": info,
        "patched_layers": len(patched),
        "main_metrics": main_metrics,
        "channel_metrics": channel_metrics,
        "sweep": sweep_results,
        "timing": {
            "original_s": t_orig,
            "sli_s": t_sli,
            "slowdown": t_sli / max(t_orig, 0.001),
        },
        "input_shapes": {
            "hidden_states": list(hidden_states.shape),
            "encoder_hidden_states": list(encoder_hidden_states.shape),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Results saved → {args.out}")
    print(f"{'='*70}")

    # Summary line for easy grep
    print(f"\nSUMMARY: cos={main_metrics['cosine']:.6f}  "
          f"PSNR={main_metrics['psnr_db']:.2f}dB  "
          f"sli_layers={len(patched)}  "
          f"t_sli={t_sli:.1f}s  t_orig={t_orig:.1f}s")


if __name__ == "__main__":
    main()
