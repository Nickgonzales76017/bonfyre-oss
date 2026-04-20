#!/usr/bin/env python3
"""
synth_teacher_paragraphs.py — T16-C synthetic paragraph-boundary teacher.

Uses bonfyre-paragraph to split documents into paragraph-level chunks,
then embeds each chunk with two sentence-transformer models (MiniLM + MPNet).
The resulting per-paragraph embeddings in sorted order give collapse_train.py
chunk-boundary a cleaner boundary signal than document-level embeddings:
cosine similarity naturally drops at paragraph/section transitions.

Outputs:
  <out_dir>/embed-a/  — MiniLM embeddings (one JSON per paragraph chunk)
  <out_dir>/embed-b/  — MPNet embeddings  (one JSON per paragraph chunk)

Usage:
  python3 synth_teacher_paragraphs.py <corpus_dir> <out_dir>
                                      [--model-a all-MiniLM-L6-v2]
                                      [--model-b all-mpnet-base-v2]
                                      [--bonfyre-paragraph bonfyre-paragraph]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

MODEL_A = "all-MiniLM-L6-v2"
MODEL_B = "all-mpnet-base-v2"
CHUNK_SIZE = 32   # sentences per "batch" for embed


def split_paragraphs_fallback(text):
    """Split text into paragraphs by blank lines or long runs."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        # Fallback: split every 3 sentences
        import re
        sents = re.split(r'(?<=[.!?])\s+', text)
        paras = [" ".join(sents[i:i+3]) for i in range(0, len(sents), 3)]
    return [p for p in paras if len(p) > 30]


def run_bonfyre_paragraph(bin_path, input_file, out_file):
    """Run bonfyre-paragraph; return True on success."""
    try:
        result = subprocess.run(
            [bin_path, "--input", input_file, "--out", out_file],
            capture_output=True, timeout=30
        )
        return result.returncode == 0 and os.path.exists(out_file)
    except Exception:
        return False


def embed_texts(model, texts):
    """Embed a list of texts with sentence-transformers."""
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir")
    p.add_argument("out_dir")
    p.add_argument("--model-a", default=MODEL_A)
    p.add_argument("--model-b", default=MODEL_B)
    p.add_argument("--bonfyre-paragraph", default="bonfyre-paragraph")
    args = p.parse_args()

    dir_a = os.path.join(args.out_dir, "embed-a")
    dir_b = os.path.join(args.out_dir, "embed-b")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    txt_files = sorted(f for f in os.listdir(args.corpus_dir) if f.endswith(".txt"))
    stems     = [f[:-4] for f in txt_files]

    if not stems:
        sys.exit(f"[synth_paragraphs] no .txt files in {args.corpus_dir}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("[synth_paragraphs] sentence-transformers required: pip install sentence-transformers")

    print(f"[synth_paragraphs] loading {args.model_a} + {args.model_b} …")
    model_a = SentenceTransformer(args.model_a)
    model_b = SentenceTransformer(args.model_b)

    # Check if bonfyre-paragraph is available
    try:
        probe = subprocess.run([args.bonfyre_paragraph, "--help"],
                               capture_output=True, timeout=5)
        bf_para_ok = True
        print(f"[synth_paragraphs] bonfyre-paragraph available")
    except Exception:
        bf_para_ok = False
        print(f"[synth_paragraphs] bonfyre-paragraph not found — using fallback splitter")

    # Collect all paragraph chunks globally
    all_chunks  = []   # list of (global_idx, text)
    global_idx  = 0

    for stem in stems:
        doc_path = os.path.join(args.corpus_dir, f"{stem}.txt")
        text     = open(doc_path, encoding="utf-8").read().strip()

        paragraphs = None
        if bf_para_ok:
            # Write doc to temp file, run bonfyre-paragraph, read result
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                             delete=False, encoding="utf-8") as tmp_in:
                tmp_in.write(text)
                tmp_in_path = tmp_in.name
            tmp_out_path = tmp_in_path + ".para"
            if run_bonfyre_paragraph(args.bonfyre_paragraph, tmp_in_path, tmp_out_path):
                try:
                    para_text = open(tmp_out_path, encoding="utf-8").read()
                    # bonfyre-paragraph writes paragraphs separated by blank lines
                    paragraphs = [p.strip() for p in para_text.split("\n\n") if p.strip()]
                except Exception:
                    pass
                finally:
                    for f in (tmp_in_path, tmp_out_path):
                        try:
                            os.unlink(f)
                        except OSError:
                            pass

        if not paragraphs:
            paragraphs = split_paragraphs_fallback(text)

        for para in paragraphs:
            if len(para) > 20:
                all_chunks.append((global_idx, para))
                global_idx += 1

    if len(all_chunks) < 3:
        sys.exit(f"[synth_paragraphs] too few paragraph chunks ({len(all_chunks)})")

    print(f"[synth_paragraphs] embedding {len(all_chunks)} paragraph chunks …")
    chunk_texts = [c[1] for c in all_chunks]

    vecs_a = embed_texts(model_a, chunk_texts)
    vecs_b = embed_texts(model_b, chunk_texts)

    written = 0
    for (gidx, _chunk_text), va, vb in zip(all_chunks, vecs_a, vecs_b):
        fname = f"{gidx:06d}.json"
        with open(os.path.join(dir_a, fname), "w") as f:
            json.dump({"embedding": va.tolist()}, f)
        with open(os.path.join(dir_b, fname), "w") as f:
            json.dump({"embedding": vb.tolist()}, f)
        written += 1

    print(f"[synth_paragraphs] wrote {written} chunk embeddings "
          f"→ {dir_a}/ + {dir_b}/")


if __name__ == "__main__":
    main()
