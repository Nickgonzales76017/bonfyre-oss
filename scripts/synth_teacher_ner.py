#!/usr/bin/env python3
"""
synth_teacher_ner.py — T14-C synthetic NER teacher.

Uses a lightweight HuggingFace NER pipeline (dslim/bert-base-NER or similar)
to produce consensus-format JSONL that collapse_train.py --task ner-bio
consumes via --consensus.

Output format (one JSON object per line):
  {"text": str, "entities": [{"start": int, "end": int, "label": str}]}

Falls back to a simple regex-based capitalized-noun extractor if transformers
NER model download fails (offline/air-gapped environments).

Usage:
  python3 synth_teacher_ner.py <corpus_dir> <out_dir>
"""
import argparse
import json
import os
import re
import sys

BATCH_SIZE  = 16
NER_MODEL   = "dslim/bert-base-NER"
MIN_SAMPLES = 10


def regex_fallback(text):
    """Extract capitalized multi-word spans as pseudo-entities."""
    entities = []
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
        if len(m.group()) >= 4:
            entities.append({
                "start": m.start(),
                "end":   m.end(),
                "label": "ENT",
            })
    return entities


def main():
    # Prevent tokenizers deadlock on macOS/forked processes
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    p = argparse.ArgumentParser()
    p.add_argument("corpus_dir")
    p.add_argument("out_dir")
    p.add_argument("--model", default=NER_MODEL)
    p.add_argument("--max-per-doc", type=int, default=3,
                   help="max sentences sampled per document (reduces size)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    consensus_path = os.path.join(args.out_dir, "consensus.jsonl")

    txt_files = sorted(f for f in os.listdir(args.corpus_dir) if f.endswith(".txt"))
    stems     = [f[:-4] for f in txt_files]

    if not stems:
        sys.exit(f"[synth_ner] no .txt files in {args.corpus_dir}")

    # Build sentence list — each doc → up to args.max_per_doc sentences
    sentences = []
    for stem in stems:
        path  = os.path.join(args.corpus_dir, f"{stem}.txt")
        text  = open(path, encoding="utf-8").read().strip()
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
        sentences.extend(sents[:args.max_per_doc])

    if len(sentences) < MIN_SAMPLES:
        sys.exit(f"[synth_ner] too few sentences ({len(sentences)}) in {args.corpus_dir}")

    # Try transformers NER pipeline
    use_hf = False
    ner    = None
    try:
        from transformers import pipeline as hf_pipeline
        print(f"[synth_ner] loading {args.model} ...")
        ner = hf_pipeline(
            "ner",
            model=args.model,
            aggregation_strategy="simple",
            device=-1,
        )
        use_hf = True
        print(f"[synth_ner] HF NER pipeline ready")
    except Exception as e:
        print(f"[synth_ner] HF NER unavailable ({e}); using regex fallback")

    print(f"[synth_ner] processing {len(sentences)} sentences …")

    written = 0
    n_entities = 0
    with open(consensus_path, "w") as out_f:
        if use_hf:
            for i in range(0, len(sentences), BATCH_SIZE):
                batch = sentences[i:i + BATCH_SIZE]
                try:
                    batch_results = ner(batch)
                except Exception:
                    batch_results = [[] for _ in batch]
                for text, ents in zip(batch, batch_results):
                    entities = [
                        {"start": int(e["start"]), "end": int(e["end"]),
                         "label": str(e.get("entity_group", e.get("entity", "ENT")))}
                        for e in ents
                    ]
                    n_entities += len(entities)
                    out_f.write(json.dumps({"text": text, "entities": entities}) + "\n")
                    written += 1
        else:
            for text in sentences:
                entities = regex_fallback(text)
                n_entities += len(entities)
                out_f.write(json.dumps({"text": text, "entities": entities}) + "\n")
                written += 1

    print(f"[synth_ner] wrote {written} records, {n_entities} entities → {consensus_path}")


if __name__ == "__main__":
    main()
