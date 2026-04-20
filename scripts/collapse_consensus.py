#!/usr/bin/env python3
"""
collapse_consensus.py — majority-vote label merger for T14 NER.

Used when three external ONNX/transformers teacher models produce
independent NER predictions that must be fused before student training.

Modes
  teach   Run a single teacher model over a corpus directory.
          Writes <out>/<stem>.json per input file.

  merge   Merge outputs from 2+ teach directories via majority vote
          (span must appear in ≥ ceil(n/2)+1 teacher outputs).
          Writes <out>/labels.jsonl consumed by collapse_train.py.

Usage
  # inside bonfyre-run stage:
  python3 scripts/collapse_consensus.py teach \\
      --model /models/dslim-bert-base-NER.onnx \\
      --corpus /data/corpus \\
      --out    /run/t14/teach-dslim

  python3 scripts/collapse_consensus.py merge \\
      --inputs /run/t14/teach-elastic,/run/t14/teach-dslim,/run/t14/teach-babel \\
      --out    /run/t14/consensus

Deps: onnxruntime transformers (for tokenizer only)
"""
import argparse
import json
import os
import sys

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


# ── teach ──────────────────────────────────────────────────────────────────────
def cmd_teach(model_path: str, corpus_dir: str, out_dir: str):
    if ort is None:
        sys.exit("[consensus] missing dep: pip install onnxruntime")
    if AutoTokenizer is None:
        sys.exit("[consensus] missing dep: pip install transformers")

    os.makedirs(out_dir, exist_ok=True)

    # Load ONNX session
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]

    # Use BERT tokenizer from model directory (or nearest match)
    model_dir = os.path.dirname(model_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        # Fallback: use bert-base-uncased tokenizer metadata
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    label2id = getattr(tokenizer, "id2label", None)

    txt_files = sorted(
        f for f in os.listdir(corpus_dir)
        if f.endswith(".txt")
    )
    if not txt_files:
        sys.exit(f"[consensus] no .txt files in {corpus_dir}")

    print(f"[consensus] teach: {len(txt_files)} files  model={os.path.basename(model_path)}")

    for fname in txt_files:
        stem = fname[:-4]
        text = open(os.path.join(corpus_dir, fname)).read().strip()
        if not text:
            continue

        enc = tokenizer(text, truncation=True, max_length=512,
                        return_tensors="np", return_offsets_mapping=True)
        offsets   = enc.pop("offset_mapping")[0]
        input_ids = enc["input_ids"]
        attn      = enc.get("attention_mask")

        feed = {"input_ids": input_ids}
        if attn is not None and "attention_mask" in input_names:
            feed["attention_mask"] = attn
        if "token_type_ids" in input_names and "token_type_ids" in enc:
            feed["token_type_ids"] = enc["token_type_ids"]

        outputs = sess.run(None, feed)
        logits  = outputs[0][0]  # [seq_len, n_tags]
        preds   = logits.argmax(axis=-1).tolist()

        # Convert token predictions to character-level spans
        entities = []
        in_entity = False
        ent_start = None
        for i, (label_id, (char_start, char_end)) in enumerate(zip(preds, offsets)):
            if char_start == 0 and char_end == 0:
                in_entity = False
                continue
            tag = (label2id or {}).get(label_id, str(label_id))
            if isinstance(tag, str) and tag.startswith("B-"):
                if in_entity and ent_start is not None:
                    entities.append({"start": ent_start, "end": int(prev_end),
                                     "label": "ENT"})
                in_entity = True
                ent_start = int(char_start)
            elif isinstance(tag, str) and tag.startswith("I-"):
                pass  # continue span
            else:
                if in_entity and ent_start is not None:
                    entities.append({"start": ent_start, "end": int(prev_end),
                                     "label": "ENT"})
                in_entity  = False
                ent_start  = None
            prev_end = char_end  # noqa: F841 — used on next iter

        if in_entity and ent_start is not None:
            entities.append({"start": ent_start, "end": int(offsets[-1][1]),
                             "label": "ENT"})

        out = {"text": text, "entities": entities}
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(out, f)

    print(f"[consensus] teach done → {out_dir}/")


# ── merge ──────────────────────────────────────────────────────────────────────
def _spans_to_set(entities):
    """Convert entity list to frozenset of (start, end) tuples."""
    return frozenset((e["start"], e["end"]) for e in entities)


def cmd_merge(input_dirs: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    n_teachers = len(input_dirs)
    threshold  = n_teachers // 2 + 1  # majority: ≥ ceil(n/2)+1

    # Collect all stems from first teacher dir
    stems = sorted(
        f[:-5] for f in os.listdir(input_dirs[0])
        if f.endswith(".json")
    )
    if not stems:
        sys.exit(f"[consensus] no .json files in {input_dirs[0]}")

    print(f"[consensus] merge: {len(stems)} stems, "
          f"{n_teachers} teachers, threshold={threshold}")

    out_path = os.path.join(out_dir, "labels.jsonl")
    written  = 0
    with open(out_path, "w") as fout:
        for stem in stems:
            text    = None
            all_spans: list[frozenset] = []

            for d in input_dirs:
                fpath = os.path.join(d, f"{stem}.json")
                if not os.path.exists(fpath):
                    continue
                with open(fpath) as f:
                    obj = json.load(f)
                if text is None:
                    text = obj.get("text", "")
                all_spans.append(_spans_to_set(obj.get("entities", [])))

            if not text or not all_spans:
                continue

            # Count votes per span
            span_votes: dict = {}
            for span_set in all_spans:
                for span in span_set:
                    span_votes[span] = span_votes.get(span, 0) + 1

            # Keep spans that meet the threshold
            agreed = [
                {"start": s, "end": e, "label": "ENT"}
                for (s, e), votes in span_votes.items()
                if votes >= threshold
            ]
            agreed.sort(key=lambda x: x["start"])

            record = {"text": text, "entities": agreed}
            fout.write(json.dumps(record) + "\n")
            written += 1

    print(f"[consensus] merge done: {written} records → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="NER teacher runner and majority-vote consensus merger")
    sub = p.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("teach", help="run single teacher ONNX model over corpus")
    t.add_argument("--model",  required=True, help="path to teacher ONNX model")
    t.add_argument("--corpus", required=True, help="directory of .txt input files")
    t.add_argument("--out",    required=True, help="output directory for per-file JSONs")

    m = sub.add_parser("merge", help="majority-vote merge across teacher directories")
    m.add_argument("--inputs", required=True,
                   help="comma-separated list of teach output directories")
    m.add_argument("--out",    required=True, help="output directory for labels.jsonl")

    args = p.parse_args()

    if args.mode == "teach":
        cmd_teach(args.model, args.corpus, args.out)
    elif args.mode == "merge":
        dirs = [d.strip() for d in args.inputs.split(",") if d.strip()]
        if len(dirs) < 2:
            sys.exit("[consensus] --inputs requires ≥2 comma-separated directories")
        cmd_merge(dirs, args.out)


if __name__ == "__main__":
    main()
