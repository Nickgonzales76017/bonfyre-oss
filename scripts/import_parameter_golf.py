#!/usr/bin/env python3
"""Convert openai/parameter-golf records into minimal Bonfyre-native recipes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def safe_name(folder: str) -> str:
    return folder.replace("_", " ").strip()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def recipe_for_record(track: str, folder: str, submission: dict) -> dict:
    record_rel = f"records/{track}/{folder}"
    rid = slugify(f"pg-{track}-{folder}")

    name = submission.get("name") or safe_name(folder)
    blurb = submission.get("blurb") or "Imported from OpenAI Parameter Golf records"

    tags = ["parameter-golf", "tiny-model", track]
    if "non_record" in track:
        tags.append("non-record")

    metadata = {
        "source_repo": "openai/parameter-golf",
        "source_record": record_rel,
        "author": submission.get("author"),
        "github_id": submission.get("github_id"),
        "date": submission.get("date"),
        "val_bpb": submission.get("val_bpb"),
        "val_loss": submission.get("val_loss"),
        "bytes_total": submission.get("bytes_total"),
        "bytes_code": submission.get("bytes_code"),
    }

    return {
        "recipe_id": rid,
        "code": rid,
        "name": name,
        "version": "1.0.0",
        "category": "parameter-golf",
        "description": blurb,
        "tags": tags,
        "metadata": metadata,
        "inputs": [
            {
                "name": "record_ref",
                "type": "string",
                "description": "Parameter Golf record path under records/"
            }
        ],
        "outputs": [
            {
                "name": "summary_json",
                "path": "{out}/summary.json",
                "type": "application/json"
            }
        ],
        "stages": [
            {
                "id": "fetch_record",
                "bin": "python3",
                "args": [
                    "scripts/parameter_golf_tool.py",
                    "fetch",
                    "--record",
                    record_rel,
                    "--out",
                    "{out}/record",
                ],
                "depends_on": [],
            },
            {
                "id": "summarize_record",
                "bin": "python3",
                "args": [
                    "scripts/parameter_golf_tool.py",
                    "summarize",
                    "--record-json",
                    "{out}/record/submission.json",
                    "--out",
                    "{out}/summary.json",
                ],
                "depends_on": ["fetch_record"],
            },
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to local parameter-golf repo")
    ap.add_argument("--out", required=True, help="Output directory for generated recipes")
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records_root = src / "records"
    entries = []

    for track_dir in sorted(p for p in records_root.iterdir() if p.is_dir()):
        track = track_dir.name
        for rec_dir in sorted(p for p in track_dir.iterdir() if p.is_dir()):
            sub_path = rec_dir / "submission.json"
            sub = load_json(sub_path) if sub_path.exists() else {}
            recipe = recipe_for_record(track, rec_dir.name, sub)

            out_path = out / f"{recipe['recipe_id']}.json"
            out_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")

            entries.append(
                {
                    "recipe_id": recipe["recipe_id"],
                    "name": recipe["name"],
                    "track": track,
                    "source_record": recipe["metadata"]["source_record"],
                    "val_bpb": recipe["metadata"]["val_bpb"],
                    "bytes_total": recipe["metadata"]["bytes_total"],
                }
            )

    index = {
        "source_repo": "openai/parameter-golf",
        "recipe_count": len(entries),
        "recipes": entries,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    print(f"generated_recipes={len(entries)}")
    print(f"output_dir={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
