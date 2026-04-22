#!/usr/bin/env python3
"""Generate minimal Bonfyre recipes from hand-curated Parameter Golf leaderboard."""

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


def recipe_from_entry(track: str, entry: dict) -> dict:
    run = entry.get("run", "unknown")
    rid = f"pgh-{slugify(track)}-{slugify(run)}"
    score = entry.get("score")

    description = f"{run} (score={score}, author={entry.get('author')}, date={entry.get('date')}). {entry.get('summary', '')}".strip()

    return {
        "recipe_id": rid,
        "code": rid,
        "name": run,
        "version": "1.0.0",
        "category": "parameter-golf-hand",
        "description": description,
        "tags": ["parameter-golf", "leaderboard", "tiny-model", track],
        "inputs": [
            {
                "name": "run_name",
                "type": "string",
                "description": "Leaderboard run name"
            }
        ],
        "outputs": [
            {
                "name": "run_card",
                "path": "{out}/card.json",
                "type": "application/json"
            }
        ],
        "metadata": {
            "source": "parameter-golf-hand",
            "track": track,
            "rank": entry.get("rank"),
            "score": score,
            "author": entry.get("author"),
            "summary": entry.get("summary"),
            "date": entry.get("date")
        },
        "stages": [
            {
                "id": "emit_card",
                "bin": "python3",
                "args": [
                    "-c",
                    (
                        "import json, pathlib; "
                        "p=pathlib.Path('{out}/card.json'); p.parent.mkdir(parents=True, exist_ok=True); "
                        f"data={json.dumps(entry, ensure_ascii=True)}; "
                        "p.write_text(json.dumps(data, indent=2)+'\\n', encoding='utf-8'); "
                        "print(json.dumps(data, indent=2))"
                    )
                ],
                "depends_on": []
            }
        ]
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(src.read_text(encoding="utf-8"))
    tracks = payload.get("tracks", {})

    recipes = []
    for track, entries in tracks.items():
        for entry in entries:
            recipe = recipe_from_entry(track, entry)
            target = out / f"{recipe['recipe_id']}.json"
            target.write_text(json.dumps(recipe, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            recipes.append(
                {
                    "recipe_id": recipe["recipe_id"],
                    "name": recipe["name"],
                    "track": track,
                    "rank": entry.get("rank"),
                    "score": entry.get("score")
                }
            )

    index = {
        "source": payload.get("source"),
        "updated": payload.get("updated"),
        "recipe_count": len(recipes),
        "recipes": recipes
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    print(f"generated_recipes={len(recipes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
