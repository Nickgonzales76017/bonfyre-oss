#!/usr/bin/env python3
"""Lightweight tooling for Parameter Golf record recipes.

This script keeps recipe payloads small by fetching record metadata on demand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

RAW_BASE = "https://raw.githubusercontent.com/openai/parameter-golf/main"


def norm_record_path(record: str) -> str:
    record = record.strip().lstrip("/")
    if not record.startswith("records/"):
        record = f"records/{record}"
    return record


def fetch_text(url: str) -> str | None:
    try:
        with urlopen(url, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError):
        return None


def cmd_fetch(args: argparse.Namespace) -> int:
    record = norm_record_path(args.record)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = ["submission.json", "README.md", "train_gpt.py", "requirements.txt"]
    fetched = []

    for name in files:
        url = f"{RAW_BASE}/{record}/{name}"
        body = fetch_text(url)
        if body is None:
            continue
        target = out_dir / name
        target.write_text(body, encoding="utf-8")
        fetched.append({"file": name, "url": url})

    manifest = {
        "record": record,
        "fetched_count": len(fetched),
        "fetched": fetched,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    record_json = Path(args.record_json)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "name": None,
        "author": None,
        "github_id": None,
        "date": None,
        "val_bpb": None,
        "val_loss": None,
        "bytes_total": None,
        "bytes_code": None,
    }

    if record_json.exists():
        try:
            payload = json.loads(record_json.read_text(encoding="utf-8"))
            for key in summary:
                if key in payload:
                    summary[key] = payload[key]
        except json.JSONDecodeError:
            pass

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parameter Golf tooling for Bonfyre recipes")
    sub = p.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="Fetch minimal record metadata files")
    fetch.add_argument("--record", required=True, help="Record path under records/")
    fetch.add_argument("--out", required=True, help="Output directory")
    fetch.set_defaults(func=cmd_fetch)

    summarize = sub.add_parser("summarize", help="Summarize submission.json")
    summarize.add_argument("--record-json", required=True, help="Path to submission.json")
    summarize.add_argument("--out", required=True, help="Output JSON path")
    summarize.set_defaults(func=cmd_summarize)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
