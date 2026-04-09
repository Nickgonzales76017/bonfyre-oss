#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --map MAP.json [--dry-run]

MAP.json format: { "mappings": [ { "demo_dir":"site/demos/xxx", "remote":"git@...", "branch":"main", "paths":["artifact.json","brief.md"] } ] }

This script clones each remote into a temporary dir, copies listed artifact files from the demo dir, commits, and pushes.
It requires git access to the remotes. Use --dry-run to print planned actions without modifying remotes.
EOF
}

DRY_RUN=0
MAP_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --map) MAP_FILE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [ -z "$MAP_FILE" ]; then
  usage; exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2; exit 1
fi

tmproot=$(mktemp -d /tmp/artifacts.XXXX)
echo "Using tmp: $tmproot"

jq -c '.mappings[]' "$MAP_FILE" | while read -r mapping; do
  demo_dir=$(echo "$mapping" | jq -r '.demo_dir')
  remote=$(echo "$mapping" | jq -r '.remote')
  branch=$(echo "$mapping" | jq -r '.branch')
  paths=$(echo "$mapping" | jq -r '.paths[]')

  echo "\n== Processing $demo_dir -> $remote#$branch =="
  if [ ! -d "$demo_dir" ]; then
    echo "Demo dir missing: $demo_dir" >&2
    continue
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would clone $remote"
    echo "DRY RUN: would copy files:"; for p in $paths; do echo " - $demo_dir/$p"; done
    echo "DRY RUN: would commit and push to $branch"
    continue
  fi

  repo_basename=$(basename "$remote" .git)
  # Prefer existing local clone if present in common locations
  candidates=("$repo_basename" "./$repo_basename" "../$repo_basename" "/Users/$(whoami)/Projects/$repo_basename" "/Users/$(whoami)/$repo_basename")
  repo_tmp=""
  for c in "${candidates[@]}"; do
    if [ -d "$c/.git" ]; then
      repo_tmp="$c"
      break
    fi
  done
  if [ -z "$repo_tmp" ]; then
    repo_tmp="$tmproot/$repo_basename"
    echo "Cloning $remote -> $repo_tmp"
    git clone --depth 1 --branch "$branch" "$remote" "$repo_tmp" || { echo "Clone failed: $remote" >&2; continue; }
  else
    echo "Using existing local repo: $repo_tmp"
  fi

  changed=0
  for p in $paths; do
    src="$demo_dir/$p"
    if [ -e "$src" ]; then
      dest_dir="$repo_tmp/$(dirname "$p")"
      mkdir -p "$dest_dir"
      cp -r "$src" "$dest_dir/" || { echo "Copy failed: $src" >&2; }
      changed=1
      echo "Copied $src -> $dest_dir/"
    else
      echo "Missing artifact (skipping): $src" >&2
    fi
  done

  if [ "$changed" -eq 1 ]; then
    pushd "$repo_tmp" >/dev/null
    git add .
    git commit -m "Add demo artifacts from $demo_dir" || echo "No changes to commit"
    git push origin "$branch" || echo "Push failed (check remote + credentials)"
    popd >/dev/null
  else
    echo "No artifacts copied; skipping commit/push"
  fi
done

echo "Done. Clean tmp: $tmproot (you may remove it)"
