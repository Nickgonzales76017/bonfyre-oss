#!/usr/bin/env bash
set -euo pipefail

usage(){
  cat <<EOF
Usage: $0 --map MAP.json [--dry-run]

Copies demo artifacts from repo root `site/demos/<demo>` into each Pages repo's `site/` folder
based on the mappings in MAP.json. Commits and pushes changes unless --dry-run is provided.
EOF
}

DRY_RUN=0
MAP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --map) MAP="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [ -z "$MAP" ]; then
  usage; exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq required" >&2; exit 1
fi

BASE_PROJECTS_DIR="/Users/$(whoami)/Projects"
tmp=$(mktemp -d /tmp/move-artifacts.XXXX)
echo "tmp: $tmp"

jq -c '.mappings[]' "$MAP" | while read -r m; do
  demo_dir=$(echo "$m" | jq -r '.demo_dir')
  remote=$(echo "$m" | jq -r '.remote')
  repo_basename=$(basename "$remote" .git)
  repo_path="$BASE_PROJECTS_DIR/$repo_basename"
  site_dir="$repo_path/site"
  echo "\n== $demo_dir -> $repo_basename/site =="

  if [ ! -d "$repo_path" ]; then
    echo "Local repo missing: $repo_path" >&2
    continue
  fi
  if [ ! -d "$site_dir" ]; then
    echo "Site dir missing: $site_dir; creating" 
    if [ "$DRY_RUN" -eq 0 ]; then mkdir -p "$site_dir"; fi
  fi

  # copy listed paths (preserve directory structure)
  paths=$(echo "$m" | jq -r '.paths[]')
  changed=0
  for p in $paths; do
    src="$demo_dir/$p"
    if [ ! -e "$src" ]; then
      echo "Missing source: $src (skipping)"
      continue
    fi
    # if src is a directory, copy its contents into site/<basename(p)>
    if [ -d "$src" ]; then
      dest_dir="$site_dir/$(basename "$p")"
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY RUN: would copy dir $src -> $dest_dir"
        changed=1
        continue
      fi
      mkdir -p "$dest_dir"
      cp -r "$src"/* "$dest_dir/" || true
      echo "Copied dir $src -> $dest_dir"
      changed=1
    else
      # src is a file; copy to site/<filename>
      filename=$(basename "$p")
      dest_file="$site_dir/$filename"
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY RUN: would copy file $src -> $dest_file"
        changed=1
        continue
      fi
      mkdir -p "$(dirname "$dest_file")"
      cp "$src" "$dest_file"
      echo "Copied file $src -> $dest_file"
      # remove legacy directory created by previous incorrect runs, if present
      legacy_dir="$site_dir/$filename"
      if [ -d "$legacy_dir" ] && [ -f "$legacy_dir/$filename" ]; then
        rm -rf "$legacy_dir"
        echo "Removed legacy directory $legacy_dir"
      fi
      changed=1
    fi
  done

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would commit/push in $repo_path"
    continue
  fi

  if [ "$changed" -eq 1 ]; then
    pushd "$repo_path" >/dev/null
    git add site || true
    git commit -m "Add demo artifacts into site/ for Pages" || echo "No changes to commit"
    git push origin HEAD || echo "Push failed for $repo_basename"
    popd >/dev/null
  else
    echo "No artifacts copied for $repo_basename"
  fi
done

echo "Done. tmp: $tmp"
