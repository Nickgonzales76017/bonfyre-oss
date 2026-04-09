#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: scripts/run_public_source_proof.sh <public-url> <title> [out-dir]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="$1"
TITLE="$2"
OUT_DIR="${3:-$(mktemp -d /tmp/bonfyre-public-proof.XXXXXX)}"

DOWNLOAD_DIR="$OUT_DIR/download"
TRANS_DIR="$OUT_DIR/transcribe"
CLEAN_DIR="$OUT_DIR/clean"
PARA_DIR="$OUT_DIR/paragraph"
BRIEF_DIR="$OUT_DIR/brief"
PROOF_JSON="$OUT_DIR/proof.json"
PROOF_MD="$OUT_DIR/proof.md"

mkdir -p "$DOWNLOAD_DIR" "$TRANS_DIR" "$CLEAN_DIR" "$PARA_DIR" "$BRIEF_DIR"

SOURCE_TEMPLATE="$DOWNLOAD_DIR/source.%(ext)s"
SOURCE_META="$DOWNLOAD_DIR/source-meta.json"

yt-dlp --dump-single-json "$URL" > "$SOURCE_META"
yt-dlp -f bestaudio --no-playlist -o "$SOURCE_TEMPLATE" "$URL"

SOURCE_FILE="$(find "$DOWNLOAD_DIR" -maxdepth 1 -type f ! -name 'source-meta.json' | head -n 1)"
if [[ -z "${SOURCE_FILE:-}" ]]; then
  echo "failed to download public source audio" >&2
  exit 1
fi

NORMALIZED_WAV="$OUT_DIR/source.wav"
"$ROOT/cmd/BonfyreMediaPrep/bonfyre-media-prep" normalize "$SOURCE_FILE" "$NORMALIZED_WAV"
"$ROOT/cmd/BonfyreTranscribe/bonfyre-transcribe" "$NORMALIZED_WAV" "$TRANS_DIR"
"$ROOT/cmd/BonfyreTranscriptClean/bonfyre-transcript-clean" --transcript "$TRANS_DIR/transcript.json" --out "$CLEAN_DIR/clean.txt"
"$ROOT/cmd/BonfyreParagraph/bonfyre-paragraph" --input "$CLEAN_DIR/clean.txt" --out "$PARA_DIR/paragraphs.txt"
"$ROOT/cmd/BonfyreBrief/bonfyre-brief" "$PARA_DIR/paragraphs.txt" "$BRIEF_DIR" --title "$TITLE"

TITLE_JSON="$(jq -Rsa . <<<"$TITLE")"
URL_JSON="$(jq -Rsa . <<<"$URL")"
OUT_DIR_JSON="$(jq -Rsa . <<<"$OUT_DIR")"

jq -n \
  --argjson source_meta "$(cat "$SOURCE_META")" \
  --argjson transcribe_meta "$(cat "$TRANS_DIR/meta.json")" \
  --arg title "$TITLE" \
  --arg url "$URL" \
  --arg out_dir "$OUT_DIR" \
  '{
    title: $title,
    public_url: $url,
    out_dir: $out_dir,
    source: {
      title: ($source_meta.title // $title),
      channel: ($source_meta.channel // ""),
      duration_seconds: ($source_meta.duration // 0)
    },
    transcribe: {
      avg_confidence: ($transcribe_meta.avg_confidence // 0),
      avg_logprob: ($transcribe_meta.avg_logprob // 0),
      rtf: ($transcribe_meta.rtf // 0),
      segments_total: ($transcribe_meta.segments_total // 0),
      segments_hallucinated: ($transcribe_meta.segments_hallucinated // 0)
    },
    outputs: {
      transcript_json: ($out_dir + "/transcribe/transcript.json"),
      transcript_txt: ($out_dir + "/transcribe/transcript.txt"),
      clean_txt: ($out_dir + "/clean/clean.txt"),
      paragraphs_txt: ($out_dir + "/paragraph/paragraphs.txt"),
      brief_md: ($out_dir + "/brief/brief.md"),
      brief_artifact: ($out_dir + "/brief/artifact.json")
    },
    retained_media: false
  }' > "$PROOF_JSON"

{
  echo "# Bonfyre Public Source Proof"
  echo
  echo "- Title: $TITLE"
  echo "- Public origin: $URL"
  echo "- Working directory: $OUT_DIR"
  echo "- Source channel: $(jq -r '.channel // "unknown"' "$SOURCE_META")"
  echo "- Source duration: $(jq -r '.duration // 0' "$SOURCE_META") seconds"
  echo "- Avg confidence: $(jq -r '.avg_confidence // 0' "$TRANS_DIR/meta.json")"
  echo "- Hallucinated segments: $(jq -r '.segments_hallucinated // 0' "$TRANS_DIR/meta.json")"
  echo "- Realtime factor: $(jq -r '.rtf // 0' "$TRANS_DIR/meta.json")"
  echo
  echo "## Outputs"
  echo
  echo "- Transcript JSON: $TRANS_DIR/transcript.json"
  echo "- Clean transcript: $CLEAN_DIR/clean.txt"
  echo "- Paragraphs: $PARA_DIR/paragraphs.txt"
  echo "- Brief: $BRIEF_DIR/brief.md"
  echo
  echo "## Brief Preview"
  echo
  sed -n '1,40p' "$BRIEF_DIR/brief.md"
} > "$PROOF_MD"

rm -f "$SOURCE_FILE" "$NORMALIZED_WAV" "$TRANS_DIR/input.denoised.wav" "$TRANS_DIR/normalized.wav"

echo "Proof JSON: $PROOF_JSON"
echo "Proof Markdown: $PROOF_MD"
