#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <offer_dir> <out.json>" >&2
  exit 2
fi
offer_dir="$1"
out_file="$2"
offer_json="$offer_dir/offer.json"
if [ ! -f "$offer_json" ]; then
  echo "Offer file not found: $offer_json" >&2
  exit 1
fi
price_cents=$(jq -r '.priceCents // 0' "$offer_json")
name=$(jq -r '.offerName // "offer"' "$offer_json")
cat > "$out_file" <<EOF
{
  "source": "gen_meter_from_offer",
  "offerName": "$name",
  "charges": [
    { "label": "offer", "amount_cents": $price_cents }
  ],
  "total_cents": $price_cents
}
EOF
echo "Generated meter: $out_file"
