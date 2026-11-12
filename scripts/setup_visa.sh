#!/usr/bin/env bash
# Download VisA (about 1.8 GB, CC BY 4.0) and put it in the MVTec layout under data/visa.
# Source: https://github.com/amazon-science/spot-diff
#
# Usage: bash scripts/setup_visa.sh [dest]
set -euo pipefail

DEST="${1:-data}"
URL="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
ARCHIVE="$DEST/VisA_20220922.tar"
RAW="$DEST/VisA"

mkdir -p "$DEST" "$RAW"
if [ ! -f "$RAW/pcb4/image_anno.csv" ] && [ ! -f "$RAW/VisA/pcb4/image_anno.csv" ]; then
  if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading $URL ..."
    curl -L --fail -C - -o "$ARCHIVE" "$URL"
  fi
  echo "Extracting to $RAW ..."
  tar -xf "$ARCHIVE" -C "$RAW"
fi
# Some copies wrap the categories in an extra VisA/ folder.
[ -d "$RAW/VisA/pcb4" ] && RAW="$RAW/VisA"
python scripts/prepare_visa.py --raw "$RAW" --out "$DEST/visa"
echo "You can delete $ARCHIVE once this has finished."
