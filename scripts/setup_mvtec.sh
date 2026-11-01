#!/usr/bin/env bash
# Extract a downloaded MVTec AD archive into <dest>/mvtec_anomaly_detection.
#
# MVTec serves the dataset through its website (direct links are session-bound),
# so download it in a browser first:
#   https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
#   -> "mvtec_anomaly_detection.tar.xz" (about 4.9 GB, CC BY-NC-SA 4.0)
#
# Usage: bash scripts/setup_mvtec.sh path/to/mvtec_anomaly_detection.tar.xz [dest]
set -euo pipefail

ARCHIVE="${1:?Usage: bash scripts/setup_mvtec.sh path/to/mvtec_anomaly_detection.tar.xz [dest]}"
DEST="${2:-data}"
TARGET="$DEST/mvtec_anomaly_detection"

if [ -f "$TARGET/zipper/train/good/000.png" ]; then
  echo "MVTec AD already present at $TARGET"
  exit 0
fi
[ -f "$ARCHIVE" ] || { echo "Archive not found: $ARCHIVE" >&2; exit 1; }

mkdir -p "$TARGET"
echo "Extracting $ARCHIVE to $TARGET ..."
tar -xJf "$ARCHIVE" -C "$TARGET"

# Some copies wrap the categories in an extra top-level folder; flatten it.
if [ -d "$TARGET/mvtec_anomaly_detection" ]; then
  mv "$TARGET/mvtec_anomaly_detection"/* "$TARGET"/
  rmdir "$TARGET/mvtec_anomaly_detection"
fi

echo "Done. Categories:"
ls "$TARGET"
echo "Check with: python scripts/inspect_data.py --root $TARGET --out results/data"
