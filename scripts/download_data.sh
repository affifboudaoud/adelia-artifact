#!/bin/bash
# Download and extract the artifact data from Google Drive.
#
# Usage:
#   ./scripts/download_data.sh
#
# Requires: gdown (pip install gdown)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(dirname "$SCRIPT_DIR")"
FILE_ID="1kJF1wHH4BOHS50T7kaLec2trcvb_JNxR"

if [ -d "$ARTIFACT_DIR/data" ]; then
    echo "data/ directory already exists. Remove it first to re-download."
    exit 0
fi

if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Install it with: pip install gdown"
    exit 1
fi

echo "Downloading data archive from Google Drive..."
gdown "$FILE_ID" -O "$ARTIFACT_DIR/data.tar.gz"

echo "Extracting..."
tar -xzf "$ARTIFACT_DIR/data.tar.gz" -C "$ARTIFACT_DIR"

rm "$ARTIFACT_DIR/data.tar.gz"
echo "Done. Data extracted to $ARTIFACT_DIR/data/"
