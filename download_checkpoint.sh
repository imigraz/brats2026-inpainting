#!/bin/bash
set -e

# Configuration
FILE_ID="1WdeLqhAZBmT5Vycyy63rfBuVYKWbvA3P"
OUT_PATH="models/model-10k.pt"

# Verify gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Error: 'gdown' is required to download from Google Drive." >&2
    echo "Install it via: pip install gdown" >&2
    exit 1
fi

# Create models directory if it doesn't exist
mkdir -p "$(dirname "$OUT_PATH")"

echo "Downloading pretrained checkpoint to: $OUT_PATH"
gdown "$FILE_ID" -O "$OUT_PATH"
echo "Download complete."