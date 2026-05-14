#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/models/project/face_tracking/weights"
MODEL_PATH="$WEIGHTS_DIR/yolo26x-face.pt"
MODEL_URL="https://github.com/jiminbae/yolo26x-face/releases/download/v0.1.0/best.pt"

mkdir -p "$WEIGHTS_DIR"

if [ -s "$MODEL_PATH" ]; then
  echo "Already exists: $MODEL_PATH"
  exit 0
fi

echo "Downloading YOLO face weights..."
echo "Source: $MODEL_URL"
echo "Target: $MODEL_PATH"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar "$MODEL_URL" -o "$MODEL_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$MODEL_PATH" "$MODEL_URL"
else
  echo "Error: curl or wget is required to download weights." >&2
  exit 1
fi

echo "Done: $MODEL_PATH"
