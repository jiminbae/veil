#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/models/project/face_tracking/weights"
MODEL_PATH="$WEIGHTS_DIR/yolo26x-face.pt"
MODEL_URL="https://github.com/jiminbae/yolo26x-face/releases/download/v0.1.0/best.pt"
LIVEPORTRAIT_DIR="$ROOT_DIR/models/project/LivePortrait"
LIVEPORTRAIT_WEIGHTS_DIR="$LIVEPORTRAIT_DIR/pretrained_weights"
LIVEPORTRAIT_REQUIRED="$LIVEPORTRAIT_WEIGHTS_DIR/liveportrait/base_models/appearance_feature_extractor.pth"

mkdir -p "$WEIGHTS_DIR"

if [ -s "$MODEL_PATH" ]; then
  echo "Already exists: $MODEL_PATH"
else
  echo "Downloading YOLO face weights..."
  echo "Source: $MODEL_URL"
  echo "Target: $MODEL_PATH"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar "$MODEL_URL" -o "$MODEL_PATH"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$MODEL_PATH" "$MODEL_URL"
  else
    echo "Error: curl or wget is required to download YOLO weights." >&2
    exit 1
  fi

  echo "Done: $MODEL_PATH"
fi

if [ ! -d "$LIVEPORTRAIT_DIR" ]; then
  echo "Error: LivePortrait submodule is missing: $LIVEPORTRAIT_DIR" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

mkdir -p "$LIVEPORTRAIT_WEIGHTS_DIR"

if [ -s "$LIVEPORTRAIT_REQUIRED" ]; then
  echo "Already exists: $LIVEPORTRAIT_REQUIRED"
else
  echo "Downloading LivePortrait pretrained weights..."
  echo "Target: $LIVEPORTRAIT_WEIGHTS_DIR"

  if command -v hf >/dev/null 2>&1; then
    hf download KlingTeam/LivePortrait \
      --local-dir "$LIVEPORTRAIT_WEIGHTS_DIR" \
      --exclude "*.git*" \
      --exclude "README.md" \
      --exclude "docs/*"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download KlingTeam/LivePortrait \
      --local-dir "$LIVEPORTRAIT_WEIGHTS_DIR" \
      --exclude "*.git*" \
      --exclude "README.md" \
      --exclude "docs/*"
  else
    python - <<'PY' "$LIVEPORTRAIT_WEIGHTS_DIR"
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="KlingTeam/LivePortrait",
    local_dir=sys.argv[1],
    ignore_patterns=["*.git*", "README.md", "docs/*"],
)
PY
  fi

  echo "Done: $LIVEPORTRAIT_WEIGHTS_DIR"
fi
