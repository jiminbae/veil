#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/models/veil/weights"
MODEL_PATH="$WEIGHTS_DIR/yolo26x-face.pt"
MODEL_URL="https://github.com/jiminbae/yolo26x-face/releases/download/v0.1.0/best.pt"
INSWAPPER_PATH="$WEIGHTS_DIR/inswapper_128.onnx"
INSWAPPER_URL="https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"

mkdir -p "$WEIGHTS_DIR"

download_file() {
  local name="$1"
  local url="$2"
  local target="$3"

  if [ -s "$target" ]; then
    echo "Already exists: $target"
    return
  fi

  echo "Downloading $name..."
  echo "Source: $url"
  echo "Target: $target"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar "$url" -o "$target"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$target" "$url"
  else
    echo "Error: curl or wget is required to download weights." >&2
    exit 1
  fi

  echo "Done: $target"
}

download_file "YOLO face weights" "$MODEL_URL" "$MODEL_PATH"
download_file "InSwapper weights" "$INSWAPPER_URL" "$INSWAPPER_PATH"
