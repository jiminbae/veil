#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${VEIL_WEIGHTS_DIR:-$ROOT_DIR/models/veil/weights}"
MODEL_PATH="$WEIGHTS_DIR/yolo26x-face.pt"
MODEL_URL="${VEIL_YOLO_URL:-https://github.com/jiminbae/yolo26x-face/releases/download/v0.1.0/best.pt}"
MODEL_SHA="${VEIL_YOLO_SHA256:-f749791ce9205e2df8bbb479201bce09b43423cac7f8e84d19cae6a01d0cea22}"
INSWAPPER_PATH="$WEIGHTS_DIR/inswapper_128.onnx"
INSWAPPER_URL="${VEIL_INSWAPPER_URL:-https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx}"
INSWAPPER_SHA="${VEIL_INSWAPPER_SHA256:-e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af}"

mkdir -p "$WEIGHTS_DIR"

download_file() {
  local name="$1"
  local url="$2"
  local target="$3"
  local expected="$4"

  if [ -f "$target" ] && printf '%s  %s\n' "$expected" "$target" | sha256sum --check --status; then
    echo "Already verified: $target"
    return
  fi
  if [ -e "$target" ]; then
    local preserved
    preserved="$(mktemp "$target.invalid.XXXXXX")"
    mv -- "$target" "$preserved"
    echo "Preserved invalid/incomplete download: $preserved" >&2
  fi
  local partial
  partial="$(mktemp "$target.part.XXXXXX")"

  echo "Downloading $name..."
  echo "Source: $url"
  echo "Target: $target"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 2 --connect-timeout 15 --max-time 600 --progress-bar "$url" -o "$partial"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --timeout=30 -O "$partial" "$url"
  else
    echo "Error: curl or wget is required to download weights." >&2
    exit 1
  fi
  if ! printf '%s  %s\n' "$expected" "$partial" | sha256sum --check --status; then
    echo "Checksum mismatch; incomplete file retained at: $partial" >&2
    return 1
  fi
  mv -- "$partial" "$target"

  echo "Done: $target"
}

download_file "YOLO face weights" "$MODEL_URL" "$MODEL_PATH" "$MODEL_SHA"
download_file "InSwapper weights" "$INSWAPPER_URL" "$INSWAPPER_PATH" "$INSWAPPER_SHA"
