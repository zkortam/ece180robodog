#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"

download() {
  local url="$1"
  local out="$2"
  if [ -f "$out" ]; then
    echo "exists: $out"
    return
  fi
  echo "download: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "curl or wget is required" >&2
    exit 1
  fi
}

download \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  "$MODELS/face_detection_yunet_2023mar.onnx"

download \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
  "$MODELS/face_recognition_sface_2021dec.onnx"

download \
  "https://huggingface.co/opencv/facial_expression_recognition/resolve/main/facial_expression_recognition_mobilefacenet_2022july.onnx" \
  "$MODELS/facial_expression_recognition_mobilefacenet_2022july.onnx"

download \
  "https://huggingface.co/opencv/facial_expression_recognition/resolve/main/facial_expression_recognition_mobilefacenet_2022july_int8.onnx" \
  "$MODELS/facial_expression_recognition_mobilefacenet_2022july_int8.onnx"

echo "models ready in $MODELS"
