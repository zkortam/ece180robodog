#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON=/private/tmp/ece180-face-uv/bin/python

# Reuse the project's voice environment when it already has the web/vision
# dependencies. This avoids rebuilding a second environment just to enroll faces.
if [ -x "$ROOT/.venv-voice/bin/python" ] &&
  "$ROOT/.venv-voice/bin/python" -c 'import cv2, flask, numpy' >/dev/null 2>&1; then
  DEFAULT_PYTHON="$ROOT/.venv-voice/bin/python"
fi

PYTHON="${FACE_EMOTION_PYTHON:-$DEFAULT_PYTHON}"

if [ -x "$ROOT/scripts/download_models.sh" ]; then
  "$ROOT/scripts/download_models.sh"
fi

if [ ! -x "$PYTHON" ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to create the local runtime" >&2
    exit 1
  fi
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/ece180-uv-cache}"
  uv venv /private/tmp/ece180-face-uv --clear
  uv pip install --python /private/tmp/ece180-face-uv/bin/python -r "$ROOT/requirements.txt"
  PYTHON=/private/tmp/ece180-face-uv/bin/python
fi

SYNC_PID=""
if [ "${FACE_SYNC_TO_BOARD:-1}" = "1" ]; then
  "$ROOT/scripts/watch_enrollments_sync.sh" &
  SYNC_PID=$!
  trap 'kill "$SYNC_PID" 2>/dev/null || true' EXIT INT TERM
fi

"$PYTHON" "$ROOT/face_emotion.py" web --host 127.0.0.1 --port 8000
