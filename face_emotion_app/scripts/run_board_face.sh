#!/usr/bin/env bash
# Run the face/emotion recognizer entirely on the UNO Q.
# A USB camera may receive any /dev/video index, so probe a small range and
# then keep recognition attached to the first device that returns a frame.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${FACE_EMOTION_PYTHON:-python3}"

while true; do
  if [ ! -s "$ROOT/data/enrollments.json" ] || [ "$(tr -d '[:space:]' < "$ROOT/data/enrollments.json")" = "{}" ]; then
    echo "Waiting for a face enrollment in $ROOT/data/enrollments.json"
    sleep 10
    continue
  fi

  for camera in 0 1 2 3 4 5 6 7 8; do
    if timeout 8 "$PYTHON" "$ROOT/face_emotion.py" --camera "$camera" camera-test; then
      echo "Using camera index $camera"
      "$PYTHON" "$ROOT/face_emotion.py" --camera "$camera" recognize --headless
      echo "Recognizer stopped; retrying camera discovery"
    fi
  done
  sleep 3
done
