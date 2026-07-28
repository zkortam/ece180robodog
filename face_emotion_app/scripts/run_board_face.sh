#!/usr/bin/env bash
# Run the face/emotion recognizer entirely on the UNO Q.
# A USB camera may receive any /dev/video index, so probe a small range and
# then keep recognition attached to the first device that returns a frame.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${FACE_EMOTION_PYTHON:-python3}"

# The voice agent owns the camera in standalone mode. Two processes opening the
# same /dev/video* node do not share it -- one gets the frames and the other
# silently gets nothing, so the robot goes half-blind in a way that looks like a
# hardware fault. Refuse to compete instead.
if pgrep -f '[v]oice_agent.main' >/dev/null 2>&1; then
  echo "The voice agent is already running and owns the camera." >&2
  echo "This CLI recognizer and robodog.service are mutually exclusive." >&2
  exit 1
fi

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
