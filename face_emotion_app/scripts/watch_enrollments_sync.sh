#!/usr/bin/env bash
# Copy compact enrollment prototypes to the UNO Q whenever the trainer updates
# them. Images and camera frames never leave the enrollment machine.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOARD_HOST="${FACE_BOARD_HOST:-zk-unoq-01.local}"
BOARD_USER="${FACE_BOARD_USER:-arduino}"
BOARD_APP_DIR="${FACE_BOARD_APP_DIR:-/home/arduino/Documents/ece180/face_emotion_app}"
INTERVAL="${FACE_SYNC_INTERVAL:-2}"
TARGET="${BOARD_USER}@${BOARD_HOST}"

case "$BOARD_HOST:$BOARD_USER:$BOARD_APP_DIR" in
  *[!A-Za-z0-9._/@:-]*)
    echo "Invalid board sync destination" >&2
    exit 2
    ;;
esac

last_fingerprint=""
while true; do
  fingerprint="$({ shasum -a 256 "$ROOT/data/enrollments.json" "$ROOT/data/emotions.json" 2>/dev/null || true; } | shasum -a 256 | awk '{print $1}')"
  if [ -n "$fingerprint" ] && [ "$fingerprint" != "$last_fingerprint" ]; then
    synced=0
    if ssh -o BatchMode=yes -o ConnectTimeout=4 "$TARGET" "mkdir -p '$BOARD_APP_DIR/data'"; then
      ok=1
      for name in enrollments.json emotions.json; do
        local_file="$ROOT/data/$name"
        remote_tmp="$BOARD_APP_DIR/data/.$name.upload"
        if [ -f "$local_file" ]; then
          scp -q -o BatchMode=yes -o ConnectTimeout=4 "$local_file" "$TARGET:$remote_tmp" || ok=0
        fi
      done
      if [ "$ok" -eq 1 ] && ssh -o BatchMode=yes -o ConnectTimeout=4 "$TARGET" \
        "mv '$BOARD_APP_DIR/data/.enrollments.json.upload' '$BOARD_APP_DIR/data/enrollments.json' && mv '$BOARD_APP_DIR/data/.emotions.json.upload' '$BOARD_APP_DIR/data/emotions.json'"; then
        synced=1
        echo "Enrollment embeddings synced to $BOARD_HOST over Wi-Fi"
      fi
    fi

    # USB is the reliable development fallback when the board is not on the same
    # Wi-Fi network or mDNS is unavailable.
    if [ "$synced" -eq 0 ] && command -v adb >/dev/null 2>&1 &&
      [ "$(adb get-state 2>/dev/null || true)" = "device" ]; then
      ok=1
      adb shell "mkdir -p '$BOARD_APP_DIR/data'" >/dev/null || ok=0
      for name in enrollments.json emotions.json; do
        local_file="$ROOT/data/$name"
        remote_tmp="$BOARD_APP_DIR/data/.$name.upload"
        if [ -f "$local_file" ]; then
          adb push "$local_file" "$remote_tmp" >/dev/null || ok=0
        fi
      done
      if [ "$ok" -eq 1 ] && adb shell \
        "mv '$BOARD_APP_DIR/data/.enrollments.json.upload' '$BOARD_APP_DIR/data/enrollments.json' && mv '$BOARD_APP_DIR/data/.emotions.json.upload' '$BOARD_APP_DIR/data/emotions.json'"; then
        synced=1
        echo "Enrollment embeddings synced to the UNO Q over USB"
        # The deployed voice service loads its databases at startup. Restart it
        # after an enrollment so the AI/LLM can use the new identity immediately.
        adb shell "pkill -f '^python3 -m voice_agent.main' || true; sleep 1; cd '$BOARD_APP_DIR' && setsid /home/arduino/run_board.sh --port 8100 >>/home/arduino/voice-agent.log 2>&1 </dev/null &" >/dev/null 2>&1 || true
      fi
    fi

    if [ "$synced" -eq 1 ]; then
      last_fingerprint="$fingerprint"
    else
      echo "Board unavailable; enrollment sync will retry" >&2
    fi
  fi
  sleep "$INTERVAL"
done
