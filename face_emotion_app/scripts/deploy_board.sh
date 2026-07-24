#!/usr/bin/env bash
# Deploy and run the voice agent ON the UNO Q, viewed from a browser on this Mac.
#
# Compute (STT, LLM call, TTS, vision) runs on the board. The laptop browser only
# supplies the camera and microphone and plays the reply.
#
# This goes over ADB on purpose. Campus and hotspot Wi-Fi isolate clients from
# each other, so the board is unreachable by IP even when both are "online";
# `adb forward` tunnels over USB and is unaffected by that.
#
#   ./scripts/deploy_board.sh              # push, install if needed, run
#   ./scripts/deploy_board.sh --reinstall  # force dependency reinstall
#
# Requires CEREBRAS_API_KEY in the environment. Never hardcode it: this repo is
# public. BOARD_SUDO_PASSWORD is only needed the first time, for apt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${BOARD_DIR:-/home/arduino/Documents/ece180/face_emotion_app}"
PORT="${VOICE_PORT:-8100}"
REINSTALL=0
[ "${1:-}" = "--reinstall" ] && REINSTALL=1

say() { printf '\n== %s\n' "$*"; }

if [ -z "${CEREBRAS_API_KEY:-}" ]; then
  echo "CEREBRAS_API_KEY is not set. export it first (the board needs it to answer)." >&2
  exit 1
fi

command -v adb >/dev/null 2>&1 || { echo "adb not found. brew install android-platform-tools" >&2; exit 1; }

say "waiting for the board over USB"
echo "   If this hangs: the cable may be charge-only, or the board is still booting."
adb wait-for-device
adb shell 'echo "   connected: $(hostname) $(uname -m)"'

# ---- push source (not the venv, not git, not the Mac-only Kokoro weights) ----
say "pushing source"
TAR="$(mktemp -t robodog).tar.gz"
trap 'rm -f "$TAR"' EXIT
tar czf "$TAR" -C "$ROOT" \
  --exclude='.venv*' --exclude='__pycache__' --exclude='.git' \
  --exclude='kokoro-v1.0.onnx' --exclude='voices-v1.0.bin' \
  --exclude='*.bak-*' \
  face_emotion.py vision_service.py voice_agent scripts requirements.txt models data 2>/dev/null || true
adb shell "mkdir -p '$REMOTE_DIR'"
adb push "$TAR" /tmp/robodog.tar.gz >/dev/null
adb shell "tar xzf /tmp/robodog.tar.gz -C '$REMOTE_DIR' && rm -f /tmp/robodog.tar.gz"
echo "   pushed to $REMOTE_DIR"

# ---- dependencies ----
NEED_INSTALL=$REINSTALL
if [ "$NEED_INSTALL" = 0 ]; then
  adb shell "[ -x '$REMOTE_DIR/.venv-voice/bin/python' ]" || NEED_INSTALL=1
fi

if [ "$NEED_INSTALL" = 1 ]; then
  say "installing board dependencies (slow the first time)"
  adb shell "cd '$REMOTE_DIR' && bash scripts/install_voice.sh" || {
    echo "install_voice.sh failed. Run it by hand over 'adb shell' to see why." >&2; exit 1; }

  # espeak-ng is the TTS that always works; Piper is preferred but needs a binary.
  if [ -n "${BOARD_SUDO_PASSWORD:-}" ]; then
    say "installing espeak-ng (TTS fallback)"
    adb shell "echo '$BOARD_SUDO_PASSWORD' | sudo -S apt-get install -y espeak-ng" >/dev/null 2>&1 \
      && echo "   espeak-ng ready" || echo "   WARN: apt failed; Piper alone must work"
  else
    echo "   skipping espeak-ng (set BOARD_SUDO_PASSWORD to install the TTS fallback)"
  fi
fi

# ---- run ----
say "starting the agent on the board"
adb shell "pkill -f voice_agent.main" >/dev/null 2>&1 || true
# Four A53 cores oversubscribe exactly like the Mac did, so cap the pools.
adb shell "cd '$REMOTE_DIR' && \
  CEREBRAS_API_KEY='$CEREBRAS_API_KEY' PYTHONUNBUFFERED=1 VOICE_CPU_THREADS=\${VOICE_CPU_THREADS:-2} \
  nohup ./scripts/run_voice.sh --host 0.0.0.0 --port $PORT --browser-camera \
  > /tmp/voice.log 2>&1 &" >/dev/null 2>&1 || true

say "forwarding localhost:$PORT to the board"
adb forward --remove tcp:$PORT >/dev/null 2>&1 || true
adb forward tcp:$PORT tcp:$PORT >/dev/null

for i in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null || true)"
  [ "$code" = "200" ] && { echo "   up after ${i}s"; break; }
  sleep 1
done

if [ "${code:-}" != "200" ]; then
  echo "   the board did not serve a page. Last log lines:" >&2
  adb shell "tail -25 /tmp/voice.log" >&2
  exit 1
fi

say "ready"
echo "   open http://127.0.0.1:$PORT   (served BY the board, over USB)"
echo "   board log:  adb shell tail -f /tmp/voice.log"
echo "   stop:       adb shell pkill -f voice_agent.main"
