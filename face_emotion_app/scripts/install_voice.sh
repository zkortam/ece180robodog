#!/usr/bin/env bash
# Install the voice-agent runtime. Works on macOS (dev) and ARM Debian (the UNO Q).
# Creates/uses a venv, installs Python deps, and fetches TTS/STT models per platform.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OS="$(uname -s)"
VENV="${VOICE_VENV:-$ROOT/.venv-voice}"
PY="$VENV/bin/python"

echo "== voice-agent install ($OS) =="

# 1) Python env (prefer uv, fall back to venv)
if [ ! -x "$PY" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

pipi() { if command -v uv >/dev/null 2>&1; then uv pip install --python "$PY" "$@"; else "$PY" -m pip install "$@"; fi; }

# 2) Base + voice deps
pipi numpy opencv-python flask openai

MODELS="$ROOT/models"; mkdir -p "$MODELS"
fetch_kokoro() {
  base="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
  [ -f "$MODELS/kokoro-v1.0.onnx" ] || curl -L "$base/kokoro-v1.0.onnx" -o "$MODELS/kokoro-v1.0.onnx"
  [ -f "$MODELS/voices-v1.0.bin" ] || curl -L "$base/voices-v1.0.bin" -o "$MODELS/voices-v1.0.bin"
}

if [ "$OS" = "Darwin" ]; then
  echo "-- macOS dev: STT=faster-whisper, TTS=kokoro (natural)"
  pipi faster-whisper kokoro-onnx
  fetch_kokoro
else
  echo "-- ARM/Linux board: STT=moonshine, TTS=piper"
  pipi useful-moonshine-onnx onnxruntime || echo "WARN: install Moonshine manually if this fails"
  # Piper: download a prebuilt binary + a low voice (adjust arch/voice as needed)
  MODELS="$ROOT/models"; mkdir -p "$MODELS"
  if [ ! -f "$MODELS/en_US-lessac-low.onnx" ]; then
    echo "-- fetching Piper voice (en_US-lessac-low)"
    base="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low"
    curl -L "$base/en_US-lessac-low.onnx"      -o "$MODELS/en_US-lessac-low.onnx"
    curl -L "$base/en_US-lessac-low.onnx.json" -o "$MODELS/en_US-lessac-low.onnx.json"
  fi
  echo "-- install the Piper binary: see https://github.com/rhasspy/piper/releases"
  echo "   (and 'sudo apt install -y espeak-ng' as the always-works TTS fallback)"
fi

# 3) Face/emotion models (reuses the existing downloader)
[ -x "$ROOT/scripts/download_models.sh" ] && "$ROOT/scripts/download_models.sh" || true

echo
echo "Done. Next:"
echo "  export CEREBRAS_API_KEY=csk-...      # your key; keep it out of the repo"
echo "  $PY scripts/smoke_cerebras.py         # Phase 0: prove the LLM tool loop"
echo "  ./scripts/run_voice.sh                # start the voice agent"
