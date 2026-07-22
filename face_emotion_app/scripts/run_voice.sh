#!/usr/bin/env bash
# Launch the voice agent. Uses the voice venv from install_voice.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VOICE_VENV:-$ROOT/.venv-voice}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "run ./scripts/install_voice.sh first"; exit 1; }

if [ -z "${CEREBRAS_API_KEY:-}" ]; then
  echo "warning: CEREBRAS_API_KEY is not set — the UI will load and the camera will run,"
  echo "         but a spoken turn will 503 until you: export CEREBRAS_API_KEY=csk-..."
fi

cd "$ROOT"
exec "$PY" -m voice_agent.main "$@"
