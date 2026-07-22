#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Give Flask a moment to bind before opening the enrollment page.
(sleep 2; open http://127.0.0.1:8000) &

cd "$ROOT"
exec ./scripts/run_local.sh
