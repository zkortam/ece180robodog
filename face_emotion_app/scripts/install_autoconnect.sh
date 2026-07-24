#!/usr/bin/env bash
# Install board_daemon.py so the tunnel is restored automatically at login and
# after every replug, with no terminal open. macOS uses launchd, Linux systemd.
# Windows users run install_autoconnect.ps1 instead.
#
#   ./scripts/install_autoconnect.sh
#   ./scripts/install_autoconnect.sh --uninstall
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.robodog.autoconnect"
CONFIG_DIR="$HOME/.robodog"
RUNNER="$CONFIG_DIR/board_daemon.py"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UNIT="$HOME/.config/systemd/user/robodog-autoconnect.service"
PYTHON="$(command -v python3 || command -v python)"

uninstall() {
  if [ "$(uname -s)" = "Darwin" ]; then
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
  else
    systemctl --user disable --now robodog-autoconnect.service 2>/dev/null || true
    rm -f "$UNIT"
  fi
  echo "uninstalled $LABEL"
}

[ "${1:-}" = "--uninstall" ] && { uninstall; exit 0; }

if [ ! -f "$CONFIG_DIR/env" ]; then
  echo "Create $CONFIG_DIR/env first (chmod 600) containing:" >&2
  echo "  CEREBRAS_API_KEY=csk-..." >&2
  exit 1
fi
[ -n "$PYTHON" ] || { echo "python3 not found on PATH" >&2; exit 1; }

# macOS TCC denies launchd agents access to ~/Documents, ~/Desktop and ~/Downloads
# unless the user grants Full Disk Access, so the daemon runs from a copy outside
# those folders. It talks to the board over adb and never reads the repo at runtime.
mkdir -p "$CONFIG_DIR"
cp "$ROOT/scripts/board_daemon.py" "$RUNNER"
chmod +x "$RUNNER"

if [ "$(uname -s)" = "Darwin" ]; then
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$PYTHON</string><string>$RUNNER</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/dev/null</string>
  <key>StandardErrorPath</key><string>$CONFIG_DIR/autoconnect.err</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
</dict>
</plist>
PLIST_EOF
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
else
  mkdir -p "$(dirname "$UNIT")"
  cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=RoboDog board autoconnect
[Service]
ExecStart=$PYTHON $RUNNER
Restart=always
RestartSec=3
[Install]
WantedBy=default.target
UNIT_EOF
  systemctl --user daemon-reload
  systemctl --user enable --now robodog-autoconnect.service
fi

echo "installed $LABEL"
echo "  log:       tail -f $CONFIG_DIR/autoconnect.log"
echo "  uninstall: ./scripts/install_autoconnect.sh --uninstall"
