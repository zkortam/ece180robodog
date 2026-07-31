"""Non-blocking status indication on the UNO Q's Linux-controlled RGB LED.

The first onboard RGB LED is exposed through sysfs and is independent of the
STM32-driven 8x13 matrix.  State changes therefore cost three tiny file writes,
need no MCU sketch, and cannot steal time from capture or playback.
"""
from __future__ import annotations

import threading
import time
import shutil
import subprocess
from pathlib import Path


# One-bit RGB channels still provide seven clear colors.  Keep green reserved for
# the most important promise: the microphone is open and waiting for the user.
COLORS = {
    "off": (0, 0, 0),
    "starting": (0, 1, 1),   # cyan
    "waiting": (1, 1, 0),    # yellow: USB hardware is not ready
    "listening": (0, 1, 0),  # green
    "hearing": (1, 1, 1),    # white: speech has crossed VAD
    "thinking": (0, 0, 1),   # blue
    "speaking": (1, 0, 1),   # magenta
    "error": (1, 0, 0),      # red
}


class StatusLED:
    """Best-effort RGB state output; absent LEDs never affect the voice service."""

    def __init__(self, root="/sys/class/leds", refresh_interval=0,
                 matrix_command=None):
        root = Path(root)
        self._paths = tuple(root / f"{channel}:user" / "brightness"
                            for channel in ("red", "green", "blue"))
        self._available = all(path.exists() for path in self._paths)
        self._state = None
        self._lock = threading.Lock()
        self._warned = False
        self._refresh_interval = max(0.0, float(refresh_interval))
        self._refresh_thread = None
        self._matrix_command = matrix_command
        self._matrix_warned = False

    @property
    def available(self):
        return self._available

    @property
    def state(self):
        return self._state

    def set(self, state):
        """Display *state* synchronously, returning False only if unavailable."""
        if state not in COLORS:
            raise ValueError(f"unknown LED state {state!r}")
        with self._lock:
            if state == self._state:
                return True
            # The UNO Q matrix is a separate Bridge device.  Some board images
            # do not expose the Linux RGB sysfs LED at all; that must not prevent
            # the large matrix from receiving state updates.
            if not self._write(state):
                return False
            self._state = state
            print(f"[status-led] {state}", flush=True)
            if self._refresh_interval and self._refresh_thread is None:
                self._refresh_thread = threading.Thread(
                    target=self._refresh, name="uno-status-led", daemon=True)
                self._refresh_thread.start()
            return True

    def _write(self, state):
        rgb_ok = True
        if self._available:
            try:
                for path, value in zip(self._paths, COLORS[state]):
                    path.write_text(str(value))
            except OSError as exc:
                if not self._warned:
                    print(f"[status-led] unavailable: {exc}", flush=True)
                    self._warned = True
                self._available = False
                rgb_ok = False
        else:
            rgb_ok = False
        matrix_ok = self._write_matrix(state)
        return rgb_ok or matrix_ok

    def _write_matrix(self, state):
        if not self._matrix_command:
            return False
        try:
            result = subprocess.run(
                [self._matrix_command, "set_robodog_status", state],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=1, check=False)
        except (OSError, subprocess.SubprocessError):
            result = None
        ok = bool(result and result.returncode == 0)
        if not ok and not self._matrix_warned:
            print("[status-led] matrix bridge not ready; retrying", flush=True)
            self._matrix_warned = True
        elif ok:
            self._matrix_warned = False
        return ok

    def _refresh(self):
        """Reassert state because board services may also touch the user LED."""
        while self._available:
            time.sleep(self._refresh_interval)
            with self._lock:
                if self._state is not None and not self._write(self._state):
                    return


# The board's own management stack can rewrite the LED after boot or an ADB
# reconnect. Reasserting once every two seconds keeps the indicator truthful.
def _matrix_cli():
    """Find the Bridge CLI even when systemd gives the service a minimal PATH."""
    configured = __import__("os").environ.get("ARDUINO_ROUTER_CLI", "").strip()
    candidates = [configured, shutil.which("arduino-router-cli"),
                  "/usr/bin/arduino-router-cli", "/usr/local/bin/arduino-router-cli"]
    return next((p for p in candidates if p and Path(p).is_file()), None)


status_led = StatusLED(refresh_interval=2, matrix_command=_matrix_cli())
