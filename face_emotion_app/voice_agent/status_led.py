"""Non-blocking status indication on the UNO Q's Linux-controlled RGB LED.

The first onboard RGB LED is exposed through sysfs and is independent of the
STM32-driven 8x13 matrix.  State changes therefore cost three tiny file writes,
need no MCU sketch, and cannot steal time from capture or playback.
"""
from __future__ import annotations

import threading
import time
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

    def __init__(self, root="/sys/class/leds", refresh_interval=0):
        root = Path(root)
        self._paths = tuple(root / f"{channel}:user" / "brightness"
                            for channel in ("red", "green", "blue"))
        self._available = all(path.exists() for path in self._paths)
        self._state = None
        self._lock = threading.Lock()
        self._warned = False
        self._refresh_interval = max(0.0, float(refresh_interval))
        self._refresh_thread = None

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
        if not self._available:
            return False
        with self._lock:
            if state == self._state:
                return True
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
        try:
            for path, value in zip(self._paths, COLORS[state]):
                path.write_text(str(value))
        except OSError as exc:
            if not self._warned:
                print(f"[status-led] unavailable: {exc}", flush=True)
                self._warned = True
            self._available = False
            return False
        return True

    def _refresh(self):
        """Reassert state because board services may also touch the user LED."""
        while self._available:
            time.sleep(self._refresh_interval)
            with self._lock:
                if self._state is not None and not self._write(self._state):
                    return


# The board's own management stack can rewrite the LED after boot or an ADB
# reconnect. Reasserting once every two seconds keeps the indicator truthful.
status_led = StatusLED(refresh_interval=2)
