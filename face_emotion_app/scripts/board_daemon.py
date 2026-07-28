#!/usr/bin/env python3
"""Keep the board reachable at http://127.0.0.1:8100, on macOS, Windows or Linux.

`adb forward` is host-side state. Unplugging the board destroys it and nothing in
the browser can recreate it, so reloading the page after replugging still fails.
This daemon owns that job: it re-establishes the tunnel and restarts the board
services whenever they are missing, and it recovers from the two failure modes
that silently look identical to an unplugged board:

  * a dead adb server, which never recovers on its own
  * a hung `adb` invocation, which blocks forever without a timeout

Every subprocess call here therefore has a timeout. That is the whole reason this
is Python and not a shell script: `timeout(1)` is not present on stock macOS and
not portable to Windows.

The API key is read from ~/.robodog/env and never from the repo, which is public.

    python scripts/board_daemon.py            # run in the foreground
    python scripts/board_daemon.py --once     # single reconcile pass, for tests
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REMOTE_DIR = os.environ.get("BOARD_DIR", "/home/arduino/Documents/ece180/face_emotion_app")
VOICE_PORT = int(os.environ.get("VOICE_PORT", "8100"))
ENROLL_PORT = int(os.environ.get("ENROLL_PORT", "8000"))
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "3"))
# `adb forward` connects to the DEVICE's loopback, so binding the board's services
# to 0.0.0.0 buys nothing and exposes unauthenticated biometric enrollment and
# camera access to everyone on the same network. Loopback by default; set
# BOARD_BIND=0.0.0.0 deliberately if you want to reach the UI from a phone.
BOARD_BIND = os.environ.get("BOARD_BIND", "127.0.0.1")

# adb occasionally wedges. Every call is bounded so one bad invocation cannot
# freeze the loop; a timeout is simply treated as "not reachable this tick".
ADB_TIMEOUT = float(os.environ.get("ADB_TIMEOUT", "10"))
START_TIMEOUT = float(os.environ.get("START_TIMEOUT", "25"))

CONFIG_DIR = Path.home() / ".robodog"
LOG_PATH = CONFIG_DIR / "autoconnect.log"
MAX_LOG_BYTES = 512_000

WINDOWS = platform.system() == "Windows"


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > MAX_LOG_BYTES:
            tail = LOG_PATH.read_text(errors="replace").splitlines()[-500:]
            LOG_PATH.write_text("\n".join(tail) + "\n")
        with LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass          # logging must never take the daemon down


def load_env() -> None:
    """Read KEY=VALUE lines from ~/.robodog/env into the environment."""
    env_file = CONFIG_DIR / "env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(errors="replace").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def find_adb() -> str | None:
    """Locate adb across platforms and common SDK layouts."""
    found = shutil.which("adb")
    if found:
        return found
    candidates = [
        Path(os.environ.get("ANDROID_HOME", "")) / "platform-tools" / "adb",
        Path(os.environ.get("ANDROID_SDK_ROOT", "")) / "platform-tools" / "adb",
        Path("/opt/homebrew/bin/adb"),
        Path("/usr/local/bin/adb"),
        Path.home() / "Library/Android/sdk/platform-tools/adb",
        Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe",
        Path("C:/platform-tools/adb.exe"),
    ]
    for path in candidates:
        exe = path.with_suffix(".exe") if WINDOWS and not path.suffix else path
        if exe and str(exe) != "." and exe.exists():
            return str(exe)
    return None


class Board:
    def __init__(self, adb: str):
        self.adb = adb
        self.online = None            # tri-state so the first tick always logs
        self.consecutive_failures = 0

    def _run(self, args, timeout=ADB_TIMEOUT):
        """Run adb with a hard timeout. Returns (ok, stdout); never raises."""
        try:
            proc = subprocess.run(
                [self.adb, *args], capture_output=True, text=True, timeout=timeout,
                # Keep console windows from flashing on Windows.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if WINDOWS else 0,
            )
            return proc.returncode == 0, proc.stdout
        except subprocess.TimeoutExpired:
            return False, "<timeout>"
        except OSError as exc:
            return False, f"<error: {exc}>"

    def device_online(self) -> bool:
        ok, out = self._run(["devices"])
        if not ok:
            return False
        for line in out.splitlines()[1:]:
            parts = line.split()
            # "unauthorized" and "offline" are NOT usable states.
            if len(parts) >= 2 and parts[1] == "device":
                return True
        return False

    def ensure_forward(self, port: int) -> None:
        ok, out = self._run(["forward", "--list"])
        if ok and f"tcp:{port} tcp:{port}" in out:
            return
        self._run(["forward", f"tcp:{port}", f"tcp:{port}"])
        log(f"forwarded localhost:{port} to the board")

    def remote_running(self, pattern: str) -> bool:
        # The shell running pgrep carries the pattern in its own command line, so a
        # plain pattern matches itself and always reports success. Bracketing the
        # first character makes the regex miss the pgrep process itself.
        bracketed = f"[{pattern[0]}]{pattern[1:]}"
        ok, out = self._run(["shell", f"pgrep -f '{bracketed}' >/dev/null 2>&1 && echo yes"])
        return ok and "yes" in out

    def tune_cpu(self) -> None:
        """Pin the board's cores to their top frequency, once per connection.

        The QRB2210 idles at a low clock and ramps only after a workload has
        already been running, so every short inference burst pays the ramp. STT
        and TTS are exactly that shape. `arduino` is in the docker group, so root
        is reachable without a password via a privileged container.
        """
        script = (
            "for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do "
            "[ -w \"$g\" ] && echo performance > \"$g\"; done 2>/dev/null; "
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null"
        )
        ok, out = self._run(["shell", script], timeout=ADB_TIMEOUT)
        if ok and "performance" in out:
            log("cpu governor: performance")
            return
        escalated = (
            "docker run --rm --privileged -u 0 --net=host --pid=host -v /:/host "
            "--entrypoint chroot ghcr.io/arduino/app-bricks/python-apps-base:0.5.0 "
            f"/host sh -c '{script}'"
        )
        ok, out = self._run(["shell", escalated], timeout=START_TIMEOUT)
        log(f"cpu governor: {'performance' if 'performance' in out else 'left as-is'}")

    def start_voice(self) -> None:
        log("starting the voice agent on the board")
        key = os.environ.get("CEREBRAS_API_KEY", "")
        cmd = (
            f"cd '{REMOTE_DIR}' && CEREBRAS_API_KEY='{key}' PYTHONUNBUFFERED=1 "
            f"VOICE_CPU_THREADS={os.environ.get('VOICE_CPU_THREADS', '4')} "
            f"VOICE_LEAD_CHUNK_MAX={os.environ.get('VOICE_LEAD_CHUNK_MAX', '32')} "
            f"VOICE_TTS={os.environ.get('VOICE_TTS', 'piper')} "
            f"nohup ./scripts/run_voice.sh --host {BOARD_BIND} --port {VOICE_PORT} "
            f"--browser-camera > /tmp/voice.log 2>&1 &"
        )
        self._run(["shell", cmd], timeout=START_TIMEOUT)

    def start_enroll(self) -> None:
        log("starting the enrollment server on the board")
        cmd = (
            f"cd '{REMOTE_DIR}' && PYTHONUNBUFFERED=1 nohup .venv-voice/bin/python "
            f"face_emotion.py web --host {BOARD_BIND} --port {ENROLL_PORT} "
            f"> /tmp/enroll.log 2>&1 &"
        )
        self._run(["shell", cmd], timeout=START_TIMEOUT)

    def reconcile(self) -> bool:
        """One pass: make reality match the desired state. Returns True if online."""
        if not self.device_online():
            if self.online is not False:
                log("board disconnected")
                self.online = False
            self.consecutive_failures += 1
            # A wedged adb server is indistinguishable from an unplugged board and
            # does not recover by itself. Restart it, but not on every tick.
            if self.consecutive_failures % 10 == 0:
                log("restarting the adb server")
                self._run(["kill-server"], timeout=ADB_TIMEOUT)
                self._run(["start-server"], timeout=ADB_TIMEOUT)
            return False

        if self.online is not True:
            log("board connected")
            self.online = True
            self.tune_cpu()          # once per connection, before anything is started
        self.consecutive_failures = 0

        self.ensure_forward(VOICE_PORT)
        self.ensure_forward(ENROLL_PORT)
        if not self.remote_running("voice_agent.main"):
            self.start_voice()
        if not self.remote_running("face_emotion.py web"):
            self.start_enroll()
        return True


def acquire_single_instance():
    """Return a held lock, or None if another daemon already owns it.

    launchd's bootstrap+kickstart, a systemd restart racing a manual run, or two
    logins can each start a copy. Duplicates both race to start the same board
    services, so exactly one must win. The lock is an open file handle: the OS
    drops it if the process dies, so a crash never leaves a stale lock behind.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(CONFIG_DIR / "daemon.lock", "w")
    try:
        if WINDOWS:
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single reconcile pass, then exit")
    args = ap.parse_args()

    load_env()

    lock = None
    if not args.once:
        lock = acquire_single_instance()
        if lock is None:
            log("another daemon is already running; exiting")
            return 0
    adb = find_adb()
    if not adb:
        log("adb not found. Install Android platform-tools and put adb on PATH.")
        return 1
    log(f"watching for the board using {adb}")

    board = Board(adb)
    if args.once:
        return 0 if board.reconcile() else 2
    while True:
        try:
            board.reconcile()
        except Exception as exc:                      # never let one bad tick kill it
            log(f"unexpected error, continuing: {type(exc).__name__}: {exc}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
