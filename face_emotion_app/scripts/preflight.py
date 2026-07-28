#!/usr/bin/env python3
"""Check everything the robot needs to run standalone, and say what is missing.

Run this ON THE BOARD, with the camera, microphone and speaker plugged in, BEFORE
you unplug the laptop. Every check maps to a failure that is invisible until the
robot is on its own: no key, no network, no capture device, no camera, a model
that was never downloaded, a read-only data directory.

    python scripts/preflight.py             # report and exit non-zero on failure
    python scripts/preflight.py --speak     # also say the verdict out loud

Exit status is 0 only when the robot can actually hold a conversation unattended.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""

FAIL, WARN, OK = "fail", "warn", "ok"
results = []


def record(status, label, detail=""):
    results.append((status, label, detail))
    mark = {OK: f"{GREEN}  ok  {RESET}", WARN: f"{YELLOW} warn {RESET}",
            FAIL: f"{RED} FAIL {RESET}"}[status]
    print(f"[{mark}] {label}" + (f"\n         {DIM}{detail}{RESET}" if detail else ""))


# --------------------------------------------------------------------- checks

def check_models():
    needed = {
        "face detector": ROOT / "models" / "face_detection_yunet_2023mar.onnx",
        "face recognizer": ROOT / "models" / "face_recognition_sface_2021dec.onnx",
    }
    optional = {"emotion model":
                ROOT / "models" / "facial_expression_recognition_mobilefacenet_2022july.onnx"}
    for label, path in needed.items():
        if path.exists():
            record(OK, f"{label} present")
        else:
            record(FAIL, f"{label} MISSING", f"{path}\n         run scripts/download_models.sh")
    for label, path in optional.items():
        record(OK if path.exists() else WARN,
               f"{label} present" if path.exists() else f"{label} missing",
               "" if path.exists() else "expressions will be unavailable; identity still works")


def check_python_deps():
    for module, why in (("cv2", "vision"), ("numpy", "everything"), ("flask", "the UI"),
                        ("openai", "the Cerebras client")):
        try:
            __import__(module)
            record(OK, f"python: {module}")
        except Exception as exc:
            record(FAIL, f"python: {module} missing ({why})", f"{exc}\n         run scripts/install_voice.sh")


def check_stt_tts():
    try:
        from voice_agent import config
    except Exception as exc:
        record(FAIL, "voice_agent config", str(exc))
        return
    record(OK, f"STT backend: {config.STT_BACKEND}", f"model {config.STT_MODEL}")
    if config.STT_BACKEND == "moonshine":
        try:
            __import__("moonshine_onnx")
            record(OK, "moonshine installed")
        except Exception:
            record(FAIL, "moonshine NOT installed",
                   "pip install useful-moonshine-onnx  (the board's STT)")
    record(OK, f"TTS backend: {config.TTS_BACKEND}")
    if config.TTS_BACKEND == "piper":
        voice = Path(config.PIPER_VOICE)
        if voice.exists():
            record(OK, "piper voice present")
        else:
            record(FAIL, "piper voice MISSING", f"{voice}\n         run scripts/install_voice.sh")
    if not (shutil.which("espeak-ng") or shutil.which("espeak")):
        record(WARN, "no espeak-ng fallback",
               "apt install espeak-ng -- the last-resort voice if piper fails")


def check_audio_devices():
    """A robot with no microphone cannot be talked to, and it cannot tell you so.

    This does not just list devices -- it records from the one that would actually
    be chosen and checks the samples are not bit-exact zero. Selecting a device
    with nothing wired to it (the board's own codec, or a hub port that lost power)
    opens fine and streams perfect silence forever, which is the one hardware fault
    that produces no symptom at all.
    """
    if not shutil.which("arecord"):
        record(WARN, "arecord not found (not a Linux board?)",
               "board audio is Linux/ALSA only; skipping device checks")
        return
    try:
        from voice_agent.board_audio import list_devices, pick_device
    except Exception as exc:
        record(FAIL, "board audio module", str(exc))
        return

    capture, playback = list_devices("arecord"), list_devices("aplay")
    mic = pick_device(capture, "capture", os.environ.get("VOICE_CAPTURE_DEVICE"),
                      os.environ.get("VOICE_CAPTURE_MATCH"))
    speaker = pick_device(playback, "playback", os.environ.get("VOICE_PLAYBACK_DEVICE"),
                          os.environ.get("VOICE_PLAYBACK_MATCH"))

    if not mic:
        record(FAIL, "NO microphone found",
               "plug in the USB mic/webcam; `arecord -l` should list it")
    else:
        others = [c["label"] for c in capture if c["device"] != mic["device"]]
        record(OK, f"microphone: {mic['device']}", mic["label"]
               + (f"\n         not chosen: {'; '.join(others)}"
                  "\n         (pin with VOICE_CAPTURE_MATCH if that is wrong)" if others else ""))
        _check_microphone_is_live(mic)

    if not speaker:
        record(FAIL, "NO speaker found",
               "plug in the USB speaker; `aplay -l` should list it")
    else:
        others = [c["label"] for c in playback if c["device"] != speaker["device"]]
        record(OK, f"speaker: {speaker['device']}", speaker["label"]
               + (f"\n         not chosen: {'; '.join(others)}"
                  "\n         (pin with VOICE_PLAYBACK_MATCH if that is wrong)" if others else ""))


def _check_microphone_is_live(mic):
    """Record a second and confirm the input is connected to something real."""
    try:
        raw = subprocess.run(
            ["arecord", "-D", mic["device"], "-q", "-f", "S16_LE", "-r", "16000",
             "-c", "1", "-t", "raw", "-d", "1"],
            capture_output=True, timeout=8, check=False).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        record(FAIL, "could not record from the microphone", str(exc))
        return
    if len(raw) < 2000:
        record(FAIL, "microphone returned almost no audio",
               f"got {len(raw)} bytes in 1s from {mic['device']}")
        return
    if not any(raw):
        # A real microphone always has a noise floor; even a silent room jitters.
        record(FAIL, "microphone is BIT-EXACT SILENT",
               "nothing is wired to this input. Check the USB hub has power and is\n"
               "         seated, or pin the right device with VOICE_CAPTURE_MATCH.")
        return
    import struct
    n = len(raw) // 2
    peak = max(abs(v) for v in struct.unpack(f"<{n}h", raw[:n * 2]))
    record(OK, "microphone is live", f"peak {peak}/32767 over 1s of room noise")


def check_camera():
    nodes = sorted(glob("/dev/video*"))
    if not nodes:
        record(FAIL if sys.platform.startswith("linux") else WARN,
               "no /dev/video* nodes", "plug in the USB webcam")
        return
    try:
        import cv2
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
        from vision_service import capture_device_indexes
    except Exception as exc:
        record(FAIL, "cannot import cv2 to test the camera", str(exc))
        return
    # On this SoC most /dev/video* nodes are the hardware encoder/decoder, not
    # cameras. Ask the kernel which are real capture devices rather than opening
    # each one to find out (opening an m2m node can block).
    candidates = capture_device_indexes()
    if not candidates:
        record(FAIL, "no /dev/video* node is a capture device",
               f"saw {', '.join(nodes)} -- all encoder/decoder nodes.\n"
               "         plug in the USB webcam (must be UVC class)")
        return
    record(OK, f"{len(candidates)} capture node(s) found",
           ", ".join(f"/dev/video{i} ({n or 'unnamed'})" for i, n in candidates))
    for index, name in candidates:
        cap = cv2.VideoCapture(index)
        ok = cap.isOpened() and cap.read()[0]
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if ok else 0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if ok else 0
        cap.release()
        if ok:
            record(OK, f"camera delivers frames: /dev/video{index}",
                   f"{name or 'unnamed'} at {width:.0f}x{height:.0f}")
            return
    record(FAIL, "no capture node actually delivers a frame",
           "the node exists but returns nothing -- is the USB hub powered?")


def check_key_and_network():
    if os.environ.get("CEREBRAS_API_KEY"):
        record(OK, "CEREBRAS_API_KEY is set")
    else:
        env_file = Path.home() / ".robodog" / "env"
        if env_file.exists() and "CEREBRAS_API_KEY" in env_file.read_text():
            record(OK, f"CEREBRAS_API_KEY in {env_file}", "systemd reads it from there")
        else:
            record(FAIL, "CEREBRAS_API_KEY not set",
                   f"put it in {env_file} (chmod 600) for the service to find it")
    # A board with no battery-backed clock boots in 1970. Every TLS certificate is
    # then "not yet valid" and the failure looks exactly like a dead network, which
    # sends you to debug Wi-Fi when the real problem is the date.
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    if now.year < 2025:
        record(FAIL, f"system clock is wrong ({now:%Y-%m-%d})",
               "TLS will reject every certificate as not-yet-valid.\n"
               "         fix NTP, or the robot cannot reach any HTTPS API")
    else:
        record(OK, f"system clock is plausible ({now:%Y-%m-%d %H:%M} UTC)")

    # The robot needs its OWN internet once the laptop is gone. This is the single
    # most common reason a board that worked tethered goes mute on its own.
    # A full TLS handshake, not just a TCP connect: it proves DNS, routing, the
    # proxy-free path AND the certificate chain (so it also catches the clock).
    try:
        import ssl
        with socket.create_connection(("api.cerebras.ai", 443), timeout=8) as raw, \
                ssl.create_default_context().wrap_socket(
                    raw, server_hostname="api.cerebras.ai"):
            record(OK, "reached api.cerebras.ai over TLS")
    except ssl.SSLCertVerificationError as exc:
        record(FAIL, "TLS certificate rejected",
               f"{exc}\n         almost always the system clock, not the network")
    except OSError as exc:
        record(FAIL, "CANNOT reach api.cerebras.ai",
               f"{exc}\n         the board needs its own Wi-Fi once the laptop is unplugged.\n"
               "         check: nmcli connection show --active")


def check_storage():
    data = ROOT / "data"
    try:
        data.mkdir(parents=True, exist_ok=True)
        probe = data / ".preflight"
        probe.write_text("ok")
        probe.unlink()
        record(OK, "data/ is writable")
    except OSError as exc:
        record(FAIL, "data/ is NOT writable", f"{exc}\n         enrollments cannot be saved")
    db = data / "enrollments.json"
    if db.exists() and db.stat().st_size > 2:
        try:
            import face_emotion as fe
            names = sorted(fe.load_db(db))
            record(OK, f"{len(names)} face(s) enrolled", ", ".join(names) or "-")
        except Exception as exc:
            record(FAIL, "enrollments.json is unreadable", str(exc))
    else:
        record(WARN, "no faces enrolled yet",
               "the robot will hear and answer, but recognize nobody")


def check_service():
    if not shutil.which("systemctl"):
        return
    for unit, want in (("robodog.service", True), ("uno-face-emotion.service", False)):
        out = subprocess.run(["systemctl", "is-enabled", unit],
                             capture_output=True, text=True, check=False).stdout.strip()
        enabled = out == "enabled"
        if want and enabled:
            record(OK, f"{unit} is enabled", "it will start on boot")
        elif want:
            record(WARN, f"{unit} is not enabled ({out or 'absent'})",
                   "sudo systemctl enable --now robodog.service")
        elif enabled:
            record(FAIL, f"{unit} is enabled and CONFLICTS with robodog.service",
                   "it opens the camera exclusively: sudo systemctl disable --now " + unit)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speak", action="store_true", help="say the verdict out loud")
    args = ap.parse_args()

    print(f"\n{DIM}RoboDog preflight -- can this board hold a conversation unattended?{RESET}\n")
    for section, fn in (("models", check_models), ("python", check_python_deps),
                        ("speech", check_stt_tts), ("audio devices", check_audio_devices),
                        ("camera", check_camera), ("key + network", check_key_and_network),
                        ("storage", check_storage), ("service", check_service)):
        print(f"{DIM}-- {section}{RESET}")
        try:
            fn()
        except Exception as exc:              # a broken check must not hide the rest
            record(FAIL, f"{section} check crashed", f"{type(exc).__name__}: {exc}")
        print()

    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]
    if failures:
        verdict = f"{RED}NOT READY{RESET}: {len(failures)} blocking problem(s)"
        spoken = f"Preflight failed. {len(failures)} problems. I am not ready to run on my own."
    elif warnings:
        verdict = f"{GREEN}READY{RESET} with {len(warnings)} warning(s)"
        spoken = "Preflight passed with warnings. I can run on my own."
    else:
        verdict = f"{GREEN}READY{RESET} -- everything the robot needs is present"
        spoken = "Preflight passed. I am ready."
    print(verdict)
    for _, label, _ in failures:
        print(f"  {RED}x{RESET} {label}")

    if args.speak:
        try:
            from voice_agent.tts import TTS
            wav = TTS().synth(spoken)
            if wav and shutil.which("aplay"):
                subprocess.run(["aplay", "-q"], input=wav, timeout=30, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            print(f"{DIM}(could not speak the verdict: {exc}){RESET}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
