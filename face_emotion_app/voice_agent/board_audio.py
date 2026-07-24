"""Board-native, half-duplex USB audio loop for the UNO Q.

The browser remains a useful remote UI, but a deployed UNO Q should not require
one just to hear and answer a person.  This loop waits for a non-board ALSA
capture device (the webcam microphone) and playback device (the USB speaker),
does lightweight local VAD, then runs one serialized agent turn and plays its
WAV response.  Capture is paused while speaking to prevent speaker feedback.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
import threading
import wave
from collections import deque
from pathlib import Path

import numpy as np


_CARD = re.compile(r"^card (\d+): (.*?) \[.*?\], device (\d+):", re.MULTILINE)


def _devices(command: str):
    """Return USB-ish ALSA hw devices, excluding the UNO Q's internal codec."""
    try:
        text = subprocess.run([command, "-l"], capture_output=True, text=True,
                              timeout=4, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [(f"plughw:{card},{device}", name) for card, name, device in _CARD.findall(text)
            if "ArduinoImola" not in name]


class BoardAudioLoop:
    RATE = 16000
    CHUNK_SAMPLES = 320             # 20 ms at 16 kHz
    MIN_SPEECH_SECONDS = 0.28
    ENDPOINT_SECONDS = 0.30
    MAX_UTTERANCE_SECONDS = 12.0
    CONFIG_FAULT_BACKOFF = 30.0

    def __init__(self, agent):
        self.agent = agent
        self._stop = threading.Event()
        self._thread = None
        self.capture_device = None
        self.playback_device = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="uno-board-audio", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _discover(self):
        capture = _devices("arecord")
        playback = _devices("aplay")
        self.capture_device = capture[0][0] if capture else None
        self.playback_device = playback[0][0] if playback else None
        return bool(self.capture_device and self.playback_device)

    def _run(self):
        waiting = False
        ready_for = None
        while not self._stop.is_set():
            if not self._discover():
                if not waiting:
                    print("[board-audio] waiting for USB microphone and speaker")
                    waiting, ready_for = True, None
                self._stop.wait(2)
                continue
            waiting = False
            # Announce the devices the first time and after any change, so the log
            # always shows what the board is actually listening and speaking on.
            devices = (self.capture_device, self.playback_device)
            if devices != ready_for:
                print(f"[board-audio] ready: mic={devices[0]} speaker={devices[1]}")
                ready_for = devices
            try:
                self._listen_once()
            except KeyboardInterrupt:
                raise
            # SystemExit is a BaseException, and the agent raises it for CONFIG
            # faults (rejected key, exhausted token budget). It used to unwind
            # straight out of this thread: the board went permanently deaf with
            # nothing in the log to say why. Such a fault fails identically every
            # turn, so report it and back off instead of burning a turn per second.
            except SystemExit as exc:
                print(f"[board-audio] cannot answer: {exc}\n"
                      f"[board-audio] retrying in {self.CONFIG_FAULT_BACKOFF:.0f}s "
                      "-- fix the configuration and it resumes on its own")
                self._stop.wait(self.CONFIG_FAULT_BACKOFF)
            except Exception as exc:
                print(f"[board-audio] {type(exc).__name__}: {exc}")
                self._stop.wait(1)

    def _listen_once(self):
        proc = subprocess.Popen(
            ["arecord", "-D", self.capture_device, "-q", "-f", "S16_LE", "-r", str(self.RATE),
             "-c", "1", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        preroll = deque(maxlen=15)  # 300 ms: do not clip the first word
        frames, speaking = [], False
        speech_s, silence_s, noise = 0.0, 0.0, 0.004
        try:
            while not self._stop.is_set():
                raw = proc.stdout.read(self.CHUNK_SAMPLES * 2)
                if len(raw) != self.CHUNK_SAMPLES * 2:
                    raise RuntimeError("microphone stopped returning audio")
                rms = float(np.sqrt(np.mean(np.frombuffer(raw, dtype="<i2").astype(np.float32) ** 2)) / 32768.0)
                if not speaking:
                    noise = noise * 0.98 + min(rms, noise * 2.0) * 0.02
                threshold = max(0.012, noise * 3.0)
                preroll.append(raw)
                if rms >= threshold:
                    if not speaking:
                        speaking, frames = True, list(preroll)
                    else:
                        frames.append(raw)
                    speech_s += 0.02
                    silence_s = 0.0
                elif speaking:
                    frames.append(raw)
                    silence_s += 0.02
                if speaking and ((silence_s >= self.ENDPOINT_SECONDS and speech_s >= self.MIN_SPEECH_SECONDS)
                                 or speech_s >= self.MAX_UTTERANCE_SECONDS):
                    break
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        if frames and speech_s >= self.MIN_SPEECH_SECONDS and not self._stop.is_set():
            self._answer(b"".join(frames))

    def _answer(self, pcm):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(self.RATE)
                out.writeframes(pcm)
            with self.agent.turn_lock:
                result = self.agent.handle_audio(path)
            audio = base64.b64decode(result.get("audio_b64") or "")
            if audio:
                subprocess.run(["aplay", "-D", self.playback_device, "-q"], input=audio,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=60, check=False)
        finally:
            # missing_ok: cleanup must never mask the real error from the turn.
            Path(path).unlink(missing_ok=True)
