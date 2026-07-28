"""Board-native, half-duplex USB audio loop for the UNO Q.

The browser remains a useful remote UI, but a deployed UNO Q should not require
one just to hear and answer a person.  This loop waits for a non-board ALSA
capture device (the webcam microphone) and playback device (the USB speaker),
does lightweight local VAD, then runs one serialized agent turn and plays its
WAV response.  Capture is paused while speaking to prevent speaker feedback.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from . import config
from .tts import sentence_chunks


def _wav_pcm(wav_bytes):
    """(raw PCM, (rate, channels, sample_width)) from in-memory WAV bytes.

    The header is read rather than assumed: Kokoro emits 24 kHz, Piper's rate
    depends on the voice, and `say`/espeak differ again. Handing aplay the wrong
    rate does not fail -- it plays the reply at the wrong speed and pitch.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return (w.readframes(w.getnframes()),
                (w.getframerate(), w.getnchannels(), w.getsampwidth()))


_CARD = re.compile(r"^card (\d+): (.*?) \[(.*?)\], device (\d+):", re.MULTILINE)

# The board's own codec has no microphone or speaker wired to it. Selecting it is
# the worst failure mode available: everything "works", the robot simply never
# hears anything and never says anything, with nothing in the log. Excluded by
# name -- and overridable, because the name is image- and revision-specific and a
# hardcoded string that stops matching would silently pick the dead codec again.
_EXCLUDED = tuple(x.strip().lower() for x in
                  os.environ.get("VOICE_EXCLUDE_AUDIO", "ArduinoImola,HDMI,Loopback,Dummy")
                  .split(",") if x.strip())

# Everything hangs off one USB hub, so `arecord -l` may legitimately list two
# capture devices (a webcam's mic AND a headset) and `aplay -l` two playback
# devices. Picking index 0 is a coin flip that changes with enumeration order --
# i.e. with which port you happened to use and how fast each device powered up.
# Score by what the device calls itself instead, and let the name be pinned.
_CAPTURE_HINTS = ("webcam", "camera", "cam", "uvc", "mic", "headset", "usb audio", "usb")
_PLAYBACK_HINTS = ("speaker", "headset", "dac", "usb audio", "audio", "usb")


def list_devices(command: str):
    """Every selectable ALSA device `command` reports, in enumeration order.

    Returns dicts so callers can log the alternatives. `arecord -l` lists only
    capture-capable devices and `aplay -l` only playback-capable ones, so the two
    lists are already role-filtered by ALSA.
    """
    try:
        text = subprocess.run([command, "-l"], capture_output=True, text=True,
                              timeout=4, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    found = []
    for card, name, short, index in _CARD.findall(text):
        label = f"{name} [{short}]"
        if any(bad in label.lower() for bad in _EXCLUDED):
            continue
        # plughw, not hw: it converts rate/format in software, so a device that
        # cannot do 16 kHz mono natively still works instead of failing to open.
        found.append({"device": f"plughw:{card},{index}", "name": name, "label": label})
    return found


def pick_device(candidates, kind, override=None, match=None):
    """Choose one device by name, not by luck of enumeration order.

    `override` is an exact ALSA device string and wins outright -- the escape
    hatch for a setup no heuristic gets right. `match` is a case-insensitive
    substring of the device name, which is the stable way to pin "the webcam's
    mic" across reboots and re-plugs where card numbers move.
    """
    if override:
        return {"device": override, "name": "(pinned)", "label": f"{override} (pinned)"}
    if match:
        wanted = match.lower()
        for c in candidates:
            if wanted in c["label"].lower():
                return c
        return None                      # asked for something specific; do not guess
    if not candidates:
        return None
    hints = _CAPTURE_HINTS if kind == "capture" else _PLAYBACK_HINTS

    def score(c):
        low = c["label"].lower()
        for rank, hint in enumerate(hints):
            if hint in low:
                return rank              # earlier hint = better match
        return len(hints)

    return min(candidates, key=score)


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
        self._capture_label = None
        self._playback_label = None
        self._alternatives = {"capture": [], "playback": []}
        # Dead-microphone detection; see _check_for_dead_microphone().
        self._silent_chunks = 0
        self._heard_anything = False
        self._warned_dead_mic = False
        # Repeated-capture-failure collapsing; see the handler in _run().
        self._last_error = None
        self._repeated_errors = 0
        self._warned_capture_broken = False

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
        capture = list_devices("arecord")
        playback = list_devices("aplay")
        mic = pick_device(capture, "capture",
                          os.environ.get("VOICE_CAPTURE_DEVICE"),
                          os.environ.get("VOICE_CAPTURE_MATCH"))
        speaker = pick_device(playback, "playback",
                             os.environ.get("VOICE_PLAYBACK_DEVICE"),
                             os.environ.get("VOICE_PLAYBACK_MATCH"))
        self.capture_device = mic["device"] if mic else None
        self.playback_device = speaker["device"] if speaker else None
        self._capture_label = mic["label"] if mic else None
        self._playback_label = speaker["label"] if speaker else None
        # Print the road not taken. When the robot ends up on the wrong device,
        # this line is the difference between a two-minute fix (set
        # VOICE_CAPTURE_MATCH) and an afternoon of guessing.
        self._alternatives = {
            "capture": [c["label"] for c in capture],
            "playback": [c["label"] for c in playback],
        }
        return bool(self.capture_device and self.playback_device)

    def _run(self):
        waiting = False
        ready_for = None
        announced_fault = False
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
                print(f"[board-audio] ready: mic={devices[0]} ({self._capture_label})")
                print(f"[board-audio]        speaker={devices[1]} ({self._playback_label})")
                for kind, labels in self._alternatives.items():
                    if len(labels) > 1:
                        print(f"[board-audio] other {kind} devices seen: {'; '.join(labels)}"
                              f"\n[board-audio]   -> pin one with VOICE_{kind.upper()}_MATCH="
                              "'<part of its name>' if the wrong one was chosen")
                ready_for = devices
                self._silent_chunks = 0
                self._heard_anything = False
                self._warned_dead_mic = False
            try:
                self._listen_once()
                # A clean pass means every fault cleared; re-arm the announcements
                # so a fault that recurs after a real recovery is reported again.
                announced_fault = False
                self._last_error, self._repeated_errors = None, 0
                self._warned_capture_broken = False
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
                # A robot that goes quiet is indistinguishable from a robot that is
                # broken, unplugged, or ignoring you. There is no screen out here, so
                # the only way to report a fault is to say it. Once per fault, not
                # once per retry -- an appliance repeating an error forever is worse
                # than one that said it clearly a single time.
                if not announced_fault:
                    self._speak("I can't reach my language service right now. "
                                "Check my configuration.")
                    announced_fault = True
                self._stop.wait(self.CONFIG_FAULT_BACKOFF)
            except Exception as exc:
                # A capture device that fails REPEATEDLY is a hardware fault, not a
                # blip: a hub that keeps browning out, a port that is dying, a mic
                # that was pulled. Left alone this loop spins forever printing the
                # same line and the robot is simply deaf -- no error anyone hears,
                # and a log that scrolls too fast to read. Collapse the repeats,
                # back off, and eventually say it out loud.
                signature = f"{type(exc).__name__}: {exc}"
                if signature == self._last_error:
                    self._repeated_errors += 1
                else:
                    self._last_error, self._repeated_errors = signature, 1
                    print(f"[board-audio] {signature}")
                if self._repeated_errors in (5, 50, 500):
                    print(f"[board-audio] {signature} (x{self._repeated_errors}) -- "
                          "check the USB hub has power and the microphone is seated",
                          flush=True)
                if self._repeated_errors == 5 and not self._warned_capture_broken:
                    self._warned_capture_broken = True
                    self._speak("Something is wrong with my microphone. "
                                "Check that it's plugged in.")
                # Back off progressively so a persistent fault costs almost nothing,
                # while a one-off blip still recovers in a second.
                self._stop.wait(1.0 if self._repeated_errors < 5 else 5.0)

    # ~45 s of 20 ms chunks. Long enough that a quiet room or a pause between
    # sentences never trips it, short enough to catch the mistake while you are
    # still standing in front of the robot wondering why it is ignoring you.
    DEAD_MIC_CHUNKS = 2250

    def _check_for_dead_microphone(self, samples):
        """Notice when the chosen capture device is not connected to anything.

        This is the one hardware fault with no symptom. A wrong-but-working device
        -- the board's own codec with no mic wired to it, or a hub port that lost
        power -- opens fine, streams happily, and returns perfect digital silence
        forever. The VAD never triggers, so the robot never speaks, never errors,
        and logs nothing: indistinguishable from being ignored.

        BIT-EXACT zero is the tell. A real microphone always has a noise floor;
        even in a silent room its samples jitter around zero. Sustained exact
        zeros mean nothing is wired to the input, so say so out loud.
        """
        if np.any(samples):
            self._silent_chunks = 0
            self._heard_anything = True
            return
        self._silent_chunks += 1
        if self._silent_chunks < self.DEAD_MIC_CHUNKS or self._warned_dead_mic:
            return
        self._warned_dead_mic = True
        seconds = self._silent_chunks * self.CHUNK_SAMPLES / self.RATE
        alternatives = self._alternatives.get("capture", [])
        print(f"[board-audio] WARNING: {self.capture_device} "
              f"({self._capture_label}) has produced {seconds:.0f}s of bit-exact "
              "silence -- not a quiet room, an input with nothing wired to it.",
              flush=True)
        if len(alternatives) > 1:
            print("[board-audio]   other capture devices: " + "; ".join(alternatives)
                  + "\n[board-audio]   pin the right one with VOICE_CAPTURE_MATCH="
                  "'<part of its name>'", flush=True)
        else:
            print("[board-audio]   check the USB hub has power and the mic is seated.",
                  flush=True)
        # Only worth saying aloud if we have never heard anything at all: that is
        # the misconfigured case. A mic that worked and then went quiet is more
        # likely unplugged, and talking to an empty room does not help.
        if not self._heard_anything:
            self._speak("I can't hear anything from my microphone. "
                        "Check that it's plugged in.")

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
                samples = np.frombuffer(raw, dtype="<i2")
                self._check_for_dead_microphone(samples)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0)
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
            # BOUNDED, like the HTTP path. `with turn_lock:` waits forever, so a
            # wedged web turn (stalled TTS subprocess, hung upload) used to make the
            # robot permanently deaf with no way to recover short of a restart --
            # the exact failure web.turn_slot() was written to prevent.
            if not self.agent.turn_lock.acquire(timeout=config.TURN_LOCK_TIMEOUT):
                print("[board-audio] previous turn is still running; dropping this one")
                self._speak("Sorry, I'm still finishing the last one.")
                return
            try:
                # Throttle perception while this turn runs: on four small cores,
                # full-rate detection competes with the STT and TTS the person is
                # actually waiting for.
                with self.agent.vision.turn_in_progress():
                    # understand_audio, not handle_audio: it stops before synthesis
                    # so we can speak the reply one sentence at a time instead of
                    # waiting for all of it. Same STT, LLM, tools and history.
                    result = self.agent.understand_audio(path)
                    # Speaking happens INSIDE the turn lock, so the lock is held for
                    # as long as the robot is talking. That is deliberate: a web
                    # client starting a turn while the robot is mid-sentence would
                    # talk over itself. It gets a clean 409 instead.
                    spoke, chunk1_ms = self._speak_reply(result.get("reply", ""))
            finally:
                self.agent.turn_lock.release()
            if spoke:
                # ONE honest line, with the same meaning as the streaming HTTP
                # endpoint's: `first_audio` is the whole silence the person sat
                # through, not just the synthesis part of it. Reporting the latter
                # under that name is how a turn looks several times more responsive
                # than it is.
                timings = result.get("timings_ms", {})
                first_audio = (timings.get("stt", 0.0) + timings.get("llm", 0.0)
                               + (chunk1_ms or 0.0))
                print(f"[board-audio] turn: stt={timings.get('stt', 0):.0f}ms "
                      f"llm={timings.get('llm', 0):.0f}ms "
                      f"chunk1={chunk1_ms or 0:.0f}ms -> first audio "
                      f"{first_audio:.0f}ms", flush=True)
            if not spoke and result.get("transcript"):
                # Heard, understood, but synthesis produced nothing. Silence here
                # reads as "it ignored me"; say so with the fallback voice instead.
                self._speak("Sorry, my voice failed on that one.")
        finally:
            # missing_ok: cleanup must never mask the real error from the turn.
            Path(path).unlink(missing_ok=True)

    def _speak_reply(self, reply):
        """Speak the reply, starting as soon as the FIRST sentence is synthesized.

        Synthesis runs at roughly real time, so waiting for a whole two-sentence
        reply before opening your mouth doubles the silence the person sits
        through. The browser transport already streams sentence chunks; this gives
        the standalone robot -- the deployment that has no browser -- the same
        behaviour.

        Everything is fed to ONE aplay process through a pipe, which matters twice:
        playback is gapless (a new aplay per sentence would click and pause between
        them), and writing blocks once its buffer is full, so synthesis is paced by
        playback for free. Chunk N+1 is therefore being made while chunk N plays,
        with no threads to get wrong.

        Returns (spoke_anything, milliseconds_to_synthesize_the_first_chunk).
        """
        chunks = sentence_chunks(reply)
        if not chunks:
            return False, None
        proc = None
        stream_format = None
        spoke = False
        first_chunk_ms = None
        started = time.perf_counter()
        try:
            for text in chunks:
                wav = self.agent.tts.synth(text)
                if not wav:
                    continue
                try:
                    pcm, fmt = _wav_pcm(wav)
                except (wave.Error, EOFError) as exc:
                    print(f"[board-audio] unplayable synthesis for {text!r}: {exc}")
                    continue
                if proc is None:
                    proc = self._open_playback(fmt)
                    if proc is None:
                        return False, None
                    stream_format = fmt
                    first_chunk_ms = (time.perf_counter() - started) * 1000
                elif fmt != stream_format:
                    # A backend that changed rate mid-reply (a fallback voice
                    # kicking in) cannot share the open stream. Finish this one and
                    # play the rest as its own stream rather than emitting noise.
                    self._close_playback(proc)
                    proc = self._open_playback(fmt)
                    if proc is None:
                        return spoke, first_chunk_ms
                    stream_format = fmt
                try:
                    proc.stdin.write(pcm)
                except (BrokenPipeError, OSError) as exc:
                    print(f"[board-audio] playback stopped early: {exc}")
                    return spoke, first_chunk_ms
                spoke = True
            return spoke, first_chunk_ms
        finally:
            if proc is not None:
                self._close_playback(proc)

    def _open_playback(self, fmt):
        """Start an aplay reading raw PCM from stdin, or None if it will not run."""
        rate, channels, width = fmt
        if not self.playback_device:
            return None
        encoding = {1: "U8", 2: "S16_LE", 3: "S24_3LE", 4: "S32_LE"}.get(width)
        if encoding is None:
            print(f"[board-audio] unsupported sample width {width}; cannot stream")
            return None
        try:
            return subprocess.Popen(
                ["aplay", "-D", self.playback_device, "-q", "-t", "raw",
                 "-f", encoding, "-r", str(rate), "-c", str(channels)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[board-audio] could not start playback: {exc}")
            return None

    @staticmethod
    def _close_playback(proc):
        """Drain and reap. Best effort: playback must never break the listen loop."""
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

    def _play(self, wav_bytes):
        """Play WAV bytes on the USB speaker. Never raises: playback failing must
        not kill the listen loop that is the robot's only input."""
        try:
            subprocess.run(["aplay", "-D", self.playback_device, "-q"], input=wav_bytes,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=60, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[board-audio] playback failed: {type(exc).__name__}: {exc}")

    def _speak(self, message):
        """Say a short status line out loud, best effort.

        Used only for faults, where the normal reply path is what failed. Every
        step is guarded: if TTS itself is broken there is nothing further to try,
        and raising here would take down the listen loop as well."""
        if not self.playback_device:
            return
        try:
            wav = self.agent.tts.synth(message)
        except Exception as exc:
            print(f"[board-audio] could not synthesize status line: {exc}")
            return
        if wav:
            self._play(wav)
