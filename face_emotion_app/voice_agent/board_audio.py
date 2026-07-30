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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import config
from .tts import sentence_chunks


# Target average level as a fraction of full scale. RMS, not peak, is what the ear
# reads as loudness. Piper already delivers ~0.18, which is a normal speech level,
# so there is far less headroom here than it looks -- see the limiter below.
TARGET_RMS = float(os.environ.get("VOICE_OUTPUT_TARGET_RMS", "0.30"))
# Where the limiter starts bending the signal. Everything below this stays exactly
# linear, so ordinary speech is untouched and only the plosive peaks are shaped.
LIMIT_KNEE = float(os.environ.get("VOICE_OUTPUT_KNEE", "0.75"))
# The level the synthesizer actually delivers, measured on this board. Used so the
# gain is CONSTANT across every sentence of a reply instead of recomputed per
# buffer, which made the volume pump audibly between sentences.
REFERENCE_RMS = float(os.environ.get("VOICE_OUTPUT_REFERENCE_RMS", "0.185"))


def _amplify(pcm_bytes, target_rms=None, knee=None):
    """Raise speech to a consistent level for a small speaker, WITHOUT distorting.

    An earlier version multiplied hard and ran the result through tanh. Measured on
    the board that saturated 39% of all samples -- a square wave. Clipping does not
    make a speaker louder, it makes it buzz, which is exactly how it sounded.

    So: normalize the average level to TARGET_RMS, then apply a soft-knee limiter
    that leaves everything below the knee perfectly linear and only rounds off the
    peaks above it. Loudness comes from the normalization; the limiter exists purely
    to stop the peaks clipping.

    The honest ceiling: Piper already outputs ~0.18 RMS, so there is maybe 4-5 dB of
    clean headroom here. Beyond that the limit is the speaker, not the signal.
    """
    target = TARGET_RMS if target_rms is None else target_rms
    knee = LIMIT_KNEE if knee is None else knee
    if not pcm_bytes or target <= 0:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms < 1e-6:
        return pcm_bytes                      # silence: amplifying it only adds hiss
    # A FIXED gain against the synthesizer's known level, not per-buffer
    # normalization. Each sentence of a reply is synthesized separately, so
    # normalizing each one to the same RMS made a quiet sentence loud and a loud
    # one quiet -- the volume audibly pumped between sentences within a single
    # answer. A constant gain keeps the whole reply at one level, and the limiter
    # below still catches anything unusually hot.
    gain = target / REFERENCE_RMS
    # Only rescue material far outside the expected range, and even then gently.
    if rms < REFERENCE_RMS * 0.4 or rms > REFERENCE_RMS * 2.5:
        gain = min(target / rms, gain * 2.0)
    samples = samples * gain
    # Soft knee: linear below `knee`, smoothly compressed above it toward 1.0.
    magnitude = np.abs(samples)
    over = magnitude > knee
    if np.any(over):
        headroom = max(1.0 - knee, 1e-6)
        excess = (magnitude[over] - knee) / headroom
        magnitude[over] = knee + headroom * np.tanh(excess)
        samples = np.sign(samples) * magnitude
    np.clip(samples, -1.0, 1.0, out=samples)
    return (samples * 32767.0).astype("<i2").tobytes()


# Fraction of the mixer's range to set the speaker to. 100% by default: a USB
# speaker's ALSA volume resets to its factory default whenever the device
# re-enumerates or the driver reloads, and on this hardware that default is 40%
# -- about 16 dB down, which no amount of software gain can honestly recover.
# Asserting it on every discovery pass is cheap and idempotent.
PLAYBACK_VOLUME = os.environ.get("VOICE_PLAYBACK_VOLUME", "100%")


def set_playback_volume(device, level=None):
    """Turn playback controls up without touching mic gain or sidetone.

    A USB headset can expose capture and playback controls on the same ALSA card.
    Blindly setting every simple control also turns up Mic/Sidetone, feeding the
    speaker into the microphone and destabilizing VAD after every reply.
    """
    level = PLAYBACK_VOLUME if level is None else level
    card = device.split(":", 1)[-1].split(",", 1)[0]
    if not card.isdigit():
        return False
    applied = []
    try:
        listing = subprocess.run(["amixer", "-c", card, "scontrols"],
                                 capture_output=True, text=True, timeout=4, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in listing.splitlines():
        # "Simple mixer control 'PCM',0"
        if "'" not in line:
            continue
        name = line.split("'")[1]
        if any(word in name.lower() for word in
               ("mic", "capture", "input", "sidetone", "auto gain")):
            continue
        try:
            detail = subprocess.run(["amixer", "-c", card, "sget", name],
                                    capture_output=True, text=True, timeout=4,
                                    check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        low = detail.stdout.lower()
        if ("playback channels:" not in low
                and "pvolume" not in low
                and "pswitch" not in low):
            continue
        try:
            done = subprocess.run(["amixer", "-c", card, "sset", name, level, "unmute"],
                                  capture_output=True, text=True, timeout=4, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            applied.append(name)
    if applied:
        print(f"[board-audio] speaker volume set to {level} on card {card} "
              f"({', '.join(applied)})", flush=True)
    return bool(applied)


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
    CHUNK_SECONDS = CHUNK_SAMPLES / RATE
    MIN_SPEECH_SECONDS = float(os.environ.get("VOICE_MIN_SPEECH_SECONDS", "0.18"))
    ENDPOINT_SECONDS = config.VAD_ENDPOINT_MS / 1000.0
    MAX_UTTERANCE_SECONDS = 8.0
    CONFIG_FAULT_BACKOFF = 30.0
    # ---- voice activity detection ----
    # Re-opened ALSA streams and the tail of the preceding reply need a short
    # settling window. It is deliberately shorter than the preroll, so someone who
    # starts speaking immediately is still captured rather than losing the opener.
    CALIBRATION_CHUNKS = int(os.environ.get("VOICE_CALIBRATION_CHUNKS", "10"))
    PREROLL_CHUNKS = max(25, CALIBRATION_CHUNKS + 5)  # at least 500 ms
    # This webcam has extremely little separation between room and speech. A
    # multiplicative threshold cannot work at that boundary: 1.25 * 0.0143 is
    # already above ordinary speech. Use a small additive margin, with a modest
    # relative component for microphones with a healthier signal.
    ONSET_MARGIN = float(os.environ.get("VOICE_ONSET_MARGIN", "0.0010"))
    ONSET_RELATIVE = float(os.environ.get("VOICE_ONSET_RELATIVE", "0.07"))
    RELEASE_MARGIN = float(os.environ.get("VOICE_RELEASE_MARGIN", "0.0006"))
    RELEASE_RELATIVE = float(os.environ.get("VOICE_RELEASE_RELATIVE", "0.035"))
    # A few isolated level spikes are not an utterance. Require three of the last
    # five 20 ms frames to cross onset before entering RECORDING.
    ONSET_WINDOW_CHUNKS = int(os.environ.get("VOICE_ONSET_WINDOW_CHUNKS", "5"))
    ONSET_REQUIRED_CHUNKS = int(os.environ.get("VOICE_ONSET_REQUIRED_CHUNKS", "3"))
    # A completed utterance also needs a real peak above the onset threshold. This
    # prevents a fan/room fluctuation hovering on the boundary from being sent to
    # Moonshine, where silence can be hallucinated as plausible text.
    MIN_PEAK_MARGIN = float(os.environ.get("VOICE_MIN_PEAK_MARGIN", "0.0015"))
    MIN_DYNAMIC_RANGE = float(os.environ.get("VOICE_MIN_DYNAMIC_RANGE", "0.0015"))
    FLAT_BASELINE_RANGE = float(os.environ.get("VOICE_FLAT_BASELINE_RANGE", "0.0012"))
    # Used only to recognize speech that starts inside the very first calibration
    # window, before any room floor exists. Ongoing turns use the learned floor.
    STARTUP_ONSET = float(os.environ.get("VOICE_STARTUP_ONSET", "0.017"))
    MIN_NOISE_FLOOR = 0.002
    MIN_ONSET = float(os.environ.get("VOICE_MIN_ONSET", "0.012"))

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
        # Learned ambient level, carried across turns. Starting from scratch each
        # turn is what made the VAD latch onto room noise; see _listen_once().
        self.noise_floor = self.MIN_NOISE_FLOOR

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
        if speaker:
            set_playback_volume(speaker["device"])
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
        preroll = deque(maxlen=self.PREROLL_CHUNKS)
        frames, speaking = [], False
        speech_s, silence_s, elapsed_s, peak_rms = 0.0, 0.0, 0.0, 0.0
        threshold = self.MIN_ONSET
        trigger_threshold = self.MIN_ONSET
        release = self.MIN_NOISE_FLOOR
        endpoint_reason = None
        onset_votes = deque(maxlen=self.ONSET_WINDOW_CHUNKS)
        onset_levels = deque(maxlen=self.ONSET_WINDOW_CHUNKS)
        recording_levels = deque(maxlen=400)
        recent_recording = deque(maxlen=100)
        calibration_speech_s = 0.0
        calibration_trigger = None
        starting_uncalibrated = self.noise_floor <= self.MIN_NOISE_FLOOR
        # Rolling window of recent quiet-room levels. A low percentile follows the
        # baseline without being dragged upward by speech, coughs, or doors.
        ambient = deque(maxlen=300)
        calibration = []
        try:
            while not self._stop.is_set():
                raw = proc.stdout.read(self.CHUNK_SAMPLES * 2)
                if len(raw) != self.CHUNK_SAMPLES * 2:
                    raise RuntimeError("microphone stopped returning audio")
                samples = np.frombuffer(raw, dtype="<i2")
                self._check_for_dead_microphone(samples)
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0)
                preroll.append(raw)

                # Always settle after playback/re-opening capture. The previous code
                # pre-filled this list after turn one, accidentally bypassing its own
                # calibration and allowing reply tail/stream startup noise to trigger.
                if len(calibration) < self.CALIBRATION_CHUNKS:
                    calibration.append(rms)
                    if len(calibration) == self.CALIBRATION_CHUNKS:
                        measured = float(np.percentile(calibration, 25))
                        # Blend across turns. A single contaminated settling window
                        # must neither deafen the robot nor make it trigger forever.
                        if self.noise_floor <= self.MIN_NOISE_FLOOR:
                            self.noise_floor = max(self.MIN_NOISE_FLOOR, measured)
                        else:
                            self.noise_floor = max(
                                self.MIN_NOISE_FLOOR,
                                self.noise_floor * 0.65 + measured * 0.35)
                        ambient.extend(calibration)
                        # Do not throw away a short answer that begins immediately
                        # after playback. Seed onset from the settling window using
                        # the carried/blended floor, then let the current frame flow
                        # through normal detection below. The previous code retained
                        # these samples in preroll but never evaluated them, so a
                        # quick "yes" inside the 200 ms window vanished completely.
                        adaptive = max(
                            self.MIN_ONSET,
                            self.noise_floor
                            + max(self.ONSET_MARGIN,
                                  self.noise_floor * self.ONSET_RELATIVE))
                        calibration_onset = (
                            min(adaptive, self.STARTUP_ONSET)
                            if starting_uncalibrated else adaptive)
                        calibration_trigger = calibration_onset
                        prior = calibration[-self.ONSET_WINDOW_CHUNKS:-1]
                        onset_levels.extend(prior)
                        onset_votes.extend(v >= calibration_onset for v in prior)
                        calibration_speech_s = (
                            sum(v >= calibration_onset for v in calibration)
                            * self.CHUNK_SECONDS)
                    else:
                        continue

                if not speaking:
                    adaptive_onset = max(
                        self.MIN_ONSET,
                        self.noise_floor
                        + max(self.ONSET_MARGIN,
                              self.noise_floor * self.ONSET_RELATIVE))
                    candidate_onset = adaptive_onset
                    # Only observations below the current onset estimate are safe
                    # room samples. Feeding probable speech back into the floor makes
                    # the threshold chase the speaker upward.
                    if rms < candidate_onset:
                        ambient.append(rms)
                        if len(ambient) >= 25:
                            self.noise_floor = max(
                                self.MIN_NOISE_FLOOR,
                                float(np.percentile(ambient, 25)))
                noise = self.noise_floor
                adaptive_onset = max(
                    self.MIN_ONSET,
                    noise + max(self.ONSET_MARGIN, noise * self.ONSET_RELATIVE))
                threshold = adaptive_onset
                # Release is tied to the measured ROOM, not to a fraction of onset.
                # With floor=.0143 and onset=.017, the old onset*.6 release was
                # .0102: even an empty room never became "silent", so recording
                # stayed latched and eventually transcribed unrelated later noise.
                release = noise + max(self.RELEASE_MARGIN,
                                      noise * self.RELEASE_RELATIVE)

                above_onset = rms >= threshold
                if not speaking:
                    onset_votes.append(above_onset)
                    onset_levels.append(rms)
                    if (len(onset_votes) == onset_votes.maxlen
                            and sum(onset_votes) >= self.ONSET_REQUIRED_CHUNKS):
                        speaking, frames = True, list(preroll)
                        trigger_threshold = calibration_trigger or threshold
                        # Only peaks belonging to this onset may validate the turn.
                        # Keeping the largest level seen during hours of idle listening
                        # let an old door slam validate unrelated boundary noise later.
                        peak_rms = max(onset_levels)
                        recording_levels.extend(onset_levels)
                        recent_recording.extend(onset_levels)
                        # Count the confirmed onset frames already in the window.
                        speech_s = max(
                            sum(onset_votes) * self.CHUNK_SECONDS,
                            calibration_speech_s)
                        elapsed_s = 0.0
                        silence_s = 0.0
                    continue

                frames.append(raw)
                elapsed_s += self.CHUNK_SECONDS
                peak_rms = max(peak_rms, rms)
                recording_levels.append(rms)
                recent_recording.append(rms)
                # If capture reopened into a genuinely higher, nearly-flat room
                # level, the carried floor is temporarily too low and ordinary room
                # audio looks like speech. Recognize that shape after one second,
                # raise the baseline, and let it endpoint/discard promptly. Speech is
                # dynamic; a fan or new steady electrical/noise floor is not.
                if len(recent_recording) >= 50:
                    low = float(np.percentile(recent_recording, 10))
                    high = float(np.percentile(recent_recording, 90))
                    if high - low < self.FLAT_BASELINE_RANGE and low > self.noise_floor:
                        self.noise_floor = low
                if above_onset:
                    speech_s += self.CHUNK_SECONDS
                    silence_s = 0.0
                elif rms <= release:
                    silence_s += self.CHUNK_SECONDS
                else:
                    # The hysteresis band is neither clear speech nor clear silence.
                    # Let an existing quiet run decay slowly instead of resetting it
                    # forever on ordinary room jitter.
                    silence_s = max(0.0, silence_s - self.CHUNK_SECONDS * 0.25)

                if silence_s >= self.ENDPOINT_SECONDS:
                    endpoint_reason = "silence"
                    break
                # This is WALL duration, not accumulated above-threshold frames.
                # The old comparison used speech_s, so a latched noisy recording
                # could run for minutes before collecting "8 seconds" of spikes.
                if elapsed_s >= self.MAX_UTTERANCE_SECONDS:
                    endpoint_reason = "timeout"
                    break
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        quality_peak = trigger_threshold + self.MIN_PEAK_MARGIN
        baseline = (float(np.percentile(recording_levels, 25))
                    if recording_levels else peak_rms)
        dynamic_range = peak_rms - baseline
        valid = (frames and speech_s >= self.MIN_SPEECH_SECONDS
                 and peak_rms >= quality_peak
                 and dynamic_range >= self.MIN_DYNAMIC_RANGE)
        if valid and not self._stop.is_set():
            if endpoint_reason == "silence":
                # Keep about 120 ms after the last voiced material; the rest is VAD
                # proof, not speech, and only invites STT hallucinations.
                trim = max(0, int((silence_s - 0.12) / self.CHUNK_SECONDS))
                if trim:
                    frames = frames[:-trim]
            # The numbers that decide whether the robot hears you at all. Without
            # these in the log, a threshold drifting out of range is invisible and
            # presents only as "it stopped responding".
            print(f"[board-audio] heard {speech_s:.1f}s/{elapsed_s:.1f}s "
                  f"(floor={self.noise_floor:.4f} threshold={threshold:.4f} "
                  f"release={release:.4f} peak={peak_rms:.4f} "
                  f"range={dynamic_range:.4f} "
                  f"end={endpoint_reason or 'stopped'})", flush=True)
            try:
                self._answer(b"".join(frames))
            except SystemExit:
                raise
            except Exception as exc:
                # STT/LLM/TTS failures are turn failures, not microphone failures.
                # Letting these escape into _run() caused five cloud or synthesis
                # errors to be announced as "check my microphone", sending diagnosis
                # in exactly the wrong direction.
                print(f"[board-audio] turn failed: {type(exc).__name__}: {exc}",
                      flush=True)
                self._speak("Sorry, that answer failed. Please try again.")
        elif speaking:
            reason = ("too short" if speech_s < self.MIN_SPEECH_SECONDS
                      else "weak/noisy trigger")
            print(f"[board-audio] discarded {speech_s:.1f}s/{elapsed_s:.1f}s: {reason} "
                  f"(floor={self.noise_floor:.4f} threshold={threshold:.4f} "
                  f"peak={peak_rms:.4f}, range={dynamic_range:.4f}, "
                  f"need={quality_peak:.4f})", flush=True)

    # Level to normalize the captured utterance to before handing it to STT.
    # 0.10 RMS is a healthy speech recording; this mic delivers about 0.013.
    CAPTURE_TARGET_RMS = float(os.environ.get("VOICE_CAPTURE_TARGET_RMS", "0.10"))

    def _normalize_capture(self, pcm):
        """Bring a quiet microphone up to a level the recognizer expects.

        Moonshine was being handed speech at ~2% of full scale, which is where
        small STT models start dropping words. Normalizing costs a millisecond and
        is not the same as turning up the mic gain -- that is already maxed -- but
        it does put the signal in the range the model was trained on.
        """
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if not samples.size:
            return pcm
        # Estimate speech loudness from short-window RMS rather than the whole file:
        # preroll and endpoint silence must not decide the gain. Also respect peak
        # headroom. The previous 30x ceiling could turn a false noise capture into a
        # clipped waveform that STT confidently hallucinated words from.
        window = self.CHUNK_SAMPLES
        usable = samples[:(samples.size // window) * window]
        if not usable.size:
            return pcm
        levels = np.sqrt(np.mean(usable.reshape(-1, window) ** 2, axis=1))
        active_rms = float(np.percentile(levels, 75))
        peak = float(np.max(np.abs(samples)))
        if active_rms < 1e-5 or peak < 1e-5:
            return pcm
        gain = min(self.CAPTURE_TARGET_RMS / active_rms, 0.95 / peak, 12.0)
        samples = np.clip(samples * gain, -1.0, 1.0)
        return (samples * 32767.0).astype("<i2").tobytes()

    def _answer(self, pcm):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(self.RATE)
                out.writeframes(self._normalize_capture(pcm))
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
        with one bounded synthesis worker that is joined before the turn exits.

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
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uno-tts")
        try:
            # Pay the first synthesis before opening playback. Thereafter synthesize
            # chunk N+1 while chunk N is being written/drained by aplay. The previous
            # sequential loop claimed to overlap these phases but could not: a pipe
            # write blocks behind playback, so Piper sat idle and replies developed
            # long gaps between sentences.
            wav = self.agent.tts.synth(chunks[0])
            for index, text in enumerate(chunks):
                next_text = chunks[index + 1] if index + 1 < len(chunks) else None
                if not wav:
                    next_wav = (pool.submit(self.agent.tts.synth, next_text)
                                if next_text else None)
                    wav = next_wav.result() if next_wav else b""
                    continue
                try:
                    pcm, fmt = _wav_pcm(wav)
                except (wave.Error, EOFError) as exc:
                    print(f"[board-audio] unplayable synthesis for {text!r}: {exc}")
                    next_wav = (pool.submit(self.agent.tts.synth, next_text)
                                if next_text else None)
                    wav = next_wav.result() if next_wav else b""
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
                next_wav = (pool.submit(self.agent.tts.synth, next_text)
                            if next_text else None)
                try:
                    proc.stdin.write(_amplify(pcm) if fmt[2] == 2 else pcm)
                except (BrokenPipeError, OSError) as exc:
                    print(f"[board-audio] playback stopped early: {exc}")
                    return spoke, first_chunk_ms
                spoke = True
                wav = next_wav.result() if next_wav else b""
            return spoke, first_chunk_ms
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
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
        # Assert the mixer level HERE, immediately before the stream opens, not only
        # when the device was discovered. Observed on this board: the volume reads
        # 100% right after discovery sets it and 40% by the time a reply plays --
        # a USB audio device reverts to its factory default across re-enumeration
        # and stream open, and 40% is 16 dB down. Doing it per playback costs about
        # ten milliseconds and makes the loudness deterministic instead of a race.
        set_playback_volume(self.playback_device)
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
        not kill the listen loop that is the robot's only input.

        Amplified like the streaming path, by decoding to PCM and feeding aplay
        raw -- otherwise status messages would be markedly quieter than replies.
        """
        try:
            pcm, fmt = _wav_pcm(wav_bytes)
            proc = self._open_playback(fmt)
            if proc is None:
                return
            try:
                proc.stdin.write(_amplify(pcm) if fmt[2] == 2 else pcm)
            finally:
                self._close_playback(proc)
        except (OSError, subprocess.SubprocessError, wave.Error, EOFError) as exc:
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
