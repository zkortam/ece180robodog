"""Text-to-speech with pluggable backends, returning WAV bytes.

mac dev : kokoro  (Kokoro-82M, ONNX, most natural on CPU)   VOICE_TTS=kokoro
board   : piper   (neural, ONNX, streams on ARM CPU)        VOICE_TTS=piper
fallback: say (macOS built-in) / espeak-ng (robotic, always-works)

If the chosen backend fails, it degrades gracefully so a turn always speaks."""
import io
import re
import select
import shutil
import subprocess
import tempfile
import threading
import wave
from collections import deque
from pathlib import Path

import numpy as np

from . import config


class TTSUnavailable(RuntimeError):
    pass


_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "a.m", "p.m",
}


def sentence_chunks(text, max_chunks=5):
    """Split spoken prose at safe sentence boundaries without changing words.

    This deliberately avoids a broad ``'. '`` regex: splitting ``Dr. Smith`` or
    ``e.g. cameras`` changes Piper's phrasing. The complete reply remains available
    in metadata; chunks are only a transport/synthesis optimization.
    """
    text = " ".join((text or "").split()).strip()
    if not text:
        return []
    parts, start = [], 0
    for match in re.finditer(r"[.!?]+(?:[\"'”’]+)?\s+", text):
        punctuation = match.group(0).lstrip()[0]
        prefix = text[start:match.start()]
        last = prefix.rsplit(None, 1)[-1].lower().rstrip(".\"'”’") if prefix else ""
        if punctuation == "." and (last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())):
            continue
        parts.append(text[start:match.end()].strip())
        start = match.end()
    if start < len(text):
        parts.append(text[start:].strip())
    parts = [p for p in parts if p]
    if len(parts) > 1 and len(parts[0]) < 18:
        parts[1] = parts[0] + " " + parts[1]
        parts.pop(0)
    if len(parts) > max_chunks:
        parts = parts[:max_chunks - 1] + [" ".join(parts[max_chunks - 1:])]
    return parts or [text]


def _wav_bytes(samples, sample_rate):
    s = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (s * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class TTS:
    def __init__(self, backend=None):
        self.backend = backend or config.TTS_BACKEND
        self._kkeng = None      # cached Kokoro engine (name must NOT collide with _kokoro())
        self._piper_proc = None
        self._piper_dir = None
        self._piper_lock = threading.Lock()
        self._piper_stderr = deque(maxlen=12)

    def synth(self, text):
        text = (text or "").strip()
        if not text:
            return b""
        try:
            return self._run(self.backend, text)
        except Exception as e:
            import sys
            print(f"[tts] backend '{self.backend}' failed ({e}); falling back", file=sys.stderr)
            for fb in ("kokoro", "say", "espeak", "piper"):   # degrade so a turn always speaks
                if fb != self.backend:
                    try:
                        return self._run(fb, text)
                    except Exception:
                        continue
            raise

    def _run(self, backend, text):
        return getattr(self, "_" + backend)(text)

    # ---- Kokoro: most natural, CPU, ONNX (mac dev) ----
    def _kokoro_engine(self):
        if self._kkeng is None:
            from kokoro_onnx import Kokoro
            if not Path(config.KOKORO_MODEL).exists():
                raise TTSUnavailable(f"Kokoro model missing: {config.KOKORO_MODEL}")
            self._kkeng = Kokoro(config.KOKORO_MODEL, config.KOKORO_VOICES)
        return self._kkeng

    def _kokoro(self, text):
        eng = self._kokoro_engine()
        samples, sr = eng.create(text, voice=config.KOKORO_VOICE, speed=1.0, lang="en-us")
        return _wav_bytes(samples, sr)

    # ---- Piper: neural, light (board) ----
    def _piper(self, text):
        # Piper's voice is a 61 MB ONNX model and took ~2.2 s to load on the UNO Q.
        # Keep one CLI process alive: it accepts one utterance per stdin line and
        # prints the completed WAV path on stdout. Startup warming in main.py then
        # pays model loading once at boot, not once in every conversation turn.
        clean = " ".join(text.replace("\x00", " ").splitlines()).strip()
        if not clean:
            return b""
        with self._piper_lock:
            for attempt in range(2):
                try:
                    proc = self._ensure_piper()
                    proc.stdin.write(clean + "\n")
                    proc.stdin.flush()
                    ready, _, _ = select.select([proc.stdout], [], [], 30.0)
                    if not ready:
                        raise TTSUnavailable("persistent piper timed out")
                    output = proc.stdout.readline().strip()
                    path = Path(output)
                    if not output or not path.is_file():
                        detail = "; ".join(self._piper_stderr)
                        raise TTSUnavailable(f"piper produced no WAV ({detail[-300:]})")
                    return self._read(path)
                except (BrokenPipeError, OSError, TTSUnavailable):
                    self._stop_piper()
                    if attempt:
                        raise
        raise TTSUnavailable("persistent piper failed")

    def _ensure_piper(self):
        if self._piper_proc is not None and self._piper_proc.poll() is None:
            return self._piper_proc
        # Reap the previous generation first. Spawning over a dead process leaked
        # its pipes and its output directory on every respawn -- and Piper is
        # respawned exactly when things are already going wrong.
        self._stop_piper()
        piper = shutil.which(config.PIPER_BIN) or config.PIPER_BIN
        voice = config.PIPER_VOICE
        if not Path(voice).exists():
            raise TTSUnavailable(f"Piper voice not found: {voice} (run install_voice.sh)")
        self._piper_dir = tempfile.TemporaryDirectory(prefix="voice-piper-")
        self._piper_stderr.clear()
        try:
            self._piper_proc = subprocess.Popen(
                [piper, "--model", voice, "--output_dir", self._piper_dir.name,
                 "--sentence_silence", "0", "--length_scale", str(config.PIPER_LENGTH_SCALE)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
        except OSError as e:
            # No binary on PATH: clean up the directory we just made, then let the
            # caller fall back to another backend.
            self._piper_dir.cleanup()
            self._piper_dir = None
            raise TTSUnavailable(f"could not start piper ({config.PIPER_BIN}): {e}") from e
        threading.Thread(target=self._drain_piper_stderr, args=(self._piper_proc,),
                         daemon=True).start()
        return self._piper_proc

    def _drain_piper_stderr(self, proc):
        # Take the process as an argument: reading self._piper_proc here would
        # follow a respawn and quietly attribute the new process's stderr to it.
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._piper_stderr.append(line.strip())
        except (ValueError, OSError):
            pass                                    # pipe closed by _stop_piper

    def _stop_piper(self):
        """Tear down the persistent Piper process and its output directory.

        Every step is best-effort and the directory cleanup is in a `finally`:
        this runs on the failure path, so it must not raise a *second* error over
        the one that brought us here.
        """
        proc, self._piper_proc = self._piper_proc, None
        try:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                for pipe in (proc.stdin, proc.stdout, proc.stderr):
                    try:
                        if pipe is not None:
                            pipe.close()
                    except OSError:
                        pass
        except Exception:
            pass
        finally:
            if self._piper_dir is not None:
                try:
                    self._piper_dir.cleanup()
                except OSError:
                    pass
                self._piper_dir = None

    # ---- macOS say (fallback) ----
    def _say(self, text):
        if not shutil.which("say"):
            raise TTSUnavailable("macOS `say` not found")
        out = self._tmp()
        subprocess.run(["say", "-v", config.SAY_VOICE, "--data-format=LEI16@22050",
                        "-o", str(out), text], check=True)
        return self._read(out)

    # ---- espeak-ng (always-works fallback) ----
    def _espeak(self, text):
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not exe:
            raise TTSUnavailable("espeak-ng not found")
        out = self._tmp()
        subprocess.run([exe, "-w", str(out), text], check=True)
        return self._read(out)

    def _tmp(self):
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.close()
        return Path(f.name)

    def _read(self, path):
        data = path.read_bytes()
        path.unlink(missing_ok=True)
        return data
