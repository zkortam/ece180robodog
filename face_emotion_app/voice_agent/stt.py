"""Speech-to-text with pluggable backends.

board  : moonshine  (Moonshine v2 tiny, streaming, ONNX) -- see VOICE_STT=moonshine
mac dev: faster-whisper (tiny.en) -- decodes webm/ogg/wav via PyAV, easy to run now

All imports are lazy so importing this module never requires the heavy deps.
transcribe() accepts a path to any audio file the browser recorded."""
from . import config


class NoSpeech(Exception):
    """Audio that could not be decoded, or held no speech. Not an error to report."""


# Whisper-family models emit a stock phrase when handed silence (small.en returns
# "You" for pure silence, every time). Hands-free VAD will occasionally ship a
# cough or a door slam, and an unguarded hallucination makes the agent answer a
# question nobody asked. vad_filter catches this on faster-whisper; this net
# catches the same artifacts on backends that have no VAD (moonshine, on the board).
#
# THIS IS THE ONLY NOISE POLICY IN THE STACK. The orchestrator used to keep a
# second, much wider list of its own that contradicted the rule below: it dropped
# "okay", "yeah", "thanks", and "bye". Those are exactly the words a person says to
# a robot, and "yeah" is the expected answer to the agent's own question during
# registration ("want me to learn a few expressions?"), so that turn was swallowed
# and the flow silently stalled.
#
# Deliberately narrow: only strings that are never a real standalone utterance.
# Filler sounds and bare punctuation qualify. Real words do not -- swallowing a
# real word is worse than answering a rare hallucination, because the user gets no
# feedback at all and simply repeats themselves.
_ARTIFACTS = {
    "you", "thanks for watching", "thank you for watching",
    "[blank_audio]", "[silence]", "[ silence ]", "[music]", "(silence)",
    "uh", "um", "hmm", "mm", "mhm", "hm", "huh", "ah", "eh",
}


def is_noise_transcript(text):
    """True when a transcript is an STT artifact rather than something a person said."""
    t = " ".join((text or "").split()).strip().strip(" .!?,").lower()
    return not t or t in _ARTIFACTS        # bare punctuation normalises to ""


_is_artifact = is_noise_transcript          # backwards-compatible alias


class STT:
    def __init__(self, backend=None, model=None):
        self.backend = backend or config.STT_BACKEND
        # Resolve the model FROM the chosen backend, not from the platform default:
        # the two are not interchangeable (see config.stt_model_for).
        self.model_name = model or config.stt_model_for(self.backend)
        self._impl = None

    def _load(self):
        if self._impl is not None:
            return
        if self.backend == "faster-whisper":
            from faster_whisper import WhisperModel
            self._impl = WhisperModel(self.model_name, device="cpu", compute_type="int8",
                                      cpu_threads=config.CPU_THREADS)
        elif self.backend == "moonshine":
            import moonshine_onnx  # pip install useful-moonshine-onnx
            # Keep the ONNX sessions resident. Passing a model-name string to
            # moonshine_onnx.transcribe() reconstructs the model on every turn,
            # which costs several seconds on the UNO Q.
            model = (f"moonshine/{self.model_name}"
                     if "/" not in self.model_name else self.model_name)
            self._impl = (moonshine_onnx,
                          moonshine_onnx.MoonshineOnnxModel(model_name=model))
        elif self.backend == "whisper":
            import whisper
            self._impl = whisper.load_model(self.model_name)
        else:
            raise SystemExit(f"unknown STT backend: {self.backend}")

    def transcribe(self, audio_path):
        """Text for the utterance, or "" if it held no speech.

        Raises NoSpeech if the file cannot be decoded at all -- a truncated or
        empty MediaRecorder blob is a routine browser event, not a server fault,
        and must not surface as a 500."""
        self._load()
        try:
            text = self._transcribe(audio_path)
        except NoSpeech:
            raise
        except Exception as e:
            if self._undecodable(e):
                raise NoSpeech(str(e)) from e
            raise
        return "" if _is_artifact(text) else text

    @staticmethod
    def _undecodable(exc):
        s = f"{type(exc).__name__}: {exc}".lower()
        return ("invalid data found" in s or "end of file" in s
                or "invaliddataerror" in s or "moov atom not found" in s)

    def _transcribe(self, audio_path):
        if self.backend == "faster-whisper":
            # vad_filter drops non-speech BEFORE decoding: without it small.en
            # returns "You" for pure silence (and it is ~8x slower on silence).
            segments, _ = self._impl.transcribe(audio_path, beam_size=1, language="en",
                                                vad_filter=True)
            return "".join(s.text for s in segments).strip()
        if self.backend == "moonshine":
            moonshine_onnx, model = self._impl
            out = moonshine_onnx.transcribe(audio_path, model)
            return (out[0] if isinstance(out, (list, tuple)) else out).strip()
        if self.backend == "whisper":
            return self._impl.transcribe(audio_path).get("text", "").strip()
        return ""
