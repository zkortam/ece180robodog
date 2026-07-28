"""Central configuration. Everything the agent needs to run, in one place, with
env-var overrides. The Cerebras key is read from the environment ONLY."""
import os
import platform
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent          # face_emotion_app/
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"


def _env(name, default):
    return os.environ.get(name, default)


# ---------- Cerebras (cloud LLM) ----------
CEREBRAS_BASE_URL = _env("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
CEREBRAS_MODEL = _env("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_API_KEY_ENV = "CEREBRAS_API_KEY"                  # never hardcode the key
MAX_TOOL_ROUNDS = int(_env("VOICE_MAX_TOOL_ROUNDS", "5"))
# gpt-oss-120b is a REASONING model: its hidden reasoning is billed against
# max_completion_tokens. A budget sized for the spoken reply alone (~140) is spent
# on reasoning before a single word is emitted -> finish_reason='length', content
# None, and the caller sees an empty reply. Leave room for reasoning + the reply.
MAX_COMPLETION_TOKENS = int(_env("VOICE_MAX_COMPLETION_TOKENS", "800"))
# 'low' keeps reasoning ~95 tokens / sub-second, which is what a voice turn needs.
REASONING_EFFORT = _env("VOICE_REASONING_EFFORT", "low")

# ---------- Vision ----------
VISION_CAMERA = int(_env("VOICE_CAMERA", "0"))
VISION_WIDTH = int(_env("VOICE_CAM_W", "320"))
VISION_HEIGHT = int(_env("VOICE_CAM_H", "240"))
VISION_FPS = int(_env("VOICE_FPS", "4"))
# Seconds between expression inferences per tracked face. Raise it on the board if
# the cores are needed for STT/TTS; identity tracking is unaffected either way.
VISION_EMOTION_INTERVAL = float(_env("VOICE_EMOTION_INTERVAL", "1.5"))
# Favor "unknown" over a wrong name. Track-level hysteresis preserves a confirmed
# identity through brief weak frames, so this can be conservative without flicker.
VISION_THRESHOLD = float(_env("VOICE_THRESHOLD", "0.58"))

# ---------- STT ----------
# board default: moonshine ; mac dev default: faster-whisper (auto-detected)
_IS_MAC = platform.system() == "Darwin"
STT_BACKEND = _env("VOICE_STT", "faster-whisper" if _IS_MAC else "moonshine")
# Model names are backend-specific and NOT interchangeable: Whisper wants
# "small.en", Moonshine wants "tiny"/"base". Deriving one global default from the
# platform's default backend silently mispaired them the moment anyone passed
# --stt (e.g. --stt moonshine on a Mac asked Moonshine for "moonshine/small.en",
# which does not exist). Resolve per backend, and let the env var still win.
_STT_DEFAULT_MODELS = {
    # base.en, NOT small.en. Measured on this project's own phrases (M-series CPU,
    # int8, beam_size=1, vad_filter on), 6 utterances including the name "Zakaria":
    #
    #     small.en   22.02 s total   (2.4-6.1 s per utterance)
    #     base.en     4.47 s total   (0.6-1.0 s per utterance)   IDENTICAL text
    #
    # small.en was the single largest component of turn latency -- bigger than the
    # LLM call and the speech synthesis combined -- and bought nothing on this
    # vocabulary. tiny.en is faster still but starts mangling words ("a 1 sentence"
    # for "a one sentence"), and a misheard name is worse than a slower reply.
    # Override with VOICE_STT_MODEL if a harder accent or a noisier room needs it.
    "faster-whisper": "base.en",
    "whisper": "base.en",
    "moonshine": "tiny",
}


def stt_model_for(backend):
    """The model name to use with `backend`; VOICE_STT_MODEL overrides everything."""
    override = os.environ.get("VOICE_STT_MODEL")
    return override or _STT_DEFAULT_MODELS.get(backend, "tiny")


STT_BACKENDS = tuple(_STT_DEFAULT_MODELS)
STT_MODEL = stt_model_for(STT_BACKEND)

# ---------- TTS ----------
# mac dev: Kokoro (most natural, ONNX, CPU) if models present, else macOS `say`
# board  : Piper (lightweight neural). espeak-ng is the always-works fallback.
KOKORO_MODEL = _env("KOKORO_MODEL", str(MODELS_DIR / "kokoro-v1.0.onnx"))
KOKORO_VOICES = _env("KOKORO_VOICES", str(MODELS_DIR / "voices-v1.0.bin"))
KOKORO_VOICE = _env("KOKORO_VOICE", "af_heart")
PIPER_BIN = _env("PIPER_BIN", "piper")
PIPER_VOICE = _env("PIPER_VOICE", str(MODELS_DIR / "en_US-lessac-low.onnx"))
# Slightly faster than the model default without the rushed/robotic quality of
# aggressive time compression. It shortens both synthesis and playback.
PIPER_LENGTH_SCALE = float(_env("PIPER_LENGTH_SCALE", "0.90"))
SAY_VOICE = _env("VOICE_SAY_VOICE", "Samantha")

if _IS_MAC:
    _default_tts = "kokoro" if Path(KOKORO_MODEL).exists() else "say"
else:
    _default_tts = "piper"
TTS_BACKEND = _env("VOICE_TTS", _default_tts)
TTS_BACKENDS = ("kokoro", "piper", "say", "espeak")

# ---------- Conversation ----------
SYSTEM_PROMPT = _env("VOICE_SYSTEM_PROMPT", (
    "You are a voice assistant that can see through a camera. You are talking out loud. "
    "Sound natural, warm, and conversational. BE BRIEF: every word is spoken aloud and the user is "
    "waiting through it. Use ONE short sentence for a normal answer, and at most two when the subject "
    "genuinely needs it. Go longer only if the user explicitly asks you to explain or elaborate. Do not "
    "ramble, repeat yourself, pad the answer with generic filler, or ask unnecessary follow-up questions. "
    "Do not comment on what you see unless the user asks. "
    "Use a vision tool ONLY when the user explicitly asks who is there, what someone looks like, or how "
    "someone feels (who_is_in_view, describe_scene, get_person_emotion, emotion_timeline, "
    "presence_events). For every other message, just answer directly with no tool. "
    "IDENTITY IS STRICT: only use a name that literally appears in a tool's 'known' list; a face that is "
    "unknown, or when 'known' is empty, is someone you do NOT recognize — say 'someone I don't "
    "recognize'. Never guess or invent a name. Never mention tools or JSON out loud.\n\n"
    "REGISTRATION (only if the user asks to register/enroll): Ask their name ONCE — do not spell it "
    "back, do not ask them to confirm it. As soon as they give a name, call enroll_face(name) that SAME "
    "turn (they are already looking at the camera), then greet them warmly by name in one sentence. "
    "After the face saves, in one short sentence offer to learn a few expressions; if they say yes, for "
    "each of neutral, happy, sad, surprise tell them to make that face and call train_emotion(name, "
    "expression) — brief, no confirmations, no spelling."
))
HISTORY_MAX_MESSAGES = int(_env("VOICE_HISTORY_MAX", "20"))

# ---------- Latency ----------
# Silence (ms) after speech before the turn is closed. 300 ms makes the device
# feel noticeably more responsive while still leaving room for a normal short
# breath; deployments in noisy rooms can override this with VOICE_ENDPOINT_MS.
VAD_ENDPOINT_MS = int(_env("VOICE_ENDPOINT_MS", "250"))
# STT and TTS both default to one thread per core. That is right on an idle box
# and actively harmful when anything else is busy: the pools oversubscribe, the
# scheduler thrashes, and a short utterance that costs 1.5s at four threads costs
# over 4s at eight. Half the cores is close to optimal loaded and costs little idle.
# Origins allowed to drive this board from a page hosted elsewhere. *.vercel.app
# is accepted implicitly; add explicit origins (e.g. a custom domain) here.
ALLOWED_UI_ORIGINS = tuple(o for o in _env("VOICE_ALLOWED_ORIGINS", "").split(",") if o)
# Largest request body the board will read. See create_app().
MAX_UPLOAD_BYTES = int(_env("VOICE_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

CPU_THREADS = int(_env("VOICE_CPU_THREADS", str(max(2, min(4, (os.cpu_count() or 4) // 2)))))
# How long a request may wait for the in-flight turn before it is told the agent
# is busy. Turns are half-duplex, so waiting is correct; waiting forever behind a
# wedged turn is not -- that leaves the UI stuck in "Thinking…" with no way out.
TURN_LOCK_TIMEOUT = float(_env("VOICE_TURN_LOCK_TIMEOUT", "45"))

# ---------- Tool policy (risk tiers for MCP; vision tools read, except the two
# enrollment writers -- see tools.py) ----------
RISK_READONLY = "readonly"
RISK_WRITE = "write"            # first-party local writes (enroll/train): allowed, but not read-only
RISK_SENSITIVE = "sensitive"
RISK_DESTRUCTIVE = "destructive"
IDENTITY_TAU_HIGH = float(_env("VOICE_TAU_HIGH", "0.6"))   # SFace cosine to authorize actions

# ---------- MCP servers (Tier 1). Empty for the MVP; add entries to integrate. ----------
# each: {"id","transport":"stdio"|"http","command"/"args" or "url","risk":RISK_*}
MCP_SERVERS = []


def require_cerebras_key():
    key = os.environ.get(CEREBRAS_API_KEY_ENV)
    if not key:
        raise SystemExit(
            f"{CEREBRAS_API_KEY_ENV} is not set. Export your Cerebras key first:\n"
            f"  export {CEREBRAS_API_KEY_ENV}=csk-...\n"
            "(Keep it out of the repo. Rotate any key that has been shared in plaintext.)")
    return key
