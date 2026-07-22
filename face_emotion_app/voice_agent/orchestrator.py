"""VoiceAgent: one turn = audio in -> STT -> Cerebras(+vision tools) -> TTS -> audio out.

Half-duplex, push-to-talk friendly. The Cerebras client is created lazily on the
first turn so the vision loop + web UI come up even before a key is set."""
import base64
import sys
import threading
import time

from . import config
from .cerebras_client import CerebrasClient, Truncated, _status, is_auth_error, is_rate_limit
from .stt import STT, NoSpeech
from .tts import TTS
from .tool_bus import ToolBus
from .tools import VisionTools


class VoiceAgent:
    def __init__(self, vision_service, stt=None, tts=None, owner_name=None):
        self.vision = vision_service
        self.tools = VisionTools(vision_service)
        self.bus = ToolBus(self.tools)
        self.bus.build()
        self.stt = stt or STT()
        self.tts = tts or TTS()
        self.owner_name = owner_name
        self.turn_lock = threading.RLock()
        self._llm = None
        self.history = []      # conversation only (user/assistant); system + scene prepended per turn

    def _llm_client(self):
        if self._llm is None:
            self._llm = CerebrasClient()   # raises if CEREBRAS_API_KEY unset
        return self._llm

    def _identity(self):
        """Identity context for gating sensitive tools (owner in view + confidence)."""
        if not self.owner_name:
            return None
        view = self.vision.who_is_in_view()
        for k in view.get("known", []):
            if k["name"] == self.owner_name:
                return {"owner_present": True, "identity_score": k["identity_score"],
                        "liveness": False}
        return {"owner_present": False, "identity_score": 0.0, "liveness": False}

    def _trim(self):
        if len(self.history) > config.HISTORY_MAX_MESSAGES:
            self.history = self.history[-config.HISTORY_MAX_MESSAGES:]

    _SCENE_PREFIX = ("SCENE (live, private context for your awareness; do not read it aloud "
                     "unless the user asks who you see): ")

    def _scene_context(self):
        """A compact, live description of who the agent is looking at, injected each turn
        so the model has situational awareness without spending a tool call. Only states
        things that are actually live: a stale feed or stale emotion is not reported."""
        try:
            view = self.vision.who_is_in_view()
        except Exception:
            return self._SCENE_PREFIX + "camera unavailable."
        stale = view.get("stale_seconds")
        if not view.get("watching") or (stale is not None and stale > 2.5):
            return self._SCENE_PREFIX + "the camera feed is idle right now (no live view)."
        known = view.get("known", [])
        if known:
            people = []
            for k in known:
                name = k["name"]
                em = self.vision.get_person_emotion(name)
                fresh = (em.get("found") and em.get("present")
                         and (em.get("sample_age_seconds") or 99) < 4.0)
                mood = em.get("emotion") if fresh else None
                people.append(name + (f" (looks {mood})" if mood else ""))
            body = "in view right now: " + ", ".join(people) + "."
            if len(known) > 1:
                body += (" Two or more known people are visible, so you cannot tell which one is "
                         "the speaker; if asked a personal question, ask which person they mean.")
        elif view.get("num_faces", 0) > 0:
            body = "a face is in view but not recognized."
        else:
            body = "no one is in view."
        return self._SCENE_PREFIX + body

    def handle_text(self, text):
        """One turn. System prompt + fresh live scene are prepended each call; only the
        user/assistant exchange is persisted to history (keeps context clean and cheap)."""
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "system", "content": self._scene_context()}]
        messages += self.history
        messages.append({"role": "user", "content": text})
        errored = False
        try:
            reply, _, trace = self._llm_client().run(
                messages, self.bus.schemas(), self.bus.dispatch, identity=self._identity())
        except SystemExit:
            raise                              # missing key -> handled as 503 upstream
        except Exception as e:
            # A rejected key or an exhausted token budget is a CONFIG fault: it fails
            # identically on every turn, so it must stop the loop with the real reason
            # rather than apologize forever. Everything else -> speak, but never silently.
            if is_auth_error(e):
                raise SystemExit(f"Cerebras rejected {config.CEREBRAS_API_KEY_ENV} "
                                 f"({_status(e) or 'auth error'}). Check the key.") from e
            if isinstance(e, Truncated):
                raise SystemExit(f"{e}") from e
            print(f"[voice] LLM turn failed: {type(e).__name__}: {e}", file=sys.stderr)
            trace, errored = [], True
            if is_rate_limit(e):
                reply = "Sorry, I'm getting rate limited. Give me a few seconds and try again."
            else:
                reply = "Sorry, something went wrong on my end. Could you say that again?"
        if not reply or not reply.strip():
            print("[voice] LLM returned an empty reply", file=sys.stderr)
            reply, errored = "Sorry, could you say that again?", True
        # Canned apologies are not conversation: persisting them teaches the model to
        # keep apologizing after the API recovers.
        if not errored:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            self._trim()
        return {"transcript": text, "reply": reply, "tools": trace}

    def understand_audio(self, audio_path):
        """Transcribe and answer, but leave synthesis to the transport.

        Keeping this as a shared primitive lets the normal JSON endpoint return one
        complete WAV while the streaming endpoint synthesizes sentence chunks. Both
        paths therefore use identical STT, LLM, history, tools, and error behavior.
        """
        t_stt = time.perf_counter()
        try:
            transcript = self.stt.transcribe(audio_path)
        except NoSpeech:
            # Truncated/empty recording from the browser. Routine; not a server error.
            transcript = ""
        stt_ms = (time.perf_counter() - t_stt) * 1000
        if not transcript:
            return {"transcript": "", "reply": "", "tools": [],
                    "note": "no speech detected", "timings_ms": {"stt": round(stt_ms, 1)}}

        t_llm = time.perf_counter()
        out = self.handle_text(transcript)
        llm_ms = (time.perf_counter() - t_llm) * 1000
        out["timings_ms"] = {"stt": round(stt_ms, 1), "llm": round(llm_ms, 1)}
        return out

    def handle_audio(self, audio_path):
        """Full compatibility turn: transcribe -> think -> one complete WAV."""
        out = self.understand_audio(audio_path)
        if not out.get("transcript"):
            out["audio_b64"] = ""
            return out

        t_tts = time.perf_counter()
        wav = self.tts.synth(out["reply"])
        tts_ms = (time.perf_counter() - t_tts) * 1000

        out["audio_b64"] = base64.b64encode(wav).decode() if wav else ""
        stt_ms = out["timings_ms"].get("stt", 0.0)
        llm_ms = out["timings_ms"].get("llm", 0.0)
        out["timings_ms"].update(tts=round(tts_ms, 1),
                                 total=round(stt_ms + llm_ms + tts_ms, 1))
        print(f"[voice] turn: stt={stt_ms:.0f}ms llm={llm_ms:.0f}ms tts={tts_ms:.0f}ms "
              f"total={stt_ms + llm_ms + tts_ms:.0f}ms", file=sys.stderr)
        return out
