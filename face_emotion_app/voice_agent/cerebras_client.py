"""Cerebras chat client (OpenAI-compatible) with a tool-calling loop.

Cerebras is not an MCP client; this runs the standard function-calling loop and
lets the ToolBus dispatch each call (to local vision tools or MCP servers). The
OpenAI SDK is imported lazily so the rest of the package imports without it."""
import json
import sys
import time

from . import config


class Truncated(Exception):
    """The token budget ran out before the model emitted a reply. A config fault:
    it must never reach the user as 'Sorry, could you say that again?'."""


def _status(e):
    return (getattr(e, "status_code", None)
            or getattr(getattr(e, "response", None), "status_code", None))


def is_rate_limit(e):
    """429 only. Substring-matching 'rate' also matches 'failed to gene(rate)'."""
    return _status(e) == 429 or type(e).__name__ == "RateLimitError"


def is_auth_error(e):
    """A rejected key is a config fault, not a bad turn -- surface it, don't apologize."""
    return (_status(e) in (401, 403)
            or type(e).__name__ in ("AuthenticationError", "PermissionDeniedError"))


class CerebrasClient:
    def __init__(self, model=None, api_key=None, base_url=None):
        from openai import OpenAI  # lazy: only needed to actually talk to Cerebras
        self.model = model or config.CEREBRAS_MODEL
        self.client = OpenAI(
            base_url=base_url or config.CEREBRAS_BASE_URL,
            api_key=api_key or config.require_cerebras_key(),
            max_retries=0,      # don't let the SDK silently sleep ~60s on a free-tier 429
            timeout=30,
        )

    def keepalive(self):
        """Keep the pooled HTTPS connection to Cerebras hot.

        The first request after an idle gap pays DNS, TCP and a TLS handshake
        again, which shows up as a few hundred extra milliseconds on exactly the
        turn a user notices most: the first thing they say after a pause. Listing
        models is a cheap, untokened request that keeps the socket alive.
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False       # the next real turn will re-establish it anyway

    def _create(self, messages, tools, retries=1):
        """Chat completion with ONE quick retry on a rate limit, then give up fast
        (free tier ~5 rpm). Better a quick 'one sec' than a 60s freeze."""
        kw = {}
        if config.REASONING_EFFORT:
            kw["reasoning_effort"] = config.REASONING_EFFORT
        # Cerebras rejects tool_choice/parallel_tool_calls unless tools are sent,
        # so omit them entirely on a toolless call (e.g. the startup warm-up).
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
            kw["parallel_tool_calls"] = True
        for attempt in range(retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                    **kw,
                )
            except Exception as e:
                if not is_rate_limit(e) or attempt == retries:
                    raise
                time.sleep(2.0)
        raise RuntimeError("unreachable")

    def run(self, messages, tools, dispatch, identity=None,
            max_rounds=config.MAX_TOOL_ROUNDS, on_tool=None):
        """Drive one assistant turn to completion. Returns (reply_text, messages, trace).

        messages : running chat history (list of dicts); mutated in place.
        tools    : merged OpenAI tool schemas (from ToolBus.schemas()).
        dispatch : callable(name, args, identity) -> JSON-able result (ToolBus.dispatch).
        """
        trace = []
        for _ in range(max_rounds):
            resp = self._create(messages, tools)
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            # append the assistant message (with any tool_calls) to history
            assistant = {"role": "assistant", "content": msg.content or ""}
            if calls:
                assistant["tool_calls"] = [{
                    "id": c.id, "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                } for c in calls]
            messages.append(assistant)

            if not calls:
                text = (msg.content or "").strip()
                if not text and resp.choices[0].finish_reason == "length":
                    # Reasoning consumed the whole budget before any reply was emitted.
                    # Say so loudly: upstream this looks identical to "didn't hear you".
                    raise Truncated(
                        "reply truncated -- reasoning used the entire "
                        f"max_completion_tokens={config.MAX_COMPLETION_TOKENS} budget. "
                        "Raise VOICE_MAX_COMPLETION_TOKENS or lower VOICE_REASONING_EFFORT.")
                return text, messages, trace

            for c in calls:
                name = c.function.name
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = dispatch(name, args, identity)
                trace.append({"tool": name, "args": args, "result": result})
                if on_tool:
                    on_tool(name, args, result)
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": json.dumps(result, default=str)})
        # Rounds exhausted with tool results already in hand. The prescribed
        # registration flow (enroll_face + train_emotion x4) is exactly MAX_TOOL_ROUNDS
        # rounds, so bailing out here apologizes for work that actually SUCCEEDED and
        # invites the user to redo it. Let the model speak once more without tools.
        try:
            resp = self._create(messages, None)
            text = (resp.choices[0].message.content or "").strip()
            if text:
                messages.append({"role": "assistant", "content": text})
                return text, messages, trace
        except Exception as e:
            print(f"[voice] final summary call failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
        return "Sorry, I got a bit tangled up there. Could you say that again?", messages, trace
