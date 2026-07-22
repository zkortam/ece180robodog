#!/usr/bin/env python3
"""Phase 0 smoke test: prove a Cerebras tool-calling round-trip works with your
key, before wiring audio. Uses a dummy tool the model must call.

  export CEREBRAS_API_KEY=csk-...
  python scripts/smoke_cerebras.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_agent import config
from voice_agent.cerebras_client import CerebrasClient

DUMMY_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]


def dispatch(name, args, identity=None):
    print(f"  [tool] {name}({args})")
    if name == "get_weather":
        return {"city": args.get("city"), "temp_c": 21, "sky": "clear"}
    return {"error": "unknown tool"}


def main():
    config.require_cerebras_key()
    client = CerebrasClient()
    print(f"model = {client.model}")
    messages = [
        {"role": "system", "content": "You are terse. Use tools when needed."},
        {"role": "user", "content": "What's the weather in San Diego? One sentence."},
    ]
    reply, _, trace = client.run(messages, DUMMY_TOOLS, dispatch)
    print("\ntool trace:", json.dumps(trace, indent=2))
    print("\nfinal reply:", reply)
    if trace and reply:
        print("\nPASS: streamed a tool-calling round-trip end to end.")
    else:
        print("\nWARN: no tool call or empty reply — check model/tool support.")


if __name__ == "__main__":
    main()
