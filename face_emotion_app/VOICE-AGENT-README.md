# Voice Agent — runbook

A voice-first assistant that can **see you**: it listens hands-free, transcribes your
speech, asks **Cerebras** (with tools wired to the face/emotion pipeline so it
knows who's in view and how they feel), and speaks the reply. Half-duplex,
hands-free, board-ready. Full design: [`../VOICE-AGENT-ARCHITECTURE.md`](../VOICE-AGENT-ARCHITECTURE.md).

## What's here

```
vision_service.py            always-on perception loop over face_emotion.py + the 8 LLM tools
voice_agent/
  config.py                  all settings + env overrides (key read from CEREBRAS_API_KEY only)
  tools.py                   the 8 read-only vision tool schemas + dispatch (Tier 0)
  tool_bus.py                merges local + MCP tools, dispatch-by-origin, identity-gate policy
  cerebras_client.py         OpenAI-compatible Cerebras client + tool-calling loop
  stt.py                     STT: faster-whisper (Mac) / moonshine (board)
  tts.py                     TTS: say (Mac) / piper (board) / espeak (fallback)
  orchestrator.py            VoiceAgent: one turn = audio -> STT -> LLM(tools) -> TTS -> audio
  web.py                     Flask server + the hyper-minimal push-to-talk orb UI
  main.py                    entrypoint
  vision_mcp_server.py       Tier 2: expose the SAME vision tools over MCP (optional)
scripts/
  install_voice.sh           venv + deps + models (macOS and ARM Debian)
  run_voice.sh               launch
  smoke_cerebras.py          Phase 0: prove the LLM tool loop with your key
```

Registration (`face_emotion.py` enroll + emotion-train) is **untouched**; this
reads its DBs and picks up new enrollments via `VisionService.reload_db()`.

## Run it (Mac dev)

```bash
cd face_emotion_app
./scripts/install_voice.sh                 # venv + openai + faster-whisper (say is built-in)
export CEREBRAS_API_KEY=csk-...             # your key — keep it out of the repo
python scripts/smoke_cerebras.py            # Phase 0: confirm the tool-calling round-trip
./scripts/run_voice.sh                      # opens on http://127.0.0.1:8100
```

Open the page and **click the face once** to grant mic/camera and start it. After
that it is **hands-free — there is no button to hold and no key to press**. Just
talk; it detects when you stop, thinks, and speaks back, then listens again. Ask
"who do you see?", "how do I look?", "how have I been feeling lately?". Click the
face again to pause. The line under the face always says what it is doing
(listening / hearing you / thinking / speaking) and shows any error.

Enroll people first with the existing app (`python face_emotion.py web`) so it can
name them.

Useful flags: `./scripts/run_voice.sh --no-camera` (UI only), `--owner zakaria`
(identity-gate sensitive actions), `--stt faster-whisper --tts say`.

## Run it (Arduino UNO Q)

```bash
./scripts/install_voice.sh                  # installs Moonshine STT + fetches a Piper voice
sudo apt install -y espeak-ng               # TTS fallback
export CEREBRAS_API_KEY=csk-...
./scripts/run_voice.sh --host 0.0.0.0       # STT=moonshine, TTS=piper auto-selected
```

Attach a **USB headset** (mic+speaker) or open the page from a laptop/phone
browser (recommended — free echo cancellation). Add a heatsink. Keep vision at
`--fps 4`. See the architecture doc §2 (board budget) and §4.5 (audio).

## What is verified vs. what needs your key/hardware

- **Verified working now:** the vision loop + all 8 tools (real models), the tool
  bus + identity-gating, the full turn loop (LLM+STT mocked, **TTS + tools + web
  real**), the entrypoint, and the UI serving.
- **Needs your key:** the real Cerebras call (`smoke_cerebras.py` proves it once
  `CEREBRAS_API_KEY` is set).
- **Needs install:** real STT (`faster-whisper` on Mac / `moonshine` on the board)
  and, on the board, the Piper binary + voice.

## Notes

- The key is read from the environment only, never hardcoded. Rotate any key
  shared in plaintext.
- Free Cerebras tier is ~1 turn / 24 s (too slow for fluid chat); the ~$10
  Developer tier is effectively required for real use.
- MCP external integrations (calendar, home automation, memory) are designed in
  §13 of the architecture doc; `tool_bus.py`/`vision_mcp_server.py` are the seams.
