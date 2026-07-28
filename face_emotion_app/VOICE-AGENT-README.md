# Voice Agent — runbook

A voice-first assistant that can **see you**: it listens hands-free, transcribes your
speech, asks **Cerebras** (with tools wired to the face/emotion pipeline so it
knows who's in view and how they feel), and speaks the reply. Half-duplex,
hands-free, board-ready. Full design: [`../VOICE-AGENT-ARCHITECTURE.md`](../VOICE-AGENT-ARCHITECTURE.md).

## What's here

```
vision_service.py            always-on perception loop over face_emotion.py + the LLM tools
voice_agent/
  config.py                  all settings + env overrides (key read from CEREBRAS_API_KEY only)
  tools.py                   the 10 vision tool schemas + dispatch (Tier 0)
  tool_bus.py                merges local + MCP tools, dispatch-by-origin, identity-gate policy
  cerebras_client.py         OpenAI-compatible Cerebras client + tool-calling loop
  stt.py                     STT: faster-whisper (Mac) / moonshine (board)
  tts.py                     TTS: kokoro (Mac) / piper (board) / say + espeak (fallbacks)
  orchestrator.py            VoiceAgent: one turn = audio -> STT -> LLM(tools) -> TTS -> audio
  board_audio.py             board-native USB mic/speaker loop (no browser needed)
  web.py                     Flask server + the hands-free animated-face UI
  main.py                    entrypoint
  vision_mcp_server.py       Tier 2: expose the read-only vision tools over MCP (optional)
scripts/
  install_voice.sh           venv + deps + models (macOS and ARM Debian)
  run_voice.sh               launch
  smoke_cerebras.py          Phase 0: prove the LLM tool loop with your key
  bench_latency.py           per-stage latency, per backend
```

Eight of the ten tools only read perception state. The other two — `enroll_face`
and `train_emotion` — write biometric data to the local databases, so they are
registered as `RISK_WRITE` in `tool_bus.py` and are not exposed over MCP.

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

Useful flags: `--browser-camera` (frames come from the page — required on macOS,
where the server cannot share the camera with the browser), `--no-camera` (UI
only), `--owner zakaria` (identity-gate sensitive actions), `--stt faster-whisper
--tts say`. Unknown backend names are rejected at startup rather than on the first
spoken turn.

## Run it (Arduino UNO Q)

There are two deployment shapes, and it matters which one you are in.

### Standalone robot — the board is the whole machine

The UNO Q listens on its own USB microphone, speaks through its own USB speaker,
and sees through its own USB webcam. No laptop, no browser, no cable. **This is
the mode to use on a robot that has to work when nothing else is plugged in.**

#### Wiring it: what the USB hub changes

Camera, microphone and speaker all reach the board through one hub, and that has
three consequences worth knowing before you blame the software.

**Power first.** A webcam and a powered speaker together can draw more than the
board's host port is willing to supply. An under-fed hub does not fail cleanly —
devices brown out and re-enumerate at random, which looks exactly like flaky code:
the camera "randomly stops", the microphone "randomly goes deaf". **Use a hub with
its own power supply,** and make sure the board itself still has a power source
once the laptop is gone — if your dongle occupies the same USB-C port the laptop
was powering the board through, it needs power-delivery passthrough or the board
needs separate power. Nothing in software can compensate for this.

**Nothing is at a fixed address.** Which port you used, and how fast each device
powers up, decides the ALSA card numbers and the `/dev/video*` indexes — and they
move between boots. So none of it is hardcoded:

- The camera is found by asking the kernel which `/dev/video*` nodes are real
  capture devices (this SoC's hardware video encoder and decoder claim nodes there
  too), then confirming one actually delivers a frame. Unplug it mid-conversation
  and the watcher re-finds it, at whatever index it comes back on.
- Audio devices are chosen **by name, not by index** — the old code took whichever
  card enumerated first, so the microphone the robot listened on changed depending
  on which port you used. The board's own codec, which has nothing wired to it, is
  excluded.

If it still picks the wrong one, the log prints every device it saw. Pin your
choice by name (survives re-plugging, unlike a card number):

```bash
VOICE_CAPTURE_MATCH='C270'          # substring of the mic's name
VOICE_PLAYBACK_MATCH='USB Audio'    # substring of the speaker's name
VOICE_CAPTURE_DEVICE='plughw:2,0'   # or pin the exact ALSA device
VOICE_EXCLUDE_AUDIO='ArduinoImola,HDMI'   # never select these
```

**Shared bandwidth.** The camera and the audio stream cross the same bus, and
starving `arecord` is what produces "microphone stopped returning audio". The
camera is therefore asked to *produce* at the configured rate (4 fps, 320x240)
rather than streaming 30 fps that gets thrown away — which also stops the robot
answering about a frame from several hundred milliseconds ago.

Before you unplug the laptop, prove it end to end:

```bash
python scripts/preflight.py --speak
```

```bash
./scripts/install_voice.sh                  # Moonshine STT + a Piper voice
sudo apt install -y espeak-ng               # TTS fallback

mkdir -p ~/.robodog && chmod 700 ~/.robodog
printf 'CEREBRAS_API_KEY=csk-...\n' > ~/.robodog/env && chmod 600 ~/.robodog/env

sudo cp systemd/robodog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now robodog.service
journalctl -u robodog -f                    # watch it come up
```

The unit restarts on crash (rate-limited to 5 failures/minute so a genuine
configuration fault stops and leaves readable evidence rather than spinning), and
binds HTTP to loopback because the robot's interface is its microphone, not its
network port.

It is **mutually exclusive with `uno-face-emotion.service`**: that unit runs the
CLI recognizer, which opens the camera exclusively. Two processes on the same
`/dev/video*` node do not share it — one gets frames and the other silently gets
none. `Conflicts=` enforces this, and `run_board_face.sh` refuses to start if the
voice agent is already running.

To run it in the foreground instead:

```bash
export CEREBRAS_API_KEY=csk-...
./scripts/run_voice.sh --host 127.0.0.1 --board-audio
```

`--board-audio` waits for a USB capture and playback device, discovers the camera
across `/dev/video*` (and keeps retrying after an unplug), and answers out loud
with no page open. Faults it cannot recover from — a rejected key, an exhausted
token budget — are **spoken aloud once**, because a robot that simply goes quiet
is indistinguishable from one that is broken or ignoring you.

### Tethered development — laptop supplies camera, mic and speaker

`scripts/board_daemon.py` keeps `adb forward` alive and runs the agent with
`--browser-camera`, so the page in your laptop browser provides the hardware and
gets free echo cancellation. Convenient for iterating; **it is not the robot.**

```bash
python scripts/board_daemon.py              # then open the hosted UI
```

Services bind to loopback by default (`adb forward` reaches the device's own
loopback, so nothing is lost). Set `BOARD_BIND=0.0.0.0` only deliberately: there
is **no authentication**, and port 8000 can enroll and delete biometric data.

Add a heatsink. Keep vision at `--fps 4`. See the architecture doc §2 (board
budget) and §4.5 (audio).

## Tests

```bash
.venv-voice/bin/python -m pytest        # ~150 tests, a couple of seconds
uvx ruff check .
```

No hardware and no API key required: `VisionService.step()` takes a plain BGR
array, and the web tests drive the real Flask app with a fake agent. Tests that
need the ONNX weights skip cleanly when `models/` has not been downloaded.

Several tests are named regressions — they encode a failure that actually
happened (a camera open that stranded the robot blind, an enrollment delete that
erased someone else, a noise filter that swallowed "yeah"). Those are the ones to
keep if anything is ever cut.

## What is verified vs. what needs your key/hardware

- **Verified working now:** the vision loop + all 10 tools (real models), the tool
  bus + identity-gating, the full turn loop (LLM mocked, **STT + TTS + tools + web
  real**), the entrypoint, and the UI serving.
- **Needs your key:** the real Cerebras call (`smoke_cerebras.py` proves it once
  `CEREBRAS_API_KEY` is set).
- **Needs install:** real STT (`faster-whisper` on Mac / `moonshine` on the board)
  and, on the board, the Piper binary + voice.
- **Needs the board:** the USB audio loop (`board_audio.py`) and camera discovery
  across `/dev/video*`. Both are Linux/ALSA paths with no macOS equivalent, so
  they are exercised on hardware, not in CI.

## Adding the legs (servo motion)

Motion belongs behind the **same tool bus** as vision — `tool_bus.py` already
dispatches by owner and enforces a per-tool risk tier, so a `walk_forward` tool
reaches the model exactly the way `who_is_in_view` does. Three constraints are
not optional, because a turn and a limb have very different timing:

1. **Motion tools must return immediately.** A conversation turn holds
   `agent.turn_lock` from STT through TTS. A tool that blocks until the robot has
   finished walking holds that lock for the whole gait, and everything else —
   including the microphone loop — waits behind it. Start the motion, return
   `{"status": "walking"}`, and expose a separate cheap `motion_status` tool the
   model can poll. (`enroll_face` is the existing example of a slow tool, and it
   is capped at a 25 s timeout for exactly this reason.)

2. **Motion needs a hardware deadman independent of Python.** Every failure in
   this stack currently degrades to silence, which is safe for a speaker and
   unsafe for a limb. The servo layer must stop on its own if it does not receive
   a heartbeat — a wedged turn, an OOM kill, or a `SystemExit` in a worker thread
   must not leave a leg driving. Do not rely on a `finally:` block for this.

3. **Register motion as `RISK_SENSITIVE`, not `RISK_WRITE`.** The identity gate in
   `Policy.check()` is already written and currently inert because no tool
   declares that tier: it requires the enrolled owner in view above
   `IDENTITY_TAU_HIGH` before it will authorize a call. Movement is the first
   thing that genuinely warrants it. Pass `--owner <name>` to arm it.

The perception side already gives motion what it needs: `describe_scene()`
returns each face's `position` and `size_frac`, which is enough to turn toward a
speaker or follow someone without any new sensing.

## Notes

- The key is read from the environment only, never hardcoded. Rotate any key
  shared in plaintext.
- Free Cerebras tier is ~1 turn / 24 s (too slow for fluid chat); the ~$10
  Developer tier is effectively required for real use.
- MCP external integrations (calendar, home automation, memory) are designed in
  §13 of the architecture doc; `tool_bus.py`/`vision_mcp_server.py` are the seams.
