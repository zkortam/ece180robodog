# RoboDog Voice and Vision Architecture — Arduino UNO Q

**Target board:** Arduino UNO Q (ABX00173) — Qualcomm Dragonwing **QRB2210**, quad Cortex-A53
@ 2.0 GHz nominal (throttles toward ~1.5 GHz under sustained all-core load without a heatsink),
**4 GB LPDDR4X**, 32 GB eMMC, ARM Debian 12, **CPU-only** (no usable GPU/NPU compute path).
**Mac = dev box only.** LLM = **Cerebras** cloud (off-board). Everything else runs on the board.

---

## 1. Vision

A voice-first assistant that you talk to and that talks back — no screen, no keyboard, no chat box —
which can also *look at you*: it knows who is in front of the camera, what each person feels right now,
and how that has drifted over the last couple of minutes. The board captures speech, transcribes it
locally, sends the text to Cerebras' `gpt-oss-120b` with a set of **vision tools** wired to the existing
YuNet/SFace/MobileFaceNet pipeline, streams the reply back through a local neural TTS, and speaks it —
all as a **turn-taking (half-duplex)** loop that runs entirely on four Cortex-A53 cores while an
always-on vision thread keeps perception fresh. The existing **registration pipeline (face enroll +
personal-emotion training) is untouched and stays separate**; face detection/recognition/emotion are
**repurposed as read-only tools the LLM calls**, never rewritten.

---

## 2. Board budget (the load-bearing section)

The whole design is dictated by one fact: **4× in-order A53 cores, CPU only, no accelerator.** The A53 is
roughly Raspberry-Pi-3-class *per core* (same µarch, higher clock), i.e. ~2.5–3× slower per core than the
Pi-5 (A76) that most "runs real-time on a Pi" benchmarks assume. So we (a) keep every local model tiny,
(b) run the LLM in the cloud, and (c) **never run STT and TTS at the same time**.

### 2.1 CPU — core occupancy per conversation phase

| Core | LISTEN (user talking) | THINK (waiting on Cerebras) | SPEAK (agent talking) |
|---|---|---|---|
| **0** | Orchestrator (asyncio) + Silero VAD | Orchestrator + VAD | Orchestrator + VAD (**barge-in watch**) |
| **1** | Moonshine STT | idle (network-bound) | Piper TTS |
| **2** | Moonshine STT | idle (network-bound) | Piper TTS |
| **3** | **Vision loop (always-on)** | **Vision loop** | **Vision loop** |

**Why this fits:** STT (LISTEN) and TTS (SPEAK) are *temporally disjoint* under half-duplex, so cores 1–2
are time-shared between them and never contended at once. Vision owns core 3 permanently and never
competes with STT/TTS. VAD (~1–2 MB, sub-ms per 32 ms frame) and the asyncio orchestrator are light
enough to share core 0. **That temporal disjointness is the entire reason 4 cores is enough.**

### 2.2 RAM — comfortable, not the bottleneck

| Component | Resident (approx) |
|---|---|
| Debian 12 + Python 3 baseline | ~0.7 GB |
| OpenCV + vision (YuNet+SFace+MobileFaceNet via `cv2.dnn`) + frame buffers | ~0.5 GB |
| Moonshine v2 tiny STT (onnxruntime + 26 M model) | ~0.25 GB |
| Silero VAD (shares onnxruntime) | ~0.02 GB |
| Piper TTS subprocess (low/x-low voice) | ~0.1 GB |
| Orchestrator + Flask/WebSocket + Cerebras HTTPS client | ~0.15 GB |
| **Peak total** | **~1.7 GB of 4 GB** (~2 GB headroom) |

**RAM is not the constraint; sustained CPU (and heat) is.**

### 2.3 Verdict — **HALF-DUPLEX turn-taking, with push-to-talk as the guaranteed fallback**

- **Full-duplex (listen while speaking): NOT feasible.** It requires continuous STT + streaming TTS +
  vision + acoustic echo cancellation *simultaneously* on 4 A53 cores. STT alone saturates 1–2 cores;
  add TTS+vision+AEC and every stage drops below real-time. **Rejected.**
- **Half-duplex + VAD-only barge-in: RECOMMENDED.** Listen → think → speak, but keep only the cheap
  Silero VAD alive during SPEAK so the user can interrupt; a confirmed barge-in kills TTS and returns to
  LISTEN. This buys ~90% of the full-duplex feel for ~0% of the extra CPU.
- **Push-to-talk: the safe demo path / degraded mode.** A button (browser or board GPIO) defines the
  listen window; release = end-of-turn. It removes the silence-endpoint wait *and* the echo problem
  (mic is closed while speaking), so it always works even on the board-native audio path. **Ship PTT
  first; add VAD barge-in as the stretch goal.**

**Thermal caveat (honest):** sustained all-4-core load (only happens briefly during STT bursts here)
heat-soaks a bare board and DVFS-throttles below 2.0 GHz within minutes. **Budget a heatsink + a little
airflow.** The half-duplex design already avoids pinning all four cores continuously, which is the main
thermal mitigation; vision on one core + bursty STT is the steady state, not a 4-core grind.

---

## 3. Component diagram + one-turn data flow

```
              CLIENT (browser: laptop/phone — primary)          BOARD  (4× A53, CPU-only)                       CLOUD
   ┌──────────────────────────────────────────────┐  ┌────────────────────────────────────────────────┐  ┌──────────────┐
   │ getUserMedia(audio)                           │  │                                                │  │  Cerebras    │
   │  ├─ WebRTC AEC3 echo-cancel ┐                 │  │  ┌──────── ORCHESTRATOR (asyncio, core 0) ────┐ │  │  Inference   │
   │  ├─ noise suppression        │ 16 kHz PCM     │WS│  │  state machine:                            │ │  │  gpt-oss-120b│
   │  └─ auto-gain               ▼  (mono)         ├─►│  │  IDLE→LISTEN→ENDPOINT→THINK→SPEAK           │ │  │  (OpenAI-    │
   │ MediaRecorder / WS ──────────────────────────►│  │  └──┬─────────┬──────────┬───────────┬───────┘ │  │  compatible, │
   │                                               │  │     │ audio   │ text     │ tokens    │ tool    │  │  tool calls) │
   │ <audio> playback ◄────────────────────────────┤◄─┤     ▼         ▼          ▼           ▼ calls   │  └──────┬───────┘
   │  (TTS PCM streamed back)                      │WS│  ┌───────┐ ┌────────┐ ┌───────┐ ┌──────────┐    │         │
   │                                               │  │  │Silero │ │Moonshine│ │ Piper │ │  TOOLS   │   │ HTTPS   │
   │ getUserMedia(video) ─── 320×240 JPEG ────────►│  │  │ VAD   │ │  tiny   │ │  TTS  │ │(tools.py)│   │◄─stream─┘
   │  (2–5 fps, downscaled) [OPTIONAL]             │  │  │core0  │ │core1-2  │ │core1-2│ └────┬─────┘   │  SSE tokens
   └──────────────────────────────────────────────┘  │  │thread │ │thread   │ │subproc│      │ read    │  + tool_calls
                                                      │  └───┬───┘ └────┬────┘ └───▲───┘      ▼ (locked) │
   ── OR board-native audio (fallback) ──             │      │barge-in  │ final    │ sentence ┌──────────┐│
   USB headset ─ ALSA card ─ sounddevice ───────────► │      │stop      ▼ text     │ chunks   │VisionState│
   (mic in + speaker out, one USB-Audio device)       │      └──────────────────────┴─────────┤ (locked) ││
                                                      │                                        │ ring buf ││
                                                      │  ┌─────────────────────────────────────┤ events   ││
                                                      │  │  VISION LOOP (core 3, always-on)     │ people   ││
                                                      │  │  VisionService over face_emotion.py: │ current  ││
                                                      │  │  YuNet detect → SFace ID (128-d) →   ├──writes──┘│
                                                      │  │  MobileFaceNet emotion (every Nth)   │ 2–5 fps   │
                                                      │  └──────────────────────────────────────┘           │
                                                      └────────────────────────────────────────────────────┘
```

**One full turn (half-duplex):**

1. **Capture** — mic audio (browser AEC, or board USB mic) → 16 kHz mono PCM → orchestrator.
2. **VAD gate (core 0)** — Silero VAD on 512-sample / 32 ms frames; speech onset → phase = `LISTEN`.
3. **STT (cores 1–2)** — Moonshine v2 tiny streams **partial** transcripts (<200 ms) as you speak.
4. **Endpoint** — ~500–700 ms of contiguous silence (or PTT release) → phase = `ENDPOINTING` → final transcript.
5. **THINK** — orchestrator appends transcript to `ConvState.history`, calls Cerebras `gpt-oss-120b`
   with `stream=True` + the tool schemas. If the model emits `tool_calls`, `tools.py` answers each from
   `VisionState` under the lock (microseconds), appends `role:"tool"` messages, and calls Cerebras again.
6. **SPEAK (cores 1–2)** — streamed reply tokens are split on sentence/clause boundaries; each finished
   chunk is handed to Piper immediately → PCM → back to the speaker. **First sentence plays while the LLM
   is still generating the rest.** Silero VAD stays alive on core 0 for barge-in.
7. Barge-in or end-of-utterance → phase = `IDLE`/`LISTEN`.

**Always-on vision loop (core 3), independent of the turn:** grabs frames at 2–5 fps, runs YuNet over
**every** face, SFace-embeds + identifies each, runs the emotion ONNX every Nth frame per track, and
writes observations + ring-buffer history + presence events into `VisionState`. Tool calls read a
snapshot of this state; they never drive the camera.

---

## 4. Concrete tech stack (board-first, with specific picks and why)

### 4.1 LLM — **Cerebras `gpt-oss-120b`** (cloud)

- **Pick:** `gpt-oss-120b` (117–120 B MoE, 131 K context, ~3,000 tok/s, **production**, native tool
  calling incl. `strict` mode). It is the fastest model in Cerebras' *current* catalog and the clear
  default for a low-latency tool-calling voice agent.
- **Why not the alternatives:** `zai-glm-4.7` (~1,000 tok/s) and `gemma-4-31b` (preview) are slower and
  offer no advantage for short conversational tool-calling turns. **The models named in older docs
  (`llama-3.3-70b`, `qwen-3-32b`, Llama-4 Scout/Maverick, `qwen-3-235b`) are DEPRECATED** — Cerebras'
  own recommended replacement is `gpt-oss-120b`.
- **Access:** OpenAI SDK pointed at Cerebras, or `cerebras-cloud-sdk`. **Key from `CEREBRAS_API_KEY`
  env var only** — never hardcoded, never sent to the browser (the key lives on the board).
  ```python
  from openai import OpenAI          # or: from cerebras.cloud.sdk import Cerebras
  client = OpenAI(base_url="https://api.cerebras.ai/v1",
                  api_key=os.environ["CEREBRAS_API_KEY"])   # keys are prefixed csk-...
  ```
- **Streaming + tools:** `stream=True`; accumulate `delta.tool_calls[i].function.arguments` fragments
  by `.index`, finalize on `finish_reason == "tool_calls"`, `json.loads` the arguments string, run the
  tool, append `{"role":"tool","tool_call_id":..., "content": json.dumps(result)}`, call again.
- **Gotcha:** on `gpt-oss-120b` you may **not** send `tools` and `response_format` together — use tools
  only (we do). Set `parallel_tool_calls=True` so one round-trip can batch e.g. `who_is_in_view` +
  `get_person_emotion`.
- **Latency:** TTFT ~170–240 ms; generation is effectively instant at 3,000 tok/s, so the LLM leg is
  **RTT-dominated, not the bottleneck** — use streaming so TTS starts on the first tokens.

### 4.2 STT — **Moonshine v2 tiny (streaming)** on the board; faster-whisper on the Mac

- **Board pick:** **Moonshine v2 tiny, streaming variant** (26 M params, ~34 MB, **MIT**, runs on
  **onnxruntime**). It is **streaming-first by design** — sliding-window attention, no 30 s Whisper
  window, sub-200 ms partials — which is exactly what near-real-time turn-taking needs. Realistic RTF on
  1–2 shared A53 cores ≈ **0.2–0.4** (well under real-time), and tiny leaves CPU for vision + TTS.
- **Board fallback:** **Vosk-small (`vosk-model-small-en-us`, Apache-2.0)** — ~50 MB / ~300 MB RAM,
  genuine word-by-word streaming, battle-tested on Pi-3/4-class ARM. Lower accuracy on hard audio but
  rock-solid latency. Use it if Moonshine's WER is too high for the vocabulary.
- **Rejected for the board:** whisper.cpp `base`+ (RTF > 2 on A53); **faster-whisper** (drags in
  Python/PyTorch-class RAM, chunked not streaming) — a Mac tool, not a board tool; **distil-whisper**
  (166 M, 5–10 s latency floor). whisper.cpp `tiny.en` (`-ac 512 -t 2`) is *arguable* but drifts past
  real-time under vision+TTS contention, and it isn't truly streaming — Moonshine wins on both counts.
- **Mac dev:** run **Moonshine** too (same models/code, a bigger variant for reference), and keep
  **faster-whisper small/medium** (or whisper.cpp CoreML/Metal) purely as a **high-accuracy ground-truth
  transcriber** to measure the board's Moonshine/Vosk WER against. Do **not** ship these to the board.

### 4.3 TTS — **Piper (low/x-low voice)** on the board; Kokoro on the Mac

- **Board pick:** **Piper** with a **low or x-low 16 kHz voice** (e.g. `en_US-*-low`). It is the only
  neural TTS in this class *designed for and proven on low-power ARM CPUs*: native **ONNX**, tens-of-MB
  footprint, **RTF ≈ 0.2–0.5** on A53 for low voices, and it **streams raw PCM to stdout sentence-by-
  sentence**, so perceived latency ≈ time-to-first-chunk, not full-utterance time.
- **Board fallback:** **espeak-ng** (`apt install espeak-ng`) — formant synth, robotic but instant and
  always-works; Piper already uses it internally for phonemization, so it's on the board anyway. Drop to
  it under thermal/CPU pressure.
- **License note:** current Piper is **GPL-3.0** — run it as a **separate subprocess/binary we shell out
  to** (which we do for streaming anyway), not statically linked. The MIT `piper-plus` fork is the escape
  hatch if GPL is a problem for productization.
- **Rejected for the board:** **Kokoro-82M** (RTF ~0.5 *on a desktop* → several seconds/sentence on
  A53 — too slow interactively), Chatterbox/XTTS (GPU-oriented, heavy; XTTS is additionally
  **non-commercial** and Coqui is defunct — avoid).
- **Mac dev:** **Kokoro (kokoro-onnx, 82 M, Apache-2.0)** for best-per-MB quality prototyping/comparison
  (same ONNX stack), understanding it won't hit interactive latency on the board.

### 4.4 VAD — **Silero VAD tiny** (MIT, ONNX)

512-sample / 32 ms chunks at 16 kHz; ~1–2 MB, sub-ms per frame. Gates the recognizer so STT only runs on
speech (saves CPU for vision+TTS), drives the ~500–700 ms silence endpoint, and stays alive during SPEAK
for barge-in (require ~150–200 ms of voiced frames to reject a cough/echo blip). **Fallback:** `webrtcvad`
(BSD, lighter, less accurate) only if the absolute-minimum footprint is needed.

### 4.5 Audio transport + physical attach — **browser front-end primary, board-native USB fallback**

This is a genuine decision (see §13). Both keep *all AI compute on the board*; they differ only in where
the mic/speaker live and where echo cancellation runs.

- **Primary — browser over WebSocket (recommended).** Extend the **existing** Flask + `getUserMedia`
  camera page (already in `face_emotion.py`'s `HTML_PAGE`). Request
  `getUserMedia({audio:{echoCancellation:true, noiseSuppression:true, autoGainControl:true}})` — this
  gives a **hardware-grade WebRTC AEC3 echo canceller for free, running on the client's CPU**, and moves
  **capture + AEC + playback off the 4 A53 cores entirely**. That offload is the single biggest reason
  the board budget closes. The board runs STT/TTS/orchestration/vision/tools; the browser is just a thin
  audio+video I/O terminal. (AEC3 needs ~1–2 s to converge — ramp mic-in at session start.)
- **Fallback — native ALSA on the board.** Plug a **single USB Audio Class headset or soundcard** into
  the board's USB port; it enumerates as a standard **ALSA card** (alongside the placeholder
  `ArduinoImolaHPH` device). Use **one** device for both mic in and speaker out. Capture/playback via
  `sounddevice`/ALSA. **Pin the card** by `/dev/snd/by-id` or an `.asoundrc`/udev rule — ALSA card
  numbers reorder across reboots. This path forces you to solve echo yourself: either run **WebRTC APM /
  SpeexDSP AEC** on the board (costs part of a core) *or* — the pragmatic choice — use **push-to-talk**
  (browser button or a board GPIO button via the STM32U585) to close the mic while speaking and sidestep
  AEC entirely. There is **no onboard codec/mic/speaker and no exposed line-audio I2S** — USB audio is
  the supported path, so a USB headset is the physical answer for the self-contained build.

---

## 5. The tool set the LLM gets (grounded in the current models)

All tools are **read-only perception** over `VisionService` (which wraps a `FaceEngine`). Registration is
**not** a tool. Every field traces to real output of YuNet (`face[:4]` bbox, `face[14]` det score,
`face[4:14]` landmarks), SFace (`best_match` → name + cosine `score` ∈ [-1,1]), and MobileFaceNet
(`EmotionModel.probabilities` → 7-way softmax over `[angry, disgust, fear, happy, neutral, sad,
surprise]`), plus personal prototypes (`classify_personal`) and `sentiment_from_emotion` (positive/
negative/neutral/not_enabled).

| Tool | Params | Returns (key fields) |
|---|---|---|
| `start_watching` | `camera=0`, `fps=4`, `emotion_every=8`, `threshold=0.5` | `{running, already_running, config, started_at}` — starts the vision loop; idempotent |
| `stop_watching` | — | `{running, uptime_seconds, frames_processed}` — stops loop, **keeps** ring buffer/events |
| `list_enrolled` | — | `{people:[{name, face_enrolled, personal_expressions[]}]}` — from `engine.db.keys()` + `engine.emotion_status()`; works with the loop off |
| `who_is_in_view` | `min_identity_score=0.0` | `{known:[{name, identity_score, bbox}], unknown_count, num_faces, as_of, stale_seconds}` — latest frame only, cheap |
| `describe_scene` | `include_probs=false` | `{as_of, frame_width, frame_height, num_faces, people:[{name, identity_score, emotion, emotion_score, emotion_source(personal\|generic\|none), sentiment, position{h,v}, size(small\|medium\|large), size_frac, bbox, probs?}]}` |
| `get_person_emotion` | `name` (req) | `{found, present, emotion, emotion_score, emotion_source, sentiment, probs, sample_age_seconds}` — latest sample from the ring buffer |
| `emotion_timeline` | `name` (req), `since_seconds=60`, `max_points=50` | `{found, sample_count, window_seconds, dominant_emotion, emotion_fractions, sentiment_fractions, series[]}` — **"how was I feeling a minute ago"** |
| `presence_events` | `since_seconds=120` | `{now_present:[name], events:[{t, name, event(enter\|leave), identity_score}]}` — **"did anyone walk in / leave"**, debounced |

Each schema is emitted OpenAI-shaped with `strict:true` and `additionalProperties:false`, e.g.:

```json
{"type":"function","function":{
  "name":"get_person_emotion","strict":true,
  "description":"Current/most-recent emotion for one named person, from the live ring buffer.",
  "parameters":{"type":"object",
    "properties":{"name":{"type":"string","description":"Enrolled identity name."}},
    "required":["name"],"additionalProperties":false}}}
```

**System-prompt guidance** tells the model: prefer `who_is_in_view`/`describe_scene` for "look at me"
questions, `emotion_timeline` for anything temporal ("lately", "a minute ago", "how have I been"),
`presence_events` for enter/leave, and to answer from the **aggregate** a tool returns (dominant emotion
over a window), not a single frame, so low-confidence blips don't mislead it.

---

## 6. Shared state (incl. temporal/history store)

Two objects, **each behind its own lock** so vision writes never block audio reads.

```python
# ===== VisionState — guarded by vision_lock; written by the vision loop @2–5 fps =====
Observation = {              # one face in one frame, all grounded in model outputs
  "track_id": int,           # greedy IoU + identity association across frames
  "name": str,               # best_match() name, or "unknown"
  "identity_score": float,   # SFace cosine (dot of L2-normed 128-d embeddings)
  "bbox": [x,y,w,h],         # ints from face[:4]
  "det_score": float,        # face[14]
  "emotion": str|None,       # personal or generic label
  "emotion_score": float,
  "emotion_source": str,     # "personal" | "generic" | "none"
  "sentiment": str,          # positive|negative|neutral|not_enabled
  "probs": {label: float},   # full 7-way vector (angry..surprise)
  "position": {"h":"left|center|right", "v":"top|middle|bottom"},  # bbox center vs frame
  "size_frac": float, "size_bucket": str,   # small<0.05, medium, large>0.20
  "t": float }               # capture timestamp

ExpressionSample = {"t","emotion","emotion_score","sentiment","source","probs"}  # per ring entry

VisionState = {
  "running": bool, "started_at": float, "frames_processed": int, "fps_est": float,
  "frame_w": int, "frame_h": int,
  "latest": [Observation, ...], "latest_t": float,          # current frame, ALL faces
  "people": { name: {"present","first_seen","last_seen","last_obs"} },   # per-identity live track
  "ring":   { name: deque[ExpressionSample] },  # RING BUFFER, bounded by BOTH
            #   RING_SECONDS≈300 (max age) AND RING_MAX≈600 (max len) → memory capped
  "events": deque[{"t","name","event":"enter|leave","identity_score"}]   # maxlen 64
}
# Presence logic (debounced for low fps): fire "enter" when a person reappears after
# absence > PRESENCE_GAP (1.5 s); fire "leave" when now - last_seen > LEAVE_TIMEOUT (2.0 s).
# unknowns keyed "unknown#<track_id>".

# ===== ConvState — guarded by conv_lock; the turn / barge-in state machine =====
ConvState = {
  "phase": str,                     # IDLE|LISTEN|ENDPOINTING|THINKING|SPEAKING
  "partial_transcript": str, "final_transcript": str,
  "speaking_utterance_id": int,     # ++ each TTS turn; barge-in bumps it to drop late chunks
  "barge_in": threading.Event,
  "history": [ {role, content} ]    # rolling chat, trimmed to a token budget
}
```

The **ring buffer is the substrate for temporal queries**: `emotion_timeline` scans `ring[name]` over the
window and aggregates (dominant emotion + fractions); `presence_events` reads the O(1) `events` deque so
"who *just* walked in" never scans frames. Sized `fps × 300 s` (~600 entries at ~2 fps effective emotion
rate) = a few hundred KB — two-plus minutes of timestamped history, well within budget.

---

## 7. Concurrency + core-affinity model

- **One asyncio orchestrator** (single thread, I/O-bound: WebSocket, Cerebras HTTPS, subprocess pipes)
  coordinates everything; heavy lifting is in native code that **runs off the GIL**.
- **STT (Moonshine):** worker thread using **onnxruntime**, which **releases the GIL** during the forward
  pass (a subprocess is the alternative for hard isolation). Pinned to cores 1–2 during LISTEN.
- **TTS (Piper):** **separate native subprocess** streaming raw PCM to stdout; reuses cores 1–2 during
  SPEAK. (Piper as a subprocess also keeps the GPL boundary clean — see §4.3.)
- **VAD (Silero):** in-process onnxruntime, its own thread on core 0; the only always-spinning audio loop.
- **Vision loop:** dedicated thread; `cv2.dnn`/OpenCV release the GIL during inference. Pinned to core 3.
- **Pinning:** `os.sched_setaffinity(tid, {core})` (Linux) or launch under `taskset`. Bridge the
  audio-callback thread to the loop with `loop.call_soon_threadsafe`; **never do heavy work in an audio
  callback**.
- **Locks:** `vision_lock` and `conv_lock`, each held only for the microseconds to append/copy. **Tool
  calls take a snapshot copy under `vision_lock`, then release** — the LLM's network latency never holds a
  lock. Reuses the existing `threading.Lock` pattern from `FaceEngine`.
- **No busy-waiting:** only VAD and vision spin (both cheap); everything else is event-driven off
  asyncio/queues, so idle cores actually idle (and stay cool).

---

## 8. Per-stage latency budget + how streaming hides it

Time from *user stops talking* → *first audio out of the speaker* (board, half-duplex):

| Stage | Est. latency | Notes |
|---|---|---|
| Endpoint (silence confirm) | 500–700 ms | Silero threshold; the "are you done?" wait. **PTT removes this** (release = endpoint). |
| Finalize STT (Moonshine) | 100–250 ms | partials already computed while you spoke; only the tail finalizes |
| Prompt assemble + tool round-trip | 50–150 ms | tool answered locally from ring buffer (µs); +1 LLM hop only if a tool is called |
| Network to Cerebras + TTFT | 200–400 ms | RTT-dominated; Cerebras TTFT ~170–240 ms |
| First sentence of tokens | 50–150 ms | ~3,000 tok/s; a short first clause arrives fast |
| Piper synth of first chunk (low voice) | 150–400 ms | near real-time on the small voice |
| WS + browser playback buffer | 30–80 ms | |
| **Perceived first-audio latency** | **≈ 1.1 – 2.1 s** | dominated by endpoint wait + network + first-Piper |

**Streaming overlaps every stage — never wait for a stage to finish before starting the next:**

1. **Partial STT** (Moonshine <200 ms partials) lets the orchestrator pre-assemble the prompt.
2. **Streamed LLM tokens** (`stream=True`) — from the board, network RTT dominates, not generation.
3. **Incremental Piper** — split the token stream on `. ? ! ;` or ~8–12 words and synth each finished
   clause immediately, so **the first sentence is playing while the LLM generates the third**. Perceived
   latency collapses to ~(first-sentence LLM time + first-chunk Piper time), independent of reply length.

**Push-to-talk shaves this toward ~0.9–1.5 s** (no silence wait). Full-duplex would *not* beat first-audio
latency — it only helps interruption latency, which the VAD-only barge-in already covers — while costing a
core we don't have. That is the core argument for half-duplex.

---

## 9. Hyper-minimal UI/UX (audio-first)

The interface is **voice**. There is **no chat box, no transcript log, no text input.** We reduce the
existing camera page to the minimum that makes the agent feel alive and "aware":

- **A single presence orb** (reusing the existing page's aesthetic) with four states:
  **idle** = dim/still · **listening** = soft pulse · **thinking** = slow breathe · **speaking** =
  animated to the TTS. That is the whole primary UI.
- **A tiny camera thumbnail** (optional) so you can see that it sees you — the "look at me" affordance.
  It can shrink to a dot; the vision loop runs regardless of whether it's shown.
- **Optional captions** (toggle, default off): live STT partial + the TTS text, for accessibility/noisy
  rooms. Never required to use the agent.
- **On the board-native path**, the UI degrades further to hardware: a **status LED** (presence/state via
  the STM32U585 GPIO) + a **push-to-talk button**. No screen at all.

---

## 10. File / module layout

**Untouched (registration stays intact and separate):** `face_emotion.py` keeps `FaceEngine`, all enroll/
train methods (`enroll_begin/frame/finish`, `emotion_enroll_begin/frame/finish`), the CLI
(`command_enroll`, `command_emotion_enroll`, `command_recognize`, `command_web`), and its Flask routes
(`/api/enroll/*`, `/api/emotion/enroll/*`). The voice agent **imports** its reusable functions/classes;
it does not modify them.

```
face_emotion_app/
├─ face_emotion.py            # UNCHANGED. Registration + FaceEngine + existing web UI.
├─ vision_service.py          # NEW. VisionService: wraps a FaceEngine, owns the vision thread,
│                             #   VisionState, presence tracking, ring buffer, and the 8 tool impls.
│                             #   Reuses create_detector/recognizer, open_camera, detect_faces,
│                             #   face_embedding, best_match, classify_personal, sentiment_from_emotion,
│                             #   EmotionModel, load_db/load_emotion_db. Its loop is command_recognize's
│                             #   body GENERALIZED from largest_face(faces) to a loop over ALL faces.
│                             #   Exposes internal (non-tool) reload_db() called after enroll_finish /
│                             #   emotion_enroll_finish so new enrollments go live without a restart.
├─ voice_agent/              # NEW package.
│  ├─ main.py                 #   entrypoint: wire everything, load CEREBRAS_API_KEY, start threads/loop.
│  ├─ orchestrator.py         #   asyncio state machine (IDLE→LISTEN→ENDPOINT→THINK→SPEAK), barge-in.
│  ├─ state.py                #   VisionState + ConvState dataclasses + vision_lock/conv_lock.
│  ├─ vad_silero.py           #   Silero VAD (onnxruntime) — 32 ms frames, endpoint, barge-in gate.
│  ├─ stt_moonshine.py        #   Moonshine v2 tiny streaming wrapper (+ stt_vosk.py fallback).
│  ├─ tts_piper.py            #   Piper subprocess wrapper, sentence-chunk streaming (+ espeak fallback).
│  ├─ cerebras_client.py      #   OpenAI-SDK-override client: streaming, tool-call accumulation + dispatch.
│  ├─ tools.py                #   8 OpenAI-shaped tool schemas + dispatch → VisionService methods.
│  ├─ audio_ws.py             #   PRIMARY transport: WebSocket audio in/out + serves the minimal orb UI.
│  ├─ audio_alsa.py           #   FALLBACK transport: sounddevice/ALSA capture+playback + PTT (+ opt AEC).
│  └─ config.py               #   model paths, thresholds, core pinning, RING_SECONDS/MAX, env.
├─ models/                    # + Moonshine tiny .onnx, Silero VAD .onnx, Piper voice .onnx/.json
└─ data/                      # enrollments.json + emotions.json (UNCHANGED, written only by registration)
```

**Relationships:** `VisionService` holds one `FaceEngine` so models/DBs load once. `orchestrator.py`
consumes `vad_silero` + `stt_moonshine`, calls `cerebras_client`, which dispatches through `tools.py` into
`VisionService`, then drives `tts_piper`. Audio flows over `audio_ws` (or `audio_alsa`). Registration and
the live tool surface share the same `FaceEngine` instance and `data/*.json`; `reload_db()` is the only
bridge — **registration mutates the models, the tool surface reads them.**

---

## 11. Phased implementation plan (MVP first; Mac → board → expand)

| Phase | Goal | Where | **Testable milestone** |
|---|---|---|---|
| **0** | Cerebras smoke test | Mac | `cerebras_client.py` completes a **streamed tool-calling round-trip** against `gpt-oss-120b` with a dummy tool, reading `CEREBRAS_API_KEY`; prints accumulated tool args + final answer. |
| **1 — MVP** | STT → Cerebras(tool) → TTS + **one** vision tool | Mac | Speak *"who do you see?"* (push-to-talk) → Moonshine transcribes → `gpt-oss-120b` calls `who_is_in_view` → answer spoken via Piper. Half-duplex, PTT. **Proves the whole loop end-to-end.** |
| **2** | Full `VisionService` + all 8 tools | Mac | *"How was I feeling a minute ago?"* → `emotion_timeline`; *"did anyone walk in?"* → `presence_events`; multi-face `describe_scene` works. Registration still runs unchanged; `reload_db()` picks up a new enrollment live. |
| **3 — RUNS ON THE UNO Q** | Port to board | **Board** | Swap Mac STT/TTS for **Moonshine tiny + Piper low voice**; attach audio (browser WS *or* USB headset); pin cores 0–3; add heatsink. **Full turn (voice→tool→voice) runs entirely on the UNO Q** within the §8 budget; log per-stage latency; confirm the vision loop stays live on core 3 throughout. |
| **4** | Barge-in + endpointing + UI | Board | Silero VAD-only **barge-in** stops TTS mid-sentence; ~600 ms silence endpoint replaces PTT (browser AEC path); presence orb + optional captions. |
| **5** | Hardening / degraded modes | Board | Sustained-load thermal check; ALSA card pinning; Cerebras 429 retry/backoff; auto-fallbacks (espeak on overload, Vosk on WER, PTT on echo, 2 fps vision under thermal pressure) all verified. |

**MVP definition (Phase 1) is deliberately minimal and Mac-first** to de-risk the audio↔LLM↔audio loop
before fighting the board; **Phase 3 is the explicit "it runs on the UNO Q" gate.**

---

## 12. Risks + mitigations

| Risk | Mitigation |
|---|---|
| **Board CPU / thermal throttling** — sustained all-core STT heat-soaks and DVFS-throttles below 2.0 GHz within minutes. | Heatsink + airflow (budgeted). Half-duplex means STT/TTS never overlap, so the steady state is vision(1 core)+bursty STT, not a 4-core grind. Monitor temp; degrade to 2 fps vision + x-low Piper voice / espeak under pressure. |
| **Echo / barge-in** — TTS bleeds into the mic (near-end-to-echo ratio < −10 dB), triggering false barge-ins. | **Primary:** browser AEC3 (offloads to client CPU). **Guaranteed fallback:** push-to-talk (mic closed while speaking — no AEC needed). VAD barge-in requires ~150–200 ms voiced to reject echo blips; bump `speaking_utterance_id` to drop late TTS chunks. On-board SpeexDSP/WebRTC-APM AEC is a stretch goal, not on the critical path. |
| **Cerebras rate limits / network** — free tier is **5 RPM / 30 K TPM** (each turn ≈ 1–2 calls with tool round-trips → a turn every ~24 s: too slow); RTT/TTFT dominates latency; historical free-tier 8 K context cap. | Use the **Developer tier** (from $10: 1 K RPM / 1 M TPM for `gpt-oss-120b`) for real conversation. Stream to hide TTFT; `parallel_tool_calls` to batch tools into one hop; exponential backoff on 429; a spoken "one sec…" filler on slow links; trim `ConvState.history` to a token budget; verify the account's context cap. |
| **Key handling** — leaking `CEREBRAS_API_KEY`. | Env var **only**, never hardcoded/logged/committed; the **key stays on the board** (the browser only streams audio, never sees the key). On the board store it in a root-readable systemd `EnvironmentFile`, not in the repo; fail fast at startup if unset. |
| **STT accuracy on hard audio** (Moonshine WER). | Vosk-small fallback (config switch); measure WER on the Mac against faster-whisper ground truth; tune Silero VAD sensitivity. |
| **ALSA card renumbering** (board-native path). | Pin the USB card by `/dev/snd/by-id` or an `.asoundrc`/udev rule; use one USB-Audio device for mic+speaker; run capture/playback under a systemd service (Arduino's `app_peripherals.speaker` helper was early/unstable). |
| **Piper GPL-3.0** in a product. | Run Piper as a separate subprocess (not statically linked); switch to the MIT `piper-plus` fork if needed. |
| **`tools` + `response_format` rejected on `gpt-oss-120b`.** | Use tools only; do all structuring via tool schemas (`strict:true`), never `response_format`. |

---

## 13. MCP integration (tiered tool architecture)

**MCP = Model Context Protocol.** The question is not "MCP *or* function-calling" — it's *where each tool sits*. This section adds an external-integration surface (calendar, home automation, notes/memory, web search, messaging, timers) **without touching the hot-path perception tools of §5**, and does it with **one impl per tool** (DRY). The whole design still bends to §2: four CPU-only A53 cores, ~2 GB of RAM headroom, cloud LLM.

### 13.1 Should we use MCP? — verdict: **yes, tiered (not for everything)**

Adopt a **three-tier "tool bus,"** mapped straight onto the board budget:

- **Tier 0 — in-process local perception (µs, NO MCP).** The 8 vision tools of §5 stay exactly where they are: direct Python calls into `VisionService` under `vision_lock` (§6–§7), **microseconds** per call. A direct in-process call is µs; MCP forces *at minimum* a **process hop** (stdio) or a **network hop** (HTTP), plus JSON-RPC encode/decode. On four in-order A53 cores, serialization and context-switching are *relatively* more expensive than on a server CPU, so wrapping a per-frame perception tool (detect/recognize/emotion, VAD) in MCP is pure waste — ms-scale overhead per frame for something that natively costs µs. These tools also produce the **identity signal** that gates everything below (§13.6).
- **Tier 1 — MCP for network-bound external integrations (overhead negligible).** Calendar, home automation, memory, web search/fetch, messaging, timers. Each of these is *already* dominated by network + the Cerebras RTT (200–400 ms, §8), so MCP's **~5–10 ms** local / one RTT remote disappears into the noise. This is exactly where MCP's interop, tool discovery, and per-server sandboxing earn their keep — and where standing up bespoke API glue would be the real waste.
- **Tier 2 — the *same* vision impls, optionally re-exposed as an MCP server (interop/demos only).** A thin FastMCP facade over the **same** `VisionService` methods, so a *different* MCP client (Claude Desktop, a teammate's agent) can ask "who's in view?" — **one impl, two surfaces.** Never on the board's own hot path; the board still calls perception in-process.

**Net:** MCP overhead is negligible against work measured in hundreds of ms (external API + cloud-Cerebras), and *disqualifying* against work measured in µs (local perception). The guidance from the 2026 literature is the same: use plain function/tool calling for deterministic in-app execution; add MCP only when integrations must be shared across clients or you cross ~10 tools / multiple providers. We do both — hence *tiered*, not all-or-nothing.

**One-line rule:** keep anything on the µs real-time path in-process; put anything already gated by the network behind sandboxed, allowlisted MCP; and let the **face/voice identity gate — not the open microphone — authorize** send-message, unlock, and purchase.

### 13.2 The bridge — Cerebras is not an MCP client; the orchestrator is the host

Cerebras (like any OpenAI-compatible `chat/completions` endpoint) is **not** an MCP client. It only understands the OpenAI `tools=[{"type":"function","function":{name, description, parameters(JSON Schema)}}]` shape — the exact shape §4.1/§5 already emit. So **our orchestrator plays the MCP host/client role** and does the mechanical translation. This is the industry-standard "MCP-to-function-calling bridge," and it is the *only* way to give a non-MCP model access to MCP tools:

1. **`tools/list`** — connect to each MCP server, list its tools → `name`, `description`, `inputSchema` (which **is already JSON Schema**).
2. **Convert (near 1:1)** — wrap each into `{"type":"function","function":{"name":t.name,"description":t.description,"parameters":t.inputSchema}}`. Light massaging only: strip JSON-Schema keywords Cerebras' `strict` mode rejects, and **namespace names** if two servers collide.
3. **Complete** — hand the merged `tools=[...]` to Cerebras `gpt-oss-120b` via the **existing `cerebras_client.py`** streaming + tool-call-accumulation loop (§4.1). No new LLM code path.
4. **Dispatch** — on `finish_reason == "tool_calls"`, for **every** `tool_call` parse `name` + `json.loads(arguments)` and route to the owning server via `session.call_tool(name, args)`.
5. **Feed back** — append a `{"role":"tool","tool_call_id":…, "content":…}` message for **each** call (a single turn can carry *multiple* `tool_calls`, and you must return a tool message for every one or the next completion errors — this is the same invariant §4.1 already handles with `parallel_tool_calls=True`), then call the model again. Loop until no more `tool_calls`.

**Concrete Python libraries that already do this:**

| Library | Role here | License |
|---|---|---|
| **Official `mcp` SDK** (`modelcontextprotocol/python-sdk`, v1.28.x stable) | Reference **client** — `session.list_tools()` / `session.call_tool()` over **stdio / Streamable HTTP / SSE**. The raw primitives we wrap for Cerebras ourselves. Also **vendors FastMCP 1.0** as `mcp.server.fastmcp.FastMCP` for building servers (Tier 2). | MIT |
| **`fastmcp`** (gofastmcp.com, v3.x) | The standalone superset — fastest way to **build** servers (`@mcp.tool` decorator, name/description/schema inferred from the function) plus a high-level client. Used for `vision_mcp_server.py`. | Apache-2.0 |
| `langchain-mcp-adapters` | `MultiServerMCPClient` + `get_tools()` → LangChain tools you `.bind_tools()` onto `ChatOpenAI(base_url=…)` pointed at Cerebras. Removes bridge boilerplate **if** we adopt LangChain. | MIT |
| `mcp-use` / OpenAI Agents SDK | Both run the whole tool loop for you (`MCPServerStdio` / `MCPServerStreamableHttp` objects). Model-agnostic for local servers. | MIT |

**Pick for this project:** the **official `mcp` client + our existing OpenAI-SDK Cerebras path** — no new framework, and `cerebras_client.py` stays the single tool loop. We deliberately **do not** pull in LangChain/`mcp-use` just for the bridge; the conversion is ~10 lines and a framework would fight the hand-tuned streaming/barge-in orchestrator of §7. `fastmcp` is used **only** to *build* our Tier-2 vision server, not to consume anything.

### 13.3 The tool bus / registry — one `tools[]` list, dispatch by origin

The orchestrator must present Cerebras with **one flat `tools[]` list** that transparently mixes local (Tier 0) and MCP (Tier 1) tools, and then route each returned `tool_call` back to *whoever owns it*. That merge/route logic is a new module, `voice_agent/tool_bus.py`, sitting between `orchestrator.py` and `cerebras_client.py`:

```python
# voice_agent/tool_bus.py  — the registry the orchestrator talks to
class ToolBus:
    def __init__(self, vision_service, mcp_sessions, policy):
        self._local   = {}                 # name -> callable  (VisionService methods, §5)
        self._mcp     = {}                 # name -> (server_id, ClientSession)
        self._schemas = []                 # merged OpenAI tool defs Cerebras sees
        self._vision  = vision_service
        self._sessions = mcp_sessions      # {server_id: ClientSession}, from config.py
        self._policy  = policy             # per-tool risk tier + allowlist (§13.6)

    async def build(self):
        # Tier 0 — the 8 vision tools; schemas already OpenAI-shaped in tools.py (§5)
        for name, fn, schema in local_vision_tools(self._vision):
            self._local[name] = fn
            self._schemas.append(schema)
        # Tier 1 — each MCP server: tools/list, convert inputSchema ~1:1, tag origin
        for sid, sess in self._sessions.items():
            for t in (await sess.list_tools()).tools:
                qn = f"{sid}__{t.name}"                      # namespace to avoid collisions
                if not self._policy.allowed(qn):             # allowlist: block unknown/new tools
                    continue                                 #   (defeats rug-pull / silent redefine)
                self._mcp[qn] = (sid, sess)
                self._schemas.append({"type": "function", "function": {
                    "name": qn, "description": t.description,
                    "parameters": t.inputSchema}})           # MCP inputSchema IS JSON Schema

    def tools(self):                        # what orchestrator passes to cerebras_client.py
        return self._schemas

    async def dispatch(self, name, args, identity):          # route BY ORIGIN
        self._policy.check(name, args, identity)             # allowlist + identity gate (§13.6)
        if name in self._local:                              # Tier 0: µs, in-process, under vision_lock
            return self._local[name](**args)
        sid, sess = self._mcp[name]                          # Tier 1: over MCP
        out = await sess.call_tool(name.split("__", 1)[1], args)
        return out.content[0].text
```

**How it slots into the §10 layout:**

- **`tools.py` — role unchanged.** Remains the local source of truth: the 8 vision tool impls + their OpenAI schemas (§5). This *is* Tier 0. `ToolBus` imports it; nothing here learns about MCP.
- **`voice_agent/tool_bus.py` — NEW.** The registry/merger/dispatcher above. Holds the local callables *and* the live MCP `ClientSession`s; enforces the allowlist and the identity gate (§13.6) *before* any dispatch.
- **`voice_agent/vision_mcp_server.py` — NEW (Tier 2).** A thin `fastmcp` wrapper that registers the **same** `VisionService` methods as MCP tools — `mcp.tool()(vision.who_is_in_view)`, etc. FastMCP derives each tool's name/description/JSON-Schema from the function name, docstring, and type hints, so **there is zero schema duplication** — `who_is_in_view` has exactly one impl, callable in-process (Tier 0) *and* over MCP (Tier 2). Runs `stdio` by default; `mcp.run(transport="streamable-http", …)` if hosted. **Not** used by the board's own agent.
- **`config.py` — NEW keys.** The MCP server table (per server: `command`/`args` for stdio or `url` for HTTP, transport, upstream API key/token), the **per-tool allowlist + risk tier**, and the identity thresholds (`τ_high`, liveness on/off).
- **`orchestrator.py` — one wire change.** It asks `tool_bus.tools()` for the merged list (built once at startup, rebuilt on server (re)connect), passes it to `cerebras_client.py`, and on `tool_calls` calls `tool_bus.dispatch(name, args, identity)` instead of reaching into `tools.py` directly. `cerebras_client.py`'s loop is untouched — it just sees a longer `tools[]` and a different dispatch callback.

### 13.4 External MCP servers to integrate (prioritized)

Each server below is a **separate process** the board talks to as an MCP *client*. Two deployment shapes matter (§13.5/§13.7): **local stdio** servers the board spawns (must be thin), and **remote HTTP** servers on a Mac / home server the board reaches over the LAN (weight doesn't matter, but auth does).

| # | Server | Category | Runtime · weight | Transport | Where it runs | Payoff |
|---|---|---|---|---|---|---|
| **1** | **Home Assistant MCP** (official, HA 2025.2+) | home automation | inside HA (Python) · **free** (already on the HA box) | Streamable HTTP | **Home Assistant host** (long-lived token) | The "wow" that *pays off* face-ID + emotion: recognized person walks in stressed → lights/scene adjust. Highest impact per effort. |
| **2** | **Memory** (official reference server) | notes / continuity | Node · **very light** | stdio | **Board (edge-OK)** | Turns face-ID into *continuity*: remember who each recognized person is and their preferences across sessions. No auth. Makes identity feel intelligent, not a gimmick. |
| **3** | **Google Calendar** (`nspady/@cocal/google-calendar-mcp`) | calendar | Node · medium | stdio + HTTP | **Home server** (owns the OAuth refresh token) | "Good morning, [recognized name] — you have 3 meetings." Natural voice payoff; one-time browser OAuth. |
| 4 | **Time** (official) | time / timezones | Python · **very light** | stdio | **Board** | Timer math + reminder phrasing. No auth. |
| 5 | **Fetch** (official) | web fetch | Python · **very light** | stdio | **Board** | "What's on this page." No auth. |
| 6 | **Brave / Tavily search** (official) | web search | Node · light | stdio | **Board** | "What's the weather / news." One API key. |
| — | Slack / Telegram / Email | messaging | Node/Go/Python · medium (**holds a live session**) | HTTP / stdio | Home server | **Defer (roadmap):** per-account OAuth + long-lived sessions that shouldn't churn on board reboot; low visual punch on stage. |
| — | Obsidian (`mcp-obsidian`) | notes | Python/Node · light | stdio | Wherever the vault lives | **Defer:** only compelling if you already use Obsidian. |

**RAM tax (reconcile with §2.2).** Each MCP server is a full language runtime: a Node server is **~50–200 MB RAM**, Go ~18 MB, Java ~220 MB; Python single-worker ~26 ms/request. Two or three on-board Node/Python servers can eat **10–25 %** of RAM before doing any work — real pressure against the ~2 GB headroom of §2.2. So keep on-board to the **very-light** servers (Time, Fetch, Memory, one search), cap heaps, and offload the rest. The **tightest demo triangle is HA + Memory (on-board) + Calendar (on host)** — it showcases identity + emotion + real-world action while staying inside the compute budget; Time + Fetch/Brave are cheap board-local rounding.

### 13.5 Transports — stdio local, Streamable HTTP remote, SSE avoided

The current spec (2025-11-25) defines **two** transports, plus deprecated legacy SSE:

- **stdio — for on-board / local servers.** The client (`tool_bus`) launches the server as a subprocess; JSON-RPC over stdin/stdout. **~0 ms protocol overhead** (message exchange at process-exec speed, ~4–9 ms cold), no network hop, no TLS, no auth server to stand up. Use for **Time, Fetch, Memory, search** and for `vision_mcp_server.py` when run locally. Lifecycle is tied to the subprocess → **supervise/restart** (§13.7).
- **Streamable HTTP — for off-board / shared servers.** A single HTTP endpoint: POST (client→server) + optional GET/SSE upgrade for streaming; plain JSON for short calls. Costs a TLS + HTTP round-trip (~5–10 ms LAN, more over internet) but scales statelessly. Use for **Home Assistant and Calendar** on the Mac/home server reached over the LAN, and anything OAuth-protected.
- **HTTP+SSE — legacy, avoid.** Deprecated in spec 2025-03-26 (mandatory persistent SSE stream, no resume on drop, needs sticky load-balancing). Only touch it for backward-compat with an old server.

**Rule:** local → **stdio**; remote/shared → **Streamable HTTP**; SSE → avoid unless forced.

**Auth trend (matters for the split).** The MCP authorization spec (Nov 2025) mandates that any **remote** server exposed over the network implement **OAuth 2.1 with PKCE (S256)**, plus RFC 9728 Protected Resource Metadata, RFC 8707 Resource Indicators (token-audience binding), and CIMD for client registration. Implication: **local stdio servers skip OAuth entirely** — they run inside the board's trust domain and authenticate to the *upstream* service with a stored API key/token. Only the LAN-exposed servers (HA, Calendar) carry the full OAuth dance. This is another reason to keep the thin servers on-board and the credential-holding ones on a host.

### 13.6 Security — the open-mic problem, and identity-gating as the answer

**Threat model:** the trigger surface is **open-air voice** — anyone within earshot, plus recorded/injected/replayed audio. Consumer precedent is the cautionary tale: Alexa/Google Home block *purchases* for unrecognized voices but will still execute security-critical commands (unlock doors) for previously-unheard voices, with speaker verification off by default. **Do not repeat that.** The board's own face recognition is the authorization factor the open mic can't provide.

**Identity-gating (the differentiator).** Because the vision loop already runs SFace identity + emotion (§5–§6), gate sensitive MCP actions on **two independent factors** so neither audio spoofing nor a photo alone suffices:

- **Factor 1 — face-ID confidence + liveness.** Read the live signal via the Tier-0 tools `who_is_in_view` / `get_person_emotion` (§5) under `vision_lock`: require the SFace cosine `identity_score ≥ τ_high` for the enrolled owner **and** a passive anti-spoof/liveness check so a printed photo or a phone screen fails.
- **Factor 2 — spoken confirmation bound to the same co-located, face-matched speaker.** A **challenge-response** phrase (a random word the agent speaks and the owner repeats) defeats pre-recorded replay; lip/voice co-location defeats a bystander shouting commands.

**Risk-tiered policy (enforced in `tool_bus.dispatch` *before* `session.call_tool`):**

| Risk tier | Example tools | Gate |
|---|---|---|
| **Read-only** | weather, time, "who's here", `describe_scene` | none — auto-run, logged |
| **Sensitive** | send-message, write a note/memory, add a calendar event | owner face `identity_score ≥ τ_high` **AND** explicit spoken "yes" **AND** allowlist pass |
| **Destructive** | unlock door, make a purchase, delete | owner face `≥ τ_high` **AND** `liveness_pass` **AND** challenge-response phrase **AND** allowlist pass |

**Other MCP-specific mitigations (each mapped to a module):**

- **Per-tool allowlist / capability policy** in `config.py`, enforced at `tool_bus` (the client/proxy). **Block dynamically-discovered tools by default** — a new tool that appears in a server's `tools/list` is *not* auto-exposed to Cerebras; it requires explicit review. This defeats **rug-pull / mutable-definition** attacks (a tool approved safe on day 1 silently redefining itself).
- **Human-in-the-loop for destructive actions** — tiered approval: low-risk auto-runs with logging; high-risk pauses for the spoken confirmation, which here *is* the face-gated challenge-response above.
- **Tool-description / tool-poisoning injection** — tool descriptions load straight into the model's context and are an injection vector. Pin/scan manifests (`mcp-scan`) before trusting a server's self-described tools; don't expose a server whose descriptions you haven't reviewed.
- **Injection via returned content** — tool *outputs* (a fetched web page, an email body, and critically an **ASR transcript of ambient speech**, or Tier-2 scene text) can carry adversarial instructions. Treat all tool output *and* ambient ASR as **untrusted**; sanitize before it re-enters the LLM context.
- **Confused deputy / over-broad scopes / token theft** — use **scoped, per-server credentials, never shared tokens**; validate token audience on every inbound request; request least-privilege OAuth scopes (read-only Calendar, not full account). Real incidents underline this: a Nov 2025 WhatsApp MCP integration leaked entire message histories, and over-scoped Gmail creates aggregation risk.
- **Sandbox each on-board server** in a container/restricted env with least filesystem/network/credential access.

**Why this closes the loop:** identity-gating also **blunts prompt injection.** Even if a poisoned tool description or a malicious ASR'd sentence convinces Cerebras to call `unlock_door`, the call **cannot execute without a live, confident owner face in front of the camera** — the vision differentiator *is* the security control.

### 13.7 Edge-cost — run only what the board needs on the board

- **On the board (stdio, thin only):** Time, Fetch, Memory, one search server. **Cap Node heaps** (`--max-old-space-size`) and **supervise/restart** — a documented lifecycle leak accumulated **141 Node processes = 10.7 GB** over ~5 hours; on a 4 GB board that is an OOM kill. Prefer Go/Rust servers, or fold multiple tools into a **single shared Python/Node process**, to bound the 50–200 MB-per-server tax. Python servers install fast with `uv` (the documented ARM/Pi pattern).
- **Offload to a Mac / home server over the LAN:** Home Assistant (already lives on the HA host — the board just points at `http://<ha-host>:8123` with a long-lived token), Calendar (owns a refresh token + browser OAuth flow — don't scatter that on the appliance), and any messaging server (holds a live session that shouldn't churn on board reboot). Weight is irrelevant there; auth (§13.5) is not.
- **Keep perception in-process (the load-bearing rule).** The 8 vision tools never go behind MCP for the board's own agent — µs vs ms/frame on a CPU-only A53. `vision_mcp_server.py` (Tier 2) exists only for *other* clients and is a thin, rate-limited, read-only facade; standing it up does not change the board's hot path.

### 13.8 Phased rollout (extends the §11 plan)

Three phases layered on top of §11 (which ends at Phase 5, board hardening). Same Mac-first-then-board spirit; each has a concrete testable milestone.

| Phase | Goal | Where | **Testable milestone** |
|---|---|---|---|
| **6 — MVP MCP bridge** | Bridge **one** external MCP server alongside the in-process vision tools | Mac → Board | Stand up `tool_bus.py`; connect **one board-local stdio server with no auth** (**Time/timer** *or* **Fetch/Brave web search**). Ask *"set a 5-minute timer"* / *"what's the weather?"* → Cerebras sees the **merged `tools[]`** (8 vision + N MCP) → calls the MCP tool → `tool_bus` routes to `session.call_tool` → answer spoken via Piper. **Proves the list→convert→complete→dispatch→feed-back bridge and dispatch-by-origin with zero new auth/security surface.** |
| **7 — Identity-gated actions** | An MCP server that *acts*, gated by face-ID | Board | Add a remote **Home Assistant** (Streamable HTTP) *or* **Calendar** (on host) server + the **Memory** server (on-board stdio, for face-ID continuity). Route sensitive/destructive tools through the §13.6 gate. *"Turn on the lights"* runs **only** when the enrolled owner is in view (`identity_score ≥ τ_high`) and speaks the confirmation; an **unknown face — or a photo — is refused.** Proves allowlist + identity gate + human-in-loop end-to-end. |
| **8 — Vision-as-MCP-server (interop)** | Expose the same perception over MCP | Board | Stand up `vision_mcp_server.py` (FastMCP over the **same** `VisionService` methods — DRY), stdio locally + optional Streamable HTTP. An **external** MCP client (Claude Desktop / another agent) calls `who_is_in_view` / `describe_scene` and gets the **same answers** the in-process agent does — **interop without touching the board's hot path.** Treat its outputs as untrusted (§13.6). |
