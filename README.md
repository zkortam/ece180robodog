# RoboDog

RoboDog is a local-first voice and vision system for an Arduino UNO Q. It gives the robot a conversational presence that can listen, speak, recognize enrolled people, estimate facial expressions, and surface its perception state through a minimal animated interface.

The current repository contains the interaction and perception stack. Locomotion and motor-control integration can be added behind the same tool bus without coupling it to the voice or vision pipeline.

## What it does

- Hands-free voice activity detection and turn-taking
- Local speech-to-text on the development machine or board
- Cerebras-backed conversational reasoning
- Local neural text-to-speech with lightweight fallbacks
- YuNet face detection and SFace identity embeddings
- Local facial-expression inference
- Private, opt-in face and expression enrollment
- Animated listening, thinking, speaking, and visual-analysis states
- Browser-camera and directly attached board-camera modes

## Repository layout

```text
face_emotion_app/
  face_emotion.py          Face enrollment and recognition
  vision_service.py        Live perception and tracking service
  voice_agent/             Voice, tools, orchestration, and web UI
  scripts/                 Install, launch, benchmark, and board helpers
  systemd/                 UNO Q service definition
```

Detailed operating instructions are in [`face_emotion_app/VOICE-AGENT-README.md`](face_emotion_app/VOICE-AGENT-README.md). The deeper system design is in [`face_emotion_app/VOICE-AGENT-ARCHITECTURE.md`](face_emotion_app/VOICE-AGENT-ARCHITECTURE.md).

## Run locally

```bash
cd face_emotion_app
./scripts/install_voice.sh
export CEREBRAS_API_KEY="your-key"
./scripts/run_voice.sh --host 127.0.0.1 --port 8100 --browser-camera
```

Open [http://127.0.0.1:8100](http://127.0.0.1:8100), allow microphone and camera access, then click the face once. After startup, the interface is hands-free.

## Run on the UNO Q

```bash
cd face_emotion_app
./scripts/install_voice.sh
export CEREBRAS_API_KEY="your-key"
./scripts/run_voice.sh --host 0.0.0.0 --board-audio
```

## Privacy and public-repository safety

Face embeddings, expression prototypes, model weights, recordings, API keys, private keys, local environments, and device credentials are excluded from Git. Enrollment is opt-in and remains local by default.

Never commit secrets. Supply API keys through the environment and keep biometric data under the ignored `face_emotion_app/data/` directory.
