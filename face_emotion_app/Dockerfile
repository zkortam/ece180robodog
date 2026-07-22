# Voice agent container (Azure App Service for Containers, HTTPS -> browser mic/camera).
# Built in the cloud with `az acr build` (no local Docker required).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ffmpeg libgl1 libglib2.0-0 espeak-ng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
      numpy opencv-python-headless flask openai faster-whisper kokoro-onnx

COPY . .

# Bake the models into the image (face detect/recognize/emotion + Kokoro voice)
RUN bash scripts/download_models.sh \
 && curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
 && curl -L -o models/voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Pre-download the STT model into the image so the first turn is fast
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8')"

ENV VOICE_STT=faster-whisper VOICE_TTS=kokoro PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "python -m voice_agent.main --host 0.0.0.0 --port ${PORT:-8000} --browser-camera"]
