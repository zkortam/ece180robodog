"""Shared test setup.

The app is a flat package rooted at face_emotion_app/ (face_emotion.py and
vision_service.py are top-level modules that voice_agent/ imports), so tests need
that directory on sys.path exactly the way main.py arranges it at runtime.
"""
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

MODELS = APP_DIR / "models"
DETECTOR = MODELS / "face_detection_yunet_2023mar.onnx"
RECOGNIZER = MODELS / "face_recognition_sface_2021dec.onnx"

# Model weights are gitignored (they are large, and the emotion model is licensed
# separately), so anything needing a real cv2 model skips rather than fails when
# they have not been downloaded.
needs_models = pytest.mark.skipif(
    not (DETECTOR.exists() and RECOGNIZER.exists()),
    reason="face models not present; run scripts/download_models.sh",
)


class _TestStatus:
    """Keep unit tests from touching real board LEDs or the RouterBridge."""

    def set(self, state):
        return True


@pytest.fixture(autouse=True)
def isolate_board_status(monkeypatch):
    # On the UNO Q, audio tests mock subprocess.Popen while the real status
    # indicator uses subprocess.run in a background thread. Without isolation the
    # two mocks cross-talk only on physical hardware, making the suite non-hermetic.
    monkeypatch.setattr("voice_agent.board_audio.status_led", _TestStatus())
