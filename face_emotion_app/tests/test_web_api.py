"""HTTP contract of the voice agent.

Every one of these is a path the browser actually hits when something goes wrong.
The UI distinguishes "server misconfigured" (stop and show why) from "busy"
(retry) from "network died" purely by status code, so the codes are the contract.
"""
import contextlib
import io
import json
import threading

import pytest

from voice_agent import config
from voice_agent.web import create_app


class FakeTTS:
    def synth(self, text):
        return b"RIFFfake" if text else b""


class FakeAgent:
    def __init__(self):
        self.turn_lock = threading.RLock()
        self.tts = FakeTTS()
        self.raise_on_turn = None
        self.reply = "I can see you."

    def _maybe_raise(self):
        if self.raise_on_turn:
            raise self.raise_on_turn

    def handle_text(self, text):
        self._maybe_raise()
        return {"transcript": text, "reply": self.reply, "tools": []}

    def understand_audio(self, path):
        self._maybe_raise()
        return {"transcript": "hello", "reply": self.reply, "tools": [],
                "timings_ms": {"stt": 1.0, "llm": 2.0}}

    def handle_audio(self, path):
        out = self.understand_audio(path)
        out["audio_b64"] = ""
        return out


class FakeVision:
    def __init__(self):
        self.frames = 0
        self.running = True
        self.source = "browser"
        self.jpeg = b"\xff\xd8\xff\xe0jpegbytes"

    def describe_scene(self):
        return {"people": [], "num_faces": 0, "feed_live": True}

    def enroll_status(self):
        return {"active": False}

    def submit_frame(self, img):
        self.frames += 1
        return {"ok": True, "num_faces": 0}

    def frame_source(self):
        return self.source

    @contextlib.contextmanager
    def turn_in_progress(self):
        self.throttled = True
        try:
            yield
        finally:
            self.throttled = False

    def snapshot_jpeg(self, quality=70):
        return self.jpeg


@pytest.fixture
def agent():
    return FakeAgent()


@pytest.fixture
def vision():
    return FakeVision()


@pytest.fixture
def client(agent, vision):
    app = create_app(agent, vision)
    app.config["TESTING"] = True
    return app.test_client()


# ------------------------------------------------------------------- liveness

def test_health_is_cheap_and_describes_the_stack(client):
    body = client.get("/api/health").get_json()
    assert body["ok"] is True
    assert {"stt", "tts", "model", "frame_source"} <= set(body)


# ------------------------------------------- one page, two deployments

def test_health_declares_who_owns_the_camera(client, vision):
    """The page must learn this BEFORE it asks for camera permission. On a
    standalone robot the board owns the camera and the browser must not open a
    second one, or two feeds fight over one perception loop."""
    vision.source = "board"
    assert client.get("/api/health").get_json()["frame_source"] == "board"
    vision.source = "browser"
    assert client.get("/api/health").get_json()["frame_source"] == "browser"


def test_the_robots_own_view_is_served_as_a_snapshot(client):
    """Lets a laptop browser see through the ROBOT's eye instead of its own."""
    response = client.get("/api/vision/snapshot.jpg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data.startswith(b"\xff\xd8")
    assert "no-store" in response.headers.get("Cache-Control", "")


def test_a_snapshot_with_no_live_frame_is_503_not_a_stale_image(client, vision):
    """Better an honest 'nothing to show' than last week's frame presented as now."""
    vision.jpeg = None
    assert client.get("/api/vision/snapshot.jpg").status_code == 503


def test_the_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<!doctype html>" in response.data.lower()


def test_favicon_is_answered_rather_than_404(client):
    assert client.get("/favicon.ico").status_code == 200


# --------------------------------------------------------------- status codes

def test_a_turn_with_no_audio_is_a_400(client):
    assert client.post("/api/voice/turn", data={}).status_code == 400


def test_a_configuration_fault_is_503_not_500(client, agent):
    """The UI stops the conversation and shows the reason on a 503. Reporting a
    bad API key as a generic 500 makes it retry forever and apologize each time."""
    agent.raise_on_turn = SystemExit("CEREBRAS_API_KEY is not set")
    response = client.post("/api/voice/text", json={"text": "hi"})
    assert response.status_code == 503
    assert "CEREBRAS_API_KEY" in response.get_json()["error"]


def test_an_unexpected_failure_is_500_with_a_json_body(client, agent):
    agent.raise_on_turn = RuntimeError("model exploded")
    response = client.post("/api/voice/text", json={"text": "hi"})
    assert response.status_code == 500
    assert "error" in response.get_json()


def test_a_busy_agent_answers_409_rather_than_hanging(client, agent):
    """Half-duplex is intentional; waiting forever behind a wedged turn is not.
    A bounded wait is what keeps the UI from sitting in 'Thinking...' until
    reload -- and, on the robot, what keeps the board audio loop from going deaf.

    The competing turn is held from another thread on purpose: turn_lock is an
    RLock, so a same-thread acquire is re-entrant and would sail straight through
    the guard rather than exercising it. Production always contends across
    threads (a Flask worker versus the board audio loop)."""
    holding = threading.Event()
    release = threading.Event()

    def hold_the_turn():
        with agent.turn_lock:
            holding.set()
            release.wait(5)

    original = config.TURN_LOCK_TIMEOUT
    config.TURN_LOCK_TIMEOUT = 0.05
    worker = threading.Thread(target=hold_the_turn, daemon=True)
    worker.start()
    try:
        assert holding.wait(5)
        response = client.post("/api/voice/text", json={"text": "hi"})
    finally:
        config.TURN_LOCK_TIMEOUT = original
        release.set()
        worker.join(5)
    assert response.status_code == 409


def test_the_turn_lock_is_released_after_a_failure(client, agent):
    """A failed turn that leaks the lock bricks every later turn."""
    agent.raise_on_turn = RuntimeError("boom")
    client.post("/api/voice/text", json={"text": "hi"})
    agent.raise_on_turn = None
    assert client.post("/api/voice/text", json={"text": "hi"}).status_code == 200


@pytest.mark.parametrize("payload", [None, "not a dict", [1, 2, 3]])
def test_malformed_text_bodies_are_400_not_500(client, payload):
    response = client.post("/api/voice/text", data=json.dumps(payload),
                           content_type="application/json")
    assert response.status_code == 400


# ------------------------------------------------------------------- streaming

def test_the_stream_emits_ndjson_meta_then_audio_then_done(client):
    response = client.post("/api/voice/turn-stream",
                           data={"audio": (io.BytesIO(b"RIFFfake"), "turn.wav")})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.data.decode().splitlines() if line.strip()]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert "audio" in kinds


def test_a_configuration_fault_is_marked_fatal_in_the_stream(client, agent):
    """fatal=True tells the page to stop; a non-fatal error lets it fall back."""
    agent.raise_on_turn = SystemExit("bad key")
    response = client.post("/api/voice/turn-stream",
                           data={"audio": (io.BytesIO(b"RIFFfake"), "turn.wav")})
    events = [json.loads(line) for line in response.data.decode().splitlines() if line.strip()]
    assert events[0]["type"] == "error"
    assert events[0]["fatal"] is True


def test_a_transient_failure_is_not_marked_fatal(client, agent):
    agent.raise_on_turn = RuntimeError("transient")
    response = client.post("/api/voice/turn-stream",
                           data={"audio": (io.BytesIO(b"RIFFfake"), "turn.wav")})
    events = [json.loads(line) for line in response.data.decode().splitlines() if line.strip()]
    assert events[0]["type"] == "error"
    assert events[0]["fatal"] is False


# ----------------------------------------------------------------- vision feed

def test_an_empty_frame_upload_is_rejected_cleanly(client):
    """cv2.imdecode ASSERTS on an empty buffer rather than returning None, and a
    tab closing mid-POST delivers exactly that."""
    response = client.post("/api/vision/frame",
                           data={"frame": (io.BytesIO(b""), "f.jpg")})
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_an_undecodable_frame_is_rejected_cleanly(client):
    response = client.post("/api/vision/frame",
                           data={"frame": (io.BytesIO(b"not a jpeg"), "f.jpg")})
    assert response.status_code == 400


def test_a_missing_frame_field_is_a_400(client):
    assert client.post("/api/vision/frame", data={}).status_code == 400


def test_an_oversized_upload_is_refused_before_it_is_read(client):
    """Without a cap the board reads any body straight into RAM."""
    huge = io.BytesIO(b"\0" * (config.MAX_UPLOAD_BYTES + 1024))
    response = client.post("/api/vision/frame", data={"frame": (huge, "f.jpg")})
    assert response.status_code == 413


# ------------------------------------------------------------------ CORS scope

def test_our_hosted_ui_may_drive_the_board(client):
    response = client.get("/api/health", headers={"Origin": "https://ece180robodog.vercel.app"})
    assert response.headers.get("Access-Control-Allow-Origin") == "https://ece180robodog.vercel.app"
    assert response.headers.get("Access-Control-Allow-Private-Network") == "true"


def test_an_unrelated_site_may_not(client):
    """Regression: any *.vercel.app page used to be granted private-network access
    to an API with no authentication that can enroll and delete faces."""
    response = client.get("/api/health", headers={"Origin": "https://attacker.vercel.app"})
    assert "Access-Control-Allow-Origin" not in response.headers
    assert "Access-Control-Allow-Private-Network" not in response.headers


def test_a_same_origin_request_needs_no_cors_headers(client):
    assert "Access-Control-Allow-Origin" not in client.get("/api/health").headers


def test_preflight_succeeds_and_carries_the_cors_grant(client):
    """What matters is that the browser's preflight passes with the private-network
    opt-in attached. Flask answers OPTIONS for a route that has other methods with
    its own 200, so the status is not fixed at 204."""
    response = client.options("/api/voice/turn",
                              headers={"Origin": "https://ece180robodog.vercel.app"})
    assert 200 <= response.status_code < 300
    assert response.headers.get("Access-Control-Allow-Private-Network") == "true"


def test_pages_are_not_cached(client):
    """A stale cached page can keep speaking an obsolete protocol to a fixed board."""
    assert "no-store" in client.get("/").headers.get("Cache-Control", "")
