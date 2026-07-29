"""Identity matching, expression prototypes, durable storage, and CORS scoping."""
import json

import numpy as np
import pytest

import face_emotion as fe


def unit(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------- identity match

def test_best_match_returns_unknown_for_an_empty_database():
    name, score = fe.best_match({}, unit(1, 0, 0), threshold=0.5)
    assert name == "unknown"


def test_best_match_picks_the_closest_identity():
    db = {"zakaria": unit(1, 0, 0), "chris": unit(0, 1, 0)}
    name, score = fe.best_match(db, unit(0.95, 0.05, 0), threshold=0.5)
    assert name == "zakaria"
    assert score > 0.9


def test_best_match_prefers_unknown_over_a_wrong_name():
    """A confident wrong name is far worse than admitting no recognition: the
    agent would greet a stranger as you and read out your expression history."""
    db = {"zakaria": unit(1, 0, 0)}
    name, score = fe.best_match(db, unit(0, 1, 0), threshold=0.58)
    assert name == "unknown"


def test_best_match_reports_the_score_even_when_rejecting():
    db = {"zakaria": unit(1, 0, 0)}
    _, score = fe.best_match(db, unit(1, 0.6, 0), threshold=0.99)
    assert 0.0 < score < 0.99


# ---------------------------------------------------------- personal emotions

def test_classify_personal_handles_no_prototypes():
    assert fe.classify_personal({}, np.zeros(7, dtype=np.float32)) == (None, 0.0)


def test_classify_personal_picks_the_nearest_prototype():
    protos = {
        "happy": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        "sad": np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
    }
    label, conf = fe.classify_personal(protos, np.array([0, 0, 0, .9, .1, 0, 0],
                                                        dtype=np.float32))
    assert label == "happy"
    assert 0.0 < conf <= 1.0


def test_classify_personal_confidence_drops_when_prototypes_are_close():
    """Two near-identical prototypes must not produce a confident answer."""
    far = {"happy": np.array([1, 0], dtype=np.float32),
           "sad": np.array([0, 1], dtype=np.float32)}
    near = {"happy": np.array([1, 0], dtype=np.float32),
            "sad": np.array([0.98, 0.02], dtype=np.float32)}
    probe = np.array([1, 0], dtype=np.float32)
    assert fe.classify_personal(far, probe)[1] > fe.classify_personal(near, probe)[1]


# ------------------------------------------------------------------- storage

def test_db_survives_a_save_load_round_trip(tmp_path):
    path = tmp_path / "enrollments.json"
    db = {"zakaria": unit(1, 2, 3), "chris": unit(0, 1, 0)}
    fe.save_db(path, db)
    loaded = fe.load_db(path)
    assert sorted(loaded) == ["chris", "zakaria"]
    np.testing.assert_allclose(loaded["zakaria"], db["zakaria"], rtol=1e-6)


def test_emotion_db_survives_a_save_load_round_trip(tmp_path):
    path = tmp_path / "emotions.json"
    db = {"zakaria": {"happy": np.arange(7, dtype=np.float32)}}
    fe.save_emotion_db(path, db)
    loaded = fe.load_emotion_db(path)
    np.testing.assert_allclose(loaded["zakaria"]["happy"], db["zakaria"]["happy"])


def test_loading_a_missing_db_is_empty_not_an_error(tmp_path):
    assert fe.load_db(tmp_path / "nope.json") == {}
    assert fe.load_emotion_db(tmp_path / "nope.json") == {}


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    """A board loses power mid-write eventually. rename() means the old file
    survives intact rather than becoming a half-written file that kills boot."""
    path = tmp_path / "enrollments.json"
    fe.save_db(path, {"zakaria": unit(1, 0, 0)})
    fe.save_db(path, {"zakaria": unit(1, 0, 0), "chris": unit(0, 1, 0)})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["enrollments.json"]
    assert sorted(json.loads(path.read_text())) == ["chris", "zakaria"]


def test_write_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "data" / "nested" / "enrollments.json"
    fe.save_db(path, {"zakaria": unit(1, 0, 0)})
    assert path.exists()


def test_file_mtime_ns_is_zero_for_absent_and_none(tmp_path):
    assert fe.file_mtime_ns(None) == 0
    assert fe.file_mtime_ns(tmp_path / "nope.json") == 0
    p = tmp_path / "there.json"
    p.write_text("{}")
    assert fe.file_mtime_ns(p) > 0


# --------------------------------------------------------------- CORS scoping

@pytest.mark.parametrize("origin", [
    # The LIVE production host. Regression: this was set from the Vercel project
    # name in .vercel/project.json ("ece180"), but the site actually serves from
    # ece180robodog.vercel.app -- which does not start with "ece180-", so the
    # deployed page was blocked from reaching the board on every API call.
    "https://ece180robodog.vercel.app",
    "https://ece180robodog-git-main-team.vercel.app",
    "https://ece180robodog-abc123.vercel.app",
])
def test_our_own_vercel_deployments_are_allowed(origin):
    assert fe.is_allowed_ui_origin(origin)


def test_the_default_matches_the_host_that_is_actually_deployed():
    """A guard against the config drifting away from reality again: the default
    must admit the production hostname with no environment override at all."""
    assert fe.is_allowed_ui_origin("https://ece180robodog.vercel.app")


@pytest.mark.parametrize("origin", [
    "https://someone-else.vercel.app",
    "https://ece180robodog.attacker.com",
    "https://evil.com",
    "https://ece180robodog.vercel.app.attacker.com",
    "https://notece180robodog.vercel.app",
    None,
    "",
])
def test_other_origins_are_rejected(origin):
    """Regression: this was `origin.endswith('.vercel.app')`, which matched every
    Vercel deployment on the internet -- and it is paired with
    Allow-Private-Network on an API that has no authentication and can delete
    enrolled faces."""
    assert not fe.is_allowed_ui_origin(origin)


def test_explicitly_configured_origins_are_allowed():
    extra = ("https://robodog.example.edu",)
    assert fe.is_allowed_ui_origin("https://robodog.example.edu", extra=extra)
    assert not fe.is_allowed_ui_origin("https://other.example.edu", extra=extra)


def test_origin_matching_ignores_port_and_case():
    assert fe.is_allowed_ui_origin("https://ECE180ROBODOG.vercel.app:443")


# ------------------------------------------------------------ label handling

@pytest.mark.parametrize("label,expected", [
    ("happy", "positive"), ("surprise", "positive"),
    ("sad", "negative"), ("angry", "negative"), ("fear", "negative"),
    ("neutral", "neutral"),
    ("fearful", "negative"),        # alias
    ("surprised", "positive"),      # alias
    (None, "not_enabled"), ("", "not_enabled"), ("none", "not_enabled"),
])
def test_sentiment_mapping(label, expected):
    assert fe.sentiment_from_emotion(label) == expected


# --------------------------------------------------- pose-aware framing checks

def face_box(x, y, w, h, yaw_px=0.0):
    """A YuNet-shaped detection: bbox + 5 landmarks (right eye, left eye, nose...)."""
    cx = x + w / 2.0
    return np.array([x, y, w, h,
                     cx - 10, y + h * 0.35,          # right eye
                     cx + 10, y + h * 0.35,          # left eye
                     cx + yaw_px, y + h * 0.55,      # nose
                     cx - 8, y + h * 0.75,
                     cx + 8, y + h * 0.75,
                     0.99], dtype=np.float32)


def frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_a_head_on_face_is_accepted():
    ok, _ = fe.enrollment_guidance(frame(), face_box(90, 60, 130, 120), "center")
    assert ok


def test_looking_down_is_accepted_despite_foreshortening():
    """THE regression the user hit: 'down it struggles'. Tilting the head shortens
    the detected box and moves it down the frame. Judged by the head-on rule that
    fails the height test and is told to 'move closer' -- punishing someone for
    doing exactly what was asked, with advice that makes it worse."""
    tilted = face_box(90, 130, 130, 60)          # short box, low in frame

    ok_down, msg_down = fe.enrollment_guidance(frame(), tilted, "down")
    ok_center, _ = fe.enrollment_guidance(frame(), tilted, "center")
    assert ok_down, f"down should accept a foreshortened face, got {msg_down!r}"
    assert not ok_center, "the head-on rule should still be strict"


def test_looking_up_is_accepted_despite_foreshortening():
    tilted = face_box(90, 18, 130, 60)           # short box, high in frame
    ok, msg = fe.enrollment_guidance(frame(), tilted, "up")
    assert ok, f"up should accept a foreshortened face, got {msg!r}"


def test_a_face_that_is_genuinely_too_small_is_still_rejected():
    """Relaxing the tilt cases must not accept someone across the room."""
    ok, msg = fe.enrollment_guidance(frame(), face_box(140, 100, 40, 30), "down")
    assert not ok
    assert "closer" in msg


def test_tilting_does_not_relax_the_horizontal_checks():
    """Nothing about looking down should let you drift off to one side."""
    ok, msg = fe.enrollment_guidance(frame(), face_box(5, 130, 130, 60), "down")
    assert not ok
    assert "Center" in msg
