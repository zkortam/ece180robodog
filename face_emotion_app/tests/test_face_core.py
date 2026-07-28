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
    "https://ece180.vercel.app",
    "https://ece180-git-main-team.vercel.app",
    "https://ece180-abc123.vercel.app",
])
def test_our_own_vercel_deployments_are_allowed(origin):
    assert fe.is_allowed_ui_origin(origin)


@pytest.mark.parametrize("origin", [
    "https://someone-else.vercel.app",
    "https://ece180.attacker.com",
    "https://evil.com",
    "https://ece180.vercel.app.attacker.com",
    "https://notece180.vercel.app",
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
    assert fe.is_allowed_ui_origin("https://ECE180.vercel.app:443")


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
