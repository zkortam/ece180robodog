"""Two processes, one enrollment database.

On the board the voice agent (VisionService) and the enrollment server
(FaceEngine) run side by side against the same data/ directory, and both persist
the WHOLE dictionary in one write. Whichever one holds a stale in-memory copy
erases the other's people. These tests pin the merge behaviour that prevents it.
"""
import threading

import numpy as np

import face_emotion as fe
from vision_service import VisionService


def unit(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def bare_engine(tmp_path):
    """A FaceEngine with only its storage state, so no cv2 model is loaded."""
    engine = object.__new__(fe.FaceEngine)
    engine.db_path = tmp_path / "enrollments.json"
    engine.emotion_db_path = tmp_path / "emotions.json"
    engine.db = fe.load_db(engine.db_path)
    engine.emotion_db = fe.load_emotion_db(engine.emotion_db_path)
    engine._db_mtime = fe.file_mtime_ns(engine.db_path)
    engine._emotion_db_mtime = fe.file_mtime_ns(engine.emotion_db_path)
    engine.lock = threading.Lock()
    engine.enroll_name = None
    engine.enroll_embeddings = []
    engine.emo_enroll_name = None
    engine.emo_enroll_expr = None
    engine.emo_enroll_feats = []
    return engine


def bare_vision(tmp_path):
    """A VisionService with only its storage state, so no cv2 model is loaded."""
    vs = object.__new__(VisionService)
    vs.db_path = tmp_path / "enrollments.json"
    vs.emotion_db_path = tmp_path / "emotions.json"
    vs.db = fe.load_db(vs.db_path)
    vs.emotion_db = fe.load_emotion_db(vs.emotion_db_path)
    vs._db_mtime = fe.file_mtime_ns(vs.db_path)
    vs._emotion_db_mtime = fe.file_mtime_ns(vs.emotion_db_path)
    return vs


# --------------------------------------------------- the enrollment server side

def test_engine_picks_up_a_person_enrolled_by_another_process(tmp_path):
    engine = bare_engine(tmp_path)
    fe.save_db(engine.db_path, {"zakaria": unit(1, 0, 0)})
    with engine.lock:
        engine._refresh_dbs()
    assert "zakaria" in engine.db


def test_deleting_someone_does_not_erase_a_voice_enrollment(tmp_path):
    """THE regression. Enroll "bob" by voice, then delete "alice" in the manage
    UI: the server used to write back its startup snapshot and take bob with it."""
    engine = bare_engine(tmp_path)
    fe.save_db(engine.db_path, {"alice": unit(1, 0, 0)})
    with engine.lock:
        engine._refresh_dbs()
    assert sorted(engine.db) == ["alice"]

    # the voice agent, in the other process, enrolls someone new
    other = fe.load_db(engine.db_path)
    other["bob"] = unit(0, 1, 0)
    fe.save_db(engine.db_path, other)

    engine.remove_person("alice")

    assert sorted(fe.load_db(engine.db_path)) == ["bob"]


def test_enrolling_here_does_not_erase_a_person_added_elsewhere(tmp_path):
    engine = bare_engine(tmp_path)
    fe.save_db(engine.db_path, {"alice": unit(1, 0, 0)})

    engine.enroll_name = "carol"
    engine.enroll_embeddings = [unit(0, 0, 1)]
    engine.enroll_finish()

    assert sorted(fe.load_db(engine.db_path)) == ["alice", "carol"]


def test_training_an_expression_does_not_erase_another_persons(tmp_path):
    engine = bare_engine(tmp_path)
    fe.save_emotion_db(engine.emotion_db_path,
                       {"alice": {"happy": np.ones(7, dtype=np.float32)}})

    engine.emo_enroll_name = "carol"
    engine.emo_enroll_expr = "sad"
    engine.emo_enroll_feats = [np.zeros(7, dtype=np.float32)]
    engine.emotion_enroll_finish()

    saved = fe.load_emotion_db(engine.emotion_db_path)
    assert sorted(saved) == ["alice", "carol"]
    assert "happy" in saved["alice"]


def test_deleting_an_absent_person_is_reported_not_guessed(tmp_path):
    engine = bare_engine(tmp_path)
    fe.save_db(engine.db_path, {"alice": unit(1, 0, 0)})
    assert engine.remove_person("nobody")["status"] == "not_found"
    assert sorted(fe.load_db(engine.db_path)) == ["alice"]


def test_engine_does_not_reload_when_nothing_changed(tmp_path):
    """The refresh runs on every request; it must be a stat(), not a re-parse."""
    engine = bare_engine(tmp_path)
    fe.save_db(engine.db_path, {"alice": unit(1, 0, 0)})
    with engine.lock:
        engine._refresh_dbs()
    identity = id(engine.db)
    with engine.lock:
        engine._refresh_dbs()
    assert id(engine.db) == identity


# ----------------------------------------------------------- the voice side

def test_voice_enrollment_does_not_erase_a_person_added_elsewhere(tmp_path):
    vs = bare_vision(tmp_path)
    fe.save_db(vs.db_path, {"alice": unit(1, 0, 0)})

    vs._enroll = {"kind": "face", "name": "bob", "buf": [unit(0, 1, 0)],
                  "samples": 1, "done": False, "status": "capturing", "last": 0.0}
    vs._finalize_enroll()

    assert sorted(fe.load_db(vs.db_path)) == ["alice", "bob"]


def test_voice_expression_training_does_not_erase_others(tmp_path):
    vs = bare_vision(tmp_path)
    fe.save_emotion_db(vs.emotion_db_path,
                       {"alice": {"happy": np.ones(7, dtype=np.float32)}})

    vs._enroll = {"kind": "emotion", "name": "bob", "expression": "sad",
                  "buf": [np.zeros(7, dtype=np.float32)], "samples": 1,
                  "done": False, "status": "capturing", "last": 0.0}
    vs._finalize_enroll()

    assert sorted(fe.load_emotion_db(vs.emotion_db_path)) == ["alice", "bob"]


def test_both_services_converge_after_interleaved_writes(tmp_path):
    """Alternating writes from both sides must accumulate, never overwrite."""
    engine = bare_engine(tmp_path)
    vs = bare_vision(tmp_path)

    engine.enroll_name = "alice"
    engine.enroll_embeddings = [unit(1, 0, 0)]
    engine.enroll_finish()

    vs._enroll = {"kind": "face", "name": "bob", "buf": [unit(0, 1, 0)],
                  "samples": 1, "done": False, "status": "capturing", "last": 0.0}
    vs._finalize_enroll()

    engine.enroll_name = "carol"
    engine.enroll_embeddings = [unit(0, 0, 1)]
    engine.enroll_finish()

    vs._enroll = {"kind": "face", "name": "dave", "buf": [unit(1, 1, 0)],
                  "samples": 1, "done": False, "status": "capturing", "last": 0.0}
    vs._finalize_enroll()

    assert sorted(fe.load_db(tmp_path / "enrollments.json")) == [
        "alice", "bob", "carol", "dave"]


# --------------------------------------- expression must not leak between people

def face_engine_with_emotion(tmp_path):
    """A FaceEngine with just the recognize_frame state, no cv2 models loaded."""
    engine = bare_engine(tmp_path)
    engine.threshold = 0.5
    engine.emotion_every = 8
    engine.emotion_counter = 1          # not a multiple of 8: no inference this frame
    engine.last_emotion = ("happy", 0.9)
    engine.last_emotion_name = "alice"
    engine.last_emotion_source = "personal"
    engine.emotion = object()           # truthy: an emotion model is loaded
    return engine


def test_a_cached_expression_is_dropped_when_the_face_changes(tmp_path, monkeypatch):
    """Expression is cached between inferences and there is one cache slot, because
    this path only looks at the largest face. When that becomes a DIFFERENT person,
    the cache holds the previous person's expression -- and reported it as the new
    person's. Reading someone else's mood off your face is not a rounding error."""
    engine = face_engine_with_emotion(tmp_path)
    monkeypatch.setattr(fe, "detect_faces", lambda d, f: ["a-face"])
    monkeypatch.setattr(fe, "largest_face", lambda faces: "a-face")
    monkeypatch.setattr(fe, "face_embedding", lambda r, f, face: (unit(0, 1, 0), None))
    engine.db = {"bob": unit(0, 1, 0)}          # the face in view is now bob
    engine.detector = engine.recognizer = object()

    result = engine.recognize_frame(object())

    assert result["name"] == "bob"
    assert result["emotion"] is None, "alice's expression must not be attributed to bob"


def test_a_cached_expression_survives_for_the_same_person(tmp_path, monkeypatch):
    """The cache exists for a reason: inference is expensive and only runs every
    emotion_every frames. Same person, keep it."""
    engine = face_engine_with_emotion(tmp_path)
    monkeypatch.setattr(fe, "detect_faces", lambda d, f: ["a-face"])
    monkeypatch.setattr(fe, "largest_face", lambda faces: "a-face")
    monkeypatch.setattr(fe, "face_embedding", lambda r, f, face: (unit(1, 0, 0), None))
    engine.db = {"alice": unit(1, 0, 0)}
    engine.detector = engine.recognizer = object()

    result = engine.recognize_frame(object())

    assert result["name"] == "alice"
    assert result["emotion"] == "happy"
