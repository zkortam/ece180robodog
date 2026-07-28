"""Face association and identity hysteresis.

These are the subtlest heuristics in the perception loop and the ones whose
failure is most visible in a robot: a wrong association hands one person's name
and expression history to another. They only touch the track list, so they are
exercised directly rather than through a camera.
"""
import numpy as np
import pytest

from vision_service import VisionService, _iou, _position, _size_bucket


def bare_service():
    """A VisionService with only the tracking state, so no cv2 model is loaded."""
    vs = object.__new__(VisionService)
    vs._tracks = []
    vs._next_track = 1
    return vs


def unit(*values):
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ geometry

def test_iou_of_identical_boxes_is_one():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert _iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


def test_iou_of_degenerate_boxes_does_not_divide_by_zero():
    assert _iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


def test_iou_is_symmetric():
    a, b = [0, 0, 10, 10], [5, 5, 10, 10]
    assert _iou(a, b) == pytest.approx(_iou(b, a))


@pytest.mark.parametrize("cx,cy,expected", [
    (10, 10, {"h": "left", "v": "top"}),
    (150, 120, {"h": "center", "v": "middle"}),
    (300, 230, {"h": "right", "v": "bottom"}),
])
def test_position_buckets(cx, cy, expected):
    assert _position(cx, cy, 320, 240) == expected


@pytest.mark.parametrize("frac,expected", [
    (0.01, "small"), (0.10, "medium"), (0.40, "large")])
def test_size_buckets(frac, expected):
    assert _size_bucket(frac) == expected


# --------------------------------------------------------------- association

def test_first_detection_creates_a_track():
    vs = bare_service()
    track = vs._associate([10, 10, 50, 50], unit(1, 0, 0), set())
    assert track["track_id"] == 1
    assert track["name"] == "unknown"
    assert len(vs._tracks) == 1


def test_same_face_in_the_same_place_reuses_its_track():
    vs = bare_service()
    embedding = unit(1, 0, 0)
    first = vs._associate([10, 10, 50, 50], embedding, set())
    first["bbox"] = [10, 10, 50, 50]
    second = vs._associate([12, 12, 50, 50], embedding, set())
    assert second["track_id"] == first["track_id"]
    assert len(vs._tracks) == 1


def test_a_different_face_in_the_same_place_gets_its_own_track():
    """Somebody stepping into a spot another person just left must not inherit
    their track -- that is how a name and an expression profile get swapped."""
    vs = bare_service()
    first = vs._associate([10, 10, 50, 50], unit(1, 0, 0), set())
    first["bbox"] = [10, 10, 50, 50]
    second = vs._associate([10, 10, 50, 50], unit(0, 1, 0), set())
    assert second["track_id"] != first["track_id"]


def test_two_faces_in_one_frame_cannot_share_a_track():
    vs = bare_service()
    embedding = unit(1, 0, 0)
    first = vs._associate([10, 10, 50, 50], embedding, set())
    first["bbox"] = [10, 10, 50, 50]
    assigned = {first["track_id"]}
    second = vs._associate([12, 12, 50, 50], embedding, assigned)
    assert second["track_id"] != first["track_id"]


def test_a_moved_face_is_still_matched_on_strong_appearance():
    vs = bare_service()
    embedding = unit(1, 0, 0)
    first = vs._associate([10, 10, 50, 50], embedding, set())
    first["bbox"] = [10, 10, 50, 50]
    moved = vs._associate([200, 150, 50, 50], embedding, set())
    assert moved["track_id"] == first["track_id"]


def test_track_embedding_stays_normalized_after_smoothing():
    """Matching is a dot product, so an un-normalized track vector silently
    rescales every future similarity score."""
    vs = bare_service()
    track = vs._associate([10, 10, 50, 50], unit(1, 0, 0), set())
    track["bbox"] = [10, 10, 50, 50]
    for _ in range(5):
        vs._associate([10, 10, 50, 50], unit(0.9, 0.1, 0), set())
    assert float(np.linalg.norm(track["embedding"])) == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------ identity

def new_track():
    return {"track_id": 1, "bbox": [0, 0, 10, 10], "name": "unknown",
            "identity_score": 0.0, "identity_misses": 0, "candidate_name": None,
            "candidate_hits": 0, "candidate_at": 0.0, "embedding": None,
            "last_emotion": (None, 0.0, "none", "unknown", {}),
            "last_emotion_at": 0.0, "last_seen": 0.0}


def test_one_good_frame_is_not_enough_to_claim_a_name():
    """Switching on a single cosine score flickers, and a flicker can attach
    another person's emotion profile."""
    vs, track = bare_service(), new_track()
    name, _ = vs._stable_identity(track, "zakaria", 0.7)
    assert name == "unknown"


def test_repeated_evidence_promotes_the_identity():
    vs, track = bare_service(), new_track()
    vs._stable_identity(track, "zakaria", 0.7)
    name, score = vs._stable_identity(track, "zakaria", 0.72)
    assert name == "zakaria"
    assert score > 0.5


def test_a_confirmed_identity_survives_a_few_weak_frames():
    """Blur, profile angle and backlight all produce brief unknown frames. The
    robot must not forget who you are because you turned your head."""
    vs, track = bare_service(), new_track()
    vs._stable_identity(track, "zakaria", 0.7)
    vs._stable_identity(track, "zakaria", 0.7)
    for _ in range(4):
        name, _ = vs._stable_identity(track, "unknown", 0.0)
        assert name == "zakaria"


def test_a_sustained_run_of_unknown_frames_drops_the_identity():
    vs, track = bare_service(), new_track()
    vs._stable_identity(track, "zakaria", 0.7)
    vs._stable_identity(track, "zakaria", 0.7)
    for _ in range(10):
        name, _ = vs._stable_identity(track, "unknown", 0.0)
    assert name == "unknown"


def test_switching_identity_clears_the_cached_expression():
    """Carrying a personalized expression across people is the worst failure
    here: the robot reports how *you* feel about a stranger's face."""
    vs, track = bare_service(), new_track()
    vs._stable_identity(track, "zakaria", 0.7)
    vs._stable_identity(track, "zakaria", 0.7)
    track["last_emotion"] = ("happy", 0.9, "personal", "positive", {})
    track["last_emotion_at"] = 1234.0
    for _ in range(3):
        vs._stable_identity(track, "chris", 0.8)
    assert track["name"] == "chris"
    assert track["last_emotion"] == (None, 0.0, "none", "unknown", {})
    assert track["last_emotion_at"] == 0.0


def test_switching_to_a_new_name_needs_more_evidence_than_a_first_claim():
    """Taking a name away from a confirmed person should be harder than
    assigning one to an unrecognized face."""
    vs = bare_service()
    fresh = new_track()
    vs._stable_identity(fresh, "chris", 0.8)
    assert vs._stable_identity(fresh, "chris", 0.8)[0] == "chris"     # 2 hits

    confirmed = new_track()
    vs._stable_identity(confirmed, "zakaria", 0.8)
    vs._stable_identity(confirmed, "zakaria", 0.8)
    vs._stable_identity(confirmed, "chris", 0.8)
    assert vs._stable_identity(confirmed, "chris", 0.8)[0] == "zakaria"   # 2 not enough
    assert vs._stable_identity(confirmed, "chris", 0.8)[0] == "chris"     # 3 is
