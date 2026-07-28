"""VisionService behaviour with the real cv2 models loaded but no camera.

step() takes a plain BGR array, so the whole perception surface is testable
without hardware. What matters here is what the agent is TOLD: a robot that
insists someone is in the room an hour after they left is worse than one that
admits it cannot see.
"""
import time

import numpy as np
import pytest

from conftest import needs_models
from vision_service import VisionService

pytestmark = needs_models


@pytest.fixture
def vs(tmp_path):
    service = VisionService(
        db_path=tmp_path / "enrollments.json",
        emotion_db_path=tmp_path / "emotions.json",
        camera=7, fps=6, threshold=0.58, emotion_interval=2.5,
        leave_timeout=0.3,
    )
    yield service
    service.stop()


def blank_frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ------------------------------------------------------- configuration safety

def test_start_watching_preserves_the_tuned_configuration(vs):
    """THE regression. These defaults used to be camera=0, fps=4, threshold=0.5,
    so any call from the model silently reset the tuned identity threshold and --
    on the board, where the webcam is discovered at whatever index it enumerated
    as -- pointed the service at camera 0 and left the robot blind."""
    vs.start(external=True)
    before = vs._config()
    vs.stop()

    vs.start_watching()

    assert vs._config()["camera"] == before["camera"] == 7
    assert vs._config()["fps"] == before["fps"] == 6
    assert vs._config()["threshold"] == before["threshold"] == 0.58


def test_start_watching_still_honours_an_explicit_camera(vs):
    vs.start_watching(camera=2)
    assert vs._config()["camera"] == 2
    assert vs._config()["threshold"] == 0.58      # untouched


def test_start_watching_keeps_the_external_frame_source(vs):
    """Called mid-conversation in browser-camera mode, this must not try to open
    a server camera that does not exist."""
    vs.start(external=True)
    vs.start_watching()
    assert vs.external is True
    assert vs._thread is None


def test_emotion_interval_is_actually_applied(vs):
    """Regression: the old `emotion_every` knob was stored and reported but never
    read, so tuning it did nothing at all."""
    assert vs.emotion_interval == 2.5
    assert vs._config()["emotion_interval"] == 2.5


def test_a_failed_camera_open_leaves_perception_recoverable(vs):
    """THE blind-robot regression. fe.open_camera raises SystemExit, which is a
    BaseException and so was NOT caught by the `except Exception` that claimed to
    handle it. The loop thread died with `running` still True, and the board's
    camera_watch only re-discovers a webcam while `running` is False -- so one
    failed open blinded the robot until someone restarted the service."""
    vs.camera = 999                      # an index that cannot possibly open
    vs.start()
    deadline = time.time() + 5.0
    while vs.running and time.time() < deadline:
        time.sleep(0.05)
    assert vs.running is False, "camera_watch would never retry"


def test_the_camera_thread_never_strands_a_restarted_loop(vs):
    """A thread dying late must not switch off the loop that replaced it."""
    vs.camera = 999
    vs.start()
    deadline = time.time() + 5.0
    while vs.running and time.time() < deadline:
        time.sleep(0.05)
    vs.start(external=True)              # a new generation takes over
    time.sleep(0.2)
    assert vs.running is True


def test_restarting_is_idempotent(vs):
    first = vs.start(external=True)
    second = vs.start(external=True)
    assert first["already_running"] is False
    assert second["already_running"] is True


# ------------------------------------------------------------- frame handling

def test_a_blank_frame_yields_no_faces(vs):
    assert vs.step(blank_frame()) == []
    assert vs.frames_processed == 1


def test_a_noise_frame_does_not_crash_perception(vs):
    rng = np.random.default_rng(0)
    vs.step(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    assert vs.frames_processed == 1


def test_frames_of_an_unusual_shape_are_handled(vs):
    vs.step(blank_frame(w=640, h=480))
    assert vs.frame_w == 640 and vs.frame_h == 480


def test_submit_frame_is_refused_when_not_watching(vs):
    assert vs.submit_frame(blank_frame())["ok"] is False


def test_submit_frame_reports_this_frames_face_count(vs):
    vs.start(external=True)
    assert vs.submit_frame(blank_frame()) == {"ok": True, "num_faces": 0}


# ------------------------------------------- one perception loop, one camera

def test_a_browser_cannot_push_frames_while_the_board_owns_the_camera(vs):
    """Two frame sources into one tracker is a fight, not a merge: association
    thrashes between the room and a laptop webcam, identities flip, and presence
    flickers between two realities. The robot's own eye wins."""
    vs.running = True
    vs.external = False               # a board camera loop owns perception
    result = vs.submit_frame(blank_frame())
    assert result["ok"] is False
    assert result["frame_source"] == "board"
    assert vs.frames_processed == 0   # the frame was not consumed


def test_frame_source_is_board_only_when_a_camera_is_actually_owned(vs):
    """--no-camera has no vision at all; telling the page to wait on a snapshot
    that will never arrive would leave the lens permanently blank."""
    vs.owns_camera = False
    assert vs.frame_source() == "browser"
    vs.owns_camera = True
    assert vs.frame_source() == "board"
    vs.external = True                # browser-camera mode overrides
    assert vs.frame_source() == "browser"


def test_frame_source_is_stable_before_the_camera_is_discovered(vs):
    """A board that has not yet enumerated its webcam must still answer 'board',
    or a browser connecting during boot opens its own camera in the gap."""
    vs.owns_camera = True
    assert vs.running is False
    assert vs.frame_source() == "board"


def test_a_snapshot_needs_a_live_frame(vs):
    assert vs.snapshot_jpeg() is None          # nothing seen yet
    vs.start(external=True)
    vs.step(blank_frame())
    jpeg = vs.snapshot_jpeg()
    assert jpeg is not None and jpeg.startswith(b"\xff\xd8")


def test_a_stale_frame_is_not_served_as_a_snapshot(vs):
    vs.start(external=True)
    vs.step(blank_frame())
    vs.latest_t = time.time() - 60.0
    assert vs.snapshot_jpeg() is None


# --------------------------------------------------------- honesty about time

def test_a_stale_feed_is_not_reported_as_a_live_view(vs):
    """A frozen feed is not a view. The tab closed, the phone locked, the board
    throttled -- whatever the cause, the last frame must not be described in the
    present tense forever."""
    vs.start(external=True)
    vs.step(blank_frame())
    vs.latest = [{"name": "zakaria", "identity_score": 0.9, "bbox": [0, 0, 10, 10]}]
    vs.latest_t = time.time() - 60.0

    view = vs.who_is_in_view()
    assert view["known"] == []
    assert view["num_faces"] == 0
    assert view["stale_seconds"] > 59


def test_describe_scene_marks_a_dead_feed(vs):
    vs.start(external=True)
    vs.step(blank_frame())
    vs.latest_t = time.time() - 60.0
    assert vs.describe_scene()["feed_live"] is False


def test_presence_expires_on_read_not_only_on_new_frames(vs):
    """Expiry used to be driven only from step(), so when frames stopped arriving
    the last state froze and the agent kept insisting someone was present."""
    now = time.time()
    vs.people["zakaria"] = {"present": True, "first_seen": now - 10,
                            "last_seen": now - 10, "last_obs": now - 10}
    assert vs.presence_events()["now_present"] == []


def test_a_departure_is_recorded_as_an_event(vs):
    now = time.time()
    vs.people["zakaria"] = {"present": True, "first_seen": now - 10,
                            "last_seen": now - 10, "last_obs": now - 10}
    events = vs.presence_events()["events"]
    assert [e["event"] for e in events] == ["leave"]


def test_who_is_in_view_reports_watching_state(vs):
    assert vs.who_is_in_view()["watching"] is False
    vs.start(external=True)
    assert vs.who_is_in_view()["watching"] is True


# ------------------------------------------------------------- name handling

def test_names_resolve_case_insensitively(vs):
    """The CLI enrolls "chris"; the model hears a name and writes "Chris". Without
    this they are two people: lookups miss and re-enrolling forks a duplicate."""
    vs.db["chris"] = np.ones(128, dtype=np.float32)
    assert vs._resolve("Chris") == "chris"
    assert vs._resolve("CHRIS") == "chris"


def test_an_unknown_name_is_returned_unchanged(vs):
    assert vs._resolve("Nobody") == "Nobody"


def test_asking_about_someone_unknown_is_not_an_error(vs):
    result = vs.get_person_emotion("nobody")
    assert result["found"] is False
    assert result["present"] is False


def test_emotion_timeline_of_someone_unseen_is_empty_not_an_error(vs):
    result = vs.emotion_timeline("nobody")
    assert result["found"] is False
    assert result["sample_count"] == 0


def test_list_enrolled_works_with_the_camera_off(vs):
    assert vs.list_enrolled() == {"people": []}


# ----------------------------------------------------------------- lifecycle

def test_stop_reports_what_happened(vs):
    vs.start(external=True)
    vs.step(blank_frame())
    result = vs.stop()
    assert result["was_running"] is True
    assert result["frames_processed"] == 1


def test_stopping_when_never_started_is_safe(vs):
    assert vs.stop()["was_running"] is False


# ------------------------------------------------------------- enrollment

def test_enrolling_with_no_camera_fails_fast_instead_of_going_deaf(vs):
    """The turn lock is held for the whole of this call, and on the standalone
    robot that lock IS the conversation. Waiting out the full 25 s timeout meant
    25 s of deafness followed by a message blaming the user's lighting for what
    was actually a missing camera."""
    vs.camera = 999
    started = time.time()
    result = vs.enroll_face("zakaria", samples=4, timeout=25.0)
    elapsed = time.time() - started
    assert result["status"] == "error"
    assert "camera" in result["reason"]
    assert elapsed < 5.0, f"took {elapsed:.1f}s; should fail fast"


def test_enrollment_without_a_name_is_rejected(vs):
    assert vs.enroll_face("")["status"] == "error"


def test_training_an_expression_without_one_is_rejected(vs):
    assert vs.train_emotion("zakaria", "")["status"] == "error"


# ------------------------------------------------- CPU budget during a turn

def test_perception_slows_while_a_turn_is_answered(vs):
    """The board has four small cores and STT, the LLM wait and TTS all land on
    them at once. Full-rate detection during that window competes with the very
    thing the person is waiting through.

    Uses the production leave_timeout: the fixture's deliberately tiny 0.3 s
    leaves no headroom to throttle into, and correctly declines to."""
    vs.leave_timeout = 2.0            # the real default
    vs.fps = 4                        # the board's configured rate
    idle = vs._current_period()
    with vs.turn_in_progress():
        assert vs._current_period() > idle
    assert vs._current_period() == pytest.approx(idle)


def test_the_throttle_never_makes_the_feed_look_dead(vs):
    """Slowed, NOT paused. If the throttled rate fell outside leave_timeout, a
    vision tool called mid-turn would be told the camera is idle and the robot
    would claim it cannot see anyone -- while looking straight at them.

    Checked across configurations because a bare 1 fps constant satisfies this
    for the default 2.0 s timeout and silently breaks it for a shorter one."""
    for leave_timeout in (0.3, 1.0, 2.0, 5.0):
        vs.leave_timeout = leave_timeout
        with vs.turn_in_progress():
            assert vs._current_period() <= leave_timeout / 2.0 or \
                vs._current_period() == pytest.approx(1.0 / vs.fps)


def test_the_throttle_never_speeds_perception_up(vs):
    """A slow-configured camera must not be driven faster by a turn starting."""
    vs.fps = 1
    vs.leave_timeout = 20.0
    with vs.turn_in_progress():
        assert vs._current_period() == pytest.approx(1.0)


def test_the_throttle_is_released_even_if_the_turn_raises(vs):
    idle = vs._current_period()
    with pytest.raises(RuntimeError), vs.turn_in_progress():
        raise RuntimeError("turn blew up")
    assert vs._current_period() == pytest.approx(idle)
