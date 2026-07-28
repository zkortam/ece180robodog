#!/usr/bin/env python3
"""VisionService: an always-on perception loop over the existing face/emotion
models, plus the read-only tools an LLM calls to ask "who is here / how do they
feel / how have they felt lately".

It REUSES face_emotion.py (YuNet detect, SFace identity, MobileFaceNet emotion,
enrollments.json / emotions.json). It never touches the registration pipeline;
registration writes the DBs, this service reads them (reload_db() picks up new
enrollments live).

The service is safe to use WITHOUT a camera: call step(frame) directly with a
BGR numpy image (used by tests). start()/stop() drive the background thread.
"""
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import face_emotion as fe

# A face we have not classified yet. Distinct from "not_enabled", which means the
# emotion model is switched off -- reporting that for someone who merely just sat
# down makes the agent announce that emotion detection is disabled when it is not.
_UNCLASSIFIED = (None, 0.0, "none", "unknown", {})


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _position(cx, cy, w, h):
    hx = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
    vy = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
    return {"h": hx, "v": vy}


def _size_bucket(frac):
    return "small" if frac < 0.05 else "large" if frac > 0.20 else "medium"


# ---- V4L2 capability query (VIDIOC_QUERYCAP), used to find the real camera ----
#
# On the UNO Q's Qualcomm SoC, /dev/video* is NOT a list of cameras: the hardware
# video encoder and decoder claim nodes there too, and with a USB hub the webcam
# lands at whatever index is left over. Probing each node by opening it with
# OpenCV is both slow and unsafe -- opening a memory-to-memory encoder node can
# block, and a block inside the camera watcher thread is unrecoverable because
# that thread is the only thing that can ever restore vision.
#
# Asking the kernel directly is one ioctl, cannot hang, and needs no v4l2-ctl
# binary (which is not installed on every image).
_VIDIOC_QUERYCAP = 0x80685600          # _IOR('V', 0, struct v4l2_capability), 104 bytes
_CAP_VIDEO_CAPTURE = 0x00000001
_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
_CAP_VIDEO_M2M = 0x00008000
_CAP_VIDEO_M2M_MPLANE = 0x00004000
_CAP_DEVICE_CAPS = 0x80000000


def v4l2_capture_capability(path):
    """(is_camera, card_name) for a /dev/video* node, via one non-blocking ioctl.

    Returns (None, "") when the node cannot be queried at all, so callers can
    decide whether to fall back to probing rather than treating "unknown" as "no".
    """
    import fcntl
    import struct
    layout = "16s32s32sII I 3I"        # driver, card, bus_info, version, caps, device_caps, reserved
    try:
        with open(path, "rb", buffering=0) as node:
            buf = fcntl.ioctl(node, _VIDIOC_QUERYCAP, bytes(struct.calcsize(layout)))
    except (OSError, ValueError):
        return None, ""
    driver, card, _bus, _ver, caps, device_caps = struct.unpack(layout, buf)[:6]
    name = card.split(b"\0", 1)[0].decode(errors="replace").strip()
    # device_caps describes THIS node; capabilities describes the whole device,
    # which on a multi-node encoder wrongly advertises capture on every node.
    effective = device_caps if caps & _CAP_DEVICE_CAPS else caps
    captures = bool(effective & (_CAP_VIDEO_CAPTURE | _CAP_VIDEO_CAPTURE_MPLANE))
    transcodes = bool(effective & (_CAP_VIDEO_M2M | _CAP_VIDEO_M2M_MPLANE))
    del driver
    return (captures and not transcodes), name


def capture_device_indexes():
    """Camera indexes worth trying, lowest first, newest-plugged last.

    Nodes the kernel says are real capture devices come first; nodes that could
    not be queried are appended as a last resort so an unusual driver still gets
    a chance instead of the robot declaring itself blind.
    """
    from glob import glob
    cameras, unknown = [], []
    for path in sorted(glob("/dev/video*")):
        suffix = path.rsplit("video", 1)[1]
        if not suffix.isdigit():
            continue
        index = int(suffix)
        is_camera, name = v4l2_capture_capability(path)
        if is_camera:
            cameras.append((index, name))
        elif is_camera is None:
            unknown.append((index, name))
    return sorted(cameras) + sorted(unknown)


class VisionService(fe.EnrollmentStore):
    def __init__(
        self,
        detector_path=fe.DEFAULT_DETECTOR,
        recognizer_path=fe.DEFAULT_RECOGNIZER,
        db_path=fe.DEFAULT_DB,
        emotion_model=fe.DEFAULT_EMOTION_MODEL,
        emotion_db_path=fe.DEFAULT_EMOTION_DB,
        emotion_size=fe.DEFAULT_EMOTION_SIZE,
        camera=0,
        width=320,
        height=240,
        fps=4,
        emotion_interval=1.5,
        threshold=0.5,
        ring_seconds=300,
        ring_max=600,
        present_gap=1.5,
        leave_timeout=2.0,
    ):
        self.detector_path = Path(detector_path)
        self.recognizer_path = Path(recognizer_path)
        self.db_path = Path(db_path)
        self.emotion_model_path = Path(emotion_model) if emotion_model else None
        self.emotion_db_path = Path(emotion_db_path) if emotion_db_path else None
        self.emotion_size = emotion_size
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        # Seconds between expression inferences PER TRACKED FACE. This used to be
        # `emotion_every`, a frame count that nothing read: scheduling moved to a
        # per-track timer in step() but the old knob stayed in the constructor, in
        # start(), and in the reported config, so tuning it did nothing at all.
        self.emotion_interval = emotion_interval
        self.threshold = threshold
        self.ring_seconds = ring_seconds
        self.ring_max = ring_max
        self.present_gap = present_gap
        self.leave_timeout = leave_timeout

        # models (loaded once)
        self.detector = fe.create_detector(self.detector_path, width, height)
        self.recognizer = fe.create_recognizer(self.recognizer_path)
        self.db = fe.load_db(self.db_path)
        self.emotion = None
        if self.emotion_model_path and self.emotion_model_path.exists():
            self.emotion = fe.EmotionModel(self.emotion_model_path, emotion_size, fe.DEFAULT_LABELS)
        self.emotion_db = fe.load_emotion_db(self.emotion_db_path) if self.emotion_db_path else {}
        self._stamp_dbs()

        # state (guarded by lock)
        self.lock = threading.Lock()
        self.running = False
        self.external = False     # True = frames pushed in via submit_frame() (e.g. browser camera)
        # Does this deployment have a camera of its own? Declared up front by
        # main.py rather than inferred from `running`, because the answer must be
        # correct BEFORE the board has finished discovering its webcam -- a browser
        # that guesses "no" in that window opens its own camera and starts pushing
        # a competing feed that the board then has to reject.
        self.owns_camera = False
        self.started_at = None
        self.frames_processed = 0
        self.frame_w = width
        self.frame_h = height
        self.latest = []          # list[Observation] for the current frame
        self.latest_t = 0.0
        self.latest_frame = None  # last BGR frame, for the snapshot endpoint
        self.people = {}          # name -> {present, first_seen, last_seen, last_obs}
        self.ring = {}            # name -> deque[ExpressionSample]
        self.events = deque(maxlen=64)  # {t, name, event, identity_score}

        # internal tracking (not exposed)
        self._tracks = []         # per-face identity/emotion state; see _associate()
        self._next_track = 1
        self._frame_counter = 0
        self._enroll = None       # active voice-driven enrollment job, or None
        # Set while a conversation turn is being processed. The board has four small
        # cores and STT, the LLM wait, and TTS all land on them at once; perception
        # running at full rate during that window is competing with the very thing
        # the person is waiting for. Slowed rather than paused: the feed must stay
        # live enough that a vision tool called mid-turn still sees the present.
        self._turn_active = threading.Event()
        self._thread = None
        # Bumped on every start(). The camera loop clears `running` on its way out
        # only if it is still the current generation, so a thread that dies late
        # cannot switch off a loop that has already been restarted in its place.
        self._generation = 0
        self._stop = threading.Event()

    # ---- DB bridge (called after a new enrollment / emotion-train) ----
    def reload_db(self):
        with self.lock:
            self.db = fe.load_db(self.db_path)
            self.emotion_db = fe.load_emotion_db(self.emotion_db_path) if self.emotion_db_path else {}
            self._stamp_dbs()
        return {"identities": sorted(self.db.keys()),
                "emotion_people": sorted(self.emotion_db.keys())}

    def _reap(self, now):
        """Expire presence on READ, not only when a frame arrives.

        All expiry used to be driven from step(), so when frames stop (tab closed,
        phone locked, camera denied, board throttled) the last frame's state froze
        and was reported as live indefinitely -- the agent would insist someone is
        in the room an hour after they left. Callers hold self.lock."""
        self._expire_absent(now)

    def _feed_dead(self, now):
        """True when no frame has arrived recently enough to describe the present."""
        return (not self.latest_t) or (now - self.latest_t) > self.leave_timeout

    def _observations(self, now):
        """Faces we can honestly claim are in front of the camera RIGHT NOW.
        A frozen feed is not a view. Callers hold self.lock."""
        return [] if self._feed_dead(now) else self.latest

    def _resolve(self, name):
        """Canonical stored key for a spoken name, matched case-insensitively.

        The CLI enrolls "chris" (README says lowercase) but the LLM hears a name and
        writes "Chris". Without this they are two different people: lookups miss, and
        enrolling forks a duplicate identity of the same face. Callers may hold
        self.lock, so this never takes it."""
        n = str(name or "").strip()
        if not n or n in self.db or n in self.ring:
            return n
        low = n.lower()
        for k in list(self.db.keys()) + list(self.ring.keys()):
            if k.lower() == low:
                return k
        return n

    # ---- lifecycle ----
    def start(self, camera=None, fps=None, emotion_interval=None, threshold=None, external=False):
        """external=True: no server camera thread; frames arrive via submit_frame().

        Every tuning argument defaults to None and means "leave it alone". Callers
        that want the current value must not pass one."""
        with self.lock:
            if self.running:
                return {"running": True, "already_running": True,
                        "config": self._config(), "started_at": self.started_at}
            if camera is not None:
                self.camera = camera
            if fps is not None:
                self.fps = fps
            if emotion_interval is not None:
                self.emotion_interval = emotion_interval
            if threshold is not None:
                self.threshold = threshold
            self._stop.clear()
            self.running = True
            self.external = external
            self.started_at = time.time()
            self._generation += 1
            generation = self._generation
        if not external:
            self._thread = threading.Thread(target=self._loop, args=(generation,),
                                            daemon=True)
            self._thread.start()
        return {"running": True, "already_running": False, "external": external,
                "config": self._config(), "started_at": self.started_at}

    def submit_frame(self, frame):
        """Feed one externally-captured BGR frame (browser camera path).

        Refused when the board is driving its own camera. Two frame sources into
        one perception loop is not a merge, it is a fight: the tracker would see
        the room and the laptop's webcam on alternating frames, so association
        thrashes, identities flip, and presence flickers between two realities.
        The robot's own eye wins -- the browser is a viewer in that mode, and
        /api/vision/snapshot.jpg shows it what the robot is actually looking at."""
        if not self.running:
            return {"ok": False, "reason": "not watching"}
        if not self.external:
            return {"ok": False, "reason": "board camera is active", "frame_source": "board"}
        observations = self.step(frame)
        # Count what THIS frame produced. Reading self.latest afterwards races with
        # a concurrently submitted frame and can report the other one's face count.
        return {"ok": True, "num_faces": len(observations)}

    def stop(self):
        with self.lock:
            was = self.running
            self.running = False
            uptime = time.time() - self.started_at if self.started_at else 0.0
            frames = self.frames_processed
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        return {"running": False, "was_running": was,
                "uptime_seconds": round(uptime, 1), "frames_processed": frames}

    # Frame rate while a turn is in flight. Must keep the feed inside
    # leave_timeout (2.0s) or a vision tool called mid-turn would be told the
    # camera is idle and the robot would claim it cannot see anyone.
    BUSY_FPS = 1

    @contextmanager
    def turn_in_progress(self):
        """Throttle perception for the duration of a conversation turn.

        Reentrant-safe for the single-turn-at-a-time design: turns are serialized
        by the agent's turn lock, so there is never more than one holder.
        """
        self._turn_active.set()
        try:
            yield
        finally:
            self._turn_active.clear()

    def _current_period(self):
        normal = 1.0 / max(1, self.fps)
        if not self._turn_active.is_set():
            return normal
        # Derived from leave_timeout, not a bare constant: the invariant "a
        # throttled feed still counts as live" must hold for ANY configuration.
        # A fixed 1 fps silently breaks it the moment leave_timeout is tuned below
        # a second, and the symptom would be the robot insisting it cannot see
        # anyone while looking straight at them. Never faster than normal either.
        return max(normal, min(1.0 / self.BUSY_FPS, self.leave_timeout / 2.0))

    def _config(self):
        return {"camera": self.camera, "fps": self.fps,
                "emotion_interval": self.emotion_interval, "threshold": self.threshold}

    # A USB camera that is unplugged keeps returning (False, None) forever rather
    # than raising. Without a ceiling the loop spun at fps producing nothing while
    # `running` stayed True, so the board's camera watcher never re-discovered the
    # device: one accidental unplug blinded the robot until someone restarted it.
    # Measured in seconds, not frames, so the behavior does not change with fps.
    DEAD_FEED_SECONDS = 3.0
    REOPEN_ATTEMPTS = 2

    def _open_camera(self):
        # fe.open_camera raises SystemExit, which is a BaseException and therefore
        # NOT caught by `except Exception` -- the comment here used to claim it was.
        # The escaping SystemExit killed this thread with `running` still True, so
        # the board's camera_watch (which only re-discovers while `running` is
        # False) never retried: one failed open left the robot blind until someone
        # restarted the service.
        try:
            # Pass our target rate so the camera itself throttles: see open_camera.
            return fe.open_camera(self.camera, self.width, self.height, fps=self.fps)
        except (Exception, SystemExit) as e:
            print(f"[vision] camera unavailable: {e}", file=sys.stderr)
            return None

    def _loop(self, generation=0):
        """Run the camera loop, and ALWAYS mark perception stopped on the way out.

        The board's camera_watch thread re-discovers a webcam only while `running`
        is False, so any exit path that leaves it True -- an unexpected exception,
        a failed open, a dying USB device -- strands the robot blind with no
        recovery. Clearing it in a finally makes every exit recoverable."""
        try:
            self._run_camera(generation)
        except BaseException as e:      # a blind robot must never be a silent one
            print(f"[vision] camera loop stopped: {type(e).__name__}: {e}",
                  file=sys.stderr)
        finally:
            with self.lock:
                # Only if we are still the current loop: a late-dying thread must
                # not switch off a loop that has already been restarted.
                if generation == self._generation:
                    self.running = False

    def _run_camera(self, generation):
        cap = self._open_camera()
        if cap is None:
            return
        # Recomputed every iteration, not cached: it drops while a turn is being
        # answered so perception stops competing with STT and TTS for the cores.
        last_good = time.time()
        healthy_since = last_good
        reopens = 0
        try:
            while not self._stop.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if ok and frame is not None:
                    if t0 - last_good >= self.DEAD_FEED_SECONDS:
                        healthy_since = t0            # first frame back after a gap
                    last_good = t0
                    # Forgive the retry budget only once the feed has been healthy
                    # for a while. Clearing it on the first frame lets a camera that
                    # yields one frame per open reopen forever without ever
                    # escalating -- which is exactly how a dying USB camera behaves.
                    if t0 - healthy_since >= self.DEAD_FEED_SECONDS:
                        reopens = 0
                    try:
                        self.step(frame)
                    except Exception as e:            # one bad frame must never kill perception
                        print(f"[vision] frame error: {e}", file=sys.stderr)
                elif t0 - last_good >= self.DEAD_FEED_SECONDS:
                    cap.release()
                    reopens += 1
                    if reopens > self.REOPEN_ATTEMPTS:
                        print(f"[vision] camera {self.camera} stopped delivering frames; "
                              "releasing it so it can be re-discovered", file=sys.stderr)
                        return                    # _loop's finally clears `running`
                    print(f"[vision] camera {self.camera} went quiet; reopening "
                          f"({reopens}/{self.REOPEN_ATTEMPTS})", file=sys.stderr)
                    self._stop.wait(1.0)
                    cap = self._open_camera()
                    if cap is None:
                        return
                    last_good = healthy_since = time.time()
                dt = time.time() - t0
                period = self._current_period()
                if dt < period:
                    self._stop.wait(period - dt)
        finally:
            if cap is not None:
                cap.release()

    # ---- the per-frame perception step (also used directly by tests) ----
    def step(self, frame):
        h, w = frame.shape[:2]
        observations = []
        seen_names = set()
        # All cv2 model access (detector/recognizer/emotion) stays under one lock:
        # the C++ objects are stateful and NOT thread-safe, and browser frames arrive
        # concurrently with tool reads under Flask threaded=True.
        with self.lock:
            # Stamp AFTER the wait, not before: two frames can queue on this lock for
            # the whole detect+embed+emotion pass, and a pre-lock timestamp lets the
            # loser write an older `now` (and older observations) over the winner --
            # as_of jumps backwards and a face that already left is resurrected.
            now = time.time()
            if self._frame_counter % 12 == 0:
                self._refresh_dbs()
            faces = fe.detect_faces(self.detector, frame)
            self._frame_counter += 1
            enroll_best = None
            enroll_area = -1.0
            assigned_tracks = set()
            for face in faces:
                x, y, fw, fh = [int(v) for v in face[:4]]
                det_score = float(face[14]) if len(face) > 14 else 1.0
                embedding, _ = fe.face_embedding(self.recognizer, frame, face)
                if embedding is None:
                    continue
                candidate_name, candidate_score = fe.best_match(self.db, embedding, self.threshold)
                if fw * fh > enroll_area:
                    enroll_area, enroll_best = fw * fh, (embedding, face)

                track = self._associate([x, y, fw, fh], embedding, assigned_tracks)
                assigned_tracks.add(track["track_id"])
                name, score = self._stable_identity(track, candidate_name, float(candidate_score))
                # Schedule emotion per visible face. A global modulo skipped a face
                # whenever it happened not to be present on that one frame and made
                # the UI alternate between stale/empty emotion states.
                run_emotion = (
                    self.emotion is not None
                    and (now - track["last_emotion_at"] >= self.emotion_interval)
                )
                emotion_label, emotion_score, emotion_source, sentiment, probs = self._emotion_for(
                    frame, face, name, track, run_emotion, now)

                cx, cy = x + fw / 2.0, y + fh / 2.0
                frac = (fw * fh) / float(w * h) if w and h else 0.0
                obs = {
                    "track_id": track["track_id"],
                    "name": name,
                    "identity_score": round(float(score), 4),
                    "bbox": [x, y, fw, fh],
                    "det_score": round(det_score, 4),
                    "emotion": emotion_label,
                    "emotion_score": round(float(emotion_score), 4),
                    "emotion_source": emotion_source,
                    "sentiment": sentiment,
                    "probs": probs,
                    "position": _position(cx, cy, w, h),
                    "size_frac": round(frac, 4),
                    "size_bucket": _size_bucket(frac),
                    "t": now,
                }
                observations.append(obs)
                track["bbox"] = [x, y, fw, fh]
                track["last_seen"] = now

                if name != "unknown":
                    seen_names.add(name)
                    self._update_presence(name, now, float(score))
                    if emotion_label is not None:
                        self._push_expression(name, now, emotion_label, float(emotion_score),
                                              emotion_source, sentiment, probs)

            if self._enroll is not None and not self._enroll["done"] and enroll_best is not None:
                self._handle_enroll(frame, enroll_best, now)
            self._expire_absent(now)
            self.latest = observations
            self.latest_t = now
            # Held by reference, not copied: callers hand us a freshly decoded or
            # freshly captured array and never mutate it afterwards, and a copy per
            # frame is pure waste on a board this small.
            self.latest_frame = frame
            self.frame_w, self.frame_h = w, h
            self.frames_processed += 1
        return observations

    def snapshot_jpeg(self, quality=70):
        """The most recent frame as JPEG bytes, or None if there is nothing live.

        Lets a laptop browser see through the ROBOT's camera instead of opening
        its own. Encoding happens per request, so a frame nobody asks for costs
        nothing."""
        with self.lock:
            frame = self.latest_frame
            stale = self._feed_dead(time.time())
        if frame is None or stale:
            return None
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None

    def frame_source(self):
        """Which camera is authoritative: 'board' or 'browser'.

        'board' only when this deployment actually has its own camera. With
        --no-camera there is no vision at all, and the browser must not be told to
        wait on a snapshot that will never arrive."""
        return "board" if (self.owns_camera and not self.external) else "browser"

    # ---- internal helpers (assume lock held) ----
    def _associate(self, bbox, embedding, assigned_tracks=None):
        """Associate a detection using motion *and* face appearance.

        IoU alone swaps identities when people cross or when somebody steps into
        a recently vacated position. Appearance alone can be noisy under pose and
        lighting. Requiring a plausible combination handles both, and the assigned
        set guarantees two faces in one frame can never share a track.
        """
        assigned_tracks = assigned_tracks or set()
        best, best_affinity = None, -1.0
        for tr in self._tracks:
            if tr["track_id"] in assigned_tracks:
                continue
            i = _iou(tr["bbox"], bbox)
            previous = tr.get("embedding")
            similarity = float(np.dot(previous, embedding)) if previous is not None else 0.0
            # A large position jump is allowed only with strong appearance; a box
            # overlap is allowed only when the faces are at least plausibly alike.
            if similarity < 0.35 or (i < 0.10 and similarity < 0.55):
                continue
            affinity = 0.55 * i + 0.45 * max(0.0, similarity)
            if affinity > best_affinity:
                best, best_affinity = tr, affinity
        if best is None:
            best = {
                "track_id": self._next_track,
                "bbox": bbox,
                "name": "unknown",
                "identity_score": 0.0,
                "identity_misses": 0,
                "candidate_name": None,
                "candidate_hits": 0,
                "candidate_at": 0.0,
                "embedding": np.asarray(embedding, dtype=np.float32).copy(),
                "last_emotion": _UNCLASSIFIED,
                "last_emotion_at": 0.0,
                "last_seen": time.time(),
            }
            self._next_track += 1
            self._tracks.append(best)
        else:
            # Smooth appearance enough to survive one imperfect crop while still
            # adapting to head rotation. Normalize because matching is cosine/dot.
            mixed = 0.75 * best["embedding"] + 0.25 * np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(mixed))
            best["embedding"] = mixed / norm if norm > 1e-8 else np.asarray(embedding).copy()
        return best

    def _stable_identity(self, track, candidate_name, candidate_score):
        """Apply temporal hysteresis to noisy per-frame face matches.

        A known identity is promoted only after repeated evidence and survives a
        few weak frames. Switching directly on one cosine score caused visible
        flicker and, worse, could attach another person's emotion profile.
        """
        current = track["name"]
        now = time.time()
        if candidate_name == current and current != "unknown":
            old = float(track["identity_score"])
            track["identity_score"] = candidate_score if old <= 0 else old * 0.7 + candidate_score * 0.3
            track["identity_misses"] = 0
            track["candidate_name"] = None
            track["candidate_hits"] = 0
            track["candidate_at"] = 0.0
            return current, track["identity_score"]

        if candidate_name == "unknown":
            track["identity_misses"] += 1
            # Keep a confirmed identity through short blur/profile/lighting dips.
            if current != "unknown" and track["identity_misses"] < 6:
                track["identity_score"] *= 0.96
                return current, track["identity_score"]
            if current != "unknown":
                self._set_track_identity(track, "unknown", 0.0)
            # Do not throw away partial evidence after one weak frame, but never
            # combine two unrelated sightings far apart in time.
            if track["candidate_name"] and now - track["candidate_at"] > 3.0:
                track["candidate_name"] = None
                track["candidate_hits"] = 0
                track["candidate_at"] = 0.0
            return "unknown", 0.0

        track["identity_misses"] = 0
        if track["candidate_name"] == candidate_name and now - track["candidate_at"] <= 3.0:
            track["candidate_hits"] += 1
        else:
            track["candidate_name"] = candidate_name
            track["candidate_hits"] = 1
        track["candidate_at"] = now

        required = 2 if current == "unknown" else 3
        if track["candidate_hits"] >= required:
            self._set_track_identity(track, candidate_name, candidate_score)
            return candidate_name, candidate_score
        return (current, track["identity_score"]) if current != "unknown" else ("unknown", 0.0)

    @staticmethod
    def _set_track_identity(track, name, score):
        if track["name"] != name:
            # Never carry an expression (especially a personalized one) across
            # people when a new face occupies the same screen position.
            track["last_emotion"] = _UNCLASSIFIED
            track["last_emotion_at"] = 0.0
        track["name"] = name
        track["identity_score"] = float(score)
        track["identity_misses"] = 0
        track["candidate_name"] = None
        track["candidate_hits"] = 0
        track["candidate_at"] = 0.0

    def _emotion_for(self, frame, face, name, track, run_emotion, now):
        if self.emotion is None:
            return None, 0.0, "none", "not_enabled", {}
        if not run_emotion:
            lbl, sc, src, sent, probs = track["last_emotion"]
            return lbl, sc, src, sent, probs
        # Mark the attempt before inference so a malformed crop does not make us
        # retry the expensive model on every camera frame.
        track["last_emotion_at"] = now
        prob_vec = self.emotion.probabilities(frame, face)
        if prob_vec is None:
            return track["last_emotion"]
        # zip() would silently drop classes if a swapped-in model emitted more than
        # the seven labels we know. Name the extras rather than hide them.
        probs = {(fe.DEFAULT_LABELS[i] if i < len(fe.DEFAULT_LABELS) else f"class_{i}"):
                 round(float(p), 4) for i, p in enumerate(prob_vec)}
        protos = self.emotion_db.get(name) if name != "unknown" else None
        if protos:
            lbl, sc = fe.classify_personal(protos, prob_vec)
            src = "personal"
        else:
            i = int(np.argmax(prob_vec))
            lbl = fe.DEFAULT_LABELS[i] if i < len(fe.DEFAULT_LABELS) else f"class_{i}"
            sc = float(prob_vec[i])
            src = "generic"
        sent = fe.sentiment_from_emotion(lbl)
        track["last_emotion"] = (lbl, sc, src, sent, probs)
        return lbl, sc, src, sent, probs

    def _update_presence(self, name, now, score):
        p = self.people.get(name)
        if p is None or not p["present"]:
            gap_ok = p is None or (now - p["last_seen"]) > self.present_gap
            self.people[name] = {"present": True,
                                 "first_seen": now if p is None else p["first_seen"],
                                 "last_seen": now, "last_obs": now}
            if gap_ok:
                self.events.append({"t": now, "name": name, "event": "enter",
                                    "identity_score": round(score, 4)})
        else:
            p["last_seen"] = now
            p["last_obs"] = now

    def _expire_absent(self, now):
        for name, p in self.people.items():
            if p["present"] and (now - p["last_seen"]) > self.leave_timeout:
                p["present"] = False
                self.events.append({"t": now, "name": name, "event": "leave",
                                    "identity_score": 0.0})
        # prune stale tracks
        self._tracks = [t for t in self._tracks if now - t["last_seen"] <= self.leave_timeout * 2]

    def _push_expression(self, name, now, label, score, source, sentiment, probs):
        dq = self.ring.get(name)
        if dq is None:
            dq = deque(maxlen=self.ring_max)
            self.ring[name] = dq
        dq.append({"t": now, "emotion": label, "emotion_score": round(score, 4),
                   "sentiment": sentiment, "source": source, "probs": probs})
        cutoff = now - self.ring_seconds
        while dq and dq[0]["t"] < cutoff:
            dq.popleft()

    # ---- voice-driven enrollment (writes the DBs; reuses the single camera) ----
    def _handle_enroll(self, frame, pack, now):
        e = self._enroll
        if now - e.get("last", 0.0) < 0.12:      # throttle so samples are diverse
            return
        embedding, face = pack
        if e["kind"] == "face":
            if embedding is not None:
                e["buf"].append(embedding)
                e["last"] = now
        elif self.emotion is not None:
            probs = self.emotion.probabilities(frame, face)
            if probs is not None:
                e["buf"].append(np.asarray(probs, dtype=np.float32))
                e["last"] = now
        if len(e["buf"]) >= e["samples"]:
            self._finalize_enroll()

    def _finalize_enroll(self):
        e = self._enroll
        self._refresh_dbs()      # merge onto the newest roster; never overwrite it
        if e["kind"] == "face":
            m = np.mean(np.stack(e["buf"]), axis=0)
            self.db[e["name"]] = (m / np.linalg.norm(m)).astype(np.float32)
            self._save_dbs(identities=True)
        else:
            m = np.mean(np.stack(e["buf"]), axis=0)
            self.emotion_db.setdefault(e["name"], {})[e["expression"]] = m.astype(np.float32)
            self._save_dbs(expressions=True)
        e["done"] = True
        e["status"] = "saved"

    # ================= TOOLS (read-only; return JSON-able dicts) =================
    def start_watching(self, camera=None, fps=None):
        """Turn perception on, keeping every tuned setting the caller did not name.

        These defaults MUST stay None. Passing concrete literals here (they used to
        be camera=0, fps=4, threshold=0.5) meant any call rewrote the running
        configuration: the identity threshold dropped from its tuned value back to
        0.5, and on the board -- where camera_watch discovers the webcam at whatever
        index it enumerated as, often not 0 -- the camera index was reset to 0 and
        perception never came back."""
        return self.start(camera=camera, fps=fps, external=self.external)

    def stop_watching(self):
        return self.stop()

    def enroll_status(self):
        with self.lock:
            e = self._enroll
            if not e:
                return {"active": False}
            return {"active": not e["done"], "kind": e["kind"], "name": e["name"],
                    "expression": e.get("expression"), "captured": len(e["buf"]),
                    "target": e["samples"]}

    def enroll_face(self, name, samples=16, timeout=25.0):
        """Capture ~samples face shots of the person in view and save them under name."""
        name = self._resolve(name)      # re-enrolling "Chris" must update "chris", not fork a twin
        if not name:
            return {"status": "error", "reason": "no name given"}
        if not self.running:
            self.start(external=self.external)
        with self.lock:
            self._enroll = {"kind": "face", "name": name, "samples": int(samples),
                            "buf": [], "done": False, "status": "capturing", "last": 0.0}
        return self._await_enroll(name, timeout)

    def train_emotion(self, name, expression, samples=16, timeout=25.0):
        """Capture ~samples shots of name making expression and save the personal prototype."""
        name = self._resolve(name)
        expression = str(expression or "").strip().lower()
        if not name or not expression:
            return {"status": "error", "reason": "need both name and expression"}
        if self.emotion is None:
            return {"status": "error", "reason": "emotion model not loaded"}
        if not self.running:
            self.start(external=self.external)
        with self.lock:
            self._enroll = {"kind": "emotion", "name": name, "expression": expression,
                            "samples": int(samples), "buf": [], "done": False,
                            "status": "capturing", "last": 0.0}
        return self._await_enroll(name, timeout, expression)

    def _await_enroll(self, name, timeout, expression=None):
        deadline = time.time() + timeout
        captured = 0
        while time.time() < deadline:
            with self.lock:
                e = self._enroll
                if e is None:
                    break
                captured = len(e["buf"])
                if e["done"]:
                    self._enroll = None
                    r = {"status": "saved", "name": name, "samples": captured}
                    if expression:
                        r["expression"] = expression
                    return r
                # Fail fast when no frame can EVER arrive: perception is stopped
                # and nobody is pushing frames in. Blocking the full timeout here
                # holds the turn lock, and on the standalone robot that is the
                # whole conversation -- 25 seconds of deafness, then a message
                # blaming the user's lighting for a missing camera.
                if not self.running:
                    self._enroll = None
                    r = {"status": "error", "name": name,
                         "reason": "no camera is running, so I can't see anyone to enroll"}
                    if expression:
                        r["expression"] = expression
                    return r
            time.sleep(0.1)
        with self.lock:
            self._enroll = None
        r = {"status": "timeout", "name": name, "captured": captured,
             "note": "I couldn't see a face clearly. Center yourself, get closer, and make sure it's well lit."}
        if expression:
            r["expression"] = expression
        return r

    def list_enrolled(self):
        with self.lock:
            emo = fe.load_emotion_db(self.emotion_db_path) if self.emotion_db_path else {}
            people = []
            for name in sorted(self.db.keys()):
                people.append({"name": name, "face_enrolled": True,
                               "personal_expressions": sorted(emo.get(name, {}).keys())})
        return {"people": people}

    def who_is_in_view(self, min_identity_score=0.0):
        with self.lock:
            now = time.time()
            self._reap(now)
            obs = self._observations(now)
            known, unknown = [], 0
            for o in obs:
                if o["name"] == "unknown":
                    unknown += 1
                elif o["identity_score"] >= min_identity_score:
                    known.append({"name": o["name"], "identity_score": o["identity_score"],
                                  "bbox": o["bbox"]})
            return {"known": known, "unknown_count": unknown, "num_faces": len(obs),
                    "as_of": round(self.latest_t, 2),
                    "stale_seconds": round(now - self.latest_t, 2) if self.latest_t else None,
                    "watching": self.running}

    def describe_scene(self, include_probs=False):
        with self.lock:
            now = time.time()
            self._reap(now)
            obs = self._observations(now)
            people = []
            for o in obs:
                entry = {k: o[k] for k in ("name", "identity_score", "emotion", "emotion_score",
                                           "emotion_source", "sentiment", "position",
                                           "size_bucket", "size_frac", "bbox")}
                if include_probs:
                    entry["probs"] = o["probs"]
                people.append(entry)
            # stale_seconds, not just as_of: an absolute epoch tells the model nothing
            # it can compare against "now", so it cannot tell a live view from a dead one.
            return {"as_of": round(self.latest_t, 2), "frame_width": self.frame_w,
                    "frame_height": self.frame_h, "num_faces": len(obs),
                    "stale_seconds": round(now - self.latest_t, 2) if self.latest_t else None,
                    "feed_live": not self._feed_dead(now),
                    "watching": self.running, "people": people}

    def get_person_emotion(self, name):
        with self.lock:
            self._reap(time.time())
            name = self._resolve(name)
            present = self.people.get(name, {}).get("present", False)
            dq = self.ring.get(name)
            if not dq:
                return {"found": False, "present": present, "name": name}
            s = dq[-1]
            return {"found": True, "present": present, "name": name,
                    "emotion": s["emotion"], "emotion_score": s["emotion_score"],
                    "emotion_source": s["source"], "sentiment": s["sentiment"],
                    "probs": s["probs"], "sample_age_seconds": round(time.time() - s["t"], 2)}

    def emotion_timeline(self, name, since_seconds=60, max_points=50):
        with self.lock:
            name = self._resolve(name)
            dq = self.ring.get(name)
            now = time.time()
            if not dq:
                return {"found": False, "name": name, "sample_count": 0,
                        "window_seconds": since_seconds}
            cutoff = now - since_seconds
            samples = [s for s in dq if s["t"] >= cutoff]
            if not samples:
                return {"found": False, "name": name, "sample_count": 0,
                        "window_seconds": since_seconds}
            emo_counts, sent_counts = {}, {}
            for s in samples:
                emo_counts[s["emotion"]] = emo_counts.get(s["emotion"], 0) + 1
                sent_counts[s["sentiment"]] = sent_counts.get(s["sentiment"], 0) + 1
            n = len(samples)
            dominant = max(emo_counts, key=emo_counts.get)
            step = max(1, n // max(1, max_points))
            series = [{"t": round(s["t"], 2), "emotion": s["emotion"],
                       "sentiment": s["sentiment"], "score": s["emotion_score"]}
                      for s in samples[::step]]
            return {"found": True, "name": name, "sample_count": n,
                    "window_seconds": since_seconds, "dominant_emotion": dominant,
                    "emotion_fractions": {k: round(v / n, 3) for k, v in emo_counts.items()},
                    "sentiment_fractions": {k: round(v / n, 3) for k, v in sent_counts.items()},
                    "series": series}

    def presence_events(self, since_seconds=120):
        with self.lock:
            now = time.time()
            self._reap(now)      # else now_present never empties once frames stop
            cutoff = now - since_seconds
            events = [e for e in self.events if e["t"] >= cutoff]
            now_present = sorted([n for n, p in self.people.items() if p["present"]])
            return {"now_present": now_present,
                    "feed_live": not self._feed_dead(now),
                    "events": [{"t": round(e["t"], 2), "name": e["name"],
                                "event": e["event"], "identity_score": e["identity_score"]}
                               for e in events]}


if __name__ == "__main__":
    import json
    vs = VisionService()
    print("VisionService ready. Enrolled:", json.dumps(vs.list_enrolled(), indent=2))
    print("Starting watch (Ctrl-C to stop)...")
    vs.start()
    try:
        while True:
            time.sleep(2.0)
            print(json.dumps(vs.describe_scene(), indent=2))
    except KeyboardInterrupt:
        print(json.dumps(vs.stop(), indent=2))
