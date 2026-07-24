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


class VisionService:
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
        emotion_every=8,
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
        self.emotion_every = emotion_every
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
        self._db_mtime = self.db_path.stat().st_mtime_ns if self.db_path.exists() else 0
        self._emotion_db_mtime = (
            self.emotion_db_path.stat().st_mtime_ns
            if self.emotion_db_path and self.emotion_db_path.exists() else 0
        )

        # state (guarded by lock)
        self.lock = threading.Lock()
        self.running = False
        self.external = False     # True = frames pushed in via submit_frame() (e.g. browser camera)
        self.started_at = None
        self.frames_processed = 0
        self.frame_w = width
        self.frame_h = height
        self.latest = []          # list[Observation] for the current frame
        self.latest_t = 0.0
        self.people = {}          # name -> {present, first_seen, last_seen, last_obs}
        self.ring = {}            # name -> deque[ExpressionSample]
        self.events = deque(maxlen=64)  # {t, name, event, identity_score}

        # internal tracking (not exposed)
        self._tracks = []         # per-face identity/emotion state; see _associate()
        self._next_track = 1
        self._frame_counter = 0
        self._enroll = None       # active voice-driven enrollment job, or None
        self._thread = None
        self._stop = threading.Event()

    # ---- DB bridge (called after a new enrollment / emotion-train) ----
    def reload_db(self):
        with self.lock:
            self.db = fe.load_db(self.db_path)
            self.emotion_db = fe.load_emotion_db(self.emotion_db_path) if self.emotion_db_path else {}
            self._db_mtime = self.db_path.stat().st_mtime_ns if self.db_path.exists() else 0
            self._emotion_db_mtime = (
                self.emotion_db_path.stat().st_mtime_ns
                if self.emotion_db_path and self.emotion_db_path.exists() else 0
            )
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
    def start(self, camera=None, fps=None, emotion_every=None, threshold=None, external=False):
        """external=True: no server camera thread; frames arrive via submit_frame()."""
        with self.lock:
            if self.running:
                return {"running": True, "already_running": True,
                        "config": self._config(), "started_at": self.started_at}
            if camera is not None:
                self.camera = camera
            if fps is not None:
                self.fps = fps
            if emotion_every is not None:
                self.emotion_every = emotion_every
            if threshold is not None:
                self.threshold = threshold
            self._stop.clear()
            self.running = True
            self.external = external
            self.started_at = time.time()
        if not external:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return {"running": True, "already_running": False, "external": external,
                "config": self._config(), "started_at": self.started_at}

    def submit_frame(self, frame):
        """Feed one externally-captured BGR frame (browser camera path)."""
        if not self.running:
            return {"ok": False, "reason": "not watching"}
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

    def _config(self):
        return {"camera": self.camera, "fps": self.fps,
                "emotion_every": self.emotion_every, "threshold": self.threshold}

    # A USB camera that is unplugged keeps returning (False, None) forever rather
    # than raising. Without a ceiling the loop spun at fps producing nothing while
    # `running` stayed True, so the board's camera watcher never re-discovered the
    # device: one accidental unplug blinded the robot until someone restarted it.
    # Measured in seconds, not frames, so the behavior does not change with fps.
    DEAD_FEED_SECONDS = 3.0
    REOPEN_ATTEMPTS = 2

    def _open_camera(self):
        try:
            return fe.open_camera(self.camera, self.width, self.height)
        except Exception as e:                        # SystemExit included
            print(f"[vision] camera unavailable: {e}", file=sys.stderr)
            return None

    def _loop(self):
        cap = self._open_camera()
        if cap is None:
            with self.lock:
                self.running = False
            return
        period = 1.0 / max(1, self.fps)
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
                        with self.lock:
                            self.running = False
                        return
                    print(f"[vision] camera {self.camera} went quiet; reopening "
                          f"({reopens}/{self.REOPEN_ATTEMPTS})", file=sys.stderr)
                    self._stop.wait(1.0)
                    cap = self._open_camera()
                    if cap is None:
                        with self.lock:
                            self.running = False
                        return
                    last_good = healthy_since = time.time()
                dt = time.time() - t0
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
                db_mtime = self.db_path.stat().st_mtime_ns if self.db_path.exists() else 0
                emotion_mtime = (
                    self.emotion_db_path.stat().st_mtime_ns
                    if self.emotion_db_path and self.emotion_db_path.exists() else 0
                )
                if db_mtime != self._db_mtime:
                    self.db = fe.load_db(self.db_path)
                    self._db_mtime = db_mtime
                if emotion_mtime != self._emotion_db_mtime:
                    self.emotion_db = fe.load_emotion_db(self.emotion_db_path)
                    self._emotion_db_mtime = emotion_mtime
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
                    and (now - track["last_emotion_at"] >= 1.5)
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
            self.frame_w, self.frame_h = w, h
            self.frames_processed += 1
        return observations

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
        if e["kind"] == "face":
            m = np.mean(np.stack(e["buf"]), axis=0)
            self.db[e["name"]] = (m / np.linalg.norm(m)).astype(np.float32)
            fe.save_db(self.db_path, self.db)
        else:
            m = np.mean(np.stack(e["buf"]), axis=0)
            self.emotion_db.setdefault(e["name"], {})[e["expression"]] = m.astype(np.float32)
            if self.emotion_db_path:
                fe.save_emotion_db(self.emotion_db_path, self.emotion_db)
        e["done"] = True
        e["status"] = "saved"

    # ================= TOOLS (read-only; return JSON-able dicts) =================
    def start_watching(self, camera=0, fps=4, emotion_every=8, threshold=0.5):
        return self.start(camera=camera, fps=fps, emotion_every=emotion_every, threshold=threshold)

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
