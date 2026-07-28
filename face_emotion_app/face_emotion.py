#!/usr/bin/env python3
import argparse
import base64
import json
import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTOR = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_RECOGNIZER = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
DEFAULT_EMOTION_MODEL = ROOT / "models" / "facial_expression_recognition_mobilefacenet_2022july.onnx"
DEFAULT_EMOTION_SIZE = 112
DEFAULT_DB = ROOT / "data" / "enrollments.json"
DEFAULT_EMOTION_DB = ROOT / "data" / "emotions.json"
DEFAULT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
POSITIVE_LABELS = {"happy", "surprise"}
NEGATIVE_LABELS = {"sad", "fear", "disgust", "angry"}
EMOTION_ALIASES = {
    "fearful": "fear",
    "surprised": "surprise",
    "none": "not_enabled",
    "n/a": "not_enabled",
    "na": "not_enabled",
}


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")


def create_detector(model_path: Path, width: int, height: int):
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise SystemExit("OpenCV build lacks FaceDetectorYN_create. Install a newer OpenCV.")
    return cv2.FaceDetectorYN_create(str(model_path), "", (width, height), 0.75, 0.3, 5000)


def create_recognizer(model_path: Path):
    if not hasattr(cv2, "FaceRecognizerSF_create"):
        raise SystemExit("OpenCV build lacks FaceRecognizerSF_create. Install a newer OpenCV.")
    return cv2.FaceRecognizerSF_create(str(model_path), "")


def open_camera(camera: int, width: int, height: int, fps: int | None = None):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera index {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        # Ask the CAMERA to slow down rather than capturing 30 fps and throwing
        # most of it away. Two reasons, both specific to a robot where the webcam,
        # microphone and speaker share one USB hub:
        #
        #  * Staleness. V4L2 queues frames in the driver. Reading at 4 fps from a
        #    camera producing 30 means read() hands back a queued frame from up to
        #    a few hundred ms ago -- the robot answers about a scene that has
        #    already moved on, and identity tracking associates against a stale
        #    position.
        #  * Bandwidth. Those discarded frames still cross the shared bus and cost
        #    the same USB budget the audio stream needs. Starving `arecord` is what
        #    produces "microphone stopped returning audio".
        cap.set(cv2.CAP_PROP_FPS, fps)
    # Keep the queue shallow so whatever the camera does deliver is the newest
    # frame available. Not honoured by every V4L2 driver, hence the FPS request
    # above rather than relying on this alone; harmless where it is ignored.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def detect_faces(detector, frame):
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return list(faces)


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: float(f[2] * f[3]))


def enrollment_guidance(frame, face, required_pose="center"):
    """Check framing and a coarse five-point head pose before accepting a sample.

    YuNet supplies eye, nose, and mouth landmarks. Ratios make this independent of
    camera resolution and are intentionally forgiving: the goal is useful variety,
    not clinical head-pose estimation.
    """
    height, width = frame.shape[:2]
    x, y, fw, fh = [float(v) for v in face[:4]]
    cx, cy = x + fw / 2.0, y + fh / 2.0
    if fw / width < 0.22 or fh / height < 0.30:
        return False, "Move closer until your face fills the guide"
    if fw / width > 0.68 or fh / height > 0.82:
        return False, "Move back slightly"
    if not (0.28 * width <= cx <= 0.72 * width and 0.25 * height <= cy <= 0.75 * height):
        return False, "Center your face inside the guide"

    right_eye = np.asarray(face[4:6], dtype=np.float32)
    left_eye = np.asarray(face[6:8], dtype=np.float32)
    nose = np.asarray(face[8:10], dtype=np.float32)
    eye_mid = (right_eye + left_eye) / 2.0
    eye_span = max(float(np.linalg.norm(left_eye - right_eye)), 1.0)
    yaw = float((nose[0] - eye_mid[0]) / eye_span)

    # The preview is mirrored, so screen-left corresponds to positive raw-image yaw.
    if required_pose == "left" and yaw < 0.08:
        return False, "Turn left  ←"
    if required_pose == "right" and yaw > -0.08:
        return False, "Turn right  →"
    # Five-point landmarks do not provide a stable pitch estimate across faces, so
    # there is deliberately no up/down test here. Keep the up/down prompt for useful
    # variety, but gate those samples on good framing instead of trapping the user
    # behind an unreliable chin-angle check.
    if required_pose == "center" and abs(yaw) > 0.15:
        return False, "Look straight at the camera"
    return True, "Hold that pose"


def face_embedding(recognizer, frame, face):
    aligned = recognizer.alignCrop(frame, face)
    feature = recognizer.feature(aligned)
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    if norm == 0:
        return None, aligned
    return feature / norm, aligned


# Origins allowed to drive this board from a page hosted elsewhere.
#
# This used to be `origin.endswith(".vercel.app")`, which matches EVERY Vercel
# deployment on the internet, not just ours -- and it is paired with
# Access-Control-Allow-Private-Network, so any page any user happened to visit on
# any *.vercel.app subdomain could reach the board on localhost, read the camera,
# and enroll or delete faces. Neither service has authentication.
#
# Scope it to this project instead. Vercel preview deployments are
# "<project>-<hash>-<team>.vercel.app", so the project prefix admits our own
# previews while excluding everyone else's.
VERCEL_PROJECT = os.environ.get("VOICE_VERCEL_PROJECT", "ece180")


def is_allowed_ui_origin(origin: str | None, extra=()) -> bool:
    """True when `origin` may call this board's API cross-origin."""
    if not origin:
        return False
    if origin in extra:
        return True
    host = origin.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    if not host.endswith(".vercel.app"):
        return False
    label = host[: -len(".vercel.app")]
    project = VERCEL_PROJECT.lower()
    # exact production host, or one of our own preview hosts
    return bool(project) and (label == project or label.startswith(project + "-"))


def file_mtime_ns(path: Path | None) -> int:
    """Modification time in ns, or 0 when the file is absent."""
    try:
        return path.stat().st_mtime_ns if path else 0
    except OSError:
        return 0


def load_db(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: np.asarray(vec, dtype=np.float32) for name, vec in raw.items()}


def _atomic_write_json(path: Path, raw) -> None:
    """Write via tmp+fsync+rename.

    Opening the real file with "w" truncates it before a byte is written, so a crash
    or power cut mid-dump (an embedded board will eventually see one) leaves a
    half-file that destroys every enrollment and makes the next boot die in load_db.
    rename() is atomic, so the old file survives intact until the new one is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def save_db(path: Path, db) -> None:
    _atomic_write_json(path, {name: np.asarray(vec, dtype=np.float32).tolist()
                              for name, vec in db.items()})


def best_match(db, embedding, threshold):
    best_name = "unknown"
    best_score = -1.0
    for name, enrolled in db.items():
        score = float(np.dot(enrolled, embedding))
        if score > best_score:
            best_name = name
            best_score = score
    if best_score < threshold:
        return "unknown", best_score
    return best_name, best_score


def load_emotion_db(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for person, exprs in raw.items():
        out[person] = {expr: np.asarray(vec, dtype=np.float32) for expr, vec in exprs.items()}
    return out


def save_emotion_db(path: Path, db) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, {
        person: {expr: np.asarray(vec, dtype=np.float32).tolist() for expr, vec in exprs.items()}
        for person, exprs in db.items()
    })


class EnrollmentStore:
    """Shared access to the identity and expression databases on disk.

    TWO processes own these files at once on the board: the voice agent
    (VisionService) and the enrollment server (FaceEngine). Both persist the WHOLE
    dictionary in a single write, so whichever one is holding a stale in-memory
    copy erases the other's people -- enroll someone by voice, then save or delete
    anyone in the manage UI, and the voice-enrolled person is gone.

    The rule that prevents it is: re-read before every read AND immediately before
    every write, which shrinks the lost-update window from "the entire uptime of
    the process" to microseconds. It lives here, once, because both sides had
    their own copy of it and a subtle correctness invariant maintained in two
    places is one that will eventually only be true in one of them.

    Users must define db_path, emotion_db_path, db, emotion_db, and call
    _stamp_dbs() once during __init__. All methods assume the caller holds the
    instance lock.
    """

    def _stamp_dbs(self):
        self._db_mtime = file_mtime_ns(self.db_path)
        self._emotion_db_mtime = file_mtime_ns(self.emotion_db_path)

    def _refresh_dbs(self):
        """Reload either database if another process rewrote it."""
        db_mtime = file_mtime_ns(self.db_path)
        if db_mtime != self._db_mtime:
            self.db = load_db(self.db_path)
            self._db_mtime = db_mtime
        if self.emotion_db_path:
            emotion_mtime = file_mtime_ns(self.emotion_db_path)
            if emotion_mtime != self._emotion_db_mtime:
                self.emotion_db = load_emotion_db(self.emotion_db_path)
                self._emotion_db_mtime = emotion_mtime

    def _save_dbs(self, identities=False, expressions=False):
        """Persist and re-stamp, so our own write is never read back as someone else's."""
        if identities:
            save_db(self.db_path, self.db)
            self._db_mtime = file_mtime_ns(self.db_path)
        if expressions and self.emotion_db_path:
            save_emotion_db(self.emotion_db_path, self.emotion_db)
            self._emotion_db_mtime = file_mtime_ns(self.emotion_db_path)


def classify_personal(protos, feat, temperature=0.2):
    exprs = list(protos.keys())
    if not exprs:
        return None, 0.0
    mats = np.stack([protos[e] for e in exprs])
    dist = np.linalg.norm(mats - feat.reshape(1, -1), axis=1)
    idx = int(np.argmin(dist))
    logits = -dist / max(temperature, 1e-6)
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    conf = float(weights[idx] / float(np.sum(weights)))
    return exprs[idx], conf


def draw_face(frame, face, label):
    x, y, w, h = [int(v) for v in face[:4]]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)


def sentiment_from_emotion(emotion_label: str) -> str:
    label = normalize_emotion_label(emotion_label)
    if label in {"", "none", "n/a", "na", "not_enabled"}:
        return "not_enabled"
    if label in POSITIVE_LABELS:
        return "positive"
    if label in NEGATIVE_LABELS:
        return "negative"
    return "neutral"


def normalize_emotion_label(emotion_label: str | None) -> str:
    label = (emotion_label or "").strip().lower()
    return EMOTION_ALIASES.get(label, label)


def decode_data_url(data_url: str):
    header, payload = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("expected base64 data URL")
    raw = base64.b64decode(payload)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode image frame")
    return frame


class EmotionModel:
    def __init__(self, model_path: Path, size: int, labels):
        self.net = cv2.dnn.readNet(str(model_path))
        self.size = size
        self.labels = [normalize_emotion_label(label) for label in labels]
        self.aligner = FaceAlignment(size)

    def probabilities(self, frame_bgr, face):
        landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
        aligned = self.aligner.align(frame_bgr, landmarks)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - 0.5) / 0.5
        blob = cv2.dnn.blobFromImage(normalized)
        self.net.setInput(blob, "data")
        outputs = self.net.forward(["label"])
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if logits.size == 0:
            return None
        shifted = logits - float(np.max(logits))
        probs = np.exp(shifted)
        return probs / float(np.sum(probs))

    def predict(self, frame_bgr, face):
        probs = self.probabilities(frame_bgr, face)
        if probs is None:
            return "neutral", 0.0
        idx = int(np.argmax(probs))
        score = float(probs[idx])
        label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
        return label, score


class FaceAlignment:
    def __init__(self, size: int):
        scale = size / 112.0
        self.std_points = np.asarray(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        ) * scale
        self.size = size

    def align(self, image, landmarks):
        transform, _ = cv2.estimateAffinePartial2D(
            landmarks,
            self.std_points,
            method=cv2.LMEDS,
        )
        if transform is None:
            x, y, w, h = [int(v) for v in landmarks_bbox(landmarks)]
            return cv2.resize(image[y : y + h, x : x + w], (self.size, self.size))
        return cv2.warpAffine(image, transform, (self.size, self.size))


def landmarks_bbox(landmarks):
    x0 = max(0, int(np.min(landmarks[:, 0])))
    y0 = max(0, int(np.min(landmarks[:, 1])))
    x1 = max(x0 + 1, int(np.max(landmarks[:, 0])))
    y1 = max(y0 + 1, int(np.max(landmarks[:, 1])))
    return x0, y0, x1 - x0, y1 - y0


class FaceEngine(EnrollmentStore):
    def __init__(self, detector_path: Path, recognizer_path: Path, db_path: Path, width: int, height: int, threshold: float, emotion_model: str | None = None, emotion_size: int = 224, emotion_every: int = 8, emotion_labels=None, emotion_db_path: Path | None = None):
        self.detector_path = detector_path
        self.recognizer_path = recognizer_path
        self.db_path = db_path
        self.width = width
        self.height = height
        self.threshold = threshold
        self.detector = create_detector(detector_path, width, height)
        self.recognizer = create_recognizer(recognizer_path)
        self.db = load_db(db_path)
        self.lock = threading.Lock()
        self.enroll_name = None
        self.enroll_embeddings = []
        self.enroll_delay = 0.15
        self.last_enroll_capture = 0.0
        self.last_emotion = (None, 0.0)
        self.last_emotion_name = None      # whose expression last_emotion describes
        self.emotion_every = emotion_every
        self.emotion_counter = 0
        self.emotion = None
        if emotion_model:
            self.emotion = EmotionModel(Path(emotion_model), emotion_size, emotion_labels or DEFAULT_LABELS)
        self.emotion_db_path = emotion_db_path
        self.emotion_db = load_emotion_db(emotion_db_path) if emotion_db_path else {}
        self._stamp_dbs()
        self.last_emotion_source = "generic"
        self.emo_enroll_name = None
        self.emo_enroll_expr = None
        self.emo_enroll_feats = []
        self.last_emo_capture = 0.0
        self.emo_enroll_delay = 0.12

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def emotion_status(self):
        with self.lock:
            self._refresh_dbs()
            return {person: sorted(exprs.keys()) for person, exprs in self.emotion_db.items()}

    def list_people(self):
        with self.lock:
            self._refresh_dbs()
            names = sorted(set(self.db) | set(self.emotion_db), key=str.casefold)
            return [
                {
                    "name": name,
                    "face_enrolled": name in self.db,
                    "expressions": sorted(self.emotion_db.get(name, {})),
                }
                for name in names
            ]

    def remove_person(self, name: str):
        with self.lock:
            self._refresh_dbs()      # delete from the CURRENT roster, not a stale copy
            existed = name in self.db or name in self.emotion_db
            if not existed:
                return {"status": "not_found", "name": name}
            self.db.pop(name, None)
            self.emotion_db.pop(name, None)
            self._save_dbs(identities=True, expressions=True)
            if self.enroll_name == name:
                self.enroll_name = None
                self.enroll_embeddings = []
            if self.emo_enroll_name == name:
                self.emo_enroll_name = None
                self.emo_enroll_expr = None
                self.emo_enroll_feats = []
            return {"status": "deleted", "name": name}

    def emotion_enroll_begin(self, name, expression, delay=0.12):
        with self.lock:
            self.emo_enroll_name = name
            self.emo_enroll_expr = expression
            self.emo_enroll_feats = []
            self.last_emo_capture = 0.0
            self.emo_enroll_delay = delay

    def emotion_enroll_frame(self, frame, required_pose="center"):
        with self.lock:
            if not self.emo_enroll_name or self.emotion is None:
                return {"status": "idle"}
            faces = detect_faces(self.detector, frame)
            face = largest_face(faces)
            if face is None:
                return {"status": "no_face", "guidance": "Place your face inside the guide"}
            compliant, guidance = enrollment_guidance(frame, face, required_pose)
            if not compliant:
                return {"status": "adjust", "guidance": guidance}
            now = time.time()
            if now - self.last_emo_capture < self.emo_enroll_delay:
                return {"status": "waiting"}
            probs = self.emotion.probabilities(frame, face)
            if probs is None:
                return {"status": "bad_sample"}
            self.emo_enroll_feats.append(probs)
            self.last_emo_capture = now
            return {"status": "captured", "count": len(self.emo_enroll_feats), "guidance": guidance}

    def emotion_enroll_finish(self):
        with self.lock:
            if not self.emo_enroll_name:
                return {"status": "idle"}
            if not self.emo_enroll_feats:
                self.emo_enroll_name = None
                self.emo_enroll_expr = None
                return {"status": "empty"}
            self._refresh_dbs()      # merge onto the newest roster, do not overwrite it
            mean_feat = np.mean(np.stack(self.emo_enroll_feats), axis=0)
            person = self.emotion_db.setdefault(self.emo_enroll_name, {})
            person[self.emo_enroll_expr] = mean_feat
            self._save_dbs(expressions=True)
            result = {"status": "saved", "name": self.emo_enroll_name, "expression": self.emo_enroll_expr, "samples": len(self.emo_enroll_feats)}
            self.emo_enroll_name = None
            self.emo_enroll_expr = None
            self.emo_enroll_feats = []
            return result

    def enroll_begin(self, name: str, delay: float = 0.15):
        with self.lock:
            self.enroll_name = name
            self.enroll_embeddings = []
            self.last_enroll_capture = 0.0
            self.enroll_delay = delay

    def enroll_frame(self, frame, required_pose="center"):
        with self.lock:
            if not self.enroll_name:
                return {"status": "idle"}
            faces = detect_faces(self.detector, frame)
            face = largest_face(faces)
            if face is None:
                return {"status": "no_face", "guidance": "Place your face inside the guide"}
            compliant, guidance = enrollment_guidance(frame, face, required_pose)
            if not compliant:
                return {"status": "adjust", "guidance": guidance}
            now = time.time()
            if now - self.last_enroll_capture < self.enroll_delay:
                return {"status": "waiting"}
            embedding, _ = face_embedding(self.recognizer, frame, face)
            if embedding is None:
                return {"status": "bad_embedding"}
            self.enroll_embeddings.append(embedding)
            self.last_enroll_capture = now
            return {"status": "captured", "count": len(self.enroll_embeddings), "guidance": guidance}

    def cancel_enrollment(self):
        with self.lock:
            self.enroll_name = None
            self.enroll_embeddings = []
            self.emo_enroll_name = None
            self.emo_enroll_expr = None
            self.emo_enroll_feats = []
            return {"status": "cancelled"}

    def enroll_finish(self):
        with self.lock:
            if not self.enroll_name:
                return {"status": "idle"}
            if not self.enroll_embeddings:
                self.enroll_name = None
                return {"status": "empty"}
            self._refresh_dbs()      # merge onto the newest roster, do not overwrite it
            mean_embedding = np.mean(np.stack(self.enroll_embeddings), axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            self.db[self.enroll_name] = mean_embedding
            self._save_dbs(identities=True)
            result = {"status": "saved", "name": self.enroll_name, "samples": len(self.enroll_embeddings)}
            self.enroll_name = None
            self.enroll_embeddings = []
            return result

    def recognize_frame(self, frame):
        with self.lock:
            self._refresh_dbs()      # a face enrolled by voice must be recognized here too
            faces = detect_faces(self.detector, frame)
            face = largest_face(faces)
            if face is None:
                emotion_label, emotion_score = self.last_emotion
                return {"name": "no_face", "score": 0.0, "emotion": emotion_label, "emotion_score": emotion_score, "sentiment": sentiment_from_emotion(emotion_label), "emotion_source": self.last_emotion_source if self.emotion else "none"}
            embedding, aligned = face_embedding(self.recognizer, frame, face)
            if embedding is None:
                emotion_label, emotion_score = self.last_emotion
                return {"name": "unknown", "score": 0.0, "emotion": emotion_label, "emotion_score": emotion_score, "sentiment": sentiment_from_emotion(emotion_label), "emotion_source": self.last_emotion_source if self.emotion else "none"}
            name, score = best_match(self.db, embedding, self.threshold)
            # Expression is cached between inferences (they only run every
            # emotion_every frames), and there is exactly ONE cache slot because
            # this path only ever looks at the largest face. So when the largest
            # face becomes a DIFFERENT person, the cached value is the previous
            # person's expression -- and it gets reported as the new person's until
            # the next inference lands. Reading someone else's mood off your face is
            # not a rounding error. VisionService already resets on identity change
            # (see _set_track_identity); this path had never been given the same fix.
            if name != self.last_emotion_name:
                self.last_emotion_name = name
                self.last_emotion = (None, 0.0)
                self.last_emotion_source = "generic"
            if self.emotion and self.emotion_every > 0 and self.emotion_counter % self.emotion_every == 0:
                probs = self.emotion.probabilities(frame, face)
                if probs is not None:
                    protos = self.emotion_db.get(name) if name not in ("unknown", "no_face") else None
                    if protos:
                        label, conf = classify_personal(protos, probs)
                        self.last_emotion = (label, conf)
                        self.last_emotion_source = "personal"
                    else:
                        idx = int(np.argmax(probs))
                        label = self.emotion.labels[idx] if idx < len(self.emotion.labels) else f"class_{idx}"
                        self.last_emotion = (label, float(probs[idx]))
                        self.last_emotion_source = "generic"
            self.emotion_counter += 1
            emotion_label, emotion_score = self.last_emotion
            payload = {"name": name, "score": float(score), "emotion": emotion_label, "emotion_score": float(emotion_score)}
            if self.emotion is None:
                payload["sentiment"] = "not_enabled"
                payload["emotion_source"] = "none"
            else:
                payload["sentiment"] = sentiment_from_emotion(emotion_label)
                payload["emotion_source"] = self.last_emotion_source
            return payload


def command_camera_test(args):
    cap = open_camera(args.camera, args.width, args.height)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit("camera opened but no frame was captured")
    print(f"camera ok: index={args.camera} frame={frame.shape[1]}x{frame.shape[0]}")


def command_enroll(args):
    detector_path = Path(args.detector)
    recognizer_path = Path(args.recognizer)
    require_file(detector_path, "face detector model")
    require_file(recognizer_path, "face recognizer model")

    cap = open_camera(args.camera, args.width, args.height)
    detector = create_detector(detector_path, args.width, args.height)
    recognizer = create_recognizer(recognizer_path)

    embeddings = []
    last_capture = 0.0
    print(f"enrolling {args.name}: need {args.samples} samples")
    try:
        while len(embeddings) < args.samples:
            ok, frame = cap.read()
            if not ok:
                continue
            faces = detect_faces(detector, frame)
            face = largest_face(faces)
            if face is not None and time.time() - last_capture >= args.delay:
                embedding, _ = face_embedding(recognizer, frame, face)
                if embedding is not None:
                    embeddings.append(embedding)
                    last_capture = time.time()
                    print(f"sample {len(embeddings)}/{args.samples}")
            if not args.headless:
                shown = frame.copy()
                if face is not None:
                    draw_face(shown, face, f"{args.name} {len(embeddings)}/{args.samples}")
                cv2.imshow("enroll", shown)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if not embeddings:
        raise SystemExit("no enrollment samples captured")

    mean_embedding = np.mean(np.stack(embeddings), axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
    db = load_db(Path(args.db))
    db[args.name] = mean_embedding
    save_db(Path(args.db), db)
    print(f"saved enrollment: name={args.name} samples={len(embeddings)} db={args.db}")


def command_recognize(args):
    detector_path = Path(args.detector)
    recognizer_path = Path(args.recognizer)
    require_file(detector_path, "face detector model")
    require_file(recognizer_path, "face recognizer model")

    db_path = Path(args.db)
    emotion_db_path = Path(args.emotion_db)
    db = load_db(db_path)
    if not db:
        raise SystemExit(f"enrollment database is empty: {args.db}")

    emotion_db = load_emotion_db(emotion_db_path)
    emotion = None
    labels = args.emotion_labels.split()
    emotion_path = Path(args.emotion_model) if args.emotion_model else None
    if emotion_path is None and DEFAULT_EMOTION_MODEL.exists():
        emotion_path = DEFAULT_EMOTION_MODEL
    emotion_size = DEFAULT_EMOTION_SIZE if emotion_path == DEFAULT_EMOTION_MODEL else args.emotion_size
    if emotion_path:
        require_file(emotion_path, "emotion model")
        emotion = EmotionModel(emotion_path, emotion_size, labels)

    cap = open_camera(args.camera, args.width, args.height)
    detector = create_detector(detector_path, args.width, args.height)
    recognizer = create_recognizer(recognizer_path)

    frame_count = 0
    last_emotion = ("none", 0.0)
    last_print = 0.0
    # 0 would mean "never", but it reaches a modulo: guard it here rather than
    # dying with a ZeroDivisionError on the first frame.
    emotion_every = max(1, args.emotion_every)
    db_mtime = db_path.stat().st_mtime_ns if db_path.exists() else 0
    emotion_db_mtime = emotion_db_path.stat().st_mtime_ns if emotion_db_path.exists() else 0
    try:
        while True:
            # Enrollment files arrive via an atomic rename. Pick them up without
            # restarting the product or interrupting the camera pipeline.
            if frame_count % 12 == 0:
                next_mtime = db_path.stat().st_mtime_ns if db_path.exists() else 0
                if next_mtime != db_mtime:
                    db = load_db(db_path)
                    db_mtime = next_mtime
                    print(f"reloaded enrollments: {', '.join(sorted(db))}")
                next_emotion_mtime = emotion_db_path.stat().st_mtime_ns if emotion_db_path.exists() else 0
                if next_emotion_mtime != emotion_db_mtime:
                    emotion_db = load_emotion_db(emotion_db_path)
                    emotion_db_mtime = next_emotion_mtime
            ok, frame = cap.read()
            if not ok:
                continue
            faces = detect_faces(detector, frame)
            face = largest_face(faces)
            label = "no_face"
            if face is not None:
                embedding, aligned = face_embedding(recognizer, frame, face)
                if embedding is not None:
                    name, score = best_match(db, embedding, args.threshold)
                    if emotion is not None and frame_count % emotion_every == 0:
                        probs = emotion.probabilities(frame, face)
                        if probs is not None:
                            protos = emotion_db.get(name) if name != "unknown" else None
                            if protos:
                                elabel, escore = classify_personal(protos, probs)
                            else:
                                ei = int(np.argmax(probs))
                                elabel = emotion.labels[ei] if ei < len(emotion.labels) else f"class_{ei}"
                                escore = float(probs[ei])
                            last_emotion = (elabel, escore)
                    emotion_label, emotion_score = last_emotion
                    sentiment = "not_enabled" if emotion is None else sentiment_from_emotion(emotion_label)
                    label = f"{name} {score:.2f} {emotion_label} {emotion_score:.2f} {sentiment}"
                    if not args.headless:
                        draw_face(frame, face, label)
            now = time.time()
            if args.headless and now - last_print >= args.print_every:
                print(label)
                last_print = now
            if not args.headless:
                cv2.imshow("recognize", frame)
                if cv2.waitKey(1) == 27:
                    break
            frame_count += 1
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face</title>
<style>
:root{
  --bg:#f4f4f4; --fg:#171717; --muted:#737373; --line:#dedede;
  --card:#ffffff; --accent:#141414;
  --pos:#2f9e5f; --neg:#d84b3f; --neu:#9a9a94;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0d0d0e; --fg:#f1f1f1; --muted:#a0a0a0; --line:#2b2b2d;
    --card:#171719; --accent:#f1f1f1;
    --pos:#4bd07f; --neg:#ff6a5c; --neu:#75756f;
  }
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;
  -webkit-font-smoothing:antialiased;
  display:flex;align-items:flex-start;justify-content:center;min-height:100%;padding:48px 32px;
}
.app{
  width:100%;max-width:1120px;display:grid;
  grid-template-columns:minmax(0,1.55fr) minmax(350px,.8fr);
  gap:24px;align-items:start;
}
.stage{
  position:relative;aspect-ratio:4/3;width:100%;background:#000;
  border-radius:12px;overflow:hidden;border:1px solid var(--line);cursor:pointer;
}
.panel{
  background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px;
}
.panel-title{font-size:22px;font-weight:650;letter-spacing:-.01em;margin:0 0 24px}
video{width:100%;height:100%;object-fit:cover;display:block;transform:scaleX(-1);visibility:hidden}
.camera-on video{visibility:visible}
.hint{
  position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  color:rgba(255,255,255,.62);font-size:14px;
  transition:opacity .3s ease;pointer-events:none;
}
.hint.gone{opacity:0}
.guide{
  position:absolute;inset:0;display:none;align-items:center;justify-content:center;
  pointer-events:none;background:rgba(0,0,0,.10);
}
.guide.on{display:flex}
.guide-box{
  width:58%;height:76%;border:2px solid rgba(255,255,255,.82);border-radius:46% 46% 42% 42%;
  box-shadow:0 0 0 999px rgba(0,0,0,.18);transition:border-color .15s,box-shadow .15s;
}
.guide.ok .guide-box{border-color:var(--pos);box-shadow:0 0 0 999px rgba(0,0,0,.12),0 0 24px rgba(47,158,95,.4)}
.guide-prompt{
  position:absolute;left:18px;right:18px;top:16px;text-align:center;color:#fff;
  font-size:17px;font-weight:600;text-shadow:0 1px 5px rgba(0,0,0,.8);
}
.readout{
  position:absolute;left:0;right:0;bottom:0;padding:20px 20px 18px;
  background:linear-gradient(to top,rgba(0,0,0,.66),rgba(0,0,0,0));
  color:#fff;opacity:0;transform:translateY(6px);
  transition:opacity .28s ease,transform .28s ease;pointer-events:none;
}
.readout.on{opacity:1;transform:none}
.name{font-size:27px;font-weight:600;letter-spacing:-.01em;line-height:1.05}
.name.unknown{color:#d0d0cb;font-weight:500}
.meta{margin-top:6px;font-size:14px;color:rgba(255,255,255,.82);
  display:flex;align-items:center;gap:9px;min-height:14px}
.tag{color:rgba(255,255,255,.62);font-size:14px}
.field{margin:0 0 12px}
.field label{display:block;margin-bottom:7px;font-size:14px;font-weight:550}
.field input{
  width:100%;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit;
  font-size:16px;text-align:left;padding:11px 12px;outline:none;transition:border-color .2s;
}
.field input:focus{border-color:var(--accent)}
.field input::placeholder{color:var(--muted)}
.row{display:flex;gap:10px}
button{
  flex:1;font:inherit;font-size:14px;font-weight:550;padding:11px 14px;
  border-radius:8px;cursor:pointer;border:1px solid var(--line);
  background:var(--card);color:var(--fg);
  transition:transform .12s ease,background .2s,border-color .2s,opacity .2s;
}
button:hover{border-color:var(--accent)}
button:active{transform:scale(.985)}
button:disabled{opacity:.4;cursor:default;transform:none}
button.primary{background:var(--accent);color:var(--bg);border-color:var(--accent)}
button.primary.active{background:var(--neg);border-color:var(--neg);color:#fff}
.cancel{margin-top:10px;width:100%;color:var(--neg);background:transparent}
.cancel[hidden]{display:none}
.chips{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-top:20px}
.clabel{width:100%;font-size:14px;font-weight:600;color:var(--fg);margin-bottom:2px}
.chip{
  flex:none;font:inherit;font-size:14px;font-weight:500;padding:7px 10px;border-radius:7px;
  border:1px solid var(--line);background:var(--card);color:var(--muted);cursor:pointer;
  transition:transform .12s ease,border-color .2s,color .2s;
}
.chip:hover{border-color:var(--accent);color:var(--fg)}
.chip:active{transform:scale(.96)}
.chip.trained{color:var(--fg);border-color:var(--accent)}
.chip:disabled{opacity:.4;cursor:default}
.status{
  text-align:left;color:var(--muted);font-size:14px;
  margin-top:16px;min-height:16px;transition:color .2s;
}
.advanced{margin-top:18px}
.advanced summary{font-size:14px;color:var(--muted);cursor:pointer;user-select:none}
.tune{
  display:flex;align-items:center;gap:12px;margin-top:14px;color:var(--muted);font-size:14px;
}
.tune input[type=range]{
  flex:1;-webkit-appearance:none;appearance:none;height:2px;
  background:var(--line);border-radius:2px;outline:none;
}
.tune input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:50%;
  background:var(--accent);cursor:pointer;
}
.tune input[type=range]::-moz-range-thumb{
  width:14px;height:14px;border:0;border-radius:50%;background:var(--accent);cursor:pointer;
}
.tune .val{font-variant-numeric:tabular-nums;min-width:30px;text-align:right}
.people{margin-top:24px}
.people-head{
  color:var(--fg);font-size:15px;font-weight:600;margin-bottom:8px;
}
.person{
  display:flex;align-items:center;gap:10px;min-height:54px;margin-top:4px;
}
.person-select{
  flex:1;border:0;background:transparent;padding:11px 0;text-align:left;border-radius:0;
}
.person-select:hover{border-color:transparent}
.person-name{display:block;font-size:15px;color:var(--fg)}
.person-meta{display:block;margin-top:3px;font-size:14px;color:var(--muted);font-weight:400}
.remove{
  flex:none;border:0;background:transparent;color:var(--neg);padding:8px;
  font-size:14px;border-radius:8px;
}
.remove:hover{border:0;background:color-mix(in srgb,var(--neg) 10%,transparent)}
.people-empty{color:var(--muted);font-size:14px;padding:10px 0}
@media (max-width:800px){
  body{padding:16px;align-items:flex-start}
  .app{grid-template-columns:1fr;gap:18px;max-width:560px}
  .stage{border-radius:10px}
  .panel{padding:20px;border-radius:10px}
}
</style>
</head>
<body>
<div class="app">
  <div class="stage" id="stage">
    <video id="video" autoplay playsinline muted></video>
    <div class="hint" id="hint">Start camera</div>
    <div class="guide" id="guide">
      <div class="guide-box"></div>
      <div class="guide-prompt" id="guidePrompt">Look straight at the camera</div>
    </div>
    <div class="readout" id="readout">
      <div class="name" id="rname">unknown</div>
      <div class="meta"><span id="remotion"></span><span class="tag" id="rtag"></span></div>
    </div>
  </div>
  <div class="panel">
  <h1 class="panel-title">Face enrollment</h1>
  <div class="field">
    <label for="name">Name</label>
    <input id="name" placeholder="Enter a name" value="zakaria" autocomplete="off" spellcheck="false">
  </div>
  <div class="row">
    <button id="enrollBtn" class="primary">Enroll</button>
    <button id="recBtn">Recognize</button>
  </div>
  <button id="cancelBtn" class="cancel" hidden>Cancel training</button>
  <div class="chips" id="chips">
    <span class="clabel">Expressions</span>
    <button class="chip" data-expr="neutral">neutral</button>
    <button class="chip" data-expr="happy">happy</button>
    <button class="chip" data-expr="sad">sad</button>
    <button class="chip" data-expr="surprise">surprise</button>
  </div>
  <div class="status" id="status">Ready</div>
  <details class="advanced">
    <summary>Recognition settings</summary>
    <div class="tune">
      <span>Match threshold</span>
      <input id="threshold" type="range" min="0.2" max="0.9" step="0.01" value="0.5">
      <span class="val" id="thval">0.50</span>
    </div>
  </details>
  <section class="people" aria-labelledby="peopleTitle">
    <div class="people-head" id="peopleTitle">Saved people</div>
    <div id="peopleList"><div class="people-empty">loading…</div></div>
  </section>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
const video=$('video'),status=$('status'),readout=$('readout'),hint=$('hint'),
  rname=$('rname'),remotion=$('remotion'),rtag=$('rtag'),stage=$('stage'),
  enrollBtn=$('enrollBtn'),recBtn=$('recBtn'),nameInput=$('name'),
  threshold=$('threshold'),thval=$('thval'),peopleList=$('peopleList'),
  guide=$('guide'),guidePrompt=$('guidePrompt'),cancelBtn=$('cancelBtn');
const chips=Array.from(document.querySelectorAll('.chip'));
let stream=null,timer=null,mode='idle';
const canvas=document.createElement('canvas');
const poses=['center','left','right','up','down'];
const samplesPerPose=12,totalSamples=poses.length*samplesPerPose;

function say(t){status.textContent=t}
function stopLoop(){if(timer){clearInterval(timer);timer=null}mode='idle'}
function lock(on){[enrollBtn,recBtn,...chips].forEach(b=>b.disabled=on)}
function poseAt(count){return poses[Math.min(poses.length-1,Math.floor(count/samplesPerPose))]}
function posePrompt(pose,expression=''){
  const action={center:'Look straight at the camera',left:'Turn left  ←',right:'Turn right  →',up:'Look up  ↑',down:'Look down  ↓'}[pose];
  return expression?(action+', keep a '+expression+' expression'):action;
}
function showGuide(on,text='',ok=false){
  guide.classList.toggle('on',on);guide.classList.toggle('ok',ok);
  if(text)guidePrompt.textContent=text;
}
function setTraining(on){cancelBtn.hidden=!on;if(!on)showGuide(false)}
async function camera(){
  if(stream)return;
  stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
  video.srcObject=stream;await video.play();stage.classList.add('camera-on');hint.classList.add('gone');
}
function grab(){
  canvas.width=video.videoWidth||640;canvas.height=video.videoHeight||480;
  canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
  return canvas.toDataURL('image/jpeg',0.85);
}
// '' when this server hosts the page; absolute when a hosted copy (Vercel) drives
// this board through the laptop's forwarded port.
const API=__API_BASE__;
async function post(url,body){
  const r=await fetch(API+url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  return r.json();
}
async function refreshChips(){
  const name=nameInput.value.trim();
  let trained=[];
  try{const m=await (await fetch(API+'/api/emotion/list')).json();trained=m[name]||[]}catch(e){}
  chips.forEach(c=>c.classList.toggle('trained',trained.includes(c.dataset.expr)));
}
async function refreshPeople(){
  let people=[];
  try{people=await (await fetch(API+'/api/enroll/list')).json()}catch(e){}
  peopleList.replaceChildren();
  if(!people.length){
    const empty=document.createElement('div');empty.className='people-empty';
    empty.textContent='No saved people';peopleList.appendChild(empty);return;
  }
  people.forEach(person=>{
    const row=document.createElement('div');row.className='person';
    const select=document.createElement('button');select.className='person-select';
    const label=document.createElement('span');label.className='person-name';label.textContent=person.name;
    const meta=document.createElement('span');meta.className='person-meta';
    meta.textContent=person.expressions.length?person.expressions.join(', '):'Face only';
    select.append(label,meta);
    select.onclick=()=>{nameInput.value=person.name;refreshChips();say('selected '+person.name)};
    const remove=document.createElement('button');remove.className='remove';remove.textContent='Remove';
    remove.setAttribute('aria-label','Remove '+person.name);
    remove.onclick=async()=>{
      if(!confirm('Remove '+person.name+' from the system?'))return;
      const r=await post('/api/enroll/delete',{name:person.name});
      if(r.status==='deleted'){
        if(nameInput.value.trim()===person.name)nameInput.value='';
        say('removed '+person.name);await Promise.all([refreshPeople(),refreshChips()]);
      }
    };
    row.append(select,remove);peopleList.appendChild(row);
  });
}
function show(o){
  if(!o||o.name==='no_face'){readout.classList.remove('on');return}
  readout.classList.add('on');
  const unknown=o.name==='unknown';
  rname.textContent=o.name;
  rname.classList.toggle('unknown',unknown);
  const parts=[];
  if(o.emotion&&o.sentiment!=='not_enabled')parts.push(o.emotion);
  if(!unknown&&o.score)parts.push(Math.round(o.score*100)+'%');
  remotion.textContent=parts.join(', ');
  rtag.textContent=o.emotion_source==='personal'?'personal':'';
}
function setRec(on){recBtn.textContent=on?'Stop':'Recognize';recBtn.classList.toggle('active',on)}

stage.onclick=async()=>{try{await camera();say('camera on')}catch(e){say('camera blocked')}};
threshold.oninput=()=>{thval.textContent=(+threshold.value).toFixed(2);post('/api/config',{threshold:+threshold.value})};
nameInput.addEventListener('input',refreshChips);

recBtn.onclick=async()=>{
  if(mode==='recognize'){stopLoop();setRec(false);say('stopped');return}
  try{await camera()}catch(e){say('camera blocked');return}
  stopLoop();mode='recognize';setRec(true);
  await post('/api/config',{threshold:+threshold.value});
  say('recognizing');
  timer=setInterval(async()=>{
    if(mode!=='recognize')return;
    try{show(await post('/api/recognize',{frame:grab()}))}catch(e){}
  },250);
};

enrollBtn.onclick=async()=>{
  const name=nameInput.value.trim();
  if(!name){nameInput.focus();return}
  try{await camera()}catch(e){say('camera blocked');return}
  stopLoop();setRec(false);mode='enroll';readout.classList.remove('on');
  const samples=totalSamples;let count=0;
  await post('/api/enroll/begin',{name,samples});
  lock(true);setTraining(true);showGuide(true,posePrompt(poseAt(0)));
  const done=async()=>{
    stopLoop();
    const r=await post('/api/enroll/finish',{});
    say(r.status==='saved'?('saved '+name):'nothing captured, try again');
    lock(false);setTraining(false);refreshPeople();
  };
  timer=setInterval(async()=>{
    if(mode!=='enroll')return;
    const pose=poseAt(count);
    const o=await post('/api/enroll/frame',{frame:grab(),pose});
    if(o.status==='captured'){
      count=o.count;showGuide(true,posePrompt(poseAt(count)),true);
      say('Captured '+count+' / '+samples+', pose '+Math.min(poses.length,Math.floor(count/samplesPerPose)+1)+' / '+poses.length);
    }else if(o.status==='adjust'||o.status==='no_face'){
      showGuide(true,o.guidance||posePrompt(pose),false);say('Paused, '+(o.guidance||'adjust your pose'));
    }
    if(count>=samples)done();
  },180);
};

chips.forEach(chip=>chip.onclick=async()=>{
  const name=nameInput.value.trim();
  if(!name){nameInput.focus();return}
  const expr=chip.dataset.expr;
  try{await camera()}catch(e){say('camera blocked');return}
  stopLoop();setRec(false);mode='train';readout.classList.remove('on');
  const samples=totalSamples;let count=0;
  await post('/api/emotion/enroll/begin',{name,expression:expr});
  lock(true);setTraining(true);showGuide(true,posePrompt(poseAt(0),expr));
  const done=async()=>{
    stopLoop();
    const r=await post('/api/emotion/enroll/finish',{});
    say(r.status==='saved'?('learned '+expr+' for '+name):'nothing captured, try again');
    lock(false);setTraining(false);refreshChips();refreshPeople();
  };
  timer=setInterval(async()=>{
    if(mode!=='train')return;
    const pose=poseAt(count);
    const o=await post('/api/emotion/enroll/frame',{frame:grab(),pose});
    if(o.status==='captured'){
      count=o.count;showGuide(true,posePrompt(poseAt(count),expr),true);
      say('Captured '+count+' / '+samples+', keep the '+expr+' expression');
    }else if(o.status==='adjust'||o.status==='no_face'){
      showGuide(true,o.guidance||posePrompt(pose,expr),false);say('Paused, '+(o.guidance||'adjust your pose'));
    }
    if(count>=samples)done();
  },160);
});

cancelBtn.onclick=async()=>{
  stopLoop();await post('/api/enroll/cancel',{});lock(false);setTraining(false);
  say('training cancelled');
};

refreshChips();refreshPeople();
</script>
</body>
</html>
"""


def command_web(args):
    try:
        from flask import Flask, jsonify, request
    except Exception as exc:
        raise SystemExit("Flask is required for web mode. Install requirements.txt.") from exc

    detector_path = Path(args.detector)
    recognizer_path = Path(args.recognizer)
    require_file(detector_path, "face detector model")
    require_file(recognizer_path, "face recognizer model")

    emotion_model = args.emotion_model
    if not emotion_model and DEFAULT_EMOTION_MODEL.exists():
        emotion_model = str(DEFAULT_EMOTION_MODEL)

    engine = FaceEngine(
        detector_path=detector_path,
        recognizer_path=recognizer_path,
        db_path=Path(args.db),
        width=args.width,
        height=args.height,
        threshold=args.threshold,
        emotion_model=emotion_model,
        emotion_size=args.emotion_size,
        emotion_every=args.emotion_every,
        emotion_labels=args.emotion_labels.split(),
        emotion_db_path=Path(args.emotion_db),
    )

    app = Flask(__name__)
    # A data-URL frame from a 640x480 camera is well under a megabyte. Cap the body
    # so a malformed or hostile POST cannot be read entirely into the board's RAM.
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("ENROLL_MAX_UPLOAD_BYTES",
                                                          str(8 * 1024 * 1024)))
    _extra_origins = tuple(o for o in os.environ.get("VOICE_ALLOWED_ORIGINS", "").split(",") if o)

    page = HTML_PAGE.replace("__API_BASE__", '""')   # same-origin when served here

    # A hosted copy of this page (Vercel) drives this board over the laptop's
    # forwarded port. That is cross-origin AND public->private, so it needs CORS
    # plus the Private Network Access opt-in, including on the preflight.
    @app.after_request
    def allow_hosted_ui(response):
        origin = request.headers.get("Origin")
        if is_allowed_ui_origin(origin, extra=_extra_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    @app.route("/api/<path:_rest>", methods=["OPTIONS"])
    def preflight(_rest):
        return ("", 204)

    @app.get("/")
    def index():
        return page

    class BadRequest(Exception):
        """A client sent something unusable. Answers 400, never a 500 stack trace."""

    def body():
        """The request body as a dict, or a clean 400.

        get_json(force=True) raises on a malformed body and every handler then
        indexed the result directly, so a truncated upload or a stray GET turned
        into an HTML 500 with a traceback -- unreadable to the page, and noise in
        the board's log during exactly the moments something is already wrong."""
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise BadRequest("expected a JSON object")
        return data

    def required(data, key):
        value = str(data.get(key, "")).strip()
        if not value:
            raise BadRequest(f"'{key}' is required")
        return value

    def frame_from(data):
        try:
            return decode_data_url(required(data, "frame"))
        except BadRequest:
            raise
        except (ValueError, TypeError, base64.binascii.Error) as exc:
            raise BadRequest(f"could not decode frame: {exc}") from exc

    @app.errorhandler(BadRequest)
    def on_bad_request(exc):
        return jsonify({"status": "invalid", "error": str(exc)}), 400

    @app.errorhandler(413)
    def on_too_large(_exc):
        return jsonify({"status": "invalid", "error": "request body too large"}), 413

    @app.post("/api/config")
    def api_config():
        data = body()
        if "threshold" in data:
            try:
                engine.set_threshold(float(data["threshold"]))
            except (TypeError, ValueError) as exc:
                raise BadRequest("threshold must be a number") from exc
        # Loading an arbitrary path as an ONNX model is a filesystem read chosen by
        # the caller. Only the model this build ships with is accepted.
        requested = str(data.get("emotion_model") or "").strip()
        if requested:
            if Path(requested).resolve() != DEFAULT_EMOTION_MODEL.resolve():
                raise BadRequest("emotion_model must be the bundled model")
            engine.emotion = EmotionModel(DEFAULT_EMOTION_MODEL, DEFAULT_EMOTION_SIZE,
                                          args.emotion_labels.split())
        return jsonify({"ok": True, "threshold": engine.threshold})

    @app.post("/api/enroll/begin")
    def api_enroll_begin():
        data = body()
        name = required(data, "name")
        try:
            delay = float(data.get("delay", 0.15))
        except (TypeError, ValueError) as exc:
            raise BadRequest("delay must be a number") from exc
        engine.enroll_begin(name, delay)
        return jsonify({"ok": True, "name": name})

    @app.post("/api/enroll/frame")
    def api_enroll_frame():
        data = body()
        return jsonify(engine.enroll_frame(frame_from(data), str(data.get("pose", "center"))))

    @app.post("/api/enroll/finish")
    def api_enroll_finish():
        return jsonify(engine.enroll_finish())

    @app.post("/api/enroll/cancel")
    def api_enroll_cancel():
        return jsonify(engine.cancel_enrollment())

    @app.get("/api/enroll/list")
    def api_enroll_list():
        return jsonify(engine.list_people())

    @app.post("/api/enroll/delete")
    def api_enroll_delete():
        return jsonify(engine.remove_person(required(body(), "name")))

    @app.post("/api/recognize")
    def api_recognize():
        return jsonify(engine.recognize_frame(frame_from(body())))

    @app.get("/api/emotion/list")
    def api_emotion_list():
        return jsonify(engine.emotion_status())

    @app.post("/api/emotion/enroll/begin")
    def api_emotion_begin():
        data = body()
        engine.emotion_enroll_begin(required(data, "name"), required(data, "expression"))
        return jsonify({"ok": True})

    @app.post("/api/emotion/enroll/frame")
    def api_emotion_frame():
        data = body()
        return jsonify(engine.emotion_enroll_frame(frame_from(data),
                                                   str(data.get("pose", "center"))))

    @app.post("/api/emotion/enroll/finish")
    def api_emotion_finish():
        return jsonify(engine.emotion_enroll_finish())

    @app.post("/api/stop")
    def api_stop():
        return jsonify({"ok": True})

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


def command_emotion_enroll(args):
    detector_path = Path(args.detector)
    require_file(detector_path, "face detector model")
    emotion_path = Path(args.emotion_model) if args.emotion_model else DEFAULT_EMOTION_MODEL
    require_file(emotion_path, "emotion model")
    emotion_size = DEFAULT_EMOTION_SIZE if emotion_path == DEFAULT_EMOTION_MODEL else args.emotion_size
    detector = create_detector(detector_path, args.width, args.height)
    emotion = EmotionModel(emotion_path, emotion_size, args.emotion_labels.split())

    cap = open_camera(args.camera, args.width, args.height)
    feats = []
    last_capture = 0.0
    print(f"training '{args.expression}' for {args.name}: need {args.samples} samples")
    try:
        while len(feats) < args.samples:
            ok, frame = cap.read()
            if not ok:
                continue
            faces = detect_faces(detector, frame)
            face = largest_face(faces)
            if face is not None and time.time() - last_capture >= args.delay:
                probs = emotion.probabilities(frame, face)
                if probs is not None:
                    feats.append(probs)
                    last_capture = time.time()
                    print(f"sample {len(feats)}/{args.samples}")
            if not args.headless:
                shown = frame.copy()
                if face is not None:
                    draw_face(shown, face, f"{args.name}:{args.expression} {len(feats)}/{args.samples}")
                cv2.imshow("emotion-enroll", shown)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if not feats:
        raise SystemExit("no emotion samples captured")

    mean_feat = np.mean(np.stack(feats), axis=0)
    db = load_emotion_db(Path(args.emotion_db))
    db.setdefault(args.name, {})[args.expression] = mean_feat
    save_emotion_db(Path(args.emotion_db), db)
    print(f"saved emotion: {args.name}/{args.expression} samples={len(feats)} db={args.emotion_db}")


def build_parser():
    parser = argparse.ArgumentParser(description="Local face enrollment and recognition for UNO Q")
    parser.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    parser.add_argument("--recognizer", default=str(DEFAULT_RECOGNIZER))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)

    sub = parser.add_subparsers(dest="command", required=True)

    camera_test = sub.add_parser("camera-test")
    camera_test.set_defaults(func=command_camera_test)

    enroll = sub.add_parser("enroll")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--samples", type=int, default=30)
    enroll.add_argument("--delay", type=float, default=0.15)
    enroll.add_argument("--headless", action="store_true")
    enroll.set_defaults(func=command_enroll)

    recognize = sub.add_parser("recognize")
    recognize.add_argument("--threshold", type=float, default=0.50)
    recognize.add_argument("--headless", action="store_true")
    recognize.add_argument("--print-every", type=float, default=0.5)
    recognize.add_argument("--emotion-model")
    recognize.add_argument("--emotion-size", type=int, default=DEFAULT_EMOTION_SIZE)
    recognize.add_argument("--emotion-every", type=int, default=8)
    recognize.add_argument("--emotion-labels", default=" ".join(DEFAULT_LABELS))
    recognize.add_argument("--emotion-db", default=str(DEFAULT_EMOTION_DB))
    recognize.set_defaults(func=command_recognize)

    emotion = sub.add_parser("emotion-enroll")
    emotion.add_argument("--name", required=True)
    emotion.add_argument("--expression", required=True)
    emotion.add_argument("--samples", type=int, default=25)
    emotion.add_argument("--delay", type=float, default=0.12)
    emotion.add_argument("--headless", action="store_true")
    emotion.add_argument("--emotion-model")
    emotion.add_argument("--emotion-size", type=int, default=DEFAULT_EMOTION_SIZE)
    emotion.add_argument("--emotion-labels", default=" ".join(DEFAULT_LABELS))
    emotion.add_argument("--emotion-db", default=str(DEFAULT_EMOTION_DB))
    emotion.set_defaults(func=command_emotion_enroll)

    web = sub.add_parser("web")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.add_argument("--threshold", type=float, default=0.50)
    web.add_argument("--emotion-model")
    web.add_argument("--emotion-size", type=int, default=DEFAULT_EMOTION_SIZE)
    web.add_argument("--emotion-every", type=int, default=8)
    web.add_argument("--emotion-labels", default=" ".join(DEFAULT_LABELS))
    web.add_argument("--emotion-db", default=str(DEFAULT_EMOTION_DB))
    web.set_defaults(func=command_web)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
