# Face Emotion App

Lightweight local face enrollment and recognition scaffold for the Arduino UNO Q.

This app uses:

- OpenCV YuNet for face detection.
- OpenCV SFace for face embeddings.
- Local cosine similarity for known-person matching.
- Bundled Hugging Face ONNX emotion classifier from `opencv/facial_expression_recognition`.
- The UI also shows a coarse sentiment label derived from the emotion result.

No face images or embeddings are uploaded by this app.

## Install on UNO Q or Mac

Prefer Debian packages where possible:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy v4l-utils
```

If the system OpenCV does not expose `cv2.FaceDetectorYN_create` and `cv2.FaceRecognizerSF_create`, install a newer OpenCV build or use a local Python environment with `opencv-python`.

Download the default OpenCV and emotion models:

```bash
cd /path/to/ece180/face_emotion_app
./scripts/download_models.sh
```

## Localhost Web UI

Run the browser-based workflow on your Mac webcam:

```bash
cd /path/to/ece180/face_emotion_app
./scripts/run_local.sh
```

Open `http://127.0.0.1:8000`, allow camera access, then enroll Zakaria and the two teammates from the page. Recognition output includes identity, emotion, and coarse sentiment.

Face enrollment and each personal-expression enrollment capture 60 accepted
samples: 12 each looking center, left, right, up, and down. The UI pauses capture
until the face is centered, appropriately sized, and matches the requested pose.

The local trainer automatically copies `data/enrollments.json` and
`data/emotions.json` to `arduino@zk-unoq-01.local:/home/arduino/app/data/`.
Only the compact embeddings/prototypes are transferred; live camera frames and
all product-time inference stay on the UNO Q. Override the destination with
`FACE_BOARD_HOST`, `FACE_BOARD_USER`, or `FACE_BOARD_APP_DIR`, or disable syncing
with `FACE_SYNC_TO_BOARD=0`.

## Check Camera

```bash
v4l2-ctl --list-devices
python3 face_emotion.py camera-test --camera 0
```

## CLI Enrollment

Run once per person. Use lowercase simple names with no spaces.

```bash
python3 face_emotion.py enroll --name zakaria --camera 0 --samples 30
python3 face_emotion.py enroll --name teammate1 --camera 0 --samples 30
python3 face_emotion.py enroll --name teammate2 --camera 0 --samples 30
```

## CLI Recognition

```bash
python3 face_emotion.py recognize --camera 0
```

Headless mode for SSH:

```bash
python3 face_emotion.py recognize --camera 0 --headless
```

## Tune Threshold

Start with:

```bash
python3 face_emotion.py recognize --threshold 0.50
```

Raise the threshold if unknown people match enrolled users. Lower it if enrolled users are rejected.

## Optional Emotion Model

The bundled default is `models/facial_expression_recognition_mobilefacenet_2022july.onnx` from Hugging Face. The int8 version is also downloaded for slower boards. If you want to swap models, place another ONNX file under `models/` and run:

```bash
python3 face_emotion.py recognize \
  --emotion-model models/emotion.onnx \
  --emotion-size 112 \
  --emotion-every 8
```

Emotion labels default to:

```text
angry disgust fear happy neutral sad surprise
```

The exact preprocessing may need adjustment for the chosen model.
