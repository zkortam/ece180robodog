"""Entrypoint: wire vision + agent + web server and run.

  python -m voice_agent.main                 # start camera + voice UI on :8100
  python -m voice_agent.main --no-camera      # UI/agent only (no vision loop)
  python -m voice_agent.main --owner zakaria  # identity-gate sensitive tools to this person
"""
import argparse
import sys
import threading
import time
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

# allow `python voice_agent/main.py` as well as `-m voice_agent.main`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_service import VisionService          # noqa: E402
from voice_agent import config                    # noqa: E402
from voice_agent.orchestrator import VoiceAgent    # noqa: E402
from voice_agent.web import create_app             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="RoboDog voice and vision agent")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--camera", type=int, default=config.VISION_CAMERA)
    ap.add_argument("--fps", type=int, default=config.VISION_FPS)
    ap.add_argument("--no-camera", action="store_true", help="do not open the camera / vision loop")
    ap.add_argument("--browser-camera", action="store_true",
                    help="get frames from the browser instead of a server camera (needed on macOS)")
    ap.add_argument("--board-audio", action="store_true",
                    help="use directly attached USB microphone and speaker; auto-discovers camera")
    ap.add_argument("--owner", default=None, help="enrolled name allowed to trigger sensitive actions")
    ap.add_argument("--stt", default=config.STT_BACKEND)
    ap.add_argument("--tts", default=config.TTS_BACKEND)
    args = ap.parse_args()

    vs = VisionService(camera=args.camera, fps=args.fps,
                       width=config.VISION_WIDTH, height=config.VISION_HEIGHT,
                       emotion_every=config.VISION_EMOTION_EVERY, threshold=config.VISION_THRESHOLD)
    if args.board_audio:
        # A board may boot before a USB camera enumerates.  The watcher starts
        # perception as soon as one becomes usable and retries after a disconnect.
        def camera_watch():
            import cv2
            from glob import glob
            import subprocess
            if hasattr(cv2, "setLogLevel"):
                cv2.setLogLevel(0)
            while True:
                with vs.lock:
                    running = vs.running
                if not running:
                    # Only test nodes Linux actually created.  Probing arbitrary
                    # indexes every two seconds wastes CPU and floods the journal.
                    indexes = []
                    for path in glob("/dev/video*"):
                        suffix = path.rsplit("video", 1)[1]
                        if not suffix.isdigit():
                            continue
                        info = subprocess.run(["v4l2-ctl", "-D", "-d", path],
                                              capture_output=True, text=True,
                                              timeout=2, check=False).stdout
                        # Qualcomm's encoder/decoder nodes are /dev/video* too,
                        # but only a node advertising Video Capture can be a camera.
                        if "Video Capture" in info:
                            indexes.append(int(suffix))
                    indexes.sort()
                    for index in indexes:
                        cap = cv2.VideoCapture(index)
                        ok, _ = cap.read() if cap.isOpened() else (False, None)
                        cap.release()
                        if ok:
                            print(f"[vision] USB camera found at index {index}")
                            vs.start(camera=index)
                            break
                time.sleep(2)
        threading.Thread(target=camera_watch, name="uno-camera-watch", daemon=True).start()
    elif not args.no_camera:
        if args.browser_camera:
            print("[vision] browser-camera mode: frames arrive from the web page")
            vs.start(external=True)
        else:
            print(f"[vision] starting server camera {args.camera} @ {args.fps} fps")
            vs.start()

    from voice_agent.stt import STT
    from voice_agent.tts import TTS
    agent = VoiceAgent(vs, stt=STT(args.stt), tts=TTS(args.tts), owner_name=args.owner)
    if args.board_audio:
        from voice_agent.board_audio import BoardAudioLoop
        BoardAudioLoop(agent).start()

    def _warm_one(label, fn):
        """Warm independent components concurrently after the HTTP server starts."""
        import time as _t
        t0 = _t.perf_counter()
        try:
            fn()
            print(f"[voice] warmed {label} in {_t.perf_counter() - t0:.1f}s")
        except BaseException as e:      # SystemExit from an unknown backend included
            print(f"[voice] WARM FAILED for {label}: {type(e).__name__}: {e} "
                  f"-- the first turn will be slow or fail", file=sys.stderr)

    def _warm_stt():
        p = NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            with wave.open(p, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 8000)   # 0.5s of silence
            agent.stt.transcribe(p)
        finally:
            Path(p).unlink(missing_ok=True)

    def _warm_llm():
        # A real, tiny completion warms DNS/TLS, the HTTP client, and Cerebras'
        # inference path without putting a fake exchange in user history.
        agent._llm_client().run(
            [{"role": "system", "content": "Reply with exactly: ready."},
             {"role": "user", "content": "ready"}],
            [], agent.bus.dispatch, max_rounds=1)

    for label, fn in (("tts", lambda: agent.tts.synth("ready")),
                      ("stt", _warm_stt), ("llm", _warm_llm)):
        threading.Thread(target=_warm_one, args=(label, fn), daemon=True).start()

    app = create_app(agent, vs)
    print(f"[voice] STT={args.stt}  TTS={args.tts}  LLM={config.CEREBRAS_MODEL}")
    print(f"[voice] open http://{args.host}:{args.port}  (click the face once, then just talk)")
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        vs.stop()


if __name__ == "__main__":
    main()
