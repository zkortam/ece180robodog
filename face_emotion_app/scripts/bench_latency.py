#!/usr/bin/env python3
"""Measure where a voice turn's latency actually goes, per stage, per backend.

Run this on an IDLE machine. Under CPU contention the numbers are meaningless:
the same STT call measured 1.9s at load 2 and 60s at load 90 on the same laptop.

  ./.venv-voice/bin/python scripts/bench_latency.py                 # current config
  ./.venv-voice/bin/python scripts/bench_latency.py --sweep         # compare STT/TTS backends

The turn budget is: VAD dead-air -> STT -> LLM -> TTS -> transfer. Only STT/LLM/TTS
are measured here; VAD dead-air is a constant you set with VOICE_ENDPOINT_MS.
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_agent import config          # noqa: E402
from voice_agent.stt import STT         # noqa: E402
from voice_agent.tts import TTS         # noqa: E402

SAMPLE_TEXT = "You look neutral."
UTTERANCE = "how do I look right now"


def make_utterance(path):
    """A real spoken wav to transcribe. `say` on macOS; else silence (STT will no-op)."""
    if sys.platform == "darwin":
        aiff = path.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(aiff), UTTERANCE], check=True)
        subprocess.run(["afconvert", str(aiff), str(path), "-d", "LEI16", "-f", "WAVE"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return path
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 32000)
    return path


def timed(fn, n=3):
    fn()                                   # warm: never time a cold model load
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def load_check():
    try:
        one, _, _ = os.getloadavg()
        cores = os.cpu_count() or 1
        if one > cores * 0.7:
            print(f"!! load average {one:.1f} on {cores} cores -- the machine is busy.")
            print("!! These numbers will be garbage. Free the CPU and re-run.\n")
    except OSError:
        pass


def bench_one(stt_backend, stt_model, tts_backend, wav, n):
    row = {"stt": f"{stt_backend}/{stt_model}", "tts": tts_backend}
    try:
        s = STT(backend=stt_backend, model=stt_model)
        row["stt_ms"] = round(timed(lambda: s.transcribe(str(wav)), n))
    except Exception as e:
        row["stt_ms"] = f"FAIL ({type(e).__name__})"
    try:
        t = TTS(backend=tts_backend)
        row["tts_ms"] = round(timed(lambda: t.synth(SAMPLE_TEXT), n))
    except Exception as e:
        row["tts_ms"] = f"FAIL ({type(e).__name__})"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="compare STT models and TTS backends")
    ap.add_argument("-n", type=int, default=3, help="repeats per stage (median reported)")
    args = ap.parse_args()

    load_check()
    tmp = Path(tempfile.mkdtemp())
    wav = make_utterance(tmp / "utter.wav")
    dur = 0.0
    with wave.open(str(wav)) as w:
        dur = w.getnframes() / w.getframerate()
    print(f"utterance: {dur:.2f}s of speech ({UTTERANCE!r})")
    print(f"VAD dead-air per turn (VOICE_ENDPOINT_MS): {config.VAD_ENDPOINT_MS} ms\n")

    combos = [(config.STT_BACKEND, config.STT_MODEL, config.TTS_BACKEND)]
    if args.sweep:
        stts = [(config.STT_BACKEND, m) for m in ("tiny.en", "base.en", "small.en")]
        ttss = ["kokoro", "say"] if sys.platform == "darwin" else ["piper", "espeak"]
        combos = [(b, m, t) for b, m in stts for t in ttss]

    print(f"{'STT':28} {'TTS':8} {'STT ms':>10} {'TTS ms':>10} {'+LLM~240':>10} {'TTFA*':>9}")
    print("-" * 80)
    for stt_b, stt_m, tts_b in combos:
        r = bench_one(stt_b, stt_m, tts_b, wav, args.n)
        s, t = r["stt_ms"], r["tts_ms"]
        if isinstance(s, int) and isinstance(t, int):
            total = s + t + 240
            ttfa = total + config.VAD_ENDPOINT_MS
            print(f"{r['stt']:28} {r['tts']:8} {s:10} {t:10} {total:10} {ttfa:8}ms")
        else:
            print(f"{r['stt']:28} {r['tts']:8} {str(s):>10} {str(t):>10} {'-':>10} {'-':>9}")
    print("\n* TTFA = time from you falling silent to first audio out")
    print("  = VAD dead-air + STT + LLM(~240ms, measured) + TTS. Lower is better.")


if __name__ == "__main__":
    main()
