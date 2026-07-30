"""Fault injection for complete standalone turns and board playback."""
import contextlib
import os
import threading
import wave
from unittest import mock

import numpy as np
import pytest

from voice_agent.board_audio import BoardAudioLoop, _amplify

from test_board_playback import FakeAgent as PlaybackAgent
from test_board_playback import FakeProc, wav_bytes
from test_board_vad import pcm_chunk


class ProbeLock:
    def __init__(self, acquired=True):
        self.acquired = acquired
        self.acquire_calls = []
        self.releases = 0

    def acquire(self, timeout=None):
        self.acquire_calls.append(timeout)
        return self.acquired

    def release(self):
        self.releases += 1


class ProbeVision:
    def __init__(self):
        self.entered = 0
        self.exited = 0

    @contextlib.contextmanager
    def turn_in_progress(self):
        self.entered += 1
        try:
            yield
        finally:
            self.exited += 1


class TurnAgent:
    def __init__(self, result=None, error=None, acquired=True):
        self.turn_lock = ProbeLock(acquired)
        self.vision = ProbeVision()
        self.result = result or {
            "transcript": "hello",
            "reply": "Hello back.",
            "timings_ms": {"stt": 10.0, "llm": 20.0},
        }
        self.error = error
        self.paths = []

    def understand_audio(self, path):
        self.paths.append(path)
        if self.error:
            raise self.error
        return self.result


def input_pcm():
    return pcm_chunk(0.014) * 8 + pcm_chunk(0.040) * 12


def test_busy_board_turn_is_bounded_and_does_not_enter_inference():
    agent = TurnAgent(acquired=False)
    loop = BoardAudioLoop(agent)
    loop._speak = mock.Mock()
    loop._speak_reply = mock.Mock()

    loop._answer(input_pcm())

    assert len(agent.turn_lock.acquire_calls) == 1
    assert agent.turn_lock.releases == 0
    assert agent.vision.entered == 0
    assert not agent.paths
    loop._speak.assert_called_once()


def test_inference_failure_releases_lock_exits_vision_and_removes_wav():
    agent = TurnAgent(error=RuntimeError("STT crashed"))
    loop = BoardAudioLoop(agent)

    with pytest.raises(RuntimeError, match="STT crashed"):
        loop._answer(input_pcm())

    assert agent.turn_lock.releases == 1
    assert agent.vision.entered == agent.vision.exited == 1
    assert agent.paths and not os.path.exists(agent.paths[0])


def test_synthesis_failure_releases_lock_and_removes_wav():
    agent = TurnAgent()
    loop = BoardAudioLoop(agent)
    loop._speak_reply = mock.Mock(side_effect=RuntimeError("Piper crashed"))

    with pytest.raises(RuntimeError, match="Piper crashed"):
        loop._answer(input_pcm())

    assert agent.turn_lock.releases == 1
    assert agent.vision.entered == agent.vision.exited == 1
    assert not os.path.exists(agent.paths[0])


def test_no_transcript_returns_to_listening_without_speaking_an_error():
    agent = TurnAgent(result={
        "transcript": "", "reply": "", "timings_ms": {"stt": 5.0}})
    loop = BoardAudioLoop(agent)
    loop._speak_reply = mock.Mock(return_value=(False, None))
    loop._speak = mock.Mock()

    loop._answer(input_pcm())

    loop._speak.assert_not_called()
    assert agent.turn_lock.releases == 1


def test_transcript_with_failed_voice_gets_audible_status():
    agent = TurnAgent()
    loop = BoardAudioLoop(agent)
    loop._speak_reply = mock.Mock(return_value=(False, None))
    loop._speak = mock.Mock()

    loop._answer(input_pcm())

    loop._speak.assert_called_once()


def test_later_tts_failure_still_closes_the_open_playback_stream():
    loop = BoardAudioLoop(PlaybackAgent())
    proc = FakeProc()
    loop._open_playback = lambda fmt: proc
    calls = 0

    def synth(text):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second chunk failed")
        return wav_bytes()

    loop.agent.tts.synth = synth
    with pytest.raises(RuntimeError, match="second chunk failed"):
        loop._speak_reply("First complete sentence. Second complete sentence.")

    proc.stdin.close.assert_called_once()
    assert proc.waited


def test_repeated_replies_do_not_leak_tts_worker_threads():
    loop = BoardAudioLoop(PlaybackAgent())
    loop._open_playback = lambda fmt: FakeProc()

    for _ in range(30):
        assert loop._speak_reply("First complete sentence. Second complete sentence.")[0]

    assert not [t for t in threading.enumerate() if t.name.startswith("uno-tts")]


def test_non_16_bit_audio_is_not_run_through_the_16_bit_amplifier():
    loop = BoardAudioLoop(PlaybackAgent())
    proc = FakeProc()
    loop._open_playback = lambda fmt: proc
    loop.agent.tts.synth = lambda text: wav_bytes(width=1, value=100)

    with mock.patch("voice_agent.board_audio._amplify",
                    side_effect=AssertionError("must not reinterpret U8 as S16")):
        assert loop._speak_reply("A complete reply.")[0]


def test_malformed_status_audio_is_logged_and_never_raises(capsys):
    loop = BoardAudioLoop(PlaybackAgent())
    loop.playback_device = "plughw:1,0"

    loop._play(b"not a wav")

    assert "playback failed" in capsys.readouterr().out


def test_amplifier_preserves_digital_silence_exactly():
    silence = b"\x00\x00" * 1000
    assert _amplify(silence) == silence


def test_amplifier_is_fixed_gain_for_normal_material():
    quiet = np.full(1000, round(0.10 * 32767), dtype="<i2")
    louder = np.full(1000, round(0.20 * 32767), dtype="<i2")
    q = np.frombuffer(_amplify(quiet.tobytes()), dtype="<i2").astype(np.float32)
    loud = np.frombuffer(_amplify(louder.tobytes()), dtype="<i2").astype(np.float32)

    # A fixed gain preserves the 2:1 sentence-level relationship; independent
    # normalization would make both sentences equally loud and cause pumping.
    assert np.mean(np.abs(loud)) / np.mean(np.abs(q)) == pytest.approx(2.0, rel=0.03)


def test_amplifier_limits_hot_peaks_without_integer_clipping():
    phase = np.linspace(0, 20 * np.pi, 16000, endpoint=False)
    hot = (np.sin(phase) * 0.95 * 32767).astype("<i2")
    out = np.frombuffer(_amplify(hot.tobytes()), dtype="<i2").astype(np.int32)

    assert np.max(np.abs(out)) < 32767
    assert np.count_nonzero(np.abs(out) >= 32766) == 0


def test_normalized_capture_is_a_valid_16khz_mono_wav_at_inference(tmp_path):
    agent = TurnAgent()
    loop = BoardAudioLoop(agent)
    loop._speak_reply = mock.Mock(return_value=(False, None))
    inspected = {}

    def inspect(path):
        with wave.open(path, "rb") as wav:
            inspected.update(rate=wav.getframerate(), channels=wav.getnchannels(),
                             width=wav.getsampwidth(), frames=wav.getnframes())
        return {"transcript": "", "reply": "", "timings_ms": {}}

    agent.understand_audio = inspect
    loop._answer(input_pcm())

    assert inspected == {
        "rate": 16000, "channels": 1, "width": 2,
        "frames": len(input_pcm()) // 2,
    }


@pytest.mark.parametrize("seed", range(30))
def test_capture_normalization_fuzz_preserves_shape_and_headroom(seed):
    rng = np.random.default_rng(5000 + seed)
    scale = float(rng.uniform(0.0001, 1.0))
    samples = np.clip(
        rng.normal(0, scale, BoardAudioLoop.CHUNK_SAMPLES * 20),
        -1, 1)
    pcm = (samples * 32767).astype("<i2").tobytes()
    loop = BoardAudioLoop(PlaybackAgent())

    out = np.frombuffer(loop._normalize_capture(pcm), dtype="<i2").astype(np.int32)

    assert out.size == samples.size
    assert np.max(np.abs(out)) <= round(0.951 * 32767)


@pytest.mark.parametrize("seed", range(30))
def test_output_amplifier_fuzz_never_wraps_or_changes_length(seed):
    rng = np.random.default_rng(9000 + seed)
    samples = rng.integers(-32768, 32768, 8000, dtype="<i2")

    encoded = _amplify(samples.tobytes())
    out = np.frombuffer(encoded, dtype="<i2").astype(np.int32)

    assert len(encoded) == samples.nbytes
    assert np.max(np.abs(out)) <= 32767
