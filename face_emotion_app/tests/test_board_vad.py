"""Regression tests for the standalone conversational turn detector.

These use the levels measured on the real webcam microphone. They exercise the
failure that made the robot ignore speech and answer ambient noise much later.
"""
from unittest import mock

import numpy as np
import pytest

from voice_agent.board_audio import BoardAudioLoop


def pcm_chunk(rms):
    """One 20 ms S16_LE chunk with the requested RMS."""
    value = max(-32767, min(32767, round(rms * 32768)))
    return np.full(BoardAudioLoop.CHUNK_SAMPLES, value, dtype="<i2").tobytes()


class CaptureProc:
    def __init__(self, levels):
        self.chunks = [pcm_chunk(level) for level in levels]
        self.stdout = self
        self.read_count = 0
        self.terminated = False

    def read(self, size):
        self.read_count += 1
        return self.chunks.pop(0) if self.chunks else b""

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


class FakeAgent:
    pass


def run_levels(levels, initial_floor=0.002):
    loop = BoardAudioLoop(FakeAgent())
    loop.capture_device = "plughw:1,0"
    loop.noise_floor = initial_floor
    loop._check_for_dead_microphone = mock.Mock()
    loop._answer = mock.Mock()
    proc = CaptureProc(levels)
    with mock.patch("voice_agent.board_audio.subprocess.Popen", return_value=proc):
        loop._listen_once()
    return loop, proc


def onset_for(loop, floor):
    return max(
        loop.MIN_ONSET,
        floor + max(loop.ONSET_MARGIN, floor * loop.ONSET_RELATIVE))


def test_low_snr_speech_endpoints_and_answers_once():
    # Real measurements: room ~.0143; normal speech crosses .017 and peaks ~.019.
    levels = (
        [0.0143] * 10          # stream settling/calibration
        + [0.0142] * 8        # listening
        + [0.0188] * 22       # 440 ms of speech
        + [0.0142] * 24       # real silence/room
    )
    loop, proc = run_levels(levels)

    loop._answer.assert_called_once()
    assert proc.terminated
    # It closes near the configured endpoint instead of waiting for an 8 s cap.
    assert proc.read_count < 75


def test_immediate_short_yes_inside_rearm_window_is_not_lost():
    # The loop already knows the room from the previous turn. The next utterance
    # starts the instant playback ends and fits almost entirely in calibration.
    levels = [0.0188] * 10 + [0.0142] * 24
    loop, proc = run_levels(levels, initial_floor=0.0143)

    loop._answer.assert_called_once()
    assert proc.read_count < 40


def test_immediate_short_yes_during_first_startup_is_not_lost():
    levels = [0.0188] * 10 + [0.0142] * 24
    loop, _ = run_levels(levels)

    loop._answer.assert_called_once()


def test_boundary_noise_is_discarded_in_under_a_second():
    # Three boundary spikes can confirm onset, but are too short and too weak to
    # become a user turn. The old code remained latched for many seconds.
    levels = [0.0143] * 10 + [0.0142] * 4 + [0.0171] * 3 + [0.0142] * 30
    loop, proc = run_levels(levels)

    loop._answer.assert_not_called()
    assert proc.read_count < 55


def test_an_old_idle_peak_cannot_validate_a_later_weak_trigger():
    levels = (
        [0.0143] * 10
        + [0.050]                    # an old door slam while idle
        + [0.0142] * 30
        + [0.0162] * 14              # enough boundary frames, but no speech peak
        + [0.0142] * 24
    )
    loop, _ = run_levels(levels)

    loop._answer.assert_not_called()


def test_max_utterance_is_wall_time_not_sparse_voiced_time():
    # Once recording starts, a hard wall-clock bound must always return control.
    # The broken implementation counted only above-threshold frames, allowing a
    # noisy latched capture to run arbitrarily long.
    levels = [0.0143] * 10 + [0.0195, 0.0160] * 250
    loop, proc = run_levels(levels)

    loop._answer.assert_called_once()
    expected_max = (
        loop.CALIBRATION_CHUNKS
        + loop.ONSET_WINDOW_CHUNKS
        + int(loop.MAX_UTTERANCE_SECONDS / loop.CHUNK_SECONDS)
        + 3
    )
    assert proc.read_count <= expected_max


def test_abrupt_higher_room_floor_is_relearned_and_never_answered():
    # A fan switching on between turns must not become an eight-second "utterance".
    levels = [0.0200] * 120
    loop, proc = run_levels(levels, initial_floor=0.0143)

    loop._answer.assert_not_called()
    assert loop.noise_floor >= 0.019
    assert proc.read_count < 90


def test_room_level_is_below_release_at_the_measured_noise_floor():
    loop = BoardAudioLoop(FakeAgent())
    floor = 0.0143
    onset = onset_for(loop, floor)
    release = floor + max(loop.RELEASE_MARGIN, floor * loop.RELEASE_RELATIVE)

    assert 0.0143 < release < onset
    # This was .0102 in the failed rewrite, below the room itself.
    assert release > onset * 0.6


def test_every_room_level_keeps_real_clearance_below_onset():
    loop = BoardAudioLoop(FakeAgent())
    floor = 0.024
    onset = onset_for(loop, floor)

    assert onset - floor >= loop.ONSET_MARGIN


def test_a_turn_failure_recovers_without_becoming_a_microphone_fault():
    levels = [0.0143] * 10 + [0.0188] * 22 + [0.0142] * 24
    loop = BoardAudioLoop(FakeAgent())
    loop.capture_device = "plughw:1,0"
    loop._check_for_dead_microphone = mock.Mock()
    loop._answer = mock.Mock(side_effect=RuntimeError("TTS failed"))
    loop._speak = mock.Mock()
    proc = CaptureProc(levels)

    with mock.patch("voice_agent.board_audio.subprocess.Popen", return_value=proc):
        loop._listen_once()  # does not escape into _run's microphone-fault handler

    loop._speak.assert_called_once()


def test_capture_normalization_uses_headroom_and_does_not_clip():
    loop = BoardAudioLoop(FakeAgent())
    quiet = pcm_chunk(0.014) * 10
    speech = pcm_chunk(0.040) * 10
    normalized = np.frombuffer(loop._normalize_capture(quiet + speech), dtype="<i2")

    assert np.max(np.abs(normalized.astype(np.int32))) < 32767
    assert np.sqrt(np.mean(normalized.astype(np.float32) ** 2)) > 0.03 * 32768


def test_short_pause_between_words_does_not_split_the_turn():
    levels = (
        [0.0143] * 10
        + [0.0190, 0.0170] * 9
        + [0.0142] * 12              # 240 ms pause, below 360 ms endpoint
        + [0.0192, 0.0168] * 9
        + [0.0142] * 24
    )
    loop, _ = run_levels(levels)

    loop._answer.assert_called_once()
    captured = loop._answer.call_args.args[0]
    # Both phrases and the pause remain in one captured utterance.
    assert len(captured) // (loop.CHUNK_SAMPLES * 2) > 45


@pytest.mark.parametrize("seed", range(60))
def test_randomized_room_and_speech_profiles_complete_once(seed):
    rng = np.random.default_rng(seed)
    room = float(rng.uniform(0.003, 0.023))
    calibration = np.clip(rng.normal(room, 0.00012, 10), 0.0001, None).tolist()
    idle = np.clip(rng.normal(room, 0.00012, 12), 0.0001, None).tolist()
    high = max(0.016, room + 0.008)
    speech = []
    for i in range(36):
        # Natural envelope: strong syllables with periodic quiet phoneme gaps.
        center = room + 0.0002 if i % 7 == 6 else high
        speech.append(float(max(0.0001, rng.normal(center, 0.00015))))
    quiet = np.clip(rng.normal(room, 0.00010, 28), 0.0001, None).tolist()

    loop, proc = run_levels(calibration + idle + speech + quiet)

    assert loop._answer.call_count == 1, f"seed={seed}, room={room:.4f}"
    assert proc.read_count < 100


@pytest.mark.parametrize("seed", range(40))
def test_random_ambient_with_isolated_impacts_never_answers_early(seed):
    rng = np.random.default_rng(1000 + seed)
    room = float(rng.uniform(0.004, 0.016))
    probe = BoardAudioLoop(FakeAgent())
    onset = onset_for(probe, room)
    calibration = np.clip(rng.normal(room, 0.00008, 10), 0.0001, None).tolist()
    ambient = []
    for i in range(220):
        if i % 23 == 11:
            ambient.append(onset + 0.005)       # door/click, always isolated
        else:
            ambient.append(float(min(onset - 0.0003,
                                     max(0.0001, rng.normal(room, 0.00010)))))
    high = max(onset + 0.005, room + 0.008)
    speech = [high if i % 8 else room + 0.0002 for i in range(36)]
    quiet = [room] * 24

    loop, proc = run_levels(calibration + ambient + speech + quiet)

    assert loop._answer.call_count == 1, f"seed={seed}, room={room:.4f}"
    # If an isolated impact had latched the turn, capture would have returned long
    # before consuming the full ambient prefix.
    assert proc.read_count > 250


def test_fifty_consecutive_turns_rearm_and_track_a_drifting_room():
    rng = np.random.default_rng(20260730)
    loop = BoardAudioLoop(FakeAgent())
    loop.capture_device = "plughw:1,0"
    loop._check_for_dead_microphone = mock.Mock()
    loop._answer = mock.Mock()
    processes = []

    for turn in range(50):
        room = 0.010 + 0.00005 * turn
        high = room + 0.008
        levels = (
            np.clip(rng.normal(room, 0.00008, 10), 0.0001, None).tolist()
            + [room] * 5
            + [high, high, room + 0.0002, high] * 8
            + [room] * 24
        )
        processes.append(CaptureProc(levels))

    with mock.patch("voice_agent.board_audio.subprocess.Popen",
                    side_effect=processes):
        for _ in range(50):
            loop._listen_once()

    assert loop._answer.call_count == 50
    assert all(proc.terminated for proc in processes)
    assert abs(loop.noise_floor - (0.010 + 0.00005 * 49)) < 0.0015


def test_capture_eof_always_terminates_and_reaps_arecord():
    proc = CaptureProc([0.0143] * 3)
    loop = BoardAudioLoop(FakeAgent())
    loop.capture_device = "plughw:1,0"
    loop._check_for_dead_microphone = mock.Mock()

    with mock.patch("voice_agent.board_audio.subprocess.Popen", return_value=proc), \
            pytest.raises(RuntimeError, match="stopped returning audio"):
        loop._listen_once()

    assert proc.terminated
