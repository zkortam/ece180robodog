"""Streaming speech on the standalone robot.

Synthesis runs at roughly real time, so waiting for a whole reply before opening
your mouth doubles the silence the person sits through. The browser transport
already streamed sentence chunks; these pin the same behaviour for the board,
which is the deployment with no browser to fall back on.
"""
import io
import subprocess
import wave
from unittest import mock

import pytest

from voice_agent.board_audio import BoardAudioLoop, _wav_pcm, set_playback_volume


def wav_bytes(seconds=0.2, rate=24000, channels=1, width=2, value=1000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        frames = int(rate * seconds)
        w.writeframes(value.to_bytes(width, "little", signed=True) * frames * channels)
    return buf.getvalue()


class FakeTTS:
    def __init__(self, rate=24000):
        self.calls = []
        self.rate = rate

    def synth(self, text):
        self.calls.append(text)
        return wav_bytes(rate=self.rate)


class FakeAgent:
    def __init__(self, rate=24000):
        self.tts = FakeTTS(rate)


class FakeProc:
    """Stands in for aplay reading raw PCM from stdin."""

    def __init__(self):
        self.stdin = mock.Mock()
        self.written = []
        self.stdin.write.side_effect = self.written.append
        self.waited = False

    def wait(self, timeout=None):
        self.waited = True

    def kill(self):
        pass


@pytest.fixture
def loop():
    board = BoardAudioLoop(FakeAgent())
    board.playback_device = "plughw:1,0"
    return board


# --------------------------------------------------------------- wav parsing

def test_pcm_and_format_are_read_from_the_header():
    """Assuming a rate does not fail loudly -- it plays the reply at the wrong
    speed and pitch. Kokoro is 24 kHz, Piper depends on the voice, espeak differs
    again."""
    pcm, fmt = _wav_pcm(wav_bytes(seconds=0.1, rate=22050))
    assert fmt == (22050, 1, 2)
    assert len(pcm) == int(22050 * 0.1) * 2


def test_a_non_wav_payload_raises_rather_than_playing_noise():
    with pytest.raises((wave.Error, EOFError)):
        _wav_pcm(b"this is not a wav file at all")


# ------------------------------------------------------------- streaming out

def test_each_sentence_is_written_as_it_is_synthesized(loop):
    reply = "I can see you clearly. You look happy today. Shall I remember you?"
    procs = []

    def open_playback(fmt):
        procs.append(FakeProc())
        return procs[-1]

    loop._open_playback = open_playback
    assert loop._speak_reply(reply)[0] is True
    assert len(loop.agent.tts.calls) >= 2, "the reply must be split, not synthesized whole"
    assert len(procs) == 1, "one continuous stream, not one aplay per sentence"
    assert len(procs[0].written) == len(loop.agent.tts.calls)


def test_playback_starts_before_the_whole_reply_is_synthesized(loop):
    """The point of the exercise: aplay must be handed the first sentence while
    later sentences are still being made."""
    import threading
    order = []
    first_written = threading.Event()
    original = loop.agent.tts.synth
    synth_calls = 0

    def synth(text):
        nonlocal synth_calls
        synth_calls += 1
        order.append(("synth-start", text))
        if synth_calls > 1:
            assert first_written.wait(1), "later synthesis must overlap first playback"
        out = original(text)
        order.append(("synth-done", text))
        return out

    loop.agent.tts.synth = synth

    def open_playback(fmt):
        order.append(("playback-open", None))
        proc = FakeProc()
        proc.stdin.write.side_effect = lambda data: (
            order.append(("write", len(data))), first_written.set())
        return proc

    loop._open_playback = open_playback
    loop._speak_reply("First sentence here. Second sentence here. Third one here.")

    kinds = [k for k, _ in order]
    assert kinds[0] == "synth-start"
    assert kinds.index("playback-open") < kinds.index("write")
    # The next chunk began synthesis concurrently and did not need to finish
    # before the first chunk was handed to playback.
    first_write = kinds.index("write")
    assert "synth-start" in kinds[first_write + 1:] or kinds.count("synth-start") > 1


def test_an_empty_reply_speaks_nothing(loop):
    loop._open_playback = mock.Mock()
    assert loop._speak_reply("")[0] is False
    loop._open_playback.assert_not_called()


def test_a_backend_returning_no_audio_reports_that_it_did_not_speak(loop):
    loop.agent.tts.synth = lambda text: b""
    loop._open_playback = mock.Mock()
    assert loop._speak_reply("Hello there, this is a reply.")[0] is False


def test_unplayable_synthesis_is_skipped_not_fatal(loop, capsys):
    """One malformed chunk must not lose the rest of the sentence."""
    outputs = [b"garbage not a wav", wav_bytes(), wav_bytes()]
    loop.agent.tts.synth = lambda text: outputs.pop(0) if outputs else wav_bytes()
    procs = []
    loop._open_playback = lambda fmt: procs.append(FakeProc()) or procs[-1]
    assert loop._speak_reply("One here. Two here. Three here.")[0] is True
    assert "unplayable" in capsys.readouterr().out


def test_a_dead_playback_process_does_not_crash_the_loop(loop):
    """The speaker being yanked mid-reply must not take down the listen loop --
    that loop is the robot's only input."""
    proc = FakeProc()
    proc.stdin.write.side_effect = BrokenPipeError("speaker gone")
    loop._open_playback = lambda fmt: proc
    assert loop._speak_reply("A reply that cannot be played out loud.")[0] is False


def test_playback_that_will_not_start_is_reported_not_hidden(loop):
    loop._open_playback = lambda fmt: None
    assert loop._speak_reply("Some reply here.")[0] is False


def test_a_rate_change_mid_reply_opens_a_new_stream(loop):
    """A fallback voice kicking in mid-reply has a different rate; feeding it to a
    stream opened for the old rate would play noise."""
    rates = [24000, 24000, 16000]
    loop.agent.tts.synth = lambda text: wav_bytes(rate=rates.pop(0) if rates else 16000)
    opened = []
    loop._open_playback = lambda fmt: (opened.append(fmt), FakeProc())[1]
    loop._speak_reply("One sentence here. Two sentence here. Three sentence here.")
    assert len(opened) == 2
    assert opened[0][0] == 24000 and opened[1][0] == 16000


def test_the_stream_is_always_drained_and_reaped(loop):
    proc = FakeProc()
    loop._open_playback = lambda fmt: proc
    loop._speak_reply("A short reply.")
    proc.stdin.close.assert_called()
    assert proc.waited


def test_the_stream_is_reaped_even_when_writing_fails(loop):
    proc = FakeProc()
    proc.stdin.write.side_effect = OSError("gone")
    loop._open_playback = lambda fmt: proc
    loop._speak_reply("A short reply.")
    assert proc.waited


# ------------------------------------------------------- aplay invocation

def test_aplay_is_told_the_right_format(loop):
    # set_playback_volume is stubbed out: it shells out to amixer, and this test is
    # about the aplay argv, not the mixer.
    with mock.patch("voice_agent.board_audio.set_playback_volume"), \
            mock.patch("subprocess.Popen") as popen:
        loop._open_playback((22050, 1, 2))
    argv = popen.call_args.args[0]
    assert argv[:3] == ["aplay", "-D", "plughw:1,0"]
    assert "-t" in argv and argv[argv.index("-t") + 1] == "raw"
    assert argv[argv.index("-f") + 1] == "S16_LE"
    assert argv[argv.index("-r") + 1] == "22050"
    assert argv[argv.index("-c") + 1] == "1"


def test_an_unsupported_sample_width_is_refused_rather_than_played_as_noise(loop):
    assert loop._open_playback((16000, 1, 7)) is None


def test_the_speaker_volume_is_asserted_before_every_playback(loop):
    """Observed on the board: the mixer reads 100% just after discovery sets it and
    40% by the time a reply plays -- USB audio reverts to its factory default across
    re-enumeration and stream open, and 40% is 16 dB down. Setting it per playback
    is what makes the loudness deterministic instead of a race."""
    with mock.patch("voice_agent.board_audio.set_playback_volume") as setvol, \
            mock.patch("subprocess.Popen"):
        loop._open_playback((16000, 1, 2))
    setvol.assert_called_once_with(loop.playback_device)


def test_volume_setting_never_changes_capture_or_sidetone_controls():
    listing = mock.Mock(stdout=(
        "Simple mixer control 'PCM',0\n"
        "Simple mixer control 'Mic',0\n"
        "Simple mixer control 'Sidetone',0\n"))
    pcm = mock.Mock(stdout="Capabilities: pvolume pswitch\nPlayback channels: Front Left\n")
    applied = mock.Mock(returncode=0)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "scontrols" in argv:
            return listing
        if "sget" in argv:
            return pcm
        return applied

    with mock.patch("voice_agent.board_audio.subprocess.run", side_effect=run):
        assert set_playback_volume("plughw:2,0")

    set_names = [argv[4] for argv in calls if "sset" in argv]
    assert set_names == ["PCM"]


def test_no_speaker_means_no_playback(loop):
    loop.playback_device = None
    assert loop._open_playback((16000, 1, 2)) is None


def test_a_missing_aplay_binary_is_reported_not_raised(loop):
    with mock.patch("subprocess.Popen", side_effect=FileNotFoundError("no aplay")):
        assert loop._open_playback((16000, 1, 2)) is None


def test_a_hung_playback_process_is_killed(loop):
    proc = mock.Mock()
    proc.wait.side_effect = [subprocess.TimeoutExpired("aplay", 60), None]
    BoardAudioLoop._close_playback(proc)
    proc.kill.assert_called_once()
