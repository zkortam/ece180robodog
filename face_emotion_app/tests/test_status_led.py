from unittest import mock

import pytest

from voice_agent.status_led import COLORS, StatusLED
from voice_agent.board_audio import BoardAudioLoop


def make_led(tmp_path):
    for channel in ("red", "green", "blue"):
        path = tmp_path / f"{channel}:user"
        path.mkdir()
        (path / "brightness").write_text("0")
    return StatusLED(tmp_path)


@pytest.mark.parametrize("state,rgb", COLORS.items())
def test_every_state_writes_the_expected_rgb(tmp_path, state, rgb):
    led = make_led(tmp_path)

    assert led.set(state)

    actual = tuple(int((tmp_path / f"{channel}:user" / "brightness").read_text())
                   for channel in ("red", "green", "blue"))
    assert actual == rgb
    assert led.state == state


def test_missing_board_led_is_a_safe_noop(tmp_path):
    led = StatusLED(tmp_path)

    assert not led.available
    assert not led.set("listening")


def test_matrix_works_when_linux_rgb_led_is_absent(tmp_path):
    """UNO Q images may omit sysfs RGB channels; the large matrix still works."""
    completed = mock.Mock(returncode=0)
    led = StatusLED(tmp_path, matrix_command="/usr/bin/arduino-router-cli")
    with mock.patch("voice_agent.status_led.subprocess.run", return_value=completed) as run:
        assert led.set("listening")
    run.assert_called_once()
    assert led.state == "listening"


def test_duplicate_state_does_not_rewrite_sysfs(tmp_path):
    led = make_led(tmp_path)
    assert led.set("listening")

    with mock.patch("pathlib.Path.write_text",
                    side_effect=AssertionError("duplicate state rewrote LED")):
        assert led.set("listening")


def test_refresh_reasserts_state_after_an_external_overwrite(tmp_path):
    led = make_led(tmp_path)
    led._refresh_interval = 0.01
    assert led.set("waiting")
    red = tmp_path / "red:user" / "brightness"
    green = tmp_path / "green:user" / "brightness"
    red.write_text("1")
    green.write_text("0")

    for _ in range(30):
        if red.read_text() == "1" and green.read_text() == "1":
            break
        import time
        time.sleep(0.01)

    assert (red.read_text(), green.read_text()) == ("1", "1")


def test_write_failure_disables_led_without_breaking_caller(tmp_path):
    led = make_led(tmp_path)
    with mock.patch("pathlib.Path.write_text", side_effect=OSError("denied")):
        assert not led.set("error")
    assert not led.available


def test_unknown_state_is_rejected_even_without_hardware(tmp_path):
    led = StatusLED(tmp_path)
    with pytest.raises(ValueError, match="unknown LED state"):
        led.set("confused")


def test_state_is_sent_to_matrix_bridge(tmp_path):
    led = make_led(tmp_path)
    led._matrix_command = "/usr/bin/arduino-router-cli"
    completed = mock.Mock(returncode=0)
    with mock.patch("voice_agent.status_led.subprocess.run",
                    return_value=completed) as run:
        assert led.set("hearing")

    run.assert_called_once_with(
        ["/usr/bin/arduino-router-cli", "set_robodog_status", "hearing"],
        stdout=-3, stderr=-3, timeout=1, check=False)


def test_matrix_bridge_failure_never_breaks_rgb_status(tmp_path):
    led = make_led(tmp_path)
    led._matrix_command = "/missing/router"
    with mock.patch("voice_agent.status_led.subprocess.run",
                    side_effect=OSError("missing")):
        assert led.set("error")
    assert led.state == "error"


class StateProbe:
    def __init__(self):
        self.states = []

    def set(self, state):
        self.states.append(state)
        return True


class Agent:
    pass


def test_missing_audio_hardware_shows_waiting(monkeypatch):
    probe = StateProbe()
    loop = BoardAudioLoop(Agent(), led=probe)
    loop._stop.wait = mock.Mock(side_effect=lambda _: loop._stop.set())
    monkeypatch.setattr(loop, "_discover", lambda: False)

    loop._run()

    assert probe.states == ["waiting"]


def test_valid_speech_moves_from_listening_to_hearing_to_thinking(monkeypatch):
    from test_board_vad import CaptureProc

    probe = StateProbe()
    loop = BoardAudioLoop(Agent(), led=probe)
    loop.capture_device = "plughw:1,0"
    loop._check_for_dead_microphone = mock.Mock()
    loop._answer = mock.Mock()
    levels = [0.0143] * 10 + [0.0188] * 22 + [0.0142] * 24
    monkeypatch.setattr("voice_agent.board_audio.subprocess.Popen",
                        lambda *args, **kwargs: CaptureProc(levels))

    loop._listen_once()

    assert probe.states[0] == "listening"
    assert "hearing" in probe.states
    assert probe.states[-1] == "thinking"


def test_reply_marks_speaking_then_returns_to_listening(monkeypatch):
    from test_board_playback import FakeAgent, FakeProc

    probe = StateProbe()
    loop = BoardAudioLoop(FakeAgent(), led=probe)
    monkeypatch.setattr(loop, "_open_playback", lambda fmt: FakeProc())

    assert loop._speak_reply("Hello there.")[0]

    assert probe.states == ["speaking", "listening"]
