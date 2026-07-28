"""USB device discovery on the board.

Everything hangs off one hub: webcam, microphone, speaker. Two consequences drive
all of this:

  * Enumeration order is not stable. Which port you used, and how fast each device
    powered up, decides the card and /dev/video numbers. Anything that assumes
    "index 0" is a coin flip that changes between boots.
  * /dev/video* is not a list of cameras. This SoC's hardware encoder and decoder
    claim nodes there too.

Real `arecord -l` / `aplay -l` output is used as fixtures so the parsing is pinned
against the format the board actually prints.
"""
import struct
from unittest import mock

import pytest

from voice_agent import board_audio
from vision_service import capture_device_indexes, v4l2_capture_capability

# Real `arecord -l` output shape: the board's own codec plus a USB webcam mic and
# a USB speaker, as they appear when all three are present through a hub.
ARECORD_HUB = """**** List of CAPTURE Hardware Devices ****
card 0: ArduinoImola [Arduino Imola], device 0: Primary [Primary]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 2: C270 [HD Webcam C270], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""

APLAY_HUB = """**** List of PLAYBACK Hardware Devices ****
card 0: ArduinoImola [Arduino Imola], device 0: Primary [Primary]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""

ARECORD_BOARD_ONLY = """**** List of CAPTURE Hardware Devices ****
card 0: ArduinoImola [Arduino Imola], device 0: Primary [Primary]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""


def fake_aplay(output):
    def run(cmd, **kwargs):
        return mock.Mock(stdout=output)
    return run


# ------------------------------------------------------------------- parsing

def test_devices_are_parsed_from_real_arecord_output():
    with mock.patch("subprocess.run", fake_aplay(ARECORD_HUB)):
        found = board_audio.list_devices("arecord")
    assert [d["device"] for d in found] == ["plughw:1,0", "plughw:2,0"]


def test_the_boards_own_codec_is_never_selectable(monkeypatch):
    """Nothing is wired to it. Choosing it is the silent-failure case: the robot
    opens the device, streams perfect silence, and never reacts to anything."""
    with mock.patch("subprocess.run", fake_aplay(ARECORD_HUB)):
        found = board_audio.list_devices("arecord")
    assert all("Imola" not in d["label"] for d in found)


def test_only_the_board_codec_present_means_no_usable_device():
    with mock.patch("subprocess.run", fake_aplay(ARECORD_BOARD_ONLY)):
        assert board_audio.list_devices("arecord") == []


def test_a_missing_alsa_binary_is_not_a_crash():
    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        assert board_audio.list_devices("arecord") == []


def test_plughw_is_used_so_odd_sample_rates_still_open():
    """hw: fails outright if the device cannot do 16 kHz mono; plughw converts."""
    with mock.patch("subprocess.run", fake_aplay(ARECORD_HUB)):
        found = board_audio.list_devices("arecord")
    assert all(d["device"].startswith("plughw:") for d in found)


# ----------------------------------------------------------------- selection

def test_the_webcam_mic_wins_over_a_generic_usb_device():
    """THE hub regression. Both are USB capture devices and the old code took
    whichever enumerated first -- so the microphone the robot listened on changed
    depending on which port things were plugged into."""
    with mock.patch("subprocess.run", fake_aplay(ARECORD_HUB)):
        found = board_audio.list_devices("arecord")
    chosen = board_audio.pick_device(found, "capture")
    assert "Webcam" in chosen["label"]
    assert chosen["device"] == "plughw:2,0"     # NOT the lower-numbered card


def test_selection_is_stable_regardless_of_enumeration_order():
    """The same hardware must give the same answer if card numbers swap."""
    with mock.patch("subprocess.run", fake_aplay(ARECORD_HUB)):
        found = board_audio.list_devices("arecord")
    forwards = board_audio.pick_device(found, "capture")
    backwards = board_audio.pick_device(list(reversed(found)), "capture")
    assert forwards["label"] == backwards["label"]


def test_an_exact_device_override_wins_outright():
    with mock.patch("subprocess.run", fake_aplay(ARECORD_HUB)):
        found = board_audio.list_devices("arecord")
    chosen = board_audio.pick_device(found, "capture", override="plughw:5,0")
    assert chosen["device"] == "plughw:5,0"


def test_a_name_match_pins_a_device_across_replugs():
    """Card numbers move when you re-plug; the name does not."""
    with mock.patch("subprocess.run", fake_aplay(ARECORD_HUB)):
        found = board_audio.list_devices("arecord")
    chosen = board_audio.pick_device(found, "capture", match="PnP")
    assert chosen["device"] == "plughw:1,0"


def test_a_name_match_that_matches_nothing_does_not_silently_fall_back():
    """Asking for a specific microphone and quietly getting a different one is
    worse than being told nothing matched."""
    with mock.patch("subprocess.run", fake_aplay(ARECORD_HUB)):
        found = board_audio.list_devices("arecord")
    assert board_audio.pick_device(found, "capture", match="Blue Yeti") is None


def test_no_candidates_selects_nothing():
    assert board_audio.pick_device([], "capture") is None


def test_a_usb_speaker_is_chosen_for_playback():
    with mock.patch("subprocess.run", fake_aplay(APLAY_HUB)):
        found = board_audio.list_devices("aplay")
    assert board_audio.pick_device(found, "playback")["device"] == "plughw:1,0"


# ------------------------------------------------- dead microphone detection

class FakeAgent:
    def __init__(self):
        self.spoken = []

        class TTS:
            def synth(_self, text):
                return b""
        self.tts = TTS()


@pytest.fixture
def loop():
    board = board_audio.BoardAudioLoop(FakeAgent())
    board.capture_device = "plughw:0,0"
    board._capture_label = "Arduino Imola [Arduino Imola]"
    board._alternatives = {"capture": ["a", "b"], "playback": []}
    board._speak = lambda message: board.__dict__.setdefault("said", []).append(message)
    return board


def test_bit_exact_silence_is_eventually_reported(loop, capsys):
    """The one hardware fault with NO symptom: a capture device with nothing wired
    to it opens fine and streams perfect zeros forever. The VAD never fires, so
    the robot never speaks and never errors -- identical to being ignored."""
    import numpy as np
    silence = np.zeros(320, dtype="<i2")
    for _ in range(loop.DEAD_MIC_CHUNKS + 1):
        loop._check_for_dead_microphone(silence)
    assert loop._warned_dead_mic
    assert "bit-exact silence" in capsys.readouterr().out
    assert loop.__dict__.get("said"), "should say it out loud when never heard anything"


def test_a_quiet_room_is_not_mistaken_for_a_dead_microphone(loop):
    """A real microphone always has a noise floor. Only exact zeros are the tell."""
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(loop.DEAD_MIC_CHUNKS * 2):
        faint = rng.integers(-3, 4, 320).astype("<i2")   # inaudible, but not zero
        loop._check_for_dead_microphone(faint)
    assert not loop._warned_dead_mic


def test_the_warning_is_given_once_not_forever(loop, capsys):
    import numpy as np
    silence = np.zeros(320, dtype="<i2")
    for _ in range(loop.DEAD_MIC_CHUNKS * 3):
        loop._check_for_dead_microphone(silence)
    assert capsys.readouterr().out.count("bit-exact silence") == 1


def test_a_microphone_that_worked_then_went_quiet_is_not_announced_aloud(loop):
    """Probably unplugged mid-session. Logging it helps; talking to an empty room
    does not."""
    import numpy as np
    loop._check_for_dead_microphone(np.full(320, 50, dtype="<i2"))   # heard something
    for _ in range(loop.DEAD_MIC_CHUNKS + 1):
        loop._check_for_dead_microphone(np.zeros(320, dtype="<i2"))
    assert loop._warned_dead_mic
    assert not loop.__dict__.get("said")


def test_any_real_audio_resets_the_silence_counter(loop):
    import numpy as np
    for _ in range(100):
        loop._check_for_dead_microphone(np.zeros(320, dtype="<i2"))
    assert loop._silent_chunks == 100
    loop._check_for_dead_microphone(np.full(320, 10, dtype="<i2"))
    assert loop._silent_chunks == 0


# --------------------------------------------------- V4L2 capability querying

def querycap_bytes(caps, device_caps, card=b"HD Webcam C270"):
    return struct.pack("16s32s32sII I 3I", b"uvcvideo", card, b"usb-0000:01:00.0",
                       0x00060009, caps, device_caps, 0, 0, 0)


CAP_CAPTURE = 0x00000001
CAP_M2M = 0x00008000
CAP_DEVICE_CAPS = 0x80000000


def test_a_uvc_webcam_is_recognized_as_a_camera(tmp_path):
    node = tmp_path / "video0"
    node.write_bytes(b"")
    payload = querycap_bytes(CAP_CAPTURE | CAP_DEVICE_CAPS, CAP_CAPTURE)
    with mock.patch("fcntl.ioctl", return_value=payload):
        is_camera, name = v4l2_capture_capability(str(node))
    assert is_camera is True
    assert name == "HD Webcam C270"


def test_a_hardware_encoder_node_is_not_a_camera(tmp_path):
    """THE Qualcomm regression. The SoC's encoder claims /dev/video* and its
    whole-device `capabilities` advertises capture, so probing it by opening it
    could hang the watcher thread -- the only thing that can restore vision."""
    node = tmp_path / "video1"
    node.write_bytes(b"")
    payload = querycap_bytes(CAP_CAPTURE | CAP_M2M | CAP_DEVICE_CAPS,
                             CAP_M2M, card=b"qcom-venus-enc")
    with mock.patch("fcntl.ioctl", return_value=payload):
        is_camera, name = v4l2_capture_capability(str(node))
    assert is_camera is False
    assert "venus" in name


def test_device_caps_is_preferred_over_whole_device_capabilities(tmp_path):
    """A multi-node device advertises capture on EVERY node in `capabilities`;
    only `device_caps` describes the node actually opened."""
    node = tmp_path / "video2"
    node.write_bytes(b"")
    payload = querycap_bytes(CAP_CAPTURE | CAP_DEVICE_CAPS, CAP_M2M)
    with mock.patch("fcntl.ioctl", return_value=payload):
        is_camera, _ = v4l2_capture_capability(str(node))
    assert is_camera is False


def test_an_unqueryable_node_is_unknown_not_rejected(tmp_path):
    """Returning False would declare an unusual driver 'not a camera' and leave
    the robot blind; unknown lets the caller still try it as a last resort."""
    node = tmp_path / "video3"
    node.write_bytes(b"")
    with mock.patch("fcntl.ioctl", side_effect=OSError("not a v4l2 device")):
        is_camera, _ = v4l2_capture_capability(str(node))
    assert is_camera is None


def test_a_missing_node_is_unknown_not_a_crash():
    assert v4l2_capture_capability("/dev/video-does-not-exist")[0] is None


def test_real_cameras_are_ordered_before_unqueryable_nodes(monkeypatch):
    """A confirmed camera must be tried before a node we merely could not rule out."""
    monkeypatch.setattr("glob.glob", lambda pattern: [
        "/dev/video0", "/dev/video1", "/dev/video2", "/dev/video-not-a-number"])

    def capability(path):
        return {"/dev/video0": (False, "enc"),      # encoder: excluded entirely
                "/dev/video1": (None, ""),          # unknown: last resort
                "/dev/video2": (True, "C270")}[path]

    monkeypatch.setattr("vision_service.v4l2_capture_capability", capability)
    assert capture_device_indexes() == [(2, "C270"), (1, "")]


def test_no_video_nodes_yields_no_candidates(monkeypatch):
    monkeypatch.setattr("glob.glob", lambda pattern: [])
    assert capture_device_indexes() == []


# ------------------------------------------- repeated capture-device failures

def test_a_persistently_broken_microphone_is_announced_and_backed_off(loop, capsys):
    """A hub that keeps browning out, or a port that is dying, raises on every
    read. Unhandled, the loop spins forever printing an identical line: the log
    scrolls too fast to read and the robot is simply deaf, with nothing said."""
    loop.capture_device = "plughw:2,0"
    loop._listen_once = mock.Mock(side_effect=RuntimeError("microphone stopped returning audio"))
    loop._stop = mock.Mock(is_set=mock.Mock(side_effect=[False] * 6 + [True]), wait=mock.Mock())
    loop._discover = mock.Mock(return_value=True)
    loop._run()

    out = capsys.readouterr().out
    # Collapsed, not repeated once per iteration.
    assert out.count("microphone stopped returning audio") <= 3
    assert "x5" in out                                  # the repeat count is surfaced
    assert loop.__dict__.get("said"), "a deaf robot must say so"
    # And it backs off harder once it is clearly not a blip.
    assert 5.0 in [c.args[0] for c in loop._stop.wait.call_args_list]


def test_a_recovered_microphone_re_arms_the_warning(loop):
    """A fault that recurs after a genuine recovery must be reported again, not
    suppressed forever by the first occurrence."""
    loop._warned_capture_broken = True
    loop._repeated_errors = 9
    loop._listen_once = mock.Mock(return_value=None)
    loop._stop = mock.Mock(is_set=mock.Mock(side_effect=[False, True]), wait=mock.Mock())
    loop._discover = mock.Mock(return_value=True)
    loop._run()
    assert loop._repeated_errors == 0
    assert loop._warned_capture_broken is False


def test_differing_errors_are_each_reported(loop, capsys):
    loop._listen_once = mock.Mock(side_effect=[RuntimeError("first"), RuntimeError("second")])
    loop._stop = mock.Mock(is_set=mock.Mock(side_effect=[False, False, True]), wait=mock.Mock())
    loop._discover = mock.Mock(return_value=True)
    loop._run()
    out = capsys.readouterr().out
    assert "first" in out and "second" in out
