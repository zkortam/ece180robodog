"""Speech chunking and noise rejection: the two pure text paths on every turn."""
import pytest

from voice_agent import config
from voice_agent.stt import is_noise_transcript
from voice_agent.tts import _split_lead, sentence_chunks


# ---------------------------------------------------------------- noise policy

@pytest.mark.parametrize("text", ["", "   ", ".", "...", "!?", "\n"])
def test_empty_and_punctuation_only_is_noise(text):
    assert is_noise_transcript(text)


@pytest.mark.parametrize("text", ["you", "You.", "  YOU  ", "thanks for watching",
                                  "[BLANK_AUDIO]", "uh", "um", "Hmm."])
def test_known_stt_artifacts_are_noise(text):
    assert is_noise_transcript(text)


@pytest.mark.parametrize("text", ["yeah", "yes", "okay", "ok", "no", "thanks",
                                  "thank you", "bye", "stop", "sit", "who am I"])
def test_real_words_are_never_noise(text):
    """Regression: the orchestrator kept a second, wider noise list that swallowed
    these. "yeah" is the expected answer to the agent's own registration question
    ("want me to learn a few expressions?"), so that turn vanished with no feedback
    and the flow stalled. Short commands matter even more once there are legs."""
    assert not is_noise_transcript(text)


def test_noise_policy_has_exactly_one_definition():
    """Regression: orchestrator._is_noise and stt._is_artifact disagreed. The
    orchestrator must not reintroduce a private list."""
    import voice_agent.orchestrator as orch
    assert not hasattr(orch, "_NOISE_TRANSCRIPTS")
    assert orch.is_noise_transcript is is_noise_transcript


# ------------------------------------------------------------ sentence chunking

def test_empty_reply_yields_no_chunks():
    assert sentence_chunks("") == []
    assert sentence_chunks(None) == []


def test_single_sentence_is_one_chunk():
    assert sentence_chunks("I see you.") == ["I see you."]


def test_splits_on_sentence_boundaries():
    # Each opener here is over the 18-char merge threshold, so none is folded
    # forward and the boundaries survive intact.
    parts = sentence_chunks("I can see you clearly. You look happy today. "
                            "Want me to sit down?")
    assert len(parts) == 3
    assert parts[0].startswith("I can see you clearly")


@pytest.mark.parametrize("text", [
    "Dr. Smith is here.",
    "That is e.g. a camera.",
    "Meet me at 3 p.m. today.",
])
def test_abbreviations_do_not_split(text):
    """Splitting mid-abbreviation changes the synthesizer's phrasing."""
    assert sentence_chunks(text) == [" ".join(text.split())]


def test_chunks_reassemble_to_the_original_words():
    reply = "Hello there. I recognize you, Zakaria. How are you feeling today?"
    assert " ".join(" ".join(sentence_chunks(reply)).split()) == reply


def test_chunk_count_is_capped():
    reply = " ".join(f"Sentence number {i}." for i in range(20))
    parts = sentence_chunks(reply, max_chunks=5)
    assert len(parts) <= 5
    assert " ".join(" ".join(parts).split()) == reply


def test_very_short_opener_merges_forward():
    """"Sure." alone is too short to be worth a synthesis round trip."""
    parts = sentence_chunks("Sure. I can see two people in the room right now.")
    assert parts[0].startswith("Sure. I can see")


def test_long_opener_splits_at_a_clause_boundary():
    """Time-to-first-word tracks the FIRST chunk, so a long opener must break."""
    opener = ("I can see you clearly right now, and you look pretty happy about "
              "something today.")
    parts = _split_lead(opener, limit=55)
    assert len(parts) == 2
    assert " ".join(" ".join(parts).split()) == opener
    assert len(parts[0]) <= 55


def test_lead_split_never_invents_or_drops_words():
    opener = "Yes I think so because the camera is working fine now"
    parts = _split_lead(opener, limit=40)
    assert " ".join(parts).split() == opener.split()


def test_unsplittable_opener_is_left_alone():
    """No comma, no conjunction: better one long chunk than a break mid-phrase."""
    opener = "Supercalifragilisticexpialidociousness notwithstanding whatsoever indeed"
    assert _split_lead(opener, limit=30) == [opener]


# ------------------------------------------------------- conversational policy

def test_default_prompt_is_conversational_not_forced_to_one_sentence():
    assert "two to four spoken sentences" in config.SYSTEM_PROMPT
    assert "ONE short sentence for a normal answer" not in config.SYSTEM_PROMPT
