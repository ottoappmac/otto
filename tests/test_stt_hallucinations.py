"""Tests for Whisper hallucination filtering in :func:`_is_hallucination`.

Whisper invents text when fed non-speech audio, and does so
non-deterministically: the same traffic-noise clip yielded "Sm Sm Sm …" ×223,
"you Thank you.", and an empty string across four runs.  The original
exact-string blocklist caught neither the looping nor rearranged filler, so
the junk reached the model under the heading "Audio transcript of the clip".
"""

from __future__ import annotations

import pytest

from backend.voice.stt import _is_hallucination

REAL_PARAGRAPH = (
    "So the way the deployment works is we push to the staging branch first "
    "and then the pipeline runs the integration tests. If those pass it "
    "promotes the build automatically, otherwise it rolls back and pages "
    "whoever is on call."
)


@pytest.mark.parametrize("text", [
    "",
    "   ",
    ".",
    "you",
    "thank you.",
    "Thanks for watching!",
])
def test_exact_blocklist_still_matches(text: str):
    assert _is_hallucination(text) is True


@pytest.mark.parametrize("text", [
    "Sm " * 223,
    "you you you you you you",
    "the the the the the the the the",
])
def test_degenerate_repetition_is_rejected(text: str):
    assert _is_hallucination(text) is True


@pytest.mark.parametrize("text", [
    "you Thank you.",          # rearrangement the exact set misses
    "See you next time",
    "Thanks for watching the video, please subscribe",
    "thank you so much.",
])
def test_filler_only_transcripts_are_rejected(text: str):
    assert _is_hallucination(text) is True


@pytest.mark.parametrize("text", [
    "Hey Otto, can you open the settings page for me?",
    "The cat sat on the mat",
    "Thank you for your help with the deployment yesterday",
    "See you tomorrow",
    "Stop",
    "Yes",
    REAL_PARAGRAPH,
])
def test_real_speech_is_kept(text: str):
    assert _is_hallucination(text) is False


def test_repetition_guard_ignores_short_transcripts():
    """Under the token floor, repetition alone must not reject real speech."""
    assert _is_hallucination("go go go") is False


def test_common_function_words_do_not_dominate_long_speech():
    """A genuine transcript repeats "the"/"to" without any token dominating."""
    assert _is_hallucination(REAL_PARAGRAPH) is False
