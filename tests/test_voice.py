"""
Tests for the voice module (Phase 5).

What's genuinely tested here, for real, in this sandbox:
  - advice_to_speech_text(): pure function, no hardware needed.
  - TTSEngine / VoiceInputEngine degrade to "unavailable" with a clear
    reason instead of crashing when pyttsx3/vosk/a microphone aren't
    present - which is the actual state of THIS sandbox, so this is a
    real, meaningful pass, not a mocked-out assumption.

What's NOT tested here (documented, not hidden):
  - Actually speaking text out loud, or actually transcribing real
    speech. Neither pyttsx3 nor vosk nor a working audio device is
    available in this sandbox - see voice/README.md for what you need
    to install on your own machine to exercise those paths for real.

A NOTE ON WHY THIS DOESN'T IMPORT core.schemas.AdviceResponse:
This sandbox doesn't have pydantic installed and has no network access
to install it, so core/schemas.py (which imports pydantic) can't be
imported here right now. advice_to_speech_text() only ever does plain
attribute access on whatever it's given (no isinstance/pydantic-specific
behaviour), so these tests use a minimal stand-in with the same shape
instead. Once you run this on your machine with pydantic installed
(`pip install -r requirements.txt`), swap in the real AdviceResponse/
Formation/PlayingStyle if you want extra confidence - the logic itself
doesn't change either way, this is purely a sandbox limitation.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice.tts import advice_to_speech_text, TTSEngine
from voice.stt import VoiceInputEngine


@dataclass
class _FakeEnumValue:
    value: str


@dataclass
class _FakeAdviceResponse:
    """Stand-in for core.schemas.AdviceResponse - see module docstring."""
    summary: str
    reasoning: str = ""
    formation_change: object = None
    style_change: object = None
    substitution_suggestions: list = field(default_factory=list)
    tactical_instructions: list = field(default_factory=list)


def test_advice_to_speech_text_headline_only():
    advice = _FakeAdviceResponse(
        summary="Hold your shape and see the game out.",
        reasoning="You're a goal up with 5 minutes left.",
    )
    text = advice_to_speech_text(advice)
    assert text == "Hold your shape and see the game out."
    print("OK - headline-only advice speaks as just the summary")


def test_advice_to_speech_text_includes_formation_and_style():
    advice = _FakeAdviceResponse(
        summary="Shut up shop.",
        formation_change=_FakeEnumValue("5-3-2"),
        style_change=_FakeEnumValue("park_the_bus"),
        reasoning="Protect the lead.",
    )
    text = advice_to_speech_text(advice)
    assert "Switch to 5-3-2." in text
    assert "Play park the bus." in text  # underscore -> space for speakability
    print("OK - formation/style changes are spoken, underscores turned into spaces")


def test_advice_to_speech_text_prefers_substitution_over_tactical():
    advice = _FakeAdviceResponse(
        summary="Freshen up the front line.",
        substitution_suggestions=["Bring on a fresh striker for the tired one"],
        tactical_instructions=["Push fullbacks higher", "Overload the left"],
        reasoning="Fatigue is showing.",
    )
    text = advice_to_speech_text(advice)
    assert "Bring on a fresh striker for the tired one." in text
    assert "Push fullbacks higher" not in text
    print("OK - one substitution suggestion is spoken; tactical instructions dropped to keep it short")


def test_advice_to_speech_text_falls_back_to_tactical_instruction():
    advice = _FakeAdviceResponse(
        summary="Change how you're pressing.",
        tactical_instructions=["Press higher up the pitch", "Close down the ball carrier faster"],
        reasoning="You're not winning the ball back quickly enough.",
    )
    text = advice_to_speech_text(advice)
    assert "Press higher up the pitch." in text
    assert "Close down the ball carrier faster" not in text
    print("OK - falls back to first tactical instruction when there's no substitution")


def test_tts_engine_degrades_gracefully_without_pyttsx3():
    """This sandbox genuinely doesn't have pyttsx3 installed - so this
    is a real check that TTSEngine() doesn't raise, and reports a clear,
    actionable reason rather than a bare stack trace."""
    engine = TTSEngine()
    assert engine.is_available() is False
    assert engine.status.reason is not None and "pyttsx3" in engine.status.reason
    print(f"OK - TTSEngine degrades cleanly: {engine.status.reason}")


def test_tts_engine_speak_raises_clear_error_when_unavailable():
    engine = TTSEngine()
    try:
        engine.speak("this should not work here")
        raise AssertionError("expected RuntimeError when TTS engine is unavailable")
    except RuntimeError as e:
        assert "pyttsx3" in str(e)
        print("OK - calling speak() on an unavailable engine raises a clear RuntimeError, not a crash")


def test_stt_engine_degrades_gracefully_without_vosk():
    """Same story as TTS - vosk isn't installed here, and even if it
    were, no model directory exists at the placeholder path. Either way
    this must fail soft with a clear reason."""
    engine = VoiceInputEngine(model_path="/nonexistent/vosk-model")
    assert engine.is_available() is False
    assert engine.status.reason is not None
    print(f"OK - VoiceInputEngine degrades cleanly: {engine.status.reason}")


def test_stt_engine_stop_recording_without_start_is_safe():
    engine = VoiceInputEngine(model_path="/nonexistent/vosk-model")
    # Never called start_recording() - must not raise, must return "".
    assert engine.stop_recording() == ""
    print("OK - stop_recording() before start_recording() returns '' instead of raising")


def test_stt_engine_start_recording_raises_clear_error_when_unavailable():
    engine = VoiceInputEngine(model_path="/nonexistent/vosk-model")
    try:
        engine.start_recording()
        raise AssertionError("expected RuntimeError when STT engine is unavailable")
    except RuntimeError as e:
        assert str(e)
        print("OK - start_recording() on an unavailable engine raises a clear RuntimeError, not a crash")


if __name__ == "__main__":
    test_advice_to_speech_text_headline_only()
    test_advice_to_speech_text_includes_formation_and_style()
    test_advice_to_speech_text_prefers_substitution_over_tactical()
    test_advice_to_speech_text_falls_back_to_tactical_instruction()
    test_tts_engine_degrades_gracefully_without_pyttsx3()
    test_tts_engine_speak_raises_clear_error_when_unavailable()
    test_stt_engine_degrades_gracefully_without_vosk()
    test_stt_engine_stop_recording_without_start_is_safe()
    test_stt_engine_start_recording_raises_clear_error_when_unavailable()
    print("\nAll voice tests passed.")
