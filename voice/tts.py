"""
Text-to-speech for the Bosman AI Coach (Phase 5).

Design mirrors vision/: keep the part that's pure logic (turning an
AdviceResponse into a natural sentence) separate from the part that
touches real hardware (the actual speech engine), so the logic can be
unit-tested without a working audio device.

pyttsx3 is used because it's fully offline (SAPI5 on Windows, no network
call, no API key) - consistent with the rest of this project running
entirely local via Ollama.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Only needed for type hints, not at runtime - this file must stay
    # importable even in an environment that doesn't have pydantic (and
    # therefore can't import core.schemas), the same way pyttsx3 itself
    # is imported lazily below rather than at module level.
    from core.schemas import AdviceResponse


def advice_to_speech_text(advice: AdviceResponse) -> str:
    """Turn a structured AdviceResponse into a short, natural sentence to
    speak aloud. Deliberately NOT just concatenating every field - a
    between-play voice line needs to be short enough to say and still
    listen to the next passage of play, so this keeps to the headline
    plus at most one instruction, not the full written breakdown (the
    GUI text panel still shows everything).
    """
    parts = [advice.top_suggestion.strip().rstrip(".") + "."]

    if advice.formation_change:
        parts.append(f"Switch to {advice.formation_change.value}.")
    if advice.style_change:
        parts.append(f"Play {_speakable_style(advice.style_change.value)}.")

    if advice.secondary_considerations:
        parts.append(advice.secondary_considerations[0].rstrip(".") + ".")

    return " ".join(parts)


def _speakable_style(style_value: str) -> str:
    """'high_press' -> 'high press' etc - enum values use underscores,
    which a TTS engine will otherwise read out literally."""
    return style_value.replace("_", " ")


@dataclass
class TTSStatus:
    available: bool
    reason: Optional[str] = None


class TTSEngine:
    """Thin wrapper around pyttsx3. Import is deferred to __init__ (not
    module level) so that importing this file never fails just because
    pyttsx3 or its OS-level speech backend isn't installed - the rest of
    the app (GUI, core, vision) must keep working either way.
    """

    def __init__(self, rate: int = 175, voice_id: Optional[str] = None):
        self._engine = None
        self._status = self._init_engine(rate, voice_id)

    def _init_engine(self, rate: int, voice_id: Optional[str]) -> TTSStatus:
        try:
            import pyttsx3
        except ImportError:
            return TTSStatus(
                available=False,
                reason=(
                    "pyttsx3 isn't installed. Run `pip install pyttsx3` "
                    "(Linux also needs espeak: `sudo apt install espeak`)."
                ),
            )
        try:
            engine = pyttsx3.init()
        except Exception as e:  # noqa: BLE001 - surface any backend init failure
            return TTSStatus(
                available=False,
                reason=f"pyttsx3 installed but couldn't start a speech engine: {e}",
            )

        engine.setProperty("rate", rate)
        if voice_id:
            engine.setProperty("voice", voice_id)
        self._engine = engine
        return TTSStatus(available=True)

    @property
    def status(self) -> TTSStatus:
        return self._status

    def is_available(self) -> bool:
        return self._status.available

    def speak(self, text: str) -> None:
        """Blocking call - speaks the full text before returning. Callers
        in the GUI must run this off the main thread (see
        gui/main_window.py's VoiceWorker) or the window will freeze for
        the duration of the speech, the same reasoning that put
        get_advice() on a background thread already."""
        if not self.is_available():
            raise RuntimeError(self._status.reason or "TTS engine unavailable")
        self._engine.say(text)
        self._engine.runAndWait()

    def speak_advice(self, advice: AdviceResponse) -> None:
        self.speak(advice_to_speech_text(advice))
