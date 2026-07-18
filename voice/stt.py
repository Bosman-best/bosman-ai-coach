"""
Speech-to-text for the Bosman AI Coach (Phase 5).

Uses vosk (fully offline, small local model file) rather than a
cloud speech API, for the same reason the whole project runs through
Ollama instead of a hosted LLM: no network dependency, no account, no
data leaving the machine.

This is used for dictating the free-text `notes` field only (e.g.
"keeper's on a yellow, left back looks gassed") - it does not attempt
to control the app or the game by voice command. Advice is still
triggered by clicking ASK COACH; voice input is a convenience for
typing less, not a hands-free automation layer.

Two hardware dependencies are required and are NOT bundled:
  1. `pip install vosk sounddevice` - the Python packages
  2. A vosk model directory downloaded separately (see voice/README.md)
Neither is available in the sandbox this was built in, so the audio
capture path is written but not exercised live - see voice/README.md
for exactly what was and wasn't tested and why.
"""

from __future__ import annotations
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class STTStatus:
    available: bool
    reason: Optional[str] = None


class VoiceInputEngine:
    """Wraps vosk + sounddevice behind a simple start/stop recording API.
    Imports are deferred to __init__, same reasoning as TTSEngine: a
    missing optional dependency must never break the rest of the app.
    """

    def __init__(self, model_path: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._model = None
        self._recognizer_cls = None
        self._sd = None
        self._status = self._init_engine(model_path)

        self._stream = None
        self._audio_q: "queue.Queue[bytes]" = queue.Queue()
        self._recording = threading.Event()

    def _init_engine(self, model_path: str) -> STTStatus:
        try:
            import vosk
            import sounddevice as sd
        except ImportError:
            return STTStatus(
                available=False,
                reason=(
                    "vosk and/or sounddevice isn't installed. Run "
                    "`pip install vosk sounddevice`. On Linux, sounddevice "
                    "also needs the PortAudio system library "
                    "(`sudo apt install portaudio19-dev`)."
                ),
            )

        if not Path(model_path).exists():
            return STTStatus(
                available=False,
                reason=(
                    f"vosk model not found at '{model_path}'. Download a "
                    "model (e.g. vosk-model-small-en-us-0.15) and point "
                    "voice.stt_model_path at it in config.yaml - see "
                    "voice/README.md."
                ),
            )

        try:
            vosk.SetLogLevel(-1)  # quiet vosk's own console logging
            self._model = vosk.Model(model_path)
        except Exception as e:  # noqa: BLE001 - surface any model load failure
            return STTStatus(available=False, reason=f"Failed to load vosk model: {e}")

        try:
            sd.check_input_settings(samplerate=self.sample_rate, channels=1)
        except Exception as e:  # noqa: BLE001 - e.g. no microphone present
            return STTStatus(available=False, reason=f"No usable microphone found: {e}")

        self._recognizer_cls = vosk.KaldiRecognizer
        self._sd = sd
        return STTStatus(available=True)

    @property
    def status(self) -> STTStatus:
        return self._status

    def is_available(self) -> bool:
        return self._status.available

    # ------------------------------------------------------------------
    # Toggle-style recording: start_recording() from one GUI click,
    # stop_recording() from the next. Chosen over push-to-talk because
    # Qt doesn't cleanly expose global key-hold state cross-platform
    # without extra OS-level permissions.
    # ------------------------------------------------------------------
    def start_recording(self) -> None:
        if not self.is_available():
            raise RuntimeError(self._status.reason or "STT engine unavailable")
        if self._recording.is_set():
            return  # already recording

        self._audio_q = queue.Queue()
        self._recording.set()

        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            if self._recording.is_set():
                self._audio_q.put(bytes(indata))

        self._stream = self._sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=_callback,
        )
        self._stream.start()

    def stop_recording(self, max_wait_s: float = 0.5) -> str:
        """Stops capture and returns the final transcribed text. Safe to
        call even if start_recording() was never called (returns "")."""
        if not self._recording.is_set():
            return ""
        self._recording.clear()

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        recognizer = self._recognizer_cls(self._model, self.sample_rate)
        while not self._audio_q.empty():
            chunk = self._audio_q.get()
            recognizer.AcceptWaveform(chunk)

        result = json.loads(recognizer.FinalResult())
        return result.get("text", "").strip()

    def is_recording(self) -> bool:
        return self._recording.is_set()
