# Voice module (Phase 5)

Two independent pieces, both fully offline (no cloud API, no account,
same "runs entirely on your machine" principle as the Ollama-based
reasoning engine):

- **`tts.py`** - reads advice aloud via `pyttsx3` (wraps your OS's own
  speech engine - SAPI5 on Windows).
- **`stt.py`** - dictates the free-text "Notes" field via `vosk` (a
  small, fully local speech-recognition model) + `sounddevice` (mic
  capture).

Neither one controls the app or the game by voice command. TTS reads
the advice text back to you; STT only fills in the notes box, the way
typing into it would. You still click ASK COACH yourself. This keeps
the same "reads/speaks, never plays the game" boundary that vision/
already commits to.

## What was tested and how

`tests/test_voice.py` genuinely runs and passes in the sandbox this was
built in, but that sandbox has none of `pyttsx3`, `vosk`, `sounddevice`'s
system library (PortAudio), or a microphone - so what actually got
exercised is:

- `advice_to_speech_text()` - the pure function that turns an
  `AdviceResponse` into a short spoken sentence. Real logic, real test,
  no hardware involved.
- Both `TTSEngine` and `VoiceInputEngine` failing **soft**: constructing
  them without the underlying package/model/mic present returns a clear,
  actionable reason (`.status.reason`) instead of raising on import or
  crashing the GUI. That's a real, meaningful pass, not a mocked
  assumption - it's the literal state of the build environment.

What was **not** tested, because it can't be from here: actually
speaking audio out loud, and actually transcribing real speech. That's
the one gap only you can close, the same way live FIFA capture was for
vision/ - calibrate/verify it on your own machine.

## Setup

### Text-to-speech (`pip install pyttsx3`)

- **Windows**: works out of the box (uses SAPI5).
- **Linux**: also needs `espeak` as a system package:
  `sudo apt install espeak`.
- **macOS**: works out of the box (uses NSSpeechSynthesizer).

### Speech-to-text (`pip install vosk sounddevice`)

1. Install PortAudio, which `sounddevice` needs at the OS level:
   - Linux: `sudo apt install portaudio19-dev`
   - macOS: `brew install portaudio`
   - Windows: usually not needed, `sounddevice` ships a bundled binary.
2. Download a vosk model (pick ONE, unzip it):
   - Small/fast, good enough for short dictation: 
     [`vosk-model-small-en-us-0.15`](https://alphacephei.com/vosk/models)
     (~40MB)
   - Bigger/more accurate, if the small one mishears you a lot: any of
     the larger `vosk-model-en-us-*` models on the same page.
3. Unzip it to `voice/model/` in the project root (so you end up with
   `voice/model/am/`, `voice/model/conf/`, etc. directly inside it), or
   set a custom path in `config.yaml`:
   ```yaml
   voice_stt_model_path: "C:/path/to/your/vosk-model"
   ```

If either dependency is missing, the GUI doesn't crash - it just
disables the "Speak" / "Read aloud" / "Dictate" controls and shows why
in a tooltip when you hover over them.

## How dictation works

Click **Dictate** to start recording, talk, then click it again (now
labeled "Stop & transcribe") to stop - there's no automatic silence
detection, this is a manual toggle rather than push-to-talk, since Qt
doesn't cleanly expose global key-hold state across Windows/Linux/macOS
without extra OS permissions. Whatever gets transcribed is appended to
the Notes field, not sent anywhere else.

## Why vosk over other options

- Fully offline - no API key, no per-request cost, no audio leaving your
  machine, consistent with the rest of the project.
- Small enough (~40MB for the model used here) to be a reasonable ask
  compared to, say, a local Whisper model, which needs meaningfully more
  RAM/CPU than this laptop's spare capacity while Ollama is also running
  the reasoning model.
