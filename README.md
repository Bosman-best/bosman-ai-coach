# Bosman AI Coach

A local, offline FIFA AI coach assistant. Runs entirely on your machine via
Ollama - no cloud APIs, no FIFA modification.

## Status: Phase 5 - voice (speak advice, dictate notes)

Phases 1-4 are done and tuned. Phase 5 adds an optional voice module:
advice can be read aloud (`pyttsx3`, offline TTS), and the Notes field
can be dictated instead of typed (`vosk` + `sounddevice`, offline STT).
Neither one controls the app or the game - TTS reads advice back to
you, STT only fills in a text box. If you don't install the voice
dependencies, everything else still works exactly as before; the
Speak/Read-aloud/Dictate controls just disable themselves with a
tooltip explaining why. See `voice/README.md` for setup (both need an
extra system-level install, and STT also needs a downloaded model file
- neither is bundled).

Phase 4 (vision) reads score, match clock, stats-screen numbers, and
team stamina from a screenshot or the live screen - but it does NOT
read formation, opponent threat side, or red cards (those aren't
reliably extractable from small screen crops). You'll still enter those
manually in the GUI; vision fills in what it can.

**Before vision works against real FIFA, you must calibrate it on your
own machine** - see `vision/README.md` for the full calibration
workflow. Region coordinates depend on your resolution and HUD
settings, so nothing here is pre-configured to "just work" against your
screen.

Run the GUI with:

```bash
python run_gui.py
```

Pick a simulated scenario from the dropdown to auto-fill the form, or enter
a match situation manually, then click ASK COACH. The request runs on a
background thread so the window won't freeze while Ollama thinks - watch
the status line under the button.

The CLI (`python main.py`) still works exactly as before, useful for quick
scripted testing without opening the GUI.

## Setup

```bash
pip install -r requirements.txt
```

For Phase 4 (vision), you also need Tesseract OCR itself installed
(separate from the `pytesseract` Python package) - see `vision/README.md`
for the Windows installer link.

For Phase 5 (voice, optional), see `voice/README.md` - TTS needs
`espeak` on Linux, STT needs PortAudio plus a downloaded vosk model.
The app runs fine without either; voice controls just disable
themselves if the dependencies aren't there.

Make sure Ollama is running and the model is pulled:

```bash
ollama serve
ollama pull qwen2.5:3b
```

## Run

```bash
python main.py              # interactive scenario picker
python main.py --list        # just list scenario names
python main.py --scenario "Losing late, opponent attacking down the left"
```

## Run the tests (no Ollama server needed - uses a mocked client)

```bash
python tests/test_core.py
python tests/test_vision.py
python tests/test_voice.py
```

## Project structure

```
core/     match_state schema, Ollama client, reasoning engine - the "brain"
data/     simulated match scenarios (edit/add your own here)
gui/      Phase 2 - PySide6 app
vision/   Phase 4 - screen capture + OCR
voice/    Phase 5 - speak advice aloud (TTS) + dictate notes (STT), optional
tests/    sanity tests for core/, vision/, and voice/ (mocked/synthetic where real hardware isn't available)
```

## Why the architecture is shaped this way

Every module downstream of `core/schemas.py` only ever sees a `MatchState`
in and an `AdviceResponse` out. That means:

- Swapping manual input -> simulated scenarios -> (eventually) OCR from FIFA
  never touches the reasoning engine or GUI.
- The GUI (Phase 2) is a thin renderer, not where tactical logic lives.
- Vision (Phase 4) and voice (Phase 5) are optional plug-ins you can build
  later, or never, without anything else needing to change.

## Model notes for this hardware (i7-8650U, integrated graphics, 16GB RAM)

Ollama runs CPU-only here (no CUDA). `qwen2.5:3b` is the recommended model
for live coaching - fast enough for a between-play "ask coach" interaction.
`qwen2.5-coder:7b` stays reserved for writing code in your editor; it's
noticeably slower per-token and isn't needed for tactical advice generation.

Advice is requested as strict JSON (`format: "json"` in the Ollama request)
so it can be validated against `AdviceResponse` directly - no fragile
free-text parsing.

## Live-play reasoning rules

The prompt builder treats the match snapshot as **read-only**: it only turns
entered `MatchState` values into advice and never controls, modifies, or seeks
additional input from the game. Its output is deliberately compact: one
prioritized `top_suggestion`, followed by at most two
`secondary_considerations`.

It reasons about related signals rather than reacting to each number alone:

- At least 8 shots with 35% or fewer on target is a finishing/shot-selection
  signal. The coach should favor composure, personnel, or better shot choices,
  not tactics intended simply to generate more shots.
- Three or fewer entered shots is a chance-creation signal. The coach should
  favor buildup, width, movement, or tempo changes.
- A falling possession trend combined with team-average stamina of 55% or less
  is an energy/control risk: conserve energy, use a compact shape and fresh
  legs, rather than pressing higher.
- Two or more yellow cards creates live card risk, so pressing and hard-tackle
  recommendations must be tempered in favor of controlled pressure.

These thresholds are only applied when every value needed for that conclusion
is entered. Every optional missing field is rendered as `unknown` in the
prompt—not as zero—and the model is told not to make a confident
stat-dependent recommendation from it. Fixed snapshot regression tests in
`tests/test_core.py` cover these rules.

## Adding your own scenarios

Edit `data/simulated_scenarios.json`. Each entry needs a `name` and a
`state` object matching `core/schemas.py::MatchState`.
