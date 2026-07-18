# Vision module (Phase 4)

Screen capture + OCR/color-analysis that produces partial match data
(score, minute, stamina) from your actual screen, to reduce (not fully
replace) manual entry once FIFA 22 is installed.

**Requires Tesseract OCR installed on your system** (separate from the
`pytesseract` Python package, which is just a wrapper around it):
- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki,
  then make sure `tesseract.exe` is on your PATH.

## Why score/clock use OCR but stamina doesn't

- Score and clock are rendered TEXT - `ocr.py` uses Tesseract.
- Stamina is a colored BAR, not text - `ocr.py` estimates how much of the
  bar is "lit up" by brightness (see `stamina_fill_percent`), which is far
  more reliable than trying to OCR a progress bar.

Formation, opponent's threat side, and red cards are NOT read by vision -
those aren't reliably extractable from a couple of small screen crops, and
trying would mean guessing. `match_reader.py` only ever returns what it
actually read; merge that with what you enter manually (e.g. in the GUI).

Vision can also read match-stats-screen numbers (shots, shots on target,
corners, yellow cards - both teams) the same way it reads the score, and
can average stamina across MULTIPLE calibrated player bars (via
`team_stamina_bars` in `regions.json`, a list rather than a single region)
instead of relying on just one player's stamina.

## Calibration workflow (do this once, on your machine, after installing FIFA)

HUD position depends on your resolution and in-game HUD scale setting, so
region coordinates can't be hardcoded - you have to find them.

1. **Get a gridded screenshot** while FIFA is running (or paused):
   ```
   python vision/calibrate.py --grab
   ```
   This saves `calibration_grid.png` with red gridlines and coordinate
   labels every 100px. Open it and note the approximate pixel position of
   the score, clock, and stamina bars.

2. **Test a candidate region** against a real screenshot before trusting it:
   ```
   python vision/calibrate.py --test-region --image calibration_grid.png --left 820 --top 40 --width 60 --height 30 --kind score
   ```
   This saves a cropped PNG so you can visually confirm the crop is right,
   and prints what OCR/color-analysis actually reads from it. Adjust
   left/top/width/height and re-run until it reads correctly. Repeat for
   `clock` and `stamina`.

3. **Write the working coordinates into `regions.json`**, e.g.:
   ```json
   "my_score": {"left": 820, "top": 40, "width": 60, "height": 30}
   ```

   **Important:** make the region generously wide, not tightly cropped -
   a two-digit number (e.g. "14" shots) needs noticeably more width than
   a single digit, and a crop that's too narrow will silently clip a
   digit and read the wrong number. When in doubt, err wide - a bit of
   extra background around the text doesn't hurt OCR, but a clipped digit
   gives you a confidently wrong answer.

   For the added menu stats, start with **80 px** wide crops for each pass
   accuracy or fouls value (room for `100%`/two-digit fouls plus 15–20 px of
   margin), and **150 px** for the formation label (room for `3-4-2-1` plus
   its menu label margin). These are starting widths, not portable
   coordinates: calibrate them against your own screenshot before enabling the
   region in `regions.json`.

4. Once all regions are calibrated, `vision/match_reader.py` can read a
   partial match state from either a saved screenshot or the live screen.

## Files

- `capture.py` - grabs a named region, from the live screen (`mss`) or from
  a saved screenshot file (for calibration/testing).
- `ocr.py` - turns a cropped region into a number (score, minute) or a
  fill percentage (stamina bar).
- `calibrate.py` - CLI helper for the calibration workflow above.
- `match_reader.py` - ties capture + ocr together into a partial match-state
  dict, reading whatever regions are calibrated and skipping the rest.
- `regions.json` - your calibrated pixel coordinates. Starts empty.

## Testing without FIFA

`tests/test_vision.py` builds a synthetic HUD image with known values and
runs the real OCR/color-analysis pipeline against it - the same code path
used for a live screenshot, just pointed at a generated test image instead.
Run it with `python tests/test_vision.py`.

## OCR change verification is mandatory

The Arena development sandbox does not include the native `tesseract`
executable, so it cannot validate real recognition behavior. **Any change to
an OCR call in `vision/ocr.py`—including PSM, OEM, character constraints, or
preprocessing—must be verified by running the vision tests against a real
local Tesseract installation before it is considered fixed.**

This requirement follows the leading-`1` stats regression: a change from PSM
7 to PSM 8 was initially made without a native Tesseract run, then real
Tesseract 5.5 testing showed that a manually supplied LSTM
`-c tessedit_char_whitelist=...` call could suppress output entirely. The
module now upscales OCR crops and uses Tesseract's packaged `digits` config
with `--psm 7` for standalone numeric OCR, but that behavior still needs a
real local Tesseract test after every OCR-related edit.

**Use `--psm 7` together with Tesseract's built-in `digits` config for all
standalone numeric OCR in this project.** Do not use `digits` alone: it falls
back to full automatic page layout and returns “Empty page” for a small
single-digit crop. Do not substitute `--psm 8` (empty output for a lone digit)
or `--psm 10` (would truncate multi-digit values). The `--psm 7 digits`
combination was proven by the target Tesseract 5.5 CLI for both a lone digit
and a two-digit `14` crop.

**`tessedit_char_whitelist` should not be used for digit OCR in this project.**
Use the packaged `digits` config instead. The manual whitelist has produced
empty output on the target LSTM installation.

`read_score` and `read_minute` deliberately remain on their existing
unconstrained `--psm 7` path: score parsing needs the score separator (`-` or
spaces), while clock input can include `:`. Applying the digits-only config to
those differently shaped strings could remove the separator before their
format parsers run. Their Python parsers already validate the expected numeric
structure after OCR.

For diagnosis, set `BOSMAN_OCR_DEBUG_PATH` to a PNG path; `ocr.py` writes the
exact prepared (upscaled) image supplied to Tesseract there.
