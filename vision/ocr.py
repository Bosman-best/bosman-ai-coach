"""
Turns cropped screen regions into actual data.

Two different techniques on purpose:
- Score and clock are TEXT -> OCR (pytesseract) is the right tool.
- Stamina bars are a GRAPHICAL fill level, not text -> OCR would be the
  wrong tool here (and unreliable). We measure how much of the bar is
  "lit up" by brightness, which is far cheaper and more robust than trying
  to run object detection on a progress bar.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional

from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _require_pytesseract() -> None:
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is not installed, or the Tesseract-OCR binary isn't on "
            "your PATH. Install the Python package with `pip install pytesseract` "
            "AND install Tesseract itself (see vision/README.md for the Windows "
            "installer link)."
        )


_OCR_SCALE = 4
# Tell Tesseract the intended working resolution as well as increasing pixels;
# otherwise it may still estimate a low DPI from a tiny HUD glyph.
_OCR_DPI_CONFIG = "--dpi 300"


def _image_for_ocr(image: Image.Image) -> Image.Image:
    """Prepare a crop for Tesseract without discarding any glyph pixels.

    HUD crops are often only about 60x45 pixels.  Resize every text/number crop
    with Lanczos interpolation before OCR so Tesseract does not have to infer
    character features from a low-resolution image.  There is deliberately no
    contour crop, whitespace trim, or thresholding here: those operations can
    discard a thin leading ``1``.
    """
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((image.width * _OCR_SCALE, image.height * _OCR_SCALE), resampling)


# Kept as a focused alias for callers/tests diagnosing single-number OCR.
def _number_image_for_ocr(image: Image.Image) -> Image.Image:
    return _image_for_ocr(image)


def _write_ocr_debug_image(image: Image.Image) -> None:
    """Optionally save the *post-preparation* OCR input for diagnosis.

    Set BOSMAN_OCR_DEBUG_PATH to a PNG path while reproducing an OCR issue.
    This is intentionally opt-in so normal live play does not write screenshots
    to disk.
    """
    debug_path = os.environ.get("BOSMAN_OCR_DEBUG_PATH")
    if debug_path:
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(debug_path, format="PNG")


def read_single_number(image: Image.Image) -> Optional[int]:
    """Read a lone number from a calibrated crop.

    PSM 8 models the crop as a single numeric word. Do not use
    ``tessedit_char_whitelist`` here: with the default LSTM OCR engine it can
    suppress all output for small HUD crops. Instead, parse digits from the
    unconstrained OCR result below.
    """
    _require_pytesseract()
    ocr_image = _number_image_for_ocr(image)
    _write_ocr_debug_image(ocr_image)
    raw = pytesseract.image_to_string(ocr_image, config=f"--psm 8 {_OCR_DPI_CONFIG}")
    match = re.search(r"\d+", raw)
    return int(match.group()) if match else None


def read_score(image: Image.Image) -> Optional[tuple[int, int]]:
    """Read a scoreline like '1 - 2' or '2-1' from a cropped image.
    Returns (my_score, opponent_score) or None if it couldn't be parsed."""
    _require_pytesseract()
    ocr_image = _image_for_ocr(image)
    _write_ocr_debug_image(ocr_image)
    # Post-parse the unconstrained result rather than using the LSTM-incompatible
    # character whitelist; parse_score_text is deliberately tolerant of spacing.
    raw = pytesseract.image_to_string(ocr_image, config=f"--psm 7 {_OCR_DPI_CONFIG}")
    return parse_score_text(raw)


def parse_score_text(raw: str) -> Optional[tuple[int, int]]:
    """Pure parsing logic, split out so it can be tested without OCR."""
    match = re.search(r"(\d+)\D+(\d+)", raw)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def read_minute(image: Image.Image) -> Optional[int]:
    """Read the match clock (e.g. '70:23' or "70'") and return just the
    minute as an int. Returns None if it couldn't be parsed."""
    _require_pytesseract()
    ocr_image = _image_for_ocr(image)
    _write_ocr_debug_image(ocr_image)
    # Do not pass a character whitelist: Tesseract's LSTM engine can return an
    # empty result for small crops when constrained this way. The parser keeps
    # only the leading digits after unconstrained recognition.
    raw = pytesseract.image_to_string(ocr_image, config=f"--psm 7 {_OCR_DPI_CONFIG}")
    return parse_minute_text(raw)


def parse_minute_text(raw: str) -> Optional[int]:
    """Pure parsing logic, split out so it can be tested without OCR."""
    match = re.match(r"\s*(\d+)", raw)
    if not match:
        return None
    minute = int(match.group(1))
    if 0 <= minute <= 130:  # sanity bound - allow generous extra time
        return minute
    return None


def stamina_fill_percent(image: Image.Image) -> Optional[int]:
    """Estimate how full a stamina/progress bar is, by brightness rather than
    OCR. Assumes the bar runs left-to-right, with a brighter "filled" color
    against a darker "empty/background" color.

    Uses an adaptive threshold (midpoint between the darkest and brightest
    column) rather than assuming any specific column is background - a
    fixed assumption like "the right edge is always empty" breaks down the
    moment the bar is fully filled, since there's no empty segment left to
    sample.
    """
    if np is None:
        raise RuntimeError("numpy is required for stamina bar reading (pip install numpy).")

    arr = np.array(image.convert("L"), dtype=float)  # grayscale, width x height
    col_means = arr.mean(axis=0)  # average brightness per column

    brightest, darkest = col_means.max(), col_means.min()

    # Whole crop is uniformly bright or uniformly dark (fully full or fully
    # empty bar) - there's no contrast to threshold against, so fall back to
    # an absolute brightness check instead.
    if brightest - darkest < 10:
        return 100 if col_means.mean() > 100 else 0

    threshold = (brightest + darkest) / 2
    filled_cols = int((col_means > threshold).sum())
    return round(100 * filled_cols / len(col_means))
