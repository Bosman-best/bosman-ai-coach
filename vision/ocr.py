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


def _number_image_for_ocr(image: Image.Image) -> Image.Image:
    """Return the exact number crop supplied to Tesseract.

    There is deliberately no contour crop, whitespace trim, or thresholding
    here. In particular, those operations can discard a thin leading ``1``.
    Keeping this helper makes that guarantee testable and gives one explicit
    point for inspecting the bytes passed to OCR.
    """
    return image.copy()


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

    PSM 8 (one word) is used rather than PSM 7 (one text line). Tesseract 5.5
    can segment a sparse leading ``1`` as margin noise in a short line and
    return only the following digit; a numeric word is the actual shape of a
    score/stat crop and retains the leading digit.
    """
    _require_pytesseract()
    ocr_image = _number_image_for_ocr(image)
    _write_ocr_debug_image(ocr_image)
    raw = pytesseract.image_to_string(
        ocr_image,
        config="--psm 8 -c tessedit_char_whitelist=0123456789",
    )
    match = re.search(r"\d+", raw)
    return int(match.group()) if match else None


def read_score(image: Image.Image) -> Optional[tuple[int, int]]:
    """Read a scoreline like '1 - 2' or '2-1' from a cropped image.
    Returns (my_score, opponent_score) or None if it couldn't be parsed."""
    _require_pytesseract()
    raw = pytesseract.image_to_string(
        image, config="--psm 7 -c tessedit_char_whitelist=0123456789-: "
    )
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
    # Whitelist digits and colon only - an apostrophe in the whitelist
    # string breaks pytesseract's shlex-based config parsing, and we don't
    # need it anyway since parse_minute_text only looks at the leading digits.
    raw = pytesseract.image_to_string(
        image, config="--psm 7 -c tessedit_char_whitelist=0123456789:"
    )
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
