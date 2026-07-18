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

    Use Tesseract's packaged ``digits`` config—not a hand-built
    ``tessedit_char_whitelist`` flag. On the target Tesseract 5.5 install,
    the former correctly recognizes thin leading digits while the latter can
    suppress all LSTM output. Pair it with PSM 7 (a single text line): this is
    verified for both lone and two-digit HUD values. ``digits`` alone falls
    back to automatic page layout and misses a lone small digit; PSM 8 also
    returns empty there, while PSM 10 would truncate multi-digit values.
    """
    _require_pytesseract()
    ocr_image = _number_image_for_ocr(image)
    _write_ocr_debug_image(ocr_image)
    raw = pytesseract.image_to_string(
        ocr_image, config=f"--psm 7 {_OCR_DPI_CONFIG} digits"
    )
    match = re.search(r"\d+", raw)
    return int(match.group()) if match else None


def _read_text_line(image: Image.Image) -> str:
    """Shared menu-text OCR path for clocks and formation labels.

    Numeric menu fields must use read_single_number(); this helper is only for
    genuinely textual values where punctuation/letters are meaningful.
    """
    _require_pytesseract()
    ocr_image = _image_for_ocr(image)
    _write_ocr_debug_image(ocr_image)
    return pytesseract.image_to_string(ocr_image, config=f"--psm 7 {_OCR_DPI_CONFIG}")


def read_score(image: Image.Image) -> Optional[tuple[int, int]]:
    """Read a scoreline like '1 - 2' or '2-1' from a cropped image."""
    return parse_score_text(_read_text_line(image))


def parse_score_text(raw: str) -> Optional[tuple[int, int]]:
    """Pure parsing logic, split out so it can be tested without OCR."""
    match = re.search(r"(\d+)\D+(\d+)", raw)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def read_minute(image: Image.Image) -> Optional[int]:
    """Read a clock and return elapsed minutes, including stoppage time."""
    return parse_match_clock_text(_read_text_line(image))[0]


def read_match_half(image: Image.Image) -> Optional[str]:
    """Read an explicit halftime/fulltime state, or infer the active half."""
    return parse_match_clock_text(_read_text_line(image))[1]


def parse_match_clock_text(raw: str) -> tuple[Optional[int], Optional[str]]:
    """Parse normal, stoppage-time, and break-state FIFA clock text.

    ``45+2`` becomes minute 47; ``HT`` and ``FT`` retain their meaningful
    state even though no running clock is present.
    """
    normalized = raw.strip().upper()
    if normalized in {"HT", "HALF TIME", "HALFTIME"}:
        return 45, "halftime"
    if normalized in {"FT", "FULL TIME", "FULLTIME"}:
        return 90, "fulltime"
    match = re.search(r"(\d{1,3})\s*(?:\+\s*(\d{1,2}))?", normalized)
    if not match:
        return None, None
    minute = int(match.group(1)) + int(match.group(2) or 0)
    if not 0 <= minute <= 130:
        return None, None
    return minute, "first_half" if minute <= 45 else "second_half"


def parse_minute_text(raw: str) -> Optional[int]:
    """Backward-compatible minute-only wrapper around clock parsing."""
    return parse_match_clock_text(raw)[0]


def parse_formation_text(raw: str) -> Optional[str]:
    """Extract one supported formation label from tactics/lineup menu text."""
    normalized = raw.replace("–", "-").replace("—", "-")
    match = re.search(r"\b([345](?:\s*-\s*[12345]){2,3})\b", normalized)
    if not match:
        return None
    formation = re.sub(r"\s*[-]\s*", "-", match.group(1))
    return formation if formation in {"4-4-2", "4-3-3", "3-5-2", "4-2-3-1", "5-3-2", "3-4-2-1"} else None


def read_formation_label(image: Image.Image) -> Optional[str]:
    """Read a formation label from a calibrated tactics/lineup menu crop."""
    return parse_formation_text(_read_text_line(image))


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
