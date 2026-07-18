"""
Tests for the vision module's PARSING logic (regex, pixel-fill math).
These don't need a real screen or FIFA - they test the pure functions with
synthetic images built with PIL.

NOTE: this does NOT test capture.py (mss) since that requires a real
display and can't be exercised in a sandboxed environment. That part can
only be verified on your actual machine - see vision/README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw

import vision.ocr as ocr_module
from vision.ocr import parse_score_text, parse_minute_text, stamina_fill_percent, read_single_number, read_minute
from vision.capture import ScreenRegion, grab_region

FIXTURE_PATH = Path(__file__).parent / "_fixture_hud.png"


def _make_hud_fixture() -> Path:
    """Build a synthetic HUD screenshot with known score/clock values at
    fixed coordinates, so OCR itself (not just the parsing regex) gets
    exercised - this is what catches things like the tesseract config
    crash on an unescaped apostrophe."""
    from PIL import ImageFont

    img = Image.new("RGB", (1000, 200), (20, 60, 20))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([800, 30, 860, 80], fill=(0, 0, 0))
    draw.text((815, 35), "2", fill=(255, 255, 255), font=font)
    draw.rectangle([870, 30, 930, 80], fill=(0, 0, 0))
    draw.text((885, 35), "1", fill=(255, 255, 255), font=font)
    draw.rectangle([460, 30, 540, 80], fill=(0, 0, 0))
    draw.text((470, 35), "73", fill=(255, 255, 255), font=font)

    img.save(FIXTURE_PATH)
    return FIXTURE_PATH


def test_score_ocr_reads_real_image():
    fixture = _make_hud_fixture()
    my_img = grab_region(ScreenRegion("my_score", 800, 30, 60, 50), source_image=fixture)
    opp_img = grab_region(ScreenRegion("opponent_score", 870, 30, 60, 50), source_image=fixture)
    assert read_single_number(my_img) == 2
    assert read_single_number(opp_img) == 1
    print("OK - OCR reads a real synthetic scoreboard image correctly (2-1)")


def test_single_number_ocr_upscales_and_post_filters_without_tesseract():
    """Exercise the non-Tesseract safeguards with a deterministic fake engine."""
    class FakeTesseract:
        def __init__(self):
            self.image = None
            self.config = None

        def image_to_string(self, image, config):
            self.image, self.config = image, config
            return "an17!"  # unconstrained LSTM-like noise around valid digits

    fake = FakeTesseract()
    original = ocr_module.pytesseract
    ocr_module.pytesseract = fake
    try:
        result = read_single_number(Image.new("RGB", (60, 45), (0, 0, 0)))
    finally:
        ocr_module.pytesseract = original

    assert result == 17
    assert fake.image.size == (240, 180)
    assert "tessedit_char_whitelist" not in fake.config
    assert "--psm 8" in fake.config
    print("OK - number OCR upscales 4x and filters digits after unconstrained OCR")


def test_single_number_ocr_preserves_leading_one():
    """Regression: Tesseract 5.5 PSM 7 dropped the leading thin ``1`` in 14."""
    from PIL import ImageFont

    image = Image.new("RGB", (60, 45), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except Exception:
        font = ImageFont.load_default()
    draw.text((15, 7), "17", fill=(255, 255, 255), font=font)

    assert read_single_number(image) == 17
    print("OK - OCR preserves a thin leading 1 in 17")


def test_clock_ocr_reads_real_image():
    fixture = _make_hud_fixture()
    clock_img = grab_region(ScreenRegion("clock", 460, 30, 80, 50), source_image=fixture)
    assert read_minute(clock_img) == 73
    print("OK - OCR reads a real synthetic clock image correctly (73)")
    fixture.unlink(missing_ok=True)


def test_parse_score_text_variants():
    assert parse_score_text("1-2") == (1, 2)
    assert parse_score_text("1 - 2") == (1, 2)
    assert parse_score_text("2  -  0") == (2, 0)
    assert parse_score_text("garbage") is None
    print("OK - score text parsing handles spacing variants and rejects garbage")


def test_parse_minute_text_variants():
    assert parse_minute_text("70:23") == 70
    assert parse_minute_text("70°") == 70  # apostrophe sometimes misread as degree sign
    assert parse_minute_text("  45  ") == 45
    assert parse_minute_text("999") is None  # out of sane bounds, rejected
    assert parse_minute_text("nothing here") is None
    print("OK - minute text parsing handles OCR quirks and rejects nonsense")


def _make_stamina_bar_image(fill_fraction: float, width: int = 200, height: int = 20) -> Image.Image:
    """Build a synthetic stamina bar: bright 'filled' portion on the left,
    dark background on the right - mimicking a typical game HUD bar."""
    img = Image.new("L", (width, height), color=20)  # dark background
    d = ImageDraw.Draw(img)
    filled_width = int(width * fill_fraction)
    d.rectangle([0, 0, filled_width, height], fill=200)  # bright filled segment
    return img.convert("RGB")


def test_stamina_fill_percent_full_bar():
    img = _make_stamina_bar_image(1.0)
    pct = stamina_fill_percent(img)
    assert 90 <= pct <= 100, f"expected ~100%, got {pct}"
    print(f"OK - full stamina bar reads as {pct}%")


def test_stamina_fill_percent_half_bar():
    img = _make_stamina_bar_image(0.5)
    pct = stamina_fill_percent(img)
    assert 40 <= pct <= 60, f"expected ~50%, got {pct}"
    print(f"OK - half-full stamina bar reads as {pct}%")


def test_stamina_fill_percent_empty_bar():
    img = _make_stamina_bar_image(0.05)
    pct = stamina_fill_percent(img)
    assert pct <= 15, f"expected ~0-15%, got {pct}"
    print(f"OK - near-empty stamina bar reads as {pct}%")


def test_match_reader_reads_stats_and_team_stamina_average():
    """End-to-end: a synthetic screenshot with a shots stat and 3 team
    stamina bars (80%, 50%, 30%) should produce shots=14 and an averaged
    team_stamina_avg_pct, using the full capture -> ocr/color pipeline."""
    import json
    import vision.capture as capture_module
    from vision.match_reader import read_partial_match_state

    img = Image.new("RGB", (1000, 700), (20, 60, 20))
    draw = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([45, 25, 105, 70], fill=(0, 0, 0))
    draw.text((60, 32), "14", fill=(255, 255, 255), font=font)

    for i, frac in enumerate([0.8, 0.5, 0.3]):
        bar_left, bar_top, bar_w, bar_h = 50, 400 + i * 40, 150, 15
        draw.rectangle([bar_left, bar_top, bar_left + bar_w, bar_top + bar_h], fill=(30, 30, 30))
        fill_w = int(bar_w * frac)
        draw.rectangle([bar_left, bar_top, bar_left + fill_w, bar_top + bar_h], fill=(220, 220, 60))

    fixture = Path(__file__).parent / "_fixture_stats.png"
    img.save(fixture)

    test_regions_path = Path(__file__).parent / "_test_regions_temp.json"
    test_regions_path.write_text(json.dumps({
        "regions": {
            "my_score": None, "opponent_score": None, "clock": None,
            "striker_stamina_bar": None, "key_player_stamina_bar": None,
            "shots": {"left": 45, "top": 25, "width": 60, "height": 45},
            "opponent_shots": None, "shots_on_target": None,
            "opponent_shots_on_target": None, "corners": None,
            "opponent_corners": None, "my_yellow_cards": None,
            "opponent_yellow_cards": None,
        },
        "team_stamina_bars": [
            {"left": 50, "top": 400, "width": 150, "height": 15},
            {"left": 50, "top": 440, "width": 150, "height": 15},
            {"left": 50, "top": 480, "width": 150, "height": 15},
        ],
    }))

    original_path = capture_module.REGIONS_PATH
    capture_module.REGIONS_PATH = test_regions_path
    try:
        result = read_partial_match_state(source_image=fixture)
    finally:
        capture_module.REGIONS_PATH = original_path
        fixture.unlink(missing_ok=True)
        test_regions_path.unlink(missing_ok=True)

    assert result["shots"] == 14, result
    assert 48 <= result["team_stamina_avg_pct"] <= 58, result  # ~53% expected
    print("OK - match_reader reads stats and averages team stamina across multiple bars")


if __name__ == "__main__":
    test_score_ocr_reads_real_image()
    test_single_number_ocr_upscales_and_post_filters_without_tesseract()
    test_single_number_ocr_preserves_leading_one()
    test_clock_ocr_reads_real_image()
    test_match_reader_reads_stats_and_team_stamina_average()
    test_parse_score_text_variants()
    test_parse_minute_text_variants()
    test_stamina_fill_percent_full_bar()
    test_stamina_fill_percent_half_bar()
    test_stamina_fill_percent_empty_bar()
    print("\nAll vision tests passed.")
