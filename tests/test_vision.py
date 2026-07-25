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
from vision.ocr import (
    parse_score_text, parse_minute_text, parse_match_clock_text,
    parse_formation_text, stamina_fill_percent, read_single_number, read_minute,
)
from vision.capture import ScreenRegion, grab_region
from vision.calibrate import build_parser, capture_to_samples, selector_image_path, validate_regions_file
from vision.match_reader import self_test_samples

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


def test_single_digit_ocr_reads_real_image():
    """Real-Tesseract regression for PSM 7 + digits on a lone HUD digit."""
    from PIL import ImageFont

    image = Image.new("RGB", (60, 45), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except Exception:
        font = ImageFont.load_default()
    draw.text((15, 7), "5", fill=(255, 255, 255), font=font)

    assert read_single_number(image) == 5
    print("OK - OCR reads a lone digit with PSM 7 + digits")


def test_single_number_ocr_upscales_and_uses_digits_config_without_tesseract():
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
    assert "--psm 7" in fake.config
    assert "digits" in fake.config
    assert "tessedit_char_whitelist" not in fake.config
    print("OK - number OCR upscales 4x and uses PSM 7 with Tesseract's digits config")


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
    assert parse_minute_text("45+2") == 47
    assert parse_minute_text("999") is None  # out of sane bounds, rejected
    assert parse_minute_text("nothing here") is None
    assert parse_match_clock_text("HT") == (45, "halftime")
    assert parse_match_clock_text("FT") == (90, "fulltime")
    print("OK - clock parsing handles stoppage time, break states, and OCR quirks")


def test_parse_formation_label_variants():
    assert parse_formation_text("FORMATION 4 - 3 - 3") == "4-3-3"
    assert parse_formation_text("4-2-3-1") == "4-2-3-1"
    assert parse_formation_text("not a formation") is None
    print("OK - formation-label parsing accepts supported tactics-menu text")


def test_region_validation_warns_for_too_narrow_stat_crop(tmp_path=None):
    """Configuration-only guard: no screen or OCR engine is required."""
    import json
    import tempfile

    temp_dir = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    regions = temp_dir / "regions.json"
    regions.write_text(json.dumps({"regions": {"shots": {"left": 1, "top": 2, "width": 25, "height": 20}}, "team_stamina_bars": []}))
    warnings = validate_regions_file(regions)
    assert any("shots" in warning and "80x40" in warning for warning in warnings)
    print("OK - narrow stats crop triggers a calibration width warning")


def test_capture_arguments_and_selector_path_plumbing():
    """Argument parsing and capture-to-selector flow require no real display."""
    import tempfile
    from datetime import datetime

    parser = build_parser()
    select_args = parser.parse_args(["--select", "--capture"])
    sample_args = parser.parse_args(["--capture-sample", "stats.png"])
    assert select_args.select and select_args.capture and select_args.image is None
    assert sample_args.capture_sample == "stats.png"

    samples = Path(tempfile.mkdtemp())
    fake_capture = lambda: Image.new("RGB", (12, 8), "navy")
    captured = capture_to_samples(
        samples, capture_fn=fake_capture, now_fn=lambda: datetime(2026, 7, 26, 14, 30, 12)
    )
    assert captured == samples / "capture_20260726_143012.png"
    assert captured.exists()
    selector_path = selector_image_path(select_args, samples_dir=samples, capture_fn=fake_capture, now_fn=lambda: datetime(2026, 7, 26, 14, 30, 13))
    assert selector_path.name == "capture_20260726_143013.png"
    assert selector_path.exists()
    print("OK - capture arguments save a shared-capture image and pass it to selector flow")


def test_garbled_numeric_ocr_is_unknown_not_zero():
    """Mock realistic weak OCR text ('l7?\\n'), not an OCR engine result."""
    import json
    import tempfile
    import vision.capture as capture_module
    import vision.match_reader as reader_module
    from vision.ocr import NumberOCRResult

    temp_dir = Path(tempfile.mkdtemp())
    screenshot = temp_dir / "sample.png"
    Image.new("RGB", (100, 80), "black").save(screenshot)
    regions = temp_dir / "regions.json"
    regions.write_text(json.dumps({"regions": {"shots": {"left": 0, "top": 0, "width": 80, "height": 40}}, "team_stamina_bars": []}))
    original_regions, original_reader = capture_module.REGIONS_PATH, reader_module.read_single_number
    capture_module.REGIONS_PATH = regions
    reader_module.read_single_number = lambda _image, diagnostic=False: NumberOCRResult(None, "l7?\\n", 18.0)
    try:
        result = reader_module.read_partial_match_state(source_image=screenshot)
    finally:
        capture_module.REGIONS_PATH, reader_module.read_single_number = original_regions, original_reader

    assert "shots" not in result
    assert "l7?" in result["_read_errors"]["shots"]
    print("OK - garbled numeric OCR leaves shots unknown and logs raw text")


def test_saved_sample_self_test_summary():
    """Self-test runner is exercised with a saved image and injected reader."""
    import json
    import tempfile

    samples = Path(tempfile.mkdtemp())
    Image.new("RGB", (20, 20), "black").save(samples / "sample.png")
    (samples / "expectations.json").write_text(json.dumps({"sample.png": {"shots": 14}}))
    passed, failed = self_test_samples(samples, reader=lambda _path: {"shots": 14})
    assert (passed, failed) == (1, 0)
    print("OK - saved-sample self-test prints a passing field summary")


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

    # All number crops are 80px wide: enough for a three-digit percent plus
    # margin and comfortably wider than expected one/two-digit fouls.
    for left, value in [(45, "14"), (150, "88"), (250, "76"), (350, "3"), (430, "1")]:
        draw.rectangle([left, 25, left + 80, 70], fill=(0, 0, 0))
        draw.text((left + 15, 32), value, fill=(255, 255, 255), font=font)
    draw.rectangle([540, 25, 690, 70], fill=(0, 0, 0))
    draw.text((555, 32), "4-3-3", fill=(255, 255, 255), font=font)

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
            "shots": {"left": 45, "top": 25, "width": 80, "height": 45},
            "opponent_shots": None, "shots_on_target": None,
            "opponent_shots_on_target": None, "corners": None,
            "opponent_corners": None,
            "pass_accuracy_pct": {"left": 150, "top": 25, "width": 80, "height": 45},
            "opponent_pass_accuracy_pct": {"left": 250, "top": 25, "width": 80, "height": 45},
            "fouls_committed": {"left": 350, "top": 25, "width": 80, "height": 45},
            "opponent_fouls_committed": {"left": 430, "top": 25, "width": 80, "height": 45},
            "formation_label": {"left": 540, "top": 25, "width": 150, "height": 45},
            "my_yellow_cards": None, "opponent_yellow_cards": None,
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
    assert result["pass_accuracy_pct"] == 88, result
    assert result["opponent_pass_accuracy_pct"] == 76, result
    assert result["fouls_committed"] == 3, result
    assert result["opponent_fouls_committed"] == 1, result
    assert result["menu_formation"] == "4-3-3", result
    assert 48 <= result["team_stamina_avg_pct"] <= 58, result  # ~53% expected
    print("OK - match_reader reads stats, menu formation, and team stamina")


if __name__ == "__main__":
    test_score_ocr_reads_real_image()
    test_single_digit_ocr_reads_real_image()
    test_single_number_ocr_upscales_and_uses_digits_config_without_tesseract()
    test_single_number_ocr_preserves_leading_one()
    test_clock_ocr_reads_real_image()
    test_match_reader_reads_stats_and_team_stamina_average()
    test_parse_score_text_variants()
    test_parse_minute_text_variants()
    test_parse_formation_label_variants()
    test_region_validation_warns_for_too_narrow_stat_crop()
    test_capture_arguments_and_selector_path_plumbing()
    test_garbled_numeric_ocr_is_unknown_not_zero()
    test_saved_sample_self_test_summary()
    test_stamina_fill_percent_full_bar()
    test_stamina_fill_percent_half_bar()
    test_stamina_fill_percent_empty_bar()
    print("\nAll vision tests passed.")
