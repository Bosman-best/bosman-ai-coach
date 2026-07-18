"""
Reads what it can from the screen (or a saved screenshot) and returns a
partial match-state dict - only the fields vision can actually see
(score, minute, stamina bars). Everything vision can't reliably determine
(formation, opponent's threat side, red cards, notes) is left out, and
the caller is expected to merge this with manually-entered context, e.g.
through the GUI form.

This deliberately does NOT try to output a full core.schemas.MatchState by
itself - vision only ever gives you a partial picture of the game, and
pretending otherwise would mean silently guessing at fields it has no way
of actually knowing.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Callable, Optional

from vision.capture import grab_region, load_regions, load_team_stamina_bar_regions
from vision.ocr import (
    NumberOCRResult, read_single_number, read_minute, read_match_half,
    read_formation_label, stamina_fill_percent,
)


# Regions that are just "a single number" - same reading logic applies to
# all of them, so they're handled generically rather than one branch each.
_SIMPLE_NUMBER_REGIONS = [
    "shots", "opponent_shots",
    "shots_on_target", "opponent_shots_on_target",
    "corners", "opponent_corners",
    "pass_accuracy_pct", "opponent_pass_accuracy_pct",
    "fouls_committed", "opponent_fouls_committed",
    "my_yellow_cards", "opponent_yellow_cards",
]

# Tesseract reports 0-100 confidences. 60 accepts clear menu digits while
# refusing marginal/fragmentary crops; it is intentionally conservative since
# a missing field becomes "unknown" downstream rather than a false fact.
MIN_NUMERIC_OCR_CONFIDENCE = 60.0
_FIELD_MAXIMUMS = {
    "my_score": 20, "opponent_score": 20,
    "pass_accuracy_pct": 100, "opponent_pass_accuracy_pct": 100,
    "my_yellow_cards": 5, "opponent_yellow_cards": 5,
    "fouls_committed": 30, "opponent_fouls_committed": 30,
    "shots": 99, "opponent_shots": 99,
    "shots_on_target": 99, "opponent_shots_on_target": 99,
    "corners": 30, "opponent_corners": 30,
}


def _accept_number(field_name: str, reading: NumberOCRResult, errors: dict[str, str]) -> Optional[int]:
    """Accept only a parseable, confident, in-range menu number."""
    if reading.value is None:
        errors[field_name] = f"numeric OCR empty/garbled; raw={reading.raw_text!r}"
        return None
    if reading.confidence is None:
        errors[field_name] = f"numeric OCR confidence unavailable; raw={reading.raw_text!r}"
        return None
    if reading.confidence < MIN_NUMERIC_OCR_CONFIDENCE:
        errors[field_name] = (
            f"numeric OCR confidence {reading.confidence:.1f} below {MIN_NUMERIC_OCR_CONFIDENCE:.0f}; "
            f"raw={reading.raw_text!r}"
        )
        return None
    maximum = _FIELD_MAXIMUMS.get(field_name)
    if reading.value < 0 or (maximum is not None and reading.value > maximum):
        errors[field_name] = f"numeric OCR value {reading.value} outside expected range; raw={reading.raw_text!r}"
        return None
    return reading.value


def read_partial_match_state(source_image: Optional[Path] = None) -> dict:
    """Read whatever HUD elements are calibrated and readable.

    Pass source_image to read from a saved screenshot file instead of the
    live screen (used for calibration testing, and for testing this code
    at all without FIFA running).

    Returns a dict with only the keys it successfully read - e.g. if the
    clock region isn't calibrated yet, "minute" simply won't be a key in
    the result. Never raises just because one region failed; each region
    is attempted independently so a bad stamina-bar crop doesn't take out
    the score reading too.
    """
    regions = load_regions()
    result: dict = {}
    errors: dict[str, str] = {}

    if regions.get("my_score") and regions.get("opponent_score"):
        try:
            my_score_img = grab_region(regions["my_score"], source_image)
            opp_score_img = grab_region(regions["opponent_score"], source_image)
            my_num = _accept_number("my_score", read_single_number(my_score_img, diagnostic=True), errors)
            opp_num = _accept_number("opponent_score", read_single_number(opp_score_img, diagnostic=True), errors)
            if my_num is not None:
                result["my_score"] = my_num
            if opp_num is not None:
                result["opponent_score"] = opp_num
        except Exception as e:  # noqa: BLE001
            errors["score"] = str(e)

    if regions.get("clock"):
        try:
            clock_img = grab_region(regions["clock"], source_image)
            minute = read_minute(clock_img)
            match_half = read_match_half(clock_img)
            if minute is not None:
                result["minute"] = minute
            if match_half is not None:
                result["match_half"] = match_half
        except Exception as e:  # noqa: BLE001
            errors["clock"] = str(e)

    if regions.get("striker_stamina_bar"):
        try:
            bar_img = grab_region(regions["striker_stamina_bar"], source_image)
            result["striker_stamina_pct"] = stamina_fill_percent(bar_img)
        except Exception as e:  # noqa: BLE001
            errors["striker_stamina"] = str(e)

    if regions.get("key_player_stamina_bar"):
        try:
            bar_img = grab_region(regions["key_player_stamina_bar"], source_image)
            result["key_player_stamina_pct"] = stamina_fill_percent(bar_img)
        except Exception as e:  # noqa: BLE001
            errors["key_player_stamina"] = str(e)

    # Match-stats-screen numbers (shots, corners, cards) - all read the
    # same way, so one loop instead of eight near-identical blocks.
    for field_name in _SIMPLE_NUMBER_REGIONS:
        region = regions.get(field_name)
        if not region:
            continue
        try:
            crop = grab_region(region, source_image)
            value = _accept_number(field_name, read_single_number(crop, diagnostic=True), errors)
            if value is not None:
                result[field_name] = value
        except Exception as e:  # noqa: BLE001
            errors[field_name] = str(e)

    # Formation is text from the tactics/lineup menu, not inferred from player
    # positions. Keep it distinct from the player's manually selected formation.
    if regions.get("formation_label"):
        try:
            formation_img = grab_region(regions["formation_label"], source_image)
            formation = read_formation_label(formation_img)
            if formation is not None:
                result["menu_formation"] = formation
        except Exception as e:  # noqa: BLE001
            errors["formation_label"] = str(e)

    # Team-wide stamina average - read every calibrated player bar and
    # average the fill percentages, rather than relying on one player.
    team_bars = load_team_stamina_bar_regions()
    if team_bars:
        readings = []
        for bar_region in team_bars:
            try:
                bar_img = grab_region(bar_region, source_image)
                pct = stamina_fill_percent(bar_img)
                if pct is not None:
                    readings.append(pct)
            except Exception as e:  # noqa: BLE001
                errors[bar_region.name] = str(e)
        if readings:
            result["team_stamina_avg_pct"] = round(sum(readings) / len(readings))

    if errors:
        result["_read_errors"] = errors

    return result


def self_test_samples(
    samples_dir: Path,
    expectations_path: Optional[Path] = None,
    reader: Callable[[Optional[Path]], dict] = read_partial_match_state,
) -> tuple[int, int]:
    """Run a saved-screen calibration regression set and print field results.

    ``expectations.json`` defaults to a mapping of screenshot filename to the
    fields expected from it, for example ``{"match_01.png": {"shots": 14}}``.
    Passing ``reader`` makes the summary logic testable without Tesseract.
    """
    expectations_path = expectations_path or samples_dir / "expectations.json"
    expectations = json.loads(expectations_path.read_text())
    screenshots = sorted(
        path for path in samples_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    passed = failed = 0
    for screenshot in screenshots:
        expected_fields = expectations.get(screenshot.name)
        if expected_fields is None:
            print(f"SKIP {screenshot.name}: no expectations entry")
            continue
        actual = reader(screenshot)
        for field_name, expected in expected_fields.items():
            actual_value = actual.get(field_name)
            if actual_value == expected:
                passed += 1
                print(f"PASS {screenshot.name} {field_name}: {actual_value!r}")
            else:
                failed += 1
                print(f"FAIL {screenshot.name} {field_name}: expected {expected!r}, got {actual_value!r}")
    print(f"Self-test summary: {passed} passed, {failed} failed")
    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bosman saved-screen OCR self-tests")
    parser.add_argument("--self-test", type=Path, metavar="SAMPLES_DIR", help="Folder containing screenshots and expectations.json")
    parser.add_argument("--expectations", type=Path, help="Optional expectations JSON path")
    args = parser.parse_args()
    if args.self_test:
        _passed, failed = self_test_samples(args.self_test, args.expectations)
        raise SystemExit(1 if failed else 0)
    parser.print_help()


if __name__ == "__main__":
    main()
