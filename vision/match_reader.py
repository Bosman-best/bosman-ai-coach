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
from pathlib import Path
from typing import Optional

from vision.capture import grab_region, load_regions, load_team_stamina_bar_regions
from vision.ocr import read_single_number, read_minute, stamina_fill_percent


# Regions that are just "a single number" - same reading logic applies to
# all of them, so they're handled generically rather than one branch each.
_SIMPLE_NUMBER_REGIONS = [
    "shots", "opponent_shots",
    "shots_on_target", "opponent_shots_on_target",
    "corners", "opponent_corners",
    "my_yellow_cards", "opponent_yellow_cards",
]


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
            my_num = read_single_number(my_score_img)
            opp_num = read_single_number(opp_score_img)
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
            if minute is not None:
                result["minute"] = minute
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
            value = read_single_number(crop)
            if value is not None:
                result[field_name] = value
        except Exception as e:  # noqa: BLE001
            errors[field_name] = str(e)

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
