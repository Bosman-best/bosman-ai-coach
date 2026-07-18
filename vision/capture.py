"""
Screen capture, scoped to small fixed regions rather than the whole screen.

This deliberately does NOT do object detection or full-frame analysis -
per the architecture decision, we only read small cropped areas (score,
clock, stamina bars) where we already know roughly where the UI element is
thanks to calibration. That keeps this fast enough to run on integrated
graphics with no GPU acceleration.

mss requires an active display - this module can only be exercised on your
actual machine while FIFA is running, not in a sandboxed/headless environment.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image

try:
    import mss
except ImportError:  # pragma: no cover - mss isn't installable in a headless sandbox
    mss = None


@dataclass
class ScreenRegion:
    """A named, calibrated rectangle on screen, in pixel coordinates."""
    name: str
    left: int
    top: int
    width: int
    height: int

    def as_mss_dict(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def as_pil_box(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


def grab_region(region: ScreenRegion, source_image: Optional[Path] = None) -> Image.Image:
    """Capture a single calibrated region and return it as a PIL Image.

    If source_image is given, crops from that saved screenshot file instead
    of the live screen. This is the key to testing and calibrating this
    whole pipeline WITHOUT FIFA running - take one screenshot manually
    (PrintScreen / Snipping Tool), then test every region against it.
    """
    if source_image is not None:
        with Image.open(source_image) as img:
            return img.crop(region.as_pil_box()).copy()

    if mss is None:
        raise RuntimeError(
            "mss is not available. Install it with `pip install mss` and run "
            "this on your actual machine (not a headless environment)."
        )
    with mss.mss() as sct:
        shot = sct.grab(region.as_mss_dict())
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def grab_full_screen(monitor_index: int = 1) -> Image.Image:
    """Capture the whole screen - used only during calibration, never during
    the regular read loop."""
    if mss is None:
        raise RuntimeError(
            "mss is not available. Install it with `pip install mss` and run "
            "this on your actual machine (not a headless environment)."
        )
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


REGIONS_PATH = Path(__file__).parent / "regions.json"


def load_regions() -> dict[str, Optional[ScreenRegion]]:
    """Load calibrated regions from regions.json. Regions that haven't been
    calibrated yet (still null in the file) come back as None, so callers
    fail with a clear message instead of silently grabbing the wrong area."""
    import json

    with open(REGIONS_PATH) as f:
        raw = json.load(f)["regions"]

    result: dict[str, Optional[ScreenRegion]] = {}
    for name, coords in raw.items():
        if coords is None:
            result[name] = None
        else:
            result[name] = ScreenRegion(name=name, **coords)
    return result


def load_team_stamina_bar_regions() -> list[ScreenRegion]:
    """Load the list of individually-calibrated team stamina bar regions
    (e.g. from the team management screen listing all 11 players). Empty
    list if none have been calibrated yet."""
    import json

    with open(REGIONS_PATH) as f:
        raw = json.load(f).get("team_stamina_bars", [])

    return [ScreenRegion(name=f"team_stamina_{i}", **coords) for i, coords in enumerate(raw)]
