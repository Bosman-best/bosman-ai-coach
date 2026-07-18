"""
Calibration helper for Phase 4.

FIFA's HUD position depends on your resolution and in-game HUD scale
setting, so region coordinates can't be hardcoded - you have to find them
on your own screen. This script has two modes:

1. Capture a full screenshot with a coordinate grid overlaid, so you can
   open it in an image viewer and read off pixel coordinates for the score,
   clock, and stamina bars.

    python vision/calibrate.py --grab

2. Test a candidate region against a saved screenshot, to check the crop
   is right and OCR can actually read it, BEFORE writing it into
   regions.json and relying on it during a real match.

    python vision/calibrate.py --test-region \\
        --image screenshot.png --left 820 --top 40 --width 60 --height 30 --kind score
"""

from __future__ import annotations
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from vision.capture import ScreenRegion, grab_region, grab_full_screen
from vision.ocr import read_single_number, read_minute, stamina_fill_percent


def grab_calibration_screenshot(out_path: Path, grid_spacing: int = 100) -> Path:
    """Grab the full screen and draw a coordinate grid on top, so you can
    read off approximate pixel positions in any image viewer."""
    img = grab_full_screen()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for x in range(0, w, grid_spacing):
        draw.line([(x, 0), (x, h)], fill=(255, 0, 0), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 0, 0))
    for y in range(0, h, grid_spacing):
        draw.line([(0, y), (w, y)], fill=(255, 0, 0), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 0, 0))

    img.save(out_path)
    return out_path


def test_region(image_path: Path, left: int, top: int, width: int, height: int, kind: str) -> None:
    """Crop the given region out of a saved screenshot, save the crop so you
    can visually check it, and run the appropriate reader against it."""
    region = ScreenRegion(name="test", left=left, top=top, width=width, height=height)
    crop = grab_region(region, source_image=image_path)

    crop_path = image_path.parent / f"{image_path.stem}_crop_test.png"
    crop.save(crop_path)
    print(f"Saved crop to {crop_path} - open it and check it actually shows what you expect.")

    if kind == "score":
        value = read_single_number(crop)
        print(f"OCR read: {value}")
    elif kind == "clock":
        value = read_minute(crop)
        print(f"OCR read minute: {value}")
    elif kind == "stamina":
        value = stamina_fill_percent(crop)
        print(f"Estimated stamina fill: {value}%")
    else:
        print(f"Unknown kind '{kind}' - use score, clock, or stamina.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bosman AI Coach - region calibration helper")
    parser.add_argument("--grab", action="store_true", help="Capture a gridded full-screen screenshot")
    parser.add_argument("--out", type=Path, default=Path("calibration_grid.png"))

    parser.add_argument("--test-region", action="store_true", help="Test a candidate region")
    parser.add_argument("--image", type=Path, help="Saved screenshot to test against")
    parser.add_argument("--left", type=int)
    parser.add_argument("--top", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--kind", choices=["score", "clock", "stamina"], help="What to read from the region")

    args = parser.parse_args()

    if args.grab:
        path = grab_calibration_screenshot(args.out)
        print(f"Saved gridded screenshot to {path}")
        print("Open it in an image viewer, find the score/clock/stamina bar, and read off pixel coordinates.")
        return

    if args.test_region:
        if not all([args.image, args.left is not None, args.top is not None,
                    args.width, args.height, args.kind]):
            parser.error("--test-region requires --image --left --top --width --height --kind")
        test_region(args.image, args.left, args.top, args.width, args.height, args.kind)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
