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
import json
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

# Support the documented `python vision/calibrate.py ...` invocation as well
# as `python -m vision.calibrate ...`.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.capture import ScreenRegion, grab_region, grab_full_screen
from vision.ocr import read_single_number, read_minute, stamina_fill_percent


# Conservative minimum crop sizes in source pixels. They deliberately include
# a margin around the widest expected value so a calibration cannot recreate a
# clipped leading/trailing digit bug. They are warnings, not resolution-agnostic
# guarantees: users must still check real FIFA screenshots.
REGION_MINIMUMS: dict[str, tuple[int, int]] = {
    "my_score": (50, 30), "opponent_score": (50, 30), "clock": (80, 30),
    "shots": (80, 40), "opponent_shots": (80, 40),
    "shots_on_target": (80, 40), "opponent_shots_on_target": (80, 40),
    "corners": (80, 40), "opponent_corners": (80, 40),
    "pass_accuracy_pct": (80, 40), "opponent_pass_accuracy_pct": (80, 40),
    "fouls_committed": (80, 40), "opponent_fouls_committed": (80, 40),
    "my_yellow_cards": (60, 40), "opponent_yellow_cards": (60, 40),
    "formation_label": (150, 40),
    "striker_stamina_bar": (80, 10), "key_player_stamina_bar": (80, 10),
}


def validate_regions_file(regions_path: Path) -> list[str]:
    """Return loud calibration warnings without needing a screenshot/OCR."""
    raw = json.loads(regions_path.read_text())
    warnings: list[str] = []
    for name, coords in raw.get("regions", {}).items():
        if coords is None:
            continue
        minimum = REGION_MINIMUMS.get(name)
        if minimum is None:
            continue
        width, height = coords.get("width", 0), coords.get("height", 0)
        if width < minimum[0] or height < minimum[1]:
            warnings.append(
                f"WARNING {name}: {width}x{height} is tighter than the recommended "
                f"minimum {minimum[0]}x{minimum[1]}; OCR may clip characters."
            )
    for index, coords in enumerate(raw.get("team_stamina_bars", [])):
        width, height = coords.get("width", 0), coords.get("height", 0)
        if width < 80 or height < 10:
            warnings.append(
                f"WARNING team_stamina_bars[{index}]: {width}x{height} is tighter than minimum 80x10."
            )
    return warnings


def _save_region(regions_path: Path, region_name: str, coords: dict[str, int]) -> None:
    raw = json.loads(regions_path.read_text())
    if region_name.startswith("team_stamina_"):
        index = int(region_name.rsplit("_", 1)[1])
        bars = raw.setdefault("team_stamina_bars", [])
        while len(bars) <= index:
            bars.append(None)
        bars[index] = coords
    else:
        raw.setdefault("regions", {})[region_name] = coords
    regions_path.write_text(json.dumps(raw, indent=2) + "\n")


class RegionSelector:
    """Small standalone Tk calibration UI: select a field, then drag its crop."""

    def __init__(self, image_path: Path, regions_path: Path):
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk

        self.tk = tk
        self.regions_path = regions_path
        self.image = Image.open(image_path).convert("RGB")
        self.photo = ImageTk.PhotoImage(self.image)
        self.start: Optional[tuple[int, int]] = None
        self.rectangle: Optional[int] = None
        raw = json.loads(regions_path.read_text())
        self.names = list(raw.get("regions", {}).keys())
        self.names.extend(f"team_stamina_{i}" for i, _bar in enumerate(raw.get("team_stamina_bars", [])))

        self.root = tk.Tk()
        self.root.title("Bosman region calibration — select a field, then drag")
        frame = ttk.Frame(self.root, padding=8)
        frame.grid(sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.listbox = tk.Listbox(frame, exportselection=False, width=28)
        for name in self.names:
            self.listbox.insert(tk.END, name)
        self.listbox.selection_set(0)
        self.listbox.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.canvas = tk.Canvas(frame, width=min(self.image.width, 1100), height=min(self.image.height, 750), scrollregion=(0, 0, self.image.width, self.image.height))
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.grid(row=0, column=1, sticky="nsew")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        vertical.grid(row=0, column=2, sticky="ns")
        horizontal.grid(row=1, column=1, sticky="ew")
        self.canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        ttk.Button(frame, text="Add team stamina bar", command=self._add_bar).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(frame, text="Drag a rectangle. Each completed drag saves regions.json immediately.").grid(row=2, column=1, sticky="w", pady=(8, 0))

    def _point(self, event) -> tuple[int, int]:
        return int(self.canvas.canvasx(event.x)), int(self.canvas.canvasy(event.y))

    def _press(self, event) -> None:
        self.start = self._point(event)
        if self.rectangle is not None:
            self.canvas.delete(self.rectangle)
        x, y = self.start
        self.rectangle = self.canvas.create_rectangle(x, y, x, y, outline="#00e5ff", width=2)

    def _drag(self, event) -> None:
        if self.start is not None and self.rectangle is not None:
            self.canvas.coords(self.rectangle, *self.start, *self._point(event))

    def _release(self, event) -> None:
        if self.start is None:
            return
        end_x, end_y = self._point(event)
        left, right = sorted((self.start[0], end_x))
        top, bottom = sorted((self.start[1], end_y))
        if right > left and bottom > top:
            name = self.names[self.listbox.curselection()[0]]
            _save_region(self.regions_path, name, {"left": left, "top": top, "width": right - left, "height": bottom - top})
            print(f"Saved {name}: left={left}, top={top}, width={right-left}, height={bottom-top}")
        self.start = None

    def _add_bar(self) -> None:
        name = f"team_stamina_{sum(item.startswith('team_stamina_') for item in self.names)}"
        self.names.append(name)
        self.listbox.insert(self.tk.END, name)
        self.listbox.selection_clear(0, self.tk.END)
        self.listbox.selection_set(self.listbox.size() - 1)

    def run(self) -> None:
        self.root.mainloop()


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
    parser.add_argument("--select", action="store_true", help="Open click-and-drag region selector for a saved screenshot")
    parser.add_argument("--regions", type=Path, default=Path(__file__).parent / "regions.json")
    parser.add_argument("--validate", action="store_true", help="Warn about regions.json crops that are too tight")

    parser.add_argument("--test-region", action="store_true", help="Test a candidate region")
    parser.add_argument("--image", type=Path, help="Saved screenshot to test against")
    parser.add_argument("--left", type=int)
    parser.add_argument("--top", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--kind", choices=["score", "clock", "stamina"], help="What to read from the region")

    args = parser.parse_args()

    if args.validate:
        warnings = validate_regions_file(args.regions)
        if warnings:
            print("\n".join(warnings))
            raise SystemExit(1)
        print(f"PASS {args.regions}: all configured regions meet recommended minimum dimensions")
        return

    if args.select:
        if not args.image:
            parser.error("--select requires --image SCREENSHOT")
        RegionSelector(args.image, args.regions).run()
        return

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
