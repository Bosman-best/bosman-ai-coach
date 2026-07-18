"""
Generates gui/assets/header_banner.png - an original illustrated header
for the GUI. Deliberately generic: a silhouetted touchline coach and a
tactics-board motif, no real person's likeness and no club crest/branding,
since neither can be reproduced here. Colors are a plain navy/sky-blue
scheme chosen for a clean "tactics app" feel, not to represent any real
team's identity.

Re-run this file directly if you ever want to tweak the banner:
    python gui/assets/generate_banner.py
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_PATH = Path(__file__).parent / "header_banner.png"
W, H = 1000, 220

TOP_COLOR = (13, 27, 54)      # deep navy
BOTTOM_COLOR = (24, 74, 120)  # muted sky blue
LINE_COLOR = (255, 255, 255)
ACCENT = (120, 200, 255)


def _vertical_gradient(w, h, top, bottom):
    img = Image.new("RGB", (w, h), top)
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def _draw_pitch_lines(draw: ImageDraw.ImageDraw):
    """Faint touchline/centre-circle motif, low opacity via a thin blended
    color rather than true alpha (keeps this a flat RGB PNG)."""
    faint = (255, 255, 255)
    # Halfway line
    draw.line([(0, H - 30), (W, H - 30)], fill=(60, 100, 140), width=2)
    # Centre circle, partially off the bottom edge
    cx, cy, r = W // 2, H - 30, 70
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(60, 100, 140), width=2)
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(60, 100, 140))


def _draw_tactics_dots(draw: ImageDraw.ImageDraw):
    """Small formation-dot-and-line motif on the right side, evoking a
    tactics board without depicting any real match or team."""
    dots = [
        (770, 70), (830, 55), (890, 70), (830, 100),
        (800, 140), (860, 140), (830, 170),
    ]
    lines = [(0, 4), (1, 4), (2, 4), (4, 5), (5, 6), (3, 4)]
    for a, b in lines:
        draw.line([dots[a], dots[b]], fill=(90, 150, 200), width=2)
    for x, y in dots:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=ACCENT, outline=(255, 255, 255))


def _draw_coach_silhouette(draw: ImageDraw.ImageDraw, base_x=95, base_y=195):
    """Flat, faceless silhouette: arms folded, tracksuit, on the
    touchline - a generic 'coach' figure, not any real individual."""
    fill = (8, 16, 32)
    # Legs
    draw.rectangle([base_x - 18, base_y - 60, base_x - 4, base_y], fill=fill)
    draw.rectangle([base_x + 4, base_y - 60, base_x + 18, base_y], fill=fill)
    # Torso (slightly trapezoidal for a jacket silhouette)
    draw.polygon(
        [
            (base_x - 26, base_y - 60),
            (base_x + 26, base_y - 60),
            (base_x + 20, base_y - 130),
            (base_x - 20, base_y - 130),
        ],
        fill=fill,
    )
    # Folded arms (simple rounded block across the torso)
    draw.rounded_rectangle(
        [base_x - 24, base_y - 108, base_x + 24, base_y - 88], radius=10, fill=fill
    )
    # Head
    draw.ellipse([base_x - 14, base_y - 158, base_x + 14, base_y - 130], fill=fill)


def build_banner() -> Path:
    img = _vertical_gradient(W, H, TOP_COLOR, BOTTOM_COLOR)
    draw = ImageDraw.Draw(img)
    _draw_pitch_lines(draw)
    _draw_tactics_dots(draw)
    _draw_coach_silhouette(draw)

    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 46
        )
        subtitle_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
        )
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()

    draw.text((190, 62), "BOSMAN AI COACH", font=title_font, fill=LINE_COLOR)
    draw.text(
        (192, 118),
        "Local, offline tactical advice \u2014 powered by Ollama",
        font=subtitle_font,
        fill=(190, 215, 235),
    )

    img.save(OUT_PATH)
    return OUT_PATH


if __name__ == "__main__":
    path = build_banner()
    print(f"Saved {path}")
