"""Render a 1200×630 atlas reply preview. Requires Pillow 10.1 or newer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SIZE = (1200, 630)
BACKGROUND = "#10131b"
PANEL = "#171c26"
LINE = "#2a3241"
TEXT = "#f3f4f3"
MUTED = "#a7afc0"
ACCENT = "#d8fa86"
BLUE = "#acbbfc"
CATEGORIES = {
    "persona": ("Persona", ACCENT),
    "changed": ("Changed assistant", BLUE),
    "off_topic": ("Off-topic", "#d7aefa"),
    "normal": ("Normal", "#9ecbbb"),
    "incoherent": ("Incoherent", "#f0b48f"),
}
FONTS = {
    "regular": ("/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"),
    "bold": ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"),
    "mono": ("/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf"),
}


@lru_cache(maxsize=24)
def load_font(role: str, size: int) -> ImageFont.FreeTypeFont:
    for name in FONTS[role]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
              width: int, max_lines: int) -> list[str]:
    remaining = " ".join(text.split())
    lines = []
    for index in range(max_lines):
        if draw.textlength(remaining, font=font) <= width:
            lines.append(remaining)
            break
        suffix = "…" if index == max_lines - 1 else ""
        limit = 1
        while limit < len(remaining) and draw.textlength(remaining[:limit + 1] + suffix, font=font) <= width:
            limit += 1
        boundary = remaining.rfind(" ", 0, limit + 1)
        if boundary > 0:
            limit = boundary
        lines.append(remaining[:limit] + suffix)
        remaining = remaining[limit:].lstrip()
    return lines


@lru_cache(maxsize=1)
def card_background() -> Image.Image:
    image = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.line((72, 116, 1128, 116), fill=LINE, width=2)
    draw.line((72, 567, 1128, 567), fill=LINE, width=2)
    draw.rounded_rectangle((72, 280, 1128, 541), radius=18, fill=PANEL, outline=LINE, width=2)
    for x in range(904, 1111, 16):
        for y in range(296, 526, 16):
            draw.point((x, y), fill="#364155")
    draw.rounded_rectangle((72, 49, 111, 88), radius=9, fill="#242e24", outline="#414c38")
    for dx, dy in ((11, 0), (0, 11), (8, 8), (8, -8)):
        draw.line((91 - dx, 68 - dy, 91 + dx, 68 + dy), fill=ACCENT, width=2)
    draw.ellipse((88, 65, 94, 71), fill=ACCENT)
    return image


def render_og(record: dict, destination: str | Path) -> None:
    """Write a share image for one selected atlas result."""
    image = card_background().copy()
    draw = ImageDraw.Draw(image)
    category, color = CATEGORIES.get(record.get("category"), ("Observation", MUTED))
    if record.get("sensitive"):
        title = "Sensitive reply"
        excerpt = "This reply is marked sensitive."
    else:
        title = record.get("name") or category
        excerpt = record["reply"].strip() or "No visible reply."

    draw.text((127, 68), "The Inkomplete Inkling Persona Atlas", fill=TEXT,
              font=load_font("bold", 25), anchor="lm")
    pill_font = load_font("mono", 19)
    pill_text = category.upper()
    pill_x = 1128 - draw.textlength(pill_text, font=pill_font) - 55
    draw.rounded_rectangle((pill_x, 49, 1128, 88), radius=9, fill=PANEL, outline=LINE)
    draw.ellipse((pill_x + 15, 65, pill_x + 22, 72), fill=color)
    draw.text((pill_x + 31, 68), pill_text, fill=color, font=pill_font, anchor="lm")

    for size in (62, 54, 46):
        title_font = load_font("bold", size)
        if draw.textlength(title, font=title_font) <= 1056:
            break
    title_line = wrap_text(draw, title, title_font, 1056, 1)[0]
    draw.text((72, 150), title_line, fill=TEXT, font=title_font, anchor="lt")

    expert = record["expert_type"].upper()
    if record["expert"] is not None:
        expert += f" {record['expert']}"
    coordinate = (f"L{record['layer']:02d}  ·  {expert}  ·  N{record['neuron']}"
                  f"  ·  PUSH {record['direction'].upper()}  ·  {record['strength']:g}×")
    draw.text((74, 238), coordinate, fill=BLUE, font=load_font("mono", 23), anchor="lt")

    draw.text((99, 309), "“", fill=ACCENT, font=load_font("bold", 64), anchor="lt")
    quote_font = load_font("regular", 37)
    for index, line in enumerate(wrap_text(draw, excerpt, quote_font, 940, 4)):
        draw.text((156, 321 + index * 47), line, fill=TEXT, font=quote_font, anchor="lt")

    footer_font = load_font("mono", 19)
    draw.text((72, 595), "INKLING-SMALL", fill=MUTED, font=footer_font, anchor="lm")
    draw.text((1128, 595), "SINGLE-NEURON STEERING", fill=MUTED, font=footer_font, anchor="rm")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
