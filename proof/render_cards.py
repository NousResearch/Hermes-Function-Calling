#!/usr/bin/env python3
"""Render title/end cards as PNG so assemble does not need a browser."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "cards"
W, H = 1280, 800
BG = (16, 16, 20)
FG = (242, 242, 245)
SUB = (154, 154, 166)

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def wrap(draw, text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def card(title: str, subtitle: str, dest: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(MONO, 42)
    sub_font = ImageFont.truetype(SANS, 22)
    max_w = int(W * 0.84)
    t_lines = wrap(draw, title, title_font, max_w)
    s_lines = wrap(draw, subtitle, sub_font, max_w)
    gap = 22
    t_h = 50
    s_h = 32
    block = len(t_lines) * t_h + gap + len(s_lines) * s_h
    y = (H - block) // 2
    for line in t_lines:
        tw = draw.textlength(line, font=title_font)
        draw.text(((W - tw) / 2, y), line, font=title_font, fill=FG)
        y += t_h
    y += gap
    for line in s_lines:
        tw = draw.textlength(line, font=sub_font)
        draw.text(((W - tw) / 2, y), line, font=sub_font, fill=SUB)
        y += s_h
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)
    print(dest, dest.stat().st_size)


def main() -> None:
    card(
        "Hermes Function Calling",
        "proving the toolkit — schemas, parser, validator, live tool execution",
        OUT / "title.png",
    )
    card(
        "the contract holds without the GPU",
        "schemas · parse · validate · execute — github.com/NousResearch/Hermes-Function-Calling",
        OUT / "end.png",
    )


if __name__ == "__main__":
    main()
