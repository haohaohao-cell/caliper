"""把生成的 SVG 粗略栅格化成 PNG，只为肉眼验证排版（不是通用渲染器）。

只支持本项目用到的那几个元素：rect / line / text。
中文字体用系统中的微软雅黑，缺字体时退化成方块——排版位置仍然可验。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _tw(s: str, size: float) -> float:
    return sum(size * (1.0 if ord(c) > 127 else 0.55) for c in s)


def render(svg_path: Path, out_path: Path, scale: float = 1.5) -> None:
    s = svg_path.read_text(encoding="utf-8")
    W = int(float(re.search(r'width="(\d+)"', s).group(1)) * scale)
    H = int(float(re.search(r'height="(\d+)"', s).group(1)) * scale)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    for m in re.finditer(r"<rect ([^/>]*)(?:/>|>(.*?)</rect>)", s, re.S):
        attrs, _ = m.groups()
        g = dict(re.findall(r'([\w-]+)="([^"]*)"', attrs))
        if "width" not in g:
            continue
        x, y = float(g.get("x", 0)) * scale, float(g.get("y", 0)) * scale
        w, h = float(g["width"]) * scale, float(g["height"]) * scale
        fill = g.get("fill", "#000000")
        if fill == "none":
            continue
        d.rectangle([x, y, x + w, y + h], fill=fill)

    for m in re.finditer(r'<line x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)" stroke="([^"]+)"', s):
        x1, y1, x2, y2, c = m.groups()
        d.line([float(x1) * scale, float(y1) * scale, float(x2) * scale, float(y2) * scale], fill=c, width=1)

    for m in re.finditer(r"<text ([^>]*)>(.*?)</text>", s, re.S):
        attrs, txt = m.groups()
        g = dict(re.findall(r'([\w-]+)="([^"]*)"', attrs))
        txt = txt.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        x, y = float(g["x"]) * scale, float(g["y"]) * scale
        size = float(g.get("font-size", 12)) * scale
        anchor = g.get("text-anchor", "start")
        bold = g.get("font-weight", "400") in ("600", "700")
        f = _font(max(8, int(size)), bold)
        w = _tw(txt, size)
        if anchor == "middle":
            x -= w / 2
        elif anchor == "end":
            x -= w
        d.text((x, y - size * 0.82), txt, font=f, fill=g.get("fill", "#000000"))

    img.save(out_path)


if __name__ == "__main__":
    for f in sorted((ROOT / "charts").glob("*.svg")):
        out = f.with_suffix(".preview.png")
        render(f, out)
        print(f"{f.name} -> {out.name}")
    sys.exit(0)
