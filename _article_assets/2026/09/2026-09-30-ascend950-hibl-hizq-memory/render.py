from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
DATA = json.loads((BASE / "source/memory-options.json").read_text())
FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]
FONT = next((p for p in FONT_CANDIDATES if p.exists()), None)
if FONT is None:
    raise SystemExit("Install a CJK font and add its path to FONT_CANDIDATES.")

SCALE = 2
im = Image.new("RGBA", (1600 * SCALE, 1030 * SCALE), "white")
d = ImageDraw.Draw(im)

def text(x, y, s, size=26, color="#25354a", anchor="mm"):
    f = ImageFont.truetype(str(FONT), size * SCALE)
    d.text((x * SCALE, y * SCALE), s, font=f, fill=color, anchor=anchor)

def box(coords, fill, outline="#d4dee9", radius=15, width=2):
    d.rounded_rectangle(tuple(v * SCALE for v in coords), radius=radius * SCALE,
                        fill=fill, outline=outline, width=width * SCALE)

def line(points, fill="#7a8898", width=3):
    d.line([(x * SCALE, y * SCALE) for x, y in points], fill=fill, width=width * SCALE)

text(800, 65, DATA["title"], 43)
text(800, 120, DATA["subtitle"], 27, "#617087")

for i, v in enumerate(DATA["variants"]):
    x = 55 + i * 775
    cx = x + 360
    color = v["color"]
    box((x, 175, x + 720, 825), "#f8fafc")
    text(cx, 220, v["name"], 33, color)
    text(cx, 270, "完整显存配置上限", 23, "#617087")
    text(cx, 325, f'{v["capacity"]}   |   {v["bandwidth"]}', 34, color)
    count = v["count"]
    bw = (590 - (count - 1) * 10) / count
    for j in range(count):
        bx = x + 65 + j * (bw + 10)
        box((bx, 380, bx + bw, 440), "#ffffff", color, 8)
        text(bx + bw / 2, 410, str(j + 1), 25, color)
    text(cx, 475, f'{count} 个显存模块', 27)
    line([(cx, 512), (cx, 565)], color)
    for yy, direction in ((512, 1), (565, -1)):
        d.polygon([(cx * SCALE, yy * SCALE), ((cx - 7) * SCALE, (yy + direction * 11) * SCALE),
                   ((cx + 7) * SCALE, (yy + direction * 11) * SCALE)], fill=color)
    text(cx + 65, 539, "访存", 22, color)
    box((x + 80, 582, x + 640, 665), "#eaf0f7")
    text(cx, 624, "2 × AI Die  +  2 × IO Die", 28)
    text(cx, 710, "每模块均分估算", 23, "#617087")
    text(cx, 752, v["average"], 29, color)
    text(cx, 795, v["focus"], 22, "#617087")

text(800, 867, DATA["facts"], 24)
text(800, 908, DATA["assumption"], 23, "#617087")
text(800, 947, DATA["scope"], 22, "#617087")
text(800, 986, DATA["source"], 20, "#617087")

exports = BASE / "exports"
exports.mkdir(exist_ok=True)
target = ROOT / "img/2026/09/30/ascend950-memory-options.png"
target.parent.mkdir(parents=True, exist_ok=True)
im.convert("RGB").save(target, optimize=True)
im.convert("RGB").save(exports / "ascend950-memory-options.png", optimize=True)
print(target)
