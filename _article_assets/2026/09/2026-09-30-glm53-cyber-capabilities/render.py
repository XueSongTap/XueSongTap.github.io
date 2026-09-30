#!/usr/bin/env python3
"""从保存的原图重建文章配图，无网络依赖，不更改图表内容。"""
from pathlib import Path

from PIL import Image


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
FIGURES = (
    "glm53-exploitation-budget",
    "glm53-caisi-benchmarks",
    "glm53-harmful-task-engagement",
)


def main():
    exports = BASE / "exports"
    destination = ROOT / "img" / "2026" / "09" / "30"
    exports.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        with Image.open(BASE / "source" / f"{name}.webp") as original:
            foreground = original.convert("RGBA")
            background = Image.new("RGBA", foreground.size, "white")
            background.alpha_composite(foreground)
            final = background.convert("RGB")
            for target in (exports / f"{name}.png", destination / f"{name}.png"):
                final.save(target, optimize=True)
            print(f"{name}: {final.width} × {final.height}")


if __name__ == "__main__":
    main()
