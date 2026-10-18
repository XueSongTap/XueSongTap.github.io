#!/usr/bin/env python3
"""Rebuild the article figure from its editable TikZ source (no Python packages)."""
from pathlib import Path
import json
import re
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
SOURCE = BASE / "source" / "flexattention-block-mask.tex"
BUILD = BASE / "build"
EXPORTS = BASE / "exports"
IMAGE = ROOT / "img" / "2026" / "10" / "18" / "flexattention-block-mask.png"
STEM = SOURCE.stem


def run(args):
    result = subprocess.run(args, cwd=BUILD, text=True, capture_output=True)
    (BUILD / (Path(args[0]).name + ".log")).write_text(
        result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed: {args}\n{result.stdout[-5000:]}\n{result.stderr}")


def main():
    for directory in (BUILD, EXPORTS, IMAGE.parent):
        directory.mkdir(parents=True, exist_ok=True)
    text = SOURCE.read_text(encoding="utf-8")
    params = {name: int(re.search(r"\\newcommand\{\\" + name + r"\}\{(\d+)\}", text)[1])
              for name in ("SeqLen", "WindowSize", "MaskBlockSize")}
    n, window, block = (params[k] for k in ("SeqLen", "WindowSize", "MaskBlockSize"))
    # This is a fixed, readable worked example. Update layout/labels for other sizes.
    assert (n, window, block) == (16, 8, 4)
    mask = [[0 <= i - j < window for j in range(n)] for i in range(n)]
    counts = [[sum(mask[i][j] for i in range(r*block, (r+1)*block)
                   for j in range(c*block, (c+1)*block))
               for c in range(n//block)] for r in range(n//block)]
    classes = [["empty" if v == 0 else "full" if v == block**2 else "partial"
                for v in row] for row in counts]
    totals = {kind: sum(v == kind for row in classes for v in row)
              for kind in ("full", "partial", "empty")}
    # Independently derived expected counts and support cardinality.
    assert counts == [[10, 0, 0, 0], [16, 10, 0, 0], [6, 16, 10, 0], [0, 6, 16, 10]]
    assert totals == {"full": 3, "partial": 6, "empty": 7}
    assert sum(map(sum, mask)) == sum(min(i+1, window) for i in range(n)) == 100

    token_tiles = []
    for i in range(n):
        for j in range(n):
            x, y = j*.36, -i*.36
            fill = "teal" if mask[i][j] else "white"
            token_tiles.append(
                rf"\path[fill={fill},draw=rule!25,line width=.22pt,rounded corners=.65pt] "
                rf"({x+.014:.3f},{y-.014:.3f}) rectangle ({x+.346:.3f},{y-.346:.3f});")
    (BUILD / "mask-cells.tex").write_text("\n".join(token_tiles), encoding="utf-8")
    block_tiles = []
    for r, row in enumerate(counts):
        for c, count in enumerate(row):
            x, y = c*1.44, -r*1.44
            kind = classes[r][c]
            fill = {"empty": "white", "full": "teal", "partial": "orange"}[kind]
            foreground = "white" if kind == "full" else "ink" if kind == "partial" else "muted"
            block_tiles.append(
                rf"\path[fill={fill},draw=rule!45,line width=.4pt,rounded corners=1pt] "
                rf"({x+.04:.3f},{y-.04:.3f}) rectangle ({x+1.40:.3f},{y-1.40:.3f});")
            block_tiles.append(rf"\node[font=\large,text={foreground}] at ({x+.72:.3f},{y-.72:.3f}) {{$ {count} $}};")
    (BUILD / "block-cells.tex").write_text("\n".join(block_tiles), encoding="utf-8")
    (BUILD / "verification.json").write_text(json.dumps(
        {"parameters": params, "valid_entries": 100, "counts": counts,
         "classes": classes, "class_totals": totals}, indent=2), encoding="utf-8")

    xelatex = shutil.which("xelatex") or "/Library/TeX/texbin/xelatex"
    pdftocairo = shutil.which("pdftocairo") or "/opt/homebrew/bin/pdftocairo"
    run([xelatex, "-interaction=nonstopmode", "-halt-on-error", str(SOURCE)])
    pdf = BUILD / f"{STEM}.pdf"
    shutil.copy2(pdf, EXPORTS / pdf.name)
    run([pdftocairo, "-svg", str(pdf), str(EXPORTS / f"{STEM}.svg")])
    run([pdftocairo, "-png", "-singlefile", "-r", "240", str(pdf), str(IMAGE.with_suffix(""))])
    run([pdftocairo, "-png", "-singlefile", "-transp", "-r", "240",
         str(pdf), str(EXPORTS / f"{STEM}-transparent")])
    print(json.dumps({"image": str(IMAGE), "class_totals": totals, "valid_entries": 100}, indent=2))


if __name__ == "__main__":
    main()
