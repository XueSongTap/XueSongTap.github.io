#!/usr/bin/env python3
"""Rebuild this article's figures from its canonical TikZ source."""
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
repo = next(p for p in root.parents if (p / "_config.yml").is_file() and (p / "_posts").is_dir())
build = root / "build"
exports = root / "exports"
image_dir = repo / "img/2025/11/19"
for folder in (build, exports, image_dir):
    folder.mkdir(parents=True, exist_ok=True)
for tool in ("xelatex", "pdftocairo"):
    if shutil.which(tool) is None:
        raise SystemExit(f"Missing dependency: {tool}")
for source in sorted((root / "source").glob("*.tex")):
    name = source.stem
    with (build / f"{name}-build.log").open("w") as log:
        subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                        f"-output-directory={build}", str(root / "source" / f"{name}.tex")],
                       cwd=root / "source", stdout=log, stderr=subprocess.STDOUT, check=True)
    pdf = build / f"{name}.pdf"
    subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "220", str(pdf), str(build / name)], check=True)
    subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-r", "220", str(pdf), str(exports / f"{name}-transparent")], check=True)
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(exports / f"{name}.svg")], check=True)
    shutil.copy2(pdf, exports / pdf.name)
    shutil.copy2(build / f"{name}.png", image_dir / f"{name}.png")
    print(f"Updated blog figure: {image_dir / (name + '.png')}")
