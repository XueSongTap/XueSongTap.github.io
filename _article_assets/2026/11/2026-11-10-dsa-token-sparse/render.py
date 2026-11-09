#!/usr/bin/env python3
"""Rebuild the DSA figures with XeLaTeX and Poppler; no fixed workspace paths."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
BUILD = ROOT / "build"
EXPORTS = ROOT / "exports"
IMAGES = REPO / "img/2026/11/10"


def run(*args, cwd=BUILD):
    result = subprocess.run([str(x) for x in args], cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        print(result.stdout[-6000:])
        result.check_returncode()


def main():
    for p in (BUILD, EXPORTS, IMAGES):
        p.mkdir(parents=True, exist_ok=True)
    run(sys.executable, ROOT / "source/gen_figures.py", BUILD)
    for tex in sorted(BUILD.glob("*.tex")):
        name = tex.stem
        run("xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name)
        pdf = BUILD / f"{name}.pdf"
        shutil.copy2(pdf, EXPORTS / pdf.name)
        shutil.copy2(tex, EXPORTS / tex.name)
        run("pdftoppm", "-singlefile", "-png", "-r", "220", pdf, BUILD / name)
        shutil.copy2(BUILD / f"{name}.png", IMAGES / f"{name}.png")
        run("pdftocairo", "-singlefile", "-png", "-transp", "-r", "220", pdf, EXPORTS / f"{name}-transparent")
        run("pdftocairo", "-svg", pdf, EXPORTS / f"{name}.svg")
        print(f"Rendered {IMAGES / (name + '.png')}")


if __name__ == "__main__":
    main()
