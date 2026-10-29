"""Rebuild MegaMoE figures and refresh the blog's white-background PNGs."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
SOURCE = ROOT / "source"
EXPORTS = ROOT / "exports"
BUILD = ROOT / "build"
IMAGES = REPO / "img/2026/10/14"
FIGURES = (
    "01-ep-token-flow",
    "02-expert-tensor-shapes",
    "03-communication-compute-overlap",
)


def run(args, log):
    result = subprocess.run(args, cwd=SOURCE, capture_output=True, text=True)
    output = result.stdout + result.stderr
    log.write_text(output, encoding="utf-8")
    if result.returncode or "Missing character:" in output or "Overfull" in output:
        raise RuntimeError(f"Figure build failed; inspect {log}\n{output}")


def main():
    for directory in (EXPORTS, BUILD, IMAGES):
        directory.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={BUILD}", f"{name}.tex"], BUILD / f"{name}-xelatex.txt")
        pdf = EXPORTS / f"{name}.pdf"
        shutil.copy2(BUILD / f"{name}.pdf", pdf)
        run(["pdftoppm", "-r", "180", "-png", "-singlefile", str(pdf), str(IMAGES / name)],
            BUILD / f"{name}-png.txt")
        run(["pdftocairo", "-r", "180", "-transp", "-png", "-singlefile",
             str(pdf), str(EXPORTS / f"{name}-transparent")], BUILD / f"{name}-transparent.txt")
        run(["pdftocairo", "-svg", str(pdf), str(EXPORTS / f"{name}.svg")],
            BUILD / f"{name}-svg.txt")
        print(f"Rendered {name}: blog PNG / PDF / SVG / transparent PNG")


if __name__ == "__main__":
    main()
