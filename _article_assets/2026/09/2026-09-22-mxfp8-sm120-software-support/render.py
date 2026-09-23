#!/usr/bin/env python3
"""Rebuild the TN / non-TN figure from its canonical editable TikZ source."""
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parent
    repo = next(p for p in root.parents if (p / "_posts").is_dir() and (p / "_config.yml").is_file())
    build, exports = root / "build", root / "exports"
    image_dir = repo / "img" / "2026" / "09" / "22"
    name = "gemm-tn-non-tn"
    for directory in (build, exports, image_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for tool in ("xelatex", "pdftocairo"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Missing dependency: {tool}")

    with (build / "render.log").open("w") as log:
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={build}", str(root / "source" / f"{name}.tex")],
            cwd=root / "source", stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    pdf = build / f"{name}.pdf"
    subprocess.run(
        ["pdftocairo", "-png", "-singlefile", "-r", "240", str(pdf), str(build / name)],
        check=True,
    )
    subprocess.run(
        ["pdftocairo", "-png", "-transp", "-singlefile", "-r", "240",
         str(pdf), str(exports / f"{name}-transparent")], check=True,
    )
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(exports / f"{name}.svg")], check=True)
    shutil.copy2(pdf, exports / pdf.name)
    shutil.copy2(build / f"{name}.png", image_dir / f"{name}.png")
    print(f"Updated article image: {image_dir / (name + '.png')}")
    print(f"Updated reusable exports: {exports}")


if __name__ == "__main__":
    main()
