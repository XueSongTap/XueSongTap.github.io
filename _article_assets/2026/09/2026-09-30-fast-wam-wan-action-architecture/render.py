#!/usr/bin/env python3
"""Rebuild the mask diagram and vector crops of the paper's original figures."""
from pathlib import Path
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parent
    repo = next(p for p in root.parents if (p / '_posts').is_dir())
    build, exports = root / 'build', root / 'exports'
    images = repo / 'img' / '2026' / '09' / '30'
    for directory in (build, exports, images):
        directory.mkdir(parents=True, exist_ok=True)
    for executable in ('xelatex', 'pdftocairo'):
        if shutil.which(executable) is None:
            raise SystemExit(f'Missing dependency: {executable}')
    for source in sorted((root / 'source').glob('*.tex')):
        name = source.stem
        with (build / f'{name}-compile.log').open('w') as log:
            subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                            f'-output-directory={build}', str(source)],
                           cwd=source.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
        pdf = build / f'{name}.pdf'
        dpi = '320' if name.startswith('fast-wam-paper-') else '220'
        subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', dpi,
                        str(pdf), str(build / name)], check=True)
        subprocess.run(['pdftocairo', '-png', '-transp', '-singlefile', '-r', dpi,
                        str(pdf), str(exports / f'{name}-transparent')], check=True)
        subprocess.run(['pdftocairo', '-svg', str(pdf), str(exports / f'{name}.svg')], check=True)
        shutil.copy2(pdf, exports / pdf.name)
        shutil.copy2(build / f'{name}.png', images / f'{name}.png')
        print(f'Updated {images / (name + ".png")}')


if __name__ == '__main__':
    main()
