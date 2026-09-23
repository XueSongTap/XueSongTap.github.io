#!/usr/bin/env python3
"""Rebuild the editable TikZ figure and all published/exported versions."""
from pathlib import Path
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
BUILD = BASE / 'build'
EXPORTS = BASE / 'exports'
NAME = 'tiled-gemm'
for directory in (BUILD, EXPORTS, ROOT / 'img/2025/10/13'):
    directory.mkdir(parents=True, exist_ok=True)
subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                f'-output-directory={BUILD}', str(BASE / 'source' / f'{NAME}.tex')],
               cwd=BASE, check=True, stdout=(BUILD / 'render.log').open('w'))
pdf = EXPORTS / f'{NAME}.pdf'
shutil.copy2(BUILD / f'{NAME}.pdf', pdf)
subprocess.run(['pdftocairo', '-svg', str(pdf), str(EXPORTS / f'{NAME}.svg')], check=True)
for transparent in (False, True):
    target = EXPORTS / f'{NAME}-transparent' if transparent else ROOT / 'img/2025/10/13' / NAME
    subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '180',
                    *(['-transp'] if transparent else []), str(pdf), str(target)], check=True)
print(ROOT / 'img/2025/10/13' / f'{NAME}.png')
