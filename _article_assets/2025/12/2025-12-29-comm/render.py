#!/usr/bin/env python3
"""Rebuild the editable TikZ figure and all article/export images."""
from pathlib import Path
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
NAME = 'alltoall-rank-exchange'
build = BASE / 'build'
exports = BASE / 'exports'
images = ROOT / 'img/2025/12/29'
for directory in (build, exports, images):
    directory.mkdir(parents=True, exist_ok=True)
subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                f'-output-directory={build}', str(BASE / 'source' / f'{NAME}.tex')],
               cwd=BASE, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
pdf = exports / f'{NAME}.pdf'
shutil.copy2(build / f'{NAME}.pdf', pdf)
subprocess.run(['pdftocairo', '-svg', str(pdf), str(exports / f'{NAME}.svg')], check=True)
for transparent in (False, True):
    target = exports / f'{NAME}-transparent' if transparent else images / NAME
    command = ['pdftocairo', '-png', '-singlefile', '-r', '180']
    if transparent:
        command.append('-transp')
    subprocess.run(command + [str(pdf), str(target)], check=True)
print(images / f'{NAME}.png')
