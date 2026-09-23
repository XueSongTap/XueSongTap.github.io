#!/usr/bin/env python3
"""Rebuild editable TikZ and all article exports; requires XeLaTeX and Poppler."""
from pathlib import Path
import shutil
import subprocess
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
NAME = 'norm-token-rows'
build, exports = ROOT / 'build', ROOT / 'exports'
build.mkdir(exist_ok=True)
exports.mkdir(exist_ok=True)
subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={build}', str(ROOT / 'source' / f'{NAME}.tex')], check=True, stdout=subprocess.DEVNULL)
pdf = exports / f'{NAME}.pdf'
shutil.copy2(build / f'{NAME}.pdf', pdf)
subprocess.run(['pdftocairo', '-svg', str(pdf), str(exports / f'{NAME}.svg')], check=True)
for suffix, extra in [('', []), ('-transparent', ['-transp'])]:
    subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '180', *extra, str(pdf), str(exports / f'{NAME}{suffix}')], check=True)
target = REPO / 'img/2025/11/09' / f'{NAME}.png'
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(exports / f'{NAME}.png', target)
print(target)
