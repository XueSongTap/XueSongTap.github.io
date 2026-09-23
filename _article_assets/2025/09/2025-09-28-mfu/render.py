#!/usr/bin/env python3
"""Rebuild the editable TikZ figure and update the blog PNG."""
from pathlib import Path
import shutil
import subprocess
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
NAME = 'linear-training-gemm'
def run(*args):
    subprocess.run([str(x) for x in args], check=True, cwd=ROOT)
def main():
    build, exports = ROOT / 'build', ROOT / 'exports'
    build.mkdir(exist_ok=True)
    exports.mkdir(exist_ok=True)
    run('xelatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={build}', ROOT / 'source' / f'{NAME}.tex')
    pdf = exports / f'{NAME}.pdf'
    shutil.copy2(build / f'{NAME}.pdf', pdf)
    run('pdftocairo', '-svg', pdf, exports / f'{NAME}.svg')
    for suffix, options in [('', []), ('-transparent', ['-transp'])]:
        run('pdftocairo', '-png', '-singlefile', '-r', '180', *options, pdf, exports / f'{NAME}{suffix}')
    dest = REPO / 'img/2025/09/28' / f'{NAME}.png'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exports / f'{NAME}.png', dest)
if __name__ == '__main__':
    main()
