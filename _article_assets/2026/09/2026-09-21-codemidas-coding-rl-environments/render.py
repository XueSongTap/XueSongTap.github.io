"""Crop original paper figures at 216 dpi without redrawing their contents."""
from pathlib import Path
import json
import shutil
import pymupdf

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
OUTPUT = ROOT / 'img/2026/09/21'
OUTPUT.mkdir(parents=True, exist_ok=True)
(BASE / 'exports').mkdir(exist_ok=True)
with pymupdf.open(BASE / 'source/codemidas-v1.pdf') as doc:
    for spec in json.loads((BASE / 'source/crops.json').read_text()):
        page = doc[spec['page'] - 1]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3),
                              clip=pymupdf.Rect(spec['rect']), alpha=False)
        target = BASE / 'exports' / (spec['name'] + '.png')
        pix.save(target)
        shutil.copyfile(target, OUTPUT / target.name)
        print(f'{target.name}: {pix.width} x {pix.height}')
