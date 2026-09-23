from pathlib import Path
import subprocess
import shutil
ROOT=Path(__file__).resolve().parent
DEST=ROOT.parents[3]/'img/2026/03/22'
(ROOT/'build').mkdir(exist_ok=True)
(ROOT/'exports').mkdir(exist_ok=True)
DEST.mkdir(parents=True,exist_ok=True)
def run(args): subprocess.run([str(x) for x in args],check=True,cwd=ROOT/'build',stdout=subprocess.DEVNULL)
run(['python3',ROOT/'source/draw.py'])
for name in ['tile-chain','online-state']:
 run(['xelatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'])
 shutil.copy2(ROOT/'build'/f'{name}.pdf',ROOT/'exports'/f'{name}.pdf')
 run(['pdftocairo','-svg',f'{name}.pdf',ROOT/'exports'/f'{name}.svg'])
 run(['pdftoppm','-png','-scale-to','2400','-singlefile',f'{name}.pdf',DEST/f'flash-attention-{name}'])
 run(['pdftocairo','-png','-transp','-scale-to','2400','-singlefile',f'{name}.pdf',ROOT/'exports'/f'{name}-transparent'])
print('Rendered both figures and updated article PNGs.')
