"""Canonical editable plotting implementation; parameters are in parameters.json."""
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

p = json.loads((ASSET_ROOT / 'source' / 'parameters.json').read_text())
c = p['colors']
N, k = p['experts'], p['experts_per_token']
assert 0 < k <= N
# Each token chooses a uniformly random k-subset; tokens choose independently.
# P(a given expert is untouched by B tokens) = (1 - k/N)^B.
def coverage(batch):
    return 1 - (1 - k / N) ** batch
xs = [2 ** (i / 100) for i in range(0, round(math.log2(p['batch_max']) * 100) + 1)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(p['width_inches'], p['height_inches']), facecolor='white')
fig.subplots_adjust(left=.10, right=.94, bottom=.24, top=.78)
fig.text(.10,.93,p['title'],fontsize=23,fontweight='bold',color=c['text'])
fig.text(.10,.885,p['subtitle'],fontsize=12,color=c['muted'])
ax.set_facecolor('white')
ax.set_xscale('log',base=2)
ax.plot(xs,[coverage(x) for x in xs],color=c['curve'],linewidth=3,zorder=3)
ax.fill_between(xs,[coverage(x) for x in xs],color=c['fill'],zorder=1)
ax.set_xlim(1,p['batch_max'])
ax.set_ylim(0,1.05)
ax.set_xticks([1,2,4,8,16,32,64,128,256])
ax.set_xticklabels(['1','2','4','8','16','32','64','128','256'])
ax.set_yticks([0,.25,.5,.75,1])
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax.tick_params(axis='both',length=0,pad=9,labelcolor=c['muted'])
ax.grid(axis='y',color=c['grid'],linewidth=1)
for spine in ['left','bottom']:
    ax.spines[spine].set_color(c['grid'])
ax.set_xlabel('Batch size B (tokens; logarithmic scale)',labelpad=15,color=c['text'])
ax.set_ylabel('Expected share of experts touched',labelpad=15,color=c['text'])
positions={1:(18,18),8:(0,21),32:(0,-43),128:(0,-44)}
for batch in p['highlight_batches']:
    y=coverage(batch)
    ax.vlines(batch,0,y,color=c['curve'],alpha=.23,linewidth=1.2,linestyle=(0,(4,3)),zorder=2)
    ax.scatter([batch],[y],s=65,color=c['curve'],edgecolors='white',linewidths=1.5,zorder=4,clip_on=False)
    dx,dy=positions[batch]
    ax.annotate(f'B = {batch}\n{y:.1%}',xy=(batch,y),xytext=(dx,dy),textcoords='offset points',
                ha='left' if batch==1 else 'center',va='bottom' if dy>0 else 'top',
                fontsize=12,fontweight='bold',color=c['text'],
                bbox={'boxstyle':'round,pad=.35','facecolor':'white','edgecolor':'none','alpha':.96},zorder=5)
fig.text(.10,.115,r'$\mathbb{E}[U_B]/N = 1 - (1 - k/N)^B$',fontsize=18,color=c['text'])
fig.text(.10,.070,p['assumptions'],fontsize=11,color=c['muted'])
fig.text(.10,.030,p['qualification'],fontsize=10,color=c['muted'])
exports=ASSET_ROOT/'exports'
exports.mkdir(exist_ok=True)
output=REPO_ROOT/p['png_output']
output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(output,dpi=p['dpi'],facecolor='white')
for suffix in ['svg','pdf']:
    fig.savefig(exports/f'hbf-moe-working-set.{suffix}',facecolor='white')
plt.close(fig)
print(output)
for b in p['highlight_batches']:
    print(f'B={b}: {coverage(b):.8%}; expected experts={N*coverage(b):.4f}')
