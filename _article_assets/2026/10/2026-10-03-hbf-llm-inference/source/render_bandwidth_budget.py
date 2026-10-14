"""Canonical renderer; editable values live in bandwidth-budget.json."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
p=json.loads((ASSET_ROOT/'source'/'bandwidth-budget.json').read_text())
c=p['colors']
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=p['figure_inches'],facecolor='white')
fig.text(.055,.955,p['title'],fontsize=24,fontweight='bold',color=c['ink'])
fig.text(.055,.918,p['subtitle'],fontsize=12,color=c['muted'])

def panel(bottom,label,title):
    ax=fig.add_axes([.055,bottom,.89,.25])
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(0,.98,label,fontsize=16,fontweight='bold',color=c['ink'],va='top')
    ax.text(.045,.98,title,fontsize=15,fontweight='bold',color=c['ink'],va='top')
    return ax

def box(ax,x,y,w,h,text,color,fill,size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.018',
                              linewidth=1.5,edgecolor=color,facecolor=fill))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=c['ink'],linespacing=1.35)

def arrow(ax,a,b,color):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,linewidth=1.8,color=color))

ax=panel(.64,'A','8-way HBM bandwidth illustration')
n=p['hbm_channels'];total=p['hbm_total_tb_s']
left=.025; step=.12; w=.095
for i in range(n):
    x=left+i*step
    box(ax,x,.52,w,.23,f'HBM {i+1}\n~{total/n:g} TB/s',c['hbm'],c['hbm_fill'],11)
    arrow(ax,(x+w/2,.50),(x+w/2,.32),c['hbm'])
box(ax,.025,.13,.935,.17,f'GPU  |  aggregate bandwidth budget: {total:g} TB/s',c['hbm'],c['hbm_fill'],13)
ax.text(.025,.02,'Equal split for illustration; not a physical package layout or a per-stack product specification.',
        fontsize=10.5,color=c['muted'])

ax=panel(.35,'B','One HBF source: a fast buffer cannot expand source bandwidth')
bw=p['hbf_stack_tb_s'];cap=p['hbf_stack_gib']
box(ax,.025,.33,.225,.39,f'HBF weights\n{cap:g} GiB\n{bw:g} TB/s interface target',c['hbf'],c['hbf_fill'],11.5)
box(ax,.405,.33,.26,.39,'Fast buffer\nHBM / SRAM\n(path-dependent)',c['buffer'],c['buffer_fill'],12)
box(ax,.80,.33,.16,.39,'GPU\ncompute',c['hbm'],c['hbm_fill'],13)
arrow(ax,(.258,.525),(.395,.525),c['hbf'])
arrow(ax,(.674,.525),(.79,.525),c['hbm'])
ax.text(.326,.76,'Miss / refill',ha='center',fontsize=11,color=c['hbf'])
ax.text(.735,.76,'Serve data',ha='center',fontsize=11,color=c['hbm'])
ax.text(.025,.13,f'Cache-miss data from HBF: <= {bw:g} TB/s at the source interface.',fontsize=12,fontweight='bold',color=c['hbf'])
ax.text(.025,.01,'Hits may reuse resident data. The buffer path is illustrative; HBM staging is not universal.',fontsize=10.5,color=c['muted'])

ax=panel(.06,'C','Four HBF stacks: a hypothetical sum across independent interfaces')
n=p['hbf_stacks']
for i in range(n):
    x=.025+i*.155
    box(ax,x,.47,.125,.28,f'HBF {i+1}\n{cap:g} GiB\n{bw:g} TB/s',c['hbf'],c['hbf_fill'],11)
    arrow(ax,(x+.0625,.455),(x+.0625,.30),c['hbf'])
box(ax,.025,.12,.59,.16,'Host / GPU-side data paths',c['hbm'],c['hbm_fill'],12)
ax.text(.69,.67,f'{n*cap/1024:g} TiB capacity',fontsize=16,fontweight='bold',color=c['ink'])
ax.text(.69,.48,f'{n*bw:g} TB/s',fontsize=20,fontweight='bold',color=c['hbf'])
ax.text(.69,.35,'ideal interface sum',fontsize=12,color=c['muted'])
ax.text(.69,.14,'Needs host links, balanced data\nand a viable thermal budget.',fontsize=11.5,color=c['ink'],linespacing=1.4)
ax.text(.025,.015,'Hypothetical bandwidth sum; not a shipping GPU configuration.',fontsize=11,fontweight='bold',color=c['hbf'])
for y in [.615,.325]:
    fig.lines.append(plt.Line2D([.055,.945],[y,y],transform=fig.transFigure,color=c['line'],linewidth=1))
fig.text(.055,.024,'Compare parallel stacks. HBF can use multiple stacks; adding layers within one stack mainly adds capacity.',
         fontsize=11,color=c['muted'])
out=REPO_ROOT/p['png_output'];out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=p['dpi'],facecolor='white')
for fmt in ['svg','pdf']:
    fig.savefig(ASSET_ROOT/'exports'/f'hbf-bandwidth-budget.{fmt}',facecolor='white')
plt.close(fig)
print(out)
