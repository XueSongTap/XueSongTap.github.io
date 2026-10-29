"""Original DSpark explanatory figures. Editable source; see ../README.md."""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]
OUT = REPO / 'img/2026/10/23'
EXPORTS = ROOT / 'exports'
BUILD = ROOT / 'build'
for p in (OUT, EXPORTS, BUILD):
    p.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    Path('/System/Library/Fonts/STHeiti Medium.ttc'),
    Path('/System/Library/Fonts/STHeiti Light.ttc'),
    Path('/System/Library/Fonts/PingFang.ttc'),
    Path('/Library/Fonts/Arial Unicode.ttf'),
]
font_file = next((p for p in FONT_CANDIDATES if p.exists()), None)
if font_file is None:
    raise RuntimeError('Install a CJK font or update FONT_CANDIDATES.')
fm.fontManager.addfont(str(font_file))
FONT = fm.FontProperties(fname=str(font_file))
plt.rcParams.update({
    'font.family': FONT.get_name(),
    'font.size': 13,
    'axes.unicode_minus': False,
    'svg.fonttype': 'path',
    'pdf.fonttype': 42,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})
DARK = '#182b3a'
MUTED = '#536473'
TEAL = '#087f5b'
TEAL_LIGHT = '#e5f5ec'
TEAL_BORDER = '#54a980'
GRAY = '#f0f2f4'
GRAY_BORDER = '#cbd2d8'
BLUE = '#255b91'
BLUE_LIGHT = '#eaf2fb'
ORANGE = '#a85d14'
ORANGE_LIGHT = '#fff2dc'
RED = '#bb4a44'


def text(ax, x, y, s, size=14, color=DARK, weight='normal', ha='left', va='center', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha,
                   va=va, fontproperties=FONT, transform=ax.transAxes, **kw)


def box(ax, x, y, w, h, fc='white', ec=GRAY_BORDER, lw=1.5, radius=0.012):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                       fc=fc, ec=ec, lw=lw, transform=ax.transAxes, zorder=2)
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color=MUTED, lw=1.6, style='-|>', mutation=14,
          connectionstyle='arc3', linestyle='solid'):
    p = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=mutation,
                        lw=lw, color=color, transform=ax.transAxes,
                        connectionstyle=connectionstyle, linestyle=linestyle,
                        zorder=3)
    ax.add_patch(p)
    return p


sizes = {}
def save(fig, name):
    png = OUT / f'{name}.png'
    fig.savefig(png, dpi=180, facecolor='white')
    fig.savefig(EXPORTS / f'{name}.svg', facecolor='white')
    fig.savefig(EXPORTS / f'{name}.pdf', facecolor='white')
    sizes[name] = Image.open(png).size
    plt.close(fig)


# Figure 1: deliberately chosen teaching budget; not a search over K.
probs = {'A': [.95, .90, .85, .80], 'B': [.90, .70, .50, .40], 'C': [.80, .40, .30, .20]}
survival = {k: np.cumprod(v).tolist() for k, v in probs.items()}
selected = {'A': 3, 'B': 2, 'C': 1}
K = sum(selected.values())
chosen_sum = sum(sum(survival[k][:n]) for k, n in selected.items())
assert K == 6 and math.isclose(chosen_sum, 4.86175)
fig = plt.figure(figsize=(15.2, 8.9))
ax = fig.add_axes([.025, .025, .95, .95]); ax.set_axis_off()
text(ax, .015, .952, '固定额外预算 K = 6，验证位置怎样分给三个请求？', 24, weight='bold')
text(ax, .015, .895, '教学示例：c 为条件接受概率，a 为累乘后的前缀存活概率；按 a 选取前 6 个位置。', 14, MUTED)
text(ax, .188, .816, '每请求基础槽', 13, BLUE, ha='center')
xs = [.315, .48, .645, .81]
w, h = .145, .135
for j, x in enumerate(xs, 1):
    text(ax, x + w/2, .816, f'draft {j}', 14, MUTED, ha='center')
ys = [.625, .435, .245]
for (name, pvals), y in zip(probs.items(), ys):
    text(ax, .018, y+.095, f'请求 {name}', 20, weight='bold')
    text(ax, .018, y+.043, f'分配 {selected[name]} 个 draft', 12.5, MUTED)
    box(ax, .125, y, .126, h, BLUE_LIGHT, '#9db9d4')
    text(ax, .188, y+.092, 'target', 17, BLUE, ha='center', weight='bold')
    text(ax, .188, y+.042, '基础槽 × 1', 12.5, BLUE, ha='center')
    # This arrow denotes logical prefix order, not additional budget.
    arrow(ax, (.26, y+h/2), (.307, y+h/2), color=GRAY_BORDER, mutation=12)
    for j, (x, pval, aval) in enumerate(zip(xs, pvals, survival[name])):
        chosen = j < selected[name]
        box(ax, x, y, w, h,
            TEAL_LIGHT if chosen else GRAY,
            TEAL_BORDER if chosen else GRAY_BORDER,
            lw=1.8 if chosen else 1.2)
        text(ax, x+w/2, y+.098, f'c = {pval:.2f}', 13, TEAL if chosen else MUTED, ha='center')
        text(ax, x+w/2, y+.047, f'a = {aval:.5f}'.rstrip('0').rstrip('.'), 17,
             TEAL if chosen else MUTED, ha='center', weight='bold' if chosen else 'normal')
        if j < len(xs)-1:
            arrow(ax, (x+w+.003, y+h/2), (xs[j+1]-.005, y+h/2), color=GRAY_BORDER, mutation=10)
# Compact legend and prefix constraint immediately under rows.
box(ax, .125, .183, .018, .020, TEAL_LIGHT, TEAL_BORDER, radius=.003)
text(ax, .151, .193, '已选', 12.5, TEAL)
box(ax, .225, .183, .018, .020, GRAY, GRAY_BORDER, radius=.003)
text(ax, .251, .193, '未选', 12.5, MUTED)
text(ax, .408, .193, '每行从左向右必须构成前缀；分配结果为 3 / 2 / 1。', 13, MUTED)
box(ax, .015, .022, .94, .115, '#f7f9fb', '#e1e6eb', radius=.01)
text(ax, .035, .096, '总预算 B = 3 个基础 target 槽 + 6 个 draft 位置 = 9', 17, weight='bold')
text(ax, .035, .053, '期望产出 τ = 3 + 已选位置的 a 之和 = 3 + 4.86175 = 7.86175', 16, TEAL)
save(fig, 'dspark-prefix-budget')


# Figure 2: scheduling-step labels are logical dependencies, not durations.
fig = plt.figure(figsize=(15.2, 8.3))
ax = fig.add_axes([.025, .025, .95, .95]); ax.set_axis_off()
text(ax, .015, .95, '异步调度：旧信息决定做多少，新信息决定分给谁', 24, weight='bold')
text(ax, .015, .894, '统一以调度第 N 步编号：进入第 N 步前，总预算已经固定，GPU 只重分配逐请求长度。', 14, MUTED)
# Lane backgrounds; dashed guide lines convey logical lanes only.
ax.add_patch(Rectangle((.015,.555),.545,.235, fc='#f5f8fc', ec='none', transform=ax.transAxes,zorder=0))
ax.add_patch(Rectangle((.015,.22),.945,.245, fc='#f4faf6', ec='none', transform=ax.transAxes,zorder=0))
text(ax, .035, .755, 'CPU', 17, BLUE, weight='bold')
text(ax, .035, .423, 'GPU', 17, TEAL, weight='bold')
# CPU historical estimate.
box(ax, .105, .592, .168, .135, 'white', '#9db9d4')
text(ax, .189, .681, '第 N−2 步', 17, BLUE, ha='center', weight='bold')
text(ax, .189, .631, '历史 confidence', 14, BLUE, ha='center')
arrow(ax, (.279,.66), (.323,.66), BLUE, mutation=15)
box(ax, .33, .588, .215, .145, BLUE_LIGHT, '#8baed0', lw=1.8)
text(ax, .4375, .696, '结合硬件耗时表', 13.5, BLUE, ha='center')
text(ax, .4375, .651, '确定总预算 $B_N$', 19, BLUE, ha='center', weight='bold')
text(ax, .4375, .610, '决定第 N 步做多少', 12.5, BLUE, ha='center')
# GPU latest confidence: separate incoming path.
box(ax, .10, .26, .23, .145, ORANGE_LIGHT, '#d8b27d')
text(ax, .215, .358, '第 N−1 步 confidence', 15, ORANGE, ha='center', weight='bold')
text(ax, .215, .308, '（设备上）', 14, ORANGE, ha='center')
# GPU uses fixed B_N to assign the request-specific draft lengths.
box(ax, .40, .26, .23, .145, TEAL_LIGHT, TEAL_BORDER, lw=1.8)
text(ax, .515, .360, '固定 $B_N$ 下重新分配', 17, TEAL, ha='center', weight='bold')
text(ax, .515, .309, '各请求的 draft 长度', 14, TEAL, ha='center')
# Old-confidence path is routed vertically then right, with explicit budget label.
arrow(ax, (.4375,.58), (.4375,.415), BLUE, lw=2, mutation=16)
text(ax, .428, .494, '$B_N$ 保持不变', 13.5, BLUE, ha='right')
arrow(ax, (.337,.332), (.392,.332), ORANGE, lw=2, mutation=16)
text(ax, .215, .435, '决定分给谁', 13.5, ORANGE, ha='center')
# Step-N execution then confidence generation.
arrow(ax, (.638,.332), (.672,.332), TEAL, mutation=15)
box(ax, .68, .26, .118, .145, 'white', TEAL_BORDER)
text(ax, .739, .359, '第 N 步', 16, TEAL, ha='center', weight='bold')
text(ax, .739, .309, '主模型验证', 14, TEAL, ha='center')
arrow(ax, (.805,.332), (.837,.332), TEAL, mutation=15)
box(ax, .845, .26, .108, .145, 'white', TEAL_BORDER)
text(ax, .899, .359, '采样', 16, TEAL, ha='center')
text(ax, .899, .309, '/ draft', 15, TEAL, ha='center')
# New confidence belongs to later scheduling steps, never the allocation above.
arrow(ax, (.899,.414), (.899,.585), MUTED, lw=1.6, mutation=15)
text(ax, .889, .503, '异步回传', 12.5, MUTED, ha='right')
box(ax, .807, .602, .145, .125, 'white', GRAY_BORDER)
text(ax, .8795, .681, '新 confidence', 14, MUTED, ha='center')
text(ax, .8795, .637, '供后续步使用', 13, MUTED, ha='center')
# Footer statements are necessary diagram constraints.
box(ax, .015, .043, .938, .105, '#f7f9fb', '#e1e6eb', radius=.01)
text(ax, .035, .110, '两条信息路径：N−2 → 总预算 $B_N$；N−1 → 固定 $B_N$ 下的逐请求分配。', 15, weight='bold')
text(ax, .035, .069, '图为逻辑依赖示意；框宽、间距和箭头长度均不代表耗时。', 12.5, MUTED)
save(fig, 'dspark-async-budget')


# Figure 3: official estimates at the pinned commit, not a local rerun.
seqs = np.array([16, 32, 48, 60, 72, 96])
gains = np.array([-1.5, 1.6, 3.6, 5.4, 6.7, 11.0])
fig = plt.figure(figsize=(12.8, 7.8))
ax = fig.add_axes([.10, .20, .85, .58])
fig.text(.07, .93, '并发变化时，动态验证长度带来的吞吐增益',
         fontsize=23, weight='bold', color=DARK, fontproperties=FONT)
fig.text(.07, .865, 'dynamic 相对 fixed5 · EP64 / DP64 · DeepInfer 官方估计吞吐',
         fontsize=14, color=MUTED, fontproperties=FONT)
ax.set_xlim(9, 103); ax.set_ylim(-3.5, 13.8)
ax.set_xticks(seqs)
ax.set_yticks([-2, 0, 2, 4, 6, 8, 10, 12])
ax.set_yticklabels([f'{x}%' for x in [-2, 0, 2, 4, 6, 8, 10, 12]])
ax.tick_params(axis='both', length=0, pad=10, labelsize=13, colors=MUTED)
ax.set_axisbelow(True)
ax.grid(axis='y', color='#e5eaee', lw=1)
ax.axhline(0, color='#99a8b3', ls=(0,(4,4)), lw=1.4)
ax.plot(seqs, gains, color=TEAL, lw=2.2, zorder=3)
ax.vlines(seqs, 0, gains, color=['#deb4b0']+['#bddfd0']*5, lw=2.5, zorder=2)
ax.scatter(seqs[1:],gains[1:], s=80, color=TEAL, edgecolors='white', linewidths=1.5,zorder=4)
ax.scatter(seqs[:1],gains[:1], s=80, color=RED, edgecolors='white', linewidths=1.5,zorder=4)
for x,y in zip(seqs,gains):
    label=f'{y:+.1f}%'
    ax.annotate(label,(x,y),xytext=(0,12 if y>=0 else -23),textcoords='offset points',ha='center',
                fontsize=16, color=TEAL if y>=0 else RED, weight='bold',fontproperties=FONT)
ax.scatter([60],[5.4], s=205, facecolors='none',edgecolors='#ba8b4d',linewidths=1.2,zorder=5)
ax.annotate('60 并发工作点',(60,5.4),xytext=(29,-32),textcoords='offset points',
            fontsize=12, color='#8b6a3f',fontproperties=FONT,
            arrowprops={'arrowstyle':'-','color':'#ba8b4d','lw':1.0})
ax.set_xlabel('并发（seqs / rank）',fontsize=15,labelpad=15,color=DARK,fontproperties=FONT)
ax.set_ylabel('相对吞吐增益',fontsize=15,labelpad=16,color=DARK,fontproperties=FONT)
for s in ['top','right','left']:
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#b8c3cb')
fig.text(.07,.055,'来源：DeepInfer · 9bd9cf2 · doc/inference_deployment_qa.md，L249–256',fontsize=11.5,color=MUTED,fontproperties=FONT)
save(fig, 'dspark-concurrency-gain')

(BUILD / 'figure-manifest.json').write_text(json.dumps({
    'font': str(font_file),
    'png_dimensions': sizes,
    'teaching_example': {'conditional_acceptance': probs, 'prefix_survival': survival,
                         'selected_draft_lengths': selected, 'K': K, 'B': 3+K,
                         'tau': 3+chosen_sum},
    'official_throughput_gain_percent': {'seqs_per_rank': seqs.tolist(), 'gain': gains.tolist()},
}, ensure_ascii=False, indent=2)+'\n')
for name,size in sizes.items():
    print(f'{OUT / (name+".png")}: {size[0]} × {size[1]}')
