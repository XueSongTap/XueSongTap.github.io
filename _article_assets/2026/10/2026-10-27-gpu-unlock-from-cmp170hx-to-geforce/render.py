#!/usr/bin/env python3
"""Rebuild the article's seven PNG figures; see README for dependencies."""
from pathlib import Path
import json, math, subprocess
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'img/2026/10/27'
BUILD=HERE/'build';EXPORT=HERE/'exports'
for p in [OUT,BUILD,EXPORT]:p.mkdir(parents=True,exist_ok=True)
FONT=next((p for p in [Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')] if p.exists()),None)
if FONT is None:raise RuntimeError('Set FONT to an installed Chinese font')
fm.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':fm.FontProperties(fname=str(FONT)).get_name(),'axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white'})
INK='#17304a';BLUE='#2463a8';GREEN='#117d73';GRAY='#64748b'
def ft(n):return ImageFont.truetype(str(FONT),n)
def page(title,sub,h):
    im=Image.new('RGB',(1800,h),'white');d=ImageDraw.Draw(im)
    d.text((70,38),title,font=ft(47),fill=INK);d.text((73,110),sub,font=ft(25),fill=GRAY)
    return im,d
def box(d,xy,title,lines,color=BLUE):
    x,y,w,h=xy;d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill='#f7f9fc',outline=color,width=3)
    d.text((x+20,y+18),title,font=ft(29),fill=color)
    for i,l in enumerate(lines):d.text((x+20,y+78+42*i),l,font=ft(25),fill=INK)
def arrow(d,a,b,color=GRAY):
    d.line((a,b),fill=color,width=4);ang=math.atan2(b[1]-a[1],b[0]-a[0])
    d.polygon([b,(b[0]-18*math.cos(ang-.5),b[1]-18*math.sin(ang-.5)),(b[0]-18*math.cos(ang+.5),b[1]-18*math.sin(ang+.5))],fill=color)

im,d=page('从公开测试密钥到算力恢复','上排是离线逆向；下排是实际启动与利用。解密能力帮助分析，签名验证仍然存在。',1030)
for xy,t,ls in [((70,240,365,195),'debug Booter',['编号 37 的已知测试密钥','解密 → 获得明文']),((505,240,365,195),'反汇编与模拟',['理解 DMA、栈与寄存器','定位越界和返回片段']),((940,240,365,195),'构造利用数据',['同时覆盖 canary 两副本','安排返回地址与 ROP'])]:box(d,xy,t,ls)
arrow(d,(440,335),(495,335));arrow(d,(875,335),(930,335))
box(d,(70,620,365,195),'已签名 Booter',['硬件正常验证签名','进入 HS 模式后运行'])
box(d,(505,620,365,195),'运行时控制执行',['DMA 越界改写数据状态','执行已有代码片段'])
box(d,(940,620,365,195),'打开 PLM',['高权限修改访问控制','主机获得覆写权限'])
box(d,(1375,620,355,195),'恢复目标功能',['写 speed-select 等状态','交还原版驱动'],GREEN)
for a,b in [((440,715),(495,715)),((875,715),(930,715)),((1310,715),(1365,715))]:arrow(d,a,b)
arrow(d,(1120,445),(1120,520));arrow(d,(1120,520),(690,520));arrow(d,(690,520),(690,610))
d.text((800,550),'把离线构造的数据交给签名程序处理',font=ft(24),fill=GRAY)
d.text((73,925),'概念图：具体固件、ROP 和目标寄存器随芯片变化。依据 Canary 论文与 CMP 社区资料重绘。',font=ft(24),fill=GRAY)
im.save(OUT/'gpu-unlock-security-path.png')

im,d=page('把解锁放到驱动绑定之前','同一段固件事务，可以由 Linux 启动早期模块或 UEFI 应用承载。',940)
box(d,(70,220,500,195),'Linux / initramfs',['内核启动后、nvidia 绑定前','模块选择目标卡并运行解锁'])
box(d,(70,520,500,195),'Windows / UEFI',['Windows Boot Manager 之前','EFI 应用完成解锁并直接交接'])
box(d,(750,285,975,350),'对应芯片的解锁事务',['准备固件、显存布局和启动状态','运行 Booter 利用 → 打开权限 → 写速率','恢复正常引用 → FLR → 验证交接状态','原版 NVIDIA 驱动接管显卡'])
arrow(d,(580,315),(735,390));arrow(d,(580,615),(735,550))
d.text((75,815),'跨 FLR 或驱动重载保留状态，不代表完整断电后仍有效；唤醒和休眠恢复也需单独实现。',font=ft(25),fill=GRAY)
im.save(OUT/'gpu-unlock-pre-driver.png')

pdf=HERE/'source/canary-in-the-crypto-mine.pdf'
for n,bounds,name in [(6,(135,145,1035,890),'gpu-unlock-canary-layout'),(7,(135,145,1035,465),'gpu-unlock-canary-overflow'),(9,(135,140,1035,635),'gpu-unlock-paper-flow')]:
    prefix=BUILD/f'paper-page-{n}'
    subprocess.run(['pdftoppm','-f',str(n),'-l',str(n),'-scale-to','2400','-singlefile','-png',str(pdf),str(prefix)],check=True)
    Image.open(str(prefix)+'.png').crop(tuple(round(x*1.6) for x in bounds)).save(OUT/(name+'.png'))

data=json.loads((HERE/'source/measurements.json').read_text());v=data['ampere'];fig,ax=plt.subplots(figsize=(11.6,6.5));y=np.arange(3)
ax.barh(y+.17,v['locked'],.30,color=GRAY,label='锁定');ax.barh(y-.17,v['unlocked'],.30,color=GREEN,label='解锁')
for i,(a,b) in enumerate(zip(v['locked'],v['unlocked'])):
    ax.text(a+2,i+.17,f'{a:.2f}',va='center');ax.text(b+2,i-.17,f'{b:.2f}（+{(b/a-1)*100:.1f}%）',va='center',color=GREEN)
ax.set(yticks=y,yticklabels=v['labels'],xlim=(0,218),xlabel='TFLOPS，dense GEMM 实测');ax.invert_yaxis();ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('RTX 3090 Ti：实际 GEMM 的收益取决于计算路径',fontsize=19,x=.045,ha='left');fig.text(.045,.89,'450 W · 8192³ · FP32 累加 · 五次样本中位数',color=GRAY);fig.text(.045,.025,'按 Zagitta/gpu-unlock 公开结果重绘；数据来自项目作者，本文未复测。',fontsize=11,color=GRAY);fig.tight_layout(rect=(0,.06,1,.86));fig.savefig(OUT/'gpu-unlock-3090ti-gemm.png',dpi=180);fig.savefig(EXPORT/'3090ti-gemm.svg');plt.close(fig)

v=data['blackwell'];fig,ax=plt.subplots(figsize=(12.8,7));x=np.arange(4);w=.24
for j,(key,label,color) in enumerate([('rtx5090','RTX 5090',GRAY),('pro6000_maxq','PRO 6000 Max-Q',BLUE),('pro6000_workstation','PRO 6000 Workstation',GREEN)]):
    bars=ax.bar(x+(j-1)*w,v[key],w,label=label,color=color)
    for bar,value in zip(bars,v[key]):ax.text(bar.get_x()+w/2,value+12,f'{value:g}',ha='center',fontsize=10)
ax.set(xticks=x,xticklabels=v['labels'],ylabel='理论稠密 Tensor 峰值（TFLOPS）',ylim=(0,1170));ax.legend(ncol=3,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('RTX 5090 与 RTX PRO 6000：不同累加路径的差距',fontsize=19,x=.05,ha='left');fig.text(.05,.90,'按官方 boost 频率列出的规格；未使用稀疏倍增指标，非应用实测。',color=GRAY);fig.text(.05,.025,'来源：NVIDIA RTX Blackwell 与 RTX PRO Blackwell 架构白皮书 v1.1，附录规格表。',fontsize=11,color=GRAY);fig.tight_layout(rect=(0,.07,1,.85));fig.savefig(OUT/'gpu-unlock-blackwell-specs.png',dpi=180);fig.savefig(EXPORT/'blackwell-specs.svg');plt.close(fig)
print('Rendered 7 figures to',OUT)
