from pathlib import Path
import random
ROOT=Path(__file__).resolve().parents[1]
PRE=r'''\documentclass[tikz,border=10pt]{standalone}
\usepackage[UTF8,fontset=fandol]{ctex}
\usepackage{amsmath,amssymb}
\usetikzlibrary{arrows.meta,backgrounds}
\definecolor{teal}{HTML}{4F8FA5}
\definecolor{orange}{HTML}{EE995B}
\definecolor{coral}{HTML}{C95B5B}
\definecolor{violet}{HTML}{8A74B5}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,>=Stealth,every node/.style={inner sep=2pt},arr/.style={->,black!60,line width=.7pt}]
'''
def txt(x,y,s,style=''):
 return rf'\node[{style}] at ({x},{y}) {{{s}}};'+'\n'
def mat(x,y,r,c,col,sym,shape):
 # One cell pitch for all axes: Br=3, Bc=5, d=4, singleton=1.
 w,h=c*.42,r*.42; out=''; rng=random.Random(sym)
 for a in range(r):
  for b in range(c):
   k=rng.choice([30,55,80]); xx=x-w/2+b*.42; yy=y-h/2+a*.42
   out+=rf'\fill[{col}!{k},rounded corners=.6pt] ({xx+.018},{yy+.018}) rectangle ({xx+.402},{yy+.402});'+'\n'
 out+=rf'\draw[black!55,line width=.5pt] ({x-w/2-.035},{y-h/2-.035}) rectangle ({x+w/2+.035},{y+h/2+.035});'+'\n'
 out+=txt(x,y-h/2-.32,'$'+sym+'$',r'font=\small')
 out+=txt(x,y-h/2-.78,'$'+shape+'$',r'font=\scriptsize,text=black!65')
 return out

def arrow(x1,y1,x2,y2,label=None):
 out=rf'\draw[arr] ({x1},{y1}) -- ({x2},{y2});'+'\n'
 if label: out+=txt((x1+x2)/2,y1+.42,label,r'font=\small,text=black!65')
 return out

def footer(y,rows):
 out=rf'\fill[black!3,rounded corners=2pt] (-.1,{y+.4}) rectangle (18.5,{y-1.7});'+'\n'
 for k,(label,body) in enumerate(rows):
  out+=txt(.2,y-.62*k,label,r'anchor=west,font=\small\bfseries,text=black!65')
  out+=txt(1.6,y-.62*k,body,r'anchor=west,font=\small')
 return out

s=PRE
s+=txt(9.2,1.6,r'$S_{ij}=Q_iK_j^{\mathsf T}/\sqrt d,\qquad \widetilde P_{ij}=\exp(S_{ij}-m_i^{\rm new})$',r'font=\large')
s+=txt(9.2,.65,'（1） 局部分数：沿 $d$ 收缩；输入块由 HBM 搬入片上',r'font=\small\bfseries,text=black!65')
s+=mat(1.3,-1,3,4,'teal','Q_i',r'B_r\times d')+txt(2.85,-1,r'$\times$',r'font=\Large')
s+=mat(4.6,-1,4,5,'teal',r'K_j^{\mathsf T}',r'd\times B_c')
s+=arrow(6.1,-1,8,-1,r'$1/\sqrt d$')
s+=mat(9.5,-1,3,5,'orange','S_{ij}',r'B_r\times B_c')
s+=arrow(11,-1,14.6,-1,'online softmax')
s+=mat(16.3,-1,3,5,'orange',r'\widetilde P_{ij}',r'B_r\times B_c')
s+=txt(9.2,-3.65,'（2） 加权累积：沿 $B_c$ 收缩；沿用上排生成的权重',r'font=\small\bfseries,text=black!65')
s+=mat(1.3,-5.1,3,5,'orange',r'\widetilde P_{ij}',r'B_r\times B_c')+txt(2.9,-5.1,r'$\times$',r'font=\Large')
s+=mat(4.6,-5.1,5,4,'violet','V_j',r'B_c\times d')
s+=arrow(5.9,-5.1,8.1,-5.1,'矩阵乘')
s+=mat(9.5,-5.1,3,4,'coral',r'\Delta U_i',r'B_r\times d')
s+=arrow(10.8,-5.1,14.8,-5.1,r'$+\;\alpha_i\odot U_i^{\rm old}$')
s+=mat(16.3,-5.1,3,4,'coral',r'U_i^{\rm new}',r'B_r\times d')
s+=r'\draw[arr] (17.65,-1) -- (18.2,-1) -- (18.2,-2.98) -- (-.6,-2.98) -- (-.6,-5.1) -- (.1,-5.1);'+'\n'
s+=txt(9.2,-7.4,r'继续下一个 $j$：保留 $(m_i,\ell_i,U_i)$，复用 $Q_i$，载入下一块 $K_j,V_j$',r'font=\small')
s+=txt(9.2,-8.1,r'所有 $K/V$ 块处理完 $\longrightarrow\quad O_i=U_i/\ell_i\quad\longrightarrow$ 写回 HBM',r'font=\small')
s+=footer(-9.1,[('轴',r'$B_r$：query 块行数；$B_c$：key/value 块行数；$d$：单头特征维。格数仅示意轴结构。'),('对象',r'$S$ 是分数，$\widetilde P$ 是非负未归一化权重；$U$ 是加权分子，尚未除以分母。'),('片上',r'$S_{ij},\widetilde P_{ij}$ 用完即丢弃；$(m_i,\ell_i,U_i)$ 跨块累积。图示 FA-2 风格扫描。')])
s+='\\end{tikzpicture}\n\\end{document}\n'
(ROOT/'build/tile-chain.tex').write_text(s)

s=PRE
s+=txt(9.2,1.6,r'$m_i^{\rm new}=\max(m_i^{\rm old},\operatorname{rowmax}S_{ij}),\qquad \alpha_i=e^{m_i^{\rm old}-m_i^{\rm new}}$',r'font=\large')
s+=txt(9.2,.65,'（1） 更新每行最大值：把旧状态和新块对齐到同一指数基准',r'font=\small\bfseries,text=black!65')
s+=mat(1.4,-1,3,5,'orange','S_{ij}',r'B_r\times B_c')
s+=arrow(2.8,-1,5,-1,r'$\mathrm{rowmax}$')
s+=mat(5.7,-1,3,1,'orange',r'\widehat m_i',r'B_r\times1')
s+=arrow(6.4,-1,10.1,-1,r'$\max(\widehat m_i,m_i^{\rm old})$')
s+=mat(10.8,-1,3,1,'orange',r'm_i^{\rm new}',r'B_r\times1')
s+=arrow(11.5,-1,15.8,-1,r'$\exp(m_i^{\rm old}-m_i^{\rm new})$')
s+=mat(16.5,-1,3,1,'coral',r'\alpha_i',r'B_r\times1')
s+=txt(9.2,-3.3,'（2） 沿用上排分数：减新基准、指数化，再沿 key 轴求和',r'font=\small\bfseries,text=black!65')
s+=mat(1.4,-5,3,5,'orange','S_{ij}',r'B_r\times B_c')
s+=arrow(2.8,-5,7.6,-5,r'$\exp(S_{ij}-m_i^{\rm new})$')
s+=mat(9,-5,3,5,'orange',r'\widetilde P_{ij}',r'B_r\times B_c')
s+=arrow(10.4,-5,15.8,-5,r'$\mathrm{rowsum}$')
s+=mat(16.5,-5,3,1,'violet',r'\Delta\ell_i',r'B_r\times1')
s+=r'\draw[arr] (.15,-1) -- (-.6,-1) -- (-.6,-5) -- (.15,-5);'+'\n'
s+=txt(9.2,-7.15,r'（3） 合并旧状态与新贡献（$\odot$ 及除法均按行广播）',r'font=\small\bfseries,text=black!65')
s+=txt(9.2,-7.9,r'$\ell_i^{\rm new}=\alpha_i\odot\ell_i^{\rm old}+\Delta\ell_i,\qquad U_i^{\rm new}=\alpha_i\odot U_i^{\rm old}+\widetilde P_{ij}V_j$',r'font=\large')
s+=txt(9.2,-8.65,r'新 $(m_i,\ell_i,U_i)$ $\longrightarrow$ 下一块的旧状态；扫描结束 $\longrightarrow$ $O_i=U_i/\ell_i$',r'font=\small')
s+=footer(-9.65,[('轴',r'每行对应一个 query；$\mathrm{rowmax}/\mathrm{rowsum}$ 沿 $B_c$ 归约，结果保留单列。'),('对象',r'$m_i$：最大分数；$\ell_i$：指数和；$\alpha_i\in[0,1]$：旧贡献缩放系数；均为逐行浮点量。'),('机制',r'初始 $m_i=-\infty,\ \ell_i=0,\ U_i=0$；图示无 mask、无 dropout，首个非空块取 $\alpha_i=0$。')])
s+='\\end{tikzpicture}\n\\end{document}\n'
(ROOT/'build/online-state.tex').write_text(s)
