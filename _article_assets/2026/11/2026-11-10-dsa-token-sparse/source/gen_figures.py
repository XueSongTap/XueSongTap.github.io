#!/usr/bin/env python3
"""Generate the TikZ sources of the DSA figures; small tensors are really computed."""
import math
import random
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

PREAMBLE = r"""\documentclass[tikz,border=7pt]{standalone}
\usepackage[UTF8,fontset=fandol]{ctex}
\usepackage{amsmath,amssymb}
\usetikzlibrary{arrows.meta,calc}
\pgfdeclarelayer{background}\pgfdeclarelayer{foreground}
\pgfsetlayers{background,main,foreground}
\definecolor{teal}{HTML}{4F8FA5}
\definecolor{orange}{HTML}{EE995B}
\definecolor{coral}{HTML}{C95B5B}
\definecolor{violet}{HTML}{8A74B5}
\definecolor{neutral}{HTML}{85898F}
\begin{document}
\begin{tikzpicture}[line cap=round,line join=round]
"""
LV = (32, 58, 85)
BR = 0.20  # horizontal room taken by each bracket
SLOT = "draw=black!55,line width=.4pt"
SEL = "draw=black!80,line width=.9pt"
FLOW = "black!55,line width=.65pt"


def levels(vals):
    """Map values to 0/1/2 by tertile."""
    s = sorted(vals)
    a, b = s[len(s) // 3], s[2 * len(s) // 3]
    return lambda v: 0 if v < a else (1 if v < b else 2)


def flat(mat):
    return [x for r in mat for x in r]


class Fig:
    def __init__(self, u, width):
        self.u, self.W, self.body, self.pos = u, width, [], {}

    def tile(self, x, y, fill, text="", border=None, tc="black!80"):
        u, g = self.u, 0.022
        st = f"fill={fill}" if fill else "fill=white"
        dr = border or ("draw=black!16,line width=.3pt" if not fill else "draw=none")
        self.body.append(f"\\draw[{st},{dr},rounded corners=.7pt] ({x+g:.3f},{y-g:.3f}) rectangle ({x+u-g:.3f},{y-u+g:.3f});")
        if text != "":
            self.body.append(f"\\begin{{pgfonlayer}}{{foreground}}\\node[text={tc},font=\\scriptsize,inner sep=0] at ({x+u/2:.3f},{y-u/2:.3f}) {{{text}}};\\end{{pgfonlayer}}")

    def bracket(self, x0, x1, y0, y1):
        o, s, c = 0.08, 0.08, "black!60,line width=.55pt"
        self.body.append(f"\\draw[{c}] ({x0-o+s:.3f},{y0+o:.3f})--({x0-o:.3f},{y0+o:.3f})--({x0-o:.3f},{y1-o:.3f})--({x0-o+s:.3f},{y1-o:.3f});")
        self.body.append(f"\\draw[{c}] ({x1+o-s:.3f},{y0+o:.3f})--({x1+o:.3f},{y0+o:.3f})--({x1+o:.3f},{y1-o:.3f})--({x1+o-s:.3f},{y1-o:.3f});")

    def row(self, heading, items, top, after=1.8):
        """Lay one horizontal algebraic stage; return y of the next stage heading."""
        u = self.u
        height = max(i["r"] * u for i in items if i["k"] == "m")
        total = sum(i["c"] * u + 2 * BR if i["k"] == "m" else i["w"] for i in items)
        x = (self.W - total) / 2
        ycenter = top - 0.95 - height / 2
        ybot = ycenter - height / 2
        self.last_bottom = ybot
        self.body.append(f"\\node[font=\\small\\bfseries,text=black!62,anchor=north] at ({self.W/2:.3f},{top:.3f}) {{{heading}}};")
        for it in items:
            if it["k"] == "m":
                w, h = it["c"] * u, it["r"] * u
                x0, y0 = x + BR, ycenter + h / 2
                for r in range(it["r"]):
                    for c in range(it["c"]):
                        f, txt, border, tc = it["f"](r, c)
                        self.tile(x0 + c * u, y0 - r * u, f, txt, border, tc)
                self.bracket(x0, x0 + w, y0, y0 - h)
                self.pos[it["id"]] = (x0, y0, w, h)
                cx = x0 + w / 2
                self.body.append(f"\\node[font=\\small,anchor=north,inner sep=0] at ({cx:.3f},{ybot-.2:.3f}) {{{it['s']}}};")
                self.body.append(f"\\node[font=\\scriptsize,text=black!58,anchor=north,inner sep=0] at ({cx:.3f},{ybot-.66:.3f}) {{{it['d']}}};")
                x += w + 2 * BR
            elif it["k"] == "op":
                self.body.append(f"\\node[font=\\Large] at ({x+it['w']/2:.3f},{ycenter:.3f}) {{{it['t']}}};")
                x += it["w"]
            elif it["k"] == "ar":
                self.body.append(f"\\draw[-{{Stealth[length=5pt]}},black!62,line width=.6pt] ({x+.12:.3f},{ycenter:.3f})--({x+it['w']-.12:.3f},{ycenter:.3f});")
                self.body.append(f"\\node[font=\\footnotesize,text=black!68,anchor=south,inner sep=0] at ({x+it['w']/2:.3f},{ycenter+.11:.3f}) {{{it['t']}}};")
                x += it["w"]
            elif it["k"] == "gap":
                if it.get("t"):
                    self.body.append(f"\\node[font=\\footnotesize,text=black!68,anchor=south,inner sep=0] at ({x+it['w']/2:.3f},{ycenter+height/2+.1:.3f}) {{{it['t']}}};")
                x += it["w"]
        return ybot - after

    def link(self, x0, y0, x1, y1):
        cx = (x0 + x1) / 2
        self.body.append(f"\\begin{{pgfonlayer}}{{background}}\\draw[black!42,line width=.55pt] ({x0:.3f},{y0:.3f}) .. controls ({cx:.3f},{y0:.3f}) and ({cx:.3f},{y1:.3f}) .. ({x1:.3f},{y1:.3f});\\end{{pgfonlayer}}")

    def bus(self, src, offs, targets, ybus, label, label_x=None):
        """Route one tensor from the bottom of src into the tops of later-stage targets."""
        x0, y0, w, h = self.pos[src]
        xs = x0 + w - offs
        ends = [(self.pos[t][0] + self.pos[t][2] * f, self.pos[t][1] + .12) for t, f in targets]
        all_x = [xs] + [e[0] for e in ends]
        layer = "\\begin{pgfonlayer}{background}"
        out = [f"{layer}\\draw[{FLOW}] ({xs:.3f},{y0-h-.11:.3f})--({xs:.3f},{ybus:.3f});"]
        out.append(f"\\draw[{FLOW}] ({min(all_x):.3f},{ybus:.3f})--({max(all_x):.3f},{ybus:.3f});")
        for ex, ey in ends:
            out.append(f"\\draw[-{{Stealth[length=5pt]}},{FLOW}] ({ex:.3f},{ybus:.3f})--({ex:.3f},{ey:.3f});")
        out.append("\\end{pgfonlayer}")
        self.body.append("".join(out))
        lx = label_x if label_x is not None else (min(all_x) + max(all_x)) / 2
        self.body.append(f"\\begin{{pgfonlayer}}{{foreground}}\\node[fill=white,font=\\footnotesize,text=black!68,inner sep=2pt] at ({lx:.3f},{ybus:.3f}) {{{label}}};\\end{{pgfonlayer}}")

    def write(self, name, formula, rows_box, build, top_gap=1.25):
        y = build(-top_gap)
        tw = self.W - 2.1
        rows = r" \\[.55em] ".join(f"\\textbf{{{a}}} & {b}" for a, b in rows_box)
        box = (f"\\node[anchor=north,fill=black!3,inner xsep=9pt,inner ysep=8pt,rounded corners=1pt,font=\\small] at ({self.W/2:.3f},{y:.3f}) "
               f"{{\\begin{{tabular}}{{@{{}}p{{1.05cm}}@{{}}p{{{tw:.2f}cm}}@{{}}}}{rows}\\end{{tabular}}}};")
        form = f"\\node[anchor=north,font=\\large] at ({self.W/2:.3f},0) {{{formula}}};"
        (OUT / f"{name}.tex").write_text(PREAMBLE + form + "\n" + "\n".join(self.body) + "\n" + box + "\n\\end{tikzpicture}\n\\end{document}\n", encoding="utf-8")


def m(id_, r, c, f, s, d):
    return dict(k="m", id=id_, r=r, c=c, f=f, s=s, d=d)


def op(t, w=0.62):
    return dict(k="op", t=t, w=w)


def ar(t, w):
    return dict(k="ar", t=t, w=w)


def gap(w, t=""):
    return dict(k="gap", w=w, t=t)


def shade(role, mat):
    lv = levels(flat(mat))
    return lambda r, c: (f"{role}!{LV[lv(mat[r][c])]}", "", None, "black!80")


# ---------------------------------------------------------------- shared data
rng = random.Random(11)
H, D, N, K_ = 4, 5, 10, 4
gm = lambda r, c: [[rng.gauss(0, 1) for _ in range(c)] for _ in range(r)]
QI, KI = gm(H, D), gm(N, D)
P = [[sum(q[i] * k[i] for i in range(D)) for k in KI] for q in QI]
R = [[max(v, 0.0) for v in row] for row in P]
W_ = [rng.uniform(0.3, 1.0) for _ in range(H)]
I = [sum(W_[j] * R[j][s] for j in range(H)) for s in range(N)]
S = sorted(sorted(range(N), key=lambda s: -I[s])[:K_])

# ---------------------------------------------------------------- figure 1
f1 = Fig(0.42, 19.4)


def p_fill(r, c):
    pos = [v for v in flat(P) if v > 0]
    neg = [-v for v in flat(P) if v <= 0]
    v = P[r][c]
    if v > 0:
        return (f"violet!{LV[levels(pos)(v)]}", "", None, "black")
    return (f"neutral!{LV[levels(neg)(-v)]}", "", None, "black")


def r_fill(r, c):
    pos = [v for v in flat(R) if v > 0]
    v = R[r][c]
    return (f"violet!{LV[levels(pos)(v)]}", "", None, "black") if v > 0 else (None, "", None, "black")


def i_fill(r, c):
    return (f"coral!{LV[levels(I)(I[c])]}", str(c), SEL if c in S else None, "black!85")


def build1(y):
    y = f1.row(
        r"1. 每个 indexer head 与全部历史 token 打分",
        [
            m("Q", H, D, shade("orange", QI), r"$Q^I_t$", r"$H\times d$"),
            op(r"$\cdot$"),
            m("K", D, N, shade("teal", [list(c) for c in zip(*KI)]), r"$(K^I)^{\top}$", r"$d\times N$"),
            op(r"$=$"),
            m("P", H, N, p_fill, r"$A_t$", r"$H\times N$"),
            ar(r"ReLU", 1.5),
            m("R", H, N, r_fill, r"$\mathrm{ReLU}(A_t)$", r"$H\times N$"),
        ],
        y,
    )
    ybus = f1.last_bottom - 1.25
    y2 = f1.row(
        r"2. 按 head 权重汇总成一行分数，再取前 $k$ 个位置",
        [
            m("w", 1, H, shade("coral", [W_]), r"$w_t^{\top}$", r"$1\times H$"),
            op(r"$\cdot$"),
            m("R2", H, N, r_fill, r"$\mathrm{ReLU}(A_t)$", r"$H\times N$"),
            op(r"$=$"),
            m("I", 1, N, i_fill, r"$I_{t,:}$", r"$1\times N$"),
            ar(r"TopKIndices", 2.2),
            m("S", 1, K_, lambda r, c: ("white", str(S[c]), SLOT, "black!85"), r"$S_t$", r"$1\times k$"),
        ],
        y,
    )
    f1.bus("R", 0.3, [("R2", 0.18)], ybus, r"同一张量进入下一步")
    return y2


f1.write(
    "dsa-indexer-topk",
    r"$\displaystyle I_{t,s}=\sum_{j=1}^{H} w_{t,j}\,\mathrm{ReLU}\!\left(q_{t,j}^{\top}k_s\right),\qquad S_t=\mathrm{TopKIndices}\!\left(I_{t,:},\,k\right)$",
    [
        ("轴", r"$t$ 是当前 query；$j$ 是 indexer head，共 $H$ 个；$s\in\{0,\dots,N-1\}$ 是历史 token 位置；$d$ 是 indexer 维度。"),
        ("对象", r"$A_t$ 是带符号打分，灰色为负，经 ReLU 后为 0（留白）；$I_{t,:}$ 是实数分数，格内数字是位置 $s$；$S_t$ 是升序的整数地址，不是 mask。"),
        ("机制", r"$(H\times d)(d\times N)\to H\times N$，$(1\times H)(H\times N)\to 1\times N$。V3.2 的配置是 $H{=}64,\ d{=}128,\ k{=}2048$。格数只示意轴结构。"),
    ],
    build1,
    top_gap=1.75,
)

# ---------------------------------------------------------------- figure 2
f2 = Fig(0.42, 19.4)
DC, HM = 5, 4
C = gm(N, DC)
G = [C[s] for s in S]
Gt = [list(c) for c in zip(*G)]
QM = gm(HM, DC)
A = [[sum(q[i] * g[i] for i in range(DC)) for g in G] for q in QM]


def softmax(row):
    e = [math.exp(v) for v in row]
    t = sum(e)
    return [x / t for x in e]


Pm = [softmax(r) for r in A]
O = [[sum(p[r] * G[r][i] for r in range(K_)) for i in range(DC)] for p in Pm]
clv = levels(flat(C))


def c_fill(r, c):
    return (f"teal!{LV[clv(C[r][c])]}", "", None, "black") if r in S else ("teal!11", "", None, "black")


def vslots(r, c):
    return ("white", str(S[r]), SLOT, "black!85")


def g_fill(r, c):
    return (f"teal!{LV[clv(G[r][c])]}", "", None, "black")


def gt_fill(r, c):
    return (f"teal!{LV[clv(Gt[r][c])]}", "", None, "black")


def build2(y):
    y1 = f2.row(
        r"1. 用整数地址把选中的 $k$ 行 KV 读成连续张量",
        [
            m("S", K_, 1, vslots, r"$S_t$", r"$k\times 1$"),
            gap(2.4, r"地址指向行"),
            m("C", N, DC, c_fill, r"$C$", r"$N\times d_c$"),
            gap(2.8, r"Gather"),
            m("G", K_, DC, g_fill, r"$G_t$", r"$k\times d_c$"),
        ],
        y,
    )
    ybus = f2.last_bottom - 1.25
    u = f2.u
    sx, sy, sw, _ = f2.pos["S"]
    cx, cy, cw, _ = f2.pos["C"]
    gx, gy, _, _ = f2.pos["G"]
    for r, s in enumerate(S):
        f2.link(sx + sw + BR, sy - (r + 0.5) * u, cx - BR, cy - (s + 0.5) * u)
        f2.link(cx + cw + BR, cy - (s + 0.5) * u, gx - BR, gy - (r + 0.5) * u)
    y2 = f2.row(
        r"2. QK 只和选中的 $k$ 个 token 打分，PV 再用同一份 $G_t$",
        [
            m("Q", HM, DC, shade("orange", QM), r"$Q_t$", r"$H\times d_c$"),
            op(r"$\cdot$"),
            m("Gt", DC, K_, gt_fill, r"$G_t^{\top}$", r"$d_c\times k$"),
            op(r"$=$"),
            m("A", HM, K_, shade("violet", A), r"$L_t$", r"$H\times k$"),
            ar(r"Softmax", 1.7),
            m("P", HM, K_, shade("violet", Pm), r"$\Pi_t$", r"$H\times k$"),
            op(r"$\cdot$"),
            m("G2", K_, DC, g_fill, r"$G_t$", r"$k\times d_c$"),
            op(r"$=$"),
            m("O", HM, DC, shade("coral", O), r"$O_t$", r"$H\times d_c$"),
        ],
        y1,
    )
    f2.bus("G", 0.25, [("Gt", 0.5), ("G2", 0.5)], ybus, r"同一份 $G_t$，QK 取转置，PV 直接用")
    return y2


f2.write(
    "dsa-gather-attention",
    r"$\displaystyle G_t[r,:]=C\big[S_t[r],:\big],\qquad O_t=\mathrm{Softmax}\!\left(Q_tG_t^{\top}\right)G_t$",
    [
        ("轴", r"$N$ 是上下文长度；$r\in\{0,\dots,k-1\}$ 是选中槽位；$H$ 是主注意力的 query head 数；$d_c$ 是 KV 的 latent 维度。"),
        ("对象", r"$C$ 是所有 head 共享的 KV，浅色行本次不读取；$S_t$ 是整数地址；$L_t$ 是 logits；$\Pi_t$ 每行和为 1；$O_t$ 是各 head 的输出。"),
        ("机制", r"Gather 后 $N$ 轴换成 $k$ 轴，这一步不再随 $N$ 增长；所有 head 共用一份 $G_t$。V3.2：$H{=}128,\ k{=}2048$，QK 用 576 维，PV 用其中不含 RoPE 的 512 维。"),
    ],
    build2,
)
# ---------------------------------------------------------------- figure 3
N3, BLK, NBLK, WIN, KD = 32, 4, 2, 4, 12
f3 = Fig(0.27, 22.1)
rng3 = random.Random(5)
pcmp = [rng3.random() for _ in range(N3 // BLK)]
chosen = sorted(sorted(range(N3 // BLK - 1), key=lambda i: -pcmp[i])[:NBLK])
nsa_read = {b * BLK + i: "orange" for b in chosen for i in range(BLK)}
nsa_read.update({s: "teal" for s in range(N3 - WIN, N3)})
Idsa = [rng3.gauss(0, 1) for _ in range(N3)]
dsa_read = set(sorted(range(N3), key=lambda s: -Idsa[s])[:KD])
plv, ilv = levels(pcmp), levels(Idsa)


def pcmp_fill(r, c):
    return (f"violet!{LV[plv(pcmp[c])]}", "", SEL if c in chosen else None, "black")


def nsa_fill(r, c):
    return (f"{nsa_read[c]}!58", "", None, "black") if c in nsa_read else (None, "", None, "black")


def idsa_fill(r, c):
    return (f"coral!{LV[ilv(Idsa[c])]}", "", SEL if c in dsa_read else None, "black")


def dsa_fill(r, c):
    return ("orange!58", "", None, "black") if c in dsa_read else (None, "", None, "black")


def build3(y):
    y = f3.row(
        r"NSA：压缩分支做真实注意力，顺带给块打分，再读整块和末尾窗口",
        [
            gap((N3 - N3 // BLK) * f3.u),
            m("pc", 1, N3 // BLK, pcmp_fill, r"$p^{\mathrm{cmp}}_t$", r"$1\times\lfloor N/d\rfloor$"),
            ar(r"选分数最高的块，加滑窗", 3.6),
            m("rn", 1, N3, nsa_fill, r"$R^{\mathrm{NSA}}_t$", r"$1\times N$"),
        ],
        y,
        after=1.45,
    )
    y = f3.row(
        r"DSA：indexer 只负责打分，主注意力只读选中的 $k$ 个 token",
        [
            m("id", 1, N3, idsa_fill, r"$I_{t,:}$", r"$1\times N$"),
            ar(r"TopKIndices", 3.6),
            m("rd", 1, N3, dsa_fill, r"$R^{\mathrm{DSA}}_t$", r"$1\times N$"),
        ],
        y,
        after=1.45,
    )
    return y


f3.write(
    "dsa-vs-nsa-read",
    r"$\displaystyle \big|R^{\mathrm{NSA}}_t\big|=\lfloor N/d\rfloor+n\,l'+w,\qquad \big|R^{\mathrm{DSA}}_t\big|=k\ \ (\text{另扫 } N \text{ 个 index key})$",
    [
        ("轴", r"$N$ 是上下文长度；$d$ 是压缩步长；$l'$ 是选择块大小，$n$ 是选中的块数；$w$ 是滑窗长度；$k$ 是 DSA 选中的 token 数。"),
        ("对象", r"$p^{\mathrm{cmp}}_t$ 和 $I_{t,:}$ 是实数分数，黑框为选中项；$R_t$ 是主注意力读取的布尔掩码，橙色是选中的 token，蓝色是滑窗，留白表示不读。"),
        ("机制", r"NSA 的读取是连续块，但粗路径也要读 $\lfloor N/d\rfloor$ 个压缩 token；DSA 的读取分散，粗路径只读更小的 index key。论文配置：NSA 为 $d{=}16,\ l'{=}64,\ n{=}16,\ w{=}512$，V3.2 为 $k{=}2048$。格数只示意。"),
    ],
    build3,
    top_gap=1.1,
)
print("S", S)
