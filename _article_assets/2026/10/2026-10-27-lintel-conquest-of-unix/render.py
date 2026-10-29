#!/usr/bin/env python3
"""Rebuild every image used by the Lintel Conquest of UNIX article.

Two kinds of output:

1. Video frames  -- already cropped from the 1080p source, only re-encoded to PNG.
2. Drawn figures -- redrawn with matplotlib from the numbers quoted in the video.

Run from anywhere:

    uv run --isolated --no-project --with matplotlib --with pillow python \
      _article_assets/2026/10/2026-10-27-lintel-conquest-of-unix/render.py

The script resolves the blog repository from its own location, so it does not
depend on the working directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
IMG_DIR = REPO / "img" / "2026" / "10" / "27"
FRAMES = HERE / "source" / "frames"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "PingFang SC",
            "Hiragino Sans GB",
            "Heiti SC",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    }
)

BLUE = "#2E5FA3"
LIGHT_BLUE = "#9db8d8"
GREEN = "#2E8B57"
DARK_GREEN = "#1f6b45"
ORANGE = "#D07A2D"
RED = "#B03A2E"
GREY = "#6b7280"

# name in source/frames  ->  published file name
FRAME_MAP = {
    "fig02-vendor-vertical": "lintel-hp-parisc-hpux.jpg",
    "fig03-nt-sales-9394": "lintel-nt-sales-1993-1994.jpg",
    "fig04-nt-sales-9396": "lintel-nt-sales-1993-1996.jpg",
    "fig05-spec-benchmark": "lintel-specint95-pentium-pro.jpg",
    "fig06-shv-p6": "lintel-intel-p6-servers-infoworld.jpg",
    "fig07-nsp": "lintel-intel-nsp-infoworld.jpg",
    "fig08-nec-nt-portable": "lintel-nec-nt-mips-ad.jpg",
    "fig09-pentiumpro-16bit": "lintel-pentium-pro-16bit-headline.jpg",
    "fig10-symbolics-lmi": "lintel-symbolics-lmi.jpg",
    "fig11-kernel-diagram": "lintel-kernel-diagram.jpg",
    "fig12-gpl-defining": "lintel-torvalds-gpl-cio.jpg",
    "fig13-redhat-box": "lintel-redhat-box.jpg",
    "fig14-linux-price-clip": "lintel-linux-philadelphia-inquirer.jpg",
    "fig15-oracle-linux": "lintel-oracle-linux-cnet.jpg",
    "fig16-future-of-linux": "lintel-future-of-linux-panel.jpg",
    "fig17-intel-pressrelease": "lintel-intel-redhat-press-release.jpg",
    "fig18-dell-redhat": "lintel-dell-redhat-press-release.jpg",
    "fig19-halloween": "lintel-halloween-documents.jpg",
    "fig20-mindcraft": "lintel-mindcraft-benchmark.jpg",
    "fig21-xeon-8way": "lintel-xeon-8way-profusion.jpg",
    "fig22-serveros-share": "lintel-server-os-share-1995-2000.jpg",
    "fig23-etrade": "lintel-etrade-sun-e4500.jpg",
    "fig24-amazon-linux": "lintel-amazon-linux-2001.jpg",
    "fig25-linux24": "lintel-linux-2.4-announcement.jpg",
    "fig26-opteron": "lintel-amd-opteron.jpg",
    "fig27-linux26": "lintel-linux-2.6-register.jpg",
    "fig28-activedirectory": "lintel-active-directory.jpg",
    "fig29-gartner-decline": "lintel-unix-install-base-gartner.jpg",
    "fig30-linuxmyths": "lintel-linux-myths-webpage.jpg",
}


# Some frames are full-screen web captures whose body text is too small at blog
# column width. These are re-cropped to keep only the part that still reads
# (headline plus the first paragraphs). Keys refer to source/frames/<key>.jpg,
# tuple is (published file name, crop box in the 1920x1080 frame).
CROPS = {
    "fig10-symbolics-lmi": ("lintel-symbolics-lmi.jpg", (100, 390, 1800, 670)),
    "fig12-gpl-defining": ("lintel-torvalds-gpl-cio.jpg", (270, 260, 1660, 660)),
    "fig15-oracle-linux": ("lintel-oracle-linux-cnet.jpg", (500, 60, 1420, 900)),
    "fig16-future-of-linux": ("lintel-future-of-linux-panel.jpg", (556, 60, 1230, 620)),
    "fig17-intel-pressrelease": ("lintel-intel-redhat-press-release.jpg", (596, 20, 1330, 790)),
    "fig18-dell-redhat": ("lintel-dell-redhat-press-release.jpg", (40, 210, 1560, 480)),
    "fig19-halloween": ("lintel-halloween-documents.jpg", (85, 100, 1720, 330)),
    "fig30-linuxmyths": ("lintel-linux-myths-webpage.jpg", (100, 90, 1920, 1040)),
    "fig24-amazon-linux": ("lintel-amazon-linux-2001.jpg", (30, 240, 1900, 700)),
    "fig25-linux24": ("lintel-linux-2.4-announcement.jpg", (320, 90, 1600, 620)),
    "fig27-linux26": ("lintel-linux-2.6-register.jpg", (40, 180, 1500, 820)),
}


def trim_dark_border(image: Image.Image, threshold: int = 22, pad: int = 8) -> Image.Image:
    """Drop the black letterbox around archive footage so the figure is not mostly black."""
    grey = image.convert("L").point(lambda p: 255 if p > threshold else 0)
    box = grey.getbbox()
    if not box:
        return image
    left, top, right, bottom = box
    if (right - left) * (bottom - top) < 0.25 * image.width * image.height:
        return image
    return image.crop(
        (
            max(0, left - pad),
            max(0, top - pad),
            min(image.width, right + pad),
            min(image.height, bottom + pad),
        )
    )


def export_frames() -> None:
    for src, dst in FRAME_MAP.items():
        path = FRAMES / f"{src}.jpg"
        if not path.exists():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        if src in CROPS:
            _, box = CROPS[src]
            image = image.crop(box)
        else:
            image = trim_dark_border(image)
        image.save(IMG_DIR / dst, quality=94)
    cover = HERE / "source" / "cover.jpg"
    if cover.exists():
        Image.open(cover).convert("RGB").save(IMG_DIR / "lintel-cover.jpg", quality=92)


def fig_timeline() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    events = [
        (1983.8, "GNU 项目启动", 1),
        (1991.6, "Linux 0.01\n首个可用版本", -1),
        (1993.5, "Windows NT 3.1\nRed Hat 创立", 1),
        (1995.9, "Pentium Pro\nNT 4.0", -1),
        (1998.5, "“Linux 之夏”\nIntel 投资 Red Hat", 1),
        (2001.1, "Linux 2.4：\nSMP 补课", -1),
        (2003.4, "Opteron / x86-64\nLinux 2.6", 1),
        (2005.0, "Windows Server\n收入追平 UNIX", -1),
    ]
    ax.axhline(0, color="#444", lw=1.4)
    for x, label, side in events:
        ax.plot([x], [0], marker="o", ms=7, color=BLUE, zorder=3)
        ax.plot([x, x], [0, side * 0.55], color="#999", lw=0.9)
        ax.text(
            x,
            side * 0.66,
            label,
            ha="center",
            va="bottom" if side > 0 else "top",
            fontsize=8.2,
        )
    ax.set_xlim(1982, 2007)
    ax.set_ylim(-1.35, 1.35)
    ax.set_yticks([])
    ax.set_xlabel("年份", fontsize=9)
    ax.set_xticks([1984, 1988, 1992, 1996, 2000, 2004])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("从 GNU 到 Lintel：一条 20 年的替代链", fontsize=11)
    fig.savefig(IMG_DIR / "lintel-timeline.png")
    plt.close(fig)


def fig_flywheel() -> None:
    """High-volume flywheel, five boxes on a circle with curved arrows."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = [
        (90, "高销量\n（PC 市场规模）"),
        (18, "更高良率\n（同样晶圆出更多好芯片）"),
        (-54, "更低单位成本"),
        (-126, "更高利润与现金流"),
        (162, "再投资：更先进制程"),
    ]
    radius = 1.0
    for angle, label in nodes:
        rad = np.deg2rad(angle)
        x, y = radius * np.cos(rad), radius * np.sin(rad)
        box = FancyBboxPatch(
            (x - 0.34, y - 0.115),
            0.68,
            0.23,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.0,
            facecolor="#eaf0f8",
            edgecolor=BLUE,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, zorder=4)

    for i in range(5):
        a0 = np.deg2rad(nodes[i][0] - 13)
        a1 = np.deg2rad(nodes[(i + 1) % 5][0] + 13)
        p0 = (radius * np.cos(a0), radius * np.sin(a0))
        p1 = (radius * np.cos(a1), radius * np.sin(a1))
        arrow = FancyArrowPatch(
            p0,
            p1,
            connectionstyle="arc3,rad=-0.18",
            arrowstyle="-|>,head_width=3.2,head_length=6",
            linewidth=1.6,
            color="#3b62a0",
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
        ax.add_patch(arrow)
    fig.savefig(IMG_DIR / "lintel-flywheel.png")
    plt.close(fig)


def fig_price_ladder() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    labels = [
        "Red Hat 盒装发行版\n（1997 年前后）",
        "Windows NT 许可\n（单份升级 → 10 座位）",
        "UNIX 整机系统\n（软硬件一体）",
    ]
    lo = [50, 150, 25_000]
    hi = [50, 1_100, 250_000]
    y = np.arange(len(labels))[::-1]
    for i, (a, b) in enumerate(zip(lo, hi)):
        yy = y[i]
        if a == b:
            ax.plot([a], [yy], marker="o", ms=9, color=GREEN)
        else:
            ax.plot([a, b], [yy, yy], lw=9, solid_capstyle="round", color=[ORANGE, RED][i - 1])
            ax.plot([a, b], [yy, yy], "|", ms=14, color="white")
        text = f"\\${a:,}" if a == b else f"\\${a:,} – \\${b:,}"
        ax.text(b * 1.35, yy, text, va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(30, 2_000_000)
    ax.set_xlabel("价格（美元，对数刻度）", fontsize=9)
    ax.set_title("同一台服务器上，操作系统的价格差跨越三个数量级", fontsize=10.5)
    ax.grid(axis="x", ls=":", color="#bbb", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"\\${v:,.0f}"))
    fig.savefig(IMG_DIR / "lintel-price-ladder.png")
    plt.close(fig)


def fig_cost_per_user() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    names = [
        "Sun Enterprise 4500\n(\\$250,000，300–400 用户)",
        "Linux + Intel 单机\n(\\$4,000，180 用户)",
        "Linux + Intel 两台\n(\\$8,000，360 用户)",
    ]
    vals = [250_000 / 350, 4_000 / 180, 8_000 / 360]
    bars = ax.bar(names, vals, color=[ORANGE, GREEN, BLUE], width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 18, f"\\${v:,.0f}/用户", ha="center", fontsize=9)
    ax.set_ylabel("每用户硬件成本\n（美元）", fontsize=8.5)
    ax.set_title("scale-out：用两台廉价机器对标一台小型机", fontsize=10.5)
    ax.set_ylim(0, 820)
    ax.grid(axis="y", ls=":", color="#bbb", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.savefig(IMG_DIR / "lintel-cost-per-user.png")
    plt.close(fig)


def fig_smp() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    items = [
        ("UNIX 变体（1990 年代）", 4, 64, ORANGE, "4–64 路"),
        ("Windows NT 4.0（1996）", 4, 8, BLUE, "4–8 路"),
        ("Windows 2000（2000）", 4, 32, LIGHT_BLUE, "最高 32 路"),
        ("Linux 2.2（1999）", 1, 2, GREEN, "仅 2 路"),
        ("Linux 2.4（2001）", 4, 32, "#3fae7a", "4–32 路"),
        ("Linux 2.6（2003）", 4, 64, DARK_GREEN, "最高 64 路"),
    ]
    y = np.arange(len(items))[::-1]
    for i, (_, a, b, color, label) in enumerate(items):
        ax.barh(y[i], b - a, left=a, color=color, height=0.6)
        ax.text(b + 1.2, y[i], label, va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([item[0] for item in items], fontsize=9)
    ax.set_xlim(0, 74)
    ax.set_xlabel("单个 OS 实例可支持的最大处理器数", fontsize=9)
    ax.set_title("多路 SMP 支持：NT 的天花板与 Linux 的补课", fontsize=10.5)
    ax.grid(axis="x", ls=":", color="#bbb", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(IMG_DIR / "lintel-smp-support.png")
    plt.close(fig)


def fig_address_space() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch((0.16, 0.58), 0.80, 0.26, boxstyle="square,pad=0",
                       facecolor="#eaf0f8", edgecolor="#3b62a0", lw=1.2)
    )
    ax.text(0.56, 0.71, "32 位地址空间：$2^{32}$ B = 4 GB", ha="center", va="center", fontsize=10)

    ax.add_patch(
        FancyBboxPatch((0.16, 0.14), 0.80, 0.26, boxstyle="square,pad=0",
                       facecolor="#e9f4ec", edgecolor=GREEN, lw=1.2)
    )
    ax.text(0.56, 0.27, "64 位地址空间：$2^{64}$ B（受硬件限制，实际可达数十 TB）",
            ha="center", va="center", fontsize=10)

    ax.text(0.13, 0.71, "x86", ha="right", va="center", fontsize=10)
    ax.text(0.13, 0.27, "x86-64", ha="right", va="center", fontsize=10)
    fig.savefig(IMG_DIR / "lintel-address-space.png")
    plt.close(fig)


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    export_frames()
    fig_timeline()
    fig_flywheel()
    fig_price_ladder()
    fig_cost_per_user()
    fig_smp()
    fig_address_space()
    made = sorted(p.name for p in IMG_DIR.glob("lintel-*"))
    print(f"{len(made)} images in {IMG_DIR}")
    for name in made:
        print(" ", name)


if __name__ == "__main__":
    main()
