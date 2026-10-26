"""Render reproducible charts from Anthropic's 2026-10-26 S-1.

Run the parent render.py to update blog PNGs and SVG exports.
All monetary values in data.json are USD million unless explicitly marked.
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageOps, ImageDraw

ASSET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ASSET_ROOT.parents[3]
ROOT = ASSET_ROOT / "exports"
IMAGE_ROOT = REPO_ROOT / "img" / "2026" / "10" / "26"
BUILD_ROOT = ASSET_ROOT / "build"
for output_dir in (ROOT, IMAGE_ROOT, BUILD_ROOT):
    output_dir.mkdir(parents=True, exist_ok=True)
D = json.loads((ASSET_ROOT / "source" / "data.json").read_text())
font_paths = [Path("/System/Library/Fonts/Hiragino Sans GB.ttc"), Path("/System/Library/Fonts/PingFang.ttc")]
for path in font_paths:
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        font_name = font_manager.FontProperties(fname=str(path)).get_name()
        break
else:
    font_name = "DejaVu Sans"

BG, INK, MUTED = "#FFFFFF", "#203442", "#5C6D76"
BLUE, TEAL, RED, AMBER = "#376DAD", "#147D70", "#BF5948", "#C88D32"
plt.rcParams.update({"font.family": [font_name, "DejaVu Sans"], "font.size": 11, "axes.titlesize": 14,
                     "figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False,
                     "axes.unicode_minus": False, "svg.fonttype": "path"})

def style(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#DDE3E3", lw=.7)
    ax.tick_params(length=0, pad=8)

def header(fig, title, subtitle):
    fig.text(.075, .958, title, fontsize=21, weight="bold", va="top")
    fig.text(.075, .904, subtitle, color=MUTED, fontsize=10.5, va="top")

def save(fig, stem, note):
    fig.text(.075, .035, note, fontsize=9, color=MUTED, va="bottom")
    fig.savefig(IMAGE_ROOT / f"anthropic-s1-{stem}.png", dpi=190, facecolor=BG)
    fig.savefig(ROOT / f"{stem}.svg", facecolor=BG)
    svg_path = ROOT / f"{stem}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)

def waterfall(ax, labels, changes, final, unit_factor=1000):
    current = 0
    for i, v0 in enumerate(changes):
        value = v0 / unit_factor
        start = current
        bottom = min(current, current + value)
        ax.bar(i, abs(value), bottom=bottom, color=TEAL if value >= 0 else RED, width=.62)
        current += value
        ax.plot([i+.31, i+.69], [current, current], color="#9EAAAF", lw=.8)
        ax.annotate(f"{value:+.3f}", (i, max(start, current)), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=10, weight="bold")
    last = final / unit_factor
    ax.bar(len(changes), abs(last), bottom=min(0, last), width=.62, color=BLUE)
    ax.annotate(f"{last:+.3f}", (len(changes), max(0, last)), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(range(len(labels)), labels)
    ax.axhline(0, color=INK, lw=.8)
    style(ax)

def quarterly():
    q = D["quarterly"]
    x = np.arange(len(q["period"]))
    rev, cost, op = [np.array(q[k])/1000 for k in ("revenue", "total_operating_expenses", "operating_income")]
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 8.2), gridspec_kw={"height_ratios": [1.45, 1]})
    fig.subplots_adjust(top=.82, bottom=.13, hspace=.34, left=.09, right=.95)
    header(fig, "季度收入与营业利润", "收入与全部经营费用使用相同季度口径；金额为十亿美元，2026 年数据未经审计")
    a.plot(x, rev, "o-", lw=2.5, color=TEAL, label="营业收入")
    a.plot(x, cost, "s--", lw=2, color=BLUE, label="全部经营费用（含收入成本）")
    for i, value in enumerate(rev):
        a.annotate(f"{value:.3f}", (i, value), xytext=(0, 12 if i != 3 else -21), textcoords="offset points", ha="center", color=TEAL)
    a.set_ylim(0, 13.8); a.set_xticks(x, q["period"]); a.set_ylabel("十亿美元")
    a.legend(loc="upper left", frameon=False); style(a)
    b.bar(x, op, width=.55, color=[TEAL if v >= 0 else RED for v in op])
    for i, value in enumerate(op):
        margin = 100 * value / rev[i]
        b.text(i, value + (.11 if value >= 0 else -.14), f"{value:+.3f}\n利润率 {margin:.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    b.set_xticks(x, q["period"]); b.set_ylabel("营业利润／亏损\n十亿美元")
    b.axhline(0, color=INK, lw=.9); b.set_ylim(-4.25, 1.9); style(b)
    save(fig, "01-quarterly-revenue-operating-profit", "来源：S-1 第 158 页。2025 Q3 费用包含 15 亿美元 Bartz 和解费用。")

def income_bridge():
    d = D["h1_2026_income_bridge"]
    fig, ax = plt.subplots(figsize=(12, 7.1))
    fig.subplots_adjust(top=.80, bottom=.25, left=.09, right=.96)
    header(fig, "807 亿美元净亏损是怎样形成的", "2026 上半年利润表精确勾稽；金额为十亿美元，负数表示减少利润")
    waterfall(ax, ["营业亏损", "利息收入", "可转债等\n估值变动", "其他收支\n净额", "所得税", "GAAP\n净亏损"],
              [d["operating_income"], d["interest_income"], d["convertible_notes_derivatives_safes_fair_value_change"], d["other_income_expense_net"], d["income_tax"]], d["net_income"])
    ax.set_ylim(-91, 14); ax.set_ylabel("十亿美元")
    fig.text(.075, .115, "可转债等估值变动损失：723.88 亿美元；其他收支净额中的特定金融工具公允价值调整：87.20 亿美元。", fontsize=10, color=INK)
    save(fig, "02-net-loss-waterfall", "来源：S-1 第 22—23、157 页。估值损失不等于同额现金流出；其他收支净额并非全部为估值变动。")

def expenses():
    h = D["half_year"]
    fig, ax = plt.subplots(figsize=(12, 7.1))
    fig.subplots_adjust(top=.80, bottom=.18, left=.09, right=.96)
    header(fig, "收入如何被经营费用消耗", "2026 上半年按职能分类；金额为十亿美元，研发费用包含持续模型开发投入")
    waterfall(ax, ["营业收入", "收入成本", "研发费用", "销售营销", "管理费用", "营业亏损"],
              [h["revenue"][1], -h["cost_of_revenue"][1], -h["research_and_development"][1], -h["sales_and_marketing"][1], -h["general_and_administrative"][1]], h["operating_income"][1])
    ax.set_ylim(-2.1, 18.8); ax.set_ylabel("十亿美元")
    ax.text(.63, .77, "按收入成本计算毛利率 ≈ 85.4%\n研发费用／收入 ≈ 59.8%\nGAAP 营业利润率 ≈ −4.3%", transform=ax.transAxes, fontsize=11,
            bbox={"facecolor": BG, "edgecolor": "#DDE3E3", "pad": 10})
    save(fig, "03-operating-expense-waterfall", "来源：财务附注 8。渠道合作成本主要计入销售营销，毛利率未扣除持续研发及渠道等费用。")

def adoption():
    h, e = D["half_year"], D["enterprise_adoption"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.8, 7.2), gridspec_kw={"width_ratios": [1, 1.65]})
    fig.subplots_adjust(top=.77, bottom=.19, left=.075, right=.95, wspace=.33)
    header(fig, "按量收入占主导，大客户数量快速增长", "左图比较同为半年的收入；右图按真实日期间隔作图，保留客户数量的约数及下限含义")
    x = np.arange(2); cons=np.array(h["consumption_revenue"])/1000; subs=np.array(h["subscription_revenue"])/1000
    a.bar(x, cons, color=TEAL, width=.5, label="按使用量收费")
    a.bar(x, subs, bottom=cons, color=BLUE, width=.5, label="订阅收费")
    for i in range(2):
        total=cons[i]+subs[i]
        a.text(i, total+.5, f"{total:.3f}", ha="center", weight="bold")
    a.text(1, cons[1]/2, "80.2%\n13.064", ha="center", va="center", color="white", fontsize=12)
    a.text(1, cons[1]+subs[1]/2, "19.8%\n3.222", ha="center", va="center", color="white", fontsize=11)
    a.set_xticks(x, h["period"]); a.set_ylabel("营业收入／十亿美元"); a.set_ylim(0,20.5)
    a.legend(frameon=False, fontsize=10, loc="upper left"); style(a)
    dates=[datetime.fromisoformat(v) for v in e["date"]]
    b.plot(dates, e["reported_count"], "o-", color=TEAL, lw=2.5)
    for i,(day, count, qualifier) in enumerate(zip(dates,e["reported_count"],e["qualifier"])):
        label=("约 " if qualifier=="approximately" else "> ")+f"{count:,}"
        b.annotate(label,(day,count),xytext=(0,13 if i%2==0 else 17),textcoords="offset points",ha="center",fontsize=10)
    b.set_xticks(dates, [d.strftime("%Y.%m") for d in dates], rotation=35, ha="right")
    b.set_ylim(-450,14000); b.margins(x=.09); b.set_ylabel("TTM 贡献收入至少 10 万美元的客户数")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f"{v:,.0f}")); style(b)
    save(fig, "04-revenue-mix-customer-adoption", "来源：财务附注 2、第 142—143 页。TTM 为过去十二个月；客户数不能当作订阅人数或付费席位数。")

def cash():
    c=D["h1_2026_cash_flow"]
    fig,(a,b)=plt.subplots(1,2,figsize=(13,7.1),gridspec_kw={"width_ratios":[1,1.05]})
    fig.subplots_adjust(top=.77,bottom=.24,left=.08,right=.96,wspace=.35)
    header(fig,"经营现金流转正，投资流出主要是现金管理","2026 上半年；金额为十亿美元。右图把投资活动总流出拆开，证券净买入不等同于经营消耗")
    values=[c["operating"]/1000,c["purchase_property_equipment"]/1000,(c["operating"]+c["purchase_property_equipment"])/1000]
    bars=a.bar(range(3),values,color=[TEAL,RED,BLUE],width=.58)
    a.set_xticks(range(3),["经营现金流","购建固定资产","两项相加"],rotation=15)
    a.set_ylim(-1.3,5.3); a.set_ylabel("十亿美元"); a.axhline(0,color=INK,lw=.8); style(a)
    for bar,v in zip(bars,values):
        a.text(bar.get_x()+bar.get_width()/2,v+(.12 if v>=0 else -.12),f"{v:+.3f}",ha="center",va="bottom" if v>=0 else "top",weight="bold")
    inv=[-(c["purchase_short_term_investments"]+c["sale_maturity_short_term_investments"])/1000,-c["purchase_property_equipment"]/1000,-c["other_investing"]/1000]
    bars=b.barh([2,1,0],inv,color=[BLUE,TEAL,AMBER],height=.53)
    b.set_yticks([2,1,0],["短期投资净买入","购建固定资产","其他投资活动"])
    for bar,v in zip(bars,inv): b.text(v+.25,bar.get_y()+bar.get_height()/2,f"{v:.3f}",va="center",weight="bold")
    b.set_xlim(0,19);b.set_xlabel("现金流出／十亿美元");style(b,"x")
    fig.text(.08,.115,"期末现金、现金等价物及短期投资：1,316.51 亿美元；上半年融资活动净流入：1,107.74 亿美元。",fontsize=11,color=INK)
    save(fig,"05-cash-flow-quality","来源：现金流量表及第 162、169 页。左图两项相加是简化计算，未扣除融资租赁本金等；未来资本开支预计显著增加。")

def commitments():
    rows=D["selected_compute_commitments"]
    fig,ax=plt.subplots(figsize=(12.5,7.5))
    fig.subplots_adjust(top=.79,bottom=.20,left=.15,right=.95)
    header(fig,"已披露的部分多年算力采购与租赁承诺","全期合同规模，金额为十亿美元；期限与合同类型不同，不能视为当年费用或完整承诺总额")
    positions=np.arange(len(rows))[::-1]
    for y,row in zip(positions,rows):
        color=BLUE if row["partner"]=="Broadcom" else TEAL
        ax.barh(y,row["amount_usd_billion"],height=.40,color=color)
        ax.text(row["amount_usd_billion"]+2,y,f'{row["amount_usd_billion"]:.1f}',va="center",weight="bold",fontsize=12)
        ax.text(1,y-.34,f'{row["period"]}  |  {row["type"]}',va="center",color=MUTED,fontsize=9.5)
    ax.set_yticks(positions,[r["partner"] for r in rows]);ax.set_xlim(0,183);ax.set_ylim(-.65,3.55);ax.set_xlabel("十亿美元");style(ax,"x")
    fig.text(.075,.115,"前三项含不论实际使用量多少的最低支出安排，适用合同例外；Broadcom 为多项五年设备租赁的全期租金。",fontsize=10,color=INK)
    save(fig,"06-selected-compute-commitments","来源：S-1 第 171—172 页。仅列四项代表性安排；不含全部供应商、额外条件性容量及租赁利息。")

def customer_concentration():
    c = D["customer_concentration"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 7.6), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(top=.78, bottom=.24, left=.09, right=.96, wspace=.29)
    header(fig, "大客户占比下降，但匿名客户份额仍需分期看", "左图逐格保留披露门槛；右图仅比较两个季度，未插补 Q2，也未把九个月累计值混入季度")
    keys = list(c["reported_revenue_share_percent"])
    vals = np.array([[np.nan if v is None else v for v in c["reported_revenue_share_percent"][k]] for k in keys])
    cmap = colors.LinearSegmentedColormap.from_list("shares", ["#D9EAE6", TEAL])
    cmap.set_bad("#E8ECEB")
    a.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=10, vmax=43, aspect="auto")
    for i in range(4):
        for j in range(5):
            v = vals[i, j]
            a.text(j, i, "<10%" if np.isnan(v) else f"{v:.0f}%", ha="center", va="center",
                   color="white" if not np.isnan(v) and v >= 25 else INK, fontsize=12, weight="bold")
    a.set_xticks(range(5), c["period"], rotation=25, ha="right")
    a.set_yticks(range(4), [f"客户 {k}" for k in keys]); a.tick_params(length=0)
    a.set_title("匿名客户收入占比", loc="left", pad=15)
    a.text(0, -.23, "灰色格表示低于 10%，不能按零处理。\n2025 全年 A、B 各 12%，合计 24%。", transform=a.transAxes, color=MUTED, fontsize=10)
    top = np.array(c["top_five_derived_approx_percent"])
    outside = np.array(c["outside_top_five_approx_percent"])
    y = np.array([1, 0])
    b.barh(y, top, color=RED, height=.43, label="前五大客户")
    b.barh(y, outside, left=top, color=TEAL, height=.43, label="其余客户")
    for yi, t, o in zip(y, top, outside):
        b.text(t/2, yi, f"约 {t}%", ha="center", va="center", color="white", fontsize=12)
        b.text(t+o/2, yi, f"约 {o}%", ha="center", va="center", color="white", fontsize=12)
    b.set_yticks(y, c["top_five_quarter"]); b.set_xlim(0,100); b.set_ylim(-.65,1.65)
    b.set_xlabel("当季收入占比／%"); b.set_title("前五大客户收入占比", loc="left", pad=15)
    b.legend(loc="upper center", bbox_to_anchor=(.5,-.25), ncol=2, frameon=False, fontsize=10)
    style(b,"x")
    save(fig,"07-customer-concentration","来源：财务附注 1 F-16、第 57、143 页。客户身份未披露；2026 前九个月前五大客户约占 25%，与季度口径不同。")

def cloud_channels():
    c = D["cloud_channel"]
    fig, (a,b) = plt.subplots(1,2,figsize=(13.8,7.4),gridspec_kw={"width_ratios":[1,1.3]})
    fig.subplots_adjust(top=.78,bottom=.22,left=.08,right=.96,wspace=.30)
    header(fig,"云市场承担近一半收入销售与过半应收结算","云平台是销售和代收款渠道，不能视为这些收入均来自平台自身采购；左图金额由约数占比估算")
    revenue = np.array(D["half_year"]["revenue"])/1000
    cloud = revenue*np.array(c["half_year_revenue_share_approx_percent"])/100
    other = revenue-cloud
    a.bar(range(2),cloud,color=BLUE,width=.5,label="经云市场")
    a.bar(range(2),other,bottom=cloud,color=TEAL,width=.5,label="其他销售渠道")
    for i,(v,total) in enumerate(zip(cloud,revenue)):
        a.text(i,total+.45,f"合计 {total:.3f}",ha="center",fontsize=10)
        if i==1:
            a.text(i,v/2,f"约 {v:.2f}\n约 48%",ha="center",va="center",color="white",fontsize=12)
        else:
            a.annotate(f"云市场约 {v:.2f}（44%）",(i,v/2),xytext=(-.36,4.3),textcoords="data",ha="left",fontsize=10,
                       arrowprops={"arrowstyle":"-","color":MUTED})
    a.set_xticks(range(2),c["half_year_period"]);a.set_ylim(0,21);a.set_ylabel("半年收入／十亿美元")
    a.legend(loc="upper left",frameon=False,fontsize=10);style(a)
    x=np.arange(3);w=.32
    b.bar(x-w/2,c["revenue_share_for_corresponding_period_approx_percent"],w,color=BLUE,label="期间收入经云市场占比")
    b.bar(x+w/2,c["receivables_share_percent"],w,color=AMBER,label="期末应收来自云市场占比")
    for i,(r,ar) in enumerate(zip(c["revenue_share_for_corresponding_period_approx_percent"],c["receivables_share_percent"])):
        b.text(i-w/2,r+1.5,f"约 {r}%",ha="center",fontsize=10)
        b.text(i+w/2,ar+1.5,f"{ar}%",ha="center",fontsize=10)
    b.set_xticks(x,["2024 全年\n／年末","2025 全年\n／年末","2026 上半年\n／6 月末"])
    b.set_ylim(0,76);b.set_ylabel("占比／%");b.legend(loc="upper left",frameon=False,fontsize=9.5);style(b)
    save(fig,"08-cloud-channel-dependence","来源：财务附注 1 F-17。收入为期间流量，应收为期末余额；不能据此直接计算渠道回款天数。")

def customer_tiers():
    e=D["enterprise_adoption"];h=D["half_year"]
    fig,(a,b)=plt.subplots(1,2,figsize=(13.8,7.4),gridspec_kw={"width_ratios":[1.2,1]})
    fig.subplots_adjust(top=.78,bottom=.22,left=.16,right=.95,wspace=.35)
    header(fig,"大额客户数量与收入贡献", "左图为截至 2026 年 9 月的累计门槛客户数；右图为两个半年的收入，客户统计时点及期间分开使用")
    counts=[11900,1900,250];yy=[2,1,0]
    a.barh(yy,counts,color=[TEAL,BLUE,AMBER],height=.50)
    for y,v,label in zip(yy,counts,[">11,900",">1,900","约 250"]):a.text(v+180,y,label,va="center",fontsize=12,weight="bold")
    a.set_yticks(yy,["TTM 至少 10 万美元","TTM 至少 100 万美元","TTM 至少 1,000 万美元"])
    a.set_xlim(0,14400);a.set_xlabel("客户数；三个门槛互相包含");style(a,"x")
    big=np.array(e["half_year_revenue_from_100k_ttm_customers_approx"])/1000
    total=np.array(h["revenue"])/1000
    b.bar(range(2),big,width=.5,color=TEAL,label="TTM 至少 10 万美元客户")
    b.bar(range(2),total-big,bottom=big,width=.5,color="#BDD0DD",label="其余客户")
    for i,(v,t,p) in enumerate(zip(big,total,[75,74])):
        b.text(i,t+.5,f"总计 {t:.3f}",ha="center",fontsize=10)
        if i==1:b.text(i,v/2,f"约 {v:.2f}\n约 {p}%",ha="center",va="center",color="white",fontsize=12)
        else:b.annotate("约 0.87（75%）",(i,v/2),xytext=(-.35,4),textcoords="data",fontsize=10,arrowprops={"arrowstyle":"-","color":MUTED})
    b.set_xticks(range(2),h["period"]);b.set_ylim(0,21);b.set_ylabel("半年收入／十亿美元")
    b.legend(loc="upper left",frameon=False,fontsize=9);style(b)
    save(fig,"09-customer-tiers-revenue","来源：S-1 第 142—143、149 页。收入贡献为约数；不把累计门槛客户数相加或据此精确反推各消费区间人数。")

def geographic_growth():
    g=D["geographic_revenue"]
    us=np.array(g["united_states"][-2:])/1000;intl=np.array(g["international"][-2:])/1000
    total=us+intl;share=intl/total*100
    fig,(a,b)=plt.subplots(1,2,figsize=(12.8,7.2),gridspec_kw={"width_ratios":[1,1]})
    fig.subplots_adjust(top=.77,bottom=.21,left=.09,right=.95,wspace=.35)
    header(fig,"海外收入达到上年约 13 倍，占比略降","2025 H1 与 2026 H1 同口径比较；地区按客户账单地址划分，不等于客户国籍或最终用户所在地")
    a.bar(range(2),us,color=BLUE,width=.5,label="美国")
    a.bar(range(2),intl,bottom=us,color=TEAL,width=.5,label="美国以外")
    for i,t in enumerate(total):a.text(i,t+.5,f"合计 {t:.3f}",ha="center",fontsize=10)
    a.text(1,us[1]/2,f"{us[1]:.3f}",ha="center",va="center",color="white",fontsize=12)
    a.text(1,us[1]+intl[1]/2,f"{intl[1]:.3f}",ha="center",va="center",color="white",fontsize=12)
    a.set_xticks(range(2),["2025 H1","2026 H1"]);a.set_ylim(0,20);a.set_ylabel("半年收入／十亿美元")
    a.legend(loc="upper left",frameon=False);style(a)
    b.bar(range(2),share,color=TEAL,width=.5)
    for i,v in enumerate(share):b.text(i,v+1,f"{v:.1f}%",ha="center",weight="bold",fontsize=13)
    b.set_xticks(range(2),["2025 H1","2026 H1"]);b.set_ylim(0,40);b.set_ylabel("美国以外收入占比／%");style(b)
    fig.text(.55,.105,"海外收入从 3.43 亿增至 45.23 亿美元；\n达到约 13.2 倍，美国收入同期约 14.3 倍。",fontsize=10,color=INK)
    save(fig,"10-geographic-revenue-growth","来源：财务附注 2 F-27。海外占比约下降 1.7 个百分点；占比下降并不意味着海外收入下降。")

def lockup_schedule():
    d=D["offering_and_lockup"]
    fig,ax=plt.subplots(figsize=(18,10.7))
    fig.subplots_adjust(top=.79,bottom=.25,left=.17,right=.98)
    header(fig,"分批解禁按股东类别与触发事件展开", "所有比例均为相应股东基准持股的解锁上限，不是总股本占比；实际可流通股数与融资额尚未披露")
    ax.set_xlim(-.5,5.5);ax.set_ylim(-.6,3.6);ax.axis("off")
    headers=["首个交易日", "招股书日期后\n第 27／48／62／76 天", "2026 年报发布后", "2027 Q1 财报\n及第 180 天节点", "2027 Q2 财报发布后", "2027 Q3 财报\n或第 365 天节点"]
    for i,title in enumerate(headers):
        ax.text(i,3.80,title,ha="center",va="bottom",fontsize=11.5,weight="bold",color=INK)
    rows=[
      ("部分员工、顾问\n及非创始人董事高管",[
       ("最多 10%",TEAL),
       ("无该类固定日批次",None),
       ("再 +35%\n第 2 个完整交易日",TEAL),
       ("余股锁定结束\nQ1 第 2 个完整交易日\n或 D+180 收盘，较早者",BLUE),
       ("—",None),("—",None)]),
      ("部分其他股东\n180 天安排",[
       ("无该类首日批次",None),
       ("27／48 天：各 +6%\n62／76 天：各 +7%",TEAL),
       ("第 2 个完整交易日 +12.5%\n年报后 13／27／41／55／69 天\n各再 +9.5%",TEAL),
       ("余股锁定结束\nQ1 第 2 个完整交易日\n或 D+180 收盘，较早者",BLUE),
       ("—",None),("—",None)]),
      ("部分较大股东\n365 天安排",[
       ("无该类首日批次",None),
       ("27／48 天：各 +1.8%\n62／76 天：各 +2.1%",AMBER),
       ("第 2 个完整交易日 +3.75%\n年报后 13／27／41／55／69 天\n各再 +2.85%",AMBER),
       ("较早节点 +4.2%\n另按 Q1 发布后 34／48／62／76 天\n各再 +5.6%",AMBER),
       ("第 2 个完整交易日 +7%\nQ2 后 27／41／55／69／83 天\n各再 +6.3%",AMBER),
       ("余股锁定结束\nQ3 第 2 个完整交易日\n或 D+365 收盘，较早者",BLUE)]),
      ("七位创始人",[
       ("IPO 售出股份另论",None),("无该类固定日批次",None),("无该类年报批次",None),
       ("最多 29%\nQ1 第 2 个完整交易日\n或 D+180 收盘，较早者",RED),
       ("无该类 Q2 批次",None),
       ("余股锁定结束\nQ3 第 2 个完整交易日\n或 D+365 收盘，较早者",BLUE)])]
    for j,(name,cells) in enumerate(rows):
        y=3-j
        ax.text(-.64,y,name,ha="right",va="center",fontsize=11.5,weight="bold",color=INK)
        for i,(label,color) in enumerate(cells):
            face="#E9EEEB" if color is None else color
            rect=FancyBboxPatch((i-.465,y-.395),.93,.79,boxstyle="round,pad=0.01,rounding_size=0.035",linewidth=0,facecolor=face,alpha=1 if color is None else .95)
            ax.add_patch(rect)
            ax.text(i,y,label,ha="center",va="center",fontsize=9.6 if len(label)>40 else 10.5,
                    color=MUTED if color is None else "white",linespacing=1.7)
    fig.text(.075,.186,"D = 最终招股书日期；这里按触发事件分栏，横向距离不代表真实日历间隔，不同栏的事件可能交错。",fontsize=11,color=INK)
    fig.text(.075,.140,"“+”表示以各组基准持股为分母的新增上限。员工与创始人使用前一个月末基准，其他两组使用招股书前 7 天基准。",fontsize=10,color=MUTED)
    fig.text(.075,.103,"锁定结束不等于立即卖出，仍须满足证券转售等限制；非交易日按合同调整，承销商可豁免，具体出售与最终条款可能变化。",fontsize=10,color=MUTED)
    save(fig,"11-lockup-release-schedule","来源：2026-10-26 S-1 第 307—310、322—323 页。各股东组的股数尚留空，无法汇总为公司整体流通比例。")

if __name__=="__main__":
    quarterly();income_bridge();expenses();adoption();cash();commitments()
    customer_concentration();cloud_channels();customer_tiers();geographic_growth()
    lockup_schedule()
    files=sorted(IMAGE_ROOT.glob("anthropic-s1-[0-9][0-9]-*.png"))
    thumbs=[]
    for path in files:
        with Image.open(path) as img:
            thumb=ImageOps.contain(img.convert("RGB"),(780,520))
        tile=Image.new("RGB",(800,555),"#ECEFED")
        tile.paste(thumb,((800-thumb.width)//2,5))
        ImageDraw.Draw(tile).text((15,535),path.name,fill="#203442")
        thumbs.append(tile)
    sheet=Image.new("RGB",(1600,555*((len(thumbs)+1)//2)),"white")
    for i,tile in enumerate(thumbs):sheet.paste(tile,((i%2)*800,(i//2)*555))
    sheet.save(BUILD_ROOT/"chart-contact-sheet.jpg",quality=90)
    customer_sheet=Image.new("RGB",(1600,1110),"white")
    for i,tile in enumerate(thumbs[6:10]):customer_sheet.paste(tile,((i%2)*800,(i//2)*555))
    customer_sheet.save(BUILD_ROOT/"customer-contact-sheet.jpg",quality=90)
    print(f"Rendered {len(files)} PNG charts, SVG counterparts and contact sheet in {IMAGE_ROOT}; SVG files in {ROOT}")
