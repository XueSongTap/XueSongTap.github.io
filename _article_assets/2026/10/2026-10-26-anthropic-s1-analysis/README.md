# Anthropic S-1 博客配图

对应文章：[Anthropic S-1 解读 收入增长背后的训练成本与算力承诺](../../../../_posts/2026/10/2026-10-26-anthropic-s1-analysis.md)。

原始依据：[Anthropic, PBC Form S-1，2026-10-26，SEC EDGAR](https://www.sec.gov/Archives/edgar/data/2130357/000162828026067168/anthropicpbc-sx1.htm)。

本文从 own_log 的同主题笔记整理而来。本目录是博客配图后续维护的编辑入口；own_log 保留笔记快照，不从另一个目录自动覆盖博客源码。正文图片使用白底 PNG，无水印。

## 目录

- source/render_charts.py：唯一的博客绘图源，包含数值图和解禁事件图。
- source/data.json：原文财务数据、客户指标、合同安排及出处。财务金额默认为百万美元，个别合同金额明确标为十亿美元。
- source/source-excerpts.txt：原文相关文本节选和下载 HTML 的 SHA-256。
- source/requirements.txt：复现依赖版本。
- render.py：按自身位置执行绘图源，不依赖调用时的工作目录。
- exports/：十一张图的 SVG，字形按路径导出。
- build/：联系表、文章预览和检查结果，由本目录的 .gitignore 排除。
- 正文 PNG：写入 ../../../../img/2026/10/26/，文件名前缀为 anthropic-s1-。

## 配图与正文顺序

| 正文图号 | 图片文件名主体 | 内容 |
| --- | --- | --- |
| 1 | 01-quarterly-revenue-operating-profit | 季度收入、经营费用与营业利润 |
| 2 | 02-net-loss-waterfall | 营业亏损到净亏损 |
| 3 | 03-operating-expense-waterfall | 经营费用中的研发与训练相关投入 |
| 4 | 04-revenue-mix-customer-adoption | 按量收入、订阅收入与客户增长 |
| 5 | 09-customer-tiers-revenue | 大额客户数量与收入贡献 |
| 6 | 07-customer-concentration | 匿名客户份额与前五大集中度 |
| 7 | 08-cloud-channel-dependence | 云销售和结算渠道依赖 |
| 8 | 10-geographic-revenue-growth | 美国与海外收入 |
| 9 | 05-cash-flow-quality | 经营现金流和证券投资 |
| 10 | 06-selected-compute-commitments | 代表性多年算力合同 |
| 11 | 11-lockup-release-schedule | 四类股东分批解禁事件 |

图号按文章阅读顺序排列，文件名保留原图身份。正文不提供制作源文件下载链接。

## 重建

需要 Python、Matplotlib、NumPy、Pillow 和中文字体。macOS 优先选择 Hiragino Sans GB 或 PingFang；其他系统可在绘图源中配置字体路径。

在本目录运行：

    uv run --no-project --with-requirements source/requirements.txt render.py

也可以在已安装依赖的环境里，从任意工作目录执行：

    python3 /path/to/2026-10-26-anthropic-s1-analysis/render.py

重建会更新对应正文 PNG、SVG，以及 build/ 中的两张联系表，不更改其他文章配图。

## 数据口径

年度和半年数字不当作等长时序混画。2026 年中期及季度财务数据未经审计。客户指标按原文时点，数量保留“约”与“超过”，累计消费门槛互相包含。

客户集中度的 null 表示低于 10% 披露阈值，不能当成零；发行金额字段的 null 表示本版留空。云渠道是代销和结算渠道，不能视为平台自身采购；估算渠道金额保留约数。RPO 包含递延收入，二者不能相加。

合同金额为不同期限的部分安排，不年化、不汇总成全部算力承诺。解禁比例针对各组基准持股，不是公司总股本。事件分栏不按真实日历比例排列，不虚构财报日期、发行股数或整体流通比例。

白底图由本目录源码重新生成。检查包括原文数据勾稽、图号及图片路径、前置文章链接、SVG 同步导出、PNG 尺寸和视觉排版。
