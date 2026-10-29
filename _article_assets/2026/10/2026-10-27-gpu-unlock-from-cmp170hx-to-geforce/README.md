# GPU Unlock 文章配图与来源

对应文章：[_posts/2026/10/2026-10-27-gpu-unlock-from-cmp170hx-to-geforce.md](../../../../_posts/2026/10/2026-10-27-gpu-unlock-from-cmp170hx-to-geforce.md)。

正式图片：[img/2026/10/27/](../../../../img/2026/10/27/)，本篇使用文件名以 `gpu-unlock-` 开头的 7 张 PNG。资料核对日期为 2026-10-27。

## 文件与重建

- `render.py`：两张概念图、三张论文裁图和两张统计图的唯一重建入口。
- `source/measurements.json`：RTX 3090 Ti GEMM 数据、RTX 5090 与 PRO 6000 理论峰值，以及来源 URL。修改数值后运行脚本更新 PNG 和 SVG。
- `source/canary-in-the-crypto-mine.pdf`：Jon Pry 的 16 页论文，从本次研究使用的 Zotero 附件复制。Zotero 元数据标注 CC BY 4.0。
- `exports/`：两张统计图的 SVG 可复用版本。
- `build/`：整页临时渲染、联系表、Markdown 预览和验证记录，由本目录 `.gitignore` 排除。

脚本根据自身位置解析博客仓库和输出目录，可从任意工作目录运行：

```bash
uv run --isolated --no-project --with matplotlib --with pillow python \
  _article_assets/2026/10/2026-10-27-gpu-unlock-from-cmp170hx-to-geforce/render.py
```

需要 Poppler 的 `pdftoppm`。macOS 使用 Arial Unicode，Linux 使用脚本中列出的 Noto Sans CJK；其他环境可修改字体候选路径。

## 图源

| 正文图 | 文件 | 来源与处理 |
| --- | --- | --- |
| 1 | `gpu-unlock-security-path.png` | 根据论文、CMP 社区 `falcon-and-booter.md` 和 Zagitta 文档绘制。区分离线解密分析与签名验证后的运行时利用 |
| 2 | `gpu-unlock-canary-layout.png` | 论文 Figure 2，PDF 第 6 页，保留原图注 |
| 3 | `gpu-unlock-canary-overflow.png` | 论文 Figure 3，PDF 第 7 页，保留原图注 |
| 4 | `gpu-unlock-paper-flow.png` | 论文 Figure 4，PDF 第 9 页，保留原图注 |
| 5 | `gpu-unlock-pre-driver.png` | 根据 Linux initramfs 与 Windows UEFI 实现绘制，概括共同事务 |
| 6 | `gpu-unlock-3090ti-gemm.png` | 按 Zagitta 公开数据重绘，标明卡、功耗、形状、累加精度和统计方法 |
| 7 | `gpu-unlock-blackwell-specs.png` | 按 NVIDIA 两份白皮书 v1.1 重绘，使用理论稠密峰值 |

论文图归 Jon Pry，提取时只裁剪和渲染。新绘图属于本文解释材料；性能结果均来自原作者报告，未运行解锁或硬件复测。完整在线参考文献放在文章末尾。

## 内容核对说明

- 将“简单密码”解释为非秘密的编号测试密钥方案，编号 37；不将密钥索引写成实际 AES 密钥，也不声称泄露生产 RSA 签名私钥。
- 以 Canary/CMP 170HX 解释共同方法，再按 Turing、Ampere、Ada 介绍实现；保留 Turing 独立上游与芯片间 payload 差异。
- Blackwell 结论限定为核对日期内尚无公开可复现方案。
- PRO 6000 比较明确区分 Max-Q 与 Workstation；GeForce/PRO 的规格来自不同官方白皮书，均采用稠密指标。
- ECC 表述使用 v1.1 更新：GDDR7 GeForce 同样具有片内单比特纠错，未将 ECC 简化为专业卡独有。

已检查图片解码、图表裁剪、公式、引用编号、站内 `post_url` 目标和图片路径。`bundle check` 报告本机缺少 Jekyll 项目依赖，因此未完成整站 Jekyll 构建；Markdown 预览与静态检查结果保存在 `build/`。
