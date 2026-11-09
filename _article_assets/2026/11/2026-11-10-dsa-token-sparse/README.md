# DSA 配图

对应文章：[稀疏注意力为什么收敛到 DSA：相比 NSA 和 MoBA，它换来了什么](../../../../_posts/2026/11/2026-11-10-dsa-token-sparse.md)。

- `source/gen_figures.py`：唯一可编辑的源文件，生成三张图的 TikZ。小矩阵的数值由脚本实际计算，所以 Top-K 结果、gather 的行与转置在图上前后一致。
- `exports/`：TeX、PDF、SVG、透明 PNG。
- `build/`：编译日志与中间文件，Git 忽略。
- 正文白底 PNG 位于 `../../../../img/2026/11/10/`。

重建依赖：Python 3、TeX Live（XeLaTeX、ctex/Fandol、TikZ）、Poppler（pdftoppm、pdftocairo）。在本目录执行 `python3 render.py`，脚本从自身路径寻找仓库。

`dsa-indexer-topk`：Indexer 打分到整数地址的路径。示意轴 H=4、d=5、N=10、k=4，单元格边长 0.42 cm；实际配置为 H=64、d=128、k=2048。

`dsa-gather-attention`：地址 → gather → QK/Softmax/PV。示意轴 N=10、d_c=5、H=4、k=4。G_t 是逻辑张量，不代表 kernel 内 SMEM 或 TMEM 的物理布局；V3.2 的 QK 为 576 维、PV 为 512 维，图中用同一个 d_c 表示。

`dsa-vs-nsa-read`：NSA 与 DSA 的粗路径和读取掩码并列对比，两排是并列方案，不是先后阶段。示意轴 N=32、块大小 4、选 2 块、滑窗 4、k=12；示意中压缩步长与块大小相同，实际 NSA 配置为 d=16、l'=64。

来源：DeepSeek-V3.2 技术报告 §2.1、官方 config.json，以及 NSA 论文 §3.3。图中未使用外部图像。
