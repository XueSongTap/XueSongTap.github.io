# 大模型并行文章配图

对应文章：[大模型并行的数学推导，第 6.3 节](../../../../_posts/2025/12/2025-12-06-paralism-basic.md)。
正文图片：[白底 PNG](../../../../img/2025/12/06/sp-activation-storage.png)。

本图延续 [TP 激活显存图](../../11/2025-11-19-tensor-parallel/README.md) 的 X、颜色和几何标尺，只展开原先重复保存的蓝绿色激活。这里具体用 LayerNorm 输出、下一列并行 Linear 的输入作为 X 的例子；橙色 H 与紫色 P 对应的存储项只保留在公式中，不重复画块。

- `source/sp-activation-storage.tex`：唯一可编辑 TikZ 源码。
- `exports/`：PDF、SVG、透明 PNG。
- `build/`：编译日志、中间 PDF、白底预览，由 Git 忽略。
- `render.py`：按自身位置定位仓库，生成导出版本并更新正文图片。

依赖：Python 3、XeLaTeX（TeX Live，含 ctex / Fandol / TikZ）、Poppler 的 pdftocairo。

从任意工作目录运行（路径按当前目录调整）：

```sh
python3 _article_assets/2025/12/2025-12-06-paralism-basic/render.py
```

## 形状、语义与几何账本

| 对象 | 类型与含义 | 全局形状 | 每卡形状 | 几何 |
| --- | --- | --- | --- | --- |
| X | 16-bit 浮点；LayerNorm 后、列并行 Linear 前的输入 | b×s×h | TP 保存、SP 临时聚合均为 b×s×h | 高 1.2 cm，宽 2.4 cm |
| X_r | X 的序列分片；供前向使用并保留待反向 | 合并后为 X | b×(s/t)×h | t=2 时高 0.6 cm，宽 2.4 cm |
| I_r | 序列位置的整数索引区间，r∈{0,…,t−1} | 区间并集为 {0,…,s−1} | 每段 s/t 个位置 | 深浅对应两段序列 |
| M | 一个 micro-batch、每卡每层保存的激活总字节数 | — | 论文近似公式 | 不是 X 单个张量的字节数 |

沿用几何标尺 s=1.2 cm、h=2.4 cm；t=2、s 能整除 t，省略前导 batch 轴。完整 X 的两条色带恰好等于两个 X_r 的高度，所有完整 X 的尺寸完全相同。图形只编码逻辑形状和保存归属，不代表硬件存储或实际序列长度。

左列为 TP 对照，虚线箭头表示改用 SP 后保存方式的变化，不是从 TP 到 SP 的实际运行时搬运。中间到右侧的实线是同一次组内 All-Gather：读取所有序列片段，按 s 拼接，再向每卡分发完整 X。右侧虚线框用于当前 GEMM，使用完可释放，反向需要完整输入计算权重梯度时再聚合；两条 GPU 行不是先后阶段。

## 来源与口径

原创绘图，参考 [Reducing Activation Recomputation in Large Transformer Models，§4.2.2、公式 (4)](https://arxiv.org/html/2205.05198v1#S4.SS2.SSS2)。论文以 Y 表示 LN 输出，本图为延续此前 TP 存储图统一用 X，并在图与正文中明确定义。

公式采用论文的 16-bit 激活、1-byte Dropout mask、4h GeLU MLP、显式保存 Attention Map 的估算。它汇总多份激活，不计模型状态和临时通信/GEMM 缓冲峰值。SP 将原来重复保存的 10sbh 改为 10sbh/t；原 TP 的 24sbh/t 与 5bas²/t 不再额外除以 t。

SP 前向的边界路径为行并行输出 → Reduce-Scatter（跨卡求和并按 s 分发）→ 本地 Dropout / 残差 / LayerNorm → All-Gather → 列并行计算。本图聚焦 LN 输出的保存与重新聚合，不重复完整算子链。边界 RS + AG 的基础数据量与 ring All-Reduce 等价；论文另在反向增加 AG 以避免长时间保存完整输入，并将其与输入梯度计算重叠。

修改后重新运行脚本，目检最终 PNG 的公式、形状、箭头和文字间距，并验证文章图片引用；所有导出版本同步更新。

## 第 7.2 节：Batch scaling 原图

- `source/scaling-book-mixed-fsdp-original.png`：2026-09-22 获取的未改动原图，作为唯一输入保存；[来源文件](https://jax-ml.github.io/scaling-book/assets/img/mixed-fsdp-comms-2.png)。
- `source/parallel-batch-scaling.tex`：仅将原图放入页面，便于现有渲染脚本输出白底 PNG；不重绘曲线、不更改图中文字或数据。
- [正文白底 PNG](../../../../img/2025/12/06/parallel-batch-scaling.png)：由同一 `render.py` 重建，PDF、SVG 和透明 PNG 同步输出到 `exports/`。这张图的 PDF/SVG 内嵌来源位图，并非曲线的可编辑矢量源码。

出处：[How to Scale Your Model，Combining FSDP and Tensor Parallelism](https://jax-ml.github.io/scaling-book/training/#combining-fsdp-and-tensor-parallelism)。现行图采用 TPU v5p 的 4×4×4 mesh、FFN 宽度约 30K 的通信模型，混合曲线允许随 batch 调整并行配置。原图横轴 B/N 对应本文 U/N，图内 100、850 的简写也按该横轴理解。此前 `img/2025/12/image.png` 是阈值不同的旧图，保留原文件但本文不再引用。

正文采用常规 Megatron TP、复制边界输入、保留反向所需输入及 ring 单卡发送量的独立推导；该表用于解释趋势，不用于复现此 TPU 图的数值阈值。

## 第 8.2 节：重计算实验图

正文继续引用 `img/2025/12/image-1.png`。来源为 [Narayanan et al., 2021，Figure 17、§5.6](https://arxiv.org/html/2104.04473v5#S5.SS6)：145B GPT、128 张 80GB A100、TP=8、PP=16、DP=1；横轴为全局 batch 的序列条数。实验段落未说明 micro-batch 大小，不从图中推断其数值。本次仅补充图源与说明，没有修改该图片。
