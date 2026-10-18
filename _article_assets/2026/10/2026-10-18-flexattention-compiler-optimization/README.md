# FlexAttention：BlockMask 配图

对应文章：[FlexAttention：自定义 Attention 的编译优化](../../../../_posts/2026/10/2026-10-18-flexattention-compiler-optimization.md)。

正文图片：[flexattention-block-mask.png](../../../../img/2026/10/18/flexattention-block-mask.png)。这是原创机制示意图，不包含性能实测或硬件线程布局。

## 文件与重建

- `source/flexattention-block-mask.tex`：唯一可编辑图形源码，包含参数、公式、布局、字体、配色和说明。
- `render.py`：读取源码中的参数，精确计算逻辑 mask 与块分类，生成 TikZ 数据，再更新白底 PNG 与可复用导出。
- `exports/`：PDF、SVG、透明 PNG；SVG 的文字转为路径以保持字体一致。
- `build/`：生成的块数据、独立验算结果、日志及编译中间文件；由 `.gitignore` 排除。

依赖：Python 3（仅标准库）、含 `standalone` / TikZ / `ctex` / Fandol 字体的 XeLaTeX、Poppler 的 `pdftocairo`。

在仓库根目录执行：

```sh
python3 _article_assets/2026/10/2026-10-18-flexattention-compiler-optimization/render.py
```

脚本根据自身位置定位仓库，也支持从任意工作目录用绝对路径执行。修改源码后运行脚本并目检正文 PNG；此图固定采用下面的算例，若改变数值，还需一起修改图中的几何与文字。

## 语义与形状验算

单个 head、`Nq = Nkv = 16`、`W = 8`。第 `i` 行第 `j` 列的布尔值由 `0 <= i-j < 8` 决定，包含当前位置和最多前 7 个位置；共有 100 个有效元素。行表示 query，列表示 key/value，两轴的整数坐标范围均为 `0..15`。

以 `4×4` 元素为一个 mask block，得到 `4×4` 块类别；`r,c` 是范围 `0..3` 的块坐标。图中两个正方形覆盖同一组 attention 坐标，聚合箭头明确改变格子的粒度，每个右图格对应左图的 `4×4` 子矩阵。右图是类别图，不是 BlockMask 内部索引张量的物理内存布局。

每块有效元素数为：

```text
10   0   0   0
16  10   0   0
 6  16  10   0
 0   6  16  10
```

由此得到 **全有效 3 块、部分有效 6 块、全空 7 块**。空块跳过；全有效块可以省去 `mask_mod`；部分有效块仍执行逐元素 mask。全有效不代表省去其他 `score_mod` 或注意力计算。左图的有效格与右图全有效块共用蓝绿色，部分有效块为橙色，空值保持白色。

图中采用 `4×4` 便于阅读；PyTorch API 的 `BLOCK_SIZE` 默认值为 `128`，即 Q 与 KV 两个方向均使用 128，具体实现可设置不同块大小。这个 mask block 不表示计算 kernel 的 tile 大小。

## 来源

- 原创图形与算例；布尔矩阵、分类及计数由 `render.py` 枚举验证，并与独立给出的预期计数核对。
- [PyTorch FlexAttention API：create_block_mask 与 BlockMask](https://docs.pytorch.org/docs/2.14/nn.attention.flex_attention.html)：核实 `mask_mod` 的布尔语义、`BLOCK_SIZE=128`、空块条件与 full-block 优化，访问日期 2026-10-18。

绘图采用 `tensor-formula-viz` 的语义、形状、几何与版面检查流程：顶部定义、中间精确结构和聚合关系、底部坐标/对象/边界说明；源码使用可移植 Fandol 字体，无水印或署名。
