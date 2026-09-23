# TP 激活显存配图

对应文章：[TP 下的激活显存公式](../../../../_posts/2025/11/2025-11-19-tensor-parallel.md)。
正文图片：[白底 PNG](../../../../img/2025/11/19/tp-activation-memory.png)。

- `source/tp-activation-memory.tex`：唯一可编辑 TikZ 源码。
- `exports/`：PDF、SVG、透明 PNG。
- `build/`：编译日志、PDF 与白底预览，Git 忽略。
- `render.py`：根据自身位置定位仓库，重建导出文件并更新正文图片。

依赖：Python 3、XeLaTeX（TeX Live，含 ctex / Fandol / TikZ）、Poppler 的 pdftocairo。

从仓库根目录运行：

```sh
python3 _article_assets/2025/11/2025-11-19-tensor-parallel/render.py
```

## 语义与几何

三行是总显存的三个加项，不是前后计算阶段；箭头表示全局对象在 TP 组中的存储归属。颜色区分三类存储，同一类的深浅区分切片。X、H 是浮点激活，P 是注意力概率（每个元素对应 query 对 key 的权重），系数还统计其他缓存和布尔 Dropout mask，并非所画单个张量的 dtype 字节数。

采用 t=a=2 的示意；b 为省略的前导 batch 轴。X 全局与局部均为 b×s×h；H 从 b×s×h 分为 b×s×(h/t)，4h 宽的 MLP 中间态遵循同一规则；P 从 b×a×s×s 分为 b×(a/t)×s×s。实际 head 数须能按 t 均分。

几何标尺：s=1.2 cm，h=2.4 cm，h/t=1.2 cm。X/H 正面高 s、宽 h；H 两个局部切片恰好拼成全局宽度；P 每个 head 正面均为 s×s 正方形。图形只说明轴结构，不代表实际数值维度或物理存储布局。

## 来源与适用范围

原创绘图，公式与存储口径参考 [Reducing Activation Recomputation in Large Transformer Models，第 4.1–4.2 节，公式 (2)](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf)。使用论文的两字节激活、一字节 mask 估算；仅 TP、无序列并行、无重计算，显式保存 Attention Map。24 与 5 是相关缓存的合计系数。

修改后重跑脚本并目检白底 PNG，核对图形尺寸、文字间距、公式和文章引用；导出文件必须同步更新。
