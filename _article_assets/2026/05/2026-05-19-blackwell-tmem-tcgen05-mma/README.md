# Blackwell TMEM 与 tcgen05 MMA 深度解析：配图附件

对应文章：[2026-05-19-blackwell-tmem-tcgen05-mma.md](../../../../_posts/2026/05/2026-05-19-blackwell-tmem-tcgen05-mma.md)。

- `source/tmem-mma-dataflow.tex`：唯一的可编辑绘图源文件，包含矩阵、文字和跨行箭头。
- `exports/`：PDF、SVG 和透明 PNG，供后续编辑、排版与复用；不在博客正文提供下载入口。
- `build/`：XeLaTeX 日志、辅助文件、编译 PDF 和检查用图片，保留在本地，由本目录 `.gitignore` 排除。
- 正文使用的白底 PNG：[img/2026/05/19/tmem-mma-dataflow.png](../../../../img/2026/05/19/tmem-mma-dataflow.png)。

`_article_assets` 是 Jekyll 默认不发布的下划线目录；不要将其配置为 collection 或加入 `include`。正文仅引用 `img/` 中的最终 PNG。

## 修改与重建

先编辑 `source/tmem-mma-dataflow.tex`，再在此目录执行：

```bash
python3 render.py
```

也可以从任意目录以脚本完整路径运行。脚本根据自身位置寻找博客根目录，不依赖临时工作区。需要 PATH 中的 XeLaTeX（TeX Live，含 ctex/Fandol、TikZ、standalone）与 Poppler 的 pdftocairo。

脚本生成所有导出格式，并更新正文 PNG。生成后打开 PNG 检查文字、箭头、矩阵边界和页边距；不自动提交或发布。

## 绘图约定与核对

- 逻辑形状：A 为 m×k，B 为 k×n，D、R、O 均为 m×n。
- 几何映射：m=2.4 cm、k=1.6 cm、n=3.2 cm；所有 m×n 面保持同尺寸。色块只表示轴结构，不是实际元素数、线程布局或数值实验。
- 上排重复 T 次，下排接续最终的 D^(T)；紫色跨行箭头表示同一 TMEM 状态的延续，不表示数据复制。
- 图示是 A/B 均来自 SMEM 的路径；使用 FP16/BF16 输入、FP32 累加的例子。D^(0)=0 表示数学初始条件。
- 无署名、水印或品牌标记。

依据：

- [NVIDIA tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html)
- [NVIDIA PTX ISA：Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)

核对日期：2026-09-20。
