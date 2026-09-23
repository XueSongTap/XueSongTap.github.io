# MXFP8 文章：TN 与 non-TN 配图

对应文章：[MXFP8 与 SM120：硬件支持之后，软件还缺什么](../../../../_posts/2026/09/2026-09-22-mxfp8-sm120-software-support.md)。

正文图片：[gemm-tn-non-tn.png](../../../../img/2026/09/22/gemm-tn-non-tn.png)。

## 文件与重建

- `source/gemm-tn-non-tn.tex`：唯一可编辑绘图源，使用 TikZ、XeLaTeX 与 `ctex` 的 Fandol 字体。
- `exports/`：同源 PDF、SVG 和透明 PNG；正文使用白底 PNG。
- `build/`：TeX 编译中间文件、日志和供目检的白底 PNG，由 Git 忽略。
- `render.py`：从自身位置定位仓库，更新全部导出文件与正文图片。

依赖：TeX Live（`xelatex`、`standalone`、`ctex`、TikZ、Fandol、Latin Modern）、Poppler（`pdftocairo`）、Python 3 标准库。

在任意工作目录运行：

```sh
python3 /path/to/repo/_article_assets/2026/09/2026-09-22-mxfp8-sm120-software-support/render.py
```

修改 TeX 后重新运行命令。应查看 `build/gemm-tn-non-tn.png`，同时检查日志内是否有文字溢出。默认导出分辨率为 240 DPI。

## 形状与语义账本

图中忽略 GEMM 的 alpha、beta 与旧 C 累加项，只表达 `C = op(A) op(B)`；输入为实数矩阵。四行表示独立的接口调用示例，不是连续阶段，也不是同一组物理 buffer 上切换 flag 的实验。

| 调用 | 输入 A | 输入 B | op(A) | op(B) | 输出 C |
| --- | --- | --- | --- | --- | --- |
| TN | K × M | K × N | M × K | K × N | M × N |
| NN | M × K | K × N | M × K | K × N | M × N |
| NT | M × K | N × K | M × K | K × N | M × N |
| TT | K × M | N × K | M × K | K × N | M × N |

- M 是输出行数，N 是输出列数，K 是归约轴；i、j、k 的索引范围分别为 0 到 M−1、0 到 N−1、0 到 K−1。
- 青色固定表示 A，橙色表示 B，紫色表示 C。明暗只是密集矩阵纹理，不编码实际数值。
- 示意格数为 M=2、K=3、N=4，每格统一 0.40 cm；因此 M、K、N 的面边长分别为 0.80、1.20、1.60 cm。
- 所有同形矩阵的几何大小相同；转置输入的高度和宽度真实交换，且色块按转置对应，不能只改文字标签。
- 箭头表示根据 T/N 选择 GEMM 取值视图，不代表执行独立 transpose kernel。图中不展示 row-major/column-major 的物理地址或线程布局。
- 左侧 non-TN 括号包含 NN、NT、TT；正文讨论的 SM120/TE 缺口具体涉及 NN、NT。

## 布局检查

图按公式、输入/运算视图、语义说明三部分排布。每一行都有独立的矩阵、符号和形状轨道；四行之间没有计算依赖箭头。主连接箭头位于背景层，连接成对输入与其运算视图。最大矩阵为 4 行高，所有标签为其预留空间。

## 来源

- [NVIDIA cuBLAS 13.1：GEMM 与转置标志](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#cublas-t-gemm)：`CUBLAS_OP_N`、`CUBLAS_OP_T` 与矩阵乘法定义。
- [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)：文章 MXFP8 背景；本图展示一般 GEMM 的轴顺序，不是 MXFP8 的物理 scale 存储图。
- 使用 `tensor-formula-viz` 的形状与几何规范；配图为本仓库原创 TikZ，未嵌入外部图片。

本文及图片未包含 GPU 性能实测，也不将某个转置组合标作普遍更快。
