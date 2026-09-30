# MXFP8 文章配图

对应文章：[MXFP8 与 SM120：硬件支持之后，软件还缺什么](../../../../_posts/2026/09/2026-09-22-mxfp8-sm120-software-support.md)。

正文图片：[gemm-tn-non-tn.png](../../../../img/2026/09/22/gemm-tn-non-tn.png)。

精度说明图片：[mxfp8-precision-blocks.png](../../../../img/2026/09/22/mxfp8-precision-blocks.png)。

SGLang Inkling KV cache 路径图：[sglang-inkling-mxfp8-kv.png](../../../../img/2026/09/22/sglang-inkling-mxfp8-kv.png)。

## 文件与重建

- `source/gemm-tn-non-tn.tex`：唯一可编辑绘图源，使用 TikZ、XeLaTeX 与 `ctex` 的 Fandol 字体。
- `source/mxfp8-precision-blocks.tex`：精度说明图的唯一可编辑源；画出 E4M3/E5M2 的位分配与两个 MXFP8 逻辑块。
- `source/sglang-inkling-mxfp8-kv.tex`：Inkling MXFP8 KV cache 路径图的唯一可编辑源；画出配置、分块量化、数据与 scale 缓存及 FA4 读取。
- `exports/`：同源 PDF、SVG 和透明 PNG；正文使用白底 PNG。
- `build/`：TeX 编译中间文件、日志和供目检的白底 PNG，由 Git 忽略。
- `render.py`：从自身位置定位仓库，对 `source/` 内各图更新导出文件与正文图片。

依赖：TeX Live（`xelatex`、`standalone`、`ctex`、TikZ、Fandol、Latin Modern）、Poppler（`pdftocairo`）、Python 3 标准库。

在任意工作目录运行：

```sh
python3 /path/to/repo/_article_assets/2026/09/2026-09-22-mxfp8-sm120-software-support/render.py
```

修改 TeX 后重新运行命令。应查看 `build/` 内三张 PNG，同时检查对应 `.log` 内是否有文字溢出。默认导出分辨率为 240 DPI。

## SGLang Inkling 路径图语义

- 以一个 token、一个 head 的 $X\in\{K,V\}$ 为写入示例；$d_X$ 是最后一维，要求可被 32 整除。图中两个相邻块只是示意，不指定真实 head dimension 为 64。
- 每个 32 元素块生成 32 个 E4M3 数据值和 1 个 E8M0 scale。K、V 各自量化；KV pool 分别保存两者的数据与 scale。图中的分离表示逻辑缓冲，不描绘实际地址或线程布局。
- `--page-size 128` 的 128 是每页 token 数；此路径下 scale 使用 FA4 的交错布局。`--kv-cache-dtype mxfp8` 选择缓存格式，`--attention-backend fa4` 选择读取它的算子；权重的 `modelopt_fp4` 是独立配置。
- 读取时当前 Q 连同自己的 scale 与缓存 K/scale 做块缩放 $QK^T$；缓存 V/scale 在 FA4 内还原为 BF16 后参与 $PV$。箭头表示数据流，不表示图中的逻辑块与物理内存一一对应。
- 图中 $P$ 表示 softmax 后的 attention 权重；省略 mask、归一化尺度、页地址查找及其他 attention 细节。

## 精度图语义

- $q_{b,i}$ 是 E4M3 或 E5M2 单元素编码，具有自己的符号、指数、尾数；E8M0 scale 是块级标量，不属于元素的 8 位。
- 两个 $1\times32$ 逻辑块拼成 $1\times64$ 向量。图中每块只画首尾各四个格子，中间省略号不改变每块有 32 个元素这一事实。
- 青色是 FP8 元素及指数字段，橙色是共享 scale，紫色是尾数字段。两条灰色箭头表示各自的 scale 作用于整块，箭头终点落在该块的一个格子上作为示意，不代表只缩放这个格子。
- 图的下半部使用正文 E4M3 算例：每块 32 个输入分别是 $2^{10}$、$2^{-12}$，所选 scale 让两块的 $q$ 都为 256，解码后恢复原数。
- $s_0=2^2$ 与 $s_1=2^{-20}$ 来自正文的手算例子，是人为选择的可表示 E8M0 数值，并非规定的量化器输出。
- 该图不表达物理数组位置、地址、线程布局或实际 E4M3/E5M2 混用。每个块按一种元素编码解释，两个编码行是候选格式比较。

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
- [OCP 8-bit Floating Point Specification v1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1)：精度图 E4M3/E5M2 的字段和数值范围来源。
- 使用 `tensor-formula-viz` 的形状与几何规范；配图为本仓库原创 TikZ，未嵌入外部图片。

本文及图片未包含 GPU 性能实测，也不将某个转置组合标作普遍更快。
