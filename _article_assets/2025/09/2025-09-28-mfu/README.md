# MFU：线性层前反向 GEMM

文章：[MFU估算](../../../../_posts/2025/09/2025-09-28-mfu.md)。
正文图片：[白底 PNG](../../../../img/2025/09/28/linear-training-gemm.png)。

- `source/linear-training-gemm.tex`：唯一可编辑绘图源，修改公式、坐标与色块请编辑此文件。
- `exports/`：PDF、SVG、白底 PNG、透明 PNG。
- `build/`：XeLaTeX 编译产物与日志，Git 忽略。
- `render.py`：从自身路径定位仓库并重建全部导出，同步正文 PNG。

依赖：Python 3、TeX Live（XeLaTeX、ctex、Fandol、TikZ、standalone）、Poppler（pdftocairo）。

从任意目录运行：

```sh
python3 /path/to/repo/_article_assets/2025/09/2025-09-28-mfu/render.py
```

## 形状与图意核对

对象均为逻辑实值矩阵，在实现中使用浮点 dtype；d 前缀为标量 loss 的梯度，不表示有限差分。图不指定存储 dtype 或硬件布局，无分片，每个形状均为所示线性层的完整逻辑形状。

| 对象 | 形状 | 含义 / 来源 |
| --- | --- | --- |
| X | B × D | 输入激活，反向复用 |
| W | D × K | 可训练权重，反向复用 |
| Y | B × K | 前向输出 XW |
| dY | B × K | 后续计算传回的梯度，两条反向分支共享 |
| dX | B × D | dY Wᵀ，沿 K 收缩 |
| dW | D × K | Xᵀ dY，沿 B 收缩 |

B、D、K 是正整数维度。统一几何映射：B=4、D=6、K=3 个示意单元，每格边长 0.34 cm；矩阵高度映射第一轴，宽度映射第二轴。转置交换长宽和色块位置，同一个 dY 在两行复用相同色块；颜色仅区分张量角色，深浅无定量含义。粗边标收缩轴，前向沿 D 收缩。每个 GEMM 约 2BDK FLOPs，计一次乘加为 2 FLOPs。

前向与反向之间的箭头表示后续计算/loss 产生 dY 的依赖。左侧括线表示两个反向分支共享 dY，不表示 dX 到 dW 的串行计算。

## 来源

原创图；矩阵乘法与梯度公式直接从链式法则推导。FLOPs 口径参考：

- [NVIDIA Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [PaLM §4.1 / Appendix B](https://arxiv.org/html/2204.02311v5#A2)

使用 tensor-formula-viz 技能的矩阵视觉规范。图示为解释用逻辑矩阵，不是实验结果。
