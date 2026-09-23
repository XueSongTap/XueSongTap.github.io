# AllToAll 四卡分块交换图

对应文章：[NCCL 通信原语](../../../../_posts/2025/12/2025-12-29-comm.md)，第 6 节。
正文图片：[白底 PNG](../../../../img/2025/12/29/alltoall-rank-exchange.png)。

## 文件与重建

- `source/alltoall-rank-exchange.tex`：唯一可编辑绘图源文件，TikZ + 中文 Fandol 字体。
- `exports/`：PDF、SVG、透明 PNG。
- `build/`：XeLaTeX 中间文件及日志，由 `.gitignore` 排除。
- `render.py`：按脚本自身路径定位仓库，重建导出并更新正文白底 PNG。

依赖 Python 3、TeX Live（XeLaTeX、ctex、standalone、TikZ、Fandol）、Poppler 的 `pdftocairo`。

在任意工作目录执行：

```sh
python3 /path/to/repo/_article_assets/2025/12/2025-12-29-comm/render.py
```

修改 TeX 后重建，目检正文 PNG 的文字、公式、分块和裁切。

## 语义与几何约定

- `s,d`：整数 rank ID，范围 0–3；分别代表来源与目标。
- `c`：每个等长数据块的元素数，正整数；数据类型沿用通信输入，不限定浮点数。
- `B_{s→d}`：由 rank s 提供、交给 rank d 的 c 元素数据向量。
- `X_s`、`Y_d`：本地发送、接收缓冲区的逻辑视图，均为 4×c。
- 映射 `Y_d[s,:] = X_s[d,:]`；全局块索引由 [来源,目标] 变为 [目标,来源]，块内部不转置。
- 颜色仅编码来源 rank；32 个前后展示块来自同一组 16 个唯一块，每个阶段各出现一次。
- 每个 tile 固定 1.3×1.15 cm，间距 0.05 cm；每行四块大小完全相同，阶段间保持几何一致。这些是带标签的数据块示意，不是按元素绘制的矩阵面。
- 行表示一张 GPU 的逻辑分块；中央箭头表示 AllToAll 搬运，不代表网络拓扑、实际执行顺序或性能。
- 对角块归本卡，不发生跨卡传输；所有块均不做规约。

## 来源

原创绘图；逻辑映射核对自 [NVIDIA NCCL Collective Communication Functions / ncclAlltoAll](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclalltoall)（2026-09-22）。
样式依据 tensor-formula-viz 技能：白底、柔和来源色、固定形状、公式与逻辑形状分离。
