# Tiled GEMM 配图

对应文章：[GPU 加速策略](../../../../_posts/2025/10/2025-10-13-gpu-acceleration.md)，第 5.1 节。
正文图片：[tiled-gemm.png](../../../../img/2025/10/13/tiled-gemm.png)。

## 文件与重建

- `source/tiled-gemm.tex`：唯一可编辑 TikZ 源码。
- `exports/`：PDF、SVG 和透明 PNG。
- `build/`：编译日志与中间文件，Git 忽略。
- `render.py`：基于自身路径定位仓库，重建导出文件及正文白底 PNG。

依赖：Python 3、TeX Live（XeLaTeX、ctex/Fandol、TikZ、standalone）、Poppler（pdftocairo）。

在本目录运行，或从任意目录指定脚本路径：

```sh
python3 render.py
```

## 图义与几何

图采用原文方阵 GEMM，示例 N=K=3T，各全局矩阵为 N×N，均用边长 1.8 cm 的正方形表示。每个 T×T tile 边长 0.6 cm；shared 副本、寄存器逻辑累加块与输出块保持该尺寸。固定 i=j=1（从 0 开始），p=0,1,2 依次选择 A 的中间块行与 B 的中间块列中对应的 tile。颜色表示张量角色与当前选择，不表示数值大小；灰格表示未选择的数据，并非零。

A、B、C、S 均为数值矩阵；i、j、p 为整数块索引。S^(p) 表示完成前 p 个 phase 的输出块部分和。每轮乘法 (T×T)(T×T)→T×T；同一 S 只初始化一次。S 是 block 内各线程寄存器值组成的逻辑矩阵，不代表连续物理存储。示例假设 N 可被 T 整除。

蓝色虚线：global/shared 之间的选中 tile 搬运及最终写回；深灰箭头：乘加计算；紫色箭头：同一累加状态沿时间延续。A、B 旁的 K 方向箭头为坐标轴。三个 phase 共用同一组 shared 缓冲区，加载后与计算后分别同步；最终写回不在 phase 循环内。

## 来源与检查

依据原文第 5 节及其链接的 [matmul_tile_full.cu](https://github.com/llmsystem/llmsys_code_examples/blob/main/cuda_acceleration_demo/matmul_tile_full.cu) 中 `matMulTiled` 的 `As`、`Bs`、`Cvalue`、`ph` 和两次 `__syncthreads()` 原创绘制，2026-09-22 核对。图是算法逻辑示意，不是硬件布局或性能测量。

重建后目检正文 PNG：公式、tile 对应关系、箭头含义、文字遮挡、裁切与逻辑形状；同步检查文章引用路径。
