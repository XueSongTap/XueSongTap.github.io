# SP 与 CP 配图

文章：[序列并行与上下文并行](../../../../_posts/2026/03/2026-03-22-sequence-context-parallel.md)。
正文使用 [白底 PNG](../../../../img/2026/03/22/)；唯一可编辑源在 `source/`，PDF、SVG、透明 PNG 在 `exports/`。`build/` 是忽略的编译中间文件。

## 重建

依赖 Python 3、TeX Live（XeLaTeX、ctex/Fandol、TikZ、standalone）、Poppler（pdftocairo）。

```sh
python3 _article_assets/2026/03/2026-03-22-sequence-context-parallel/render.py
```

脚本根据自身位置定位仓库，可从任意工作目录执行；重建三个源文件并同步正文图片。修改 source 后须重建并目检。

## 语义与几何台账

- `ulysses-layout`：MHA，省略 batch，P=2。X 是 Q/K/V 任一个实值张量；T_i 是序列区间，H_j 是 head 分组，i,j ∈ {0,1}。单块 X_ij 为 (L/2,h/2,d_h)。图明确合并末两轴为矩阵面：L 对应 3.2 cm，h*d_h 对应 4 cm；每块 1.6×2 cm，所有块尺寸一致。左边按 head 拆分，右边按序列拼接；颜色表示 head 归属。输出逆向通信还原序列分片。
- `ring-online-attention`：单个 batch、单个 head，无 mask；n=L/P，n 对应 1.2 cm，d_h 对应 1.8 cm。Q、V 为 n×d_h，K 转置为 d_h×n，S 为 n×n，保持相同收缩边长与方形分数面。S 是实值 logits；m、ell 是长度 n 的逐行最大值和重标定指数和，A 是 n×d_h 的实值加权和。每轮沿 key 轴归约，并按 query 行广播；最后 A/ell 得到 O。循环中的整数 j 是 KV 源 rank，范围 [0,P)，图例次序为 i=0、P=4、递减方向。箭头表示数据依赖或循环状态，邻卡传入的是 K 与 V。输出只显示符号与形状，不另外画矩阵面。
- 图形尺寸表达轴关系，不是物理内存布局、实际 token 数、硬件拓扑或性能测量。Ring 完整分数块是逻辑对象，实际内核可继续切 tile。

## 来源

图为依据公式自制，未复制外部图片。机制参考：

- [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Megatron CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

验证：独立核对切分元素数、矩阵乘法收缩维、在线 softmax 更新及 P 次计算/P-1 次交换；渲染后目检公式、标签、连接和裁切。正文不放制作文件下载链接。

## Megatron SP + TP 补图

`source/megatron-sp-tp.tex` 展示两卡、无 bias 的普通 MLP 前向。X_i、Y_i 为 L/2×d，X 与每个 P^(r) 为 L×d，H^(r) 为 L×f/2；W1^(r) 为 d×f/2，W2^(r) 为 f/2×d。所有对象均为实值，r,i∈{0,1}，T_i 为第 i 段 token。φ 为逐元素激活。

几何台账：L=2 cm，d=1.6 cm，f/2=3.2 cm；因此序列分片高 1 cm、完整激活与部分和高 2 cm，等形状矩阵面完全一致。权重仅在算子框中标注形状，不用矩阵面表示。颜色按激活角色编码；行并行层的部分和需跨 rank 相加，ReduceScatter 再沿序列轴切片。长连接线表示同一计算路径向下一行延续。省略 batch、bias 和反向。

机制来源：[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)。
