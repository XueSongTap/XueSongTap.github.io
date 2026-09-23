# FlashAttention 分块与在线累积配图

对应文章：[FlashAttention](../../../../_posts/2026/03/2026-03-22-flash-attention.md)。

- `source/draw.py`：唯一可编辑绘图源，生成 TikZ；修改坐标、形状和文案均在此处完成。
- `render.py`：从自身位置定位仓库；编译、导出并更新正文白底 PNG。
- `exports/`：两图的 PDF、SVG 和透明 PNG。
- `build/`：生成的 TeX、编译中间文件与日志，由 Git 忽略。
- 正文图片：[计算链](../../../../img/2026/03/22/flash-attention-tile-chain.png)、[统计量](../../../../img/2026/03/22/flash-attention-online-state.png)。

依赖：Python 3、XeLaTeX（ctex / Fandol、TikZ、amsmath）、Poppler（pdftoppm、pdftocairo）。

重建（可从任意工作目录调用）：

```sh
python3 /path/to/repo/_article_assets/2026/03/2026-03-22-flash-attention/render.py
```

## 符号和几何台账

范围为单个 head、无 mask、无 dropout，Q/K/V 全局均为 N×d，固定 query tile 扫描所有 KV tile。浮点值的具体存储精度由 kernel 决定，图不指定 dtype 位宽。

| 对象 | 局部形状 | 语义 / 生产与消费 |
|---|---|---|
| Q_i | Br×d | query 浮点特征，参与 QKᵀ |
| K_jᵀ | d×Bc | key 特征的转置，d 为收缩轴 |
| V_j | Bc×d | value 浮点特征，参与权重乘 V |
| S_ij | Br×Bc | 缩放点积分数，生成最大值和指数权重 |
| P̃_ij | Br×Bc | 未归一化非负指数权重，沿 Bc 求和或乘 V |
| m_i、m̂_i | Br×1 | 逐行累计 / 当前块最大分数 |
| α_i | Br×1 | 旧贡献缩放系数，范围 [0,1] |
| ℓ_i、Δℓ_i | Br×1 | 累计 / 当前块指数和 |
| U_i、ΔU_i、O_i | Br×d | 累积分子 / 新贡献 / 最终归一化输出 |

几何以 0.42 cm 为单元间距：Br=3 单元、Bc=5 单元、d=4 单元，单列=1 单元。每个矩阵的高度对应行轴、宽度对应列轴；同形对象的外框完全相同。格数仅展示轴结构，不是 kernel 参数或数值样本。颜色深浅是示意纹理，不是测量值。折线为同一中间结果的延续，水平箭头为计算；循环条件和 HBM 搬运用正文及图内文字说明。

## 来源

- https://arxiv.org/abs/2205.14135 ：FlashAttention，IO-aware 分块与 online softmax。
- https://arxiv.org/html/2307.08691v1 ：FlashAttention-2，§3.1.1 / Algorithm 1，固定 query 块、累积未归一化输出、最后除以分母。

本图为原创逻辑张量示意，不表示实际寄存器分配、shared memory 布局或性能测量。
