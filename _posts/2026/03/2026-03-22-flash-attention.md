---
layout: article
title: FlashAttention：IO 感知的精确 Attention 计算
tags: LLM
---

> 关联阅读：[Transformer 加速技巧](https://xuesongtap.github.io/2025/10/15/transformer-acc.html) | [混合精度训练（AMP）](https://xuesongtap.github.io/2026/03/22/amp-mixed-precision.html)

---

## 1. 标准 Attention 的内存瓶颈

标准 Attention 计算为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

朴素实现的显存复杂度为 $O(N^2)$（需要存储 $N \times N$ 的 attention score 矩阵），时间复杂度也是 $O(N^2 d)$。

**瓶颈在哪里？** 不是 FLOP 不够，而是 **HBM（显存带宽）** 太慢。

以 A100（80GB HBM2e，带宽 2TB/s，FP16 算力 312 TFLOPS）为例：
- 计算强度（FLOP/Byte）：312T / 2T = **156**；
- 标准 Attention 的算术强度：对于 $N=1024$，$d=64$，约为 $2N^2d / (4N^2)$ bytes ≈ **32**（远低于 156）；
- 结论：标准 Attention 是**显存带宽受限（memory-bound）**，大量时间在等数据，而不是在算。

---

## 2. FlashAttention 的核心思想

FlashAttention [1] 不改变最终计算结果（精确，非近似），只改变**计算顺序**，做到：
1. **Tiling（分块）**：把 $Q$、$K$、$V$ 分成小块，放进 SRAM（片上缓存，带宽高 10-20×）分块计算；
2. **Online Softmax**：用数值稳定的在线算法，在不完整地看完所有 $K$ 的情况下，增量地维护 softmax 分母；
3. **Recomputation（重计算）**：反向传播时不存中间的 attention score 矩阵，而是重新从 $Q$、$K$、$V$ 计算，节省大量 HBM 写入。

### 2.1 Tiling：从 HBM 到 SRAM

```
标准实现（HBM-bound）:
  1. Q, K, V 从 HBM 读入
  2. 计算 S = QKᵀ → 写回 HBM（N×N 矩阵）
  3. P = softmax(S) → 写回 HBM
  4. O = PV → 写回 HBM

FlashAttention（SRAM-first）:
  1. 把 Q 切成 Tr 块，K/V 切成 Tc 块
  2. 对每个 (Q_i, K_j, V_j) 的组合，在 SRAM 里做局部 attention
  3. 维护在线 softmax 状态（m, l）增量更新 O_i
  4. 只在最后把最终的 O 写回 HBM，S/P 从不写入 HBM
```

**关键参数**：块大小由 SRAM 大小决定，典型值 $B_r = B_c = 128$（tokens/block）。

### 2.2 Online Softmax（数值稳定 + 增量）

标准 softmax 需要先看完所有 $K$ 才能知道最大值（用于数值稳定）和分母 $\sum \exp(\cdot)$。Online softmax 用两个统计量做增量更新：

$$
m_i^{(\text{new})} = \max(m_i^{(\text{old})},\ \max_j s_{ij})
$$

$$
\ell_i^{(\text{new})} = e^{m_i^{(\text{old})} - m_i^{(\text{new})}} \cdot \ell_i^{(\text{old})} + \sum_j e^{s_{ij} - m_i^{(\text{new})}}
$$

$$
O_i^{(\text{new})} = \frac{\ell_i^{(\text{old})} \cdot e^{m_i^{(\text{old})} - m_i^{(\text{new})}}}{\ell_i^{(\text{new})}} O_i^{(\text{old})} + \frac{e^{s_{ij} - m_i^{(\text{new})}}}{\ell_i^{(\text{new})}} V_j
$$

这样对每个 $K$ 块只需一遍扫描，每块更新 $(m, \ell, O)$ 三个统计量。

---

## 3. 内存和速度收益

| 指标 | 标准 Attention | FlashAttention |
|------|--------------|---------------|
| **HBM 访问量** | $O(N^2)$ | $O(N^2 d / M)$（$M$ 为 SRAM 大小）|
| **存储 attention score** | $O(N^2)$ | **不存，重计算** |
| **数值精确性** | 精确 | **精确（等价结果）** |
| **序列长度支持** | $N$ 受 HBM 限制 | 大幅扩展，仅 $O(N)$ 显存 |

FlashAttention 实测在 A100 上对长序列 attention 的加速约 **2-4×**（取决于序列长度），且显存使用从 $O(N^2)$ 降到 $O(N)$。

---

## 4. FlashAttention-2 / 3 的改进

### FlashAttention-2 [2]

主要工程优化：
1. **减少非矩阵乘 FLOP**：把 online softmax 中的 rescale 操作放到最后一次做，减少不必要的重新缩放；
2. **更好的并行化**：将 $Q$ 的 tile 分配给不同 warp，提高 GPU 占用率；
3. **序列维度的 warp 分配**：避免 warp 间通信，充分利用 register 文件。

FlashAttention-2 比 FA-1 快约 **2×**，在 A100 80GB BF16 下，实测达到 **~72% 的 Tensor Core 利用率**（接近理论峰值）。

### FlashAttention-3 [3]（H100 优化）

针对 H100 的硬件特性做了专项优化：
1. **Warp Specialization**：把 attention 的 producer（GEMM/数据加载）和 consumer（softmax/输出）分配给不同 warp，流水线执行；
2. **WGMMA（Warp Group Matrix Multiply-Accumulate）**：利用 H100 的 TMA（Tensor Memory Accelerator）硬件指令；
3. **FP8 支持**：实现了 FP8 的 attention，进一步降低带宽开销。

H100 上 FA-3 比 FA-2 快约 **1.5-2×**。

---

## 5. 与 Sequence Parallel 的配合

在 Ulysses SP 模式下，attention 前后各有一次 All-to-All：
```
(L/P, d) → All-to-All → (L, d/P) → FlashAttention → All-to-All → (L/P, d)
```

FlashAttention 此时在 `(L, d/P)` 的张量上做计算，$L$ 是全序列长度，每张卡负责 $d/P$ 个 head。

在 Ring CP 模式下，FlashAttention 配合 Ring 通信，对每个 KV 块增量计算局部 attention 后累积（FlashAttention 内置的 online softmax 天然支持这种分块累积）。

---

## 6. 使用

PyTorch 2.0+ 原生集成（`scaled_dot_product_attention`）：

```python
import torch.nn.functional as F

# 自动选择最优 backend（含 FlashAttention）
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,          # 因果 mask
    scale=None,               # 默认 1/sqrt(d_k)
)
```

Megatron-LM / Transformer Engine 默认会在支持的设备上自动使用 FlashAttention：

```bash
--use-flash-attn               # Megatron 开关
```

---

## 参考

[1] Dao, T., et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. [arxiv:2205.14135](https://arxiv.org/abs/2205.14135)

[2] Dao, T. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024. [arxiv:2307.08691](https://arxiv.org/abs/2307.08691)

[3] Shah, J., et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* 2024. [arxiv:2407.08608](https://arxiv.org/abs/2407.08608)
