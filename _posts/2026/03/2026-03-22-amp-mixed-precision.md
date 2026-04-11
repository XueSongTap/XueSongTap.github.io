---
layout: article
title: 混合精度训练（AMP）：从 FP32 到 BF16
tags: LLM
---

> 关联阅读：[FP8 训练](https://xuesongtap.github.io/2025/11/03/fp8.html) | [Float16Module](https://xuesongtap.github.io/2025/11/29/float16module.html)

---

## 1. 为什么要降低精度

全精度（FP32）训练的问题：
- **显存**：一个 1B 参数模型，FP32 参数占 4GB，再加上同等量级的梯度和优化器状态，显存开销达 16GB+；
- **带宽**：FP32 比 BF16 多占 2× 带宽，矩阵乘法的吞吐受限于内存带宽时损失明显；
- **硬件利用率**：A100/H100 的 Tensor Core 在 BF16/FP16 下的峰值算力是 FP32 的 **2-4×**。

---

## 2. 数值格式基础

| 格式 | 符号 | 指数 | 尾数 | 数值范围 | 特点 |
|------|------|------|------|---------|------|
| FP32 | 1 | 8 | 23 | ±3.4×10³⁸ | 精度最高，显存最大 |
| FP16 | 1 | 5 | 10 | ±65504 | 范围窄，易溢出 |
| BF16 | 1 | 8 | 7 | ±3.4×10³⁸ | 范围=FP32，精度低，LLM 首选 |

**BF16 vs FP16**：BF16 把 FP32 的指数位保留下来，牺牲尾数精度换来更宽的数值范围。LLM 训练中权重/梯度的绝对值分布范围大，BF16 几乎不出现 overflow/underflow，而 FP16 需要额外的 loss scaling 来避免梯度下溢。

---

## 3. 混合精度训练（AMP）的核心思路

AMP [1] 不是把所有计算都降精度，而是**分类处理**：

| 操作 | 精度 | 原因 |
|------|------|------|
| **前向计算（矩阵乘、卷积）** | BF16/FP16 | 利用 Tensor Core 加速，精度损失可接受 |
| **参数主拷贝（Master Weights）** | FP32 | 梯度累积时精度损失小，优化器更新稳定 |
| **梯度累积** | FP32（或 BF16） | 多 micro-batch 累积时保证精度 |
| **LayerNorm / Softmax** | FP32 | 数值稳定性要求高（归约操作敏感）|
| **损失缩放（仅 FP16）** | — | 防止梯度下溢（BF16 通常不需要）|

### 3.1 Master Weight 机制

```
参数更新流程（以 AdamW 为例）:

BF16 权重（前向/反向）
    ↓ 反向传播得到 BF16 梯度
    ↓ 梯度转 FP32 累积
    ↓ AdamW 在 FP32 下更新一阶/二阶矩阵和主权重
    ↓ 主权重转 BF16，覆盖前向用的 BF16 权重
```

优化器状态（m、v）常驻 FP32，占额外 8 bytes/参数。这是 AMP 相比纯 FP16/BF16 的主要额外显存来源。

### 3.2 Loss Scaling（仅 FP16 需要）

FP16 的最小正规数约为 6×10⁻⁵，LLM 训练中梯度经常比这更小，导致 FP16 下溢为 0（梯度消失）。

解决方案：在损失值上乘以一个大的缩放因子 $S$（如 $2^{15}$），使梯度"放大"后不会下溢；更新参数前再除以 $S$ 还原真实梯度。

```python
# PyTorch AMP（FP16）
scaler = torch.amp.GradScaler()
with torch.amp.autocast('cuda', dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**BF16 不需要 GradScaler**，直接：

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
loss.backward()
optimizer.step()
```

---

## 4. Megatron-LM 中的混合精度

Megatron 把 AMP 封装在 `Float16Module` 中（另见 [Float16Module 详解](https://xuesongtap.github.io/2025/11/29/float16module.html)），核心逻辑：

1. **参数降精度**：把模型权重从 FP32 转到 BF16/FP16 存储（前向用）；
2. **保留主权重**：在 `DistributedOptimizer`/`optimizer_state` 里保留 FP32 主拷贝；
3. **梯度 bucket**：梯度以 BF16 传输（AllReduce），在 FP32 主权重上累积；
4. **数值敏感算子**：通过 `autocast` 的 op-level 配置，让 LayerNorm / Softmax / CrossEntropy 等在 FP32 下计算。

Megatron 启用参数：

```bash
--bf16                          # 使用 BF16（推荐，H100/A100）
--fp16                          # 使用 FP16（需配合 loss scale）
--loss-scale 4096               # 静态 loss scale（FP16 专用）
--loss-scale-window 200         # 动态 loss scale 窗口
```

---

## 5. 显存节省分析

以 7B 模型为例（BF16 训练，AdamW 优化器）：

| 组件 | 精度 | 大小 |
|------|------|------|
| 参数 | BF16 | 7B × 2B = **14 GB** |
| 梯度 | BF16 | 14 GB |
| 优化器一阶矩 m | FP32 | 28 GB |
| 优化器二阶矩 v | FP32 | 28 GB |
| Master weights | FP32 | 28 GB |
| **合计** | | **~112 GB** |

纯 FP32 训练需要 ~224 GB，AMP 节省约 **50%**。结合 ZeRO-3 / FSDP 可进一步把每卡显存压到极低。

---

## 6. BF16 的限制与注意事项

1. **精度敏感操作慎用 BF16**：BF16 尾数只有 7 位（十进制约 2-3 位），对于需要高精度累积的归约操作（如跨很多 token 的 softmax），建议保留 FP32；
2. **LLM 训练中 BF16 基本够用**：大量实验表明，用 BF16 训练的 LLM 与 FP32 在损失曲线上几乎一致，BF16 已成为 LLM 预训练的默认选项；
3. **注意力机制的精度**：FlashAttention 默认在 BF16/FP16 下运行，但其内部使用了 FP32 的归约中间值（online softmax），保证了数值稳定。

---

## 参考

[1] Micikevicius, P., et al. *Mixed Precision Training.* ICLR 2018. [arxiv:1710.03740](https://arxiv.org/abs/1710.03740)

[2] PyTorch AMP 文档：[torch.amp.autocast](https://docs.pytorch.org/docs/stable/amp.html)

[3] Kalamkar, D., et al. *A Study of BFLOAT16 for Deep Learning Training.* 2019. [arxiv:1905.12322](https://arxiv.org/abs/1905.12322)
