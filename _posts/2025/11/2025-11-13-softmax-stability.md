---
layout: article
title: Softmax 数值稳定性的危机
tags: Transformer Stability
---

大模型训练里，最常被忽略的炸弹往往埋在最熟悉的算子下。Softmax 同时包含指数与除法，一旦输入 logits 偏离正常尺度，就会把梯度链条整段炸成 NaN。本文把课堂随笔整理成一篇可查的博客，集中梳理 softmax 在注意力与输出层中的不稳定来源，以及工业界常用的工程缓解手段

## 1 Softmax 为什么会失稳

Softmax 把向量 $\mathbf{z}$ 转成概率分布：
$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

在自注意力中它直接作用在 $\frac{QK^\top}{\sqrt{d_k}}$ 上。问题在于，$QK^\top$ 的取值范围会随 hidden size、head dim、训练阶段而大幅波动，导致 softmax 出现三个常见故障。

![softmax-structure](/img/2025/11/softmax_structure.png)

### 1.1 指数放大

- 当 $z_i \gg 0$ 时，$e^{z_i}$ 在 fp16/bf16 精度下极易溢出（例如 $e^{100} \approx 2.7 \times 10^{43}$）。
- 溢出会直接把 softmax 输出推到 NaN，或者形成极端的 one-hot 分布，使梯度尖锐到无法训练。

### 1.2 分母极小

- 如果所有 logits 都是大负数（例如 $-1000$），$e^{z_j}$ 几乎为 0，分母接近 0。
- 这种 underflow 让概率分布失效，反向传播中的 $\frac{1}{\sum e^{z_j}}$ 放大成巨量噪声，梯度直接 NaN。

### 1.3 低精度累积误差

- 混合精度训练（尤其是 H100/A100 上的 bf16/fp16）会压缩指数的动态范围。
- 如果实现中忘了做“减最大值”操作，softmax 在 attention 层会频繁崩溃，训练日志里常见到 `grad norm is NaN`。

## 2 注意力模块的工程防线

注意力矩阵在每个 token、每个 head、每个 batch 都要算一次 softmax，一次失稳就会把整条流水线污染。下面这几个做法几乎是所有现代框架的必备选项。

### 2.1 减最大值（Max Subtraction）

$$
p_i = \frac{e^{z_i - \max_j z_j}}{\sum_j e^{z_j - \max_j z_j}}
$$

把最大值平移到 0，可以同时避免 overflow 与 underflow。PyTorch、Megatron、Transformer Engine、FlashAttention 都默认执行这一操作。

### 2.2 缩放与范数控制

- 通过 $\frac{1}{\sqrt{d_k}}$ 缩放 $QK^\top$，防止 head dim 增长带来的能量放大。
- 保持权重与梯度的范数可控（gradient clipping、精细的学习率 warmup）能避免 logits 在训练中无穷放大。

### 2.3 Mask 与可疑值拦截

- 在 softmax 前屏蔽 padding、未来位置或无效 token，避免这些位置的 logits 干扰归一化。
- 高端实现会在 kernel 内做 NaN/Inf 检测，第一时间把异常 batch 丢弃或回退。

### 2.4 精度策略

- 采用 **fp32 accumulator** 或者 “compute in fp32, store in bf16”。
- FlashAttention 等实现把归一化核心逻辑保留在高精度路径，避免混合精度下指数爆炸。

### 2.5 梯度裁剪

梯度裁剪无法直接修复 softmax，但它能阻断“梯度 → 权重 → logits”这条放大链。大部分 LLM 都保持全局梯度范数在 0.5～1.0 区间。

### 2.6 对照清单

| 问题来源 | 数学原因 | 结果 | 解决方式 |
| --- | --- | --- | --- |
| 指数过大 | $e^{z_i}$ 溢出 | NaN / Inf | 减最大值、缩放、范数控制 |
| 指数过小 | $e^{z_i} \approx 0$ | 分母近 0 | 稳定化计算、mask |
| 精度不足 | fp16 动态范围窄 | 随机 NaN | fp32 accumulator |
| logits 失控 | 权重/梯度太大 | one-hot attention | 梯度裁剪、正则化 |

## 3 输出层的 z-loss：控制归一化常数

语言模型在输出层仍然使用 softmax：
$$
P(x) = \frac{e^{U_r(x)}}{Z(x)}, \quad Z(x) = \sum_{r'} e^{U_{r'}(x)}
$$

当 vocab 达到 128k、参数尚未收敛或使用低精度训练时，$Z(x)$ 很容易呈指数级发散。PaLM 引入的 **z-loss** 通过惩罚 $\log Z$ 偏离 0 来约束 logits 的整体尺度：
$$
L = \text{CE}(x) + \alpha \cdot \big(\log Z(x)\big)^2
$$

- $\alpha$ 一般取 $10^{-4}$，足够温和，不会改变原始交叉熵的最优点。
- 训练时模型被迫让 $\log Z \approx 0$，等价于 $Z \approx 1$，整体概率分布保持在可控范围。
- PaLM、Baichuan 2、DeepSeek DCLM、OLMo 2 都报告 z-loss 能减少 NaN 与梯度爆炸。

```python
log_z = torch.logsumexp(logits, dim=-1)
z_loss = alpha * (log_z ** 2).mean()
ce_loss = F.cross_entropy(logits, targets)
total_loss = ce_loss + z_loss
```

## 4 QK Norm：在 softmax 前归一化 Query/Key

最早在视觉/多模态模型（Dehgani 2023, Idefics, Chameleon）中流行的 **QK norm**，现在也成了语言模型的稳定性工具。核心做法是对 $Q$、$K$ 做 LayerNorm 或 RMSNorm，再送入 softmax：
$$
\tilde{Q} = \text{Norm}(Q), \quad \tilde{K} = \text{Norm}(K)
$$
$$
\text{Attention} = \text{softmax}\left(\frac{\tilde{Q}\tilde{K}^\top}{\sqrt{d_k}}\right)V
$$

- 归一化后，$QK^\top$ 的数值范围被锁定在一个可控的壳层里，减少极端头的出现。
- DCLM、OLMo 2、Gemma 2 都把 “QK RMSNorm” 当作默认配置，特别适合长上下文或 FP8 推理。

## 5 Logit Soft-Capping：用 Tanh 平滑封顶

随着模型尺寸增大，即便做了缩放和归一化，仍可能出现个别极大 logits。**Soft-capping** 使用平滑函数（常见是 $\tanh$）把 logits 融化到一个上限，而不是硬裁剪：
$$
y = \alpha \cdot \tanh\left(\frac{x}{\alpha}\right)
$$

- 当 $|x| \ll \alpha$ 时几乎不受影响；当 $|x| \gg \alpha$ 时以平滑方式逼近 $\pm \alpha$。
- 该技巧可以用于 attention logits 或语言头 logits。Gemma 2、OLMo 2、DCLM、部分 DeepSeek/Llama3 变种都在关键路径上做 soft-capping。
- 与硬 `clamp` 相比，tanh 保持可导且不会制造额外拐点，训练更平滑。

```python
max_logit = 20.0
logits = torch.tanh(logits / max_logit) * max_logit
probs = torch.softmax(logits, dim=-1)
```

## 6 实战检查清单

- Attention logits 是否在 kernel 内减过最大值，并使用 fp32 accumulator？
- 是否对 Q、K 做 RMSNorm/QK norm，或者至少保证初始化方差受控？
- 梯度裁剪、learning-rate schedule、权重衰减是否限制了 logits 的整体尺度？
- 输出层是否考虑 z-loss 或 soft-capping，尤其在大词表、低精度场景？
- 遇到 NaN 时，日志里是否保留了 `logsumexp`、`grad norm` 的观测指标，方便定位是哪一级的 softmax 崩溃？

只要把这些“隐形工程项”列入 checklist，softmax 就不再是训练曲线里随时引爆的未爆弹，而会成为一段可预测、可调优的算子。
