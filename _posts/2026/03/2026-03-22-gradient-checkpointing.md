---
layout: article
title: 梯度检查点（Gradient Checkpointing）：用计算换显存
tags: LLM
---

> 关联阅读：[训练中的显存优化](https://xuesongtap.github.io/2025/10/25/memory-optim-in-training.html) | [ZeRO 优化](https://xuesongtap.github.io/2026/03/22/zero-redundancy-optimizer.html)

---

## 1. 问题：反向传播需要存激活

标准反向传播（Backpropagation Through Time / Chain Rule）需要在反向时访问每层前向传播的**中间激活（intermediate activations）**。

以一个 $n$ 层的模型为例：

```
前向：x₀ → [Layer 1] → x₁ → [Layer 2] → ... → xₙ₋₁ → [Layer n] → xₙ = loss

反向需要：
  dL/dx_{n-1} 需要 x_{n-1}（Layer n 的输入）
  dL/dx_{n-2} 需要 x_{n-2}（Layer n-1 的输入）
  ...
  dL/dx₀     需要 x₀
```

**全部存下来（Vanilla Backprop）**：
- 显存 $O(n)$：$n$ 层的激活全部驻留显存，直到该层反向传播结束才释放；
- 对于 LLM，激活占用随序列长度、batch 大小线性增长，是大 batch 训练的主要显存瓶颈。

---

## 2. Gradient Checkpointing 的核心思想

Chen et al. [1] 提出：**不保存所有中间激活，只在某些层（checkpoint）保存，反向时重新计算其余层**。

### 2.1 两个极端

**极端 1：全保存（Vanilla Backprop）**
- 显存 $O(n)$，计算 $O(n)$（每层算一次）

**极端 2：全不存（Memory-Poor Backprop）**
- 显存 $O(1)$（只存当前层激活），计算 $O(n^2)$（反向传播第 $k$ 层时，要从第 0 层重新前向到第 $k$ 层）

### 2.2 最优折中：$\sqrt{n}$ 分段

把 $n$ 层分成 $\sqrt{n}$ 段，每段 $\sqrt{n}$ 层：
- **只在段边界保存激活**（checkpoint），共 $\sqrt{n}$ 个；
- 反向到某段时，**从该段的入口 checkpoint 重新前向**计算该段内所有层的激活；
- 每段内的激活在反向完成后立刻丢弃。

**复杂度**：
- 显存：$O(\sqrt{n})$（$\sqrt{n}$ 个 checkpoint，每个大小为一层激活）
- 计算：$O(n)$（前向走一遍）+ $O(n)$（每段重计算一遍，共 $\sqrt{n}$ 段 × $\sqrt{n}$ 层）= $O(n)$，系数约 **2×**

---

## 3. LLM 中的实际应用

现代 LLM 框架中，Gradient Checkpointing 通常以**每个 Transformer Block**为粒度做 checkpoint，而不是严格的 $\sqrt{n}$ 分段。

### 3.1 显存节省

以 LLaMA 7B 为例（$n=32$ 层，seq=2048，batch=4）：

| 方案 | 激活显存估算 |
|------|------------|
| 全保存 | ~32 GB |
| 按 Block checkpoint | ~2-4 GB（只存层边界激活）|
| 节省比例 | **~90%** |

代价：训练吞吐降低约 **30-33%**（每层 forward 多算一次）。

### 3.2 PyTorch 中使用

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        # 不使用 checkpoint（保存所有激活）
        return self.attn(x) + self.ffn(x)

# 使用 checkpoint（反向时重计算）
output = checkpoint(block.forward, x, use_reentrant=False)
```

Hugging Face Transformers：

```python
model.gradient_checkpointing_enable()
```

### 3.3 Megatron-LM

Megatron 提供三个级别：

```bash
# 每个 transformer layer 做 checkpoint（最省显存）
--recompute-granularity full --recompute-method uniform

# 只对 core attention 做 checkpoint（省部分显存，计算代价小）
--recompute-granularity selective

# 不做 checkpoint（最快）
# 默认不开
```

`selective` 模式只重计算 attention 的中间矩阵（$QK^T$ softmax 结果），其他激活仍保留，在速度和显存间取得更好的平衡。

---

## 4. 与 Pipeline Parallel 的交互

在流水线并行（PP）中，每个 stage 在 1F1B steady state 下最多持有 $p$ 个 micro-batch 的激活（$p$ 为 stage 数）。

开启 Gradient Checkpointing 后，**每个 stage 不需要存每层的激活**，只存该 stage 的输入（来自上游 stage 的激活）——这大幅降低了 PP 中的 in-flight 激活显存：

```
显存占用（每 micro-batch，每 stage）:

不开 checkpoint：num_layers_per_stage × per_layer_activation
开 checkpoint：   1 × stage_input_activation（重计算该 stage 内所有层）
```

这使得在 PP 下能用更大的 $p$（更多 stage）而不爆显存，从而降低每 stage 的层数。

---

## 5. 与 ZeRO 的组合

ZeRO 减少**参数、梯度、优化器状态**的冗余，Gradient Checkpointing 减少**激活**的冗余，两者互补：

| 显存来源 | 优化手段 |
|---------|---------|
| 参数 | ZeRO-3 / FSDP |
| 梯度 | ZeRO-2/3 |
| 优化器状态 | ZeRO-1/2/3 |
| **激活** | **Gradient Checkpointing** |
| 临时 buffer | ZeRO-R CB |

生产环境中，通常同时开 ZeRO-3 + Gradient Checkpointing，这是训练超大模型时最常见的组合。

---

## 参考

[1] Chen, T., et al. *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174, 2016. [arxiv](https://arxiv.org/abs/1604.06174)

[2] Korthikanti, V., et al. *Reducing Activation Recomputation in Large Transformer Models.* MLSys 2023. [arxiv:2205.05198](https://arxiv.org/abs/2205.05198)（Megatron selective recomputation）
