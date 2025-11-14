---
layout: article
title: TP 下的激活显存公式
tags: ModelParallel
---

大模型训练时，激活显存往往比参数显存更先成为瓶颈。为了在同样的硬件预算下堆更长的序列或更大的 batch，我们会把 Transformer 的矩阵乘拆到多张 GPU 上，也就是常说的 **Tensor Parallel (TP)**。这篇笔记把原始草稿整理成一篇小博文，记录 TP 如何影响每层激活的内存占用，以及从公式里可以读出的工程直觉。


## 1 每层激活显存的估算式

$$
\text{Activations memory per layer} = s b h \left( 10 + \frac{24}{t} + 5 \frac{a s}{h t} \right)
$$

这个式子描述了 **单层 Transformer 激活显存** 与多种变量之间的关系：序列长度 $s$、micro-batch 大小 $b$、隐藏维度 $h$、注意力头数 $a$，以及张量并行度 $t$。当 $s$、$b$、$h$ 任一放大时，整个式子会线性增长；只有分到多张 GPU 的部分会对 $t$ 产生反比关系。

### 1.1 变量说明

| 符号 | 含义 |
| --- | --- |
| **s** | 序列长度（sequence length） |
| **b** | micro-batch 大小 |
| **h** | hidden size（隐藏维度） |
| **a** | 注意力头数（attention heads） |
| **t** | tensor parallel size（张量并行的 GPU 数） |
| **p** | pipeline parallel size（流水线并行度） |
| **L** | Transformer 层数 |
| **v** | 词表大小（vocab size） |

## 2 公式拆解

### 2.1 LayerNorm / Dropout：固定项 `10`

> “The remaining 10 term is for the LayerNorm (4sbh), Dropout (2sbh), and inputs to the attention and MLP (4sbh).”

这些激活无法被 TP 分摊：LayerNorm、Dropout 以及送入注意力 / MLP 的输入都需要完整的 hidden vector，每张卡都必须保留一份。因此 **无论 t 多大，该项恒等于 `10·sbh`**，也是 TP 无法触碰的显存下界。

### 2.3 Attention / MLP 的中间激活：`24/t`

自注意力和 FFN 的矩阵乘能被拆分到多张 GPU，激活也随之切片。理论上，`t` 张卡会把这部分均匀分担成 `1/t`，所以显存随并行度成反比下降。这也是我们在实践中提升 TP 数量后最直观能看到的收益：Attention/MLP 中间态不再集中到单卡。

### 2.4 Attention Map：`5 · (a s) / (h t)`

注意力权重矩阵（softmax(QKᵀ)）会占用额外显存。由于 head 数为 `a`，每个 head 需要存一个 `s × s` 的矩阵，在 TP 下可以按 head 切分，让每张卡负责 `a/t` 个头。该项因此包含 `5as/(ht)`，说明**序列越长、头越多，attention map 的内存越难压缩**，但增加并行度仍可提供线性回报。

## 3 综合直觉

- **固定成本占比高**：LayerNorm、Dropout、输入缓存约占 `10sbh`，TP 对它们无能为力，决定了激活显存不会随并行度无限下降。
- **可压缩项随 `1/t` 衰减**：Attention 与 MLP 的中间激活是主要的可优化部分，24/t 一项告诉我们只要能分片，显存就能几乎线性缩小。
- **注意力图依赖 head 与长度**：`5as/(ht)` 揭示了序列长度和 head 数摸高时，attention map 可能变成第二大瓶颈，这也是长序列模型常配合 FlashAttention、序列并行的原因。

| 项目 | 是否随 TP 增大而下降 | 说明 |
| --- | --- | --- |
| LayerNorm / Dropout / 输入 | 否 | 每张卡都需要全量 hidden，固定成本 |
| Attention / MLP 中间激活 | 是 | 张量并行按 `1/t` 切片 |
| Attention map | 是 | head 被切分后按 `1/t` 缩小 |
| 总体激活显存 | 部分下降 | 固定项仍然存在 |

## 4 工程提示

- **TP 与其他手段配合**：既然有固定项，就需要与流水线并行、序列并行或激活检查点结合，才能在更大批量下稳定训练。
- **关注 head 与长度**：当 `a` 或 `s` 很大时，优先优化 attention map，比如调低 head 数、引入块状注意力，或者把 KV cache 做压缩（如 MLA）。
- **估算前提清晰**：该公式假设每张 GPU 均匀分到 `1/t` 的工作，实战中通信、padding、容量裁剪都会引入额外常数，应该把它视作 trend 指标而非精确 profiler。

TP 确实能帮忙压缩激活，但它只解决了“能拆分的那一部分”。理解这条公式后，就能迅速判断下一步是加卡继续分片，还是回头从 LayerNorm/输入缓存等固定项着手做进一步的内存优化
