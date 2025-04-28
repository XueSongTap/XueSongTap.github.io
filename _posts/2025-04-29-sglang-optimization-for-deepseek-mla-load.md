---
layout: articles
title: sglang 对 DeepSeek的 MLA的矩阵乘 load阶段融合的优化
tags: sft
---



### 主要优化点
**PR标题**:  
> Fuse `q_a_proj` and `kv_a_proj` for DeepSeek models

**动机（Motivation）**：  
在 DeepSeek V3 模型的 Self-Attention 结构中，  
`q_a_proj`（query after attention projection）和 `kv_a_proj`（key-value after attention projection）都以同一个 hidden state 作为输入。  
因此可以**把它们合并为一个模块**，  
这样：
- 只需要 **一次 DeepGemm 调用**（矩阵乘），  
- 减少了一次 kernel launch 开销，
- 提升整体推理速度。

---

### 具体修改（Modifications）
- 当 `q_lora_rank > 0`（DeepSeek V3 和 DeepSeek R1 确实满足这个条件）时：
  - `self.q_a_proj` 和 `self.kv_a_proj_with_mqa` 合并成新的 `self.fused_qkv_a_proj_with_mqa`。
- 加载权重时：
  - 把 `q_a_proj` 和 `kv_a_proj` 的权重（weights）和 block scales **拼接（concatenate）**后加载到新的 fused 模块里。

---

### 准确率验证（Accuracy Test）
对比基线，性能没有下降，准确率保持稳定：

| Benchmark | 准确率 (Accuracy) |
| :--- | :--- |
| GSM8K（数学推理） | 0.951 |
| MMLU（多领域理解） | 0.875 |

---

### 🚀 性能提升（Benchmark Results）

对比主分支（main branch）和这个 PR：

| Batch Size | 原来 Latency | 优化后 Latency | Latency 降低 | Throughput 提升 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 10.85s | 10.41s | ↓ 4.2% | ↑ |
| 32 | 16.83s | 16.21s | ↓ 3.8% | ↑ |

特别是 **小 batch size（bs=1）** 提升明显，说明单次推理（如在线服务）也能受益。

---

### 总结一句话
> **将原来两个独立的投影操作 `q_a_proj` 和 `kv_a_proj` 融合成一个，共享一次矩阵乘，加速推理，且准确率不受影响。**

---

要不要我顺便也帮你画一下这个合并优化前后的结构变化示意图？这样可能更直观。要的话告诉我！🎯  
要继续的话，也可以说下要不要顺便解释一下为什么这样融合可以减少 DeepGemm 的开销？（可以更深入一点）