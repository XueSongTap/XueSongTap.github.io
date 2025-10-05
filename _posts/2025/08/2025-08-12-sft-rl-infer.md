---
layout: articles
title: 为什么相比于SFT训练，RL训练更依赖推理引擎
tags: RL
---


在大模型的训练中，常见的 RL（Reinforcement Learning）训练框架往往对推理引擎（如 vLLM、SGLang）依赖更强，甚至会将其作为安装和运行的必选组件。

核心原因很简单：**SFT 在训练时只需要一次性计算 logits（Prefill），而 RL 的 rollout 阶段必须走完整的 Prefill + Decode 推理流程**

---

## 1. Prefill vs Decode

在 Transformer 推理中，我们可以把计算过程分为两种模式：

| 阶段          | 输入内容                          | 输出内容              | 并行性           |
| ----------- | ----------------------------- | ----------------- | ------------- |
| **Prefill** | 一次性输入整个序列（prompt + 已知目标）      | 输出所有位置的 logits    | 高             |
| **Decode**  | 输入是上一步生成的 token（加历史 KV Cache） | 当前 token 的 logits | 低（逐 token 串行） |

---

## 2. SFT 的训练流程：只用 Prefill

SFT 的训练数据是成对的 `(prompt, target)`，目标序列是已知的：

1. 一次性将 `prompt + target` 输入模型。
2. 模型一次前向计算得到所有位置的 logits。
3. 直接计算 loss（预测 vs target），反向更新。

**特点：**

* **无需生成**：目标 token 已知，不需要逐步 decode。
* **计算全并行**：一次 Prefill 就能得到所有结果。
* **推理速度不是瓶颈**：主要计算耗时在反向传播（Backward）。

---

## 3. RL 的训练流程：必须 Prefill + Decode

RL 训练（例如 RLHF）中的 rollout 阶段，需要模型**自由生成**未知的输出：

1. **Prefill**：输入 prompt，得到初始 context 的 logits。
2. **Decode**：逐 token 生成输出，每一步的输入依赖上一步生成的 token。
3. 收集生成轨迹（trajectory），再进入奖励（Reward）计算与策略更新。

**特点：**

* **目标未知**：必须自回归生成，无法一次性并行算出所有 token。
* **逐步串行**：每个 decode step 都依赖前一步输出，速度慢。
* **推理是瓶颈**：rollout 阶段往往占据 RL 训练的大部分时间。

---

## 4. 为什么 RL 更依赖推理引擎

由于 rollout 的推理过程包含大量逐 token 的 decode，RL 对推理性能的要求非常高：

* **高并发**：需要同时处理大量 prompt rollout（continuous batching）。
* **低延迟**：减少 decode 步之间的空转时间。
* **显存优化**：利用 KV Cache、PagedAttention 等技术降低显存占用。
* **分布式支持**：方便跨卡、跨节点 rollout 生成。

vLLM、SGLang 等推理引擎正是针对这些需求优化的，因此在 RL 训练中几乎是标配。
