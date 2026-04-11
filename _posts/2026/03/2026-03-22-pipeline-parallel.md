---
layout: article
title: 流水线并行（Pipeline Parallelism）：1F1B 调度与气泡分析
tags: LLM
---

> 本文是分布式训练并行系列的一篇。前置背景可参考：
> - [分布式模型并行](https://xuesongtap.github.io/2025/10/21/distribute-model-parallel.html)（含 PP 朴素方案 + GPipe 概览）
> - [NCCL 通信原语](https://xuesongtap.github.io/2025/12/29/comm.html)

---

## 1. 为什么需要流水线并行

单机模型并行（Naïve Model Parallel）把模型不同层放在不同 GPU，同一时刻只有一张卡在算，其余全部空等——**GPU 利用率极低**。

GPipe [1] 提出用 **micro-batch** 填充流水线空隙：把一个 mini-batch 切成 $m$ 份 micro-batch，各阶段可以流水线化执行。但 GPipe 的 "全 F 全 B" 策略导致：
1. **显存峰值高**：设备 0 要把所有 $m$ 个 micro-batch 的激活值全存到反向传播结束；
2. **气泡（Bubble）时间大**：前向全部跑完才开始反向，等待时间约 $(p-1)$ 个 stage 的前向 + 反向时间。

1F1B（One Forward One Backward）调度[2] 是 GPipe 的工程替代方案，也是 Megatron-LM 的默认 PP 实现，显著改善了这两个问题。

---

## 2. 1F1B 调度

### 2.1 核心思想

**不等全部前向完成，尽量早开始反向**。具体规则：

- **Warmup 阶段**：阶段 $i$（从 0 开始）先跑 $p - i$ 个 micro-batch 的前向（其中 $p$ 是 stage 总数），填充流水线；
- **Steady state**：此后每做完一个 micro-batch 的前向，立刻做一个（更早的）micro-batch 的反向，保持 in-flight 激活数量恒为 $p$；
- **Cooldown 阶段**：流水线排空，只做反向。

```
时间轴（p=4, m=8）:

Device 0: F0  F1  F2  F3  F4  F5  F6  F7  B7  B6  B5  B4  B3  B2  B1  B0
Device 1:     F0  F1  F2  F3  F4  F5  F6  F7  B7  B6  B5  B4  B3  B2  B1  B0
Device 2:         F0  F1  F2  F3  F4  F5  F6  F7  B7  B6  ...
Device 3:             F0  F1  F2  F3  F4  F5  F6  F7  B7  ...
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^               ^^^^^^
                     Steady State（1F1B）                 Cooldown
```

### 2.2 气泡比率（Bubble Ratio）

**GPipe**（All-F-then-All-B）的气泡时间约为 $2(p-1)$ 个单元时间（前向 + 反向各 $(p-1)$ 个 stage 等待）。

**1F1B**（PipeDream-Flush）的气泡比率：

$$
\text{bubble ratio} = \frac{p - 1}{m}
$$

其中 $p$ 是 pipeline stage 数，$m$ 是 micro-batch 数。

**推导**：
- Warmup 阶段，设备 $p-1$（最后一台）等了 $(p-1)$ 个前向时间才接到数据；
- Cooldown 阶段，设备 $0$（第一台）在最后 $(p-1)$ 个反向里无法叠上新前向；
- 总气泡时间为 $(p-1) \cdot (t_f + t_b)$，理想总时间为 $m \cdot (t_f + t_b)$。

因此，当 $m \gg p$ 时气泡可以压得很低。实践中通常要求 $m \geq 4p$。

### 2.3 内存分析

| 方案 | 最大 in-flight 激活 | 显存主要瓶颈 |
|------|-------------------|-----------|
| Naïve MP | 整个 batch | 严重 OOM |
| GPipe | $m$ 个 micro-batch 全部 | 随 $m$ 线性增长 |
| 1F1B | $p$ 个 micro-batch | 固定为 $p$，与 $m$ 无关 |

1F1B 的关键优势：**把内存峰值从 $O(m)$ 压到 $O(p)$**。

---

## 3. 关键实现细节

### 3.1 激活值的生命周期

1F1B 中，每个 stage 在 steady state 需要同时持有最多 1 个 micro-batch 的激活值供反向使用（其它的要么已释放，要么还未产生）。实际实现中会用 **double buffering** 存两个激活以隐藏通信延迟：做 micro-batch $k$ 的反向时，把 micro-batch $k+1$ 的激活通过 P2P 通信提前 prefetch 进来。

### 3.2 P2P 通信

相邻 stage 间通过 `send_forward` / `recv_forward` / `send_backward` / `recv_backward` 四类 P2P 操作传激活和梯度：

```python
# Megatron-LM schedules.py（简化版）
# 前向：从上游 stage 接收激活，计算，发给下游
input_tensor = recv_forward(recv_tensor_shapes, config)
output_tensor = forward_step(input_tensor, ...)
send_forward(output_tensor, config)

# 反向：从下游接梯度，反向计算，发梯度给上游
output_tensor_grad = recv_backward(send_tensor_shapes, config)
input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)
send_backward(input_tensor_grad, config)
```

Megatron-LM 使用非阻塞（`isend`/`irecv`）实现通信计算重叠。

### 3.3 梯度累积与 Optimizer Step

多个 micro-batch 的梯度在同一个 pipeline stage 上累积，只在整个 batch（所有 $m$ 个 micro-batch）的反向都完成后才做一次 `optimizer.step()`，确保参数更新同步（PipeDream-Flush 中无 weight staleness 问题）。

---

## 4. 配置建议

Megatron-LM 开启 PP 的核心参数：

```bash
--pipeline-model-parallel-size 4    # p 值
--num-micro-batches 8               # m 值，建议 m >= 4*p
```

组合并行下的经验法则：
- **PP** 用于节点间（通信量小，只传激活值，带宽敏感度低）；
- **TP** 用于节点内（每层都有 AllReduce，依赖 NVLink 高带宽）；
- **DP** 在剩余 GPU 上扩展吞吐。

---

## 5. 局限与改进方向

1. **气泡无法为零**：1F1B 的气泡比率 $(p-1)/m$ 在 $p$ 大时仍可观。更大的 $m$ 可以降低气泡，但会增加显存（每个 stage 持有更多 micro-batch 的激活）。
2. **Warmup/Cooldown 不对称**：激活内存在 warmup 阶段快速积累，到 steady state 才稳定。
3. **改进方向**：
   - **VPP（Interleaved PP）**：通过让每个设备负责多个不连续层，进一步降低气泡比率（见下篇）；
   - **DualPipe（DeepSeek-V3 [3]）**：双向流水线，实现近零气泡，同时不增加激活内存。
   - **ZeroBubble [4]**：通过将权重梯度计算（`dW`）从激活梯度计算（`dX`）中分离，实现真正零气泡。

---

## 参考

[1] Huang, Y., et al. *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.* NeurIPS 2019. [arxiv:1811.06965](https://arxiv.org/abs/1811.06965)

[2] Narayanan, D., et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC 2021. [arxiv:2104.04473](https://arxiv.org/abs/2104.04473)

[3] DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. [arxiv:2412.19437](https://arxiv.org/abs/2412.19437)

[4] Qi, P., et al. *Zero Bubble Pipeline Parallelism.* ICLR 2024. [arxiv:2401.10241](https://arxiv.org/abs/2401.10241)
