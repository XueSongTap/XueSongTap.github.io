---
layout: article
title: 虚拟流水线并行（VPP）：更低气泡，更高吞吐
tags: LLM
---

> 前置阅读：[流水线并行（1F1B）](https://xuesongtap.github.io/2026/03/22/pipeline-parallel.html)

---

## 1. 回顾：1F1B 的气泡极限

1F1B 的气泡比率为：

$$
\text{bubble ratio}_{\text{1F1B}} = \frac{p - 1}{m}
$$

想降低气泡，有两条路：
1. **增大 $m$（micro-batch 数）**：但显存需求也随之增大（更多 in-flight 激活），且每个 micro-batch 变小会降低 GPU 的矩阵乘效率；
2. **减小 $p$（stage 数）**：但这意味着要么缩减模型规模，要么每个 stage 承载更多层——两者都不是我们想要的。

VPP（Virtual Pipeline Parallelism，也叫 Interleaved PP）提供了第三条路：**在不增加物理 GPU 数的前提下，通过更细粒度的层分配降低气泡**。

---

## 2. VPP 的核心思想

### 2.1 层分配方式的变化

设模型有 $L$ 层，物理设备 $p$ 个，每个设备负责 $L/p$ 层（连续块）。

**VPP** 将每台设备负责的层拆成 $v$ 个不连续的**块（chunk）**，每个块负责 $L/(p \cdot v)$ 层（假设 $L$ 整除 $p \cdot v$）：

```
                  [不使用 VPP，v=1，每设备连续 L/p 层]
Device 0:  [层 0 ~ L/p-1]
Device 1:  [层 L/p ~ 2L/p-1]
Device 2:  [层 2L/p ~ 3L/p-1]
Device 3:  [层 3L/p ~ L-1]

                  [使用 VPP，v=2，每设备 2 块，各 L/(2p) 层]
Device 0:  [层 0 ~ L/(2p)-1]  +  [层 L/2 ~ L/2+L/(2p)-1]
Device 1:  [层 L/(2p) ~ L/p-1]  +  [层 L/2+L/(2p) ~ L/2+L/p-1]
...
```

每台设备仍然只持有 $L/p$ 层参数，**不增加模型存储**，但每个 micro-batch 需要在设备间"走 $v$ 轮"。

### 2.2 调度图

以 $p=4$，$m=8$，$v=2$ 为例，每个 micro-batch 要走 $2 \times 4 = 8$ 个 chunk：

```
时间 →

Device 0:  F0₁ F1₁ F2₁ F3₁ F4₁ F0₂ F1₂ ...  B3₂ B2₂ ...
Device 1:       F0₁ F1₁ F2₁ F3₁ F4₁ ...
Device 2:            F0₁ F1₁ ...
Device 3:                 F0₁ ...

（下标 k₁ 表示 micro-batch k 的 chunk 1，k₂ 表示 chunk 2）
```

Interleaved PP 让 steady state 中的调度更密集：设备 0 在做完 chunk 1 的某个 micro-batch 前向后，立即接着做 chunk 2 的前向，这样填满了原本 1F1B 中设备等待其他 stage 时的空档。

---

## 3. 气泡比率公式

Megatron-LM 论文 [1] 给出 VPP 的气泡比率：

$$
\text{bubble ratio}_{\text{VPP}} = \frac{1}{v} \cdot \frac{p - 1}{m} = \frac{p-1}{v \cdot m}
$$

相比 1F1B，**气泡比率缩小了 $v$ 倍**。

**直觉推导**：VPP 相当于把原来 $p$ 个 stage 的流水线，扩展成 $p \times v$ 个逻辑 stage 的流水线，但每个 micro-batch 流经的 stage 变多了 $v$ 倍。在总 micro-batch 数 $m$ 不变的情况下，等待阶段（warmup/cooldown）被更均匀地分摊，气泡被稀释了 $v$ 倍。

### 3.1 数值对比

设 $p=8$，$m=16$，对比不同 $v$：

| 方案 | 气泡比率 |
|------|---------|
| GPipe | ~100%（极端情况）|
| 1F1B（$v=1$）| $(8-1)/16 = 43.75\%$ |
| VPP $v=2$ | $(8-1)/(2 \times 16) = 21.9\%$ |
| VPP $v=4$ | $(8-1)/(4 \times 16) = 10.9\%$ |

---

## 4. 代价：通信开销增加

VPP 并非免费午餐。每个 micro-batch 在 $v$ 轮流经所有设备时，每轮都需要一次 P2P 前向激活传输 + 一次反向梯度传输：

- **基础 PP（$v=1$）**：每个 micro-batch 共 $2p$ 次 P2P 通信（$p$ 个前向 + $p$ 个反向）；
- **VPP（$v$ 块）**：每个 micro-batch 共 $2p \cdot v$ 次 P2P 通信。

**$v$ 越大，通信次数越多**。因此 $v$ 的选择需要在气泡减少和通信开销之间权衡：

```
吞吐增益 ≈ f(m, p, v, bandwidth)
```

实践中 $v = 2$ 或 $v = 4$ 是常见选择。

### 4.1 Megatron-LM 中开启 VPP

```bash
--pipeline-model-parallel-size 8              # p 值
--virtual-pipeline-model-parallel-size 2      # v 值（每设备 chunk 数）
--num-micro-batches 16                        # m 值
```

要求：总层数能被 $p \times v$ 整除。

---

## 5. 进一步优化：DualPipe

DeepSeek-V3 [2] 提出的 **DualPipe** 思路更激进：**同时在两个方向跑流水线**，一组设备跑正向（chunk 0→v-1），另一组跑反向（chunk v-1→0），让前向和反向的通信与计算完全重叠。

关键结论：**DualPipe 的气泡时间几乎为零**，且不增加激活内存。代价是需要在 DeepSeek 的 3FS 存储 + 自研推理框架等整体基础设施的配合下才能高效工作（在单机或通用集群上移植难度较高）。

DeepSeek-V3 使用了 **16-way PP + 64-way EP** 的组合并行，DualPipe 是其中 PP 高效化的关键组件。

---

## 6. Zero Bubble（ZB-H1/ZB-H2）

Qi et al. [3] 指出：1F1B 气泡的根本来源是 **$dX$（输入梯度）和 $dW$（权重梯度）耦合在同一个反向步骤里**。

ZeroBubble 的思路：
- **分离 $dX$ 和 $dW$**：$dX$ 需要尽快做（因为上游 stage 等着），$dW$ 可以推迟到 cooldown 阶段再做；
- 推迟 $dW$ 后，反向路径中 $dX$ 快速"穿过"所有 stage，warmup 的气泡被填满。

结果：**ZB-H1 气泡比率 = 0**，实测比 1F1B 提升约 30% 吞吐（$p=8$，$m=24$）。代价是 $dW$ 堆积，需要在权重更新前确保梯度全部完成。

---

## 7. 横向对比

| 方案 | 气泡比率 | P2P 通信次数 | 实现复杂度 |
|------|---------|------------|---------|
| GPipe | $\approx (p-1)/m$（含 warmup+cooldown）| $2p$ | 低 |
| 1F1B | $(p-1)/m$ | $2p$ | 中 |
| VPP | $(p-1)/(v \cdot m)$ | $2pv$ | 中-高 |
| Zero Bubble | $\approx 0$ | $2p$ | 高 |
| DualPipe | $\approx 0$ | $4pv$ | 极高（需配套基础设施）|

---

## 参考

[1] Narayanan, D., et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC 2021. [arxiv:2104.04473](https://arxiv.org/abs/2104.04473)

[2] DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2024. [arxiv:2412.19437](https://arxiv.org/abs/2412.19437)

[3] Qi, P., et al. *Zero Bubble Pipeline Parallelism.* ICLR 2024. [arxiv:2401.10241](https://arxiv.org/abs/2401.10241)
