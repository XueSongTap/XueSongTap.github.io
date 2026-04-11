---
layout: article
title: ZeRO：从优化器状态到残差状态的显存优化
tags: LLM
---

> 关联阅读：[ZeRO 作为数据并行](https://xuesongtap.github.io/2025/11/17/zero-as-dp.html) | [训练中的显存优化](https://xuesongtap.github.io/2025/10/25/memory-optim-in-training.html)

---

## 1. 训练显存的组成

训练一个参数量为 $\Psi$ 的模型，全精度 Adam 优化器（FP16 混合精度）下，显存占用主要来自：

| 组件 | 精度 | 大小（bytes/参数）|
|------|------|-----------------|
| 参数（Master Weights）| FP32 | 4 |
| 梯度 | FP16 | 2 |
| 一阶矩 m | FP32 | 4 |
| 二阶矩 v | FP32 | 4 |
| FP16 参数副本 | FP16 | 2 |
| **合计（参数相关）** | | **16 bytes/参数** |

还有两类常被忽视的显存来源：
- **激活值（Activations）**：前向传播时保留，供反向传播使用，随 batch 大小和序列长度增长；
- **临时缓冲区（Temporary Buffers）**：梯度聚合、all-reduce 等操作的工作空间。

ZeRO [1] 分两个阶段针对这两类来源做优化。

---

## 2. ZeRO-DP：消除参数冗余

ZeRO-DP 的核心思路是：在数据并行（DP）的 $N_d$ 张卡中，不必每张卡都持有**完整的**参数、梯度和优化器状态，可以把这些东西切分存储。

| 阶段 | 切分内容 | 每卡显存（理论）| 通信开销 |
|------|---------|--------------|---------|
| **ZeRO-1** | 优化器状态 | $16/N_d + 2$ B/param | = Baseline |
| **ZeRO-2** | 优化器状态 + 梯度 | $16/N_d$ B/param | = Baseline |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | $16/N_d$ B/param | 1.5× Baseline |

三阶段逐步扩大切分范围，ZeRO-3 的极端情况下每张卡只需存 $1/N_d$ 的参数，$N_d = 64$ 时单卡参数显存缩小 64 倍。

通信开销方面，ZeRO-3 前向时需要 AllGather 参数（每层用完即丢），反向时 ReduceScatter 梯度，总通信量约为标准 AllReduce 的 1.5 倍——这是获得极致显存节省的代价。

---

## 3. ZeRO-R：消除残差状态冗余

ZeRO-R（ZeRO Residual State Memory Reduction）针对的是 ZeRO-DP 没有处理的**残差显存（Residual States）**：激活值、临时缓冲区、碎片化显存。

### 3.1 Pa：激活分片

**问题**：标准 ZeRO-DP 下，每张 DP 卡虽然数据不同，但前向传播的激活值与数据大小成正比，并不因为模型被分片而减少。

**解法**：将每层的激活值也按 DP 维度切分（Partitioned Activation，Pa），每张卡只存 $1/N_d$ 的激活。在反向传播需要时，通过 AllGather 重新拼出完整激活。

**开销**：AllGather 激活的通信量与激活大小成正比，对长序列代价较高，因此 Pa 往往与 Checkpoint（不存激活，重计算）结合使用（Pa+cpu 或 Pa+checkpoint）。

```
激活内存对比（单层，batch=1，seq=2048，hidden=4096）：
- 不切分：2048 × 4096 × 2 bytes ≈ 16 MB
- Pa，N_d=8：16 MB / 8 = 2 MB（+ AllGather 开销）
```

### 3.2 Pa+cpu：激活卸载到 CPU

更激进的版本：将激活直接卸载到 CPU 内存，需要时通过 PCIe 读回。对于长序列 / 大 batch 训练中激活占大头的场景，CPU 内存远比 HBM 便宜，但 PCIe 带宽（~16 GB/s）会成为新的瓶颈。

### 3.3 CB：固定大小缓冲区

**问题**：AllReduce / AllGather 等通信操作需要临时工作空间（workspace buffer）。如果允许其随通信量动态增长，在大模型 / 大 DP 规模下，这些 buffer 可能突然申请大块显存，触发 OOM 或显存碎片化。

**解法**：预先设置一个固定大小的通信 buffer（CB），通信数据量超过阈值时分批（chunked）进行，保证 buffer 占用恒定。

```bash
# DeepSpeed 中配置固定 buffer
"communication_data_type": "fp16",
"allgather_bucket_size": 200000000,   # ~200MB
"reduce_bucket_size": 200000000
```

### 3.4 MD：显存碎片整理

**问题**：训练过程中频繁申请和释放激活、梯度、临时张量，会导致显存碎片化——理论上有 N GB 空闲，但申请连续的 N GB 会失败（类似 malloc 的碎片化问题）。

**解法**：在训练开始前，根据运行时的张量生命周期，**预先分配**一块连续的显存池（Memory Defragmentation）。张量的申请和释放在这个池内管理，避免系统 allocator 产生碎片。

---

## 4. ZeRO-Infinity：向 NVMe 扩展

ZeRO-Infinity [2] 把卸载目标从 CPU 扩展到 NVMe SSD：

```
GPU HBM → CPU DRAM → NVMe SSD
       ↑ 越来越慢，越来越便宜
```

通过带宽感知的数据流（bandwidth-centric partitioning），在 NVMe 顺序读写带宽（~3-7 GB/s）的约束下，仍能维持合理的训练吞吐。DeepSpeed ZeRO-Infinity 允许在单机上训练 **trillion 参数量级**的模型（代价是速度极慢）。

---

## 5. 各阶段效果对比

以 7.5B 参数模型、64 个 GPU 为例（来自 ZeRO 论文）：

| 方案 | 单卡显存（GB）| 最大可训练模型 |
|------|-------------|-------------|
| 不使用 ZeRO | 120 | 1.4B（单卡 32GB）|
| ZeRO-1 | 31.4 | 7.5B |
| ZeRO-2 | 16.6 | 14B |
| ZeRO-3 | 1.9 | 1T（理论）|
| ZeRO-3 + Pa | 更低 | 无激活瓶颈 |
| ZeRO-Infinity | 接近 0（HBM）| 无上限（受 NVMe 速度限制）|

---

## 6. 实践建议

- **ZeRO-1/2** 通信开销最小，优先选；模型放得下时不必开 ZeRO-3；
- **ZeRO-3** 适合超大模型，但对通信带宽要求高，在高速 IB 互联集群上效果好；
- **Pa（激活分片）** 适合超长序列训练，通常和 Checkpoint 配合使用；
- **CB（固定 buffer）** 是生产环境中预防 OOM 的稳定性手段，推荐默认开启；
- ZeRO-3 与 PP / TP 混用时，需要额外注意 AllGather 时机和梯度 reduce 的顺序（三者并行时调度复杂）。

---

## 参考

[1] Rajbhandari, S., et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC 2020. [arxiv:1910.02054](https://arxiv.org/abs/1910.02054)

[2] Rajbhandari, S., et al. *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning.* SC 2021. [arxiv:2104.07857](https://arxiv.org/abs/2104.07857)
