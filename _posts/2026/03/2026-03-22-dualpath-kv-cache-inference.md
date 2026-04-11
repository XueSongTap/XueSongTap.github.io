---
layout: article
title: DualPath：用双路径 KV-Cache 加载打破 Agentic 推理的存储瓶颈
tags: LLM
---

> 论文：*DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference*
> arXiv:2602.21548，2026.02.25，作者：Yongtong Wu, Shaoyuan Chen, Yinmin Zhong 等（北大 + 字节 + 清华）

---

## 1. 问题：Agentic 推理的 I/O 瓶颈

近年来，以 Coding Agent 为代表的 Agentic LLM 应用呈现出与批量推理截然不同的访问模式。论文给出了来自生产环境的真实 trace 数据（coding task）：

| 指标 | 数值 |
|------|------|
| 平均交互轮次 | 157 轮 |
| 平均上下文长度 | 32.7k tokens |
| 每轮新增 token | 仅 429 个 |
| **KV-Cache 命中率** | **98.7%** |
| Cache-compute ratio（DeepSeek-V3.2）| ~22 GB/PFLOP |

**核心矛盾**：GPU 算力代际间暴增（Ampere → Blackwell 计算增益约 14.4×），但网卡（NIC）带宽没跟上。Agentic 场景下几乎所有时间都在等 KV-Cache 从存储载入，不在做矩阵计算。

### PD 分离架构下的不对称

当前主流 PD 分离（Prefill-Decode 分离）推理架构中，KV-Cache 只由 Prefill 节点（PE）从存储加载：

- Prefill 节点的 SNIC（慢网卡）被打满；
- Decode 节点（DE）的 SNIC 完全闲置。

**DualPath 的直觉**：既然 DE 的网卡空着，为什么不让它也帮忙加载 KV-Cache？

---

## 2. DualPath 系统设计

### 2.1 双路径 KV-Cache 加载

```
传统单路径（PE read path）：
  存储 --SNIC--> PE DRAM --> PE HBM --> DE DRAM

新增 DE read path：
  存储 --SNIC--> DE DRAM --RDMA(CNIC)--> PE HBM --> DE DRAM
```

两条路径按**层（layer-wise）**流水执行，与计算重叠（overlap）。调度器在运行时动态决定每一层的 KV-Cache 走哪条路径。

### 2.2 CNIC-Centric 流量管理（QoS 隔离）

**问题**：KV-Cache 传输如果与 AllToAll / ReduceScatter 等集合通信争抢带宽，会在 sub-millisecond 级别上产生延迟 spike，影响推理尾延迟。GPUDirect Storage 和 CUDA copy engine 都无法提供 QoS 隔离。

**解法**：强制所有数据（包括本地 H2D/D2H）走 **CNIC 的 GPUDirect RDMA**，利用 InfiniBand Virtual Lane 做 QoS：
- 模型推理通信 → 高优先级 VL（占 ~99% 带宽）；
- KV-Cache 传输 → 低优先级 VL（利用空闲带宽）。

工程细节：RDMA write 提交约 1μs（mmio 写寄存器），vs `cudaMemcpyAsync` 的 5-7μs，支持 doorbell batching 摊销开销。论文直接表示这是**目前唯一实用的隔离方案**。

### 2.3 自适应调度器

两级调度：

| 层级 | 职责 |
|------|------|
| **Inter-engine** | 按 token 数均衡 GPU/NIC 负载 + 磁盘读队列长度，决定请求分配和路径选择（PE 优先短队列节点）|
| **Intra-engine** | 按"compute quota"控制 forward batch 大小，避免各 GPU attention 计算时间不一致导致 pipeline bubble |

### 2.4 Bottleneck-Free 条件分析

论文推导了在以下参数范围内系统无瓶颈（$g$=GPU数/节点，$s$=SNIC数/CNIC数之比，$M$=内存带宽，$B$=CNIC带宽）：

$$
\frac{s}{g-s} \leq \frac{P}{D} \leq \min\!\left\{ \frac{g-2s}{s},\ \frac{g-s}{2s},\ \frac{M/Bs-3}{2} \right\}
$$

典型配置（$g=8$，$s=1$）：$1/7 \leq P/D \leq 7/2$，覆盖绝大多数实际部署场景。

---

## 3. 实验结果

**测试环境**：NVIDIA Hopper 集群，InfiniBand，每节点 8×400Gbps CNIC + 1 SNIC + DeepSeek 3FS 分布式存储（无 DRAM 缓存，能打满 SNIC 带宽）

### 离线推理（RL rollout 场景）

| 模型 | 吞吐提升（vs baseline）| 备注 |
|------|-----------------|------|
| DeepSeek 660B (MoE) | **最高 1.87×** | 接近 Oracle（几乎消除 I/O 开销）|
| DeepSeek 27B | **最高 1.78×** | 受 1P1D 存储带宽限制 |
| P/D 比变化下 | **平均 1.64×，最高 2.46×** | |

关键结论：**DualPath 1P1D ≈ Basic 2P1D**，等效节省一半推理 GPU 预算。

### 在线服务

SLO（TTFT ≤ 4s，TPOT ≤ 50ms）下，在线吞吐提升平均 **1.96×**。

---

## 4. 评价

**亮点：**
- 问题诊断有真实 trace 支撑，不是讲故事；
- 3FS 是关键依赖——正因为能打满 SNIC，DE 侧闲置带宽才有价值；
- CNIC-centric 设计诚实：承认这是目前唯一可行的隔离方案；
- 理论推导给出了 bottleneck-free 的参数范围，有工程指导价值。

**局限：**
- 深度绑定自研基础设施（3FS + 自研推理框架 + InfiniBand），迁移到 vLLM/SGLang + RoCE 有待验证；
- 调度器的 α、β 阈值如何调优未详述；
- 论文承认与 SGLang 的对比基准不完全公平。

---

## 参考

[1] Wu, Y., et al. *DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference.* arXiv:2602.21548, 2026. [arxiv](https://arxiv.org/abs/2602.21548)
