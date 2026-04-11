---
layout: article
title: 序列并行（SP）与上下文并行（CP）：超长上下文训练的关键
tags: LLM
---

> 前置阅读：[张量并行与通信](https://xuesongtap.github.io/2025/12/30/tensor-parallel-comm.html) | [分布式训练并行基础](https://xuesongtap.github.io/2025/12/06/paralism-basic.html)

---

## 1. 为什么需要序列并行

训练长上下文（如 128K、1M token）模型时，内存瓶颈主要来自：

- **激活值**：self-attention 的 $Q$、$K$、$V$ 和中间结果，内存随序列长度 $L$ 线性增长；
- **KV Cache**：每层的 $K$、$V$ 需要存储，$O(L \cdot d)$ 的显存；
- **TP 的 Embedding / LayerNorm 副本**：张量并行（TP）要求每张卡持有相同的完整激活，$L$ 大时这个"完整副本"变得难以承受。

**序列并行（Sequence Parallelism，SP）** 和 **上下文并行（Context Parallelism，CP）** 都是把序列维度 $L$ 切到多张卡上，让每张卡只处理一段序列片段，从而降低单卡显存。两者切分的思路相同，但**通信方式**不同，适用场景各有侧重。

---

## 2. DeepSpeed-Ulysses（SP）：All-to-All

### 2.1 核心设计

DeepSpeed-Ulysses [1] 沿序列维度切分输入：$P$ 张卡，每张卡持有 $(L/P, d)$ 的激活片段。

在进入 Attention 计算之前，需要每张卡都能看到**完整序列的一段 head 维度**，因此做一次 **All-to-All** 通信：

```
[前 All-to-All]
每张卡输入：(L/P, d)  →  All-to-All  →  每张卡输出：(L, d/P)
```

- 通信后：每张卡持有**完整序列长度 $L$**，但只负责 **$d/P$ 个 head**；
- Attention 计算：每张卡对自己负责的那部分 head 独立计算；
- 计算后再做一次反向 All-to-All 还原序列分片：

```
[后 All-to-All]
每张卡输入：(L, d/P)  →  All-to-All  →  每张卡输出：(L/P, d)
```

### 2.2 通信量分析

每次 All-to-All 的数据量为 $L \times d$（整个激活张量）分摊到 $P$ 张卡：每张卡发送/接收 $L \times d / P$。

关键结论：**每张卡的通信量与 $P$ 无关**（总量固定为 $L \times d$，分摊后每卡 $L \times d / P$），通信复杂度 $O(Ld/P)$，即**随 GPU 数线性扩展**，不存在通信膨胀问题。

### 2.3 约束

Attention 计算时每张卡负责 $d/P$ 个 head，因此要求：

$$
\text{head 数} \bmod P = 0
$$

即 SP 度 $P$ 必须能整除 attention head 数。对于 GQA（Group Query Attention），约束变为 $P$ 能整除 KV head 数。

### 2.4 与 TP 的关系

Megatron-LM 的 **Sequence Parallelism** [2] 是 TP 的配套方案：在 TP 中，LayerNorm 和 Dropout 等算子作用于完整激活，每张 TP 卡持有相同副本（浪费显存）。开启 SP 后，这些算子所需的"完整激活"被切成 $L/P$ 段分布在各 TP 卡，通过 `ReduceScatter` / `AllGather` 与 TP 的 `AllReduce` 等价替换，消除激活冗余。

---

## 3. Ring Attention（CP）：Ring 传递 KV

### 3.1 核心设计

Ring Attention [3] 同样沿序列维度切分：每张卡持有序列片段 $Q_i$、$K_i$、$V_i$（$i$ 为卡编号）。

计算 Attention 时，每张卡需要用自己的 $Q_i$ 对**所有卡的 $K, V$** 做注意力。

Ring CP 的做法：让 KV 在环形拓扑中**逐步传递**，每一步：
1. 每张卡持有当前接收到的 $(K_j, V_j)$，用本地 $Q_i$ 计算部分 Attention；
2. 将当前 $K_j, V_j$ 传给下一张卡，同时接收来自上一张卡的 $K_{j-1}, V_{j-1}$；
3. 重复 $P$ 步，每张卡累积到完整 Attention 结果。

```
环形传递示意（P=4 卡，1 步）:

   Card 0 --KV₀--> Card 1 --KV₁--> Card 2 --KV₂--> Card 3
      ↑                                               |
      └─────────────────────────────KV₃──────────────┘
```

在每一步传递的同时，卡上的 Attention 计算与通信**重叠（overlap）**进行，通信几乎被隐藏。

### 3.2 通信量分析

每步传递的数据量：$K_j$ 和 $V_j$ 各为 $(L/P, d_k)$，共 $2Ld_k/P$；
一共 $P$ 步，总通信量 $= P \times 2Ld_k/P = 2Ld_k$（与 $P$ 无关）。

**每张卡的通信量为 $O(Ld_k/P)$ 每步，共 $P$ 步**，总接收量为 $O(Ld_k)$。

关键差异：相比 All-to-All（$O(P^2)$ 点对点通信），Ring 只需环形相邻通信，**无需高速全互联（如 NVLink）**，适合节点间的低带宽网络（InfiniBand / 以太网）。

### 3.3 Flash Decoding 与因果 Mask

对于 causal（下三角）Attention Mask，Ring Attention 每步还需要处理哪些 token 应该被遮掩的问题（上三角被 mask 的部分无需计算）。FlashAttention-3 等实现已经原生支持 Ring 模式下的 causal mask 处理。

---

## 4. SP vs CP 对比

| 维度 | SP（DeepSpeed-Ulysses）| CP（Ring Attention）|
|------|---------------------|---------------------|
| 通信原语 | All-to-All | P2P 环形传递 |
| 每卡通信量 | $O(Ld/P)$（1 次）| $O(Ld_k/P)$（$P$ 次，可 overlap）|
| 互联要求 | 需要高带宽全互联（NVLink）| 只需相邻卡互联，低带宽友好 |
| Head 整除约束 | $P$ 整除 head 数 | 无约束 |
| 实现复杂度 | 低（2 次 All-to-All）| 较高（需管理 $P$ 步循环 + overlap）|
| 适用场景 | 节点内（8 卡，NVLink）| 节点间（多机，IB 网络）|
| 通信与计算重叠 | 部分可 overlap | 天然 overlap（ring 步骤）|

---

## 5. 混合使用 SP + CP

两者并不冲突，可以**同时开启**，序列维度被分成 $P_{SP} \times P_{CP}$ 份：

```
GlobalSeqLen = 256K
SP = 4  →  每 SP 组处理 64K
CP = 2  →  每 CP 组内再分 2 份，每卡最终处理 32K
```

Megatron-LM 的参数：`--context-parallel-size 2`（CP），TP 自带 SP；
DeepSpeed 的参数：`--ds-seq-parallel-size 4`（SP）。

**推荐搭配**：
- **节点内**（NVLink）：优先用 SP（All-to-All 快）；
- **节点间**（InfiniBand）：用 CP（环形，不需要全互联）；
- 极长序列（如 1M token）：同时开 SP + CP，进一步切细。

---

## 6. 对并行度 DP 的影响

引入 SP/CP 后，可用于数据并行的 GPU 数会减少：

$$
\text{DP} = \frac{W}{P_{FSDP} \times P_{TP} \times P_{SP} \times P_{CP} \times P_{PP}}
$$

以 8 卡为例：

| FSDP | SP | CP | TP | PP | 有效 DP |
|------|----|----|----|----|--------|
| 1    | 1  | 1  | 1  | 1  | 8      |
| 1    | 2  | 1  | 1  | 1  | 4      |
| 1    | 2  | 2  | 1  | 1  | 2      |
| 2    | 2  | 2  | 1  | 1  | 1      |

DP 减小时，可以用**梯度累积**弥补全局 batch 大小：

$$
\text{GlobalBatch} = \text{MicroBatch} \times \text{DP} \times \text{GradAccStep}
$$

---

## 参考

[1] Jacobs, S., et al. *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models.* 2023. [arxiv:2309.14509](https://arxiv.org/abs/2309.14509)

[2] Korthikanti, V., et al. *Reducing Activation Recomputation in Large Transformer Models.* MLSys 2023. [arxiv:2205.05198](https://arxiv.org/abs/2205.05198)（Megatron SP）

[3] Liu, H., et al. *Ring Attention with Blockwise Transformers for Near-Infinite Context.* ICLR 2024. [arxiv:2310.01889](https://arxiv.org/abs/2310.01889)

[4] 参考知乎整理：[序列并行综述](https://zhuanlan.zhihu.com/p/698447429)
