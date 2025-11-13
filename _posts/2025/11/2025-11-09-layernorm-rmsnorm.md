---
layout: article
title: Norm 的两种计算方式, LayerNorm vs RMSNorm
tags: norm
---


## 1 LayerNorm：从提出到落地

### 1.1 提出背景
LayerNorm 是 Transformer（Vaswani et al., 2017）原始论文中使用的标准归一化方案，用于在训练初期稳定每个 token 的激活分布。

### 1.2 数学定义
$$
y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x]} + \epsilon} \times \gamma + \beta
$$

其中：

* 对输入 $x \in \mathbb{R}^{d_{model}}$ 的每个样本（token 向量）独立归一化；
* 减去均值 $\mathbb{E}[x]$、除以标准差 $\sqrt{\mathrm{Var}[x]}$；
* 通过可学习参数 $\gamma$、$\beta$ 做仿射变换。

### 1.3 作用与目标
* 让特征在不同维度上保持零均值、单位方差；
* 避免深层网络在训练初期出现激活爆炸或梯度消失；
* 提供可控的缩放与偏移，方便后续层匹配分布。

### 1.4 应用代表模型
> GPT-1 / GPT-2 / GPT-3、OPT、GPT-J、BLOOM

这些早期大模型均沿用标准 LayerNorm 结构。

---

## 2 RMSNorm：轻量级替代者

### 2.1 提出背景
随着模型和上下文不断增大，LayerNorm 的减均值与偏置操作带来的计算开销与数值噪声被放大。RMSNorm（Zhang & Sennrich, 2019）以「只保留均方根缩放」为核心思路，被迅速用于超大规模训练。

### 2.2 数学定义
$$
y = \frac{x}{\sqrt{\frac{1}{d}\lVert x\rVert_2^2 + \epsilon}} \times \gamma
$$

### 2.3 关键差异

| 特征 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 是否减均值 | 是（$\mathbb{E}[x]$） | 否 |
| 是否加偏置 | 有 β | 无 |
| 归一化依据 | 方差 $\mathrm{Var}[x]$ | 均方根 $\mathrm{RMS}(x)$ |
| 可学习参数 | γ、β | 仅 γ |
| 稳定性 & 实现 | 精度好、算子复杂 | 更快、更易融合 |
| 计算量 | 较高（含均值） | 更低 |

### 2.4 直观理解
* LayerNorm = “标准化后再居中”；
* RMSNorm = “只缩放幅值，不过度调整均值”。

RMSNorm 只依赖向量的平方和 $\lVert x\rVert^2$，因此实现简单、内存访问少，在分布式训练下更稳定。

### 2.5 应用代表模型
> LLaMA 系列、PaLM、Chinchilla、T5

多采用 RMSNorm 或其 Pre-Norm 变体。

---

## 3 现代 LLM 为什么偏爱 RMSNorm

### 3.1 数值稳定性
* 减均值步骤在超深网络中易放大浮点误差；
* RMSNorm 避免这一步，残差分布不被频繁“拉回”零点，梯度传播更平滑。

### 3.2 计算代价
* 相比 LayerNorm 少一次 mean 计算与一次 bias 加法；
* 在每层都执行归一化的 Transformer 中，累计可降低约 1%~2% FLOPs，同时减少 kernel 数量。

### 3.3 与 Pre-Norm 结构耦合
* RMSNorm 常放在子层输入（Pre-Norm）处，仅做幅值缩放；
* 不破坏残差的均值结构，减轻残差漂移问题，利于极深网络的优化。

---

## 4 Infra 视角：通信与实现对比

### 4.1 LayerNorm 的通信路径
LayerNorm 需要同时统计均值与方差：
$$
y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x]} + \epsilon} \times \gamma + \beta
$$

在 tensor parallel 等场景，每个 GPU 仅持有部分 hidden 维度（例：$8192/8=1024$），因此必须通过两次 AllReduce 收集：

* 均值 $\mathbb{E}[x]$；
* 方差 $\mathrm{Var}[x] = \mathbb{E}[x^2] - (\mathbb{E}[x])^2$。

缺点是：通信延迟高、同步点多、与主算子 overlap 困难。

### 4.2 RMSNorm 的通信路径
RMSNorm 只需一次平方和：
$$
\lVert x\rVert_2^2 = \sum_i x_i^2
$$

因此：

* 只需一次 reduce；
* 无均值通路，无偏置读写；
* 更容易与矩阵乘等主算子在同一 CUDA stream 上重叠。

### 4.3 通信复杂度对比

| 操作 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 统计量 | mean + var | sum($x^2$) |
| AllReduce 次数 | 2 | 1 |
| 同步依赖 | 强 | 弱 |
| 通信占比 | 高 | 低 |

结论：RMSNorm 减少一次全局求均值，通信抖动更低，适合作为分布式训练的“轻量归一化骨干”。

---

## 5 性能指标：算子占比与数据流

### 5.1 FLOPs 与 Runtime 占比

| Operator 类别 | % FLOPs | % Runtime |
| --- | --- | --- |
| 矩阵乘（Tensor contraction） | 99.8% | 61.0% |
| 归一化（Stat. normalization） | 0.17% | 25.5% |
| 逐元素操作（Element-wise） | 0.03% | 13.5% |

> 矩阵乘法计算量巨大却能高效执行；  
> LayerNorm / RMSNorm FLOPs 极低却耗时高，主要卡在频繁的内存访问与跨设备通信。

### 5.2 FLOP-to-Memory Ratio
FLOP-to-memory ratio（计算密度）表示“访问 1 单位内存可执行多少计算”。数字越大说明越算密集、GPU 越容易被“喂饱”。

* MHA 的比值 153 → 计算密集，IO 比例低；
* LayerNorm / Dropout / Add 的比值 1/3~3.5 → 典型内存密集算子，主要消耗带宽。

| 模块 | FLOPs | 比值 | 含义 |
| --- | --- | --- | --- |
| MHA（多头注意力） | 43G | 153 | 计算密集 |
| Dropout | 4M | 1/3 | 内存密集 |
| Add (+) | 4M | 1/3 | 内存密集 |
| LayerNorm | 29M | 3.5 | 内存密集 |

**计算密度低 → 带宽压力大 → IO 成为瓶颈。**

![alt text](/img/2025/11/flops_to_memory_ratio.png)

### 5.3 Infra 侧收益总结
尽管 RMSNorm 带来的 FLOPs 节省仅约 0.2%，但在数据移动层面具备更明显的复合收益：

1. **减少一次全局归约**，通信与同步更少；
2. **减少内存访问**，提升带宽利用率；
3. **去掉偏置 β**，显存压力更小；
4. **Kernel 更易与主算子重叠**，提高流水效率。

因此在大模型、长上下文、深层网络中，RMSNorm 以“数值更稳 + 通信更轻 + 实现更简单”成为主流选择。
