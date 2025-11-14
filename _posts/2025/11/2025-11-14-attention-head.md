---
layout: article
title: 注意力头设计
tags: Attention
---

在不牺牲模型质量的前提下，如何从注意力头这一层入手，压缩推理显存、带宽与算力成本


## 1 绝大多数LLM为何仍守着标准多头

Transformer 诞生至今，最常见的还是经典多头注意力（MHA）：
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
每个 head 拥有独立的 Q、K、V，与其他 head 并行计算。GPT-3、LLaMA、Qwen 这些主流 LLM 之所以鲜少改动，原因主要有三：

- **训练可复现**：生态里所有框架、算子库都为 MHA 做了极致优化，工程风险最小。
- **表达力充足**：每个 head 面向不同子空间，模型容量不受限制。
- **推理成本仍可接受**：在 4K token 以内，MHA 的 $O(n^2)$ 与 KV 缓存还没成为瓶颈；单个 head 的 KV size 为 `seq_len × d_head × precision`，16~32 head 也能塞进 40GB GPU。

真正的痛点来自长上下文和大规模部署——KV 缓存抢显存、注意力矩阵抢带宽：对 70B 模型，把上下文拉到 32K、batch 到 8，就意味着单层 KV 近 40GB，任何一处 head 冗余都会被无限放大，于是才有了下面这些头部改造方案。

## 2 GQA / MQA：共享 K/V 的推理减法

核心动机：把多个 head 的 K/V 合并，从而减少缓存尺寸和内存读写。可把注意力拆成 Query 路径（负责表示多样性）与 KV 路径（负责上下文复用），只在 KV 上做“压缩”。

### 2.1 MQA（Multi-Query Attention）

- 所有 head 共享同一份 K、V，仅保留独立的 Q；等价于把原来的 $W_k,W_v$ 约束为一个矩阵。
- 推理时 KV 缓存只保存一次，显存峰值随 head 数量降为常数，带宽成本也随之下降；flash decoding 场景通常能节省 3～4× KV IO。
- GPT-3.5、PaLM、Claude 2 等大模型用它来在多租户场景里塞更多 batch；很多云厂商的 KV Cache Manager 默认按 MQA 估算配额。

**缺点**：

- 表达力被压缩，每个 head 看到的是同一份上下文，Q 的差异只能做线性重加权；
- 训练时更容易崩梯，需要更低的学习率 warmup、额外正则约束；
- 大语料上稍不注意就会造成困惑度上升。Meta 在 LLaMA-2 的技术报告中就提到，直接把 MHA 改成 MQA 会损失 0.2~0.3 ppl。

因此实践中常见的折中做法是：**训练阶段保持 MHA，推理阶段把头合并成 MQA，再用 LoRA/MoE 在少量数据上补偿**。多家模型开源 repo 中提供的 `merge_mqa.py` 就是这个套路：通过最小化 $||W_k - \hat{W}_k||_2$ 的线性回归把多个头的权重压缩成一份。

### 2.2 GQA（Grouped Query Attention）

- 把所有 head 划分成若干组，同组共享 K/V，组间保持独立；每组通常 4～8 个 head。
- 在 MQA 与 MHA 之间寻找平衡：既能节省显存，又保留一部分多样性；与 MQA 相比，困惑度下降幅度可控。
- LLaMA-2、Gemma、Qwen-2.5 在 8～16K 上下文长度时普遍采用它，`n_kv_heads` 正是它的超参。

**实践建议**：

- 如果服务目标是**单推理延迟**而非极致吞吐，GQA 往往是更稳妥的选择；单次请求只需加载成对的 KV；
- group size 不宜过大，否则 GQA≈MQA；可通过 eval perplexity + 合成长上下文 Q&A 进行网格搜索；
- 大多数厂商提供 MHA→GQA 的微调脚本：先初始化共享 KV，再对合并后的 head 做 1~3B token 的继续预训练，就能找回绝大部分困惑度。

下图给出一组典型数字（70B / 32K / bsz=8 / fp16）：

| 方案 | KV 缓存峰值 | 准确率回退 | 迟滞 |
| --- | --- | --- | --- |
| MHA | 38.4 GB | 0 | baseline |
| GQA (group=4) | 12.8 GB | +0.05 ppl | -5% |
| MQA | 6.4 GB | +0.25 ppl | -8% |

## 3 稀疏 / 滑动窗口：改连接模式

另一条思路是不触碰 head，而是改「谁能注意谁」。只要每个 token 不再对全序列做 softmax，注意力矩阵的尺寸、带宽就能大幅下降，对头的设计也更自由。

- **稀疏注意力（Sparse Attention）**：让每个 token 只连接局部、周期或全局少量节点。Longformer、BigBird 通过设计图样，把复杂度从 $O(n^2)$ 降到接近 $O(n)$。
- **滑动窗口（Sliding Window）**：每个 token 仅关注前后固定窗口，Mistral-7B、GPT-4 解码阶段靠它撑起 32K 以上上下文。FlashAttention-2 提供了窗口化 kernel，能直接在 GPU 上做局部 softmax；推理时只需维护最近 `w` 个 KV，远端 token 可通过「全局 token」或「跨层残差」间接影响结果。

实践中需要留意三个问题：

1. **模式设计**：局部/周期/全局 token 的比例直接决定困惑度；通常做法是在每层保留 1~2 个完全全局的 head 来兜底；
2. **缓存管理**：窗口移动时要滚动 KV 缓存，可以用 `ring buffer` 或 `[t mod w]` 方式复用内存；
3. **与采样策略的耦合**：若在推理中途改动窗口大小，要确保 beam search、speculative decoding 的缓存也同步裁剪。

共性仍然是：不再让所有 token 参与全连接。带来的副作用是全局信息流减弱，需要通过周期性全局 token、跨层跳连或混合注意力来补救。

## 4 SSM 路线：用状态空间替代注意力

Jamba、Falcon 3 等模型尝试把 Mamba 一类的状态空间模型（SSM）嵌入 Transformer：

- SSM 通过线性状态更新实现长距离依赖，理论复杂度线性。
- 混合架构通常把局部模式交给 SSM、把全局模式交给少量注意力层；典型做法是「Attention-SSM-FFN」堆叠。
- 工程门槛高：需要自研 kernel、权重初始化，也尚未形成统一范式；尤其是 ONNX / TensorRT 导出阶段，状态缓存和注意力缓存需要分别管理。

SSM 的优势在于：恒定大小的状态向量、可 streaming 的推理路径，非常适合边缘设备或 RAG agent 的长对话。但目前业界仍面临：

- **训练难度**：需要专门的正则稳定状态矩阵，否则会出现梯度爆炸；
- **生态缺乏**：模型加速、量化、KV 管理工具多数仍服务于注意力；
- **评测缺口**：缺少统一的长序列 benchmark，很难直接证明收益。

当上下文上万、推理硬件有限时，SSM 是值得关注但仍在试验阶段的方向，适合愿意承担科研风险的团队。


## 5 MLA 

TODO

参考：https://www.spaces.ac.cn/archives/10091