---
layout: article
title: Transformer 超参数的取舍与经验法则
tags: Transformer
---

从 small model 到万亿 token 级的大模型，表现和效率往往被少量关键超参数左右：hidden size、前馈层扩张倍数、注意力头数与 head dim、词表大小、正则化策略以及深宽比例（aspect ratio）。随着参数预算与训练数据呈指数级增长，工程团队反而更依赖这些“旋钮”来稳定收益——一次错误的配置可能让数百万美元的训练算力付诸流水。本文把我在课程里的零散记录整理成一篇可查的博客，并加入实践中反复被问到的背景解释，方便之后设计或复现 Transformer。

阅读顺序上，可以先浏览第 1 节的速览表迅速建立数量级感知，再根据实际需求跳到单独的小节。如果你正打算把某篇论文的配置迁移到自己的项目里，建议直接对照第 7 节的顺序进行 sanity check。

## 1 关键超参数速览

决定 Transformer “性格”的核心参数大致三类：描述形状的（hidden size、层数、head 数），描述激活容量的（FFN 扩张倍数、head dim），以及围绕数据/优化的（vocab、正则化）。把它们放在一个表里可以快速看到主流模型之间的共性。

| 类别         | 常见配置                           | 代表模型                               | 核心影响                     |
|------------|--------------------------------|--------------------------------------|----------------------------|
| hidden size | 4k–12k                         | GPT-3、PaLM、LLaMA                   | 表达能力、显存占用               |
| feedforward | $r_{ffn}=4$（GLU 取 $8/3$）     | GPT-3、PaLM、LLaMA、DeepSeek         | 非线性容量、计算开销             |
| num heads   | $d_{head}=64\sim128$           | GPT-3: 96×128, LLaMA: 32×128         | 上下文分辨率、并行粒度             |
| vocab size  | 32k–256k（取决于语料/语言）        | LLaMA 32k, PaLM 256k, Qwen 150k      | 嵌入层参数量、序列长度             |
| regularization | Dropout≈0；依赖 AdamW + weight decay | GPT-3、PaLM、LLaMA、Claude、DeepSeek | 收敛稳定性、优化噪声               |
| aspect ratio | $n_{layers}/d_{model} \approx 0.006\sim0.01$ | GPT-3, LLaMA, PaLM, Chinchilla | 延迟、并行化、深层表达 vs 宽度效率 |

## 2 前馈层：4× hidden size 不是偶然

Transformer 的每个 block 包含一个扩大后再压缩的前馈层。它承担把多头注意力收集到的信息进行非线性变换、再写回主分支的职责，所以容量不足时模型通常表现为“记得住信息但融会贯通不了”。
$$
\text{FFN}(x) = W_2 \, \sigma(W_1 x)
$$

- 输入与输出维度等于 hidden size，$W_1$ 把维度扩张至 $r_{ffn} \cdot d_{model}$。扩张的意义在于给激活函数提供足够宽的表示空间，再通过 $W_2$ 投回主干。
- 经典 Transformer 和 GPT 系列普遍采用 **$r_{ffn}=4$**；GLU 或 SwiGLU 激活通常用 **$r_{ffn}=8/3$**，保持 FLOPs 相近。换句话说，激活类型不同但耗费的算力基本一致。
- 更大的 $r_{ffn}$ 提升非线性表达能力，但 FFN 计算和显存随之线性增长，尤其在推理阶段会拖慢每个 token 的延迟。实践里只有在大幅增加计算预算、并确认 attention 已经不是瓶颈时才会上调该比例。

## 3 注意力头：保持 head dim × heads = hidden size

注意力模块把 hidden 向量拆成多个子空间，让不同 head 关注不同的上下文模式。为了方便实现和硬件并行，通常约束：
$$
d_{model} = n_{heads} \times d_{head}
$$

- **必须整除** 是工程上的刚需：只有这样才能把 $Q/K/V$ 张量 reshape 成 $(n_{heads}, d_{head})$ 并行计算。
- $d_{head}$ 常落在 64～128；过小会让单个 head 缺乏容量，过大则降低 head 数目，失去多样化关注。这个区间也是 CUDA warp、张量核等硬件友好的倍数。
- 典型设置：GPT-3 (12288 hidden, 96 heads, $d_{head}=128$)、LLaMA-2 7B (4096 hidden, 32 heads, $d_{head}=128$)。可以看到不同模型通过调节 head 数而不是 head dim 来适配更大的 hidden。
- 更宽的 hidden size 不代表更多少头，反而会把 $n_{layers}/d_{model}$ 约束在 100～200 的“aspect ratio” 范围内，以便在计算/通信与表达力之间取得平衡。这也是很多团队迁移配置时最容易忽略的隐形规则。

## 4 词表大小：语料、分词与显存的三角平衡

词表大小看似与网络结构无关，但它直接决定 embedding 与输出层的参数量，还会影响每条输入序列的 token 数。词表越大，embedding/LM head 的参数量越多，显存和吞吐都会被拉高；词表太小则导致序列变长、分词碎片化，训练时同样浪费 FLOPs。

- **分词方式**：BPE（GPT-2/3）倾向英文；SentencePiece（LLaMA、PaLM、Qwen）更适合多语言。
- **经验区间**：
  - 单语英文：32k–50k（GPT-2/3 50k，LLaMA 32k）。
  - 多语言 or 代码：150k–256k（PaLM 256k，Qwen 150k）。
- 设计词表时先看：语料复杂度、SOP（sequence of processing）最长长度、部署显存预算。embedding 层在 175B 级模型中占比可达 5–8%，对推理吞吐也有显著影响，所以千万别把它当“无成本”参数。

## 5 正则化策略：从 Dropout 转向 Weight Decay

过去的 Transformer 习惯在 attention 与 FFN 中使用 0.1 左右的 dropout，如今的超大模型基本弃用，只保留 **AdamW 的 weight decay（0.1–0.01）**。训练规模上来后，人为引入的噪声往往弊大于利：

- **大规模数据** 自带噪声和正则化效果，训练集本身很难过拟合，即便不过拟合也更多是数据/指令分布的问题。
- Dropout 在分布式训练中引入额外随机性，会干扰流水线与参数同步，推理阶段还得额外关掉，增加实现复杂度。
- Weight decay 提供更平滑、可控的收缩，且不会破坏计算图 determinism。结合精心设计的 LR schedule，已经能覆盖大部分正则需求。
- 仍然会保留 LayerNorm / RMSNorm 以及混合 batch、噪声数据等隐式正则方式。

Qwen 等少数模型继续在注意力内用极小的 dropout（<0.1）来改善优化噪声，但这已经是特例而非常规。

## 6 深宽取舍：Aspect Ratio 的共识

“模型该更深还是更宽？”——答案是 **在给定 FLOPs 下保持 $n_{layers}/d_{model}$ 的稳定区间**。这里的“深”指 Transformer block 的堆叠层数，也就是前向路径上串联多少次注意力+FFN；“宽”指单层内部的表示维度（hidden size）以及由它派生的 head dim 和 FFN 扩张维度，决定每层矩阵乘法的横向规模。行业公开模型的 ratio 极为接近，说明大家都在遵循同一条经验曲线。

| 模型           | 层数 | hidden | $n_{layers}/d_{model}$ |
|--------------|-----|--------|------------------------|
| GPT-3 175B   | 96  | 12,288 | 0.0078                 |
| LLaMA-2 7B   | 32  | 4,096  | 0.0078                 |
| PaLM 540B    | 118 | 18,432 | 0.0064                 |
| Chinchilla 70B | 80 | 8,192 | 0.0097                 |

这个 ratio 的好处是：对训练来说可以把梯度路径长度和通信量控制在某个可管理的范围；对推理来说则意味着延迟与吞吐都不会单方面失衡。偏离这一带宽基本意味着浪费算力，或是模型表现远低于参数上限。

### 6.1 为什么不盲目加深？

- 层越多越难 pipeline 并行，通信和调度成本上升。
- 推理延迟与层数线性相关，极深模型响应慢。
- 训练更易遭遇梯度消失/爆炸，对初始化和学习率极端敏感，需要额外技巧（深度残差缩放、层跳连等）才能稳定。

### 6.2 为什么不过度加宽？

- Attention/FFN 的矩阵乘法复杂度随 $d_{model}^2$ 升级，显存爆炸；在 TPU/GPU 上，方阵越大越难充分利用带宽。
- 层数太少虽然宽，但缺乏逐层抽象能力。

Scaling Law（Kaplan 2020, Hoffmann 2022）说明在固定计算预算下，存在一条最优曲线，大致满足 $n_{layers} \propto d_{model}^{0.8}$。现代 LLM 的“aspect ratio” 正是沿着这条曲线演化的，也为训练调度、内存分配提供了简单的 thumb rule。

## 7 设计超参数的顺序建议

1. **先定算力预算与推理目标**：确定 hidden size、层数范围以及允许的延迟。很多人一上来盯着参数量，结果推理时才发现延迟不可接受。
2. **按 hidden size 推出 FFN、head dim、head 数**：保持 $d_{head}\in[64,128]$ 且 $r_{ffn}$ 在 4 或 $8/3$。这一步既能保证矩阵形状适配硬件，也能方便地估算单步 FLOPs。
3. **根据语料设计分词方案与 vocab**：预估 embedding 占用与序列长度，必要时用多语言语料做 ablation，避免上线后才发现 token 化碎片严重。
4. **正则化默认只用 weight decay**，除非训练噪声过低或 batch 极大才考虑轻量 dropout。保持训练 determinism 可以让调参效率更高。
5. **回到 aspect ratio 检查深宽平衡**，超出 0.006–0.01 往往意味着资源使用不均且优化困难。必要时用小规模 pilot run 评估 loss scaling 情况，再决定是否重新分配宽度或深度。

掌握这些经验法则，在复现论文或定制新模型时就能少踩不少坑：超参数不再是拍脑袋的选择，而是一系列围绕计算、表示和数据约束的工程折中。希望这篇更“啰嗦”的笔记能在下一次开会讨论配置时，为你节省几轮 trial-and-error。
