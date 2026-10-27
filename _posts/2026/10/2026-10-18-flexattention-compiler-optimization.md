---
layout: article
title: FlexAttention：自定义 Attention 的编译优化
tags: LLM
---

> 前置阅读：[FlashAttention：IO 感知的精确 Attention 计算]({% post_url 2026/03/2026-03-22-flash-attention %}) · [LLM 推理（Inference）性能推导]({% post_url 2025/12/2025-12-13-infer %})

给 Attention 加一个 sliding window，看起来只需要改一行代码：

```python
scores = scores.masked_fill(~window_mask, float("-inf"))
```

但如果执行到这里时 `Q @ K.T` 已经算完了，窗口外的 dot product 其实一个都没省。要是想把这段逻辑塞进 fused Attention，又会碰到新问题：现有 kernel 支不支持这种窗口？能不能再叠加 document mask、相对位置 bias 或 soft-capping？

FlexAttention 想同时解决这两件事：用户用几行 PyTorch 代码写规则，生成的 kernel 既保留 FlashAttention 的分块与融合，又能利用规则里的块稀疏性。本文就沿着一条主线往下看：**Python 里写的规则，怎样变成 kernel 里的运算，以及哪些规则能让整个计算块直接消失。**

<!--more-->

## 1. FlashAttention 省了搬运，窗口还可以省计算

固定 batch 和 head，设 $Q\in\mathbb R^{N_q\times d}$、$K,V\in\mathbb R^{N_{kv}\times d}$，标准 Attention 为：

$$
S=\frac{QK^{\mathsf T}}{\sqrt d},\qquad
P=\operatorname{softmax}(S),\qquad O=PV.
$$

两个矩阵乘的计算量都是 $O(N_qN_{kv}d)$。朴素实现还得在 HBM 里存下 $N_q\times N_{kv}$ 的分数矩阵和概率矩阵，在矩阵乘和 softmax 之间来回读写。

FlashAttention 把这条链分块执行，用 online softmax 维护每行的统计量，完整的 $S$、$P$ 不用再写进 HBM。对 dense self-attention 来说，计算量仍是 $O(N^2d)$，但输入、输出和保存的状态只要 $O(Nd)$ 存储（这里把 head 维度 $d$ 也算进了复杂度）。[1] 原论文还讨论过 block-sparse 扩展，可见“减少 IO”和“利用稀疏性”本来就能结合。

现在加上 causal sliding window。约定 $W$ **包含当前 token**，位置从零开始，那么 query $i$ 能看到的 key 满足：

$$
0\le i-j<W.
$$

比如 $W=4$ 时，query $i=6$ 只能看 $j=3,4,5,6$。整个序列的有效配对数是：

$$
\sum_{i=0}^{N-1}\min(i+1,W)
=NW-\frac{W(W-1)}2,\qquad 1\le W\le N.
$$

也就是前 $W$ 行贡献 $1+2+\cdots+W$，后面每行各贡献 $W$ 项。拿 $N=8192,W=1024$ 算一下，有效配对约 786 万；完整 dense 矩阵有 6711 万个位置，普通 causal Attention 也有约 3356 万个。窗口留出的优化空间很大，但到底能省多少，要看 kernel 是不是真的跳过了窗口外的块。另外，配对数之比也不能直接当成耗时之比。

## 2. 把变化部分留给两个函数

常见的 Attention 变体主体都一样：算分数、做 softmax、加权求和，变化一般发生在 softmax 之前。比如 ALiBi 给分数加位置 bias，soft-capping 对分数做非线性变换，document mask 让不同文档之间互相看不见。

FlexAttention 把这些变化拆成两个 callable：[2][3]

```python
def score_mod(score, b, h, q_idx, kv_idx):
    return modified_score

def mask_mod(b, h, q_idx, kv_idx):
    return allowed  # True 表示参与 Attention
```

其中 `score` 是已经乘过缩放系数的单个分数，`b`、`h`、`q_idx`、`kv_idx` 说明它属于哪个 batch、head 和 query-key 配对。这些参数在语义上是标量 Tensor，不是普通的 Python 循环变量，所以不能拿它们写依赖数据的 `if`，也不能调 `.item()` 来分支。

沿用前面的窗口，再叠一个 soft-cap，规则写出来是这样：

```python
W = 256
CAP = 20.0

def causal_window(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx < W)

def soft_cap(score, b, h, q_idx, kv_idx):
    return CAP * torch.tanh(score / CAP)
```

`< W` 对应“含当前 token 共 $W$ 个”，写成 `<= W` 窗口就多了一个位置。`CAP=20` 只是演示用的参数，实际要和模型定义保持一致。

两条规则合起来，分数变成：

$$
\widetilde S_{bhij}=
\begin{cases}
c\tanh(S_{bhij}/c),&0\le i-j<W,\\
-\infty,&\text{其他位置},
\end{cases}
\qquad O=\operatorname{softmax}(\widetilde S)V.
$$

顺序很关键：**先修改有效分数，再把不允许的位置设为 $-\infty$。** 要是先做 mask、再对所有分数做 soft-cap，$-\infty$ 会被压成有限的 $-c$，本该屏蔽的位置又拿到了非零概率。

单论表达能力，让 `score_mod` 返回 $-\infty$ 也能实现 mask。之所以还要单独提供 `mask_mod`，是为了把“这里不参与计算”的信息交给块稀疏路径。实际调用时，`mask_mod` 先传给 `create_block_mask` 得到 `BlockMask`，再把这个对象传给 `flex_attention`；`flex_attention` 自己并没有直接接收 `mask_mod` 的同名参数。[4]

当然这套抽象也有边界。它适合表达逐元素的规则，能用的输入是当前分数、位置索引和捕获的 Tensor；如果新算子要跨整行排序、做全局 top-k，或者要改 softmax 的归约方式，这两个函数就表达不了了。

## 3. 编译器具体生成了什么

一说“用 Python 生成高性能 Attention”，很容易以为最难的部分全靠自动编译搞定。FlexAttention 实际的分工更具体：框架先准备好优化过的 Attention 模板，再把用户规则编译进去。[2]

按论文描述的 Triton 路径，TorchDynamo 捕获自定义函数的计算图，TorchInductor 把这些子图生成对应的 Triton 运算，嵌进 forward、backward 和 decoding 模板。模板管 tiling、online softmax、数据加载和并行分工，用户函数只管分数和 mask 怎么变。对受支持的 `score_mod`，自动求导还会生成反向计算，接进 Attention 的 backward 路径。[2][3]

以窗口加 soft-cap 为例，forward 内部大致是下面这样。伪代码省略了线程布局和 softmax 的更新公式，但执行顺序是对的：

```text
固定一个 query tile，初始化 online-softmax 状态
    从 BlockMask 读取需要访问的 KV block 编号
    对每个保留的 block：
        计算 Q_tile @ K_tile.T，并缩放
        在片上分数 tile 中执行 score_mod
        若为部分有效块，执行 mask_mod，把无效分数设为 -inf
        更新 online-softmax 状态，累加与 V_tile 的乘积
    归一化并写回输出
```

这样一来，`torch.tanh` 就成了 Attention 循环里的一段运算，不用先把整个分数矩阵写回去、再单独起一个 kernel 处理。即使函数捕获了额外的 Tensor，比如 `bias[h]`，编译器也能把对应的加载放进这条计算链。[3]

这里其实有两种收益。一是融合：把修改放进 kernel 里，省掉中间结果和单独的算子调度。二是跳块：把不需要的块从遍历列表里拿掉，少做矩阵乘、少加载 K/V，这一点要靠 `BlockMask` 提供结构信息。不过非线性函数、额外加载和索引本身都有开销，融合进去也不是免费的。

想继续读实现的话，可以从固定版本源码里的 `flex_attention`、`create_block_mask` 入口进去，再看 Inductor 的 `kernel/flex/` 目录。[5] 这样“规则怎么写”“块怎么表示”“kernel 怎么生成”可以分开追，不用一上来就啃完整的 Triton 模板。

这种分工也让底层模板可以继续演进。2026 年 3 月，PyTorch 官方介绍了基于 FlashAttention-4 的 backend：Inductor 生成 CuTeDSL 形式的 score/mask 修改，再接到 FA4 的计算实现上。[9] 本文讲的是经典 Triton 路径，但同一套编程接口也能接新的计算模板。所以比较性能时，除了规则和输入形状，还要记下实际用的是哪个 backend。

## 4. BlockMask 怎样把 mask 变成少算

`BlockMask` 先把 query-key 平面切成块，再记下每个 query 块要访问哪些 KV 块。下面用 $N=16,W=8$、每块 $4\times4$ 的小例子演示一下。左图每个有色单元是一个允许的配对，右图每个单元对应左图的一整块。

![Causal sliding window 从逐元素 mask 转换为全空、全有效和部分有效的块分类](/img/2026/10/18/flexattention-block-mask.png)

全空块，kernel 直接跳过，既不加载对应的 K/V，也不做这块的矩阵乘。全有效块要算，但可以省掉逐元素的 `mask_mod`。部分有效块照样按块算，算完再把其中的无效位置屏蔽掉。`score_mod` 则要作用在保留下来的全有效块和部分有效块上。[2][4]

以图中最后一个 query 块（位置 $12$ 到 $15$）为例：KV 块 $0$ 到 $3$ 完全用不到，$4$ 到 $7$ 部分有效，$8$ 到 $11$ 全部有效，$12$ 到 $15$ 又是部分有效。运行时直接按这份列表访问 K/V，不用每次都扫一遍完整的 key 维度。

右图只是方便阅读的分类示意。实际的 `BlockMask` 用块数量、块索引这类 Tensor 来存，部分块和全有效块的信息分开保存，另外还有给反向传播用的 query 方向索引。它是面向执行的稀疏元数据，并不是把右图原样存成一张彩色矩阵。[4][5]

$4\times4$ 也只是为了看图方便，示例代码用的是 `BLOCK_SIZE=128`。这个参数描述的是稀疏元数据的粒度，并不直接对应最终 Triton kernel 的 `BLOCK_M`、`BLOCK_N`：计算 tile 还受 backend 配置约束，一个稀疏块可以再切成多个计算 tile。[5]

这也是为什么元素稀疏度高，速度却不一定快。如果有效元素散落在几乎每个块里，大部分块都会变成“部分有效”，矩阵乘照样得做。滑动窗口、文档边界这类规则的块级局部性好，容易形成成片可跳过的区域。但窗口很窄时，边界块里多算的元素可能占比很高，收益得结合块粒度来看。

`BlockMask` 小，也不代表没有成本。它把元数据粒度从元素降到了块，但索引缓冲有多大，仍取决于块网格和具体表示，没法保证所有辅助内存都随 $N$ 线性增长。通用的 `create_block_mask` 还要对规则求值、做块内归约，所以构建过程的峰值内存和耗时，要和最终的 Attention kernel 分开算。[4][5]

## 5. 把窗口与 soft-cap 接到一次调用里

下面是完整的 CUDA 示例，API 按 PyTorch 2.14 文档和 v2.14.1 源码核对过，这个版本里 FlexAttention 仍标为 prototype。示例用 FP16，Q/KV 的 head 数和 head 维度相同，不开 dropout，足够把前面两条规则串起来。[4][5]

```python
import math
import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)

if not torch.cuda.is_available():
    raise RuntimeError("此示例需要支持 FlexAttention 的 CUDA 环境")

torch.manual_seed(0)
B, H, N, D = 2, 8, 1024, 64
W, CAP = 256, 20.0

def causal_window(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx < W)

def soft_cap(score, b, h, q_idx, kv_idx):
    return CAP * torch.tanh(score / CAP)

# mask 与 batch/head 无关，所以只构建一份并广播。
# 当前 API 推荐编译构建函数，不再传弃用的 _compile=True。
build_mask = torch.compile(create_block_mask)
block_mask = build_mask(
    causal_window,
    B=None, H=None,
    Q_LEN=N, KV_LEN=N,
    device="cuda",
    BLOCK_SIZE=128,
)

q, k, v = [
    torch.randn(
        B, H, N, D,
        device="cuda", dtype=torch.float16, requires_grad=True,
    )
    for _ in range(3)
]

compiled_flex = torch.compile(flex_attention)
out = compiled_flex(
    q, k, v,
    score_mod=soft_cap,
    block_mask=block_mask,
)  # 默认 scale = 1 / sqrt(D)，out 的形状为 [B, H, N, D]

# 小规模 dense reference 只用于核对语义，不作为高性能实现。
with torch.no_grad():
    pos = torch.arange(N, device="cuda")
    distance = pos[:, None] - pos[None, :]
    keep = (distance >= 0) & (distance < W)
    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(D)
    scores = CAP * torch.tanh(scores / CAP)
    scores = scores.masked_fill(~keep, float("-inf"))
    ref = torch.softmax(scores, dim=-1) @ v.float()
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=2e-3)

# 检查反向路径能执行；完整训练验证还需对照 reference 梯度。
out.float().square().mean().backward()
for x in (q, k, v):
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

这段代码在写作时核对过 API 和语法，但还没在本机 CUDA 设备上跑过。误差阈值只是这组 FP16 示例的起点；换成别的 dtype、形状或更复杂的规则时，输出和梯度都要验证，并报告实际误差。

有两个参数值得单独说。`B=None,H=None` 是因为窗口规则跟 batch 和 head 完全无关。如果换成 packed sequence，每个样本的文档边界不同，`B` 就得把这些差异表达出来，不能为了省元数据，把不同样本的规则错误地广播成一份。

另一个是 `W`。它既决定可见性，也决定 `BlockMask` 里的有效块列表。改了 `W` 或文档划分，就得重新构建一个匹配的 mask。光让闭包读到新的 Tensor 值没用，旧对象里“哪些块为空、哪些块全有效”的记录不会跟着更新。

窗口和长度固定时，`BlockMask` 可以缓存复用；document mask 每个 batch 都会变，可以在 batch 入口构建一次，让规则相同的多个 Attention 层共用。注意 **重建 mask 不等于重新编译 kernel**：前者更新的是稀疏元数据，后者是重新生成代码。元数据内容变了不一定触发编译，但形状、dtype、函数图或被特化的 Python 参数变了，还是可能触发重新编译。[3][5]

## 6. Decode 先解决并行度，再核对位置

训练和 prefill 时 query 很多，可以按 query 块分配工作。单 token decode 则是 $N_q=1$、$N_{kv}$ 很长，如果 batch 和 head 提供的并行度又不够，光靠按 query 分块很难把 GPU 占满。

所以 FlexDecoding 给短 query 准备了另一条 backend 路径：沿 KV 长度拆分，让多个 program 并行算局部结果，再合并 softmax 统计量和输出。具体怎么分派取决于形状和版本实现，prefill 和 decode 可以编译成不同的 kernel。[6] 这种先拆分再归并的思路沿用自 Flash-Decoding。[7]

局部输出为什么不能直接平均？固定一个 query，设第 $r$ 个 KV 分片的局部输出为 $o_r$，局部分数的 log-sum-exp 为：

$$
\ell_r=\log\sum_{j\in\mathcal K_r}\exp(\widetilde S_j).
$$

每个 $o_r$ 只在自己的分片内做了归一化，它在全体 key 里该占多大权重，取决于这个分片的分母。所以最终输出是：

$$
\ell=\operatorname{logsumexp}_r(\ell_r),\qquad
o=\sum_r\exp(\ell_r-\ell)\,o_r.
$$

这正是 online softmax 的可合并性，只不过从“一个 program 依次接收下一块”变成了“多个 program 并行算完再归并”。上式默认分片非空，局部归一化输出才有定义；空分片的贡献要直接置零，不能拿零权重去乘一个未定义的局部输出。[7]

把前面的窗口函数搬到 decode，还有一个更直接的正确性问题：`q_idx` 是当前 query Tensor 内部的位置。只传一个 query 时它就是 $0$，而这个 token 在完整序列里的位置可能是 $8191$。原样用 `q_idx >= kv_idx` 的话，只会留下 cache 的第一个位置。[6]

假设 KV Tensor 从全局位置 $0$ 开始、当前 query 的起点是 $t$，就要先换算位置：

$$
i_{\mathrm{abs}}=q\_idx+t,\qquad
0\le i_{\mathrm{abs}}-kv\_idx<W.
$$

依赖相对位置的 `score_mod` 也要带上这个 offset。如果 KV 用的是环形缓冲或分页存储，还得先把物理位置映射回逻辑位置，再判断规则。

官方推理教程给过一个做法：预先算好一个大的 `BlockMask`，decode 时按 query 块取切片。[6] 用的时候要同时处理好块行、实际的 Q/KV 长度和带 offset 的 `mask_mod`，块编号可不是 token 位置。另外，v2.14.1 的切片实现要求给切片重新设置 `mask_mod`，所以旧教程里那段短代码不能直接当完整的 decode 实现用。[5]

PagedAttention 也是在这一层接上 FlexAttention 的：页表把逻辑 KV 块转成物理块索引，score/mask 规则再用对应的逻辑位置。[2][6] 这里还牵涉 cache 的存储和地址映射，光写一个逐元素的 bias 函数管不了分页。

## 7. 怎样判断这次改造是否值得

如果模型用的是标准 dense 或 causal Attention，直接从 PyTorch SDPA 开始就好。SDPA 是统一接口，实际走哪个 backend 由设备、dtype、形状等条件决定；对标准模式来说，现成的专用 kernel 是很强的基线。[8] FlexAttention 真正有用武之地的，是模型需要组合自定义规则，或者规则能带来可利用的块稀疏性。

顺着前面的执行过程，成本可以拆成三类：

1. **首次编译。** 图捕获、代码生成，开了 autotune 还有候选搜索。它影响的是冷启动，别混进每次调用的稳态延迟里。
2. **mask 构建。** `create_block_mask` 的执行、内存分配和可能的重建。固定模式可以摊销；每个 batch 都变的模式，要按实际调用频率算进去。
3. **Attention 执行。** 保留块上的矩阵乘、`score_mod`、边界 mask、间接索引和输出归并。训练场景还要单独测 backward。

测的时候，先在目标 GPU 上用较短序列对齐输出和梯度，再用相同形状预热编译好的函数，然后用 CUDA events 或带同步的计时方式测稳态调用。PyTorch/Triton 版本、GPU、dtype、$B/H/N_q/N_{kv}/d$、窗口宽度和 mask 块大小都要记下来，后面速度有变化才解释得了。`mode="max-autotune"` 能让 Inductor 搜更多配置，但多出来的编译成本要单独记。[6]

对比时还得保证 Attention 语义一致。滑动窗口 kernel 比完整 causal kernel 快，有一部分只是因为模型本来就少看了一些 key；只有拿同样的窗口、同样的 bias 或 soft-cap 实现放在一起比，比出来的才是实现本身的效率。

总结一下，FlexAttention 把 Attention 开发的分工划得很清楚：框架维护高性能计算模板，模型作者定义变化的规则，`BlockMask` 把可见性变成可执行的块列表。实际调优时，顺着这三处逐一排查，通常比只盯着一个总稀疏率管用：规则有没有被融合，无效块有没有被跳过，剩下的构建和索引成本划不划算。

## 参考

[1] Dao, T., et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)。

[2] Dong, J., et al. *Flex Attention: A Programming Model for Generating Optimized Attention Kernels.* 2024，MLSys 2025. [arXiv:2412.05496](https://arxiv.org/abs/2412.05496)，模板编译、BlockMask 与分页映射见 §4–5。

[3] PyTorch. *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention.* 2024. [官方介绍](https://pytorch.org/blog/flexattention/)，含 score/mask 示例与重编译、mask 缓存说明。

[4] PyTorch. *torch.nn.attention.flex_attention.* [2.14 API 文档](https://docs.pytorch.org/docs/2.14/nn.attention.flex_attention.html)，含 `flex_attention`、`create_block_mask`、`BlockMask` 定义。

[5] PyTorch v2.14.1 源码：[Python API 与 BlockMask](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/nn/attention/flex_attention.py)、[Inductor FlexAttention](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/_inductor/kernel/flex/flex_attention.py)、[Inductor FlexDecoding](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/_inductor/kernel/flex/flex_decoding.py)。

[6] PyTorch. *FlexAttention Part II: FlexAttention for Inference.* 2025. [官方博客](https://pytorch.org/blog/flexattention-for-inference/)，涵盖短 query、位置偏移、BlockMask 复用、PagedAttention 与 autotune。

[7] Dao, T., et al. *Flash-Decoding for Long-Context Inference.* 2023. [官方博客](https://pytorch.org/blog/flash-decoding/)，KV 分片及 log-sum-exp 归并见正文算法说明。

[8] PyTorch. *scaled_dot_product_attention.* [2.14 API 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。

[9] PyTorch. *FlexAttention + FlashAttention-4: Fast and Flexible.* 2026. [官方博客](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)，说明 FA4 模板与 Inductor 生成 CuTeDSL 规则的衔接。
