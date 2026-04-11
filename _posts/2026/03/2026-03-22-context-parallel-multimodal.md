---
layout: article
title: 上下文并行在多模态模型中为什么会失效
tags: LLM
---

> 前置阅读：[序列并行（SP）与上下文并行（CP）](https://xuesongtap.github.io/2026/03/22/sequence-context-parallel.html)

---

## 1. 背景：为什么需要 CP

对于纯文本 LLM，开启上下文并行（Context Parallelism，CP）可以将长序列的激活显存分摊到多张卡上：

- CP 组内每张卡只持有序列的 $L/P$ 段；
- KV-Cache 也按序列维切分，显存随 CP 度线性减少；
- Ring Attention 或 Ulysses 负责跨卡的 attention 计算。

Megatron-LM 的 `get_batch_on_this_cp_rank` 函数负责做这个切分，并且有一个**负载均衡**的细节：

### 1.1 Causal Mask 下的负载均衡

对于因果（causal）LLM，序列靠后的 token 需要 attend 到更多的 token，计算量比靠前的 token 重。

如果简单地把序列分成 $P$ 段平均分配，GPU 分到尾部的卡会比分到头部的卡忙得多。

**解决方案**：把序列切成 $2P$ 段，然后"头尾配对"分配：

```python
# CP=2 时，分成 4 块：chunk_0, chunk_1, chunk_2, chunk_3
# GPU0 得到 chunk_0（最轻）+ chunk_3（最重）→ 负载均衡
# GPU1 得到 chunk_1（较轻）+ chunk_2（较重）→ 负载均衡

index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)])
val = val.index_select(seq_dim, index)
```

这样每张卡的计算量大致相等，避免了 CP 慢卡拖后腿。

---

## 2. 多模态序列的特殊性

VLM（Vision-Language Model）的输入序列结构通常是：

```
[image_token_1, ..., image_token_N, text_token_1, text_token_2, ...]
                 ↑                              ↑
            ViT + Projector 产出           语言模型部分
```

**image_token 是"前缀"，是不可分割的整体**：它们共同表示一张图像的信息，任意截取其中一部分都无法构成语义完整的图像特征。

---

## 3. 为什么直接套用 CP 会失败

### 3.1 图像特征被物理割裂

假设 CP=2，序列被切成 4 块，GPU0 拿到 `chunk_0`（序列头部）和 `chunk_3`（序列尾部）：

```
序列：[img_0, img_1, ..., img_N, text_0, text_1, ..., text_M]
       ←── chunk_0 ──→            ←── chunk_3 ──→

GPU0 看到的：一部分图像 token + 尾部文本 token
GPU1 看到的：剩余图像 token + 中部文本 token
```

没有任何一张 GPU 能看到**完整的图像特征**，LLM 无法做完整的图文交叉注意力，模型性能会严重退化。

### 3.2 梯度路径被切断（ViT 训不到）

这是更根本的问题：

LLM 的 loss 根据生成的文本 token 计算，梯度需要通过以下路径反传：

```
text loss → LLM layers → image_embeddings → Projector → ViT weights
```

如果 image_embeddings 和用于计算 loss 的文本 token 分布在**不同 GPU** 上，它们之间的 attention 关系被物理割断，梯度无法从文本侧流回图像侧。

**结果：Projector 和 ViT 完全接收不到来自 LLM 的梯度，权重无法更新。**

---

## 4. 正确的实现方式：复制视觉特征

正确方案不是切分图像特征，而是**把完整的图像特征复制到每张 CP 卡上，只切分文本部分**：

### 4.1 Forward Pass

```
1. ViT + Projector 在某张卡上跑，生成 image_embeddings（完整）
   ↓
2. AllGather / Broadcast image_embeddings 到所有 CP 组内的卡
   ↓
3. 只对 text_tokens 做 CP 切分（get_batch_on_this_cp_rank 只作用于文本）
   ↓
4. 每张卡构造自己的输入：[image_embeddings（完整）, text_chunk_i（局部）]
   ↓
5. 各卡独立做 forward（ring attention 处理文本部分的序列并行）
```

### 4.2 Backward Pass

```
1. 每张卡根据 text_chunk_i 计算 loss 和梯度
   ↓
2. image_embeddings 在每张卡上都参与了计算，每张卡各有一份梯度
   ↓
3. AllReduce image_embeddings 的梯度（所有卡的贡献求和）
   ↓
4. 聚合后的梯度传回 Projector → ViT，完成视觉模块的训练
```

---

## 5. 实现要点

### 5.1 显存权衡

复制图像特征意味着每张 CP 卡都需要存一份完整的 `image_embeddings`，其大小为 $N_{img} \times d$（$N_{img}$ 为图像 token 数，$d$ 为隐层维度）。

对于高分辨率图像（如 4K 图像，$N_{img}$ 可能达到数千），这个开销不可忽视。折中方案：
- 让 ViT 在 CP 组内的某张卡上跑，再广播（省去多卡 ViT 推理）；
- 对 image_embeddings 做 TP 切分（按 head 维度切），在合并 image token 时 AllGather。

### 5.2 AllReduce 时机

image_embeddings 梯度的 AllReduce 必须在 Projector 反向传播之前完成，否则梯度不完整。实现时通常挂 `register_hook` 在 `image_embeddings` 的梯度上触发 AllReduce。

### 5.3 Packed Sequence 的兼容性

当使用 packed sequence（多样本拼成一条长序列）时，sequence 里混有多个样本的 image token 和 text token，切分逻辑更复杂，需要额外的 `cu_seqlens` 记录边界信息，CP 的负载均衡策略也需要同步调整。

---

## 6. 总结

| | 纯文本 LLM | 多模态 VLM（直接套用 CP）| 多模态 VLM（正确实现）|
|--|---------|---------------------|-----------------|
| 图像特征 | — | 被切分，不完整 | AllGather，各卡持有完整副本 |
| ViT 梯度 | — | 被切断，无法更新 | AllReduce 聚合后正确回传 |
| CP 效果 | 正常降低显存 | 模型退化 | 文本侧正常降显存 |
| 额外开销 | — | — | image_embeddings AllGather + AllReduce |

CP 在多模态模型中的关键限制来自**图像前缀的不可分割性**。正确的做法是区分"同质序列（text）"和"异质前缀（image）"，对前者做 CP，对后者做复制 + 梯度聚合。

---

## 参考

[1] Megatron-LM context parallel 实现：[megatron/core/utils.py#L1893](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/utils.py#L1893)

[2] Liu, H., et al. *Ring Attention with Blockwise Transformers for Near-Infinite Context.* ICLR 2024. [arxiv:2310.01889](https://arxiv.org/abs/2310.01889)
