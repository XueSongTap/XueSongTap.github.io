---
layout: article
title: 顺着 FlashAttention 看 SM120 的实现
tags: LLM GPU CUDA FlashAttention
---

> 前置阅读：[FlashAttention]({% post_url 2026/03/2026-03-22-flash-attention %}) · [Blackwell TMEM 与 tcgen05 MMA 深度解析]({% post_url 2026/05/2026-05-19-blackwell-tmem-tcgen05-mma %})

这次看 `flash_fwd_sm120.py`，注意到它没有自己实现 mainloop，而是继承了 `FlashAttentionForwardSm80`，然后在构造函数里写了一句 `self.arch = Arch.sm_80`。[1]

目标卡明明是 SM120，这个变量为什么要写成 80？先去父类查它的使用位置。

<!--more-->

## arch 影响了哪个分支

父类中有一处输出路径选择：[6]

```python
self.use_tma_O = self.arch >= Arch.sm_90
```

SM120 子类把 arch 改成 80 后，这里就是 False。再看 `_get_tiled_mma()`，QK 和 PV 都由 `warp.MmaF16BF16Op` 构造，MMA shape 为 `(16, 8, 16)`。这部分是直接继承来的，并不是 arch 赋值后才从某种 SM100 实现切换过来。

到这里可以串起来了：子类复用 SM80 的 warp-level MMA 主循环，arch 负责让父类的相关分支配套。这不是 nvcc 的目标架构参数，DSL 仍然面向实际的 SM120 GPU。[1]

## can_implement 又改了什么

另一个改动是共享内存检查。代码按 Q/K/V tile 算缓冲大小，最后用 SM120 的容量判断配置能不能用。[1]

用最直观的配置代入：tile_m、tile_n、head_dim、head_dim_v 都是 128，FP16/BF16 每元素 2 bytes。Q 不放寄存器、单 stage 时，三个 tile 共 96 KiB；K/V 增加到双 stage 后，共 160 KiB。

SM120 单 block 只有 99 KiB，第二种配置放不下，SM80 的 163 KiB 则能容纳这部分缓冲。[5][9] 所以即使主循环没重写，tile 配置也必须重新筛选。如果减小 tile_n，K/V 缓冲会缩小，但相同长度下循环次数也会增加。

还有 `Q_in_regs` 分支，会将 Q/V 的占用由相加改成取最大值，利用两块缓冲生命周期的差异复用空间。这些条件要连起来看，不能只抄一个容量数字。

## 为什么没有直接用 SM100

因为 SM100 的 `tcgen05`/TMEM 路线在 SM120 上不成立。[2] 这里选 SM80 父类，和两者同属 Blackwell 并不冲突。


## 附录：PRO 6000 为什么显存更大，shared memory 却更小

RTX PRO 6000 Blackwell 有 96 GB 显存，怎么一个 block 能用的 shared memory 反而比 A800 少？

96 GB GDDR7 是 GPU 核心芯片外的 DRAM，用来放权重、KV cache 和激活；shared memory 是每个 SM 里的 SRAM，用来存当前参与计算的 tile。[4][10]

外部显存变大，模型更容易整份放进单卡，但每个 block 搬进来计算的数据块有多大，仍要看 SM 内部的资源。

### 容量小了，会卡在哪里

| Shared memory 上限 | A800 所属 SM80 | RTX PRO 6000 所属 SM120 | B200 所属 SM100 |
| --- | ---: | ---: | ---: |
| 每 SM 总容量 | 164 KiB | 128 KiB | 228 KiB |
| 每 block 最大可用 | 163 KiB | 99 KiB | 227 KiB |

表中是架构上限，按官方 KB 的 1024 bytes 口径写作 KiB；实际可用量还受分配配置影响。[5][9]

这里不能只看其中一行。一个申请 120 KiB 的 block，在 SM120 上先撞到 99 KiB 的单 block 上限，即使整个 SM 有 128 KiB 也没用。

换成每 block 申请 72 KiB，则两边都能放下一个。但两个 block 合计 144 KiB，SM80 的总容量允许，SM120 的 128 KiB 就不够了。这时受限的是并发驻留数量。能够同时执行的 block 变少，可能减少可用于隐藏延迟的活跃 warp。


对 FlashAttention 来说，这些约束最终都会回到 tile 和 stage 的选择

## 参考
[1] Dao-AILab, [FlashAttention `flash_fwd_sm120.py`](https://github.com/Dao-AILab/flash-attention/blob/6e646e0099952b768ff2fe229ea9027c26438e7c/flash_attn/cute/flash_fwd_sm120.py), 2025–2026.

[2] NVIDIA, [CUTLASS `tcgen05/mma.py`：SM120/SM121 的架构限制](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py)，2026，访问日期 2026-09-19。

[3] NVIDIA, [Warp-Level MMA Instructions Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/wmma_programming.html), CUTLASS Documentation.

[4] NVIDIA, [CUDA C++ Programming Guide: asynchronous copy and async groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), CUDA Documentation.

[5] NVIDIA, [Blackwell Tuning Guide: Occupancy](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-tuning-guide/index.html#occupancy), CUDA 12.8 Documentation.

[6] Dao-AILab, [FlashAttention `flash_fwd.py`：MMA 与输出路径](https://github.com/Dao-AILab/flash-attention/blob/6e646e0099952b768ff2fe229ea9027c26438e7c/flash_attn/cute/flash_fwd.py#L588-L659)，2026。


[8] Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022，[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)。Attention 算法背景；本文 SM120 路径以实现源码为依据。

[9] NVIDIA, [Ampere Tuning Guide: Occupancy 与 Shared Memory](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy)，访问日期 2026-09-19。

[10] NVIDIA, [RTX Blackwell GPU Architecture 白皮书 v1.1](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf)，2025。

[11] NVIDIA, [Recommended NVIDIA GPUs for NVIDIA RTX vWS：NVLink 支持对比](https://docs.nvidia.com/vgpu/sizing/virtual-workstation/latest/gpus-vws.html)，访问日期 2026-09-19。

[12] NVIDIA, [Introduction to NVIDIA DGX B200 Systems](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)，访问日期 2026-09-19。

[13] NVIDIA, [CUTLASS Changelog：SM120 TMA 与低精度 GEMM 支持](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)，访问日期 2026-09-19。

[14] NVIDIA, [RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)，访问日期 2026-09-19。
