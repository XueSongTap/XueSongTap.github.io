---
layout: articles
title: 通过vLLM的ROCm适配揭示AMD GPU在AI推理领域的支持现状
tags: gpu llm vllm amd rocm
---


## 从vllm的rocm 适配来看AMD gpu的 支持情况

### vllm 的 setup.py 中支持的RCOM structure 有：

```py
# Supported NVIDIA GPU architectures.
NVIDIA_SUPPORTED_ARCHS = {"7.0", "7.5", "8.0", "8.6", "8.9", "9.0"}
ROCM_SUPPORTED_ARCHS = {"gfx90a", "gfx908", "gfx906", "gfx1030", "gfx1100"}
# SUPPORTED_ARCHS = NVIDIA_SUPPORTED_ARCHS.union(ROCM_SUPPORTED_ARCHS)
```
### 支持的架构有：


#### gfx90a： cDNA2	MI200
买不到/未全面发售
#### gfx908： cDNA1	MI100

32G 显存

#### gfx906： GCN 5.0 Radeon VII / Radeon Pro VII

2019年上市，GCN 末代，HBM2显存，16g显存，老矿工，1000多就可以买到，计算能力和寿命存疑

#### gfx1030：RNA 2	RX 6900/6800 . 系列

RX6800: 16G显存，60个计算单元 面向游戏


6800XT：16G显存 72个计算单元 面向4K游戏

6900XT：16G显存  80个计算单元 面向4K游戏


注： gfx1032 架构的 6600 卡通过覆盖 成gfx1030 可以支持大部分ROCM？ 

https://zhuanlan.zhihu.com/p/566112395

待验证

#### gfx1100：RNA 3	Radeon RX 7900

20G 显存
### 参考




https://www.eddiba.com/amd-rdna-3-navi-31-gfx1100%E3%80%81navi-32-gfx1102%E3%80%81navi-33-gfx1101-%E7%A6%BB%E6%95%A3%E6%98%BE%E7%A4%BA%E6%A0%B8%E5%BF%83-next-3-2-gpu%E3%80%81apu-%E8%8E%B7%E5%BE%97-dcn-3-1-4/、