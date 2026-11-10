---
layout: article
title: kernel 的利用率怎么算：标准 MFU、TCU 与 TCA
tags: LLM GPU
---

> 前置阅读：[MFU估算](https://xuesongtap.github.io/2025/09/28/mfu.html)

同一个 kernel，有人报 66.7%，有人报 83.3%，还有厂商报出超过 100% 的 MFU。差别往往不在 kernel，而在分母怎么取。本文只对比三种算法：标准 MFU、TCU、TCA。

## 1. 标准 MFU

PaLM 给出的定义是 [1]：

$$
\text{MFU}=\frac{\text{模型 FLOPs}/\text{耗时}}{P_{\text{peak}}}
$$

- **分子**：模型本身需要的 FLOPs，不计重算。计入重算的叫 HFU（Hardware FLOPs Utilization）。
- **分母**：规格书上对应精度的 dense 峰值，例如 B200 BF16 为 2.25 PFLOPS [2]。

问题出在分母上。

规格书峰值是按标称频率算的：

$$
P_{\text{peak}}=N_{\text{SM}}\times\text{FLOPs/clk/SM}\times f_{\text{rated}}
$$

以 A100 为例：$108\times2048\times1.41\,\text{GHz}\approx312$ TFLOPS [3]。但实际运行时，SM 会因为功耗墙、温度墙降频，越接近满载的 kernel 越容易降频。标准 MFU 不区分这一点：kernel 调度不好和 GPU 降频，都会表现为 MFU 低。

## 2. TCU：按标称峰值算

TCU 的分母就是规格书上的标称峰值，不考虑降频：

$$
\text{TCU}=\frac{\text{实测 FLOP/s}}{P_{\text{peak}}}
$$

口径规范时，它和标准 MFU 基本一致，回答的是"离规格书上的数还有多远"。降频造成的损失也算在里面，所以更接近部署时真正拿到的吞吐。

## 3. TCA：按实际频率算

TCA 把分母按实际频率折算：

$$
\text{TCA}=\frac{\text{实测 FLOP/s}}{P_{\text{peak}}\times f_{\text{actual}}/f_{\text{rated}}}
$$

它衡量的是 tensor core 在实际跑过的每个周期里，平均干了满载的几成（Tensor Core Active）。这个比例和频率无关，适合横向比较两个 kernel 的调度好坏。


$f_{\text{actual}}$ 要取 kernel 运行期间的平均 SM 频率，可以用 `nvidia-smi --query-gpu=clocks.sm --format=csv -lms 100` 采样。

## 4. 例子

kernel 实测 1500 TFLOP/s，标称峰值 2250 TFLOP/s，运行时降频 20%：

$$
\text{TCU}=\frac{1500}{2250}=66.7\%,\qquad
\text{TCA}=\frac{1500}{2250\times0.8}=83.3\%
$$

| 算法 | 分母 | 结果 | 适合回答 |
|---|---|---|---|
| 标准 MFU / TCU | 标称峰值 | 66.7% | 部署时实际拿到多少吞吐 |
| TCA | 标称峰值 × 实际频率比 | 83.3% | kernel 的调度做得好不好 |

两者之比就是频率比。差得多，说明降频吃掉了不少；两者都低，说明 kernel 的调度本身有问题。

## 5. 两种 TCA：算出来的，和计数器读出来的

上面的 TCA 是用 FLOPs 算出来的。社区里说的 TCA 更多指硬件计数器的读数，例如 DCGM 的 `PIPE_TENSOR_ACTIVE`，即采样周期内 tensor core 处于激活状态的周期占比 [4]。

NVIDIA 的 OFU 论文把两者的关系写得很清楚 [5]。它把计数器读数记为 TPA（Tensor Pipe Activity），并假设：

$$
\text{TPA}\approx\frac{\text{实际 FLOPs/cycle}}{\text{峰值 FLOPs/cycle}}
$$

再乘上频率比，换算回按时间算的利用率：

$$
\text{OFU}=\text{TPA}\times\frac{f_{\text{SM}}}{f_{\text{SM}}^{\max}}\approx\text{MFU}
$$

对照本文的记号：TPA 对应计数器版的 TCA，OFU 对应 TCU。

问题出在 TPA 那个约等号上。计数器只知道 tensor core 在这个周期"忙不忙"，不知道它这个周期做了多少 FLOPs。只有当每条指令都能用满整个 datapath、也没有补零计算时，计数器版 TCA 才等于算出来的 TCA。

FA2 在 B200 上就是反例 [4]：

| 后端 | MFU | 计数器 TCA |
|---|---|---|
| FA2（Dao flash_attn） | 8.49% | 57% |
| cuDNN Attention | 15.25% | 26% |

FA2 的计数器 TCA 比 cuDNN 高一倍多，速度却只有 cuDNN 的一半左右。原因是 FA2 是 Ampere 时代的代码，用的是 per-warp 的 `HMMA` 指令，cuDNN 用的是 Blackwell 的 tcgen05（SASS 里是 `UTCHMMA`）。NCU 显示，每个活跃周期 FA2 只做了 512 FLOPs，cuDNN 是 2048 FLOPs。tensor core 一直"活跃"，但每条 `HMMA` 只用到约 1/4 的计算能力，所以 MFU 和计数器 TCA 的比值稳定在 0.25 左右。

同一篇文章还提到两个会让两者对不上的原因 [4]：

- **补零计算**：cuBLAS 选的 tile 不能整除矩阵维度时，补零的部分会被计入计数器 TCA，但不计入 MFU。
- **频率读数不准**：H100 上 `nvidia-smi` 和 NVML 读到约 1515 MHz，用 `clock64()` 实测是 1376 MHz，前者偏高约 10%，按它折算出的 TCA 会偏低。

所以计数器版 TCA 适合做集群监控，能低成本地发现 tensor core 在空转。要评价一个 kernel，还是用 FLOPs 自己算 TCU 和 TCA。两者差距明显时，先检查指令代际和 padding。

## 参考

[1] Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, 2022, §4.1. <https://arxiv.org/abs/2204.02311>

[2] NVIDIA DGX B200 规格：8 卡 FP8 72 PFLOPS（sparse，dense 为其一半），折合单卡 FP8 dense 4.5 PFLOPS，BF16 dense 2.25 PFLOPS. <https://www.nvidia.com/en-us/data-center/dgx-b200/>

[3] NVIDIA Technical Blog, *NVIDIA Hopper Architecture In-Depth*, 2022, Table 3（A100 boost clock 1410 MHz，BF16 dense 312 TFLOPS）. <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>

[4] 九老师，*TCA 51%，MFU 不足 8%——GPU 的隐藏性能损耗*，Recsys Frontier，2026-02-18. <https://blog.recsys-frontier.com/article/tca-mfu-gap>

[5] Pedersen et al. (NVIDIA), *Instant GPU Efficiency Visibility at Fleet Scale*, 2026, 公式 (1)（OFU 定义）. <https://arxiv.org/abs/2605.20799>
