---
layout: article
title: MXFP8 与 SM120：硬件支持之后，软件还缺什么
tags: FP8 GPU CUDA TransformerEngine
---

> 前置阅读：[训练中使用 FP8 精度]({% post_url 2025/11/2025-11-03-fp8 %}) · [顺着 FlashAttention 看 SM120 的实现]({% post_url 2026/09/2026-09-19-flash-attention-sm120-sm80 %})
>
> 本文的 issue、PR 和源码状态核对于 2026-09-22；讨论 NVIDIA Transformer Engine 的 PyTorch 路径，没有做 SM120 GPU 性能实测。

SM120 原生支持 MXFP8，但在本文核对的 Transformer Engine（下文简称 TE）v2.19 中，MXFP8 可用性检查仍会返回“不支持”。这两个结果分别回答了不同的问题：前者说的是 Tensor Core 能执行什么指令，后者说的是这套训练软件已经接好了哪些路径。

[TE issue #2668](https://github.com/NVIDIA/TransformerEngine/issues/2668) 恰好讨论了这个落差。提问者在 SM120 上使用 `Float8BlockScaling`，发现它最终也会进入 MXFP8 GEMM，于是追问：既然底层计算已经能用，为什么 `MXFP8BlockScaling` 的支持还不完整？

要看懂这个问题，需要把数据格式、GEMM 的布局，以及训练中的前向和反向连起来。

<!--more-->

## MXFP8 给 FP8 加了什么

MX 是 Microscaling。按照 [OCP MX 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)，MXFP8 把 32 个 FP8 元素放在一个逻辑块里，共享一个 8-bit 的 E8M0 缩放因子。对一个有限值，解码关系为：

$$
\hat{x}_i=s_b q_i,\qquad i\in\text{block }b.
$$

这里的 $q_i$ 仍然是完整的 FP8 数，规范允许 E4M3 和 E5M2 两种编码。每个元素自己的指数还在，块外又多了一层共享的 scale。E8M0 没有符号位和尾数位，其有限值是 $2^k$；它另外保留了 NaN 编码。

忽略 padding 和对齐，一个块占 $32\times8+8=264$ bits，平均每个元素是 8.25 bits。这里的“32 个数据加一个 scale”描述的是编码组成，实际实现可以把数据和 scale 放在不同的数组里。

量化时先为每块选择 $s_b$，再计算 $q_i=Q_{\mathrm{FP8}}(x_i/s_b)$。它解决的是不同区域数值范围不一致的问题。普通 FP8 是元素编码，本身并不规定整张张量只能用一个 scale；MXFP8 则把块大小和 scale 编码一起标准化了。

可以手算一个极端例子。假设一个向量有 64 个数，前 32 个全是 $2^{10}$，后 32 个全是 $2^{-12}$，使用 E4M3 和就近舍入。若整段共用 $s=4$，大数会变成可精确表示的 $q=256$，小数则变成 $2^{-14}$。E4M3 的最小正非正规数是 $2^{-9}$，这个小数会舍入到零。[E4M3 编码定义](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1)

如果分成两个 MXFP8 块，分别取 $s_0=2^2$ 和 $s_1=2^{-20}$，两块的 $q$ 都可以是 256，解码后正好恢复原值。这是人为构造的算例，用来说明局部 scale 的作用。若大数和小数混在同一个块里，它们仍要共享 scale；MXFP8 没有增加单个 FP8 元素的尾数位数。

## SM120 确实有对应的硬件指令

原生支持的证据可以直接落到 [NVIDIA CUTLASS 的 `mma_sm120.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm120.hpp)。其中 E4M3 × E4M3 的块缩放操作使用以下 PTX 指令名称，操作数省略：

```text
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
```

这条指令接收 FP8 数据和 UE8M0 scale，沿 K 方向每 32 个元素共享 scale，执行 $16\times8\times32$ 的矩阵乘加，并使用 FP32 累加。CUTLASS 对缩放粒度也有 `VS == 32` 的检查。

[PTX 的目标架构说明](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)明确将这组 block-scale 扩展列在 `sm_120a` 下，PTX 8.8 起也支持 `sm_120f`。因此自己编译这类 kernel 时，要启用对应的架构特性，不能只看设备查询返回的 `(12, 0)`。

硬件需要处理的数学关系也很直观。把点积分成若干个长度为 32 的块，有：

$$
\hat{y}=\sum_b\left(s_b^A s_b^B
\sum_{i=0}^{31}q_{b,i}^A q_{b,i}^B\right).
$$

每个块的部分和要用自己的 scale，再跨块相加。原生 MXFP8 指令把这类缩放纳入计算路径；软件也可以通过解码、转换或分块计算实现相同的数据语义，只是执行代价不同。

到这里能确认的是 SM120 的计算能力。一个框架还要准备正确的数据布局、组织 scale、选择 kernel，并把结果交给下一步计算。

## 先把 TN 和 non-TN 读明白

TN 是 GEMM 接口中两个转置标志的组合，依次对应输入 $A$、$B$。先只讨论实数，忽略 GEMM 的标量系数和旧结果累加，计算写成：

$$
C=\operatorname{op}(A)\operatorname{op}(B).
$$

N 表示直接使用输入，T 表示使用它的转置；因此 TN 是 $A^T B$，NN 是 $AB$，NT 是 $AB^T$，TT 是 $A^T B^T$。non-TN 泛指 TN 以外的组合，在这个 issue 中主要涉及 NN、NT。[cuBLAS GEMM 定义](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#cublas-t-gemm)

![TN、NN、NT、TT 的输入轴顺序与统一 GEMM 计算形状](/img/2026/09/22/gemm-tn-non-tn.png)

图中四行是四种输入方式的比较，并非连续阶段。每行经过 `op` 后，参与相乘的形状都是 $[M,K]\times[K,N]\to[M,N]$：$M$、$N$ 是输出的行数、列数，$K$ 是乘积求和的归约长度。例如 TN 要求输入 $A$ 为 $[K,M]$、$B$ 为 $[K,N]$；NN 的 $A$ 则是 $[M,K]$。图里的矩形和转置箭头表示逻辑轴及其交换，不表示实际搬运或线程布局。

把输出元素展开，区别更清楚。令 $0\le i<M$、$0\le j<N$，四种组合分别为：

$$
\begin{aligned}
\mathrm{NN}:\quad C_{ij}&=\sum_{k=0}^{K-1}A_{ik}B_{kj},\\
\mathrm{TN}:\quad C_{ij}&=\sum_{k=0}^{K-1}A_{ki}B_{kj},\\
\mathrm{NT}:\quad C_{ij}&=\sum_{k=0}^{K-1}A_{ik}B_{jk},\\
\mathrm{TT}:\quad C_{ij}&=\sum_{k=0}^{K-1}A_{ki}B_{jk}.
\end{aligned}
$$

例如 TN 固定 $i,j$ 后，取原输入 $A$ 的第 $i$ 列与 $B$ 的第 $j$ 列做点积；NN 则取 $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列。两者都沿长度为 $K$ 的轴求和，只是这个轴在原输入中的位置不同。

这里每种组合的 $A$、$B$ 都按各自形状解释。T 指定的是取哪个元素，并不要求调用前先启动一个 transpose kernel：库可以直接按交换后的索引读取已有数据，也可以为计算另行准备转置表示。四种组合也不意味着对同一对数组切换两个字母，就会得到相同的 $C$；每种组合都要提供与其 `op` 对应的输入表示。原数组不满足该表示时，上游仍可能需要搬运数据或调整操作数顺序。

为什么访问方向值得单独适配？若按列主序存储，零起始索引的元素 $(r,c)$ 位于偏移 $r+c\,\mathrm{ld}$，其中 leading dimension（`ld`）是相邻两列起点的元素间距。[cuBLAS 数据布局](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html#data-layout)

由此看 TN：沿 $k$ 归约时，$A_{ki}$、$B_{kj}$ 的偏移分别为 $k+i\,\mathrm{ld}_A$、$k+j\,\mathrm{ld}_B$，两侧步长都是 1。换成 NN，$A_{ik}$ 的步长成为 $\mathrm{ld}_A$；NT 则两侧都跨列访问，TT 只有 $B$ 跨列。实际 kernel 还会分块和重排，这些地址关系解释了布局为何影响实现，不直接给出性能高低。

所以，“前向 TN、反向 NN/NT”要结合下面 TE 的具体调用来看。它包含存储约定和传参顺序，不能把 TN 当成前向计算的固有属性，也不能仅凭训练公式中有没有上标 $T$ 来判定。

## issue #2668 卡在前向和反向之间

2026-02-11，TE 维护者在 [issue 回复](https://github.com/NVIDIA/TransformerEngine/issues/2668#issuecomment-3881426940)中说明：当时 SM120 的 MXFP8 支持主要受 cuBLAS 的 non-TN GEMM 限制，前向可以执行，但反向所需的路径还不行；这不是根本性的硬件限制。

为什么训练会遇到不止一种 GEMM？用线性层说明最容易。设输入 $X\in\mathbb{R}^{M\times K}$、权重 $W\in\mathbb{R}^{N\times K}$，前向为 $Y=XW^T$。令输出梯度为 $G=\partial L/\partial Y$，反向需要：

$$
\frac{\partial L}{\partial X}=GW,\qquad
\frac{\partial L}{\partial W}=G^T X.
$$

这三次乘法对操作数的访问方向不同。[关联 PR #3050](https://github.com/NVIDIA/TransformerEngine/pull/3050)将 TE 调用的布局写得很明确：

| 训练阶段 | 计算内容 | PR 中的 cuBLAS GEMM 布局 |
| --- | --- | --- |
| forward | 输出 $Y$ | TN |
| dgrad | 输入梯度 $\partial L/\partial X$ | NN |
| wgrad | 权重梯度 $\partial L/\partial W$ | NT |

表中的 TN、NN、NT 正是前面所说的接口组合；它们对应的是这条 TE 路径的实际调用。

这也解释了为什么“一次 MXFP8 matmul 跑通”还不足以说明 TE 的训练 recipe 可用。库可能只实现了 TN，而训练还会走 NN、NT；即使三种 single GEMM 都有了，MoE 等场景需要的 grouped GEMM 又是另一个入口。

块量化还让转置变得更复杂。行方向的 $1\times32$ 分组和列方向的 $32\times1$ 分组包含不同的元素，独立量化得到的 scale、FP8 数值也可能不同。把已量化矩阵直接转置，不能代替从高精度数据生成另一方向的量化结果。TE 需要管理这些版本及其 scale 布局。[TE 的 MXFP8 转置说明](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html#handling-transposes)

## 为什么 Float8BlockScaling 也能走 MXFP8 GEMM

这里容易混淆“怎么量化”和“用什么指令计算”。TE 的 `Float8BlockScaling` 使用 128 元素的一维块，或 $128\times128$ 的二维块，scale 存为 FP32；`MXFP8BlockScaling` 默认按 32 元素的一维块量化，scale 存为 E8M0。[TE blockwise 格式说明](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html#data-format)

下面讨论默认的一维量化路径。TE 2.19 另有 `enable_2d_quantization=True` 选项，让权重按 $32\times32$ 块选择共享 scale，再展开为 MXFP8 所需的 scale 表示；它没有改变 FP8 元素和 E8M0 的编码。[recipe 选项](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/common/recipe/__init__.py#L364-L373)、[二维量化参考实现](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/tests/pytorch/test_mxfp8_2d_quantize.py#L65-L91)

在 TE 的 Blackwell 路径中，前一种 recipe 的 scale 被限制为 2 的幂。这使它能够转换成 E8M0，再按 MXFP8 GEMM 需要的粒度展开。例如，一个 128 元素块共用 $s$，可以拆成四个 32 元素块，让这四块继续使用同一个 $s$。原来的 FP8 数据不必因此重新按 32 元素求最大值。

所以，即使两条路径最终使用同类 GEMM，它们的量化误差仍可能不同：`Float8BlockScaling` 保留原来较大块的量化结果；原生 `MXFP8BlockScaling` 可以让这四个小块各自选择 scale。二维大块转换时，也需要将对应 scale 广播到所需位置。

这是复用原生计算能力的一种软件实现。维护者在 issue 中也确认了 scale 广播这条路线，并指出转换开销取决于矩阵尺寸，小矩阵中相对更明显；issue 没有提供可直接推广的加速百分比。[维护者答复](https://github.com/NVIDIA/TransformerEngine/issues/2668#issuecomment-3881426940)

沿着源码往下看，还能找到两者在 SM120 上表现不同的直接原因。`gemm.cpp` 在 `fp8_block_scaling && sm_arch() >= 100` 的分支中，把两个输入转换为供 MXFP8 GEMM 使用的表示，随后固定 `transa = true`、`transb = false`，也就是走 TN。[GEMM 分支源码](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/pytorch/csrc/extensions/gemm.cpp#L312-L323)

它能这样做，是因为 `Float8BlockQuantizer` 已经准备了物理转置的 columnwise buffer；转换函数从已有表示中选择数据，复用数据指针，再转换和重排 scale。相对地，`MXFP8Quantizer` 的 rowwise、columnwise 数据保持相同的逻辑 shape，通过不同量化方向和 GEMM 转置标志参与计算。这两条路径对 cuBLAS 布局的需求并不相同。[两种 quantizer 的存储准备](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/pytorch/csrc/quantizer.cpp#L1090-L1110)、[MXFP8 buffer 分配](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/pytorch/csrc/quantizer.cpp#L1525-L1535)、[转换函数](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/pytorch/csrc/extensions/swizzle.cpp#L303-L355)

因此，`Float8BlockScaling` 可以借助已准备好的转置表示复用 TN，而 `MXFP8BlockScaling` 的训练路径还需要补齐 NN、NT。这比“两个 recipe 都调用了 MXFP8 kernel”多追了一层，也就解释了 issue 中看似矛盾的现象。

## 当前代码和关联 PR 到了哪里

截至本文核对日期，issue #2668 仍然 open。TE `main` 固定到提交 [`969320524aa14c06477d16bf90a8872066811d70`](https://github.com/NVIDIA/TransformerEngine/commit/969320524aa14c06477d16bf90a8872066811d70) 后，`quantization.py` 中的 `_compute_mxfp8_support()` 仍然对计算能力 12.0 及以上返回 False，理由明确指向“全部 GEMM 布局”尚未支持。[支持检查源码](https://github.com/NVIDIA/TransformerEngine/blob/969320524aa14c06477d16bf90a8872066811d70/transformer_engine/pytorch/quantization.py#L162-L184)

正式发布的 [v2.19](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.19)（2026-09-11 发布）也保留同样的判断，并非只有主分支如此。[v2.19 固定提交的门禁](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/pytorch/quantization.py#L162-L168)

这比只读文档中的“Blackwell 支持 MXFP8”更有用。当前 TE 2.19 文档的 [Supported devices](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html#supported-devices) 也只列出 SM 10.0、SM 10.3，没有列 SM 12.0。

PR #3050 正在补这段接入。核对时它仍是 **Draft，尚未合并**。该分支拟开放 SM120/SM121 的 MXFP8 single GEMM，覆盖 TN、NN、NT；同时把 grouped GEMM 的能力检查拆开，后者在这些设备上仍被拒绝。[PR 说明](https://github.com/NVIDIA/TransformerEngine/pull/3050)

PR 描述和提交说明写出的依赖是 **cuBLASLt 13.6.0.2**；固定到 PR 提交 `66df3eba33c644d97bfc17b848c479974f1bd500`，代码实际通过 `tex.get_cublasLt_version() >= 130600` 判断版本。这是该开发分支的条件，不是“当前正式版 TE 安装对应 CUDA 就能用”的保证。[PR 分支源码](https://github.com/NVIDIA/TransformerEngine/blob/66df3eba33c644d97bfc17b848c479974f1bd500/transformer_engine/pytorch/quantization.py)

它还关联了尚未合并的 [PR #2833](https://github.com/NVIDIA/TransformerEngine/pull/2833)，后者处理 SM120 的非 attention 路径适配、资源限制及测试覆盖。#3050 作者建议配合 #2833 和 SM120 CI 一起推进合并；这说明接入还涉及测试基础设施，并非只放开一个架构判断。

这里也需要收紧一句常见的版本描述：[CUDA 12.8 Update 1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cublas-release-12-8-update-1) 的确已经加入 Blackwell GeForce 的 block-scaled FP8/FP4 支持，但这个事实不能扩展成“从该版本起，TE 的全部 MXFP8 训练路径都可用”。它证明底层库开始提供相关 GEMM 能力，具体覆盖哪些布局、哪些接口，仍要继续查。

## Megatron-LM #4175 把同一个问题带到了训练入口

[Megatron-LM issue #4175](https://github.com/NVIDIA/Megatron-LM/issues/4175) 于 2026-04-07 提出。用户在 RTX PRO 6000 上使用 MXFP8 时遇到了下面的错误：

```text
AssertionError: MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet.
```

这段消息与前面 TE 的支持检查一致，关键仍是括号里的 `for all gemm layouts`。它反映的是训练所需路径尚未全部接通。当前 TE 的 `autocast_enter()` 也保留了这条检查：遇到 `MXFP8BlockScaling` 时获取可用性及原因，并据此断言。[TE 检查入口](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/pytorch/quantization.py#L758-L763)

从当前 Megatron 源码也能看到这层依赖。固定到提交 `dae135048e9fcb9f75c8fe02ccdd54375711162a`，`get_fp8_recipe()` 会把 `mxfp8` 选项构造成 TE 的 `MXFP8BlockScaling`；启用 FP8 的层通过 `get_fp8_context()` 将该 recipe 传给 `transformer_engine.pytorch.fp8_autocast()`。Megatron 在这里选择并调用 TE 的量化策略，最终仍要通过 TE 的支持检查。[recipe 构造](https://github.com/NVIDIA/Megatron-LM/blob/dae135048e9fcb9f75c8fe02ccdd54375711162a/megatron/core/fp8_utils.py#L786-L828)、[autocast 调用](https://github.com/NVIDIA/Megatron-LM/blob/dae135048e9fcb9f75c8fe02ccdd54375711162a/megatron/core/fp8_utils.py#L866-L884)

issue 的负责人 `sbhavani` 回复说，团队仍在 Transformer Engine 中添加 SM120／RTX PRO 6000 支持，会在有初步 PR 后补充计划时间。后续参与者也将剩余依赖指向 TE，并指出 Megatron 侧还需要验证和兼容性修复。截至 2026-09-22，该 issue 仍然 open，线程中没有给出明确完成日期。[负责人回复](https://github.com/NVIDIA/Megatron-LM/issues/4175#issuecomment-4224546211)、[后续讨论](https://github.com/NVIDIA/Megatron-LM/issues/4175#issuecomment-4634209257)

因此，只更新 Megatron 或打开一个 MXFP8 参数，不能补齐 TE、cuBLASLt 尚未提供的执行路径；删除支持检查也只会让程序继续走向原本被拦住的后端。TE 接通以后，Megatron 仍需验证自己的训练配置，尤其是前后向、并行组合和 grouped GEMM。#4175 提供了上层训练框架中的实际报错，#2668 解释了底层布局缺口，而 #3050 展示了正在进行的接入工作。

## 在自己的环境里怎么判断

首先记录 GPU、PyTorch、TE 以及实际使用的 cuBLASLt 版本，再询问当前安装的 TE。下面的检查在已安装 TE、CUDA 可用的 NVIDIA GPU 环境中运行，针对本文核对的 API，不需要先启动完整训练：

```python
from importlib.metadata import version

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

print("GPU:", torch.cuda.get_device_name())
print("Compute capability:", torch.cuda.get_device_capability())
print("PyTorch:", torch.__version__)
print("PyTorch build CUDA:", torch.version.cuda)
print("Transformer Engine:", version("transformer-engine"))
print("cuBLASLt version code:", tex.get_cublasLt_version())
print("TE MXFP8 availability:", te.is_mxfp8_available(return_reason=True))
print("TE blockwise availability:", te.is_fp8_block_scaling_available(return_reason=True))
```

`is_mxfp8_available(return_reason=True)` 返回 `(bool, reason)`。它回答的是安装版本的 TE 是否开放该功能；在 SM120 上得到 False，并不否认硬件指令的存在。[公开检查 API 源码](https://github.com/NVIDIA/TransformerEngine/blob/5e52befd5262c06289106338c308079d6adb391f/transformer_engine/pytorch/quantization.py#L334-L349)

`torch.version.cuda` 是 PyTorch 的构建 CUDA 版本，也不能替代实际加载的 cuBLASLt 版本。上面额外调用了 TE 使用的 `tex.get_cublasLt_version()`，便于对照 PR 的版本门槛；`tex` 是内部扩展接口，后续版本可能变化。两个 recipe 的可用性检查也只是框架的功能门禁，具体形状和配置还会继续影响运行。

如果之后支持检查通过，下一步应先验证一个线性层的 forward 和 backward，对照 BF16 检查输出、输入梯度、权重梯度；涉及 MoE 时再单独验证 grouped GEMM。性能分析则把量化、scale 转换、GEMM 和通信放进同一次 profile，不能仅凭 GEMM 名称推断整步训练更快。这里给的是验证顺序，本文没有这些测试的实测结果。

这个 issue 后续最值得跟进的是 #3050 是否合并、配套改动是否进入正式 release，以及 single/grouped GEMM 的支持范围如何更新。对当前问题，明确的回答是：**SM120 有 MXFP8 硬件能力，TE 仍需要软件适配，而且要适配到实际使用的前向、反向和算子入口。**
