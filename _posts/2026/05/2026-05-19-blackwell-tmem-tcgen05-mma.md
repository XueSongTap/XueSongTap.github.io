---
layout: article
title: Blackwell TMEM 与 tcgen05 MMA 深度解析
tags: GPU CUDA Blackwell TensorCore CUTLASS
---

Blackwell 数据中心 GPU（B200、GB200，SM100/SM103）围绕第五代 Tensor Core 引入了一块专用片上存储——TMEM（Tensor Memory）。本文整理 TMEM 的设计动机、tcgen05.mma 指令族的工作方式，以及从 Ampere 到 Blackwell 的 MMA 架构演进。

<!--more-->

## 核心逻辑

每个 SM 配备 256 KB TMEM，与寄存器文件大小相同。TMEM 主要服务 `tcgen05.mma`，即 Blackwell 这一代新的 Tensor Core MMA 指令族（PTX ISA 86，支持 SM_100a/f、SM_103a/f）。

GEMM 的核心计算是：

```text
D = A * B + D
```

MMA 指令把这个矩阵乘加以固定 tile shape 交给 Tensor Core 执行。由于 Tensor Core 吞吐远高于普通 SIMT CUDA core，一个原本用 SIMT 写的 GEMM 迁移到 MMA/Tensor Core 后，计算本身会变得非常快，瓶颈可能从 compute-bound 转移到 memory-bound。

这里的 memory 不只是 global memory，也包括 SMEM 布局、TMA 搬运、TMEM 读写、epilogue 从 TMEM 拷回寄存器等路径。

## 为什么需要 TMEM：accumulator 越来越大

随着 Tensor Core 吞吐逐代提升，为了喂饱它，每个 warpgroup 需要维护的 accumulator tile 也越来越大。Hopper 上 accumulator 放在线程寄存器里，一个 warpgroup（128 线程）的 D tile 可能占掉几百个寄存器。寄存器是线程私有的，128 个线程加起来占用极大，导致：

- **occupancy 下降**：SM 上能同时跑的 warp 减少，hiding latency 的能力变弱
- **register spilling**：寄存器不够时溢出到 local memory（本质是 global memory），直接打爆带宽

因果链：

```text
Tensor Core 吞吐逐代提升
  → 需要更大的 tile 才能喂饱它
    → accumulator 越来越大
      → 占用大量线程寄存器
        → occupancy 下降，成了新瓶颈
          → 把 accumulator 单独拎出来放 TMEM
            → 寄存器压力解除，Tensor Core 直连 TMEM 读写更高效
```

TMEM 的核心动机不只是带宽，更直接的是**把越来越胖的 accumulator 从线程寄存器里解放出来**，让寄存器文件腾出空间，同时给 Tensor Core 一块专属的、硬件直连的存储。

## 架构演进

```text
Ampere:     mma.sync        / Tensor Core warp-level MMA
Hopper:     wgmma.mma_async / warpgroup-level MMA
Blackwell:  tcgen05.mma     / UMMA / TMEM-backed MMA
```

Blackwell 上 `tcgen05.mma` 是 Hopper `wgmma.mma_async` 的替代者。`wgmma.mma_async` 在 Blackwell 上已被标记为 deprecated，不应在 SM100 上继续依赖。

## TMEM 在 tcgen05.mma 里的位置

`tcgen05.mma` 的 PTX 签名大致为：

```text
tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, ...
```

操作数来源：

| 操作数 | 来源 |
|--------|------|
| A | SMEM（通过 SMEM descriptor a_desc 传入） |
| B | SMEM（通过 SMEM descriptor b_desc 传入） |
| D / accumulator | TMEM（[d_tmem] 为 TMEM 地址） |

Blackwell 的 MMA 累加器不再占用普通寄存器，而是放进 TMEM，从而大幅降低寄存器压力。

`tcgen05.mma` 由单个线程发射（issued by a single thread），而非 warp 集体发射，这是与 Hopper `wgmma.mma_async`（warpgroup 集体发射）的重要区别。

计算完成后，结果不能直接在 TMEM 里做普通 CUDA 运算。需要用 `tcgen05.ld` 把 accumulator 从 TMEM load 回寄存器，然后 epilogue 再做 scale、bias、activation、store 等后处理。

## MMA shape 与 tiling

Tensor Core 是专用硬件，电路在设计时就固定了"一次能处理多大的矩阵块"。MMA 指令是 Tensor Core 的原子操作：

```text
一条 MMA 指令 = 把一个固定大小的 A_tile 和 B_tile 喂给 Tensor Core，
                执行 D_tile += A_tile × B_tile，结果写回 D_tile。
```

shape 是硬件电路决定的，不能任意指定。比如 Ampere 的一个典型 atom 是 `m16n8k16`，硬件一次只能做这个尺寸。

实际 GEMM 的 M/N/K 远大于一条 MMA 指令的 tile，所以 CUTLASS/CuTe 做的事是分块（tiling）：

```text
大矩阵
  → CTA-level tile（一个 thread block 负责的块）
    → warp/warpgroup-level tile
      → MMA atom 大小的 tile（对应一条 MMA 指令）
```

整个大 GEMM 变成很多条 MMA 指令的循环叠加。例如 K=256、MMA atom K 维度为 16，则沿 K 循环 16 次，每次把 accumulator D 累加一次，最终 D 里存的就是完整的 `A × B` 结果。

dense FP16 的 `tcgen05.mma` 支持类似下面的 shape：

```text
64 x N x 16
128 x N x 16
```

其中 N 也有倍数和上限约束。

## 兼容性与代际差异

不同代的 MMA 指令是架构绑定的：

| 指令 | 架构 | 状态 |
|------|------|------|
| `mma.sync` | Ampere (SM80) | 老路径，后续架构可能仍支持 |
| `wgmma.mma_async` | Hopper (SM90) | 在 Blackwell 上已 deprecated |
| `tcgen05.mma` | Blackwell (SM100/SM103) | 数据中心 Blackwell 新路径 |

### 消费级 vs 数据中心 Blackwell

消费级 RTX 50 系列（RTX 5090 等）使用 SM120（GB202），与数据中心 Blackwell SM100 是不同的架构目标：

- **SM100**（B200、GB200）：有 TMEM，支持 `tcgen05.mma`，是本文讨论的主体。
- **SM120**（RTX 5090 等消费级）：没有 TMEM，使用不同的 Tensor Core 编程模型，不能直接使用 SM100 的 `tcgen05.mma` 代码路径。

因此，FlashAttention-4 等依赖 TMEM 的内核无法在消费级 RTX 50 系列上运行。

## 调用层次

底层最终是 PTX/inline asm 指令，但通常不直接手写裸 asm，而是走 CUTLASS/CuTe 的封装：

```text
CUTLASS GEMM kernel
  → CuTe TiledMMA / MMA_Atom
    → SM100_MMA_* atom
      → tcgen05.mma PTX / inline asm
```

相关底层指令：

| 指令 | 功能 |
|------|------|
| `tcgen05.alloc` | 分配 TMEM |
| `tcgen05.mma` | Tensor Core MMA（A/B 来自 SMEM，D 在 TMEM） |
| `tcgen05.ld` | TMEM → register |
| `tcgen05.st` | register → TMEM |
| `tcgen05.cp` | SMEM → TMEM |
| `tcgen05.dealloc` | 释放 TMEM |
| `tcgen05.commit` | 提交异步 MMA 操作 |

## tcgen05.mma 特点总结

1. CTA 级，不再是传统 thread/warp 级语义
2. 由单个线程发射（非 warpgroup 集体发射）
3. A 来自 SMEM（通过 SMEM descriptor）
4. B 来自 SMEM（通过 SMEM descriptor）
5. accumulator/D 必须在 TMEM
6. 结果后处理前，需要用 `tcgen05.ld` 从 TMEM 读回寄存器
7. MMA shape 是受限的固定 tile shape，由 CUTLASS/CuTe 做 tile 映射

## 性能直觉

- SIMT GEMM 可能 compute-bound
- 换成 Tensor Core MMA 后，计算吞吐大幅提升，瓶颈经常转移到数据搬运和布局上
- 关键路径包括 GMEM → SMEM（TMA）、SMEM → TMEM、TMEM → register（epilogue）

## 参考资料

- [NVIDIA CUTLASS Blackwell SM100 GEMM 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [libcu++ tcgen05.mma PTX wrapper](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tcgen05_mma.html)
- [Colfax CUTLASS Blackwell TMEM 教程](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [Colfax CUTLASS Blackwell 2-SM UMMA 教程](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/)
- [cuda-oxide Matrix Multiply Accelerators](https://nvlabs.github.io/cuda-oxide/advanced/matrix-multiply-accelerators.html)
- [Mojo SM100 MMA API 文档](https://docs.modular.com/mojo/std/gpu/compute/arch/mma_nvidia_sm100/mma/)
- [SemiAnalysis: Dissecting NVIDIA Blackwell Tensor Cores](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor)
- [Microbenchmarking NVIDIA's Blackwell Architecture (arXiv)](https://arxiv.org/html/2512.02189v1)
