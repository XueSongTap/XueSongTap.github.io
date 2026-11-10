---
layout: article
title: DeepGEMM 的 TMA multicast 与 peer 信号：为什么停在 2，以及“先写数据再写信号”
tags: LLM GPU CUDA DeepGEMM
---

> 前置阅读：[Blackwell TMEM 与 tcgen05 MMA 深度解析](https://xuesongtap.github.io/2026/05/19/blackwell-tmem-tcgen05-mma.html)
>
> 源码核对于 2026-11-10，对应 DeepGEMM 提交 `5081f85`（2026-10-20）

前阵子 TileLang 群里有人问：为什么没人给 DeepGEMM 贡献 multicast 代码？接着话题转到了 peer semaphore。有群友说他们的硬件就是把信号写进 peer 的 semaphore 寄存器，对方要么 WFI（等中断）要么轮询。另一位接话说，真正难的是 memory model：必须先写完数据才能写信号量。

翻了一遍 DeepGEMM 源码，第一个问题的前提并不成立：DeepGEMM 早就有 multicast，SM90 和 SM100 都有，只是 cluster 大小固定为 2。所以该问的是另一个问题：**为什么只做到 2？**

第二个问题在 DeepGEMM 里也能找到现成的例子，而且有两层：cluster 内跨 CTA 的 mbarrier，以及 MegaMoE 里跨卡的 NVLink barrier。两层的写法不同，规则是同一条：先让数据可见，再发信号。

## 1. TMA multicast 是什么

Hopper 引入了 thread block cluster：一组被同时调度到同一个 GPC 的 CTA，可以通过 DSMEM 访问彼此的共享内存。TMA 在此基础上提供了 multicast 形式：

```text
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
```

一条 TMA load 带一个 16 位的 CTA mask，从 global 读一次 tile，写到 mask 里每个 CTA 共享内存的**同一偏移**，并在每个目标 CTA **同一偏移**的 mbarrier 上记账（complete_tx）。

GEMM 里，如果 cluster 内两个 CTA 算的输出块 M 坐标相同、N 不同，它们用的是同一块 A，A 只需读一次，B 各读各的。反过来就共享 B。省下来的是 L2 到 SM 的流量：对 L2 带宽和功耗都有好处，HBM 流量本来就靠 L2 复用，multicast 改变不大。

## 2. DeepGEMM 里的 multicast

### 2.1 SM90：一个 CTA 发 TMA，两边都收到

`tma_copy.cuh` 里，SM90 分支只让 cluster 里的 rank 0 发 multicast load，mask 为 `(1 << num_tma_multicast) - 1`：[1]

```cpp
#elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
    if (cute::block_rank_in_cluster() == 0) {
        Ops::LoadMulticast::copy(desc_ptr, barrier_ptr,
                                 (1 << num_tma_multicast) - 1, kCacheHint, smem_ptr, ...);
    }
#endif
```

共享的那个操作数走这条路，另一个操作数 `num_tma_multicast = 1`，两个 CTA 各自加载。哪一边共享由模板参数 `kIsTMAMulticastOnA` 决定。

### 2.2 SM100：multicast 变成 2-CTA UMMA

SM100 分支没有用 multicast load，而是 `SM100_TMA_2SM_LOAD`，注释写的是“2-CTA function will send signals to the leader CTA only”。[1] 这里的 multicast 其实是 2SM UMMA（`cta_group::2`）：

```cpp
using Allocator = cute::conditional_t<kNumMulticast == 1,
                                      cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
```

两个 CTA 合成一条 M=256 的 MMA，每个 CTA 只加载被切开那个操作数的一半，另一个操作数由 2SM MMA 在两个 SM 间共享。[2] 数据复用由 MMA 指令本身表达，不需要再手写 multicast load。

### 2.3 启发式：SM90 会关，SM100 优先开

SM90 的启发式只枚举 `cluster_m × cluster_n ≤ 2`（源码注释：“We only support cluster 2”），并在三种情况下关掉 multicast：[3]

- batched GEMM（不支持）；
- K-grouped GEMM 且 group 数大于 4；
- 只有一个 wave 时，multicast 方案的代价直接记为无穷大。

SM100 正好相反，比较函数里写着“Doing multicast is better”：只要波数相同，cluster 大的方案优先。m-grouped GEMM（MoE 场景）直接固定 swap A/B，N 方向块数和 SM 数都是偶数时就开 `cluster_n = 2`。[4]

## 3. 为什么停在 2

### 3.1 动态调度下，peer 的身份要能算出来

DeepGEMM 是 persistent kernel，每个 CTA 循环调用 `scheduler.get_next_block()` 领下一个块。multicast 要求 cluster 里的 CTA 领到的块在共享维上对齐。DeepGEMM 的做法是让 peer 固定为 `block_idx ^ 1`：同一 cluster 里的两个 CTA 永远领相邻的两个块，再由 swizzle 保证这两块共享 A 或 B。[5]

这个技巧只对 2 成立。异或 1 就是“找配对”，换成 4 就要处理 4 个 CTA 是否都在界内、是否在同一个 group、边界上剩 1 到 3 个块时怎么办。SM90 kernel 里直接写着：[6]

```cpp
DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");
```

### 3.2 边界块：SM90 能动态关，SM100 不能

块数不是偶数时，最后一个块没有 peer。调度器用 `is_peer_cta_alive` 判断 peer 是否在界内；swizzle group 里块数为奇数时，把多出的那一块单独切出去，让它不做 multicast。[5] 源码注释点出了 SM90 和 SM100 的区别：

> for SM90 only, as SM90 can dynamically disable TMA multicast while SM100 uses 2-CTA, which can not be dynamically disabled

SM90 的 multicast 只是一条 load 指令的 mask，运行时可以逐块改成 1。SM100 的 2-CTA 写在 TMEM 分配器和 UMMA 形状里，是编译期决定的，所以启发式只能要求形状本身能被 2 整除（`ceil_div(n, block_n) % cluster_n == 0`）。[4]

m-grouped contiguous 布局还要额外检查两个 M 块是否属于同一个 expert，`is_tma_multicast_valid` 比较的是 `grouped_layout[m_block_idx * BLOCK_M]` 和 `grouped_layout[(m_block_idx ^ 1) * BLOCK_M]`，不同就关掉。[5]

### 3.3 更大的 cluster 换来的不多

- cluster 越大，同时被调度的 CTA 越多，GPC 内放不下时会降低 occupancy，尾部 wave 也更难填满；
- SM100 上，2SM UMMA 已经消化了一部分复用；再叠加 multicast 去做 2×2 的 CGA，需要同时改调度器、屏障计数和 TMEM 分配；
- 群里有人提到，2×2 CGA 的 multicast 能进一步减轻 L2 的功耗压力，这是更大 cluster 的主要收益。但在 DeepGEMM 已经接近算力上限的形状上，这部分收益未必能覆盖调度上的损失。

另外，消费级的 RTX 5090（SM120）没有 2SM UMMA，大 cluster 在那里本来就没有意义。

所以“没人贡献 multicast”的准确说法是：2-way 已经做进去了；4-way 要重写的是调度器，不是一条 TMA 指令。

## 4. cluster 内的 peer 信号：empty barrier 要通知两边

multicast 引出了一个同步问题。rank 0 发出的 TMA 会写进 rank 1 的共享内存，所以 rank 0 在复用某个 stage 之前，必须确认**两个** CTA 的 consumer 都用完了这个 stage。否则 rank 1 还在读，rank 0 的下一次 TMA 就把数据覆盖了。

DeepGEMM 的做法是：每个 CTA 的 empty barrier 按两边 consumer 的数量初始化，consumer 用完一个 stage 后，同时给自己和 peer 的 empty barrier 各 arrive 一次：[6]

```cpp
empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
...
auto empty_barrier_arrive = [&](uint32_t s) {
    if constexpr (kNumTMAMulticast == 1) {
        lane_idx == 0 ? empty_barriers[s]->arrive() : void();
    } else {
        auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
        lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
    }
};
```

每个 math warp 的 lane 0 和 lane 1 分别 arrive 到 rank 0 和 rank 1。peer 不在界内时，两次都 arrive 到自己，计数照样凑够。

这就是群里说的“写到 peer 的 semaphore”：信号是对 peer 共享内存里 mbarrier 的一次 remote arrive。反方向上，full barrier 由 TMA 的 complete_tx 在每个目标 CTA 上记账，数据字节数到齐后 barrier 才翻转，“先写数据再写信号”由硬件保证，软件不需要手写 fence。

## 5. 跨卡的 peer 信号：release 与 acquire

到了跨卡，就没有 mbarrier 和 complete_tx 了。DeepGEMM 的 MegaMoE 里有一个 `nvlink_barrier`，可以看到软件怎么自己维护这个顺序：[7]

```cpp
// 1. 本卡所有 SM 先做一次 grid sync
grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);

// 2. 只有 SM 0 参与跨卡：给每个 rank 的信号做 release.sys 原子加
if (thread_idx < kNumRanks)
    ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);

// 3. 轮询本地信号，直到所有 rank 都到了
wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; }, ...);

// 4. 再做一次 grid sync，把结果告诉本卡其他 SM
grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
```

对应的 PTX 是 `red.release.sys.global.add.s32` 和 `ld.acquire.sys.global.s32`。[8] 这几步分别解决：

1. **数据先于信号。** release 语义保证：本线程在它之前的写，对任何 acquire 到这个值的线程都可见。grid sync 内部用的是 `atom.release.gpu` 和 `ld.acquire.gpu`，先把其他 SM 的写汇总到 SM 0 的因果链上，SM 0 再用 `.sys` 范围的 release 发出去。
2. **scope 要够大。** `.gpu` 只保证同一张卡内的顺序，跨 NVLink 必须用 `.sys`。
3. **等待方用轮询。** 对应群里说的 pull。GPU 上没有 WFI 可用，`wait_until` 自旋并带 60 秒超时打印，方便定位死锁。
4. **信号要能复用。** 计数器的低 2 位保存 phase 和 sign，两块信号区交替使用，每轮在 +1 计到 kNumRanks 与 −1 减回 0 之间切换，不需要每次清零。

为什么不能只用 `volatile`？`volatile` 只阻止编译器删掉或合并访存，不约束硬件的可见顺序。弱内存模型下，flag 的写可能先于数据到达对端。只要漏掉一次 release，对端就可能读到旧数据，而且这种错误很难稳定复现。

群友提到的“peer semaphore 寄存器 + WFI”是另一种设计：信号进专用寄存器，等待方睡眠等中断，省电但唤醒延迟更高。GPU 上做这种同步，目前还是靠全局内存里的 flag 和 acquire/release，NVLink 上没有 mbarrier 那样硬件记账的原语。

## 6. 小结

| 层级 | 数据路径 | 信号 | “先数据后信号”由谁保证 |
| --- | --- | --- | --- |
| cluster 内（SM90） | TMA multicast 写两个 CTA 的 SMEM | full：complete_tx；empty：remote arrive 到 peer | 硬件（TMA 记账 + mbarrier） |
| cluster 内（SM100） | 2-CTA TMA + 2SM UMMA | 信号只发给 leader CTA | 硬件 |
| 跨卡（MegaMoE） | NVLink P2P 写对端显存 | `red.release.sys` / `ld.acquire.sys` | 软件（release/acquire + grid sync） |

DeepGEMM 的 multicast 停在 2，原因在调度器：`block_idx ^ 1` 的配对、奇数块的拆分、按 expert 检查，以及 SM100 上 2-CTA 不能动态关闭，都是围绕 2 写的。peer 信号这一侧，cluster 内硬件替你做了“数据到齐再翻转”，跨卡就要自己写 release 和 acquire。

## 参考

[1] DeepGEMM `deep_gemm/include/deep_gemm/common/tma_copy.cuh`，`copy_nd` 的 SM90 / SM100 分支

[2] DeepGEMM `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh`，L56–L71

[3] DeepGEMM `csrc/jit_kernels/heuristics/sm90.hpp`，L62–L73、L234

[4] DeepGEMM `csrc/jit_kernels/heuristics/sm100.hpp`，L34–L45、L105–L118、L281

[5] DeepGEMM `deep_gemm/include/deep_gemm/scheduler/gemm.cuh`，`get_swizzled_block_idx`、`get_next_block`、`is_tma_multicast_valid`

[6] DeepGEMM `deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh`，L116、L161–L166、L238–L245

[7] DeepGEMM `deep_gemm/include/deep_gemm/comm/barrier.cuh`，`grid_sync`、`nvlink_barrier`

[8] DeepGEMM `deep_gemm/include/deep_gemm/ptx/ld_st.cuh`；NVIDIA PTX ISA，Memory Consistency Model 一节
