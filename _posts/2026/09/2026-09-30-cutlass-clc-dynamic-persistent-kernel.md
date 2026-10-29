---
layout: article
title: 理解 CUTLASS 的 CLC dynamic persistent kernel
tags: LLM GPU CUDA Blackwell CUTLASS
---

> 前置阅读：[Blackwell TMEM 与 tcgen05 MMA 深度解析]({% post_url 2026/05/2026-05-19-blackwell-tmem-tcgen05-mma %})

读 CUTLASS 的 Blackwell Cluster Launch Control 文档时，有个地方很容易卡住：persistent kernel 一般只启动一批 worker，让它们在内部循环处理多个 tile；可 CLC dynamic persistent kernel 却按全部输出 tile 启动 grid，看上去又回到了普通 GEMM kernel 的老路。

每个 tile 都有自己的 block 启动位置，那凭什么还叫 persistent？

答案是：**grid 里有一个启动位置，不等于最后真会启动一个新的 block 实例。** 已经在跑的 block 可以通过 CLC 取消某个还没开始的启动，拿到它的坐标，自己接着把那份活干了。这样同一个 worker 能连续处理多个 tile，而后续的活由硬件在运行时分配。[CUTLASS 文档][cutlass-doc]

下面按"为什么需要动态分配 → CLC 怎样接手工作 → CUTLASS 怎样把领取和计算重叠起来"的顺序展开。文档和源码都以 CUTLASS 4.3.1 为准，源码链接固定到对应 tag。

<!--more-->

## 从一个 tile 一个 block，到一个 worker 连续干活

先说清两个概念。tile 是要算的任务，比如 GEMM 输出矩阵里的一个分块；worker 是干活的 block 或 cluster。下面先把 cluster shape 设为 `1×1×1`，这样一个 worker 就是一个 block；同时假设每份任务就是一个完整的输出 tile，不涉及 split-K、Stream-K 这类进一步拆分工作的调度方式。

普通 non-persistent GEMM 可以让一个 block 负责一个输出 tile。比如共有 400 个 tile，就启动 400 个 block 的 grid，硬件按资源情况分批运行，每个 block 算完自己的 tile 就退出。

Persistent kernel 则在 block 里多套一层工作循环：算完当前 tile，再去找下一个。这样，能跨任务保留的公共初始化、数据结构和执行资源都可以复用，block 启动和公共准备的开销也摊到了多个 tile 上。当然，每个 tile 自己的输入加载、累加和写回还是得做，最终收益要看具体 kernel 里有多少工作能复用或重叠。[CUDA 编程指南][cuda-clc]

于是问题变成了：下一个 tile 从哪来？这就是调度器要管的事：

| 方式 | grid 怎么启动 | 一个 worker 后续做什么 |
| --- | --- | --- |
| 普通 non-persistent | 按全部 tile 启动 | 做完自己的 tile 就退出 |
| Static persistent | 按预设 worker 数启动 | 沿软件预先确定的任务序列继续计算 |
| CLC dynamic persistent | 按全部 tile 启动 | 通过 CLC 接手尚未启动的任务 |

可见 persistent 说的是 worker 的执行方式，即同一个实例连续处理多个任务。它既不要求 grid 只有固定数量的启动位置，也不是说 block 会永久驻留在 GPU 上。

## 固定任务序列为什么会拖尾

看个小例子。12 个 tile，计划启动 4 个 worker，按固定步长分配：

```text
W0：tile 0 → 4 → 8
W1：tile 1 → 5 → 9
W2：tile 2 → 6 → 10
W3：tile 3 → 7 → 11
```

如果 GPU 这时只能同时容纳 3 个 worker，W3 就暂时起不来。W0、W1、W2 各自算完三个 tile 后退出，而分给 W3 的三个 tile 得等它启动后才能开始。

假设每个 tile 都耗时 $t$，前三个 worker 同时开工，W3 要等某个 worker 退出、腾出资源才能运行；忽略启动和调度开销，前三个 worker 约在 $3t$ 结束，W3 再花约 $3t$，总共约 $6t$。

注意，这里 W3 并没有绑死在某个 SM 上。问题出在任务已经按 4 个 worker 划好了，实际并发资源却只够跑 3 个；就算某个 worker 先闲下来，也没有办法去接别人序列里的活。CUTLASS 文档用"部分 SM 被其他 kernel 占用"的场景说明了这种不均衡。[CUTLASS 文档][cutlass-doc]

CLC 改的是任务归属确定的时机。同样 12 个 tile，这回启动 12 个 block 的 grid。先跑起来的 worker 做完手头的活，就去接后面还没启动的任务。同样只有 3 个并发 worker 的理想情况下，每个 worker 处理 4 个 tile，总耗时约 $4t$。

$6t$ 和 $4t$ 只是为了说明机制的手算，不是实测性能。其实在这个例子里，普通 non-persistent kernel 也能把 12 个 block 动态分给可用资源；CLC 想要的是两头兼顾：既有这种分配上的弹性，又有 persistent worker 连续干活带来的复用收益。

## 取消的是启动，拿到的是工作坐标

CLC 全称 **Cluster Launch Control**，是 Blackwell 新增的调度能力。运行中的 worker 可以发出 `clusterlaunchcontrol.try_cancel`，请求取消同一 grid 中某个还没开始运行的 cluster 的启动。成功的话，响应里会带上被取消 cluster 的首个 CTA 坐标（CTA 就是 CUDA 的 thread block）。[CUDA 编程指南][cuda-clc]、[PTX 指令说明][ptx-clc]

在 cluster 大小为 1 的例子里，假设 block 0 成功接手了 block 7：

```text
原本：block 7 启动 → 初始化 → 计算 tile 7 → 退出

现在：block 7 的启动被取消
      block 0 在原有执行循环里继续计算 tile 7
```

这次取消由硬件原子地完成，同一个坐标不会出现在同一 grid 的两个成功取消响应里。所以每份工作最终只有一个归宿：要么正常调度、启动它自己的 worker，要么被已经在跑的 worker 接走。至于已经跑到一半的 block，是接不走的。[PTX 指令说明][ptx-clc]

先把线程协作和异步细节放一边，执行逻辑大致如下。其中 `clc_fetch_and_wait` 是本文为了讲解编的占位函数，不是能直接调用的 CUTLASS API：

```cpp
// 概念伪代码，cluster shape = 1x1x1
setup_common_data_structures();
dim3 work = blockIdx;

while (true) {
    compute_tile(work);

    auto [success, next_work] = clc_fetch_and_wait();
    if (!success) {
        break;
    }
    work = next_work;
}
```

Persistent 体现在这层循环上，dynamic 则体现在 `next_work` 是运行时通过 CLC 请求拿到的。`blockIdx` 始终是当前 block 自己的启动坐标，变的是程序里保存的 `work`。CUTLASS 文档把启动 grid 中的三维坐标叫作 `ClcID`：第一份工作来自自己的 `blockIdx`，后面的都来自 CLC。[CUTLASS 文档][cutlass-doc]

所以，400 个 block 的 grid 只是列出了全部任务的启动位置，并不要求真有 400 个 block 实例各自初始化、各自执行。worker 接手任务时，也不需要在循环里再发起一次 kernel launch。

## 沿 CUTLASS 源码看：发请求与消费响应

上面的伪代码是"算完再领"，方便说明语义。真正的 CUTLASS kernel 会让领取任务和当前计算重叠，尽量不让计算 warps 干等下一份工作坐标。

读 v4.3.1 源码，可以从 [`sm100_gemm_tma_warpspecialized.hpp`][kernel-source] 里 scheduler warp 的分支入手，顺着这条路径往下走：

```text
scheduler warp
  → advance_to_next_work：申请可写的 pipeline stage，发出 CLC 请求
  → issue_clc_query：执行 clusterlaunchcontrol.try_cancel
  → 硬件异步写入 shared memory，并通过 mbarrier 通知完成
  → fetch_next_work：等待响应，解码并转换成下一份 WorkTileInfo
  → 各工作 warp 用新坐标推进后续计算
```

[`PersistentTileSchedulerSm100`][scheduler-source] 管工作坐标和请求，`PipelineCLCFetchAsync` 管响应缓冲的生产与消费。前者的 `advance_to_next_work` 先拿到一个可写 stage，再由选出的单个线程调用 `issue_clc_query`。请求发出去之后线程可以接着往下走，响应要过一会儿才写进 shared memory。

所以消费端不能默认缓冲已经可读。v4.3.1 的 `fetch_next_work` 先调 `consumer_wait`，再从当前 stage 读响应，用 `work_tile_info_from_clc_response` 解码，最后 `consumer_release`，把这个 stage 让给后续请求复用。解码时先看成功标志，成功了才提取坐标，然后再结合 cluster 内 CTA 的位置、swizzle 和 raster order，得到实际的 `WorkTileInfo`。[v4.3.1 调度器源码][scheduler-source]

这里文档和源码有个小出入：4.3.1 文档用 `get_current_work` 来介绍读取响应这一步，而对应 tag 的这条实现路径实际用的是 `fetch_next_work`。所以对照源码时，与其按函数名搜，不如抓住"发请求"和"等响应、读坐标"这两个动作。

[`PipelineCLCFetchAsync`][pipeline-source] 是生产者—消费者模型。cluster 中第 0 个 CTA 的 scheduler warp 是 producer，由其中选出的一个线程发请求；需要工作坐标的 warps 是 consumer。响应会广播到 cluster 中各 CTA 对应的 shared memory 地址，各 CTA 等到完成后再读。

文档给的几个参数正好能和这个流程对上：

```cpp
// CLC pipeline 参数节选
transaction_bytes = 16;  // 一份 CLC 响应占 16 字节
producer_arv_count = 1;  // 选出一个线程发请求
producer_blockid = 0;    // cluster 的首个 CTA 负责生产
```

`consumer_arv_count` 按实际参与消费的线程数来配。scheduler warp 自己也是消费者：它除了给其他 warps 提前备好任务，还得读成功标志，判断领取循环什么时候结束。[CUTLASS 文档][cutlass-doc]

文档里展示的 warp-specialized GEMM 用的是深度为 3 的 CLC pipeline，让多轮领取和计算交叠。这个深度是该示例配置的取值，在 v4.3.1 kernel 里由 `SchedulerPipelineStageCount` 传入。另外还有一条 CLC throttle pipeline，用来限制 scheduler 提前领取的进度，免得各个 persistent worker 推进快慢不一，又造成新的负载不均衡。[v4.3.1 kernel 源码][kernel-source]

总结一下分工：CLC 决定下一份工作由哪个 worker 接手，pipeline 负责把这份工作坐标及时、正确地送到各个消费者手里。

## Cluster 粒度与失败后的退出

前面的例子都是单 block worker。如果 cluster shape 改成 `2×2×1`，一次成功的请求接手的就是整组 4 个 CTA 的工作，响应返回的是被取消 cluster 的首个 CTA 坐标。接手方的各个 CTA 要根据自己在 cluster 里的位置算出对应的任务，再按调度器的坐标映射处理，而不是 4 个 CTA 都拿着首个坐标去算同一个 tile。[CUTLASS 文档][cutlass-doc]、[v4.3.1 调度器源码][scheduler-source]

另一个要认真对待的是失败。取消请求可能因为已经没有可领的任务而失败，也可能因为硬件要去调度更高优先级的 kernel。所以失败只是告诉当前 worker 别再领了，整个 grid 未必算完：其他 worker 可能还在算，剩下的启动也可能稍后继续被硬件调度。[CUDA 编程指南][cuda-clc]

按 PTX 语义，CTA **观察到取消失败之后再发新的取消请求，属于未定义行为**，失败响应也不能拿来提取任务坐标。所以正确做法是先查成功标志，一旦失败就结束领取循环，并做好流水线的收尾。[PTX 指令说明][ptx-clc]

回到开头的疑问，CLC dynamic persistent kernel 其实有两层：grid 列出全部工作，由 GPU 调度器决定哪些位置真正启动 worker；已经在跑的 worker 再通过 CLC 接走其他还没启动的位置，在同一套执行资源上接着算。把"启动坐标"和"执行实例"区分开，就能明白它为什么既有普通 grid 的动态分配能力，又保留了 persistent kernel 连续干活的执行方式。

## 参考

[1] NVIDIA，[Blackwell Cluster Launch Control][cutlass-doc]，CUTLASS 4.3.1，2025。主要阅读对象，包含三种 kernel 的伪代码、CLC pipeline 与 warp 分工。

[2] NVIDIA，[Work Stealing with Cluster Launch Control][cuda-clc]，CUDA Programming Guide，访问日期 2026-10-19。补充动机、失败语义与 cluster 用法。

[3] NVIDIA，[PTX ISA — `clusterlaunchcontrol.try_cancel` 与 `query_cancel`][ptx-clc]，访问日期 2026-10-19。核实原子取消、16 字节异步响应、坐标解码及失败约束。

[4] NVIDIA，[`sm100_tile_scheduler.hpp`][scheduler-source]，CUTLASS v4.3.1。重点看 `advance_to_next_work`、`issue_clc_query`、`fetch_next_work` 和 `work_tile_info_from_clc_response`。

[5] NVIDIA，[`sm100_pipeline.hpp`][pipeline-source]，CUTLASS v4.3.1。`PipelineCLCFetchAsync` 的 barrier、缓冲 stage 与生产者—消费者接口。

[6] NVIDIA，[`sm100_gemm_tma_warpspecialized.hpp`][kernel-source]，CUTLASS v4.3.1。CLC pipeline 初始化、scheduler warp 循环和 throttle pipeline 的使用。

[cutlass-doc]: https://docs.nvidia.com/cutlass/4.3.1/media/docs/cpp/blackwell_cluster_launch_control.html
[cuda-clc]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html
[ptx-clc]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel
[scheduler-source]: https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp
[pipeline-source]: https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cutlass/pipeline/sm100_pipeline.hpp
[kernel-source]: https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp
