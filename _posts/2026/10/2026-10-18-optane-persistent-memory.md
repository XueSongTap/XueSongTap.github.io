---
layout: article
title: "Intel 傲腾：从 3D XPoint 到持久内存"
tags: AIInfra Optane Memory
---

一个大型数据库，工作数据全放在 DRAM 里，访问快，但容量太贵；放到 SSD 上，容量好扩，处理器却得绕一趟存储 I/O 才能拿到数据。Intel 傲腾（Optane）想在这两者之间再加一层：比 NAND 更适合低延迟访问，容量成本又比 DRAM 低，能装下更大的数据集。

傲腾后来既做成了 SSD，也做成了插在服务器内存通道上的持久内存。底层是同一种介质，接法不同，软件能拿到的能力就差得很远。所以下面顺着材料、接口、软件访问方式一层层往下看。

本文基于 Asianometry 的[《Intel’s Optane Memory: A Promise Unfulfilled》](https://www.youtube.com/watch?v=A2BsmrnegrY)整理，并结合 Intel、Micron 的原始资料补充技术细节。[1]

<!--more-->

## 1. 傲腾想补上哪一层

DRAM 和 NAND 的差别不只在速度和价格，还在访问方式。DRAM 适合处理器频繁读写工作数据；NAND 擅长大容量的持久存储，但有页编程、块擦除这些约束。所以系统里少不了缓存、批量 I/O 和数据搬运，把存储里的内容送到计算侧。

存储级内存（Storage-Class Memory，SCM）瞄准的就是这段空隙：让更多数据离处理器更近，同时断电不丢。SCM 说的是系统里的一个位置，不限定用哪种材料。

2015 年，Intel 和 Micron 发布了 3D XPoint，宣称相比 NAND，速度和耐久性最高可达 1000 倍；相比 DRAM，密度可达 10 倍。[2]

先理清几个名字：**PCM 是相变存储的物理机制，3D XPoint 是存储器技术，Optane 是 Intel 的产品品牌，SCM 则是它想占的系统位置。**

Intel 最初公开的重点是交叉点阵列、选通器和独立单元寻址；后来 TechInsights 拆解实际芯片，确认了 PCM 与选通器的集成结构。[2][4] 所以下面先从相变原理讲起，再看这些单元怎么组织成存储产品。

## 2. 相变存储：用材料状态记录信息

### 2.1 晶态与非晶态

常见的 NAND 靠浮栅或电荷俘获层里的电荷记录信息。PCM（Phase Change Memory）用的则是材料在不同相态下的电阻差：晶态原子排列有序，电阻低；非晶态缺少长程有序结构，电阻高。读的时候测一下电阻，就能分辨存的是哪个状态。[3]

写入要控制一个很小区域的温度变化。想得到高阻的非晶态，先把材料加热熔化，再迅速冷却，让原子来不及重新排成晶体，这一步叫 **RESET**。想得到低阻的晶态，就要让材料结晶，常见做法是加热到结晶温度以上、熔点以下并保持一段时间，这一步叫 **SET**。[3]

![PCM 的 SET、RESET 与读出过程](/img/2026/10/18/optane-pcm-set-reset.jpg)

*图：RESET 通过熔化与快速淬火形成非晶态，SET 使材料结晶，读取时用较小的扰动分辨电阻。摘自原视频 [02:28](https://www.youtube.com/watch?v=A2BsmrnegrY&t=148s)，原图署名 Zhang、Mazzarello、Ma，MRS Bulletin，2019；展示的是通用 PCM 原理。*

写完之后，断电了材料也能保持状态，非易失性就是这么来的。但每次写入都要付出加热和相变的代价，结晶速度、写入电流、保持时间和反复切换的寿命得放在一起优化。把单元做小能减少要加热的材料体积，可又会带来新的可靠性和制造问题。[3]

这条路线的意义在于，存储单元可以用一种不同于 NAND 的机制更新状态，从而往更细粒度、更低延迟的非易失访问上探索。

### 2.2 从单元到 3D XPoint 阵列

3D XPoint 把存储单元放在互相垂直的导线交点上，每个单元配一个 selector，用来选中要访问的位置。首代技术在一颗裸片里堆叠两层存储阵列，容量 128 Gbit，密度靠交叉点结构和多层集成一起撑起来。[2]

![3D XPoint 的两层交叉点阵列示意](/img/2026/10/18/optane-xpoint-crossbar.jpg)

*图：首代 3D XPoint 的交叉点与两层结构示意。摘自原视频 [15:52](https://www.youtube.com/watch?v=A2BsmrnegrY&t=952s)，原始资料为 Intel/Micron 2015 年发布材料。[2]*

TechInsights 拆解第二代 Optane SSD 里的芯片，识别出四层 PCM/OTS 集成结构，说明这条路线也在靠堆叠更多阵列层来提高容量（这里的层数指裸片内的存储阵列层）。[4]

阵列解决了大量单元怎么组织的问题，接下来决定实际体验的，是这些单元通过什么接口连到处理器。

## 3. 同一种介质，SSD 和 DIMM 两条路线

提到傲腾，很多人先想到的是给机械硬盘加速的小容量 M.2 模块。其实 Optane 的产品线要宽得多：2017 年，数据中心级的 P4800X SSD 已经开始早期供货；之后，服务器持久内存又把它带进了 CPU 的内存通道。[5][6]

| 产品形态 | 接入与使用方式 | 主要解决的问题 |
| --- | --- | --- |
| 消费级 Optane Memory 模块 | 经 PCIe 接入，由缓存软件加速存储访问 | 给 HDD 等大容量存储增加一个较快的缓存层 |
| Optane SSD | 经 PCIe／NVMe 提供块存储 | 改善低延迟存储访问，沿用已有 I/O 接口 |
| Optane Persistent Memory DIMM | 接入受支持的 CPU 内存通道 | 扩大内存容量，或向应用提供持久内存 |

SSD 路线的好处是容易接进现有系统。应用照旧读写文件或块设备，驱动、数据库和存储软件都能沿用熟悉的接口。但介质变快以后，软件处理、请求排队、总线和控制器照样要花时间，应用也可能卡在等 CPU 计算或别的资源上。所以介质层快了多少倍，程序并不会原样快多少倍。

DIMM 路线则直接改变了数据所在的位置。大型工作集可以放到 CPU 的内存侧，走内存访问路径取数，不用每次都组织成传统的存储 I/O。代价是平台约束：Optane PMem 的外形和 DDR4 插槽兼容，实际通信用的却是 DDR-T，需要配套的 Xeon、BIOS 和平台支持。[6]

就算接进了内存通道，软件怎么用它也还有两种选择，这就是 Memory Mode 和 App Direct 的区别。

## 4. Memory Mode 与 App Direct：容量和持久性怎样交给软件

![Optane 的 Memory Mode、App Direct 与存储兼容路径](/img/2026/10/18/optane-access-modes.jpg)

*图：Memory Mode 将 DRAM 用作缓存；App Direct 暴露持久内存，既可直接 load/store，也可经存储兼容路径使用。摘自原视频 [25:20](https://www.youtube.com/watch?v=A2BsmrnegrY&t=1520s)，原图为 Intel 2019 年 Xeon 技术概览。[6]*

### 4.1 Memory Mode：把 DRAM 变成缓存

在 Memory Mode 下，Optane PMem 提供大容量的系统内存，DRAM 由内存控制器管理，充当更快的缓存。应用用的还是普通的易失性内存，一般不用为持久化重新设计数据结构。[6][7]

这时 DRAM 的容量不会额外算进操作系统的可用内存。可以这么理解：软件看到的是 PMem 那一大块地址空间，访问频繁的数据由 DRAM 缓存加速，所以两者的标称容量不能直接相加。

另外，介质本身断电能保住数据，Memory Mode 却不向应用提供持久性。这个模式下的软件语义仍是易失内存，应用不能指望重启后还能接着用原来的对象。[7]

这条路线迁移门槛低，但收益要看工作集。热数据能留在 DRAM 缓存里，大容量层就能分担存储空间；访问老是落到 PMem 上，它和 DRAM 在延迟、带宽上的差距就会显现出来。至于能不能省钱，还得看应用的访问模式和当时两种介质的相对价格。

### 4.2 App Direct：把持久内存直接交给应用

App Direct 把 DRAM 和 PMem 作为两种资源分别暴露出来。应用可以把需要持久保存的数据放到 PMem，经过适当映射后直接 load/store；DRAM 继续承载易失的工作数据。图里的 Storage over App Direct 是另一条兼容路径，让软件还能通过存储接口使用这部分容量。[6][7]

这对内存数据库很有吸引力。大型数据库重启后，往往要把工作数据从 SSD 重新加载回 DRAM；如果持久内存里已经留着可恢复的数据结构，就有机会省掉一部分重载。Intel 2019 年产品简报给过一个 SAP HANA 案例：列存储重载从 20 分钟缩短到 1.5 分钟，整个系统重启从 32 分钟缩短到 13.5 分钟。[7] 提速集中在恢复中的具体阶段，不是所有数据库操作都能快这么多。

不过，普通的 `store` 指令执行完，数据未必已经进入掉电后能保住的范围，修改可能还停在 CPU cache 里。所以软件要根据平台的持久域安排写回和排序。ADR 和 eADR 平台的要求不一样：eADR 扩大了受保护的范围，可以少做显式 flush，但持久化顺序照样要保证。PMDK 等持久内存库提供了对应的持久化接口。[8][9]

再往前一步，数据留住了，恢复后还得保持一致。比如一次更新要同时改两个关联字段，掉电时只改完了第一个，持久内存就可能恰好留下一个不完整的状态。所以应用照样需要日志、事务或其他恢复机制，保证这些修改作为一个完整操作生效。持久性、事务一致性和备份容灾，解决的本来就是不同的问题。

于是两种模式走向了不同的采用路径：Memory Mode 尽量不动原有的应用模型，用 DRAM 缓存配合大容量；App Direct 让应用直接用上持久数据，但软件设计和恢复的责任也一并交给了应用。

## 5. 技术做出来了，为什么还是退场

从这两条路径就能看出傲腾商业化难在哪里。好采用的方案主要靠容量和价格证明价值，会一直受 DRAM 和 NAND 产品演进的挤压；更能发挥持久内存特点的方案，又要客户投入平台迁移、软件改造和长期维护。

制造规模和采用速度也互相牵制：客户采用得慢，量产投入就难摊薄；成本降得不够快，客户又更难证明迁移的收益。对打算长期跑数据库和基础设施的用户来说，后续供货和平台支持同样会影响选择。

2021 年 3 月，Micron 宣布停止 3D XPoint 开发，把资源转向支持 CXL 的内存产品。公告说得很直接：市场验证不足以支撑继续投入大笔资金做规模化商业化。[10]

2022 年第二季度，Intel 也开始收缩 Optane 业务，确认了 5.59 亿美元的 Optane 库存减值。[11] 随后，原计划中的 PMem 300 系列被取消；Intel 在公告里把停止未来开发和现有产品支持分开做了说明。[12] 傲腾留下了实际的产品和应用经验，却没能做出足以让这条路线继续扩张的商业规模。

如今讨论[高带宽闪存 HBF]({% post_url 2026/10/2026-10-03-hbf-llm-inference %})，又会碰到“在快速内存之外再加一层容量”的问题。两者定位相近，技术路线却不同：傲腾用 3D XPoint 支撑低延迟存储和 CPU 持久内存；HBF 用的是 NAND，靠宽接口和并行读取，探索怎么存放权重、MoE 专家和可复用的 KV。[13] 傲腾能提供的参考，是新加一层介质以后，接口和软件怎样把它用起来；HBF 适不适合存这些推理数据，还得沿着各自的数据路径去验证。

## 参考

[1] Asianometry，[*Intel’s Optane Memory: A Promise Unfulfilled*](https://www.youtube.com/watch?v=A2BsmrnegrY)，2026-09-20。本文的阅读起点，三张插图沿用视频中的资料画面。

[2] Intel、Micron，[*Intel and Micron Produce Breakthrough Memory Technology*](https://www.intc.com/news-events/press-releases/detail/324/intel-and-micron-produce-breakthrough-memory-technology)，2015-07-28。技术发布指标、首代阵列与 selector。

[3] Geoffrey W. Burr 等，[*Phase Change Memory Technology*](https://arxiv.org/abs/1001.1164)，2010，arXiv:1001.1164。PCM 机制与器件约束。

[4] TechInsights，[*Intel 2nd Generation XPoint Memory*](https://techinsights.com/blog/memory/intel-2nd-generation-xpoint-memory)。Optane P5800X 芯片的一手拆解，识别四层 PCM/OTS 结构；访问日期 2026-10-18。

[5] Intel，[*Intel Introduces the World’s Most Responsive Data Center Solid State Drive*](https://download.intel.com/newsroom/2021/archive/2017-03-19-news-intel-introduces-worlds-most-responsive-data-center-solid-state-drive.pdf)，2017-03-19。P4800X 早期供货公告。

[6] Intel，[*Second Generation Intel Xeon Processor Scalable Family Technical Overview*](https://www.intel.com/content/www/us/en/developer/articles/news/second-generation-intel-xeon-processor-scalable-family-technical-overview.html)，2019。平台要求、DDR-T、运行模式及访问路径。

[7] Intel，[*Intel Optane DC Persistent Memory Product Brief*](https://www.intel.com/content/dam/www/public/us/en/documents/product-briefs/optane-dc-persistent-memory-brief.pdf)，2019。Memory Mode、App Direct 与 SAP HANA 案例；阶段耗时见脚注 3。

[8] Intel，[*Intel Xeon Processor Scalable Family Technical Overview*](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-xeon-processor-scalable-family-overview.html)。第 3 代 Xeon 的 ADR/eADR 与持久域说明；访问日期 2026-10-18。

[9] PMDK，[*libpmem*](https://pmem.io/pmdk/libpmem/)。持久化、flush 与 drain 接口说明；访问日期 2026-10-18。

[10] Micron，[*Micron Updates Data Center Portfolio Strategy to Address Growing Opportunity for Memory and Storage Hierarchy Innovation*](https://investors.micron.com/news/press-release/2021/Micron-Updates-Data-Center-Portfolio-Strategy-to-Address-Growing-Opportunity-for-Memory-and-Storage-Hierarchy-Innovation-03-16-2021/default.aspx)，2021-03-16。

[11] Intel，[*Intel Reports Second-Quarter 2022 Financial Results*](https://www.intc.com/news-events/press-releases/detail/1563/intel-reports-second-quarter-2022-financial-results)，2022-07-28。Optane 业务收缩及库存减值。

[12] Intel，[*Announcement: Intel Optane Persistent Memory 300 Series*](https://www.intel.com/content/www/us/en/support/articles/000093792/technologies/memory-and-storage.html)。公告所述取消生效日期为 2023-01-31。

[13] Chinnakrishnan Ballapuram、Dongsop Lee，[*High Bandwidth Flash (HBF™) High-Level Base Die Specification*, v0.7.0](https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf)，OCP，2026-08-03。HBF 的 NAND、主机接口与软件使用方式。
