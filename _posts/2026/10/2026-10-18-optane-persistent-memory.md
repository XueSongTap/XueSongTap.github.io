---
layout: article
title: "Intel 傲腾：从 3D XPoint 到持久内存"
tags: AIInfra Optane Memory
---

一个大型数据库，把工作数据全部放在 DRAM 中，可以快速访问，但容量成本很高；放在 SSD 上，容量容易扩大，处理器却要经过存储 I/O 路径才能取到数据。Intel 傲腾（Optane）试图在这两者之间增加一层：比 NAND 更适合低延迟访问，又能以低于 DRAM 的容量成本保存更大的数据集。

傲腾后来既做成了 SSD，也做成了插在服务器内存通道上的持久内存。同一种底层介质，接入方式不同，软件得到的能力就很不一样。理解它的关键，是顺着材料、接口和软件访问方式往下看。

本文基于 Asianometry 的[《Intel’s Optane Memory: A Promise Unfulfilled》](https://www.youtube.com/watch?v=A2BsmrnegrY)整理，并结合 Intel、Micron 的原始资料补充技术细节。[1]

<!--more-->

## 1. 傲腾想补上哪一层

DRAM 与 NAND 的差别，除了速度和价格，还有访问方式。DRAM 适合处理器频繁读写工作数据；NAND 擅长提供大容量的持久存储，但有页编程、块擦除等约束。系统因此需要缓存、批量 I/O 和数据搬运，把存储里的内容送到计算侧。

存储级内存（Storage-Class Memory，SCM）就是在这段空隙中寻找机会：让更多数据靠近处理器，同时保留断电不丢失的能力。它是一种系统定位，并不指定某种材料。

2015 年，Intel 与 Micron 发布了 3D XPoint。发布时宣称，相比 NAND，速度和耐久性最高可达 1000 倍；相比 DRAM，密度可达 10 倍。[2]

这里先把几个名字分开：**PCM 描述相变存储的物理机制，3D XPoint 是存储器技术，Optane 是 Intel 的产品品牌，SCM 则描述它想占据的系统位置。**

Intel 最初公开的重点是交叉点阵列、选通器和独立单元寻址；TechInsights 后来对实际芯片的拆解进一步确认了 PCM 与选通器的集成结构。[2][4] 因此，可以先从相变原理理解它，再看这类单元怎样被组织成存储产品。

## 2. 相变存储：用材料状态记录信息

### 2.1 晶态与非晶态

常见 NAND 通过浮栅或电荷俘获层中的电荷状态记录信息。PCM（Phase Change Memory）利用的则是材料在不同相态下的电阻差：晶态中原子排列更有序，电阻较低；非晶态中缺少长程有序结构，电阻较高。读取时检测电阻，便能区分存储状态。[3]

写入需要控制一个很小区域的温度变化。要得到高阻非晶态，先把材料加热熔化，再迅速冷却，使原子来不及重新排列成晶体，这个过程称为 **RESET**。要得到低阻晶态，则需要让材料完成结晶，常见方式是加热到结晶温度以上、熔点以下并保持一定时间，这个过程称为 **SET**。[3]

![PCM 的 SET、RESET 与读出过程](/img/2026/10/18/optane-pcm-set-reset.jpg)

*图：RESET 通过熔化与快速淬火形成非晶态，SET 使材料结晶，读取使用较小扰动辨别电阻。摘自原视频 [02:28](https://www.youtube.com/watch?v=A2BsmrnegrY&t=148s)，原图署名 Zhang、Mazzarello、Ma，MRS Bulletin，2019；展示的是通用 PCM 原理。*

写入完成后，材料可以在断电时保持状态，这就是非易失性的来源。但写入仍要付出加热和相变的代价。结晶速度、写入电流、保持时间与反复切换的寿命，都需要一起优化。把单元做小，有助于减少需要加热的材料体积，也会带来新的可靠性和制造问题。[3]

这条路线的意义，是让存储单元可以用不同于 NAND 的机制更新状态，探索更细粒度、更低延迟的非易失访问。

### 2.2 从单元到 3D XPoint 阵列

3D XPoint 将存储单元放在互相垂直的导线交点上，每个单元配有 selector，用于选择要访问的位置。首代技术在一颗裸片中堆叠两层存储阵列，容量为 128 Gbit。交叉点结构和多层集成共同提高了密度。[2]

![3D XPoint 的两层交叉点阵列示意](/img/2026/10/18/optane-xpoint-crossbar.jpg)

*图：首代 3D XPoint 的交叉点与两层结构示意。摘自原视频 [15:52](https://www.youtube.com/watch?v=A2BsmrnegrY&t=952s)，原始资料为 Intel/Micron 2015 年发布材料。[2]*

TechInsights 对第二代 Optane SSD 中芯片的拆解识别出四层 PCM/OTS 集成结构，说明这条路线也在通过增加阵列层数提高容量。[4] 这里的层数指裸片内的存储阵列。

阵列解决了如何组织大量单元的问题。接下来决定应用体验的，是这些单元经过什么接口与处理器连接。

## 3. 同一种介质，SSD 与 DIMM 是两条路线

提到傲腾，很容易先想到给机械硬盘加速的小容量 M.2 模块。但 Optane 的产品范围更广：2017 年，数据中心 P4800X SSD 已经开始早期供货；随后，服务器持久内存又把它带到了 CPU 的内存通道。[5][6]

| 产品形态 | 接入与使用方式 | 主要解决的问题 |
| --- | --- | --- |
| 消费级 Optane Memory 模块 | 经 PCIe 接入，由缓存软件加速存储访问 | 给 HDD 等大容量存储增加一个较快的缓存层 |
| Optane SSD | 经 PCIe／NVMe 提供块存储 | 改善低延迟存储访问，沿用已有 I/O 接口 |
| Optane Persistent Memory DIMM | 接入受支持的 CPU 内存通道 | 扩大内存容量，或向应用提供持久内存 |

SSD 路线的优势是容易接入已有系统。应用仍然读写文件或块设备，驱动、数据库和存储软件可以沿用熟悉的接口。介质访问变快之后，软件处理、请求排队、总线和控制器仍然会消耗时间；应用还可能在等待 CPU 计算或其他资源。因此，介质层的性能倍率不会原样变成程序的加速倍率。

DIMM 路线则进一步改变了数据的位置。大型工作集可以放到 CPU 的内存侧，通过内存访问路径取得，而不必把每次访问都组织成传统存储 I/O。它也带来了平台约束：Optane PMem 虽然采用与 DDR4 插槽兼容的外形，实际通信使用 DDR-T，需要匹配的 Xeon、BIOS 和平台支持。[6]

即使接入了内存通道，软件如何使用它仍然有两种选择。这正是 Memory Mode 与 App Direct 的区别。

## 4. Memory Mode 与 App Direct：容量和持久性怎样交给软件

![Optane 的 Memory Mode、App Direct 与存储兼容路径](/img/2026/10/18/optane-access-modes.jpg)

*图：Memory Mode 将 DRAM 用作缓存；App Direct 暴露持久内存，既可直接 load/store，也可经存储兼容路径使用。摘自原视频 [25:20](https://www.youtube.com/watch?v=A2BsmrnegrY&t=1520s)，原图为 Intel 2019 年 Xeon 技术概览。[6]*

### 4.1 Memory Mode：把 DRAM 变成缓存

在 Memory Mode 下，Optane PMem 提供较大的系统内存容量，DRAM 由内存控制器管理，作为较快的缓存。应用仍然使用普通的易失性内存，通常不需要为持久化重新设计数据结构。[6][7]

这时，DRAM 的容量不会额外计入操作系统可用内存。可以把它理解为：软件使用的是 PMem 对应的大容量地址空间，访问频繁的数据由 DRAM 缓存加速。DRAM 与 PMem 的标称容量不能直接相加。

介质本身能断电保持数据，也不代表 Memory Mode 向应用提供持久性。此模式的软件语义仍然是易失内存，应用不能依靠它在重启后继续使用原有对象。[7]

这条路线提供了较低的迁移门槛，但收益依赖工作集。频繁访问的数据能留在 DRAM 缓存里时，大容量层可以分担存储空间；访问不断落到 PMem 时，就更容易感受到它与 DRAM 的延迟、带宽差异。能否降低成本，还要结合应用的访问方式和当时两种介质的相对价格判断。

### 4.2 App Direct：把持久内存直接交给应用

App Direct 将 DRAM 和 PMem 暴露为不同的资源。应用可以把需要持久保存的数据放到 PMem，通过适当映射直接 load/store；DRAM 则继续承载易失工作数据。图中的 Storage over App Direct 是另一条兼容路径，让软件仍可通过存储接口使用这部分容量。[6][7]

这对内存数据库很有吸引力。大型数据库重启后，往往要将工作数据从 SSD 重新加载到 DRAM。如果持久内存中已经保留了可恢复的数据结构，就有机会减少重载。Intel 2019 年产品简报给出的一个 SAP HANA 案例中，列存储重载从 20 分钟缩短到 1.5 分钟，而整个系统重启从 32 分钟缩短到 13.5 分钟。[7] 改善集中在具体恢复阶段，不能把它解释成所有数据库操作都加速相同倍数。

不过，普通的 `store` 指令执行完，并不总意味着数据已经进入掉电后可保留的范围。修改可能仍在 CPU cache 中，软件需要根据平台的持久域安排写回与排序。ADR 与 eADR 平台的要求有所不同；eADR 可以扩大受保护的范围、减少显式 flush，但持久化顺序仍需保证。PMDK 等持久内存库提供了相应的持久化接口。[8][9]

更进一步，数据留得住，也需要在恢复后保持一致。例如，一次更新要同时修改两个关联字段，掉电时只完成了第一个，持久内存可能恰好保留了一个不完整的状态。应用仍然需要日志、事务或其他恢复机制，保证这些修改作为完整操作生效。持久性、事务一致性与备份容灾，分别处理不同的问题。

两种模式由此形成了不同的采用路径：Memory Mode 尽量保留原有应用模型，用缓存配合容量；App Direct 让应用直接使用持久数据，但也要求它承担相应的软件设计和恢复责任。

## 5. 技术做出来以后，为什么仍然退出

从上面的两条路径，可以看出傲腾商业化的难处。较容易采用的方案，主要通过容量和价格证明价值，会持续受到 DRAM 与 NAND 产品演进的影响；更能体现持久内存特点的方案，则需要客户投入平台迁移、软件改造和长期维护。

制造规模也与采用速度有关。客户采用慢，量产投入就更难摊薄；成本下降不够快，客户又更难证明迁移收益。对准备长期运行数据库和基础设施的用户，后续供货与平台支持同样会影响选择。

2021 年 3 月，Micron 宣布停止 3D XPoint 开发，并将资源转向支持 CXL 的内存产品。公告给出的判断很直接：市场验证不足以支撑继续为规模化商业化投入大量资金。[10]

2022 年第二季度，Intel 也开始收缩 Optane 业务，并确认了 5.59 亿美元的 Optane 库存减值。[11] 随后，原计划中的 PMem 300 系列被取消；Intel 的公告将停止未来开发与现有产品支持分开说明。[12] 傲腾留下了实际产品和应用经验，但没有形成足以让这条路线持续扩张的商业规模。

今天讨论[高带宽闪存 HBF]({% post_url 2026/10/2026-10-03-hbf-llm-inference %})，会再次遇到“在快速内存之外增加容量层”的问题。两者定位有相似之处，技术路线却不同：傲腾以 3D XPoint 支撑低延迟存储与 CPU 持久内存；HBF 使用 NAND，通过宽接口和并行读取，探索权重、MoE 专家及可复用 KV 的存放方式。[13] 傲腾的经历提供的参考，是新增一层介质以后，接口和软件怎样把它用起来；HBF 是否适合这些推理数据，还要沿着各自的数据路径验证。

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
