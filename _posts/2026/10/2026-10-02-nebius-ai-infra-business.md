---
layout: article
title: 从 Yandex 到 Nebius：AI 云的生意
tags: LLM
description: 从 Yandex 拆分后的产品路线出发，梳理 Nebius 的 AI 云平台、技术与成本优势，以及合同、融资和现金回流如何支撑这门生意。
---

> 前置阅读：[NVL72 架构解析：72 GPU 全互连拓扑]({% post_url 2026/05/2026-05-04-nvl72-node %}) · [DualPath：用双路径 KV-Cache 加载打破 Agentic 推理的存储瓶颈]({% post_url 2026/03/2026-03-22-dualpath-kv-cache-inference %})

一家搜索公司卖掉了占原集团收入 95% 以上的业务，只留下一部分工程团队和海外资产，然后把主要精力押在 AI 云上。这就是 Nebius 的起点。[1]

从 Yandex 分出来以后，Nebius 做了哪些东西？它凭什么认为自己比其他云厂商技术更强、成本更低？这些能力又怎么变成一门能持续扩张的生意？

本文资料截至 2026 年 10 月 2 日，经营数字主要用已披露的 2026 年第二季度财报。

## 1 拆分 Nebius

原来的 Yandex N.V. 是荷兰上市母公司，主要业务在俄罗斯。2022 年俄罗斯全面入侵乌克兰后，它的纳斯达克股票被停牌，集团开始研究怎么重组所有权和治理结构。[1][3]

2024 年定下的方案是：俄罗斯及部分其他市场的业务卖给由俄罗斯管理层和金融投资者组成的 Consortium.First，原荷兰母公司保留特定国际业务和非俄罗斯资产。7 月 15 日完成最终交割，母公司从此不再持有被出售业务的任何股份；8 月改名 Nebius Group N.V.，股票代码从 YNDX 改成 NBIS，10 月公告了恢复交易的安排。[1][2][3]

所以，今天的俄罗斯 Yandex 继承了品牌和原集团绝大部分业务，Nebius 继承的则是荷兰上市主体。这既是资产出售，也是业务分离，远不是改个名那么简单。

这次出售代价很大。被卖掉的业务占原集团 2023 年前九个月合并收入的 95% 以上，交易还受到俄罗斯要求的至少 50% 强制折价影响。最终公告的交易估值约 54 亿美元，包括现金和股票，实际到手的现金约 28 亿美元。[1][2]

拆分时 Nebius 留下了 AI 云、Toloka 数据服务、Avride 自动驾驶、TripleTen 技术教育，以及芬兰的数据中心等资产。新集团由 Arkady Volozh 领导，重心转向 AI 基础设施。[4]

它的起点优势在团队和系统经验：做搜索要常年运营服务器、存储、网络和大规模计算系统，其中一部分能力可以搬到 AI 云上。但搜索广告收入已经卖掉了，旧集团的用户和利润不会跟着过来，客户关系和商业规模都得从头建。

## 2 分出来之后，Nebius 做了哪些东西

Nebius 没打算只做 GPU 出租。拆分后公开的产品路线，逐步覆盖了云平台、集群管理和模型服务。

| 时间 | 产品或技术动作 | 具体内容 |
|---|---|---|
| 2024 年 | 开源 Soperator | 把 Slurm 集群放到 Kubernetes 上管理，提供统一的软件环境与硬件健康检查。[16] |
| 2024 年 10 月 | 发布新的 AI-native 云平台 | 公告介绍约 400 名云工程师和内部 LLM 研发团队参与的平台，覆盖数据处理、训练、微调和推理。[17] |
| 2025 年 | 完善 AI 存储与集群可靠性 | 提供面向数据流读取、checkpoint 的存储服务，公开主动与被动硬件检查、故障节点隔离方案。[18][19] |
| 2025 年 11 月 | 发布 Token Factory | 将原 AI Studio 路线扩展为生产推理平台，提供开放模型及自定义模型托管、微调和企业权限管理。[20] |
| 2026 年 3 月 | 发布 AI Cloud 3.5 | 增加 Serverless AI、跨 S3 兼容存储的数据迁移、Managed Soperator 配置和集群观测等功能；发布时 Serverless 为公开预览。[21] |
| 2026 年 5 月 | 宣布整合 Eigen AI 与 Clarifai 团队能力 | 前者侧重模型与运行时优化，后者侧重 serving、编排和硬件支持，继续补齐推理技术栈。[22] |

其中最能说明方向的是三个产品。

**Soperator 是开源的集群软件，实现可以直接看。**它用 Kubernetes Operator 管理 Slurm，提供共享的运行环境，并接入 GPU 和网络健康检查。Nebius 的可靠性文章列出了 DCGM、NCCL All-Reduce、InfiniBand 带宽与延迟测试等检查项。[16][19]

**AI Cloud 是一套完整的云产品。**Nebius 公开资料里列的不只是 GPU，还有自有的云软件、服务器与机架设计、存储和托管服务。[17][18] 这里的自有设计指的是系统和基础设施层，GPU 本身还是 NVIDIA 的。

**Token Factory 是模型服务产品。**它覆盖开放模型、客户自有模型和后训练流程。[20] 2026 年 Nebius 又继续整合推理团队，看得出是想把云基础设施和模型服务放在同一个平台上做。[22]

## 3 Nebius 声称的技术和成本优势

Nebius 宣称的优势集中在三处：自有硬件与数据中心设计、接近裸金属的云实例性能，以及面向模型的推理平台。

### 3.1 自有设计与更低 TCO

Nebius 在 2025 年发布 2024 年可持续发展报告时声称，基础设施效率带来约 **20% 更低的总拥有成本（TCO）**，自定义服务器比现成方案在 2024 年省电约 10 GWh，数据中心 PUE 做到了 1.1。[9]

### 3.2 云实例性能与 MLPerf 结果

Nebius 在介绍 MLPerf Inference v5.1 的技术文章里说，它的虚拟机不对 NVIDIA GPU 和 ConnectX InfiniBand 网卡做虚拟化，并拿提交结果来支撑“接近裸金属性能”的说法。[23]

同一篇文章给出了 Llama 3.1 405B 在 GB200 系统上的成绩：和上一轮 MLPerf 同类硬件的最好结果相比，offline 和 server 场景分别提高约 **6.7% 和 14.2%**。[23]

### 3.3 推理平台与更低模型服务成本

Token Factory 的发布公告称，集成的微调和蒸馏流程最多能把推理成本和延迟降低 **70%**。[20]

2026 年的软件战略文章又补充说，Eigen AI 补的是模型优化、量化和部署，Clarifai 补的是服务系统、编排和硬件支持。[22]

有了这些产品、开源项目和基准成绩，Nebius 的技术定位算是有了具体支撑。不过基础设施 TCO、特定配置下的吞吐和模型优化后的推理成本各算各的账，最后能不能变成客户愿意买单的服务，还得放到交付、定价和资金成本里看。

## 4 AI 云扩张：从资本开支（capex）到现金回流

AI 云要扩张，先得扛住投入和回款之间的时间差。机房、供电、GPU 和网络都要先花资本开支（capex），等容量上线、客户验收之后，才能持续提供服务、确认收入。新容量建得越快，前期要垫的钱就越多。

长期合同能让这段时间差好安排一些。Nebius 在 2025 年 9 月宣布和 Microsoft 签下多年基础设施协议时，计划用合同现金流加上以合同为支撑的债务融资来支付相关资本开支，还明确提到 Microsoft 的信用质量有助于改善融资条件。[5] 对出资方来说，已经有客户承诺购买的容量，回款比纯押未来需求的扩建好评估得多。

客户预付款则把一部分未来回款挪到了建设期。Nebius 2026 年第二季度股东信披露，当季签下的交易里约 70% 带预付款，这个比例按交易数量算。[7] 别家也有类似安排：IREN 在 2025 年 11 月披露了和 Microsoft 约 97 亿美元的五年合同，每批容量对应合同金额的 20% 在交付前支付，之后在服务期内抵扣费用。[14] 预付款缓解了建设期的现金压力，但也背上了将来交付服务的义务，收入还是要随服务提供逐步确认。

合同总额也得结合交付时间和购买安排来看。2026 年 3 月和 Meta 签的新协议包括五年内 120 亿美元的专用容量，计划 2027 年初开始交付；另外还有最高 150 亿美元的额外容量购买承诺，这部分 Nebius 打算先卖给其他 AI 云客户，剩下的由 Meta 购买。[6] 这给扩建多加了一层需求保障。公告里最高约 270 亿美元是多年合同金额的上限，实际收入和回款要随容量交付和购买逐步兑现。

从财报能更直观地看到扩张有多占钱。Nebius 2026 年上半年经营现金流约 45.04 亿美元，其中递延收入增加约 43.95 亿美元，是重要的现金来源，客户提前付款对当期现金流的支撑很大。同期购买固定资产、设备及无形资产花掉约 81.30 亿美元现金。[8] 两项相减约为负 36.26 亿美元，也就是说经营现金流还覆盖不了这部分资产购置，缺口要靠已有现金或其他资金来补。这只是两个披露数字的简单相减，没算其他投资和融资活动。

## 5 与 CoreWeave 竞争：融资能力与 TCO

CoreWeave 是最直接的对照：两家都做 AI 云平台，在训练、推理和专用容量上正面竞争。[4][10] 它们都得把大笔资本变成能交付的算力，再靠后面的服务把钱收回来，所以融资条件和算力的实际交付成本，会一起决定扩张速度和盈利空间。

融资能力不只是能拿到多少钱，还包括利率、期限和还款安排。对先建集群、再靠几年服务回款的项目来说，利率决定最后能留下多少收益，期限和还款节奏决定资金能不能周转开。就算合同收入差不多，资金成本不同，两家公司的盈利和继续扩建的能力也可能拉开差距。

CoreWeave 的财报很能说明为什么资金成本要单独拿出来看。2026 年第二季度，它收入 25.75 亿美元，调整后 EBITDA 15.10 亿美元，净亏损却有 6.26 亿美元；当季折旧摊销 13.93 亿美元，净利息费用 6.40 亿美元。[11] 调整后 EBITDA 剔除了折旧、摊销、利息等项目，还含有其他调整。可 GPU 和机房的投入要在使用期内摊进成本，借来的钱也要付利息，所以 EBITDA 利润率很高，照样可能净亏损。

再说 TCO。前面讲的自有服务器、能耗优化和集群可靠性，主要压的是云厂商自己建设和运行基础设施的成本；客户关心的则是跑完一次训练，或者在给定质量和延迟要求下提供推理服务，总共要花多少钱。GPU 每小时单价只是其中一项，任务耗时、故障重跑、存储和网络费用、运维投入，都会算进最终账单。

这正是 Nebius 的技术路线和商业模式接上的地方：基础设施成本更低，可以换成更低的报价或更大的利润空间；集群更稳、有效吞吐更高，则可能让客户完成同样的任务花更少的钱。

大型云厂商还多一层优势。AWS、Azure、Google Cloud 和 Oracle 都自己提供 GPU 计算产品，[15] 客户的数据、权限和其他业务服务往往早就放在这些平台上。专业 AI 云得让客户在具体负载的性能、交付速度、支持或总成本上看到足够的好处，客户才有理由多加一个供应商。

## 参考

公司公告、技术文章和财报用于核实产品与商业事实。成本优势及性能声明保留来源口径，初次资料核对日期为 2026-10-02，第 4、5 节的合同与财务资料于 2026-10-04 复核，产品文档为持续更新页面。

- [1] Yandex，2024，*Yandex N.V. Announces Binding Agreement to Divest its Russia-based Businesses*，[重组方案](https://yandex.com/company/news/05-02-2024)。
- [2] Nebius，2024，*YNV announces successful completion of the divestment of its Russia-based businesses*，[最终交割公告](https://nebius.com/newsroom/ynv-announces-successful-completion-of-the-divestment-of-its-russia-based-businesses)。
- [3] Nebius，2024，*Nebius Group announces planned resumption of trading on Nasdaq and provides investor update*，[更名与交易安排](https://nebius.com/newsroom/nebius-group-announces-planned-resumption-of-trading-on-nasdaq-and-provides-investor-update)。
- [4] Nebius，2024，*Nebius Group to build leading European AI infrastructure company*，[新集团介绍](https://nebius.com/newsroom/nebius-group-to-build-leading-european-ai-infrastructure-company)。
- [5] Nebius，2025，*Nebius announces multi-billion dollar agreement with Microsoft for AI infrastructure*，[合同与融资说明](https://nebius.com/newsroom/nebius-announces-multi-billion-dollar-agreement-with-microsoft-for-ai-infrastructure)。
- [6] Nebius，2026，*Nebius signs new AI infrastructure agreement with Meta*，[五年协议](https://nebius.com/newsroom/nebius-signs-new-ai-infrastructure-agreement-with-meta)。
- [7] Nebius，2026，*Letter to shareholders, Q2 2026*，[股东信](https://assets.nebius.com/assets/a6ecfd85-a6cb-4967-8ef7-9a25bd261f9c/SHLQ226.pdf)。
- [8] Nebius，2026，*Second quarter financial results*，[SEC 财务披露](https://www.sec.gov/Archives/edgar/data/1513845/000110465926094568/tm2622968d1_ex99-1.htm)。
- [9] Nebius，2025，*Nebius Group 2024 Sustainability Report*，[TCO、服务器能耗及 PUE 声明](https://nebius.com/newsroom/nebius-group-2024-sustainability-report-highlights-importance-of-sustainability-to-long-term-value-creation-in-ai-infrastructure)。
- [10] CoreWeave，*CoreWeave Cloud Platform*，[官方平台](https://www.coreweave.com/platform)。
- [11] CoreWeave，2026，*CoreWeave Reports Strong Second Quarter 2026 Results*，[季度财报](https://investors.coreweave.com/news/news-details/2026/CoreWeave-Reports-Strong-Second-Quarter-2026-Results/default.aspx)。
- [12] Nscale，[官方平台介绍](https://www.nscale.com/)；Lambda，[1-Click Clusters 文档](https://docs.lambda.ai/public-cloud/1-click-clusters/)。
- [13] Crusoe，*Crusoe Cloud*，[训练、微调与推理服务](https://www.crusoe.ai/cloud)。
- [14] IREN，2025，*IREN Secures $9.7bn AI Cloud Contract with Microsoft*，[合同公告](https://irisenergy.gcs-web.com/news-releases/news-release-details/iren-secures-97bn-ai-cloud-contract-microsoft)；[Form 8-K：分批预付款与费用抵扣安排](https://www.sec.gov/Archives/edgar/data/1878848/000114036125040072/ef20058139_8k.htm)。
- [15] AWS，[EC2 P5](https://aws.amazon.com/ec2/instance-types/p5/)；Microsoft，[Azure GPU 分布式训练](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu)；Google Cloud，[GPU 实例](https://docs.cloud.google.com/compute/docs/gpus)；Oracle，[OCI GPU](https://www.oracle.com/cloud/compute/gpu/)。
- [16] Nebius，2024，*Explaining Soperator, Nebius’ open-source Kubernetes operator for Slurm*，[实现介绍](https://nebius.com/blog/posts/soperator-in-open-source-explained)；[Soperator 开源仓库](https://github.com/nebius/soperator)。
- [17] Nebius，2024，*Nebius launches new AI-native NVIDIA cloud platform*，[平台发布公告](https://nebius.com/newsroom/nebius-launches-new-ai-native-nvidia-cloud-platform-built-from-the-ground-up-to-accelerate-ai-innovation)。
- [18] Nebius，[AI 存储产品](https://nebius.com/storage)。此页面持续更新，用于核对服务形态；2025 年存储进展另见[第二季度股东信](https://assets.nebius.com/assets/98fceb3b-2951-4647-9864-4b0654af057c/Nebius%20-%20Letter%20to%20shareholders%20-%20Q2%202025.pdf)。
- [19] Nebius，2025，*Fault-tolerant training: How we build reliable clusters for distributed AI workloads*，[健康检查与故障处理](https://nebius.com/blog/posts/how-we-build-reliable-clusters)。
- [20] Nebius，2025，*Nebius launches Nebius Token Factory to deliver production AI inference at scale*，[产品及成本声明](https://nebius.com/newsroom/nebius-launches-nebius-token-factory-to-deliver-production-ai-inference-at-scale)。
- [21] Nebius，2026，*Nebius AI Cloud 3.5 introduces serverless AI*，[平台更新](https://nebius.com/newsroom/nebius-ai-cloud-3-5-introduces-serverless-ai-to-give-developers-frictionless-compute-for-real-world-ai)。
- [22] Nebius，2026，*The AI cloud will be won at the software layer*，[Eigen AI、Clarifai 与推理软件路线](https://nebius.com/blog/posts/the-ai-cloud-will-be-won-at-the-software-layer)。
- [23] Nebius，2025，*Nebius proves bare-metal-class performance for AI inference workloads in MLPerf Inference v5.1*，[配置、结果与对比口径](https://nebius.com/blog/posts/bare-metal-class-performance-mlperf-inference)。
