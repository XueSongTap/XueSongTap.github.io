---
layout: article
title: 从 Yandex 到 Nebius：AI 云的生意
tags: LLM
description: 从 Yandex 拆分后的产品路线出发，梳理 Nebius 的 AI 云平台、技术与成本优势，以及合同、融资和现金回流如何支撑这门生意。
---

> 前置阅读：[NVL72 架构解析：72 GPU 全互连拓扑]({% post_url 2026/05/2026-05-04-nvl72-node %}) · [DualPath：用双路径 KV-Cache 加载打破 Agentic 推理的存储瓶颈]({% post_url 2026/03/2026-03-22-dualpath-kv-cache-inference %})

一个搜索公司出售了占原集团收入 95% 以上的业务，留下部分工程团队和海外资产，随后把主要精力投入 AI 云。这就是 Nebius 的起点。[1]

从 Yandex 分出来以后，Nebius 具体做了哪些东西？它为什么认为自己相比其他云厂商，拥有更强的技术能力和更低的成本？这些能力又怎样转化成一门能持续扩张的生意？

本文资料截至 2026 年 10 月 2 日，经营数字主要采用已披露的 2026 年第二季度财报。

## 1 拆分 Nebius

原来的 Yandex N.V. 是荷兰上市母公司，旗下主要经营业务位于俄罗斯。2022 年俄罗斯全面入侵乌克兰后，其纳斯达克股票停牌，集团开始研究所有权与治理重组。[1][3]

2024 年的方案是出售俄罗斯及部分其他市场业务，由俄罗斯管理层与金融投资者组成的 Consortium.First 承接；原荷兰母公司保留特定国际业务和非俄罗斯资产。7 月 15 日完成最终交割后，母公司退出被出售业务的全部持股；8 月更名为 Nebius Group N.V.，股票代码由 YNDX 改成 NBIS，10 月公告恢复交易安排。[1][2][3]

所以，今天的俄罗斯 Yandex 延续了品牌和原集团绝大部分经营业务，Nebius 则延续了荷兰上市主体。这里既有资产出售，也有业务分离，远比一次改名复杂。

出售代价相当大。被出售业务占原集团 2023 年前九个月合并收入的 95% 以上，交易也受到俄罗斯要求的至少 50% 强制折价影响。最终公告中的约 54 亿美元交易估值包含现金和股票，实际收到现金约 28 亿美元。[1][2]

Nebius 在拆分时保留了 AI 云、Toloka 数据服务、Avride 自动驾驶、TripleTen 技术教育，以及芬兰数据中心等资产。新集团由 Arkady Volozh 领导，重点转向 AI 基础设施。[4]

它的优势起点在于团队与系统经验。搜索需要长期运营服务器、存储、网络和大规模计算系统，部分能力可以用于 AI 云。但原搜索广告收入已经出售，旧集团的用户和盈利并不会自动转移到新业务。Nebius 必须重新建立客户关系和商业规模。

## 2 分出来之后，Nebius 做了哪些东西

Nebius 没有停留在租赁 GPU 的业务定位上。拆分后的公开产品路线逐步覆盖云平台、集群管理和模型服务。

| 时间 | 产品或技术动作 | 具体内容 |
|---|---|---|
| 2024 年 | 开源 Soperator | 把 Slurm 集群放到 Kubernetes 上管理，提供统一的软件环境与硬件健康检查。[16] |
| 2024 年 10 月 | 发布新的 AI-native 云平台 | 公告介绍约 400 名云工程师和内部 LLM 研发团队参与的平台，覆盖数据处理、训练、微调和推理。[17] |
| 2025 年 | 完善 AI 存储与集群可靠性 | 提供面向数据流读取、checkpoint 的存储服务，公开主动与被动硬件检查、故障节点隔离方案。[18][19] |
| 2025 年 11 月 | 发布 Token Factory | 将原 AI Studio 路线扩展为生产推理平台，提供开放模型及自定义模型托管、微调和企业权限管理。[20] |
| 2026 年 3 月 | 发布 AI Cloud 3.5 | 增加 Serverless AI、跨 S3 兼容存储的数据迁移、Managed Soperator 配置和集群观测等功能；发布时 Serverless 为公开预览。[21] |
| 2026 年 5 月 | 宣布整合 Eigen AI 与 Clarifai 团队能力 | 前者侧重模型与运行时优化，后者侧重 serving、编排和硬件支持，继续补齐推理技术栈。[22] |

其中有三个产品最能说明它的方向。

**Soperator 是可以查看实现的集群软件。**它使用 Kubernetes Operator 管理 Slurm，提供共享的运行环境，并接入 GPU 和网络健康检查。Nebius 的可靠性文章列出了 DCGM、NCCL All-Reduce、InfiniBand 带宽与延迟测试等检查项目。[16][19]

**AI Cloud 是完整的云产品。**Nebius 的公开资料不仅列出 GPU，也包括自有云软件、服务器与机架设计、存储和托管服务。[17][18] 自有设计主要指系统与基础设施层，不是宣称 NVIDIA GPU 由 Nebius 自研。

**Token Factory 是模型服务产品。**它覆盖开放模型、客户自有模型及后训练流程。[20] 2026 年继续整合推理团队，表明它希望把云基础设施与模型服务做在同一平台上。[22]

## 3 Nebius 声称的技术和成本优势

Nebius 的优势声明主要集中在三个方向：自有硬件与数据中心设计、接近裸金属的云实例性能，以及面向模型的推理平台。

### 3.1 自有设计与更低 TCO

在 2025 年发布的 2024 年可持续发展报告公告中，Nebius 声称基础设施效率带来约 **20% 更低的总拥有成本（TCO）**，自定义服务器相比现成替代方案在 2024 年节省约 10 GWh 电力，并提及数据中心 PUE 达到 1.1。[9]

### 3.2 云实例性能与 MLPerf 结果

Nebius 在 MLPerf Inference v5.1 的技术文章中介绍，其虚拟机环境不对 NVIDIA GPU 和 ConnectX InfiniBand 适配器进行虚拟化，并以提交结果支持“接近裸金属性能”的定位。[23]

同一篇文章给出 Llama 3.1 405B 在 GB200 系统上的结果：相比上一轮 MLPerf 同类硬件的最佳结果，offline 与 server 场景分别提高约 **6.7% 和 14.2%**。[23]

### 3.3 推理平台与更低模型服务成本

Token Factory 发布公告声称，集成微调与蒸馏流程能够让推理成本和延迟最多降低 **70%**。[20]

2026 年的软件战略文章进一步说明，Eigen AI 补强模型优化、量化和部署，Clarifai 补强服务系统、编排与硬件支持。[22]

这些产品、开源项目和基准成绩，让 Nebius 的技术定位有了具体支撑。不过，基础设施 TCO、特定配置的吞吐和模型优化后的推理成本衡量的是不同问题。它们如何转化成客户愿意购买的服务，还要放回交付、定价和资金成本中来看。

## 4 AI 云扩张：从资本开支（capex）到现金回流

AI 云的扩张，首先要跨过投入与回款之间的时间差。机房、供电、GPU 和网络需要先投入资本开支（capex），容量上线并通过客户验收后，才能持续提供服务、确认收入。新容量建设得越快，前期需要垫付的资金就越多。

长期合同的作用，是让这段时间差更容易安排。Nebius 在 2025 年 9 月宣布 Microsoft 多年基础设施协议时，计划结合合同现金流和以合同为支持的债务融资支付相关资本开支，并明确提到 Microsoft 的信用质量有助于改善融资条件。[5] 对资金提供方而言，已有客户承诺购买的容量，比完全押注未来需求的扩建更容易评估回款。

客户预付款则把一部分未来回款提前到了建设期。Nebius 的 2026 年第二季度股东信披露，约 70% 的当季签成交易包含预付款；这里的 70% 指交易数量占比。[7] 其他厂商也采用类似安排：IREN 在 2025 年 11 月披露与 Microsoft 的约 97 亿美元五年合同，每批容量对应合同金额的 20% 在交付前支付，并在后续服务期抵扣费用。[14] 预付款改善了建设期的现金状况，同时也留下了未来交付服务的义务，收入仍要随着服务提供逐步确认。

合同总额也要结合交付时间和购买安排来看。2026 年 3 月的 Meta 新协议包括五年内 120 亿美元专用容量，计划从 2027 年初开始交付；另有最高 150 亿美元的额外容量购买承诺。对于后一部分，Nebius 计划先向其他 AI 云客户销售，剩余容量由 Meta 购买。[6] 这为扩建增加了需求保障。公告中最高约 270 亿美元描述的是多年合同金额的上限，实际收入和回款仍随容量交付与购买逐步实现。

财报能更直观地看出这种扩张对资金的占用。Nebius 2026 年上半年经营现金流约 45.04 亿美元，其中递延收入增加约 43.95 亿美元，是重要的现金流来源；客户提前付款对当期现金流形成了很大支持。同期，购买固定资产、设备及无形资产的现金支出约 81.30 亿美元。[8] 两项相减约为负 36.26 亿美元，说明当期经营现金流仍不足以覆盖这部分资产购置，差额需要由已有现金或其他资金来源补足。这是按两项披露数值计算的简化差额，尚未纳入其他投资和融资活动。

## 5 与 CoreWeave 竞争：融资能力与 TCO

CoreWeave 是最直接的对照对象：双方都提供 AI 云平台，竞争覆盖训练、推理和专用容量。[4][10] 两家公司都需要把大量资本变成可交付的算力，再靠后续服务收回投入。因此，融资条件和算力的实际交付成本，会共同影响它们的扩张速度与盈利空间。

这里的融资能力，既包括能拿到多少钱，也包括利率、期限和还款安排。对于先建集群、再靠数年服务回款的项目，利率影响未来能留下多少收益，期限和还款节奏则影响资金周转。即使合同收入相近，不同的资金成本，也可能让两家公司的盈利和继续扩建的能力拉开差距。

CoreWeave 的财报说明了资金成本为什么值得单独看。2026 年第二季度，它的收入为 25.75 亿美元，调整后 EBITDA 为 15.10 亿美元，净亏损却达到 6.26 亿美元；当季折旧摊销为 13.93 亿美元，净利息费用为 6.40 亿美元。[11] 调整后 EBITDA 剔除了折旧、摊销和利息等项目，也包含其他调整。GPU 和机房的投入要在使用期内计入成本，借来的钱也要付利息，所以较高的 EBITDA 利润率，仍可能与净亏损同时出现

再看 TCO。前面提到的自有服务器、能耗优化和集群可靠性，主要影响云厂商建设、运行基础设施的成本；客户更关心的，则是完成一次训练，或者在给定质量、延迟要求下提供推理服务，总共要花多少钱。GPU 每小时单价只是其中一项，任务完成时间、故障重跑、存储与网络费用、运维投入，都会影响最终账单。

这也是 Nebius 的技术路线与商业模式真正相接的地方：更低的基础设施成本，可以转成更低的报价或更大的利润空间；更稳定的集群和更高的有效吞吐，则可能降低客户完成同一任务的费用

大型云厂商又多了一层优势。AWS、Azure、Google Cloud 和 Oracle 都自己会提供 GPU 计算产品。[15] 客户往往已经把数据、权限和其他业务服务放在这些平台上。专业 AI 云需要让客户在具体负载的性能、交付速度、支持或总成本上看到足够收益，才有理由增加一个供应商。

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
