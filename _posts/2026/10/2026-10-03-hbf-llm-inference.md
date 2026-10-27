---
layout: article
title: "高带宽闪存 HBF：会给推理带来什么样的变化"
tags: LLM HBF AIInfra
last_modified_at: "2026-10-18"
---

> 前置阅读：[LLM 推理（Inference）性能推导]({% post_url 2025/12/2025-12-13-infer %}) · [DualPath：用双路径 KV-Cache 加载打破 Agentic 推理的存储瓶颈]({% post_url 2026/03/2026-03-22-dualpath-kv-cache-inference %}) · [昇腾 950 的两种显存方案：白鹭、朱雀与 PD 分离]({% post_url 2026/09/2026-09-30-ascend950-hibl-hizq-memory %})

推理时，显存里有两类数据值得单独拿出来看：模型权重加载后基本只读；KV cache 随上下文增长，读多写少。两者都要占很大的空间，也都要求很高的读取速度。

HBF（High Bandwidth Flash，高带宽闪存）用 NAND 换来大容量，再靠宽接口和内部并行把读取带宽拉上去，正好能和这两类数据对上：**只读权重可以试着放进 HBF，尤其是总参数量大、每个 token 只激活部分专家的 MoE；有复用价值的 KV，则可以考虑放进 HBM 之外的二级缓存。**

本文以 Asianometry 的[《High Bandwidth Flash Is Coming》](https://www.youtube.com/watch?v=3nTpW52nioI)为起点，结合 OCP 规格和两篇 HBF 推理研究，沿这两条路径聊聊可行性。[1–6] 重点放在哪些数据适合放进去、怎样读回来；具体的性能和成本收益，要等实现和实验来回答。

<!--more-->

## 1. 推理的时候，什么在占显存

### 1.1 权重：容量大，加载后基本只读

用固定模型做推理时，权重会被不同请求、不同 token 反复使用，一般只在加载或切换模型时才更新。这种“写一次、长期读”的访问方式，刚好对上闪存擅长读的特点。

设模型有 $P$ 个参数，每个参数用 $q$ bit 表示，参数本体的存储量为

$$
M_{\mathrm{weights}}=P\frac{q}{8}.
$$

比如一个万亿参数模型用 8 bit 权重，光参数本体就约 1 TB，还没算量化元数据和运行时空间。MoE 也得保存全部专家的权重，哪怕单个 token 只调用其中一部分。所以推理系统既要装得下完整模型，又要及时把当前计算用到的权重读出来。

### 1.2 KV cache：持续追加，历史数据反复读取

KV cache 存的是已处理 token 的 Key 和 Value。常规自回归解码每生成一个新 token，就追加一份它的 KV；后续 token 做 attention 时，再读取所需的历史 KV。如果 attention 用的是完整历史上下文，每步只新增一个 token 的 KV，却要读此前许多 token 的 KV，所以上下文越长，逻辑访问上就越是读多写少。

它的容量同样随上下文长度和并发请求数增长。对于各层形状相同的常规 MHA／GQA，设层数为 $L$，KV head 数为 $h_{\mathrm{KV}}$，每个 head 的维度为 $d_h$，每个缓存元素占 $s_{\mathrm{KV}}$ byte，第 $i$ 个请求保留 $S_i$ 个 token，逻辑 KV 总量为

$$
M_{\mathrm{KV}}
=2Lh_{\mathrm{KV}}d_hs_{\mathrm{KV}}\sum_i S_i.
$$

前面的 2 对应 Key 和 Value。注意这里代入的是 KV head 数：GQA 让多组 Query head 共享较少的 KV head，不能直接拿 Query head 数来算缓存。[4]

举个具体配置：32 层、8 个 KV head、head dimension 为 128，每个元素 2 byte。一个请求保留 32768 个 token，KV cache 就有 4 GiB；32 个这样的并发请求就要 128 GiB。

这就给了 HBF 另一个可以探索的方向：在 HBM 之外多留一些可复用的上下文。不过，“读多写少”说的是模型怎么用 KV；放进二级缓存以后实际有多少保存和读回流量，还要看缓存策略。

## 2. HBF 提供了什么：大容量与高读取带宽

HBF 的基本结构是把 NAND core die 堆叠在逻辑 base die 上，用 TSV 连起来，再由基底上的控制器和 PHY 通过 UCIe 接口和计算芯片通信。说白了就是借用 HBM 的封装思路，把存储介质换成密度更高的闪存。这里的裸片堆叠，和 3D NAND 在单颗裸片内部堆叠存储单元层，不是一个层级的东西。[2]

![HBF 的 NAND 裸片、TSV、逻辑基底与计算芯片连接示意](/img/2026/10/03/hbf-package-die-stack.jpg)

*图：原视频 08:04 的结构示意，展示多颗 NAND 裸片与逻辑基底的连接。[1]*

OCP 0.7.0 版规格明确了 HBF 和计算芯片怎么连：**封装内的 UCIe 互连，上面跑 AXI 协议**。UCIe 用来连接同一封装里的不同芯粒，GPU／AI 处理器通过这条链路和 HBF 逻辑基底通信，基底再去访问 NAND 堆叠。规格选的是先进封装，通道最长支持 2 mm（按两端连接凸点之间测量）。所以这套方案里的 HBF 是贴着计算芯片放、靠短距离宽接口连接的独立闪存堆栈，走的不是普通 SSD 那条 PCIe／NVMe 路线。[2]

连接方式规格已经定了，但哪些 GPU 支持、集成几颗 HBF、怎么和 HBM 搭配，要看具体产品。计算芯片这一侧也得有配套的接口、控制器和数据通路，现有显卡光凭这份规格是加不上 HBF 的。[2][3]

OCP 0.7.0 版规格里的最高档配置是 16 个 host channel，每通道 64 bit、32 GT/s。按表 2 的 75% 接口效率算，有效接口带宽为 3072 GB/s；表 3 中，16 层裸片对应 512 GiB 用户容量。[2] SK hynix 也把 HBF 定位在 HBM 和 SSD 之间。[3]

下面列几个量级做对照：HBM、HBF 按单堆栈，SSD 按单块盘，带宽用十进制单位。产品规格、接口目标和论文假设在表里分别标注。

| 对象 | 容量示例 | 读取／接口带宽 | 写入说明 |
| --- | --- | --- | --- |
| KIOXIA CM7-V，单块 PCIe 5.0 SSD | 3.2 TB | 顺序读最高 14 GB/s | 顺序写最高 6.75 GB/s，产品规格 [7] |
| Micron HBM3E，单堆栈 | 24／36 GB | 总接口带宽超过 1200 GB/s | 读写共享接口带宽 [8] |
| Micron HBM4，12-high 单堆栈 | 36 GB | 总接口带宽超过 2800 GB/s | 读写共享接口带宽 [9] |
| OCP HBF，最高档单堆栈 | 512 GiB | 有效接口带宽 3072 GB/s | 接口数值不代表 NAND 持续写带宽 [2] |
| 《Exploring HBF》建模的单堆栈 | 512 GB | 读 1600 GB/s | 写 16 GB/s，为论文假设 [5] |

HBF 的吸引力就在这里：单堆栈几百 GB 容量，读取目标到了 TB/s 量级，计算芯片有机会从闪存层及时拿到数据。至于整机能拿到多少带宽，还要看堆栈数量和访问并行度。

写入就是另一回事了。《Exploring HBF》按每堆栈 1.6 TB/s 读、16 GB/s 写来建模，其中 100 μs 的页编程延迟是因为缺少公开的写性能数据而设的假设。[5] 这组数字是用来说明读写不对称的，不能当产品规格看。再加上 NAND 的编程／擦除寿命有限，少写多读的数据才更能发挥它的长处。[5][6]

顺着这个特点，先看最直接的一类数据：权重。

## 3. 权重放进 HBF：为什么 MoE 尤其值得一试

### 3.1 保存完整权重，按需送到计算侧

一种可以尝试的组织方式是：完整权重常驻 HBF，计算时读取当前需要的权重块；HBM 或 SRAM 做计算侧缓冲，也可以缓存反复用到的热点权重。[5] 这样大模型的存储容量由 HBF 分担，HBM 就能腾出空间给活跃数据和临时张量。

接下来的问题是：每一步到底要读多少。

### 3.2 MoE 把总参数量与当前使用量分开

Dense 模型一次前向会用到各层的主要权重。MoE 的路由专家不一样：全部专家都得存着，但一个 token 只选其中几个。**模型需要的总容量很大，单个 token 要读的专家权重却可能只占一小部分。** 这正是 MoE 和 HBF 合得来的地方。

比如一层有 256 个等大小的路由专家，每个 token 选 8 个，那么单个 token 只用到这一层 3.125% 的路由专家权重。剩下的专家照样要有地方放，但不必为这个 token 全读到计算侧。Attention 和共享专家等公共部分另算。

多个 token 一起跑时，用到的专家集合会变大。为了看清这个趋势，用一个简化模型：一层有 $N$ 个等大小专家，每个 token 均匀选择 $k$ 个不同专家，token 之间独立选择。一个专家没被 $B$ 个 token 中任何一个选中的概率为 $(1-k/N)^B$，所以该步用到的专家比例期望为

$$
f(B)=1-\left(1-\frac{k}{N}\right)^B.
$$

![均匀 MoE 路由下，batch 增大时专家工作集占全部专家的比例](/img/2026/10/03/hbf-moe-working-set.png)

*图：按上述简化模型绘制，$N=256$、$k=8$，只统计路由专家的覆盖比例。*

在这组配置下，batch 为 8 时期望覆盖约 22.4% 的专家，batch 到 128 时覆盖率已接近 98.3%。但同一个专家的权重也能被这一批里的多个 token 复用，所以这张图反映的是每步工作集怎样扩大，实际搬运量还得结合权重缓存和 kernel 的复用方式来看。

所以 MoE 给 HBF 留的机会是：用大容量存下全部专家，再按路由结果读当前需要的那部分。小批量时激活的专家少，单步读取量可能不大；批量上去以后，就要靠权重复用和更多并行读取来满足供数需求。

### 3.3 关键在于：需要的权重能不能及时送到

设一个 step 需要从 HBF 新取的权重为 $D_{\mathrm{miss}}$，HBF 到计算侧的有效读取带宽为 $\beta_{\mathrm{delivered}}$，纯搬运时间的下界就是

$$
t_{\mathrm{read,min}}
=\frac{D_{\mathrm{miss}}}{\beta_{\mathrm{delivered}}}.
$$

权重缓存能减小 $D_{\mathrm{miss}}$，预取能让一部分搬运和计算重叠。但 MoE 选哪些专家取决于路由结果，能提前多久读、哪些专家值得常驻 HBM，都得由运行时来定。而且计算侧的 HBM 再快，没命中缓存的权重还是要走 HBF 这条供数路径。[5]

读取量大的时候，也可以多加几个独立的 HBF 堆栈和接口，把同一步要用的数据分散到不同堆栈上，OCP 规格就给了一个 host 连接四个 HBF chipset 的示例。[2] 说到底，模型容量、主机接收能力和当前计算的读取需求要同时满足。

所以 MoE 权重是验证可行性比较直接的起点：只读省掉了写入负担，稀疏激活又给按需读取留了空间。能不能维持目标推理速度，就看缓存、预取和供数能力配合得怎么样了。

## 4. KV cache：作为 HBM 之外的二级缓存

### 4.1 把可复用的上下文留在更大的容量层

KV 和权重不同，它会随请求不断产生：一次 prefill 就能生成大量 KV，decode 又一点点往上追加。好在上下文不变时，已有的 KV 通常可以继续复用，所以它也有“存一次、读多次”的可能。

这里先讨论的是把 HBF 当作 **HBM 之后、由运行时管理的二级 KV 缓存**。这个“二级”指运行时管理的数据分层，不是 GPU 硬件上的 L2 cache，也不是通过 PCIe 外接；HBM 和 HBF 可以同在计算芯片的封装里。

一种可以尝试的安排是：当前请求频繁用到的 KV 留在 HBM；暂时不活跃、但预计还会用到的共享前缀或会话块，放到 HBF；后续请求命中时，再把需要的块读回 HBM 参与计算。

比如一段较长的公共前缀被 HBM 淘汰后，可以继续留在 HBF。之后有请求用到相同前缀，运行时先核对模型版本、token 前缀等缓存匹配条件，命中就恢复对应的 KV，新增部分照常走推理流程。这样缓存能留住的范围更大，可以减少重复 prefill，或者少从更远的存储层读数据。

长会话也类似：暂时没有生成任务时，把值得保留的 KV 块放到 HBF，等会话恢复时再读回来。保存、恢复和淘汰都按块进行，由运行时根据复用价值来安排。这是分层缓存的设计思路，实际收益还得结合请求行为来验证。[6]

### 4.2 历史 KV 不一定是冷数据

给 KV 分层时，放在哪一层要按访问频率来定。一个 token 离当前生成位置很远，不代表它很少被读：如果活跃请求还在用完整历史 attention，这块旧 KV 每一步都可能参与计算。

所以活跃请求正在反复读的 KV 应该优先留在 HBM。如果工作集超出了 HBM 容量，解码期间必须从 HBF 分块读入，那 HBF 的读取就进了每一步的计算路径，得专门安排读取并行和计算重叠。这和“请求恢复时读回一次”的缓存用法相比，对带宽和时延的要求不一样。[5][6]

### 4.3 读多写少，得在 HBF 这一层也成立

KV 在 attention 里被反复读，这是考虑 HBF 的出发点。但热数据留在 HBM 之后，这些读取大多也就留在了 HBM。如果 HBF 一直在接收被淘汰的块，而这些块以后又很少再命中，HBF 自己承受的流量反而可能以写为主。

所以保存策略要挑有复用价值的块。稳定的共享前缀，存一次可能换来后面多次恢复；很快过期、之后不再用的 KV，存下来只会增加写入和容量占用。合并小写入、按块组织数据有助于适配闪存，但首先得决定哪些内容值得存。OCP 的 KV 指南也把布局和读写组织交给了理解模型访问模式的主机软件。[2]

两篇 HBF 研究各给了一个侧面：《Exploring HBF》讨论权重和 KV 驻留在 HBF 的主内存方案；《HBF Sucks?》研究带 save/restore 策略的分层 KV 池，指出热复用被近端内存吸收后，容量层可能接到大量低复用的写入。[5][6] 后者也讨论了共享前缀靠多次复用摊薄写入成本的机会。所以 KV 二级缓存行不行，最终要落到具体的保存和复用策略上。

这么看，HBF 给推理提供了两条可以探索的路：一是给只读权重，尤其是 MoE 的完整专家集合，提供更大的存放空间；二是让值得复用的 KV 离开 HBM 后还能继续留着。前者的关键是按需供数，后者的关键是缓存命中和保存成本。

## One More thing

说到在 HBM 和 SSD 之间加一层，很容易想到 Intel 傲腾。它用 3D XPoint 做过低延迟 SSD 和 CPU 持久内存，最后退出了市场。但傲腾当年的需求其实很明确：扩大内存容量、改善存储访问延迟、缩短内存数据库的恢复时间。Intel 的产品资料里就有 SAP HANA 减少数据重载时间的案例，Memory Mode 也提供了一条不怎么改应用就能扩容的路径。[10]

从产品路线看，傲腾得分别证明几种不同的价值：做 SSD，要让低延迟值回额外的容量成本；做内存扩展，要在容量收益和访问性能之间找平衡；用 App Direct 发挥持久性，又牵涉到数据结构、持久化顺序和恢复逻辑的适配。每一项需求单独看都成立。Micron 在 2021 年停止 3D XPoint 开发时给出的理由是，市场验证不足以支撑继续投入大量资金做规模化商业化；Intel 随后也收缩了 Optane 业务。[11][12]

{% assign optane_post = site.posts | where: "slug", "optane-persistent-memory" | first %}
{% if optane_post %}
> 延伸阅读：[Intel 傲腾：从 3D XPoint 到持久内存]({{ optane_post.url | relative_url }})。
{% endif %}

**相比之下，HBF 面向 LLM 推理，需求要集中得多，尤其是只读的 MoE 权重。**

需求集中还可能带来一个好处：软件适配可以围绕推理引擎来做，专家放置、预取、KV 的保存与淘汰，都有明确的地方来管。


## 参考

[1] Asianometry，[*High Bandwidth Flash Is Coming*](https://www.youtube.com/watch?v=3nTpW52nioI)，2026-09-24。本文的阅读起点；结构插图摘自视频 08:04。

[2] Chinnakrishnan Ballapuram、Dongsop Lee，[*High Bandwidth Flash (HBF™) High-Level Base Die Specification*, v0.7.0](https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf)，OCP，2026-08-03。第 4 章的容量、带宽与系统组织，第 5.1.1、5.2.1 节的封装内 UCIe 与 AXI 协议，以及第 13.3 节的权重和 KV 使用指南。

[3] SK hynix，[*SK hynix Unveils First HBF Standard Specifications with Sandisk, Presenting AI Memory Solutions at ‘FMS 2026’*](https://news.skhynix.com/en/hbf-at-fms-2026/)，2026-08-04。

[4] Joshua Ainslie 等，[*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245)，2023，arXiv:2305.13245。

[5] Dowon Son 等，[*Exploring High-Bandwidth Flash for Modern LLM Inference: Opportunities and Challenges*](https://arxiv.org/abs/2608.13868)，2026，arXiv:2608.13868。本文引用的读写带宽为论文建模参数。

[6] Zhuoran Li 等，[*HBF Sucks? A Full-Stack Characterization of High-Bandwidth Flash for KV-Centric LLM Serving*](https://arxiv.org/abs/2608.11668v4)，2026，arXiv:2608.11668，v4。重点参考第 7.1 节对不同放置策略与 KV 复用的讨论。

[7] KIOXIA，[*CM7-V Series (2.5-inch)*](https://americas.kioxia.com/en-us/business/ssd/enterprise-ssd/cm7-v.html)，产品规格，访问日期 2026-10-18。采用 3.2 TB 型号的顺序读写数值。

[8] Micron，[*HBM3E*](https://www.micron.com/products/memory/hbm/hbm3e)，产品规格，访问日期 2026-10-18。

[9] Micron，[*HBM4*](https://www.micron.com/products/memory/hbm/hbm4)，产品规格，访问日期 2026-10-18。

[10] Intel，[*Intel Optane DC Persistent Memory Product Brief*](https://www.intel.com/content/dam/www/public/us/en/documents/product-briefs/optane-dc-persistent-memory-brief.pdf)，2019。Memory Mode、App Direct 与 SAP HANA 数据重载案例。

[11] Micron，[*Micron Updates Data Center Portfolio Strategy to Address Growing Opportunity for Memory and Storage Hierarchy Innovation*](https://investors.micron.com/news/press-release/2021/Micron-Updates-Data-Center-Portfolio-Strategy-to-Address-Growing-Opportunity-for-Memory-and-Storage-Hierarchy-Innovation-03-16-2021/default.aspx)，2021-03-16。停止 3D XPoint 开发的市场验证与投入判断。

[12] Intel，[*Intel Reports Second-Quarter 2022 Financial Results*](https://www.intc.com/news-events/press-releases/detail/1563/intel-reports-second-quarter-2022-financial-results)，2022-07-28。Optane 业务收缩。

[13] SK hynix，[*SK hynix Announces 2Q26 Financial Results*](https://news.skhynix.com/en/q2-2026-business-results/)，2026-07-29。AI 基础设施带动的内存需求与供给情况；所述供需判断针对其内存市场整体。
