---
layout: article
title: "高带宽闪存 HBF：更大的容量，能否让推理更便宜？"
tags: LLM HBF AIInfra
---

> 前置阅读：[LLM 推理（Inference）性能推导]({% post_url 2025/12/2025-12-13-infer %}) · [DualPath：用双路径 KV-Cache 加载打破 Agentic 推理的存储瓶颈]({% post_url 2026/03/2026-03-22-dualpath-kv-cache-inference %}) · [昇腾 950 的两种显存方案：白鹭、朱雀与 PD 分离]({% post_url 2026/09/2026-09-30-ascend950-hibl-hizq-memory %})

本文从 Asianometry 的 [《High Bandwidth Flash Is Coming》](https://www.youtube.com/watch?v=3nTpW52nioI)出发，结合 OCP 规格和两篇 HBF 推理研究，讨论权重、MoE 与 KV cache 的不同访问模式。资料核验截至 2026 年 10 月 6 日；下面区分单堆栈、整颗 GPU 与多 GPU 的带宽口径，算例是手算和简化模型，论文结果按各自的仿真配置解释。

<!--more-->

## 1. 先算容量：到底是什么把显存挤满了


### 1.1 权重

权重的容量比较容易估算。设模型一共需要保存 $P$ 个参数，每个参数用 $q$ bit 表示，参数本体占用的字节数就是

$$
M_{\mathrm{weights}}=P\frac{q}{8}.
$$

例如，一个假设的万亿参数模型，参数本体在 8 bit 下约为 1 TB

MoE 也要保存所有专家的权重，尽管一个 token 只调用其中一部分。

### 1.2 KV cache

KV cache 的容量则随请求而变化。

对于各层形状相同的常规 MHA／GQA，设层数为 $L$，KV head 数为 $h_{\mathrm{KV}}$，每个 head 的维度为 $d_h$，每个缓存元素占 $s_{\mathrm{KV}}$ byte，第 $i$ 个请求保留 $S_i$ 个 token，那么逻辑 KV 总量为

$$
M_{\mathrm{KV}}
=2Lh_{\mathrm{KV}}d_hs_{\mathrm{KV}}\sum_i S_i.
$$

前面的 2 分别对应 Key 和 Value。

这里需要代入 KV head 数：GQA 让多组 Query head 共享较少的 KV head，因此不能直接拿 Query head 数计算缓存。[GQA 论文](https://arxiv.org/abs/2305.13245)解释了这种共享关系。[4] 

取一个具体配置：32 层、8 个 KV head、head dimension 为 128、KV 使用 2 byte 元素。当一个请求保留 32768 个 token 时，缓存正好是 4 GiB。32 个这样的并发请求就需要 128 GiB


## 2 HBF、HBM、SSD：先对齐单堆栈的带宽口径

HBF 的基本结构是 NAND core die 堆叠在逻辑 base die 上，通过 TSV 连接，再由基底上的控制器和 PHY 经 UCIe 主机接口与计算芯片通信。它借用了 HBM 的封装思路，存储介质则换成密度更高的闪存。这里的裸片堆叠，与 3D NAND 在一颗裸片内部堆叠存储单元层，是两个层级的结构。[2]

![HBF 的 NAND 裸片、TSV、逻辑基底与计算芯片连接示意](/img/2026/10/03/hbf-package-die-stack.jpg)

*图：原视频 08:04 的结构示意，展示封装中多颗 NAND 裸片与逻辑基底的连接。[1] 示意图不代表某款量产加速器的实际封装布局。*

OCP 0.7.0 版规格的表 2、表 3 给出了一组很直观的配置：16 个 host channel，每通道 64 bit、32 GT/s，按 75% 接口效率计算，有效接口带宽为 3072 GB/s；16 层裸片对应的用户容量为 512 GiB。[2] 按表 2 的单位复算：

$$
B_{\mathrm{interface}}
=32\times10^9\times\frac{64}{8}\times16\times0.75
=3.072\times10^{12}\ \mathrm{byte/s}.
$$

这是规格中的接口计算值，应用还要靠内部并行、地址分布和足够多的请求，把通道利用起来。SK hynix 的公告也把 HBF 放在 HBM 与 SSD 之间，并给出了约 0.4～3 TB/s 的三个带宽档位。[3]

先把它与 HBM 和 SSD 放在一张表里。下表中的 GB/s 都使用十进制单位；HBM、HBF 按单堆栈比较，SSD 按单块盘比较。厂商接口规格、SSD 顺序吞吐和论文假设分别注明，不能当成同一负载下的实测排名。

| 对象与统计单位 | 容量示例 | 读取／接口带宽 | 写入说明 |
| --- | --- | --- | --- |
| KIOXIA CM7-V，单块 PCIe 5.0 SSD | 3.2 TB | 顺序读最高 14 GB/s | 顺序写最高 6.75 GB/s，产品规格 [8] |
| Micron HBM3E，单堆栈 | 24／36 GB | 总接口带宽超过 1200 GB/s | 读写共享接口带宽 [9] |
| Micron HBM4，12-high 单堆栈 | 36 GB | 总接口带宽超过 2800 GB/s | 读写共享接口带宽 [10] |
| OCP HBF，最高档单堆栈 | 512 GiB | 有效接口带宽 3072 GB/s | 该数字不能当作 NAND 持续写带宽 [2] |
| 《Exploring HBF》建模的单堆栈 | 512 GB | 读 1600 GB/s | 写 16 GB/s，为论文假设 [5] |

按数字复算，1600／14 约为 114，3072／14 约为 219：HBF 的读取目标确实到了单块高端 SSD 的百倍量级。不过，写入不能套用这个倍率。《Exploring HBF》因为缺少公开写性能，假设 100 μs 的页编程延迟；其每堆栈 16 GB/s 写带宽只比这块 SSD 的 6.75 GB/s 高约 2.4 倍。论文同时为 HBM4 设置了 1600 GB/s 的读、写带宽：在这组模型里，HBF 与 HBM 读吞吐相同，写吞吐却相差 100 倍。[5] 这些读、写数字是分别进行读或写时的预算，不能相加成同时可用的双倍带宽。

这也回答了“权重不是都在显存里吗”。权重常驻 HBM 是常见部署方式；HBF 探索的是让大量权重常驻近计算 NAND，使用时把所需块读到计算侧缓冲，HBM 或 SRAM 留给更频繁读写的数据。具体设计可以直接从 HBF 供数，也可以把 HBF 当作 HBM 后面的容量层，软件并不总要先把整个模型完整装入 HBM。[5]

按上面的容量和接口带宽，理想扫描时间为

$$
t_{\mathrm{scan}}
=\frac{512\times2^{30}}{3.072\times10^{12}}
\approx0.179\ \mathrm{s}.
$$

这里把 GiB 与 GB/s 转成了相同的字节口径。它只反映理想容量／带宽比：512 GiB 能放进去，与每一步能否在目标时间内读完，是两个不同约束。

NAND 还带来读写不对称、有限的编程／擦除寿命，以及高并行工作时的散热问题。[5][6] 

于是，HBF 的使用方式开始分化：静态权重、反复读取的缓存，以及不断写入的短命数据，不能只因都占空间，就按同一种策略放置。

## 3 B200／B300：整颗 GPU 的带宽来自多个并行数据源

上一节比较的是单堆栈；B200、B300 的显存带宽则是整颗 GPU 的合计值。以下统一采用 NVIDIA HGX 企业参考架构表 1 的名义上限，避免把单 GPU 与整台八卡服务器混在一起。[11]

| 配置 | 单 GPU HBM3E 容量 | 单 GPU 显存带宽 | 八 GPU 节点的本地显存带宽合计 |
| --- | --- | --- | --- |
| B200 SXM，参考架构口径 | 180 GB | 最高 8 TB/s | 最高 64 TB/s |
| B300 SXM，参考架构口径 | 288 GB | 最高 8 TB/s | 最高 64 TB/s |

因此，按这份参考架构，B200 到 B300 的容量增加了 60%，显存带宽上限却仍是 8 TB/s。容量增长不自动带来带宽增长。具体 SKU 要以对应资料为准：Blackwell Ultra 数据表另列 HGX B300 为 270 GB、7.7 TB/s，不能与上表拼成一套规格。[12] 同样，Micron 单堆栈的带宽上限，也不能直接乘上堆栈数，替代 NVIDIA 对整颗 GPU 的规格。

显存聚合带宽来自多个存储堆栈和控制器并行供数。用八路 HBM、整 GPU 8 TB/s 做均分示意，每路平均约 1 TB/s；这是帮助理解的预算模型，不是每路实测值。八卡节点的 64 TB/s 又是八颗 GPU 各自本地带宽的合计，单颗 GPU 访问其他 GPU 的数据仍受互联约束，不能把它看成任意一个 kernel 都能使用的统一 64 TB/s 显存接口。

![并行 HBM、多堆栈 HBF，以及单 HBF 向快速缓冲供数时的带宽预算](/img/2026/10/03/hbf-bandwidth-budget.png)

*图：带宽预算与数据流示意，不复现实际封装布局。八路 HBM 按 8 TB/s 均分；单 HBF 采用 OCP 接口目标。四个 HBF 的 12.288 TB/s 是独立接口可并行时的算术上界，具体主机还须提供相应接收能力。[2][11]*

那么 HBF 能不能也放多个？可以。OCP 第 4.3 节、图 2 就展示了一个 host 连接四个 HBF chipset 的系统，每个容量为 512 GiB；《Exploring HBF》也建模了七个和八个 HBF 堆栈的配置。[2][5] 要区分“在一个堆栈里增加 NAND 裸片层数”和“增加独立堆栈及其接口”：前者主要扩容，后者才增加独立供数通路。

若每堆栈容量为 $C_{\mathrm{stack}}$，可持续读取带宽为 $\beta_{\mathrm{stack}}$，对 $n$ 个同规格独立堆栈，理想预算为

$$
C_n=nC_{\mathrm{stack}},\qquad
\beta_{n,\mathrm{ideal}}=n\beta_{\mathrm{stack}}.
$$

沿用 OCP 最高档的接口数值，一、四、八个 HBF 分别对应 512 GiB、2 TiB、4 TiB，以及 3.072、12.288、24.576 TB/s 的理想接口合计。这是扩展算术，并不是现有 GPU 的 HBF 配置。实际收益还要看 xPU 端口、控制器与内部数据通路、封装面积、供电散热，以及当前访问是否均衡分布到各堆栈。OCP 的四堆栈示例也没有规定主机最多只能接四个。

**HBF 的大容量允许系统用更少的堆栈满足容量需求，却不会让所需带宽也自动变少。** 例如只计算 500 GB 参数本体的存放空间：按上表两个 B300 的名义容量合计为 576 GB，本地显存带宽合计最高 16 TB/s；一个 512 GiB HBF 也放得下约 500 GB 权重，但接口目标是 3.072 TB/s。这里忽略 KV 和运行时空间，只比较容量及带宽预算。多 GPU 方案的额外带宽伴随额外硬件成本和通信开销，不能只凭这个合计值预测推理速度。

所以，“HBF 堆栈大，因此不容易叠加”不是准确的因果关系。容量更高给设计者减少堆栈数、减少加速器数的机会；若读取量很大，仍可能需要增加 HBF 堆栈和主机接口来补足带宽。对高密度介质，真正要优化的是单位容量带宽与系统成本的配比。

## 4 MoE 的机会：保存全部专家，每步只读一部分

权重在模型加载后通常保持不变，天然适合读多写少的介质。MoE 又把“需要保存的参数”和“每个 token 使用的参数”分开，让容量与单步读取量之间出现更大的差距。

但这条差距会随 batch 改变。单个 token 只激活少数专家，一个 batch 的 token 合起来，可能覆盖很多专家。为了看清这个趋势，先做一个理想化推导：一层有 $N$ 个等大小专家，每个 token 均匀选择 $k$ 个不同专家，各 token 的选择相互独立。

对于某一个专家，一个 token 没有选择它的概率为 $1-k/N$。若一个 decode step 中有 $B$ 个 token，这个专家一次也没被选中的概率就是 $(1-k/N)^B$。因此，该步被至少一个 token 使用的专家数，期望为

$$
\mathbb{E}[U_B]
=N\left[1-\left(1-\frac{k}{N}\right)^B\right].
$$

再假设同一个专家的权重在该步只需从目标内存读取一次，所有专家权重合计为 $W$，专家权重读取量的期望便是

$$
\mathbb{E}[D_{\mathrm{experts}}]
=W\left[1-\left(1-\frac{k}{N}\right)^B\right].
$$

这里只估算路由专家权重，没有计入共享专家、Attention、KV 读取、跨卡通信，也没有模拟缓存和实际 kernel 的搬运次数。

![均匀 MoE 路由下，batch 增大时专家工作集占全部专家的比例](/img/2026/10/03/hbf-moe-working-set.png)

*图：本文按上述简化模型绘制，$N=256$、$k=8$；曲线是期望值，不是某个模型的实测路由或硬件带宽。*

在这组配置下，batch 为 1 时只访问 3.1% 的专家；batch 为 8、32 时，期望覆盖率约为 22.4%、63.8%；batch 到 128 时，已经接近 98.3%。每步读取集合扩大了，但该步也会生成更多 token。在每专家权重每步只读一次的假设下，batch=128 的专家权重读取量按每个输出 token 分摊约为 $98.3\%/128\approx0.77\%$，反而低于 batch=1 的 3.1%。

因此，不能只看专家覆盖率趋近 100%，就断言大 batch 不适合 HBF。固定同一个模型、batch 和数据布局，设每步未被缓存命中的权重读取量为 $D_{\mathrm{miss}}(B)$，只考虑搬运带宽，其时间下界为

$$
t_{\mathrm{read,min}}(B)
=\frac{D_{\mathrm{miss}}(B)}{\beta_{\mathrm{delivered}}}.
$$

若 HBF 与 HBM 的实际可持续读带宽相同，搬运同样字节数的下界也相同。大 batch 的专家并集增长对两者都适用；更好的权重复用、更多独立读取，也可能让 HBF 更容易发挥带宽。

低并发、大总参数、小激活比例的 MoE 值得研究，是因为较少的 HBF 堆栈就可能同时满足“保存全部专家”和“按时读出少数专家”。大 batch 下则要重新核算聚合带宽和时延目标。区别来自堆栈数、持续供数能力与访问模式；低并发本身也可能不足以隐藏 NAND 延迟，需要预取提供并行性。普通 MoE 的专家位于 FFN，KV 容量还要按 Attention 结构和保留的 token 数计算，不会因为 batch 小而自动增加。

## 5 权重放在 HBF：哪些限制与 SSD offload 相似

权重常驻 HBF 可以有不同组织方式：运行时从 HBF 读取所需权重块，经过计算侧缓冲供给运算；也可以用 HBM 缓存热点权重，未命中时从 HBF 预取。[5] 后一种策略与 SSD 权重 offload 的共同问题，是大容量层要及时把需要的数据送到计算侧。HBF 换成近计算 UCIe 接口和更高并行度，数据放置与缓存策略仍然需要设计。

假设全部权重只放在一个 HBF，计算侧即使有多个 HBM 堆栈，HBM 的带宽也不会把这个 HBF 的源端吞吐放大。缓存命中的权重可由 HBM 快速供给；未命中的新权重仍受 HBF 源端、传输路径和请求调度限制。若一次计算需要从容量层新取 100 GB 数据，以 3.072 TB/s 理想带宽计算，仅搬运下界就是约 32.6 ms。SSD 按 14 GB/s 则约为 7.14 s。这是纯带宽算例，不含等待与软件开销，也不是两种设备的实测推理延迟。

预取可以把搬运与当前层计算重叠，但需要足够的提前量和缓冲；平均消费速率持续超过源端带宽时，缓冲迟早会耗尽。MoE 的专家选择还依赖路由结果，缓存命中率、提前预测以及热点专家的放置会影响实际读取量。也因此，上式应代入真正的 $D_{\mathrm{miss}}$，而不是机械地每步读取全部模型参数。

要利用多个 HBF 的带宽，可以把同一步所需的数据条带化或分片到不同堆栈，并发读取。仅把第 1 层放在堆栈 A、第 2 层放在堆栈 B，不保证当前层能同时用满 A、B 的带宽；层间依赖需要通过预取或流水安排来处理。这是系统设计中的工作集问题，而不是容量能否继续相加的问题。

HBF 因而有机会缓解 SSD offload 的供数限制，却不是把 SSD 替换掉就自动得到完整的推理收益。需要同时确定：权重从哪些堆栈读取，多少数据能命中快速缓存，主机接口能接收多少数据，以及目标 token 间隔允许暴露多少等待。

## 6 KV cache 的难点：热数据留在哪里，写流量去了哪里

KV cache 容量大，看起来也是 HBF 的合适对象。但“KV”这个名字下面，至少有两种截然不同的生命周期。

正在 decode 的请求，会在后续每一步读取需要的历史 KV，并追加新 token 的 KV。长上下文的活跃缓存可能产生很强的读取压力。另一类 KV 是为了未来的请求复用而保留的前缀或会话状态；它们离开活跃工作集后，什么时候再被使用，取决于请求和缓存策略。

如果把后者迁入容量层，系统支付一次保存成本，期待以后命中时省掉重算。对于始终没有再被命中的块，这次写入只消耗了带宽和寿命。更大的容量让系统能保留更多块，却也可能把更多低价值写入留在系统里。

![传统 host-staged KV 卸载与 GPU 直达 SSD 的数据路径](/img/2026/10/03/hbf-kv-offload-paths.jpg)

*图：原视频 16:20 展示的 SSD KV 卸载路径，对应研究 [6] 的图 3。下方路径减少了数据经过主机内存的中转；两种路径都仍需要决定哪些 KV 值得保存和读回。图中没有画 HBF 接口。*

两篇近期论文恰好从不同位置回答这个问题。[《Exploring High-Bandwidth Flash for Modern LLM Inference》](https://arxiv.org/abs/2608.13868)把 HBF 作为 GPU 的主要内存组件，分析把权重和 KV 放入其中的收益：更大的容量有机会减少所需 GPU 数、扩大 batch；但收益依赖接近 HBM 的持续读取带宽，KV 写入也造成性能和耐久压力。其 1.6 TB/s、1 μs page read 等参数属于建模假设。[5]

[《HBF Sucks?》](https://arxiv.org/abs/2608.11668v4)则保留 Mooncake 风格的 SSD KV 池策略，把容量层换成 HBF。在其 TokenSim 扩展、生产 trace 和 H100／B200 系统配置的仿真中，平均端到端延迟反而上升 2～5.5 倍。一个关键原因是：热复用留在近端，HBF 收到大量保存冷块的写入；与此同时，系统配置的近端容量和带宽也变少了。[6]

这两个结果的差别来自部署位置、访问模式和资源预算。前者问“更多主内存能否减少容量约束”，后者问“维持原有分层策略，换上更快的闪存是否划算”。它们都把问题指向了同一个地方：必须把读取收益、写入代价和保留下来的 HBM 一起计算。

对我来说，比较有价值的方向是先按数据行为分配位置。中间张量和每步依赖的热点，应优先获得低延迟、可反复写入的空间；静态权重、具有稳定复用价值的前缀，则值得探索 HBF。短命冷块是否落盘，要依据以后复用的概率和重算成本，而不是有空闲容量就全部接纳。

## 7 真正要评估的参数，是工作集与服务目标

在规划这种系统时，第一步可以先用现有模型配置算容量。下面的小脚本复算前面的 KV 示例和专家覆盖率，不依赖任何 HBF 硬件：

```python
def kv_bytes(layers, kv_heads, head_dim, bytes_per_element, lengths):
    return 2 * layers * kv_heads * head_dim * bytes_per_element * sum(lengths)

def expert_fraction(experts, top_k, batch):
    return 1 - (1 - top_k / experts) ** batch

gib = 2 ** 30
one_request = kv_bytes(32, 8, 128, 2, [32768])
print(f"KV per request: {one_request / gib:.1f} GiB")
print(f"KV for 32 requests: {32 * one_request / gib:.1f} GiB")
for batch in [1, 8, 32, 128]:
    print(f"batch={batch:3d}, expert coverage={expert_fraction(256, 8, batch):.1%}")
```

随后需要从服务 trace 中取得更难的几个量：活跃 KV 的峰值、前缀命中和复用间隔、每秒向容量层写入的字节数，以及缓存读回暴露在关键路径上的时间。只看总 KV 大小，无法判断它应该放在什么层。

例如，如果某项放置策略导致 HBF 持续承接 10 GB/s 的逻辑写入，一天就是 864 TB。这个算例没有计入写放大，也没有据此推断产品寿命；它说明评估耐久时，应输入写入速率与运行时长，而不是仅仅看到“一共有几 TB 容量”。更新更频繁的 KV 和模型加载一次后长期复用的权重，即使体积相同，成本也很不同。

比较 HBF 与 HBM 方案时，还要固定同一套 TTFT、token 间隔或端到端时延目标，再统计符合这些目标的有效产出。否则，大容量允许接入更多请求，却可能让排队或读取等待拉长，账面吞吐提高，用户体验反而变差。

用一个简化成本指标表达，就是

$$
\mathrm{cost/token}
=\frac{\text{系统每秒总成本}}
{\text{满足服务时延目标的有效输出 token/s}}.
$$

这里的成本口径需要前后一致，包含硬件摊销、电力和维护等项目。假设某方案每秒成本降低 20%，有效 token 吞吐却下降 30%，单位 token 成本会变成原来的 $0.8/0.7\approx1.14$ 倍。便宜的容量是否带来便宜的推理，最终会在这个分母里体现出来。

## 8 标准化让方案可落地，软件决定容量如何兑现

视频最后把视角拉回内存产业，用 Optane 的经历提醒读者关注软件、供应和生态。[1] 对 HBF，我更关注它是否能成为芯片设计者可以长期采用的接口，以及运行时是否能安排合适的数据进入这个层级。

Sandisk 与 SK hynix 在 2026 年 8 月通过 OCP 发布首份技术规格，Google、Tenstorrent 参与了标准化过程；公开内容覆盖接口、封装、可靠性及软件读写指南。[7] 这是芯片与软件团队协同设计的起点。量产方案的持续性能、成本与软件支持，还需要后续产品来回答。

我对 HBF 的期待，是它能减少那些为了凑容量而增加的 GPU，让权重和上下文有更合适的存放位置。要实现这一点，运行时需要识别实际工作集、利用读取并行性，并对写入和保留策略做选择。512 GiB 和 3 TB/s 给了很大的设计空间；哪些数据值得占用这个空间，才决定它最终能省下多少推理成本。

## 参考

[1] Asianometry，[*High Bandwidth Flash Is Coming*](https://www.youtube.com/watch?v=3nTpW52nioI)，2026-09-24。本文的阅读起点；结构与 SSD 卸载插图摘自视频 08:04、16:20。

[2] Chinnakrishnan Ballapuram、Dongsop Lee，[*High Bandwidth Flash (HBF™) High-Level Base Die Specification*, v0.7.0](https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf)，OCP，2026-08-03。第 4 章，尤其表 2、表 3；带宽计算采用表 2 的 GB/s 口径。

[3] SK hynix，[*SK hynix Unveils First HBF Standard Specifications with Sandisk, Presenting AI Memory Solutions at ‘FMS 2026’*](https://news.skhynix.com/en/hbf-at-fms-2026/)，2026-08-04。

[4] Joshua Ainslie 等，[*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245)，2023，arXiv:2305.13245。

[5] Dowon Son 等，[*Exploring High-Bandwidth Flash for Modern LLM Inference: Opportunities and Challenges*](https://arxiv.org/abs/2608.13868)，2026，arXiv:2608.13868。主要内存方案与带宽、写入耐久的分析。

[6] Zhuoran Li 等，[*HBF Sucks? A Full-Stack Characterization of High-Bandwidth Flash for KV-Centric LLM Serving*](https://arxiv.org/abs/2608.11668v4)，2026，arXiv:2608.11668，v4。基于 TokenSim 的 KV 分层系统、热与耐久模型分析。

[7] Sandisk，[*Sandisk and SK hynix Advance Global Standardization of High Bandwidth Flash with Release of First OCP Technical Specification*](https://www.sandisk.com/company/newsroom/press-releases/2026/2026-08-03-Sandisk-and-sk-hynix-advance-global-standardization-of-hbf)，2026-08-03。

[8] KIOXIA，[*CM7-V Series (2.5-inch)*](https://americas.kioxia.com/en-us/business/ssd/enterprise-ssd/cm7-v.html)，产品规格，访问日期 2026-10-06。本文采用 3.2 TB 型号的顺序读写数值。

[9] Micron，[*HBM3E*](https://www.micron.com/products/memory/hbm/hbm3e)，产品规格，访问日期 2026-10-06。单堆栈 24／36 GB、超过 1.2 TB/s。

[10] Micron，[*HBM4*](https://www.micron.com/products/memory/hbm/hbm4)，产品规格，访问日期 2026-10-06。12-high 单堆栈 36 GB、超过 2.8 TB/s。

[11] NVIDIA，[*Enterprise Reference Architecture — HGX AI Factory: Components*](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)，2026，表 1。本文 B200／B300 名义容量、单 GPU 与八 GPU 节点的带宽采用此表口径。

[12] NVIDIA，[*Blackwell Ultra Datasheet*](https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf)，技术规格，第 5 页，访问日期 2026-10-06。HGX B300 与 GB300 NVL72 的具体配置分别列示，不与参考架构上限混用。
