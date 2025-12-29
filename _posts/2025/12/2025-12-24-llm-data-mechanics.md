---
layout: article
title: LLM 数据的过滤、去重
tags: LLM
---




**本讲：深入 mechanics**
- 过滤（filtering）的算法工具：如何从海量原始数据里找“像目标数据”的子集
- 过滤的应用：语言识别、质量过滤、毒性过滤
- 去重（deduplication）：从精确重复到近似重复（Bloom filter、MinHash、LSH）

  

## 1. 过滤（Filtering）总览：Target vs Raw

**基本问题（算法模块）**
- 给定小但高质量的“目标数据” `T` 与海量“原始数据” `R`
- 目标：在 `R` 中找到一个子集 `T'`，使其“更像” `T`（但不要求与 `T` 完全相同）

**对过滤算法的要求（Desiderata）**
- 能从 `T` 泛化：希望 `T` 与筛出的 `T'` 不同，但风格/分布相近
- 极快：必须在巨大 `R` 上跑，吞吐量是第一约束

**参考**
- 数据选择（data selection）综述：`https://arxiv.org/abs/2402.16827`



## 2. 过滤算法（Algorithmic Tools）

### 2.1 KenLM：Kneser–Ney 平滑的 n-gram 语言模型

**定位**
- KenLM：最初服务机器翻译的超快 n-gram LM 实现，常用于数据过滤
- 核心优点：极其简单（计数+归一化），因此非常快；缺点：表达力粗糙

**n-gram 最大似然估计（MLE）**
- 以 trigram 为例：  
$$
p(\text{in}\mid \text{the cat})=\frac{\text{count(the cat in)}}{\text{count(the cat)}}
$$
- 问题：稀疏性（n 变大后，绝大多数 n-gram 计数为 0）
- 解决：Kneser–Ney smoothing（未见过的 n-gram 也能有合理概率；例如让 $p(\text{in}\mid \text{the cat})$ 也依赖 $p(\text{in}\mid \text{cat})$）

**用 LM 打分：log-prob 与困惑度（perplexity）**
- 对文本计算 log 概率 `score = log p(text)`
- 用困惑度归一化长度，避免偏好短文本：  
$$
\text{perplexity}=\exp\left(-\frac{\text{score}}{\#\text{tokens}}\right)
$$
- 直觉：越“像训练语料/语言”、越“流畅”的文本，困惑度越低

**CCNet（CommonCrawl 清洗的一种经典做法）**
- 粒度：段落（paragraph）
- 按困惑度从小到大排序
- 保留最好的 1/3（困惑度最低的部分）
- 被用于 LLaMA 的数据处理链条之一

**小结**
- KenLM（Kneser–Ney n-gram）是“快但粗”的过滤工具：适合做大规模第一道筛



### 2.2 fastText：超快文本分类器（Bag of embeddings + Hashing trick）

**定位**
- 任务：文本分类（情感、语言、质量、毒性等）
- 目标：在速度很快的前提下，效果接近更慢的神经网络分类器

**对比：bag-of-words vs fastText**
- bag-of-words（基线）：直接学习词→类别的嵌入/权重，参数量约为 $V\times K$（词表大时很重）
- fastText：先做词嵌入（维度 $H$），再接一个线性分类头：参数量约为 $H\times (V+K)$

**工程要点**
- 并行、异步 SGD
- 学习率线性衰减到 0（实现里做线性插值）

**Bag of n-grams + hashing trick（关键技巧）**
- 为了捕获局部顺序信息，使用 n-gram（如 bigram：`"the cat"`, `"cat in"`…）
- 问题：n-gram 空间可能非常大甚至无界
- 解决：hashing trick，把 n-gram 映射到固定数量的桶（实践中可用上千万 bins）；实现里用 MurmurHash（`mmh3`）

**二分类时的简化**
- 质量过滤经常是 `K=2`（good vs bad）
- 此时 fastText 本质上就是一个线性分类器（概念上更容易理解与部署）



### 2.3 DSIR：用重要性重采样做数据选择（Importance Resampling）

**论文：Data Selection for Language Models via Importance Resampling (DSIR)**
- 核心：用“分布匹配”的方式选数据，比纯粹的“好/坏分类”更原则化（principled）

**重要性采样/重采样（importance sampling）复习**
- 有目标分布 $p$（想从这里采样），但只能从提议分布 $q$ 采样
- 对从 $q$ 采到的样本 $x$，赋权重：  
$$
w(x)\propto \frac{p(x)}{q(x)}
$$
- 再按归一化后的权重进行重采样，样本分布会更接近 $p$

**DSIR 的数据选择设定**
- 目标数据集：$D_p$（小但想要的风格/领域）
- 原始/提议数据集：$D_q$（大但混杂）

**“Take 1” 直觉方案**
- 分别拟合 $p$ 与 $q$，再对 $D_q$ 做重要性重采样
- 难点：$D_p$ 太小，直接拟合高质量模型很难

**“Take 2” 实用方案：hashed n-grams（降低建模难度）**
- 用 hashing trick 把 n-gram（示例里从 unigram 开始）映射到有限 bins
- 在 hashed 空间里学习一个非常简单的 n-gram（unigram）模型
- 用 $\frac{p_T(x)}{p_R(x)}$ 做打分并重采样（对应“更像目标分布、且在原始里不常见”的内容会被放大）

**结果与对比**
- 论文报告：DSIR 在 GLUE 上略优于启发式分类（如 fastText）
- 与 fastText 的比较：  
  - DSIR：建模分布，更能刻画“多样性/覆盖面”，更原则化  
  - fastText：直接判别式分类，简单好用  
  - 复杂度上相近；二者都能通过更好的建模进一步提升



### 2.4 过滤的一般框架（把三种方法放在同一个模板里）

**统一视角**
- 输入：目标 `T` 与原始 `R`
- 步骤：
  1) 基于 `T`/`R` 训练某种模型，得到打分函数 `score(x)`
  2) 按 `score(x)` 保留/抽样 `R` 中样本得到 `T'`

**三种实例**
- KenLM（生成式，拟合目标分布）：`score(x)=p_T(x)`，阈值过滤/随机保留
- fastText（判别式分类）：`score(x)=p(T|x)`，阈值过滤/随机保留
- DSIR（重要性重采样）：`score(x)=p_T(x)/p_R(x)`，按比例重采样



## 3. 过滤的应用（Applications）

### 3.1 语言识别（Language identification）

**目标**
- 从混杂文本中筛出特定语言（例如英语）

**为什么不直接全做多语？（课程给出的现实考量）**
- 数据：高质量多语清洗/处理很难在每种语言都做到位
- 计算：计算预算固定时，更多语言会分走 token/compute
- 模型差异：  
  - BLOOM 的英语仅约 30%，英语表现受影响（undertrained）  
  - 许多前沿模型（GPT-4/Claude/Gemini/Llama/Qwen）仍然高度多语，但需要足够训练量支撑

**fastText 语言识别**
- 现成的 off-the-shelf 分类器，支持 176 种语言
- 训练来源：Wikipedia、Tatoeba（翻译语料）、SETimes（新闻）
- 例：Dolma 保留满足 $p(\text{English})\ge 0.5$ 的页面

**常见坑（Caveats）**
- 短文本难
- 低资源语言难
- 可能误伤英语方言/变体
- 相近语言易混（如 Malay vs Indonesian）
- code-switching（混合语言）本身就难以定义“属于哪种语言”

**案例：OpenMathText（从 CommonCrawl 挖数学文本）**
- 规则过滤（如包含 LaTeX 命令）
- KenLM（在 ProofPile 上训练）过滤：困惑度 < 15000
- fastText 数学写作分类器：阈值 0.17（判为数学）；阈值 0.8（判为非数学）
- 产出：14.7B tokens；训练 1.4B 模型，优于用 20× 数据训练的对照



### 3.2 质量过滤（Quality filtering）

**两派做法**
- 不用“模型过滤”（更偏规则/启发式）：C4、Gopher、RefinedWeb、FineWeb、Dolma
- 用“模型过滤”（越来越常见）：GPT-3、LLaMA、DCLM

**GPT-3 的质量过滤（思路）**
- 正样本（positives）：Wikipedia、WebText2、Books1、Books2
- 负样本（negatives）：CommonCrawl
- 用分类器打分，把 CommonCrawl 中“更像正样本”的内容筛出来
- 课件补充：后续实践中也会混入合成数据（如 GPT-3.5/后来 GPT-4 生成）与过滤后的数据

**一个以代码数据为例的模型过滤流程（课件示例）**
- `R`：The Stack 的 Python 子集（原始）
- 用 GPT-4 按提示词标注 `R` 的 100K 子集得到 `T`（正例：更有教育价值）
- 用预训练 codegen 模型的 embedding 做特征，训练随机森林分类器
- 用分类器从 `R` 中挑选被判为正的样本
- HumanEval 结果（训练 1.3B LM）：  
  - 直接用 Python 子集：96K steps 后 12.19%  
  - 用过滤后的子集：36K steps 后 17.68%（更好且更省步数）



### 3.3 毒性过滤（Toxicity filtering）

**案例：Dolma 的毒性过滤**
- 数据集：Jigsaw Toxic Comments（2018，Wikipedia talk page 评论）
- 标注维度：toxic / severe_toxic / obscene / threat / insult / identity_hate
- 训练 2 个 fastText 分类器：  
  - `hate`：正例={unlabeled, obscene}；负例=其他  
  - `NSFW`：正例={obscene}；负例=其他



## 4. 去重（Deduplication）：从精确到近似

### 4.1 为什么要去重？重复的两种形态

**两类重复**
- 精确重复（exact duplicates）：镜像站、GitHub forks 等
- 近似重复（near duplicates）：只差少量 token/格式的文本

**近似重复常见来源**
- 服务条款、许可证等模板化内容（如 MIT license）
- 模板/表格化写作（复制粘贴、模板生成）
- 轻微格式差异导致的“几乎一样”
- 课件例子：C4 中某条商品描述被重复了 61,036 次（高度模板化/拷贝导致）

**去重的收益**
- 训练更高效（token 更少）
- 降低记忆化（缓解版权/隐私风险）
  - 相关工作：`https://arxiv.org/pdf/2107.06499`（指出去重可提升 LM 训练效果）

**设计空间（做去重要先定清楚）**
1) item 粒度：句子/段落/文档？
2) 匹配标准：精确匹配？共享子片段？共享比例（如 Jaccard）？
3) 动作策略：全删？只保留一个？

**关键挑战**
- 本质是“把 item 与其它 item 比较”
- 必须用近线性/线性时间算法才能规模化



### 4.2 Hash：去重的基础工具

**哈希函数**
- $h$：把 item 映射到更短的 hash 值
- 碰撞（collision）：$h(x)=h(y)$ 但 $x\ne y$

**速度 vs 抗碰撞的权衡**
- 加密哈希（如 SHA-256）：更抗碰撞，但慢（例如比特币）
- 非加密哈希（DJB2/MurmurHash/CityHash）：快但不抗恶意碰撞（适合数据工程/哈希表）

课件采用：MurmurHash（`mmh3`）



### 4.3 精确去重（Exact deduplication）

**简单范式（MapReduce 思路）**
- 对每个 item 计算 hash
- 按 hash 分组（同 hash 的视为候选重复）
- 每组保留一个（其余删掉）

**优缺点**
- 优：语义清晰、精确度高、易并行可扩展
- 缺：无法处理“近似重复”

**C4 的精确去重设定**
- item：3 句跨度（3-sentence spans）
- 匹配：精确匹配
- 动作：删掉重复，只保留一个
- 风险提示：如果从文档中间删掉 3 句片段，文档可能变得不连贯



### 4.4 Bloom Filter：近似集合成员查询（用于大规模精确去重/查重）

**目标**
- 以极低内存实现“是否在集合里？”的近似查询

**性质**
- 省内存
- 可插入（update），不可删除（delete）
- 查询返回 “no”：一定不在集合
- 查询返回 “yes”：很可能在，但存在假阳性（false positive）
- 通过更多计算/更多哈希函数可把假阳性概率指数级压低

**直观构造**
- 用长度为 $m$ 的 bitarray 作为表 `B[0..m-1]`
- 插入：把 item 用 $k$ 个哈希函数映射到 $k$ 个桶，把对应位都置 1
- 查询：若 $k$ 个位置都为 1，则返回 “可能存在”

**从“加内存”到“加哈希函数”的对比（课件直觉）**
- 仅增大 $m$：假阳性率大致按 $O(1/m)$ 下降（随内存多项式变好）
- 增大 $k$：能把假阳性率压得更快（更接近指数级下降，但需要更多计算）

**假阳性率分析（假设独立）**
- 插入 $n$ 个 item、$k$ 个哈希函数、表大小 $m$
- 某个特定位为 1 的概率约为：  
$$
1 - \left(1-\frac{1}{m}\right)^{kn}
$$
- 对不在集合的查询项，其 $k$ 个位置都为 1 的概率（假阳性率）：
$$
f=\left(1 - \left(1-\frac{1}{m}\right)^{kn}\right)^k
$$
- 给定 $m/n$ 时的最优 $k$：  
$$
k^*=\ln 2\cdot \frac{m}{n}\quad (\text{此时 } f\approx 0.5^k)
$$

**例：Dolma 的设置（课件给出的量级）**
- 目标假阳性率：$10^{-15}$
- item 粒度：段落（paragraph）



### 4.5 Jaccard 相似度：定义“近似重复”

**Jaccard(A,B)**
$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

**近似重复的定义**
- 若 $J(A,B)\ge \tau$（阈值），则 A 与 B 是 near-duplicates

**挑战**
- 想在线性时间里找出所有 near-duplicate 对



### 4.6 MinHash：让“碰撞概率 = 相似度”

**核心性质**
- 取一个随机哈希函数（等价于对“元素集合”做随机排列），定义：
  - $h(S)=\min\limits_{x\in S}\text{hash}(x)$
- 则有：
$$
\Pr[h(A)=h(B)]=J(A,B)
$$

**直觉解释（课件的排列视角）**
- 随机排列后，看哪个元素最先出现（min）
- 如果最先出现的是 $A\cap B$ 里的元素，则两者 min 一样；否则不一样

**实践**
- 用很多个随机种子（很多个哈希函数）来估计 Jaccard
- 但：一次碰撞只能说明“相似度可能高”，并不能直接判断是否超过阈值



### 4.7 LSH（Locality Sensitive Hashing）：把“可能高相似”变成“有阈值的碰撞”

**问题**
- 单个 MinHash：$P[\text{collide}]=J(A,B)$ 很“软”，噪声大

**Banding 技巧（AND-OR 结构）**
- 使用 $n=b\times r$ 个哈希函数
- 划分为 $b$ 个 band，每个 band 有 $r$ 个哈希值
- 判定：只要**存在某个 band** 的 **r 个值全相同**，就认为碰撞

**碰撞概率（给定相似度 sim=J(A,B)）**
- 一个 band 全匹配概率：$\text{sim}^r$
- 至少一个 band 匹配概率：
$$
P[\text{collision}] = 1-(1-\text{sim}^r)^b
$$

**数值感受（课件示例）**
- 例如 `sim=0.8`、`b=5`、`r=10` 时，可用上式计算碰撞概率（相似度越高、越容易碰撞）

**参数直觉**
- 增大 $r$：阈值更“尖锐”，曲线右移（更严格、更难碰撞）
- 增大 $b$：曲线左移（更宽松、更容易碰撞）

**课件举例（来自去重论文的典型设置）**
- $n=9000,\; b=20,\; r=450$
- 相变阈值（phase transition）约为：
$$
\tau \approx \left(\frac{1}{b}\right)^{1/r}
$$
- 当 sim≈阈值附近时，碰撞概率会从很小快速跃迁到接近 1（“阈值化”的效果）



## 5. 本讲总结（Takeaways）

- 过滤算法工具：KenLM（n-gram）、fastText（快速分类）、DSIR（重要性重采样）
- 过滤应用：语言识别、质量过滤、毒性过滤；同一套 machinery 可复用在不同目标上
- 去重：从精确去重到近似去重；用哈希把“全量两两比较”变成可扩展的近线性流程
- 课程主旨：mechanics（工具）学会后，还要花时间与数据相处，建立直觉与判断标准
