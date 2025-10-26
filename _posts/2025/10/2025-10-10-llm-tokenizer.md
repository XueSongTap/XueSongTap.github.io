---
layout: article
title: LLM的tokenizer
tags: tokenizer
---

tokenizer 有多种计算token，进行转换的方式


## 1. Byte Pair Encoding (BPE)

### 1.1 BPE 算法概述

BPE 最早是为数据压缩提出的算法, 其核心思想是：

- 统计字节（或字符）对的频率
- 把频率最高的一对替换成一个新的"符号"
- 重复这个过程，不断合并频繁出现的对，形成越来越大的"词典"


需要注意的是，BPE 拆分的子单词不一定都具有语义意义


### 1.2 BPE 词表构建流程

#### 1.2.1 初始化词表（Initialize vocabulary）

一开始，BPE 的词表只包含：

- 所有单个字符（例如英文字母 a-z，标点，空格等）
- 一个词尾标志（end-of-word symbol，通常用 `</w>` 或 `▁` 表示）

示例：
```
单词：low
初始 token 序列：l o w </w>
```


同时统计它们在语料库中的出现频率。例如：

```
l o w </w> : 5
l o w e r </w> : 2
n e w e s t </w> : 6
w i d e s t </w> : 3
```

---
#### 1.2.2 迭代构建词表（Loop until vocabulary reaches capacity）


词表的最终大小通常是预先设定的（比如 30k、50k）。在达到目标之前，BPE 不断执行以下步骤：


##### (1) 统计词语中连续 token 对的频率

例如对于 `l o w </w>`，连续 token 对有：

```
(l, o), (o, w), (w, </w>)
```

然后在整个语料里统计所有出现的 token 对频率，比如：

```
(o, w) 出现 7 次
(e, s) 出现 5 次
(s, t) 出现 4 次
...
```

---

##### (2) 选择出现频率最高的一对

比如频率最高的是 `(o, w)`，说明 "ow" 这个组合在语料里很常见

##### (3) 把这一对合并成一个新 token，并加入词表

把 `(o, w)` 合成新 token `ow`，更新词表：
```
[l], [o], [w], [</w>], [ow]
```

同时，语料中的所有 `(o, w)` 都替换为 `ow`：
```
l o w </w> → l ow </w>
```

然后进入下一轮统计，继续合并频率最高的 pair，比如 `(l, ow)` → `low`，依次进行

---
#### 1.2.3 输出最终词表（Output final vocabulary）

当词表大小达到预设容量，或者没有更多高频 pair 可以合并时，算法停止。

示例：
```
初始字符词表：{l, o, w, e, r, s, t, </w>}
最终词表可能是：{l, o, w, ow, low, e, r, s, t, est, new, ...}
```

### 1.3 BPE 的优势

- 处理未登录词（OOV）时，可以回退到更小的 subword 或字符，不会"崩溃"
- 常见词可以合并成更大的 token，减少序列长度，提高训练/推理效率
- 分词规则是从语料中自动学出来的，而不是人工定义



### 1.4 BPE 核心理念

用频率驱动的贪心合并策略，把语料从字符层压缩到高频子词层

参考：https://github.com/llmsystem/llmsys_code_examples/blob/main/tokenization/tokenization.ipynb

## 2 VOLT（Using Entropy to Learn Vocabulary）

### 2.1 VOLT 方法概述


VOLT 是一种近年来提出的、替代 BPE 的词表学习方法，其核心思想是：

不再仅仅依赖频率（像 BPE 那样贪心合并高频 token 对），而是引入**信息论中的熵（entropy）**来指导 subword 词表的构建，使得最终的分词更符合语言信息分布特性。

>It measures semantic-information-per-char


### 2.2 VOLT 的目标

>It measures semantic-information-per-char

VOLT 衡量的是 semantic-information-per-char（每字符的语义信息量），目标是找到一组 token（subword），使得这些 token 在整个语料中能最大化区分信息、最小化编码冗余。


## 3 MUV（Maximum Utility of Information for Vocabulary selection） 方法

### 3.1 VOLT 的局限性

VOLT 的问题在于：它把所有 token 评价标准都归结为"加入它能让语料整体熵下降多少"，但在实际任务（如翻译、语言建模）中，有些 token 不一定显著降低整体熵，却可能对任务预测能力非常有帮助（比如稀有实体名、人名、技术词汇等）。

### 3.2 MUV 方法概述

MUV（Vocabulary Selection with Maximum Utility of Information）不仅看 token 对编码熵的贡献，还要衡量它对任务效用（utility）的提升。其核心指标是 Marginal Utility of information for Vocabulary (MUV)。


## 4 LLM 的tokenizer 实践




### 4.1 语料去重与过滤（Corpus Deduplication and Filtering）

#### 4.1.1 LLaMA 3 的去重处理

LLaMA 3 采用了三级去重策略：

* **URL 级去重（URL-level deduplication）**
  按网页地址去重，去掉重复抓取的同一网页内容。

* **文档级去重（Document-level deduplication）**
  使用 **minHash** 算法，对文档内容进行近似哈希去重，识别并去除重复或高度相似的文档。

* **行级去重（Line-level deduplication）**
  对每 3000 万（30M）个文档，计算每一行的 **SHA-1（64 位）哈希值**，去除重复行。
  主要目的是清理掉网页上的模板内容（boilerplate），例如：



#### 4.1.2 过滤规则（Filtering）

- 单行内 n-gram 重复检查：如果同一行里出现大量重复的 n-gram（例如垃圾内容、灌水文本），则过滤掉
- "脏词"统计（dirty word counting）：检查文本中出现敏感或不适宜词汇的频率，超过阈值则剔除
- token 分布的 KL 散度（KL divergence）检测：如果某段文本的 token 分布与整个语料的平均分布差异过大（KL 散度过高），说明这段文本可能是异常或噪声，会被过滤掉




### 4.2 n-gram 详解

#### 4.2.1 基本定义
**n-gram** 是指在文本中连续出现的 **n 个 token（或字符）** 组成的序列。
例如，句子：

```
I like playing football
```

如果我们把每个单词视为一个 token：

* 1-gram（unigram）：
  `["I", "like", "playing", "football"]`

* 2-gram（bigram）：
  `["I like", "like playing", "playing football"]`

* 3-gram（trigram）：
  `["I like playing", "like playing football"]`

n-gram 的 n 可以是任意整数，常见的是 1～5 之间。

---

#### 4.2.2 在语料过滤中的用途

在清洗网页文本或训练语料时，常见低质量文本的一个特征就是：

* 同一行或同一段中，**n-gram 重复次数非常多**。
  例如：

  ```
  Buy now buy now buy now buy now click here click here
  ```

* 其中 2-gram “buy now” 出现了多次，这通常是垃圾广告或模板内容。

所以在 LLaMA 3 的过滤流程中，会检测一行中 n-gram 的重复频率。如果重复率超过阈值，就认为这行是噪声，直接过滤掉。

---

### 4.3 KL 散度（Kullback–Leibler Divergence）

#### 4.3.1 基本定义

**KL 散度**（Kullback–Leibler Divergence）是一种衡量两个概率分布差异的指标。
给定两个离散分布 ( P ) 和 ( Q )，KL 散度定义为：

$$
D_{\mathrm{KL}}(P | Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
$$

* ( P )：真实或参考分布
* ( Q )：被比较的分布（比如某段文本的 token 频率分布）
* 值越大，表示两者差异越大
* KL 散度是非对称的

#### 4.3.2 在语料过滤中的用途

在大规模语料中，大多数高质量文本的 **token 频率分布**（例如常见词 vs 罕见词的比例）与整个语料总体分布比较接近。

但一些异常文本，比如：

* 二进制垃圾
* 随机字符
* 非目标语言文本
* 爬虫错误页面

它们的 token 分布会和总体分布差异很大。

例如，如果整体语料主要是英文，但某一文档是阿拉伯文或乱码，那么它的 token 分布和主语料的分布会有很大的 KL 散度。

因此，LLaMA 3 的过滤规则中会：

* 计算每个文档或行的 token 分布；
* 与整个语料的参考分布做 KL 散度；
* 如果 KL 值超过阈值，就认为它是异常文本，直接剔除。


## 5 更多子词分词方法（More Subword Tokenization）

### 5.1 BBPE（Byte-level BPE）

* 全称：**Byte-level Byte Pair Encoding**
* 特点：以字节为单位进行 BPE 分词，对所有语言都通用（不依赖具体字符集）。
* 优点：不需要预先分词，能够处理任何语言和符号，包括罕见字符、表情符号等。

---

### 5.2 WordPiece

* 与 BPE 类似，也是基于子词合并的分词方法。
* 不同之处在于，它不是简单地合并**出现频率最高的 token 对**，而是选择能**最大化条件概率 p(b|a)** 的 a、b 组合来合并。
  换句话说，WordPiece 关注的是「在 a 后面出现 b 的可能性」，而不是单纯频率最高的 pair。

---

### 5.3 SentencePiece


* 对空格和标点符号使用统一的处理方式。
* 不依赖预先的分词工具，而是直接对原始句子操作：
  * 将空格 `' '` 替换成下划线符号 `▁`（Unicode U+2581）
  * 然后将句子拆成字符，再在字符层面上执行 BPE 或 Unigram 分词。
* 这样做的好处是：整个流程完全数据驱动，不依赖语言规则，可以用于多语言和低资源语言。

参考文献：
Kudo and Richardson, *SentencePiece*, 2018

## 6. 处理代码文本：预分词（Handling Code: Pre-tokenization）

### 6.1 预分词的必要性

在处理包含代码的语料时，常常需要在正式的子词分词（如 BPE、SentencePiece）之前，先进行 **预分词（Pre-tokenization）**，以便更好地保留代码结构。

### 6.2 使用正则表达式进行预分词

- 利用正则表达式对代码序列进行分割，可以更准确地识别代码中的有意义片段
- 例如，在 Python 代码中出现的 `.append`，如果直接用 BPE 分词，可能会被切成 `.`、`app`、`end` 等多个子词，造成语义和结构的破坏
- 通过正则预分词，可以将 `.append` 整体识别为一个单独的 token：`.append → [.append]`

### 6.3 预分词的优势

- 更好地保留函数名、属性名等语义单元
- 减少无意义的切分，提升模型对代码结构的理解能力


## 7、处理数字：xVal 方法


### 7.1 问题背景

传统的分词方式会把数字分解成多个 token，导致：
- 数字被切碎，难以进行数学运算
- 数字之间的关系难以捕捉（例如 7.4 和 7.5 的接近性）



### 7.2 xVal 方法（xVal: A Continuous Numerical Tokenization）

来源：Golkar et al., 2023

#### 7.2.1 编码（Encoding）


示例文本：`5 trials at pH 7.4`


**xVal 的方法**是：

* 使用特殊标记 `[NUM]` 来替换文本中的数字；
* 另起一个数值通道 `x_num` 来存储真实的数值（如 `5` 和 `7.4`）；
* token 序列中 `[NUM]` 会映射到一个通用的数字 token embedding；
* embedding 中再附加数字的连续值（如 5.0 或 7.4）作为额外维度。

这样，模型的输入就同时包含：

1. 文本 token 序列（`x_text`）；
2. 对应的数字值向量（`x_num`）；
3. 二者结合后的整体 embedding（`h_emb`）。

#### 7.2.2 解码（Decoding）


* 模型最后一层分成两部分输出：

  1. **Token Head**：生成下一个 token（如文本、标点、特殊符号等）。
  2. **Number Head**：预测具体的数值（例如从 7.4 预测到 7.5）。

例如：

示例：`5 trials at pH 7.4 → 5 trials at pH 7.5`


这里 `[NUM]` 在解码时对应的数值由 Number Head 输出为 7.5，实现了数字的连续建模和可计算性。


#### 7.2.3 与传统编码的比较

传统的 text-based 分词（如 BPE）对数字使用纯文本拆分，例如：

```
+ 500 e-2
```

会被分成多个 token `[82], [9283], [402], ...`
这种方式不能体现数字的连续性，也不便于进行数学推理。

---

### 7.3 xVal 方法的核心优势

- 把数字从离散的文本 token 中剥离出来
- 使用一个单独的通道对数字进行连续表示
- 在模型输出端增加一个 Number Head，支持对数值的回归预测
- 能显著提升大模型在科学、工程、金融等场景中处理数字和执行数学运算的能力


## 8 LLaMA 3 的多语言词表

### 8.1 词表规模扩大

- 从 LLaMA 2 的 32k tokens 扩展到 LLaMA 3.1 的 128k tokens

### 8.2 词表来源

- 其中 100k 个 token 来自 OpenAI 的 tiktoken（原始 200k 中筛选而来）
- 另外 28k 个 token 专门分配给多语言支持部分，用于覆盖非英语语言的字符和词汇


### 8.3 改进意义

相比 LLaMA 2，LLaMA 3 的词表在设计上：
- 不再只局限于英语，增强了多语言处理能力
- 通过引入更多的子词 token，可以更高效地表示不同语言的文本，减少切分碎片，提升模型在多语言任务中的表现

参考：https://www.icodeformybhasa.com/p/exploring-multilingual-aspects-and


## 9 如何构建多语言词表


### 9.1 多语言词表构建流程
LLaMA 3 使用了系统化的步骤来为多语言模型构建共享的子词词表，核心流程如下：

#### 9.1.1 合并多语言语料（Combining multilingual documents）

- 从 176 种语言中收集文档，占整个训练语料的约 8%
- 将这些不同语言的语料合并成一个联合语料库（joint corpus）
- 在这个联合语料上应用 BPE 算法，以便学习覆盖多语言的初始子词片段

#### 9.1.2 对每种语言分别应用 BPE（Per-language BPE）

- 对每种语言单独执行 BPE 分词
- 为每种语言生成相同数量的 BPE token（例如每种语言 N 个子词）
- 这样做是为了避免高资源语言（如英语）主导整个词表，保证低资源语言也有足够的 token 表达能力

#### 9.1.3 合并不同语言的 token 集（Merge token sets）

- 将各语言的 BPE token 集合合并成一个统一的多语言词表
- 在合并时，需要控制不同语言之间的 token 数量分配，避免某些语言过多占用词表容量

#### 9.1.4 通过 ALP 平衡词表容量

使用平均对数概率（Average Log Probability, ALP）来动态分配每种语言的词表容量：
- ALP 衡量的是模型在不同语言上对 token 的平均预测能力
- 通过平衡 ALP，可以让词表容量的分配既考虑语言规模，也考虑建模难度
- 这种方法来自 Zheng et al. (EMNLP 2021)，并在 Liang et al. (EMNLP 2023, XLM-V) 中进一步扩展用于跨语言大模型


### 9.2 参考文献

* Zheng et al. **Allocating large vocabulary capacity for crosslingual language model pre-training.** EMNLP 2021.
* Liang et al. **XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models.** EMNLP 2023.


### 9.3 在线演示 Demo
- https://belladoreai.github.io/llama-tokenizer-js/example-demo/build/
- https://koala.sh/tools/free-gpt-tokenizer



## 10 词表共享及其对多语言性能的影响

### 10.1 词表共享（Vocabulary Sharing）


#### 10.1.1 多语言拼写相似性示例

以"电视"一词为例：



| 语言   | 词汇         |
| ---- | ---------- |
| 英语   | television |
| 西班牙语 | televisión |
| 法语   | television |
| 意大利语 | television |
| 荷兰语  | televisie  |
| 葡萄牙语 | televisão  |
| 瑞典语  | television |
| 芬兰语  | televisio  |

这些语言中，“电视”一词的拼写非常相似。这种相似性使得在使用 **子词分词（如 BPE 或 SentencePiece）** 时，这些词在多个语言中会**共享相同或部分相同的 subword token**。
例如：

* `tele`、`vision`、`televis` 这样的子词会在多种语言中出现。
* 这意味着多个语言会使用相同的 embedding 参数表示相似的片段，实现了 **跨语言的表示共享**。

---

### 10.2 嵌入微调实验（Embedding Finetuning for LLM）

#### 10.2.1 实验设计

论文 **Yuan et al., ACL** 的研究展示了词表共享如何帮助多语言泛化

1. 构造小规模双语微调数据集
   - 使用 1 万条双语平行数据构建一个小型指令微调（instruction-finetuning）数据集
   - 平行数据指的是成对的翻译句子，例如英语-西班牙语

2. 微调 LLaMA-7B
   - 使用上述小数据集对 LLaMA-7B 模型进行微调，仅调整 embedding 与模型参数

3. 评估翻译性能
   - 监督的双语方向（bilingual）：即模型在微调时见过的语言对，例如英语-西班牙语
   - 其他所有语言方向（multilingual）：即模型没有直接微调过的语言对，例如西班牙语-法语、意大利语-葡萄牙语等

#### 10.2.2 实验结果与结论
- 模型在监督的双语方向上（如英-西）翻译性能显著提升
- 更重要的是：**模型在未见过的其他语言对上（multilingual）也出现了性能提升**
  - 例如，通过对英语-西班牙语进行微调，法语-意大利语的翻译质量也得到了提升

这说明：
- 由于这些语言共享大量子词（如 `tele`, `vision`），微调时的 embedding 参数更新能够迁移到其他语言
- 这种共享词表的机制自然地促进了多语言能力的扩展

参考文献：Yuan et al. How Vocabulary Sharing Facilitates Multilingualism in LLaMA. ACL


### 10.3 语言类型对微调效果的影响

embedding 微调的影响是语言依赖的，和以下因素密切相关：

- 与英语及其他语言的词表共享程度
- 语言类型（印欧 vs 非印欧）
- 资源丰富度
- 拼写和形态结构的相似性

不同类型语言的特征：

- Reciprocal 语言（多数欧洲语言）：能实现双赢，是构建多语言系统时的"核心语言"
- Altruistic 语言（如中文、韩语）：对多语言有帮助，但自己可能不增反降
- Stagnant 语言：需要专门的机制（如扩展词表）来提升效果
- Selfish 语言：微调可能导致负迁移，需要谨慎处理



## 11 Stagnant Quadrant —— 过度切分问题（Over-tokenization）


### 11.1 Stagnant Quadrant 问题
#### 11.1.1 问题来源

对于一些语言（尤其是非印欧语系、使用特殊字符的语言），Byte-level BPE 在分词时会将一个单独的字符切成多个 token，造成过度切分（over-tokenization）。


#### 11.1.2 示例：汉字"饕"

- 拼音：[tāo]，意思是"贪吃的"
- Unicode 码点：U+9955
- UTF-8 编码：0xE9 0xA5 0x95
- 字节序列：[233, 165, 149]
- 在 BBPE 下的 token 序列为：[227, 234, 260]

这导致：
- 一个字符被切成了 3 个 byte-level token
- 序列长度比字符数还长
- 模型输入变长，训练与推理效率下降
- 对该语言的表示质量降低，embedding 学不到紧凑的语义单元
- 导致这类语言容易落入 Stagnant Quadrant（双语、多语言性能都不提升）



### 11.2 改进方向

#### 11.2.1 缩短 token 序列

移除所有 token 的共同前缀（例如 227），将其合并，以减少 token 数量。

在上例中，如果多个字符都以 `227` 作为前缀，那么可以把 `227` 抽掉或合并成一个标记，从而让"饕"这个字只对应更少的 token（理想是一个）。

#### 11.2.2 改进效果

- 词表可以更高效地表示非拉丁文字
- 序列长度缩短
- 多语言表示能力提升
- 对 Stagnant Quadrant 语言的支持会更好
