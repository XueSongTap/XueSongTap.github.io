---
layout: article
title: LLM 数据的生产
tags: LLM
---



数据不是凭空出现的，数据管线（收集→转换→过滤→去重→聚合）往往是决定模型能力与差异化的关键。

## 1. 核心观点：为什么数据最重要？

- Hot take：训练语言模型时，**最重要的是把数据搞对**。
- 一个理由：看公司披露的信息。
  - 开源权重模型（例如 Llama 3）在架构、训练流程上可能比较透明，但对**训练数据**几乎不公开。
- 数据保密的两个主要原因：
  1. 竞争优势（数据是壁垒）
  2. 版权风险（数据来源可能涉及侵权）
- 历史对比：
  - 基础模型之前：数据工作更多是“重标注”的监督学习。
  - 现在：标注减少，但“**筛选、清洗、策划、去重**”的工作量依然巨大。
- 数据是一个典型的**长尾问题**：大量边角情况需要人工与规则打磨；它不像架构/系统那样容易规模化“自动变好”

## 2. 训练阶段与术语

### 2.1 训练阶段（现实中边界会模糊）

1. **Pre-training（预训练）**：用大量原始文本训练（如网页文档）。
2. **Mid-training（中期训练）**：继续用更高质量/更针对的数据训练，以增强特定能力。
3. **Post-training（后训练）**：面向指令跟随/对话做微调（或 RLHF 等）。

总体趋势可以理解为：**从“大量低质量”逐步过渡到“少量高质量”**（但阶段之间不总是清晰分割，也可能有更多中间阶段）

### 2.2 术语

- **Base model**：完成预训练 + 中期训练后的模型。
- **Instruct / Chat model**：完成后训练（指令/对话/对齐）后的模型。

### 2.3 一个全流程例子：OLMo（AI2）

- OLMo 2：`https://allenai.org/olmo`
- 1) Pretraining：  
  ![OLMo2 Pretraining](olmo2-pretraining.png)
- 2) Mid-training：  
  ![OLMo2 Mid-training](olmo2-dolmino.png)
- 3) Post-training：`https://arxiv.org/pdf/2411.15124`  
  ![Tulu / Post-training](tulu.png)


## 3. 框架：把“数据”当成可工程化对象（补充框架）

> 这一部分在代码里作为 `framework()` 给出：用来帮助你在脑中组织“数据管线”和“能力目标”。

### 3.1 数据对象类型（data objects）

- **Live service**：在线服务产生的数据（例：Reddit）。
- **Raw snapshot**：通过爬虫/API/公开 dump 获得的原始快照。
- **Processed text**：经过转换、过滤、去重等处理后的文本。
- **Aggregated datasets**：多源聚合后的大数据集（例：Dolma、The Pile）。

### 3.2 数据来源类型（sources）

- 标注者（例：Llama 2 的部分指令数据）
- 真实用户（例：ShareGPT）
- 策划/爬取（例：从 Common Crawl 清洗）
- 从更强模型蒸馏（例：GPT-4 生成的合成数据）
- 自蒸馏（用正在训练的模型生成合成数据）

### 3.3 想通过数据增强的能力（capabilities）

- 任务能力（例：信息抽取）
- 指令跟随与对话
- 长上下文（4096 → 100,000）
- Infilling（例：the cat __ the hat）
- 领域能力（代码/数学/医疗等）
- 安全（拒答/对齐）
- 推理（chain-of-thought 等）

预训练数据集纵览（BERT→GPT-2→CCNet→C4→GPT-3→The Pile→Gopher→LLaMA→RefinedWeb/FineWeb→Dolma→DCLM→Nemotron-CC）见下篇：[预训练数据集巡礼](https://xuesongtap.github.io/2025/12/22/llm-pretrain-datasets.html)

## 5. 法律与伦理：版权（Copyright）

- 生成式 AI 相关诉讼很多，尤其围绕版权：`https://www.bakerlaw.com/services/artificial-intelligence-ai/case-tracker-artificial-intelligence-copyrights-and-class-actions/`

### 5.1 知识产权法（Intellectual property law）

- 目标：**激励**知识产品的创造
- 类型：版权、专利、商标、商业秘密

### 5.2 版权法（Copyright law）基础

- 历史：
  - 1709 英国《Statute of Anne》：`https://en.wikipedia.org/wiki/Statute_of_Anne`
  - 美国现行主要框架：1976 年版权法：`https://en.wikipedia.org/wiki/Copyright_Act_of_1976`
- 保护对象（课程中引用的定义）：
  - 对“固定在任何有形媒介上的原创作品”的保护（能被感知、复制、传播）
- 关键原则：
  - 原创作品才受保护；纯“集合”一般不受保护（如电话簿），除非选择/编排有创意
  - 版权保护的是**表达（expression）**，不保护**思想/想法（ideas）**（例：quicksort 的思想可复现，但不能直接抄具体代码表达）
  - 从 1909 年“published”扩展到 1976 年“fixed”
  - 不需要注册也自动受保护（对比专利）
  - 版权门槛很低（你的网站也可能自动受版权保护）
- 诉讼与期限：
  - 起诉前通常需要注册
  - 注册费用 $65：`https://www.copyright.gov/about/fees.html`
  - 持续 75 年，之后进入 public domain（如莎士比亚、贝多芬，以及 Project Gutenberg 多数书）
- 小结：互联网上**大多数内容其实都有版权**。

### 5.3 使用受版权保护作品的两条路

1. 获得许可（license）
2. 适用合理使用（fair use）

#### 5.3.1 Licenses（许可）

- 许可（合同法概念）：许可方给被许可方授权
- 可理解为：“许可就是承诺不告你”

Creative Commons（CC）许可：

- 允许更自由地分发与使用作品
- 例：Wikipedia、Open Courseware、Khan Academy、Free Music Archive、Flickr 上 3.07 亿图片、MusicBrainz 3900 万图片、YouTube 1000 万视频等
- 2001 年由 Lessig 和 Eldred 创建，试图在 public domain 与传统版权之间搭桥

现实做法：很多模型开发者会**直接购买/签订数据授权**来训练基础模型，例如：

- Google ↔ Reddit：`https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/`
- OpenAI ↔ Shutterstock：`https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year`
- OpenAI ↔ StackExchange：`https://stackoverflow.co/company/press/archive/openai-partnership`

#### 5.3.2 Fair use（合理使用，Section 107）

判断是否合理使用的四个因素：

1. 使用目的与性质（教育 > 商业；转换性/transformative > 复制性）
2. 原作品性质（事实性 > 虚构；非创作性 > 创作性）
3. 使用量与关键性（片段 > 全部）
4. 对市场的影响（是否损害原作品市场或潜在市场）

合理使用例子：

- 看完电影写摘要
- 复现算法思想而不复制具体代码表达
- Google Books 建索引并展示 snippets（Authors Guild v. Google, 2002–2013）

一个容易误解的点：

- 版权不只是“逐字记忆/逐字复制”的问题
  - 情节与角色（如 Harry Potter）也可能受保护
  - 戏仿（parody）更可能被认为合理使用
- 版权更接近“语义 + 经济影响”的问题

对基础模型的考虑（课程讨论方向）：

- 即使只是“复制数据进入训练管线”，在法律上也可能已经构成侵权的起点
- 训练模型通常具有转换性（远非 copy/paste）
- 视觉领域直觉：系统学的是“概念/想法”（如 stop sign），而不是某张图片的具体表达细节
- 但语言模型确实可能影响写作者/艺术家的市场，这一点会变得复杂

#### 5.3.3 Terms of service（服务条款）

- 即使你有许可或能主张合理使用，ToS 仍可能施加额外限制
  - 例：YouTube ToS 禁止下载视频，即便视频是 CC 许可

进一步阅读：

- Stanford CS324 课程笔记：`https://stanford-cs324.github.io/winter2022/lectures/legality/`
- Fair learning（Lemley & Casey）：`https://texaslawreview.org/fair-learning/`
- Foundation models and fair use：`https://arxiv.org/pdf/2303.15715`
- The Files are in the Computer：`https://arxiv.org/abs/2404.12590`

## 6. 中期训练 + 后训练：面向能力的“专项数据”

### 6.1 长上下文（Long context）

需求：长上下文（例如对整本书做 QA）

- DeepSeek v3：128K tokens
- Claude 3.5 Sonnet：200K tokens
- Gemini 1.5 Pro：1.5M tokens

为什么不直接在预训练阶段就用超长序列？

- Transformer 的计算/显存开销与序列长度近似**二次增长**
- 因此更常见策略：先在较短上下文上预训练，再在后续阶段“补长上下文”

LongLoRA：`https://arxiv.org/pdf/2309.12307`

- 把 Llama2 7B 的上下文从 4K 扩展到 100K
- 方法要点：
  - shifted sparse attention（论文 Figure 2）
  - positional interpolation（Chen+ 2023）
- 训练数据：长文档
  - PG-19（书籍）
  - Proof-Pile（数学）

### 6.2 任务型数据（Tasks）：把现有 NLP 数据集“改写成提示词”

TL;DR：把大量已有的 NLP 数据集统一转换成 prompts 形式，用来增强任务泛化能力。

Super-Natural Instructions：`https://arxiv.org/pdf/2204.07705`

- 数据：1600+ 任务（论文 Figure 2）
  - HuggingFace 数据集：`https://huggingface.co/datasets/Muennighoff/natural-instructions`
- 用于微调 T5 做 k-shot 学习（Tk-instruct）
- 任务来自社区（通过 GitHub 贡献）
- 每个任务的样例来自既有数据集，再被转成模板化 prompts
- 论文中声称：尽管模型更小，但在对比中优于 InstructGPT（值得进一步对齐评测设置）

Flan 2022：`https://arxiv.org/pdf/2301.13688`

- 数据：1800+ 任务
  - HuggingFace 数据集：`https://huggingface.co/datasets/Muennighoff/flan`
- 对同一任务构造多种形式：zero-shot、few-shot、chain-of-thought（论文 Figure 7）

### 6.3 指令/对话数据（Instruction + Chat）：更开放的指令，合成数据占比高

TL;DR：相比任务型数据，更开放、更接近对话场景；合成数据（由更强模型生成）使用非常普遍。

Alpaca：`https://crfm.stanford.edu/2023/03/13/alpaca.html`（课程代码中引用：`alpaca`）

- 用 self-instruct 从 `text-davinci-003` 生成 52K 指令样本：`https://arxiv.org/pdf/2212.10560`
- 用这些数据对 LLaMA 7B 做监督微调

Vicuna：`https://lmsys.org/blog/2023-03-30-vicuna/`

- 用 ShareGPT 上 70K 对话微调 LLaMA
  - ShareGPT：用户分享 ChatGPT 对话（目前已基本停用）

Baize：`https://arxiv.org/pdf/2304.01196`

- 用 GPT-3.5 进行 self-chat 生成 111.5K 数据
- 种子问题来自 Quora 与 StackOverflow
- 微调 LLaMA

WizardLM：`https://arxiv.org/pdf/2304.12244`

- Evol-Instruct：把问题“进化”得更广/更难（论文 Figure 1）
- 用该数据微调 LLaMA

MAmmoTH2：`https://arxiv.org/pdf/2405.03548`

- 策划 WebInstruct：从 Common Crawl 提取 1000 万条指令
- 过滤：用 fastText 在 quiz sites 上训练分类器
- 抽取：用 GPT-4 与 Mixtral 抽取 QA pairs
- 用这些数据微调 Mistral 7B
- 目标效果：提升数学能力

OpenHermes 2.5：

- 多数据集聚合：`https://huggingface.co/datasets/teknium/openhermes`
- 用 GPT-4 生成的 100 万样本微调 Mistral 7B：`https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B`

Llama 2 chat：`https://arxiv.org/pdf/2307.09288`

- 通过供应商标注得到 27,540 条高质量指令数据
- 论文声称：比使用开源数据集的数百万样本效果更好
- 课程评论：也许可以标注更少 SFT 数据，把更多精力留给 RLHF 数据获取/标注

Llama-Nemotron 后训练数据（NVIDIA, 2024）：`https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset`

- Prompts：来自公共数据集（如 WildChat）或合成生成，然后再过滤
- 合成回复：由 Llama、Mixtral、DeepSeek r1、Qwen 等生成（相对更具商业可行性，不依赖 GPT-4）
- 包含 reasoning traces
- 示例：`https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/viewer/SFT/code`