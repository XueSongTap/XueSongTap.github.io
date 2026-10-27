---
layout: article
title: MiMo-V2.6-RL-oss：一条任务怎样成为 Agentic RL 环境
tags: LLM Agent RL
description: >-
  从 MiMo-V2.6-RL-oss 的具体任务出发，说明需求、容器环境与 verifier 如何组成 Agentic RL 任务，比较 Code、Cyber、Webdev、Music 和 General 的评分方式，并区分 Explorer 在线评分与训练奖励。
excerpt: >-
  从 MiMo-V2.6-RL-oss 的具体任务出发，说明需求、容器环境与 verifier 如何组成 Agentic RL 任务，比较 Code、Cyber、Webdev、Music 和 General 的评分方式，并区分 Explorer 在线评分与训练奖励。
---

> 关联阅读：任务构造可接着看 [CodeMidas：从源代码构造 Coding Agent 的 RL 训练环境]({% post_url 2026/09/2026-09-21-codemidas-coding-rl-environments %})；训练方法可参考 [RLHF 到 GRPO：大模型强化学习训练方法梳理]({% post_url 2026/03/2026-03-22-rlhf-grpo %})。

一条 Code 任务要求修改某个项目的 GitHub Action：让 `secrets` 参数变成可选项，这样只需要 Vault token 时也能调用它。乍看就是一个很常见的开源开发需求，难点在于怎样在 RL 场景下给这种任务一个准确的评判。

在 [MiMo-V2.6-RL-oss][dataset] 中，这条需求和项目的容器镜像、工作目录、测试补丁、测试命令放在同一条数据里。模型进入容器读代码、改代码，结束后 verifier 再把测试应用到结果上，运行测试并给出奖励。所以数据集里的一行不是“问题—标准答案”对，而是一次 agent 任务的启动配置，真正被训练的是模型在环境里的整段操作轨迹。[1][2]

小米一共发布了 7,780 条任务，覆盖 Code、Webdev、Cyber、Music 和 General 五类。FineEnvs 的 [MiMo RL Environment Explorer][explorer] 把这些配置、评分说明和试跑轨迹整理到了一起。

## 一条 Code 任务里面有什么

回到开头那条 Vault token 任务。`prompt` 描述需求，`extra_info.instance_json` 指定工作目录、Docker 镜像、测试命令、测试补丁和 verifier 超时。只看 `prompt` 是复现不了这道题的：仓库的起始状态在镜像里，做没做对由测试补丁和执行命令决定。[2]

这条任务的测试补丁检查了几个具体行为：空 `secrets` 能被解析；开启 `exportToken` 时只导出 `VAULT_TOKEN`；读取 `secrets` 时不再把它标为必填。agent 可以在容器里自己测，评分用的材料则要等它做完才放进评分环境。Explorer 展示的 Code 奖励很直接：隐藏测试通过得 1，否则得 0；如果是测试环境本身出错、没法评分，这种基础设施故障不会算成模型的 0 分。[2][4][5]

## 都是 reward，检查点各不相同

Code 的测试能把“改好了没有”变成 0 或 1。但换成安全、网页或音乐任务，交付物变了，verifier 也得跟着换。

先看 Cyber 子集的 `arvo_35858`。题面只给出漏洞目标：让 dnsmasq 在 `rfc1035.c` 的 `extract_name` 中触发 AddressSanitizer 报告的 `heap-buffer-overflow`。任务配置指定了 `arvo-rl:v1-arvo-35858` 镜像和 `/home/agent` 工作目录，agent 可以读源码、用本地的 `run.sh` 测试，最后向 `submit.sh` 交一个 PoC 文件。提交反馈分 `crash` 和 `match` 两种：光让程序崩溃还不够，崩溃类型要一致，指定函数还得是 sanitizer 栈里最上面的项目代码帧。Explorer 把最后提交的 PoC 是否 `match` 映射成 0/1 奖励。所以这里的 verifier 不看漏洞解释写得对不对，而是把输入实际跑一遍，定位崩溃。[4][9]

Webdev 的 `dasyn_260638_00014` 要求用 React 给一家叫 Study Bites 的辅导咖啡馆做静态网站：粗野主义风格，包含课程菜单、导师推荐、活动、评论，以及不需要后端的预约表单。配置给了 `/workspace` 和 `webdev-rl-opensource:v2` 镜像，却没有 Code 题那样的测试补丁，这里的交付物是运行后渲染出来的页面。Explorer 的在线评分先截取整页，再让视觉模型评估视觉表现、需求覆盖和素材质量；其中视觉表现是五项准则的平均，三大项再取平均。这个分数依赖指定的裁判模型。另外 Explorer 也写明了，训练时的 Webdev 评分是在同组至多八条轨迹里做相对选择，并扣掉未满足需求的部分，所以在线的绝对分数不能直接当训练 reward 用。[4][5][10]

Music 子集第一条 `gK-0768` 是道中文题：写一首 E 大调、129 BPM、4/4 拍、56 小节、三个声部的英式乡村舞曲，用 ABC 记谱输出。行内的 `extra_info` 直接存了 `bpm: 129`、`meter: 4/4`、`nvoice_want: 3` 这些条件。Explorer 展示的评分流程是：先用 `abc2midi` 解析作品，再计算 18 项音乐特征，和人类音乐的特征范围做比较；奖励由加权特征组和直方图相似度按 0.85 和 0.15 合成，最后归一到 0～1。解析不了或测不出来的作品直接得 0。可见规则式奖励也可以是连续分数，不一定是 Code 那种通过/失败。[4][11]

三个例子放在一起看，`prompt` 只是任务入口：Cyber 还要有目标二进制和崩溃判定，Webdev 要有可渲染的页面和视觉裁判，Music 要有能解析的乐谱和特征计算。`reward_model.style: rule` 这种外层字段也说明不了奖励是怎么来的：这些任务都归在“规则式”下面，实际检查的却是完全不同的交付物。所以读一条 RL 数据，至少要把 agent 能操作的环境、最终交付物和任务专属的 verifier 放在一起看。[2][4]

## General 为什么需要 rubric

General 的 989 条任务又分两类：925 条模拟工作场景，64 条是终端任务。前者可能让 agent 查表格、在模拟业务系统里检索记录，最后交一份报告；后者更像“改文件、过测试”的终端题。前者很难写成一条“运行测试是否通过”的断言。[3][4]

举一条法律合规任务：题目要求核对会议律师名单，把订单摘录、最终 Excel 名单和模拟的 Litify 登记系统对起来，排除已被替代的记录，并说明姓名、案件和当事人身份。环境里有工作区文件和多种模拟系统；`verifier_meta.json` 则拆出 7 项检查，包括去重后的律师人数、完整的身份集合、不同案件的归属关系等。也就是说，“核对名单”这个笼统要求被拆成了一组能逐项判分的事实。[3]

统计 925 条模拟工作任务的 `verifier_meta.json`，一共有 5,125 项检查，每题中位数 5 项；其中 4,437 项标为模型裁判，688 项标为规则检查。也就是说，**这一部分约 87% 的检查项依赖模型裁判**。Explorer 按权重汇总这些检查得到奖励，没设权重的按 1 处理。至于“critical / important / sanity”，它们只是标签，别自己把它们当成奖励系数。[3][4]

这组数字也能看出 General 的难点：它把知识工作纳入了可训练范围，同时把评分可靠性压在了 rubric 和裁判模型上。律师名单题把人数、身份和排除条件都写清楚了，裁判至少有可核对的锚点；如果只问“报告是否充分”，同样的 0/1 判断就很难稳定。当然，这是从公开检查项的结构推出来的，没有实测裁判的误判率。[3][4]

## 从发布集看任务设计

把这些任务并排看，有三个共同条件：起点能复现，agent 有可操作的空间，结果在结束后能检查。Code 靠项目镜像和测试做到这一点；Cyber 靠固定的目标程序和预期崩溃；General 则要同时准备工作区、模拟系统和答案锚点。说到底，这些任务不是把普通问答题换了个名字，而是给 agent 搭了一个能真正干活的现场。[1][2][3][4]

发布集里 Code 有 2,698 条，Webdev 有 2,093 条，两者合计约占 62%；Cyber 和 Music 各 1,000 条，General 989 条。不过任务行数不等于混合 RL 训练时的采样比例。一条 Webdev rollout 要渲染页面、调用视觉裁判，一条 Code rollout 可能要反复读仓库、跑命令，光凭条目数算不出各类任务消耗的 token 和算力。[1][6]

数据集卡片给了领域、任务家族和 verifier 的概览，但没有逐条展开候选任务从哪来、经过哪些过滤、每一步留下多少。CodeMidas 论文倒是详细写了自己的环境清理、参考答案检查和 rollout 筛选流程，但这些步骤不能直接套成 MiMo-V2.6-RL-oss 所有领域统一的入库规则。[1][7]

小米把任务、Docker 镜像和基于 verl 的训练代码一起发布了。FineEnvs 和 Hugging Face 做的 [Explorer][explorer] 是非官方的浏览和试跑界面：用 OpenCode 当 agent harness，模型和裁判由试跑的人自己选。打开某道题的轨迹，能看到 agent 做了什么、哪项检查没过；但它的一次在线得分不能直接和小米报告里的训练 reward 或基准结果放在一起比。[5][6][8]


## 参考

[1] XiaomiMiMo, [MiMo-V2.6-RL-oss 数据集与 Dataset Card](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)，2026，访问日期 2026-09-30。

[2] XiaomiMiMo, [Code 子集首条任务的公开数据行](https://datasets-server.huggingface.co/rows?dataset=XiaomiMiMo%2FMiMo-V2.6-RL-oss&config=code&split=train&offset=0&length=1)，2026，访问日期 2026-09-30。

[3] XiaomiMiMo, [General 任务 `s3k_2076_legal_compliance_en_t5_rl_009` 的 verifier 元数据](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/blob/main/general/envs/s3k_2076_legal_compliance_en_t5_rl_009/verifier_meta.json)，2026；统计方法参考 [Explorer 的数据构建脚本](https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer/blob/main/build_data.py)。

[4] Hugging Face 与 FineEnvs, [Explorer 的 Reward design 实现](https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer/blob/main/web/js/rewards.js)，2026，访问日期 2026-09-30。

[5] Hugging Face 与 FineEnvs, [MiMo RL Environment Explorer 说明](https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer/blob/main/README.md)，2026，访问日期 2026-09-30。

[6] XiaomiMiMo, [基于 verl 的 Agentic RL 复现代码](https://github.com/XiaomiMiMo/verl)，2026，访问日期 2026-09-30。

[7] Bowen Ye 等, *CodeMidas: Scaling Agentic Coding RL Environments from Code Itself*, 2026，[arXiv:2609.22068](https://arxiv.org/abs/2609.22068)。

[8] Xiaomi LLM-Core, *MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement*, 2026，[技术报告 PDF](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)；[发布说明](https://mimo.mi.com/docs/en-US/news/latest/v2-6)。

[9] XiaomiMiMo, [Cyber 子集首条任务 `arvo_35858`](https://datasets-server.huggingface.co/rows?dataset=XiaomiMiMo%2FMiMo-V2.6-RL-oss&config=cyber&split=train&offset=0&length=1)，2026，访问日期 2026-09-30。

[10] XiaomiMiMo, [Webdev 子集首条任务 `dasyn_260638_00014`](https://datasets-server.huggingface.co/rows?dataset=XiaomiMiMo%2FMiMo-V2.6-RL-oss&config=webdev&split=train&offset=0&length=1)，2026，访问日期 2026-09-30。

[11] XiaomiMiMo, [Music 子集首条任务 `gK-0768`](https://datasets-server.huggingface.co/rows?dataset=XiaomiMiMo%2FMiMo-V2.6-RL-oss&config=music&split=train&offset=0&length=1)，2026，访问日期 2026-09-30。

[dataset]: https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss
[explorer]: https://huggingface.co/spaces/FineEnvs/MiMo-RL-Envs-Explorer
