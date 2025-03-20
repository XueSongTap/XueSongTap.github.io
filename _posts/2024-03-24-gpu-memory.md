---
layout: articles
title: GPU MEM 训练和微调估算
tags: gpu
---


在训练或微调大型语言模型(LLM)和其他深度学习模型时，GPU显存限制往往是主要瓶颈。了解模型所需的最低显存是避免OOM(内存不足)错误的关键一步。以下是两个有用的在线工具，可以帮助你估算显存需求：

## 1. HuggingFace的模型内存使用估算器

HuggingFace提供了一个专门的工具，可以帮助估计训练和微调所需的最低显存大小：

- **名称**：Model Memory Utility
- **链接**：[https://huggingface.co/spaces/hf-accelerate/model-memory-usage](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)
- **用途**：估算训练或微调模型所需的最低GPU显存
- **受欢迎程度**：918个赞

这个工具允许你选择特定模型，并根据批量大小、精度等参数计算所需的显存。它可以帮助你在购买GPU资源或规划训练任务时做出明智的决策。

## 2. LLM检查工具

另一个类似工具是Rahul S Chand开发的LLM检查工具：

- **名称**：LLM check (GPU poor)
- **链接**：[https://rahulschand.github.io/gpu_poor/](https://rahulschand.github.io/gpu_poor/)
- **注意**：需要启用JavaScript才能运行

这个工具可以帮助评估你的GPU配置是否足以运行特定的LLM模型。
