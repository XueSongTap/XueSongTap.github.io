---
layout: articles
title:  LLaMA Factory 训练中 SDPA 引发 grad_norm 为 NaN 问题解决
tags: redis
---


在使用 LLaMA Factory 进行模型训练时，问题：训练起步的时候梯度范数（`grad_norm`）显示为 `NaN`。相关的日志输出：

```bash
/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py:825: UserWarning: cuDNN SDPA backward got grad_output.strides() != output.strides(), attempting to materialize a grad_output with matching strides... (Triggered internally at /opt/pytorch/pytorch/aten/src/ATen/native/cudnn/MHA.cpp:674.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
{'loss': 80.8002, 'grad_norm': nan, 'learning_rate': 2.9411764705882354e-05, 'epoch': 0.18}
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 4.995131923687488e-05, 'epoch': 0.36}
```

### 环境信息
- PEFT: 0.15.0
- Transformers: 4.50.0
- PyTorch: 2.6.0a0+df5bbc09d1.nv24.12
- Datasets: 3.4.1
- Tokenizers: 0.21.0

### 问题根源
分析问题出在 PyTorch 的 cuDNN SDPA（Scaled Dot-Product Attention）实现上。在反向传播过程中，cuDNN 的多头注意力（Multi-Head Attention, MHA）模块要求输入梯度张量（`grad_output`）的步幅（strides）与输出张量（`output`）的步幅保持一致。然而，日志中的警告清楚地表明两者步幅并不匹配：

```
grad_output.strides() != output.strides()
```

步幅（strides）描述了张量在内存中的布局，决定了从一个元素到下一个元素的移动步长。当步幅不匹配时，cuDNN 无法直接处理这些张量。为解决这一问题，PyTorch 会尝试“物化”（materialize）一个新的 `grad_output` 张量，使其步幅与 `output` 对齐。虽然这一过程是自动完成的，但它会触发警告，并可能导致额外的性能开销（例如内存操作）或数值不稳定性。

最终，梯度范数（`grad_norm`）变为 `NaN`，这可能是步幅调整过程中引入了无效值，或者模型优化本身发生了数值溢出等问题。

### 社区反馈与临时解决方案
LLaMA Factory 的 GitHub Issue（[#7388](https://github.com/hiyouga/LLaMA-Factory/issues/7388)）中，社区确认这是 SDPA 实现的一个已知缺陷，并建议将注意力机制从 SDPA 切换到 Flash Attention 2，以规避 cuDNN 的步幅问题。

具体的切换方法可以参考 LLaMA Factory 官方文档：[高级参数配置](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/arguments.html)。默认情况下，`flash_attn` 参数被设置为 `auto`，而在我的环境中，这导致系统选择了 SDPA。通过手动设置 `flash_attn=fa2`，即可有效解决问题。

### PyTorch 官方修复
PyTorch 社区也在相关 Issue（[#138581](https://github.com/pytorch/pytorch/issues/138581)）中深入讨论了这一问题。NVIDIA 提供的 PyTorch 镜像（[PyTorch 24.12 发布说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-12.html)）基于 commit [df5bbc09d1](https://github.com/pytorch/pytorch/commit/df5bbc09d191fff3bdb592c184176e84669a7157) 构建，虽然版本号标为 2.6.0，但并未包含 2024 年 10 月 19 日的完整修复 PR（[#138354](https://github.com/pytorch/pytorch/pull/138354)）。
### 最终解决方案
我当前使用的 PyTorch 版本（2.6.0a0+df5bbc09d1.nv24.12）是一个预发布版本，未包含最新的修复。
因此，
1. 升级到 PyTorch 2.6 正式版是最彻底的解决方案。
2. 升级到 nv [pytorch镜像的 25.1 版本](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-01.html), 即  2.6.0a0+ecf3bae40a.

### 总结
- **短期方案**：将注意力机制切换到 Flash Attention 2，避免使用 SDPA。
- **长期方案**：升级到 PyTorch 2.6 正式版，应用官方修复，或者  2.6.0a0+ecf3bae40a nv镜像版本

