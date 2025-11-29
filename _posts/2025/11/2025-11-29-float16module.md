---
layout: article
title: Megatron Float16Module 的参数传法
tags: Megatron Precision
---

Megatron 的 `Float16Module` 只会把**位置参数**转换成半精度（bf16/fp16）。一旦只用关键字参数，输入会保持原 dtype，导致前向混用 fp32/bf16。

## 1 相关代码

代码位置：`megatron/core/transformer/module.py#L404` https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/module.py#L404

```py
    def forward(self, *inputs, fp32_output=True, **kwargs):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=self.vp_stage):
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if (
            parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=self.vp_stage)
            and fp32_output is True
        ):
            outputs = float16_to_fp32(outputs)
        return outputs
```

## 2 行为解读

- 仅在流水线 **第一阶段** 把位置参数元组 `inputs` 转成 bf16/fp16；`kwargs` 完全不改。
- 调用是 `self.module(*inputs, **kwargs)`，因此 `module(hidden_states=x)`（全关键字）不会被半精度化。
- **最后阶段** 才会按 `fp32_output` 决定是否把输出升到 fp32，非末阶段不会升。

## 3 推荐用法

- 需要半精度输入时，用位置参数传入：`module(hidden_states, attn_mask)`。
- 确实要关键字时，采用“位置参数 + * + 关键字”的写法，在 Megatron 内部也常见：`module(hidden_states, *, attention_mask=mask)`。
- 如果外层封装只给关键字，显式在调用前 `tensor = tensor.to(torch.bfloat16)` 再喂入，避免隐藏的 dtype 混用。

## 4 TL;DR

Float16Module 只转换 `*inputs`，不碰 `**kwargs`。想要自动 bf16/fp16，就把张量放在位置参数里；关键字路径记得手动或语法上确保半精度。
