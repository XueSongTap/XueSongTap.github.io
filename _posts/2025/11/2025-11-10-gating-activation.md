---
layout: article
title: Gated Activation 与 ReGLU
tags: Transformer
---

## 1 标准前馈层回顾

Transformer 前馈层（Feed-Forward Layer, FFN）是注意力块外的主要非线性单元，通常由两层线性映射加一次激活构成

### 1.1 基础公式

$$
FF(x) = \text{ReLU}(x W_1) W_2 = \max(0, xW_1) W_2
$$

#### 1.1.1 关键组件

* $x$：来自自注意力或上一层输出的输入向量。
* $W_1$：升维线性变换，产生中间隐空间。
* $\max(0, \cdot)$：共享阈值的 ReLU 激活。
* $W_2$：把激活后的特征投影回原始维度。

ReLU 固定使用零阈值，所有神经元以同一规则开关，难以针对不同 token 动态调节信息流。

## 2 门控机制的引入

为提升自适应能力，**Gated Linear Units（GLU）** 及其变体在激活阶段增设门分支，让 FFN 学会对不同特征分配不同“通行证”。

### 2.1 数学化表达

门控的核心替换是：

$$
\max(0, xW_1) \rightarrow \max(0, xW_1) \otimes (xV)
$$

#### 2.1.1 表达含义

* $\otimes$：逐元素乘，确保每个维度都能独立调节。
* $xV$：额外线性层生成的门控信号，可视作“输入相关的滤波器”。
* 结果：输出不再只由主分支决定，而是由门分支控制通过比例。

## 3 ReGLU 流程

**ReGLU** 选择保持主分支的 ReLU 激活，同时使用线性门分支，得到：

$$
FF_{\text{ReGLU}}(x) = \left(\max(0, xW_1) \otimes (xV)\right) W_2
$$

| 模型     | 第一层输出                        | 是否门控 | 额外参数 |
| -------- | ---------------------------- | ---- | ------ |
| 标准 FFN | $\max(0, xW_1)$              | 否    | 无      |
| ReGLU    | $\max(0, xW_1) \otimes (xV)$ | 是    | $V$    |

### 3.1 设计直觉

* 门信号依赖输入 $x$，不同 token 激活不同通道。
* 结构保持残差拓扑不变，却显著改善梯度流。
* 额外参数 $V$ 成本小，却提供更细粒度的特征选择能力。

## 4 GLU 家族与实践

不同激活函数对应不同 GLU 变体，常见组合如下：

| 变体     | 激活函数    | 定义                                |
| -------- | ------- | --------------------------------- |
| GLU      | sigmoid | $(xW_1) \otimes \sigma(xV)$       |
| GeGLU    | GELU    | $\text{GELU}(xW_1) \otimes (xV)$  |
| ReGLU    | ReLU    | $\max(0, xW_1) \otimes (xV)$      |
| SwiGLU   | Swish   | $\text{Swish}(xW_1) \otimes (xV)$ |

> LLaMA、PaLM、Chinchilla 更偏好 **SwiGLU**，T5 使用 **ReGLU**，而 GPT 早期版本仍沿用标准 FFN。

### 4.1 典型应用提示

* 变体选择要在训练稳定性、硬件友好度、推理延迟之间权衡。
* SwiGLU 常带来更强表达力，但 ReGLU 参数更少、实现更轻量。
