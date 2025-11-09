---
layout: article
title: torch的参数处理化特殊处理
tags: FP8
---

## 1 nn.Parameter 的作用与原理


代码：
```python
w = nn.Parameter(torch.randn(input_dim, output_dim))
```


### 1.1 含义

* `nn.Parameter` 是 PyTorch 的特殊张量类型，用来定义 **模型的可训练参数**；
* 当你把它放在 `nn.Module` 中时，`model.parameters()` 会自动包含它；
* 它在反向传播时会自动累积 `.grad`。

等价于一个带标志的 tensor：

```python
param = torch.Tensor(...)
param.requires_grad_(True)
```

### 1.2 参数与输入维度的关系

```python
input_dim = 16384
output_dim = 32
x = nn.Parameter(torch.randn(input_dim))
output = x @ w
assert output.size() == torch.Size([output_dim])
```

这里：

* `x` 是输入向量 `[16384]`
* `w` 是参数矩阵 `[16384, 32]`
* `output = x @ w` → `[32]`

## 2 问题：输出随输入维度增大而“爆炸”


> “Note that each element of output scales as sqrt(input_dim): 18.9”

如果 `w ~ N(0, 1)`，即标准正态分布，那么：
$$
\text{Var}(x @ w) = input_dim × Var(w) = input_dim
$$
所以输出的标准差约为：
$$
σ_{output} = \sqrt{input_dim}
$$

当 `input_dim = 16384` 时，
$\sqrt{16384} = 128$，
说明输出数值会非常大 → **梯度容易爆炸**，训练不稳定


## 3 解决方案：按维度缩放

目标：**让输出分布不随输入维度变化**。

做法：在初始化时缩放参数：

```python
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
```

此时有：

$$
Var(w) = \frac{1}{input_dim}
\Rightarrow Var(x @ w) = 1
$$

即输出方差恒定，稳定训练过程




## 4 截断正态分布（Truncated Normal）

> “To be extra safe, we truncate the normal distribution to [-3, 3]”

普通正态分布可能出现极端值（outlier），影响训练稳定性。
PyTorch 提供：

```python
nn.init.trunc_normal_(tensor, std=1/√input_dim, a=-3, b=3)
```

即生成服从正态分布但裁剪在 $[-3σ, 3σ]$ 范围内的值，
以减少异常权重带来的不稳定。