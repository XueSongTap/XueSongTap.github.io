---
layout: articles
title: d2l ai 动手学习深度学习 02 预备知识
tags: ubuntu d2l ai
---

# 2.1 数据操作
n维数组 张量tensor

maxnet: ndarray

pytorch/tensorflow: Tensor

tensor 可以支持gpu计算，支持微分

下面主要是pytorch版本

## 2.1.1 入门

引入pytorch

```python
import torch
```
## 2.1.2 运算符

## 2.1.3 广播机制

## 2.1.4 索引和切片

## 2.1.5 节省内存


## 2.1.6 转换为其他python对象

# 2.2 数据预处理

## 2.2.1 读取数据集



### 2.2.2 处理缺失值

"NaN" 代表缺失值 


用插值法处理缺失的数据

inputs里面的缺失值用同一列均值替代

```python
inputs, outputs = data.iloc[:, 0:2], data.illoc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

### 2.2.3 转换为张量格式

数值转换成张量格式

```python
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```
(tensor([[3., 1., 0.],
         [2., 0., 1.],
         [4., 0., 1.],
         [3., 0., 1.]], dtype=torch.float64),
 tensor([127500, 106000, 178100, 140000]))
```


# 2.3 线性代数

# 2.3.1 标量 Scalar
标量用小写字母表示

## 2.3.2 向量

## 2.3.3 矩阵


## 2.3.4 张量

## 2.3.5 张量算法的本质

## 2.3.6 降维

## 2.3.7 点积

## 2.3.8 矩阵向量积

## 2.3.9 矩阵-矩阵乘法

## 2.3.10 范数



# 2.4 微积分


# 2.5

# 2.6

# 2.7