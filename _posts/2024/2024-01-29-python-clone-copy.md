---
layout: article
title: Python深浅拷贝详解：内存管理机制、实现原理与常见陷阱
tags: python
---

## 引言

写torch过程中接触到torch.clone() 和 python.copy()
## 理解Python中的对象复制概念

在Python中，有三种主要的方式来"复制"一个对象：

1. **赋值操作（=）**：创建对原始对象的引用，而非复制对象本身
2. **浅复制（Shallow Copy）**：创建一个新对象，但内部元素仍指向原始对象
3. **深复制（Deep Copy）**：创建一个全新的对象，包括所有嵌套的对象

### 赋值操作

```python
list1 = [1, 2, [3, 4]]
list2 = list1  # 赋值操作，list2引用与list1相同的对象

list2[0] = 5
print(list1)  # 输出：[5, 2, [3, 4]]
```

在这个例子中，`list1`和`list2`指向内存中的同一个对象。修改`list2`也会影响`list1`，因为它们实际上是同一个列表的两个名称。

## `.copy()` 方法详解

在Python的标准库中，`.copy()`方法用于创建对象的浅复制。

### 内置数据类型中的`.copy()`

Python的多种内置容器类型都提供了`.copy()`方法：

#### 列表的`.copy()`

```python
original_list = [1, 2, [3, 4]]
copied_list = original_list.copy()

# 修改基本类型元素
copied_list[0] = 5
print(original_list)  # 输出：[1, 2, [3, 4]]
print(copied_list)    # 输出：[5, 2, [3, 4]]

# 修改嵌套列表
copied_list[2][0] = 30
print(original_list)  # 输出：[1, 2, [30, 4]]
print(copied_list)    # 输出：[5, 2, [30, 4]]
```

这个例子展示了浅复制的本质：外层容器是独立的，但内部嵌套的可变对象仍然是共享的。

#### 字典的`.copy()`

```python
original_dict = {'a': 1, 'b': [2, 3]}
copied_dict = original_dict.copy()

# 修改键值
copied_dict['a'] = 10
print(original_dict)  # 输出：{'a': 1, 'b': [2, 3]}
print(copied_dict)    # 输出：{'a': 10, 'b': [2, 3]}

# 修改嵌套列表
copied_dict['b'][0] = 20
print(original_dict)  # 输出：{'a': 1, 'b': [20, 3]}
print(copied_dict)    # 输出：{'a': 10, 'b': [20, 3]}
```

与列表类似，字典的`.copy()`方法也创建浅复制。

### 其他创建浅复制的方式

除了使用`.copy()`方法，Python还提供了其他创建浅复制的方式：

```python
# 使用切片操作创建列表的浅复制
list_copy = original_list[:]

# 使用list()构造函数
list_copy = list(original_list)

# 使用dict()构造函数
dict_copy = dict(original_dict)
```

### `.copy()`的局限性

浅复制对于简单的数据结构可能足够，但当处理包含嵌套可变对象的复杂数据结构时，修改复制对象的嵌套元素会影响原始对象，这可能导致难以追踪的错误。

## 实现深复制

当需要创建对象的完全独立副本时，我们需要使用深复制。Python的`copy`模块提供了`deepcopy()`函数：

```python
import copy

original_list = [1, 2, [3, 4]]
deep_copied_list = copy.deepcopy(original_list)

# 修改嵌套列表
deep_copied_list[2][0] = 30
print(original_list)      # 输出：[1, 2, [3, 4]]
print(deep_copied_list)   # 输出：[1, 2, [30, 4]]
```

深复制会递归地复制所有嵌套对象，确保原始对象和复制对象完全独立。

## `torch.clone()` 方法详解



在PyTorch中，张量（Tensor）对象提供了`.clone()`方法，用于创建张量的深复制：

```python
import torch

original_tensor = torch.tensor([1, 2, 3])
cloned_tensor = original_tensor.clone()

# 修改克隆张量
cloned_tensor[0] = 5
print(original_tensor)  # 输出：tensor([1, 2, 3])
print(cloned_tensor)    # 输出：tensor([5, 2, 3])
```

PyTorch的`.clone()`方法确保了新张量有自己的内存，与原始张量完全分离。

### Pandas中的复制操作

在Pandas库中，DataFrame对象提供了`.copy()`方法，但与Python标准库不同，它可以接受一个`deep`参数来指定是浅复制还是深复制：

```python
import pandas as pd

df_original = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_shallow = df_original.copy(deep=False)  # 默认是deep=True
df_deep = df_original.copy(deep=True)
```

### NumPy中的复制

NumPy数组可以使用`.copy()`方法创建深复制：

```python
import numpy as np

arr_original = np.array([1, 2, 3])
arr_copied = arr_original.copy()
```