---
layout: article
title: 深入分析dynamic_cast的工作机制、RTTI实现、性能开销及最佳实践
tags: cpp
---

### 底层实现

`dynamic_cast` 的底层实现涉及两个主要的机制：虚函数表（vtable）和运行时类型信息（RTTI）。

#### 1. 虚函数表（vtable）：
   虚函数表是用于实现多态性的一种机制。它是一个存储在对象内存布局中的特殊表格，用于存储虚函数的地址。每个具有虚函数的类都有一个对应的虚函数表。虚函数表中的每个条目对应一个虚函数，包含了函数的地址。派生类的虚函数表会继承基类的虚函数表，并可以在其中添加或重写虚函数。

#### 2. 运行时类型信息（RTTI）：
   运行时类型信息是用于在运行时获取对象的实际类型的机制。C++ 中的 RTTI 通过 `type_info` 类型和 `typeid` 运算符来实现。每个具有虚函数的类都会自动生成一个与之对应的 `type_info` 对象，其中包含了类的名称和其他相关信息。`typeid` 运算符可以用于获取对象的类型信息，返回一个指向 `type_info` 对象的指针。


### 实现步骤
在 `dynamic_cast` 的底层实现中，它会使用虚函数表和运行时类型信息来进行类型转换的检查和操作。具体步骤如下：

1. 首先，`dynamic_cast` 会检查源指针或引用是否为空指针，如果是空指针，则转换结果也将是空指针。

2. 如果源指针或引用不是空指针，则 `dynamic_cast` 会根据源对象的虚函数表和运行时类型信息来进行类型转换的检查。

3. `dynamic_cast` 会在源对象的虚函数表中查找目标类型的虚函数，并检查其地址是否与源指针或引用指向的对象的虚函数表中的对应函数地址相同或兼容。

4. 如果类型转换是合法的，`dynamic_cast` 将返回指向目标类型的指针或引用；否则，如果转换不合法，它将返回空指针（对于指针类型）或引发 `std::bad_cast` 异常（对于引用类型）。

需要注意的是，`dynamic_cast` 的底层实现依赖于编译器对虚函数表和运行时类型信息的支持。不同的编译器可能有不同的实现方式，但基本原理是相似的。此外，为了使用 `dynamic_cast` 进行类型转换，需要在编译时启用 RTTI 功能。


## 参考 

http://www.selfgleam.com/rtti_layout

http://www.uusystem.com/C++/C++%E4%B8%AD%E5%9F%BA%E6%9C%AC%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B/C++%E5%9B%9B%E7%A7%8D%E5%BC%BA%E5%88%B6%E7%B1%BB%E5%9E%8B%E8%BD%AC%E6%8D%A2/C++%E5%9B%9B%E7%A7%8D%E5%BC%BA%E5%88%B6%E7%B1%BB%E5%9E%8B%E8%BD%AC%E6%8D%A2.html

https://github.com/grmaple/cpp_mianshi/blob/master/C%2B%2B/C%2B%2B%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/%E5%9B%9B%E7%A7%8Dcast%E7%B1%BB%E5%9E%8B%E8%BD%AC%E6%8D%A2.md

https://cirnoo.github.io/2019/08/12/dynamic_cast/

https://zhuanlan.zhihu.com/p/580330672

https://lancern.xyz/2022/11/04/dynamic-cast-benchmark/