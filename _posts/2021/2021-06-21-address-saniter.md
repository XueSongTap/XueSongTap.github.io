---
layout: article
title: 深入剖析AddressSanitizer实现机制，内存错误检测原理与性能开销分析
tags:  cpp Asan
---

## 引入
ASan是google提供的一个内存检测工具，(来自gpt3.5)

ASan通过在编译时插入额外的代码来实现内存错误检测，并提供了相应的运行时库来捕获和报告错误。可以知道通过一下方式实现：

1. 插桩：ASan使用编译器插桩技术，在编译时修改源代码，插入额外的代码。这些额外的代码用于跟踪内存分配、释放和访问操作，以及检测内存错误。

2. 内存分配器：ASan使用自定义的内存分配器，用于跟踪分配的内存块，并在每个内存块之前和之后添加红区（redzone）。红区是一段未分配的内存，用于检测缓冲区溢出。

3. 彩色标记：ASan使用彩色标记技术，将分配的内存块分为不同的颜色，并将颜色信息存储在内存块的元数据中。这样，在访问内存时，ASan可以根据元数据中的颜色信息检测出内存错误。


## 原理理解速通 Short Version

run-time 的 library 取代了free 和malloc函数，

## 插桩

## 内存分配器

## 彩色标记


## 参考

https://learn.microsoft.com/zh-cn/cpp/sanitizers/asan?view=msvc-170

https://github.com/google/sanitizers/wiki/AddressSanitizer

https://github.com/google/sanitizers

https://zhuanlan.zhihu.com/p/382994002


https://github.com/google/sanitizers/wiki/AddressSanitizerAlgorithm