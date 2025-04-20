---
layout: articles
title:  .pyx 文件不是编译产物，是Cpython 代码 
tags: redis
---


最近在参与 [tilelang](https://github.com/tile-ai/tilelang) 项目开发时，我提了一个 [PR #267](https://github.com/tile-ai/tilelang/pull/267)，遇到了需要修改`.pyx` 的情况
---

## 起因：py 修改

这个pr主要是修复非CUDA环境的编译问题，找到问题的 parser.py 进行了提交，但是reviwer 提了

> You should also update `parser.pyx` accordingly.

我当时一愣：`.pyx`？这不是编译后的产物吗？为什么还要改？我一度以为这类似 `.pyc` 或 `.so` 文件，是由某些构建流程生成的

---

## `.pyx` 文件并不是编译产物！

我很快发现，`.pyx` 实际上是 **Cython 的源代码文件**，**和 `.py` 作用类似**，但它允许使用 C 语言扩展语法，最终会被编译成 C，再生成 `.so` 或 `.pyd` 扩展模块。

简单说：
> `.pyx` 是源码，不是中间产物！

这是我此次 PR 学到的最关键一点。

Cython 是 Python 的一个超集语言，它允许你在写 Python 的同时，插入 C 级别的类型声明、函数调用等，大大提高运行效率。比如：

```cython
# example.pyx
def add(int a, int b):
    return a + b
```

这个 `.pyx` 文件最终会通过 Cython 编译生成 C 代码，再通过编译器生成共享对象（`.so`）文件，供 Python 直接 `import` 使用。这个过程不是自动完成的，源码文件 `.pyx` 是需要**人工维护和提交**的。

---

## 为什么 tilelang 会用 `.pyx`？

TileLang 是一个在图编译、深度学习 IR 等场景中使用的 DSL，它对运行效率有一定要求。使用 Cython 能带来两个好处：

1. **性能加速**：比如在处理 Token 流、解析器、匹配规则等密集计算任务时，通过类型声明和 C 层调用提高速度。
2. **静态接口绑定**：很多 DSL 项目会需要和底层 C++/C 接口交互，Cython 可以做天然桥梁。

因此，tilelang 中的 `.pyx` 文件其实是项目最核心的源码之一

