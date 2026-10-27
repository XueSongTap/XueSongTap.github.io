---
layout: article
title: 从一行 print 看 torch.compile 的 Graph Break
tags: LLM PyTorch torch.compile
description: >-
  从一行 print 引起的切图出发，说明 torch.compile 中 Dynamo 的捕获与恢复过程，区分 Graph Break、数据依赖分支、动态 shape 和重编译，并介绍 fullgraph、日志与 trace 的排查方法。
excerpt: >-
  从一行 print 引起的切图出发，说明 torch.compile 中 Dynamo 的捕获与恢复过程，区分 Graph Break、数据依赖分支、动态 shape 和重编译，并介绍 fullgraph、日志与 trace 的排查方法。
---

> 前置阅读：[torch.compile 里 Inductor 是怎么把编译结果包进 CUDA Graph]({% post_url 2026/05/2026-05-19-inductor-cudagraph-analysis %})。这篇往前看一步：Inductor 收到的图，是 Dynamo 怎么捕获出来的。

给函数加上 `torch.compile`，代码正常跑完，很容易以为整个函数都交给了编译器。先看这段代码：

```python
import torch


def f(x):
    y = x.sin()
    print("intermediate shape:", y.shape)
    return y.cos() + 1


compiled = torch.compile(f)
compiled(torch.randn(8, 16))
```

中间只插了一行打印。默认配置下，Dynamo 通常会在这里切图：先捕获 `sin`，回到 Python 执行 `print`，再从后面的 `cos` 和加法接着捕获。程序照样能跑，但后端拿到的计算已经被切成了两段。[1][3]

这就是 Graph Break。下面顺着这个例子看它发生在哪里，再看排查时常和它混在一起的 `.item()`、动态 shape 和重编译。文中代码用于说明和复现，没有在本文整理环境里实际跑过，具体报错文字以本机 PyTorch 版本为准。

## 先把 Dynamo 交出去的图打印出来

先不接 Inductor，写一个很小的 backend：

```python
def inspect_backend(gm, example_inputs):
    print("=== Dynamo captured ===")
    print(gm.code)
    return gm.forward


compiled = torch.compile(f, backend=inspect_backend)
compiled(torch.randn(8, 16))
```

Dynamo 会把捕获到的 FX `GraphModule` 和示例输入交给 backend，backend 返回一个可执行函数。这里直接返回 `gm.forward`，省掉后面的代码生成，方便观察捕获结果。[2] 注意 backend 里的打印发生在编译回调中，和 `f` 里那个正在被追踪的 `print` 不在一个地方。

开头那个例子的执行过程可以简化成下面这样（只是示意，变量名不必和实际生成的代码一致）：

```text
x → [图 1：sin] → y → Python print → [图 2：cos、add] → output
```

`y` 要从前一段传到后一段，Dynamo 在这个边界上还得保存恢复执行所需的 Python 状态。于是同一个 Python 函数里，就同时有了编译区域和 eager 区域。[1]

删掉 `print`，三个 tensor 运算才有机会一起交给后端，Inductor 也才能在更大的范围里安排 fusion；一旦切图，两个编译区域之间的这类优化机会就没了。至于具体多出几个 kernel、多花多少时间，得看后端实际生成的结果，没法从图的数量直接推算。

这里也正好接上前一篇的 CUDA Graph：FX graph 描述的是编译器捕获的计算，CUDA Graph 管的是 GPU 工作的捕获与重放。所以排查 Dynamo 时看到“两张图”，不等于“两次 CUDA Graph replay”。

## 用 fullgraph 把断点暴露出来

默认的 `fullgraph=False` 允许上面这种切图和恢复。调试时可以把同一个函数改成：

```python
strict_f = torch.compile(f, backend="eager", fullgraph=True)
strict_f(torch.randn(8, 16))
```

`backend="eager"` 仍然会走 Dynamo，适合先单独排查捕获问题；`fullgraph=True` 则要求这次捕获形成一张完整的 FX graph，碰到 graph break 直接报错。[4] 顺着报错里的用户代码位置，就能找回那行 `print`。

fullgraph 约束的只是捕获范围。后端照样可以为一张 FX graph 生成多个 kernel，打开这个选项也不会自动得到一个 CUDA Graph。

如果想让程序继续跑，同时看看在哪里切了图，可以把第一个例子存成 `graph_break_demo.py`，然后执行：

```bash
TORCH_LOGS="graph_breaks" python graph_break_demo.py
```

日志里先看用户代码的文件和行号，再看原因。一开始没必要把所有日志都打开，单开 `graph_breaks` 更容易看清是谁打断了当前区域。[5]

开头这种调试输出，最直接的办法是挪出热点计算。如果打印只是为了确认输入维度，在调用 compiled 函数之前打就行；如果确实需要看中间 tensor 的值，就得掂量保留这个观察点的代价。用 `torch.compiler.is_compiling()` 跳过打印会改变编译模式下的日志行为，算不上透明的修复。

## 去掉 item，为什么还是断

接下来把打印换成一个依赖数据的分支：

```python
def choose(x):
    score = x.sum().item()
    if score > 0:
        return x + 1
    return x - 1
```

这段代码做了两件事：把 tensor 标量取出来，再让 Python 根据这个值选分支。默认配置下，`.item()` 本身就可能造成 graph break；但就算改成 `if x.sum() > 0`，Python 还是得知道这次求和的结果，才能决定往哪边走。[3]

`if x.shape[0] > 8` 则是另一种情况。shape 是 tensor 的元信息，Dynamo 可以对它做特化或符号推理；而 `x.sum()` 的结果取决于 tensor 里的实际数据。同样是 `if`，问题出在条件依赖的是什么，而不是编译器不支持分支。

如果 `.item()` 取出的标量只参与后续计算，可以试试打开 scalar capture：

```bash
TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 python graph_break_demo.py
```

这个选项只解决标量捕获。如果后面还是拿这个标量做依赖数据的 Python 分支，控制流的问题依然在。[3]

对上面这种加减分支，可以把选择写进 tensor 表达式：

```python
def choose_where(x):
    return torch.where(x.sum() > 0, x + 1, x - 1)
```

注意 Python 会先把调用参数求出来，所以 `x + 1` 和 `x - 1` 两边都会算。这个例子里两边都便宜、也都合法，这么写很自然；但如果两边是不同的大模块，或者某一边会出现除零之类的问题，就不适合直接套 `where`。

需要保留条件执行时，可以用 `torch.cond` 把两个分支都交给编译器：[6]

```python
def choose_cond(x):
    return torch.cond(
        x.sum() > 0,
        lambda t: t + 1,
        lambda t: t - 1,
        (x,),
    )
```

`torch.cond` 会捕获两边的计算，运行时再按条件选一边执行。分支要满足接口约束，比如返回 tensor 的 shape、dtype 等元信息要兼容，也不能随意修改全局状态。而且两个分支都得能被捕获，把原本不支持的代码塞进某个 lambda 里，问题并没有解决。

## 没有 Graph Break，也可能一直编译

换一个没有打印、也没有数据分支的函数，并显式关掉动态 shape：

```python
def pointwise(x):
    return x.sin() + x.cos()


compiled_pointwise = torch.compile(
    pointwise, backend="eager", fullgraph=True, dynamic=False
)
compiled_pointwise(torch.randn(8, 16))
compiled_pointwise(torch.randn(12, 16))
```

这段计算本身能完整捕获。但第二次调用时，输入第一维从 8 变成了 12，第一次编译时做的 shape 假设可能就不成立了。Dynamo 用 guard 检查这些假设：如果已有的缓存版本都对不上，就要重新追踪、编译一个新版本。[5]

把这段单独存成 `recompile_demo.py`，用下面的命令观察：

```bash
TORCH_LOGS="recompiles,guards" python recompile_demo.py
```

`guards` 显示图的使用条件，`recompiles` 显示是哪个条件失败、触发了重编译。Graph break 关心的是一次捕获在哪里被截断，recompile 关心的是下次调用能不能复用已有的结果，所以 `fullgraph=True` 的函数照样可能重编译。

这里特意用了 `dynamic=False`，好把尺寸变化的问题暴露出来。默认的 `dynamic=None` 会在尺寸变化后尝试生成更动态的版本。如果事先知道某个维度会变，也可以提前标记：[7]

```python
x = torch.randn(8, 16)
torch._dynamo.mark_dynamic(x, 0, min=2, max=64)

compiled_dynamic = torch.compile(pointwise, backend="eager", fullgraph=True)
compiled_dynamic(x)
compiled_dynamic(torch.randn(12, 16))
```

`mark_dynamic` 要放在编译函数外、第一次调用之前，表示希望第 0 维在给定范围内变化。它管不了前面的 `print` 或数据依赖分支。另外，复杂代码还可能引入额外的尺寸约束，标了动态维度也不保证所有输入都能复用同一版图。

所以看到吞吐忽高忽低时，我会分开查两件事：热点计算是不是频繁切回 Python，warm-up 之后是不是还在不断编译新版本。不然很容易花时间清掉几个无关的打印，却没发现真正的耗时来自尺寸变化。

## 哪些边界值得留下来

实际模型里，计算前后经常夹杂着预处理、统计、日志和第三方调用。不打算编译的辅助函数，可以明确地划出去：[8]

```python
@torch.compiler.disable
def report_shape(t):
    print("shape:", t.shape)


def step(x):
    y = x.sin()
    report_shape(y)
    return y.cos() + 1


compiled_step = torch.compile(step)  # 使用允许 graph break 的默认模式
```

这样改并没有去掉边界，只是让它更明确；`disable` 默认还会递归禁用辅助函数内部的编译。如果对这个 `step` 打开 `fullgraph=True`，调用 disabled 函数依然会成为障碍。

我的处理顺序是先找高频计算里的断点。比如每个 decode step 都要经过的 Python 调用，就值得细看它是不是打断了连续计算；初始化时只跑一次的辅助逻辑，优先级就低得多。决定挪走某个操作之前，还要确认这样做有没有改变日志时机、状态更新或分支语义。

大型工程里，光看一份终端日志很难把握全局，可以收集 trace 后用 `tlparse` 浏览：[5]

```bash
TORCH_TRACE="/tmp/torch-compile-trace" python train.py
pip install tlparse
tlparse /tmp/torch-compile-trace
```

先定位哪个函数被切开、哪个函数在重复编译，再缩成前面这样的小例子。捕获问题修好后，换回实际的 backend，对比输出和稳态耗时。这样每次改动都能对上一个具体问题，也能看清去掉这个边界到底有没有收益。

## 参考

本文由 2026-06-13 的 Graph Break 阅读笔记整理，接口与机制对照 PyTorch 官方文档核实。文档访问日期：2026-09-19。

[1] PyTorch, [Dynamo Core Concepts](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html).

[2] PyTorch, [Custom Backends](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_custom_backends.html).

[3] PyTorch, [Common Graph Breaks](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html).

[4] PyTorch, [Use fullgraph=True to Identify and Eliminate Graph Breaks](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html).

[5] PyTorch, [torch.compile Troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html).

[6] PyTorch, [torch.cond](https://docs.pytorch.org/docs/stable/generated/torch.cond.html).

[7] PyTorch, [Dynamic Shapes](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html).

[8] PyTorch, [torch.compiler.disable](https://docs.pytorch.org/docs/stable/generated/torch.compiler.disable.html).
