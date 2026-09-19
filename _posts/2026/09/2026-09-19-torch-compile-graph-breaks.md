---
layout: article
title: 从一行 print 看 torch.compile 的 Graph Break
tags: LLM PyTorch torch.compile
---

> 前置阅读：[torch.compile 里 Inductor 是怎么把编译结果包进 CUDA Graph]({% post_url 2026/05/2026-05-19-inductor-cudagraph-analysis %})。这篇往前看一步：Inductor 收到的图，是 Dynamo 怎么捕获出来的。

给一个函数加上 `torch.compile` 后，代码能正常跑完，很容易以为整个函数已经交给编译器了。可以先看这段代码：

```python
import torch


def f(x):
    y = x.sin()
    print("intermediate shape:", y.shape)
    return y.cos() + 1


compiled = torch.compile(f)
compiled(torch.randn(8, 16))
```

中间只插了一行打印。默认配置下，Dynamo 通常会在这里切图：先捕获 `sin`，回到 Python 执行 `print`，再从后面的 `cos` 和加法继续捕获。程序仍然能运行，但后端看到的计算已经分成了两段。[1][3]

这就是 Graph Break。下面沿着这个例子看它发生在哪里，再看实际排查时经常混在一起的 `.item()`、动态 shape 和重编译。文中代码用于说明和复现，未在本文整理环境执行；具体报错文字以本机 PyTorch 版本为准。

## 先把 Dynamo 交出去的图打印出来

暂时不用 Inductor，可以写一个很小的 backend：

```python
def inspect_backend(gm, example_inputs):
    print("=== Dynamo captured ===")
    print(gm.code)
    return gm.forward


compiled = torch.compile(f, backend=inspect_backend)
compiled(torch.randn(8, 16))
```

Dynamo 把捕获到的 FX `GraphModule` 和示例输入交给 backend，backend 返回可执行函数。这里直接返回 `gm.forward`，方便观察捕获结果，省去了后面的代码生成。[2] backend 里的打印发生在编译回调中，与 `f` 内正在被追踪的 `print` 是两个位置。

对开头的例子，可以把执行过程简化为下面的示意，变量名和实际生成代码不必一致：

```text
x → [图 1：sin] → y → Python print → [图 2：cos、add] → output
```

`y` 需要从前一段计算传到后一段。Dynamo 处理这个边界时，还要保留恢复执行所需的 Python 状态。于是一个 Python 函数可以同时包含编译区域和 eager 区域。[1]

把 `print` 删掉，三个 tensor 运算就有机会一起交给后端。这样 Inductor 才能在更大的范围内安排 fusion；切图后，两个独立编译区域之间的这类优化机会就丢掉了。具体多出多少 kernel、耗时多少，要看后端生成结果，不能由图的数量直接换算。

这也接上了前一篇的 CUDA Graph：FX graph 描述编译器捕获的计算，CUDA Graph 处理 GPU 工作的捕获与重放。排查 Dynamo 时看到“两张图”，并不能直接读成“两次 CUDA Graph replay”。

## 用 fullgraph 把断点暴露出来

默认 `fullgraph=False` 会允许上述切图和恢复过程。调试时可以把同一个函数换成：

```python
strict_f = torch.compile(f, backend="eager", fullgraph=True)
strict_f(torch.randn(8, 16))
```

`backend="eager"` 仍然经过 Dynamo，适合先排查捕获问题；`fullgraph=True` 要求这次捕获形成完整的 FX graph，遇到 graph break 就报错。[4] 此时应该顺着错误里的用户代码位置找回那行 `print`。

这里的 fullgraph 约束的是捕获范围。后端仍然可以为一张 FX graph 生成多个 kernel，也不会因为打开这个选项就自动得到一个 CUDA Graph。

如果希望程序继续跑，同时查看切图位置，可以把第一个例子保存为 `graph_break_demo.py`，执行：

```bash
TORCH_LOGS="graph_breaks" python graph_break_demo.py
```

先看用户代码的文件和行号，再看原因。通常没必要一开始就把所有日志打开；单独的 `graph_breaks` 更容易看清是谁打断了当前区域。[5]

对于开头这种调试输出，最直接的处理是移出热点计算。如果输出本来只是用来确认输入维度，在调用 compiled 函数前打印即可。如果确实需要中间 tensor 的值，就要考虑保留这个观察点的成本。用 `torch.compiler.is_compiling()` 跳过打印会改变编译模式下的日志行为，不能当成透明修复。

## 去掉 item，为什么还是断

接下来把打印换成一个数据分支：

```python
def choose(x):
    score = x.sum().item()
    if score > 0:
        return x + 1
    return x - 1
```

这里有两件事：把 tensor 标量提取出来，以及让 Python 根据这个值选分支。默认配置下，`.item()` 本身就可能造成 graph break；但即使写成 `if x.sum() > 0`，Python 仍然需要知道这次求和的结果，才能决定往哪边执行。[3]

这和 `if x.shape[0] > 8` 不同。shape 是 tensor 的元信息，Dynamo 可以对它做特化或符号推理；`x.sum()` 的结果取决于 tensor 里的实际数据。不能把两种 `if` 都归为“编译器不支持分支”。

如果 `.item()` 取出的标量只参与后续计算，可以尝试开启 scalar capture：

```bash
TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1 python graph_break_demo.py
```

这个选项解决的是标量捕获。后面若仍然拿标量做数据依赖的 Python 分支，控制流问题还在。[3]

对于上面的加减分支，可以把选择写在 tensor 表达式里：

```python
def choose_where(x):
    return torch.where(x.sum() > 0, x + 1, x - 1)
```

注意 Python 会先求出调用参数，因而这里的 `x + 1` 和 `x - 1` 都会计算。这个例子两边很便宜，也都合法，所以比较自然。若两边是不同的大模块，或者某一边会出现除零等问题，直接套 `where` 就不合适。

需要保留条件执行时，可以用 `torch.cond` 把两个分支交给编译器：[6]

```python
def choose_cond(x):
    return torch.cond(
        x.sum() > 0,
        lambda t: t + 1,
        lambda t: t - 1,
        (x,),
    )
```

它会捕获两边的计算，但运行时按条件选择分支。分支需要满足接口约束，例如返回 tensor 的 shape、dtype 等元信息要兼容，也不能随意修改全局状态。两个分支都要能被捕获，不能把原先不支持的代码藏进其中一个 lambda 就算完成改写。

## 没有 Graph Break，也可能一直编译

换一个没有打印、没有数据分支的函数，并明确关闭动态 shape：

```python
def pointwise(x):
    return x.sin() + x.cos()


compiled_pointwise = torch.compile(
    pointwise, backend="eager", fullgraph=True, dynamic=False
)
compiled_pointwise(torch.randn(8, 16))
compiled_pointwise(torch.randn(12, 16))
```

这段计算本身可以完整捕获。第二次调用时，输入第一维从 8 变成了 12，第一次编译的 shape 假设却可能已经不成立。Dynamo 用 guard 检查这些假设：如果已有缓存版本都不匹配，就需要重新追踪、编译新版本。[5]

把这段单独保存为 `recompile_demo.py`，观察：

```bash
TORCH_LOGS="recompiles,guards" python recompile_demo.py
```

`guards` 展示图的使用条件，`recompiles` 展示哪个条件失败并触发了重新编译。Graph break 关注一次捕获在什么地方被截断，recompile 关注下次调用还能不能复用已有结果。`fullgraph=True` 的函数也完全可能重编译。

这里特意用了 `dynamic=False`，方便把尺寸变化的问题暴露出来。默认的 `dynamic=None` 会尝试在尺寸变化后生成更动态的版本。已经知道某个维度会变化时，也可以提前标记：[7]

```python
x = torch.randn(8, 16)
torch._dynamo.mark_dynamic(x, 0, min=2, max=64)

compiled_dynamic = torch.compile(pointwise, backend="eager", fullgraph=True)
compiled_dynamic(x)
compiled_dynamic(torch.randn(12, 16))
```

`mark_dynamic` 放在编译函数外、第一次调用前，表示这里希望第 0 维在一定范围内变化。它不负责解决前面的 `print` 或数据依赖分支。复杂代码还可能引入额外的尺寸约束，动态维度也不能保证所有输入都复用同一版图。

所以看到吞吐忽高忽低时，我会先分开查两个问题：热点计算是否被频繁切回 Python，warm-up 之后是否还在不断产生新的编译版本。否则很容易花时间清理了几个无关的打印，却没有发现真正的耗时来自尺寸变化。

## 哪些边界值得留下来

实际模型里，计算前后经常混着预处理、统计、日志和第三方调用。可以明确把不准备编译的辅助函数划出去：[8]

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

这个改法保留了边界，并让边界更明确；`disable` 默认也会递归禁用辅助函数内部的编译。若对这个 `step` 开启 `fullgraph=True`，调用 disabled 函数仍然会成为障碍。

我的处理顺序会是先找高频计算里的断点。例如每个 decode step 都要经过的 Python 调用，值得进一步看它是否妨碍了连续计算；初始化时只执行一次的辅助逻辑，优先级就低很多。决定搬走一个操作前，还要确认是否改变了日志时机、状态更新或分支语义。

大型工程里，单份终端日志不容易看清整体，可以收集 trace 后用 `tlparse` 浏览：[5]

```bash
TORCH_TRACE="/tmp/torch-compile-trace" python train.py
pip install tlparse
tlparse /tmp/torch-compile-trace
```

先定位哪段函数被切开、哪段函数重复编译，再缩成前面这样的小例子。修好捕获问题后，再换回实际 backend，对比输出和稳态耗时。这样每次改动都能对应到一个具体问题，也能看清消除这个边界到底有没有收益。

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
