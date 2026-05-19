---
layout: article
title: torch.compile 里 Inductor 是怎么把编译结果包进 CUDA Graph
tags: PyTorch Inductor CUDAGraph
---

## 1. Inductor 后端注册与调用链

### 1.1 注册点

`torch/_dynamo/backends/inductor.py` — 只有 32 行，是字符串 `"inductor"` 的绑定入口：

```python
from torch._dynamo import register_backend

@register_backend   # 把 "inductor" 注册到 _COMPILER_FNS 字典
def inductor(*args, **kwargs):
    from torch._inductor.compile_fx import compile_fx  # 懒加载
    return compile_fx(*args, **kwargs)
```

### 1.2 查找机制

`torch/_dynamo/backends/registry.py`：

```python
_COMPILER_FNS: dict[str, CompilerFn] = {}
_default_backend: str | CompilerFn = "inductor"  # 默认后端就是 inductor

def lookup_backend(compiler_fn: str | CompilerFn) -> CompilerFn:
    if isinstance(compiler_fn, str):
        compiler_fn = _COMPILER_FNS[compiler_fn]  # "inductor" → 上面注册的函数
    return compiler_fn
```

### 1.3 完整调用链

```
torch.compile(model, backend="inductor")
  │
  ▼
torch/_dynamo/eval_frame.py
  get_compiler_fn("inductor") → lookup_backend("inductor") → _COMPILER_FNS["inductor"]
  │
  ▼
torch/_dynamo/backends/inductor.py :: inductor(gm, example_inputs)
  └─ compile_fx(gm, example_inputs)
  │
  ▼
torch/_inductor/compile_fx.py :: compile_fx()
  └─ 定义 fw_compiler / bw_compiler / inference_compiler
  └─ aot_autograd(fw_compiler, bw_compiler, decompositions, partition_fn)
       │
       ├─ (3a) 算子分解（decompositions，如 F.linear → mm + add）
       ├─ (3b) min-cut 分区：切出 forward graph + backward graph
       └─ (3c) fw_compiler(forward_gm) / bw_compiler(backward_gm)
                │
                ▼
             compile_fx_inner(gm)
                │
                ▼
             fx_codegen_and_compile(gm)
                │
                ├─ GraphLowering(gm).run()      ← FX op → Inductor IR
                ├─ graph.compile_to_fn()        ← Inductor IR → Triton/C++ kernel
                └─ CompiledFxGraph.post_compile()
                     └─ cudagraph_post_compile() ← 包 CUDA Graph
```

### 1.4 Inductor 核心：GraphLowering

`compile_fx_inner` 里最关键的两步（`compile_fx.py` ~1486）：

```python
graph = GraphLowering(gm, example_inputs=example_inputs, ...)

with V.set_graph_handler(graph):
    graph.run(*example_inputs)     # FX node → Inductor IR (TensorBox, Pointwise, Reduction...)

with dynamo_timed("GraphLowering.compile_to_fn"):
    compiled_fn = graph.compile_to_fn()  # Inductor IR → Triton kernel → 编译
```

`GraphLowering.run()` 遍历 FX graph 每个节点，调用 `torch/_inductor/lowering.py` 里注册的
lowering 函数，把 `torch.ops.aten.xxx` 转成 Inductor IR 节点。

---

## 2. Inductor 与 CUDA Graph 的集成

### 2.1 开关

```python
# config.py，triton 子配置
cudagraphs = os.environ.get("TORCHINDUCTOR_CUDAGRAPHS") == "1"  # 默认关闭
cudagraph_trees = True   # 树形内存池共享，默认开启

# 开启方式
torch.compile(model, options={"triton.cudagraphs": True})
# 或
TORCHINDUCTOR_CUDAGRAPHS=1 python train.py
```

`compile_fx_inner` 入口处创建 `BoxedBool`（可变容器，允许下游原地 disable）：

```python
# compile_fx.py ~900
if graph_kwargs.get("cudagraphs") is None:
    graph_kwargs["cudagraphs"] = BoxedBool(config.triton.cudagraphs)
```

### 2.2 编译期检查（哪些情况会 skip）

在 `fx_codegen_and_compile` 阶段，结果存入 `CompiledFxGraph.disabled_cudagraphs_reason`，
随编译产物一起缓存：

| 检查项 | 触发条件 |
|--------|---------|
| 动态 shape | `cudagraph_skip_dynamic_graphs=True` 且有 SymInt 输入 |
| 不兼容 op | `get_first_incompatible_cudagraph_node(gm)` 返回非空 |
| CPU/多设备 | `check_lowering_disable_cudagraph(device_node_mapping)` |
| 输入 mutation | 非 cudagraph 管理的 tensor 被 mutate |

### 2.3 `post_compile` — 实际包装发生的地方

`output_code.py :: CompiledFxGraph.post_compile()`：

```python
if cudagraphs:
    if self.disabled_cudagraphs_reason:
        BoxedBool.disable(cudagraphs)   # 有 skip 原因，放弃
    else:
        if config.graph_partition and policy is None:
            cudagraph_partition_post_compile(...)  # 按 partition 分别包装
        else:
            cudagraph_post_compile(...)            # 整图包装
```

`cudagraph_post_compile` 最终替换 `compiled_graph.current_callable`：

```python
compiled_graph.current_callable = cudagraphify(
    current_callable,
    static_input_idxs=...,
    device_index=...,
    is_backward=...,
    is_inference=...,
    ...
)
```

### 2.4 `cudagraphify` — 两条路径的分叉

`compile_fx.py` ~1883：

```python
def cudagraphify(model, static_input_idxs=(), *, device_index, ...):
    if config.triton.cudagraph_trees:
        cudagraphify_fn = functools.partial(new_cudagraphify_impl, ...)  # 树形版
    else:
        cudagraphify_fn = cudagraphify_impl   # 简单版

    compiled_fn = None

    def run(new_inputs):   # 懒惰录制：第一次真正调用时才录制
        nonlocal compiled_fn
        if compiled_fn is None:
            compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)
        return compiled_fn(new_inputs)

    return run
```

---

## 3. 两种 CUDA Graph 实现

### 3.1 简单版 `cudagraphify_impl`（`cudagraph_trees=False`）

`compile_fx.py` ~1951，无内存池共享：

```python
def cudagraphify_impl(model, inputs, static_input_idxs=()):
    # 1. 分配 static_inputs
    static_inputs = [static_input(x) if idx not in static_input_idxs else x.detach()
                     for idx, x in enumerate(inputs)]

    # 2. warmup（side stream）
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        model(list(static_inputs))
    torch.cuda.synchronize()

    # 3. 录制
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        static_outputs = model(list(static_inputs))

    # 4. replay 闭包
    def run(new_inputs):
        for idx in copy_indices:
            index_expanded_dims_and_copy_(static_inputs[idx], new_inputs[idx], ...)
        new_inputs.clear()
        graph.replay()
        return static_outputs

    return run
```

### 3.2 树形版（`cudagraph_trees=True`，默认）

`cudagraph_trees.py :: cudagraphify_impl`，按 int 输入（动态 shape）做 key 缓存：

```python
def cudagraphify_impl(model, inputs, static_input_idxs, *args, **kwargs):
    fn_cache: dict[tuple[int,...], Callable] = {}

    def deferred_cudagraphify(inputs):
        int_key = get_ints(inputs)
        fn = fn_cache.get(int_key)
        if fn is not None:
            return fn(inputs)
        # 第一次：录制，结果缓存
        fn, out = cudagraphify(model, inputs, ...)  # → CUDAGraphTreeManager.add_function()
        fn_cache[int_key] = fn
        return out

    return deferred_cudagraphify
```

---

## 4. CUDAGraphTreeManager

每个 device 一个实例，存在 thread-local 的 `TreeManagerContainer` 里。

### 4.1 初始化：创建共享内存池

```python
def __init__(self, device_index):
    with torch.cuda.device(device_index):
        torch.cuda.synchronize()
        self.stream = torch.cuda.Stream()
        self.graph = torch.cuda.CUDAGraph()
        self.cuda_graphs_thread_pool = torch.cuda.graph_pool_handle()  # 共享内存池

        # 用空图激活 pool，保持其存活
        with torch.cuda.graph(self.graph, pool=self.cuda_graphs_thread_pool, ...):
            pass
```

### 4.2 `_run()` 热路径状态机

```
_run(new_inputs, function_id)
  │
  ├─ 还在 recording/warmup 中？→ try_end_curr_recording/warmup
  │
  ├─ 检测到非 cudagraph 管理的 mutation？→ 直接 eager 执行
  │
  ├─ 还没 warmup？→ run_eager() → CUDAWarmupNode（在 pool 内存里跑）
  │
  ├─ 已 warmup，扫描 child_nodes[function_id]：
  │   ├─ check_invariants() == SUCCESS → execute_node() → graph.replay()
  │   └─ 不匹配 → 计入 re-record 次数
  │       └─ 超过 cudagraph_unexpected_rerecord_limit(128) → 永久 fallback eager
  │
  └─ 没有匹配的 child → record_function() → 新建 CUDAGraphNode
```

### 4.3 节点类型

| 类型 | 作用 |
|------|------|
| `CUDAWarmupNode` | 第一次执行（warmup），在 pool 内存里 eager 跑，建立 allocator 状态 |
| `CUDAGraphNode` | 录制一次 CUDA Graph，后续 replay |

---

## 5. CUDAGraphNode — 实际录制

构造函数里直接调用 `_record()`，使用共享内存池：

```python
def _record(self, model, inputs):
    with (
        preserve_rng_state(),
        torch.cuda.device(self.device),
        clear_cublas_manager(),
        torch.cuda.graph(
            self.graph,
            stream=self.stream,
            pool=self.cuda_graphs_pool,           # 共享内存池
            capture_error_mode="thread_local",
        ),
        CUDAGraphCaptureControlFlowOpDispatchMode(),
        get_history_recording(),
    ):
        static_outputs = model(inputs)
    return static_outputs
```

`run()` 方法（replay 路径）：

```python
def run(self, new_inputs):
    self.check_static_inputs_are_stable(new_inputs)   # 检查 static ptr 没变
    self._copy_inputs_and_remove_from_src(            # 新 input 拷到 static slot
        self.reconstructed_inputs, new_inputs
    )
    self.run_graph()                                   # graph.replay()
    return self.reconstruct_outputs()                  # 从 metadata 重建 tensor
```

---

## 6. 前向/后向协调

`cudagraph_post_compile` 对 backward 有特殊处理（`output_code.py`）：

```python
if is_backward and config.triton.cudagraph_trees:
    def compiled_artifact(new_inputs):
        manager.set_to_running_backward()  # 通知 manager 进入 backward 模式
        return compiled_graph_callable(new_inputs)
    compiled_graph.current_callable = compiled_artifact
```

`set_to_running_backward()` 把 `running_forwards_with_pending_backwards` 置为 False，
允许 manager 在 backward 完成后开始新的 generation（新的训练迭代）。

---

## 7. 关键配置

```python
# 主开关（默认关）
torch.compile(model, options={"triton.cudagraphs": True})

# 树形 pool 共享（默认开）
torch.compile(model, options={"triton.cudagraph_trees": True})

# 动态 shape 下跳过（默认不跳过，会为每个 shape 录制一个图）
torch.compile(model, options={"triton.cudagraph_skip_dynamic_graphs": True})

# 最多允许多少次 re-record（默认 128）
torch.compile(model, options={"triton.cudagraph_unexpected_rerecord_limit": 128})

# 强制要求 cudagraph，不兼容时报错而非 skip
torch.compile(model, options={"triton.cudagraph_or_error": True})
```

---

## 8. 各层职责总结

| 层 | 文件 | 职责 |
|----|------|------|
| 注册 | `torch/_dynamo/backends/inductor.py` | 把字符串 `"inductor"` 绑定到 `compile_fx` |
| 查找 | `torch/_dynamo/backends/registry.py` | `lookup_backend("inductor")` |
| 调度 | `torch/_dynamo/eval_frame.py` | Dynamo 追踪后把 FX graph 交给 backend |
| AOT 层 | `torch/_inductor/compile_fx.py :: compile_fx` | 调 AOT Autograd，拆 fw/bw |
| Inductor 编译 | `compile_fx_inner` + `GraphLowering` | FX op → Inductor IR → Triton kernel |
| CUDA Graph 开关 | `compile_fx_inner` | 创建 `BoxedBool(config.triton.cudagraphs)` |
| CUDA Graph 包装 | `CompiledFxGraph.post_compile` | 替换 `current_callable` |
| 分发 | `compile_fx.cudagraphify` | 懒惰录制，分发到 trees 或简单版 |
| 内存池管理 | `CUDAGraphTreeManager` | 共享 pool，维护 warmup/record/replay 状态机 |
| 单次录制 | `CUDAGraphNode` | 持有 `torch.cuda.CUDAGraph`，管理 tensor 生命周期 |
