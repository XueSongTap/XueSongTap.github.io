---
layout: article
title: 在线 C++ 性能测试：Quick Bench 工具
tags: cpp benchmark assembly 
---


## Quick Bench 简介

[Quick Bench](https://quick-bench.com/) 是一个强大的在线 C++ 基准测试工具，让开发者能够快速比较不同代码实现的性能差异。无需复杂的环境配置，只需在浏览器中编写代码，就能获得精确的性能测试结果，使性能优化工作变得更加高效直观。


## 主要特点

- **多编译器支持**：可选择不同版本的 GCC、Clang 等编译器
- **C++ 标准灵活选择**：支持从 C++11 到最新标准
- **自定义编译选项**：可添加特定的编译标志和优化级别
- **精确的性能比较**：自动计算 CPU time 与 noop time 比率，消除系统噪声
- **结果可视化**：直观图表展示不同实现的性能差异
- **代码共享**：生成唯一链接，方便分享结果与讨论
- **汇编代码查看**：可检查生成的汇编代码，深入分析性能问题


## 本地部署

如果需要在本地环境运行 Quick Bench，可以通过 Fred Tingaud 开发的 [bench-runner](https://github.com/FredTingaud/bench-runner) Docker 容器实现：

1. 克隆仓库：`git clone https://github.com/FredTingaud/bench-runner.git`
2. 启动 Quick Bench：`./quick-bench`
3. 启动 Build Bench：`./build-bench`

## 示例

### 案例1：vector vs. list 性能对比

下面是一个比较 `std::vector` 和 `std::list` 在不同场景下性能差异的例子：

```cpp
static void VectorPushBack(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<int> v;
    for (int i = 0; i < 1000; ++i)
      v.push_back(i);
    benchmark::DoNotOptimize(v.data());
  }
}
BENCHMARK(VectorPushBack);

static void ListPushBack(benchmark::State& state) {
  for (auto _ : state) {
    std::list<int> l;
    for (int i = 0; i < 1000; ++i)
      l.push_back(i);
    benchmark::DoNotOptimize(&l);
  }
}
BENCHMARK(ListPushBack);
```
![vector-list-push-benchmark](/img/240517/vector-list-push-benchmark.png)

### 案例2：字符串连接方法比较

```cpp
static void StringPlus(benchmark::State& state) {
  for (auto _ : state) {
    std::string result;
    for (int i = 0; i < 100; ++i)
      result = result + "x";
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(StringPlus);

static void StringAppend(benchmark::State& state) {
  for (auto _ : state) {
    std::string result;
    for (int i = 0; i < 100; ++i)
      result.append("x");
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(StringAppend);

static void StringBuilder(benchmark::State& state) {
  for (auto _ : state) {
    std::string result;
    result.reserve(100);
    for (int i = 0; i < 100; ++i)
      result.append("x");
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(StringBuilder);
```

![string-append-benchmark](img/240517/string-append-benchmark.png)


## 使用技巧

### 1. 使用 DoNotOptimize 和 ClobberMemory

为避免编译器过度优化掉基准测试代码，可使用：
```cpp
benchmark::DoNotOptimize(var); // 确保变量被读取
benchmark::ClobberMemory(); // 防止指令重排
```

### 2. 定制测试参数


```cpp
BENCHMARK(BenchmarkName)->Arg(8)->Arg(64)->Arg(512);
BENCHMARK(BenchmarkName)->Range(8, 8<<10);
```


### 3. 多线程

```cpp
BENCHMARK(BenchmarkName)->Threads(2)->Threads(4)->Threads(8);
```