---
layout: article
title: c++ 分支预测
tags: c++ 预测
---

## 背景知识-流水线

现代CPU为了提高执行指令执行的吞吐量，使用了流水线技术，它将每条指令分解为多步，让不同指令的各步操作重叠，从而实现若干条指令并行处理。在流水线中，一条指令的生命周期可能包括：

1. 取指：将指令从存储器中读取出来，放入指令缓冲区中。

2. 译码：对取出来的指令进行翻译

3. 执行：知晓了指令内容，便可使用CPU中对应的计算单元执行该指令

4. 访存：将数据从存储器读出，或写入存储器

5. 写回：将指令的执行结果写回到通用寄存器组

流水线技术无法提升CPU执行单条指令的性能，但是可以通过相邻指令的并行化提高整体执行指令的吞吐量

## likely unlikely
```cpp
#define likely(x) __builtin_expect(!!(x), 1) 
#define unlikely(x) __builtin_expect(!!(x), 0)
```

c++20 成为关键字

## 测试
分支预测一个，使用likely
```cpp
#define likely(x) __builtin_expect(!!(x), 1) 
#define unlikely(x) __builtin_expect(!!(x), 0)

#include <iostream>
#include <cstdlib>
#include <chrono>

int run_likely() {
    int res = 0; 
    for (int i = 0; i < 1e6; ++ i) {
        int x = std::rand() % 100;

        if (likely(x < 90)) {  // likely(x < 50)
            res ++;
        }
    }
    return res;
}

int run() {
    int res = 0; 
    for (int i = 0; i < 1e6; ++ i) {
        int x = std::rand() % 100;

        if (x < 90) { 
            res ++;
        }
    }
    return res;
}


void benchmark_likely() {

    std::cout << "Benchmark likely =========================" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    int tmp = run_likely();
    auto end = std::chrono::high_resolution_clock::now();

    // 计算执行时间
    std::chrono::duration<double> duration = end - start;

    // 输出执行时间（以秒为单位）
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

}
void benchmark() {

    std::cout << "Benchmark=========================" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    int tmp = run();
    auto end = std::chrono::high_resolution_clock::now();

    // 计算执行时间
    std::chrono::duration<double> duration = end - start;

    // 输出执行时间（以秒为单位）
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

}


int main() {
    benchmark();
    benchmark_likely();
    return 0;
}

```
test1:
```shell
Benchmark=========================
Execution time: 0.0121381 seconds
Benchmark likely =========================
Execution time: 0.0119101 seconds
```
test2:
```shell
Benchmark=========================
Execution time: 0.0200414 seconds
Benchmark likely =========================
Execution time: 0.0192134 seconds
```

likely 可以显著降低用时，具体细节测试可以考虑用perf进行

## 参考文献

https://www.cnblogs.com/qiangz/p/17088276.html

http://irootlee.com/juicer_branch_predictor4/

https://zhuanlan.zhihu.com/p/48145176