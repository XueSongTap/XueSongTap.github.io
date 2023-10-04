---
layout: articles
title: c++ 计时器（支持多线程）
tags: c++ timer thread_safe
---

本项目能够对指定代码块或函数等进行计时，并利用Chrome tracing进行可视化

将InstrumentorTimer.h和InstrumentorMacro.h（可选的，一些宏定义）正确引入后。

```cpp
Instrumentor::BeginSession("SessionName");               // Begin session 
{
    InstrumentationTimer timer("Profiled Scope Name");   // Place code like this in scopes you'd like to include in profiling
    // Code Blocks
    // timer.Stop();                                     // (Optional) Stop timing manually, timer's destructor will call this function automatically
}
// Instrumentor::EndSession();                           // (Optional) End Session manually, Instrumentor's destuctor will call this function automatically
```


```cpp
int Fibonacci(int x) {
    std::string name = std::string("Fibonacci ") + std::to_string(x);
    InstrumentationTimer timer(name.c_str());
    // PROFILE_SCOPE(name.c_str());     // Available only when include header file 'InstrumentorMacro.h'

    if (x < 3) return 1;
    std::cout << "not finished" << std::endl;
    int part1 = Fibonacci(x - 1);
    int part2 = Fibonacci(x - 2);
    return part1 + part2;
}

int main() {
    Instrumentor::BeginSession("Benchmark");
    Fibonacci(5);
}
```

计时器统计时长的类型为 std::chrono::microseconds，即本计时器对小于1微秒的时间开销不敏感。



https://github.com/XueSongTap/InstrumentorTimer