---
layout: articles
title: 性能优化工具
tags: c++ 优化
---



## perf工具

### 火焰图分析
![fire](/img/0315/fire.png)

y轴：调用栈深度，火焰高度表示调用栈的深度。

x轴：函数采样数，宽度表示函数被采样到的次数，即消耗的资源多少。

#### 常用的分析流程

1. 检查线程采样点占比，火焰图最底层即为特定线程名的资源占比，让同名线程绘制到一起

2. 检查火焰图的平顶
  - 函数执行CPU消耗较高，如RLECompress函数，存在大量计算操作
  - 函数执行次数较多，比如线程数量较多时，线程的频繁切换会导致schedule()消耗较高

### 常用的指令

抓取所有进程包含主进程的火焰图

抓取火焰图前执行  perf top -a 抓取全局火焰图

### 在线调试
- 监控全局CPU函数热点：./perf top -a -F 1000
- 监控全局CPU函数热点，并打印所有调用栈（打印量非常多）：./perf top -a -F 1000 -g
- 监控进程CPU函数热点：./perf top -a -F 1000 -p PID
- 监控进程branch-miss：./perf top -aK -e branch-misses -F 1000 -p PID
- 监控进程cache-miss：./perf top -aK -e cache-misses -F 1000 -p PID
## 内存分析
内存泄漏情况，

heap上的虚拟内存分配情况

进程内存占用情况，但是一般只能精确到模块，无法精确到函数
## 执行耗时
针对主线程的执行耗时进行统计


## 日志分析

感知日志分析，根据log提取 cpu占用，fps delay等信息



## 参考文献

https://lgl88911.github.io/2020/03/19/Perf%E5%92%8C%E7%81%AB%E7%84%B0%E5%9B%BE/


https://www.brendangregg.com/overview.html