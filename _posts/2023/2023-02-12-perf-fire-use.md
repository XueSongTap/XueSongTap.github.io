---
layout: articles
title: perf和火焰图
tags: memory leak location
---

## perf 环境

保证内核开启CONFIG_PERF_EVENTS

`sysctl -a | grep -i "perf"`命令查看perf是否开启

```shell
$ sysctl -a | grep -i "perf"
kernel.perf_cpu_time_max_percent = 25
kernel.perf_event_max_contexts_per_stack = 8
kernel.perf_event_max_sample_rate = 100000
kernel.perf_event_max_stack = 127
kernel.perf_event_mlock_kb = 516
kernel.perf_event_paranoid = 3
```
## 手动执行过程

### 采样进程的信息
使用进程名进行采样
```shell
procss="xxx"
pid=$(ps | grep $procss | head -n1 | awk '{print $1}')
./perf record -p $pid -ag -- sleep 10
```
这段代码是一个Shell脚本，它的作用是使用perf工具对指定的进程进行性能记录。下面是代码的逐行解释：

- procss="xxx"：将变量procss设置为要查找的进程名或关键字，这里使用"xxx"作为示例。

- pid=$(ps | grep $procss | head -n1 | awk '{print $1}')：通过执行ps命令获取当前系统中所有进程的列表，并通过grep命令过滤出包含关键字procss的行。然后使用head -n1命令获取第一行（即匹配到的第一个进程），最后使用awk '{print $1}'提取出该进程的PID，并将其赋值给变量pid。

- ./perf record -p $pid -ag -- sleep 10：使用perf工具对指定的进程进行性能记录。-p $pid参数指定要记录的进程的PID，-ag参数表示记录所有事件（包括CPU、内存、I/O等），-- sleep 10表示在记录性能数据前等待10秒钟。

### 处理采样结果

```shell
./perf script -i perf.data > perf.unfold
```

保存为perf.unfold
### 绘制火焰图
```shell
git clone https://github.com/brendangregg/FlameGraph.git
cd FlameGraph
./FlameGraph/stackcollapse-perf.pl perf.unfold > perf.folded
./FlameGraph/flamegraph.pl perf.folded > perf.svg
```

## 可能的异常处理
生成的svg图没有符号，可能是perf工具编译时没有用 libelf 和 libdw


https://stackoverflow.com/questions/58928506/linux-perf-not-resolving-symbols

## 参考

https://medium.com/coccoc-engineering-blog/
things-you-should-know-to-begin-playing-with-linux-tracing-tools-part-i-x-225aae1aaf13

https://www.brendangregg.com/perf.html


https://github.com/brendangregg/FlameGraph

https://docs.rockylinux.org/de/gemstones/view_kernel_conf/