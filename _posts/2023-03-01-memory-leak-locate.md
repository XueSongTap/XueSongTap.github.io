---
layout: articles
title: 内存泄漏定位
tags: memory leak location
---


## 1 初步确认内存泄漏

在进行长期稳定的测试的时候，定期查看内存使用情况，初步确定是否存在，一般需要较长时间的稳定测试

### 1.1 linux free 命令

https://blog.csdn.net/qq_35462323/article/details/105724468

执行free 命令可以得到系统当前内存的情况

此时所有的数据默认都是 KB，如果想要得到MB, 则输入free -m

```shell
#free -m
             total       used       free     shared    buffers     cached
Mem:         16081      15285        796          0        154       6901
-/+ buffers/cache:       8229       7852
Swap:            0          0          0
```

对于输出的第一行，我们先纵向看，可以发现除去第一列，后面一共有六列，分别为total、used、free、shared、buffers、cached

- total：物理内存大小，就是机器实际的内存
- used：已使用的内存大小，这个值包括了 cached 和 应用程序实际使用的内存
- free：未被使用的内存大小
- shared：共享内存大小，是进程间通信的一种方式
- buffers：被缓冲区占用的内存大小
- cached：被缓存占用的内存大小
对于Mem对应的行：total = used + free

下面一行，代表应用程序实际使用的内存：

- 前一个值表示 - buffers/cached，即 used - buffers/cached，表示应用程序实际使用的内存。
- 后一个值表示 + buffers/cached，即 free + buffers/cached，表示理论上都可以被使用的内存。
这两个值相加=total
所以说我们在说机器的内存使用率的时候，简单的使用used/total，并不准确，应为total不代表应用程序实际使用过的内存，只有减去buffers/cached才准确，所以计算公式应该为：（used -buffers-cached）/total，结果则为应用程序实际使用内存的内存使用率。

用shell命令可以这么写：
```shell
free |grep Mem|awk -F ' ' '{printf("%.2f",($3-$7-$6)/$2)}'
```
另外执行 `free -h`

```shell
$ free -h
              total        used        free      shared  buff/cache   available
Mem:           1.8G        366M        514M         18M        1.0G        1.3G
Swap:            0B          0B          0B
```

### 1.2 top命令

使用top指令可以查看当前系统的内存&CPU占用情况

- 按m，可以按内存占用高低显示CPU占用率

- 按s，或使用top -m可以显示内存占用


通常关注以下值：
- RSS：表示该进程分配的内存大小，不包括swap，包括共享库（实际加载到内存的部分）、堆、栈
- VSZ：进程已分配的虚拟内存大小，包括swap、共享库、堆、栈


### 1.3 cat /proc/xxx/status

https://blog.csdn.net/weixin_40584007/article/details/88847745

执行ps可获取进程运行的pid，假设为xxx

执行cat /proc/xxx/status 可获取进程当前的资源占用状态，包括内存

通常关注以下值：

- VmSize：虚拟内存使用量
- VmRSS：物理内存使用量

## 2 详细定位

### tcmalloc 定位

tcmalloc分析需要符号表，请使用-g参数编译程序


tcmalloc本质是替换libc库中的malloc、new等内存相关函数。


### valgrind 定位


## 3 脚本分析

上位机 ubuntu perf 进行分析