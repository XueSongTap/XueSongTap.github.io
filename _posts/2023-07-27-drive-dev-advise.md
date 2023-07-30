---
layout: articles
title: C++ 开发推荐tips
tags: c++ guide
---


为了减少不确定性，固定时延，避免内存使用的不确定性，以下是一些推荐做法。

## 1.mlockall() 锁定内存
调用mlockall()在初始化时锁定进程的虚拟内存,包括代码段、数据段、堆和栈,防止不确定的内存回收。


内存回收是可能会把只读映射的代码段区域回收掉，用到时再触发异常，重新从文件系统加载，重而引入了不确定的延迟。
```shell
mlockall(MCL_CURRENT | MCL_FUTURE )
```
这可以防止代码段意外回收后重新加载造成的不确定延迟。




注：
- 对于链接了多个库的大型应用,可以只锁定关键代码区域。
- 另外一个手段，初始化时malloc一块大内存，mlock住，再释放掉，后面再有malloc不再产生缺页中断。

## 2.malloc

libc中的malloc函数，会通过sbrk向内核申请内存，也可能通过mmap直接映射来分配内存。

在应用开头，设置相关参数，来防止反复向内核的分配释放，内核的内存管理有快路径和慢路径，慢路径可能触发内存回收，时间不确定 。
```shell
mallopt (M_TRIM_THRESHOLD, -1);
```
这个调用关闭向内核释放的过程，应用内存一旦从内核分配出来，不再释放回内核，防止内存使用的不确定性和额外延迟。

如果各应用使用的最大内存有可能超过系统所能提供的，可以在测试阶段暴露出来，避免在实际产品中在某个场景下崩溃。

```cpp
mallopt (M_MMAP_MAX, 0)
```
这个调用可能触发mmap分配方式禁调，这样malloc都经过sbrk系统调用，配合上一条，固定内存


注意
- 大块内存和小块内存交错分配释放，有可能会造成大块内存不断被拆分，内存碎片化后，重而不断向系统申请新的内存。所以驾驶应用应该尽量自建内存池管理
- 最优的做法是在初始化是把可能用到的内存池按最大可能性都分配好，后续用空闲队列和使用队列管理，工作循环中如果用内存获取和释放就是确定时间了


## 3.Thead

推荐在应用初始化时提前创建所有需要的线程,不要动态创建和销毁线程。

创建线程后立即启动线程运行,并锁定其内存,防止释放。

线程数宜与 CPU 核数匹配,不要超过 CPU 核数太多,否则会增加不必要的 CPU 切换开销。

所有线程初始化后,可以等待在信号量上,准备处理任务。

这样可以减少线程创建/销毁带来的不确定性,并尽量减小线程切换对性能的影响。

## 4.系统调用

在要求时间延迟确定的循环中，不进行可能触发异常的系统调用，比如 fopen

注:  c++ 17中文件系统操作,filesystem 类进行封装，可以优先考虑
## 5.ulimit

通过ulimit命令设置应用权限，设成实时进程的优先级别，这样应用可以以非root用户运行
```cpp
ulimit -r 40 <pid>
```
同时也可以设置应用最大能分配的内存，在某个应用出错内存泄露后，可以不影响其它进程

```shell
ulimit -m 4096 <pid>
```
在程序里通过setrlimit来限制资源。

## 6.设置实时线程

根据应用场景，设置各线程，各进程的实时优先级，设置两种策略：FIFO, RR，先进先出，和时间片轮询。

Linux内核的实时优先级分为1-99，99个级别。可以通过sched_setscheduler来设

置，但必须有root权限。
```cpp
struct sched_param param
param.sched_priority =50; //实时优先级50
sched_setscheduler(0, SCHED_FIFO, &param)
```

在内核加上RT patch后，中断线程化，默认优先级为50，应用优先级一般不应该超过它，具体需要根据场景全面考虑后再定优先级

## 7.优先级反转应对

在设置了实时的线程中，在需要关键区保护的地方，使用优先级继承的互斥锁。
```cpp
// 创建互斥锁属性变量
pthread_mutexattr_t mutexAttr; 
// 初始化互斥锁属性
ret = pthread_mutexattr_init(&mutexAttr);
// 设置互斥锁的调度策略为优先级继承
ret = pthread_mutexattr_setprotocol(&mutexAttr, PTHREAD_PRIO_INHERIT);
// 使用优先级继承属性初始化互斥锁
ret = pthread_mutex_init(&mutex, &mutexAttr);
```
注：需要glibc 2.5以上


## 8.CPU绑定

对于场景和CPU算力分析清楚以后，可以把某些线程绑定在某个CPU上，这个可以充分利用本址cache, 减少CPU migration引起的额外延迟和CPU开销，和cache在不同核中同步的开销。
```cpp
cpu_set_t set;
CPU_SET(0,&set);
sched_setaffinity(0,sizeof(cpu_set_t),&set);
```
在命令行或脚本中可以通过taskset来把某进程绑定到某个或某些CPU上。
## 分支预测 likely

对于对性能影响很大的循环中判断，如果绝大部分情况是为真或为假，可以用gcc的编译器指令告诉它，把大概率的分支作为主线分支，
```cpp
# define likely(x) __builtin_expect(!!(x), 1)
# define unlikely(x) __builtin_expect(!!(x), 0)
if (likely( cond == 1)) {
    printf("often hit");
} else {
    printf("rarely hit")
}
```
这个cond==1的分支是大概率分支。

注：arm编译有进行优化，大部分情况下都默认编译进行了优化，复杂分支需要设置
## 10.watchdog
关键的应用线程需要被独立的看门狗(watchdog)进程所监控。

看门狗进程用于监测关键进程和线程的运行状态。如果检测到某进程无响应,它可以重启该进程。

看门狗进程自己也需要具备崩溃自动重启的能力,可以由系统的 init 进程来监督它的运行。

看门狗进程的优先级需要根据整体场景来综合确定。