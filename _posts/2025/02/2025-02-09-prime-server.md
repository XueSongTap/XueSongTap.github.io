---
layout: article
title: prime_server：基于ZeroMQ的高性能分布式服务框架
tags: ZeroMQ
---

## 介绍

从Valhalla项目发现了一个非常有趣的服务器库—`prime_server`，这是一个基于ZeroMQ构建的非阻塞式web服务器API，专为分布式计算和面向服务架构(SOA)设计。

## 架构特点：蝴蝶模式

`prime_server`采用了ZeroMQ中的"蝴蝶模式"（Butterfly Pattern）或"并行管道模式"（Parallel Pipeline Pattern）。这种设计模式在[ZeroMQ教程](http://wiki.zeromq.org/tutorials:butterfly)中有详细描述。

基本架构如下：

```
client <---> server ---> proxy <---> [worker pool] <---> proxy <---> [worker pool] <---> ...
               ^                         |                                |
               |                         |                                |
                \ ____________________ /_________________________________/
```

![alt text](/img/250428/prime_server_core_architecture.png)

这种架构有几个关键优势：

1. **请求管道化**：请求可以通过多个阶段的处理流程
2. **并行处理**：每个阶段可以有多个worker并行处理任务
3. **负载均衡**：proxy确保工作均匀分配给可用的workers
4. **容错性**：任何阶段出现错误都可以立即返回给客户端
5. **可扩展性**：可以独立扩展各个处理阶段的worker数量


### 请求数据流
<img src="/img/250428/request_flow.png" alt="alt text" width="500">


### 代码结构


![alt text](/img/250428/code_structure.png)

## 技术基础：ZeroMQ

`prime_server`建立在ZeroMQ之上，ZeroMQ是一个高性能的消息传递库，提供了套接字API，支持多种通信模式。ZeroMQ以其低延迟、高吞吐量和可扩展性而闻名，非常适合构建分布式系统。

在这个框架中，ZeroMQ提供了关键的通信模式：

- **ZMQ_STREAM**：用于HTTP服务器部分
- **负载均衡模式**：在workers之间分配工作
- **发布/订阅模式**：用于结果的收集和分发

## 性能测试

简单的性能测试，使用Apache Benchmark (ab)向服务发送素数检测请求，其中计算素数是典型的CPU密集计算：

```bash
ab -k -n 1000 -c 8 http://localhost:8002/is_prime?possible_prime=32416190071
```

测试结果：

```bash
yxc@yxc-MS-7B89:~/code/2502/prime_server$ ab -k -n 1000 -c 8 http://localhost:8002/is_prime?possible_prime=32416190071
This is ApacheBench, Version 2.3 <$Revision: 1903618 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking localhost (be patient)
Completed 100 requests
Completed 200 requests
Completed 300 requests
Completed 400 requests
Completed 500 requests
Completed 600 requests
Completed 700 requests
Completed 800 requests
Completed 900 requests
Completed 1000 requests
Finished 1000 requests

Server Software:
Server Hostname: localhost
Server Port: 8002

Document Path: /is_prime?possible_prime=32416190071
Document Length: 11 bytes

Concurrency Level: 8
Time taken for tests: 2.060 seconds
Complete requests: 1000
Failed requests: 0
Keep-Alive requests: 1000
Total transferred: 74000 bytes
HTML transferred: 11000 bytes
Requests per second: 485.43 [#/sec] (mean)
Time per request: 16.480 [ms] (mean)
Time per request: 2.060 [ms] (mean, across all concurrent requests)
Transfer rate: 35.08 [Kbytes/sec] received

Connection Times (ms)
min mean[+/-sd] median max
Connect: 0 0 0.0 0 0
Processing: 4 16 2.2 16 32
Waiting: 4 16 2.2 16 32
Total: 5 16 2.2 16 32

Percentage of the requests served within a certain time (ms)
50% 16
66% 16
75% 16
80% 16
90% 16
95% 16
98% 28
99% 31
100% 32 (longest request)
yxc@yxc-MS-7B89:~/code/2502/prime_server$
```

同时我观察到服务的线程分布情况：

```bash
top - 14:09:07 up 2 days, 16:44,  2 users,  load average: 0.20, 0.30, 0.63
Threads:   9 total,   0 running,   9 sleeping,   0 stopped,   0 zombie
%Cpu(s):  2.0 us,  1.0 sy,  0.0 ni, 97.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :  80372.7 total,    714.7 free,   6606.8 used,  73982.1 buff/cache
MiB Swap:   8192.0 total,   3105.7 free,   5086.2 used.  73765.9 avail Mem

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
3731507 yxc       20   0  490748  38692  10880 S   0.0   0.0   0:00.00 prime_serverd
3731508 yxc       20   0  490748  38692  10880 S   0.0   0.0   0:00.00 prime_serverd
3731509 yxc       20   0  490748  38692  10880 S   0.0   0.0   0:00.00 ZMQbg/Reaper
3731510 yxc       20   0  490748  38692  10880 S   0.0   0.0   0:14.79 ZMQbg/IO/0
3731511 yxc       20   0  490748  38692  10880 S   0.0   0.0   0:05.68 prime_serverd
3731512 yxc       20   0  490748  38692  10880 S   0.0   0.0   0:02.65 prime_serverd
3731513 yxc       20   0  490748  38692  10880 S   0.0   0.0   0:02.91 prime_serverd
3731514 yxc       20   0  490748  38692  10880 S   0.0   0.0   0:02.00 prime_serverd
3731515 yxc       20   0  490748  38692  10880 S   0.0   0.0   2:14.21 prime_serverd
```

可以看到，prime_serverd自动创建了多个线程来处理请求，包括ZeroMQ的后台线程和多个worker线程。

