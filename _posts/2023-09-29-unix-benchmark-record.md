---
layout: articles
title: 手头设备的 unixbenchmark 跑分记录
tags: benchmark unix arm x86
---



## 跑分方式

```shell
git clone https://github.com/kdlucas/byte-unixbench.git
cd byte-unixbench/UnixBench
make
# 等待编译完成

# 编译完成后，需要给 执行程序赋予执行权限
sudo chmod u+x ./Run

# Ps：除了 Run 程序测试多核的时候，需要执行其他脚本
# 为了避免报错，最好把 UnixBench 目录下脚本都赋予执行权限
sudo chmod u+x -R ./*

sudo ./Run
# 等待测试结果即可
```


## 跑分结果

### pather x2 rk3566

新到的板子 pather x2

该设备型号为瑞芯微 rk3566 内存是ddr4 4g 32g emmc 5.1 支持内存卡扩展。


到手armbian系统


```shell
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: panther-x2: GNU/Linux
   OS: GNU/Linux -- 6.1.38-rockchip64 -- #3 SMP PREEMPT Wed Jul  5 17:27:38 UTC 2023
   Machine: aarch64 (aarch64)
   Language: en_US.utf8 (charmap="UTF-8", collate="UTF-8")
   CPU 0:  (48.0 bogomips)
          
   CPU 1:  (48.0 bogomips)
          
   CPU 2:  (48.0 bogomips)
          
   CPU 3:  (48.0 bogomips)
          
   10:22:39 up 17:58,  3 users,  load average: 0.54, 0.30, 0.17; runlevel 2023-09-28

------------------------------------------------------------------------
Benchmark Run: ven Sep 29 2023 10:22:39 - 10:50:58
4 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       11440979.1 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     2986.6 MWIPS (10.0 s, 7 samples)
Execl Throughput                                629.4 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        199779.9 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks           61658.0 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks        477773.9 KBps  (30.0 s, 2 samples)
Pipe Throughput                              342800.4 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                  30566.3 lps   (10.0 s, 7 samples)
Process Creation                               1800.5 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   2241.4 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                    666.6 lpm   (60.1 s, 2 samples)
System Call Overhead                         528658.1 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   11440979.1    980.4
Double-Precision Whetstone                       55.0       2986.6    543.0
Execl Throughput                                 43.0        629.4    146.4
File Copy 1024 bufsize 2000 maxblocks          3960.0     199779.9    504.5
File Copy 256 bufsize 500 maxblocks            1655.0      61658.0    372.6
File Copy 4096 bufsize 8000 maxblocks          5800.0     477773.9    823.7
Pipe Throughput                               12440.0     342800.4    275.6
Pipe-based Context Switching                   4000.0      30566.3     76.4
Process Creation                                126.0       1800.5    142.9
Shell Scripts (1 concurrent)                     42.4       2241.4    528.6
Shell Scripts (8 concurrent)                      6.0        666.6   1110.9
System Call Overhead                          15000.0     528658.1    352.4
                                                                   ========
System Benchmarks Index Score                                         374.1

------------------------------------------------------------------------
Benchmark Run: ven Sep 29 2023 10:50:58 - 11:19:22
4 CPUs in system; running 4 parallel copies of tests

Dhrystone 2 using register variables       44510748.9 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                    11599.6 MWIPS (10.0 s, 7 samples)
Execl Throughput                               1881.0 lps   (29.9 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        706944.7 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          233903.4 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       1231581.0 KBps  (30.0 s, 2 samples)
Pipe Throughput                             1340279.0 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 153598.7 lps   (10.0 s, 7 samples)
Process Creation                               4437.7 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   5133.3 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                    684.8 lpm   (60.2 s, 2 samples)
System Call Overhead                        2056180.5 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   44510748.9   3814.1
Double-Precision Whetstone                       55.0      11599.6   2109.0
Execl Throughput                                 43.0       1881.0    437.4
File Copy 1024 bufsize 2000 maxblocks          3960.0     706944.7   1785.2
File Copy 256 bufsize 500 maxblocks            1655.0     233903.4   1413.3
File Copy 4096 bufsize 8000 maxblocks          5800.0    1231581.0   2123.4
Pipe Throughput                               12440.0    1340279.0   1077.4
Pipe-based Context Switching                   4000.0     153598.7    384.0
Process Creation                                126.0       4437.7    352.2
Shell Scripts (1 concurrent)                     42.4       5133.3   1210.7
Shell Scripts (8 concurrent)                      6.0        684.8   1141.3
System Call Overhead                          15000.0    2056180.5   1370.8
                                                                   ========
System Benchmarks Index Score                                        1147.3
```


### b450 + 3700x

微星迫击炮，b450主办 默频，120一体水冷
```shell
========================================================================
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: yxc-MS-7B89: GNU/Linux
   OS: GNU/Linux -- 5.4.0-150-generic -- #167~18.04.1-Ubuntu SMP Wed May 24 00:51:42 UTC 2023
   Machine: x86_64 (x86_64)
   Language: en_US.utf8 (charmap="UTF-8", collate="UTF-8")
   CPU 0: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 1: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 2: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 3: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 4: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 5: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 6: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 7: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 8: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 9: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 10: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 11: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 12: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 13: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 14: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 15: AMD Ryzen 7 3700X 8-Core Processor (7199.9 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   10:39:15 up 16 days, 16:31,  6 users,  load average: 0.15, 0.09, 0.09; runlevel 2023-09-12

------------------------------------------------------------------------
Benchmark Run: Fri Sep 29 2023 10:39:15 - 11:09:51
16 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       54086383.7 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     3163.3 MWIPS (25.8 s, 7 samples)
Execl Throughput                               8707.6 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       1515267.4 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          415111.5 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       3331194.9 KBps  (30.0 s, 2 samples)
Pipe Throughput                             2576068.1 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 336019.2 lps   (10.0 s, 7 samples)
Process Creation                              12847.4 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                  10209.5 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   8141.9 lpm   (60.0 s, 2 samples)
System Call Overhead                        3628805.9 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   54086383.7   4634.7
Double-Precision Whetstone                       55.0       3163.3    575.1
Execl Throughput                                 43.0       8707.6   2025.0
File Copy 1024 bufsize 2000 maxblocks          3960.0    1515267.4   3826.4
File Copy 256 bufsize 500 maxblocks            1655.0     415111.5   2508.2
File Copy 4096 bufsize 8000 maxblocks          5800.0    3331194.9   5743.4
Pipe Throughput                               12440.0    2576068.1   2070.8
Pipe-based Context Switching                   4000.0     336019.2    840.0
Process Creation                                126.0      12847.4   1019.6
Shell Scripts (1 concurrent)                     42.4      10209.5   2407.9
Shell Scripts (8 concurrent)                      6.0       8141.9  13569.9
System Call Overhead                          15000.0    3628805.9   2419.2
                                                                   ========
System Benchmarks Index Score                                        2426.8

------------------------------------------------------------------------
Benchmark Run: Fri Sep 29 2023 11:09:51 - 11:40:14
16 CPUs in system; running 16 parallel copies of tests

Dhrystone 2 using register variables      605906477.1 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                    44021.5 MWIPS (23.9 s, 7 samples)
Execl Throughput                              58219.8 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       6795623.6 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks         1972258.4 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       6973802.1 KBps  (30.0 s, 2 samples)
Pipe Throughput                            25790784.0 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                3162968.6 lps   (10.0 s, 7 samples)
Process Creation                             114767.0 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                 108349.5 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                  16314.3 lpm   (60.0 s, 2 samples)
System Call Overhead                       38884205.2 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0  605906477.1  51920.0
Double-Precision Whetstone                       55.0      44021.5   8003.9
Execl Throughput                                 43.0      58219.8  13539.5
File Copy 1024 bufsize 2000 maxblocks          3960.0    6795623.6  17160.7
File Copy 256 bufsize 500 maxblocks            1655.0    1972258.4  11917.0
File Copy 4096 bufsize 8000 maxblocks          5800.0    6973802.1  12023.8
Pipe Throughput                               12440.0   25790784.0  20732.1
Pipe-based Context Switching                   4000.0    3162968.6   7907.4
Process Creation                                126.0     114767.0   9108.5
Shell Scripts (1 concurrent)                     42.4     108349.5  25554.1
Shell Scripts (8 concurrent)                      6.0      16314.3  27190.5
System Call Overhead                          15000.0   38884205.2  25922.8
                                                                   ========
System Benchmarks Index Score                                       16376.5

```


### b85 + i5-4590

实验室的陈年老物，b85平台，普通风冷，甚至没有清灰

```shell

========================================================================
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: yxc-B85M-HD3-A: GNU/Linux
   OS: GNU/Linux -- 5.4.0-84-generic -- #94~18.04.1-Ubuntu SMP Thu Aug 26 23:17:46 UTC 2021
   Machine: x86_64 (x86_64)
   Language: en_US.utf8 (charmap="UTF-8", collate="UTF-8")
   CPU 0: Intel(R) Core(TM) i5-4590 CPU @ 3.30GHz (6584.8 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 1: Intel(R) Core(TM) i5-4590 CPU @ 3.30GHz (6584.8 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 2: Intel(R) Core(TM) i5-4590 CPU @ 3.30GHz (6584.8 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 3: Intel(R) Core(TM) i5-4590 CPU @ 3.30GHz (6584.8 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   10:43:06 up 2 days, 23:33,  5 users,  load average: 0.35, 0.43, 0.51; runlevel 2023-09-26

------------------------------------------------------------------------
Benchmark Run: 五 9月 29 2023 10:43:06 - 11:12:24
4 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       45229178.8 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     3511.0 MWIPS (17.6 s, 7 samples)
Execl Throughput                               5307.2 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        710363.4 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          185084.1 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       1973666.7 KBps  (30.0 s, 2 samples)
Pipe Throughput                              927325.2 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 226391.7 lps   (10.0 s, 7 samples)
Process Creation                               6034.4 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   9032.2 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   4070.7 lpm   (60.0 s, 2 samples)
System Call Overhead                         546048.0 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   45229178.8   3875.7
Double-Precision Whetstone                       55.0       3511.0    638.4
Execl Throughput                                 43.0       5307.2   1234.2
File Copy 1024 bufsize 2000 maxblocks          3960.0     710363.4   1793.8
File Copy 256 bufsize 500 maxblocks            1655.0     185084.1   1118.3
File Copy 4096 bufsize 8000 maxblocks          5800.0    1973666.7   3402.9
Pipe Throughput                               12440.0     927325.2    745.4
Pipe-based Context Switching                   4000.0     226391.7    566.0
Process Creation                                126.0       6034.4    478.9
Shell Scripts (1 concurrent)                     42.4       9032.2   2130.2
Shell Scripts (8 concurrent)                      6.0       4070.7   6784.5
System Call Overhead                          15000.0     546048.0    364.0
                                                                   ========
System Benchmarks Index Score                                        1294.6

------------------------------------------------------------------------
Benchmark Run: 五 9月 29 2023 11:12:24 - 11:41:32
4 CPUs in system; running 4 parallel copies of tests

Dhrystone 2 using register variables      171072449.5 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                    14452.4 MWIPS (16.2 s, 7 samples)
Execl Throughput                              18175.8 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       2272404.8 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          649415.1 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       4655279.3 KBps  (30.0 s, 2 samples)
Pipe Throughput                             3496408.8 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 755908.2 lps   (10.0 s, 7 samples)
Process Creation                              35968.3 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                  28266.7 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   4508.5 lpm   (60.0 s, 2 samples)
System Call Overhead                        2052749.8 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0  171072449.5  14659.2
Double-Precision Whetstone                       55.0      14452.4   2627.7
Execl Throughput                                 43.0      18175.8   4226.9
File Copy 1024 bufsize 2000 maxblocks          3960.0    2272404.8   5738.4
File Copy 256 bufsize 500 maxblocks            1655.0     649415.1   3924.0
File Copy 4096 bufsize 8000 maxblocks          5800.0    4655279.3   8026.3
Pipe Throughput                               12440.0    3496408.8   2810.6
Pipe-based Context Switching                   4000.0     755908.2   1889.8
Process Creation                                126.0      35968.3   2854.6
Shell Scripts (1 concurrent)                     42.4      28266.7   6666.7
Shell Scripts (8 concurrent)                      6.0       4508.5   7514.1
System Call Overhead                          15000.0    2052749.8   1368.5
                                                                   ========
System Benchmarks Index Score                                        4205.3

```


### 腾讯云轻量服务器 2核的 6133

应该就是docker化的服务器
```shell
========================================================================
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: VM-4-5-ubuntu: GNU/Linux
   OS: GNU/Linux -- 4.15.0-206-generic -- #217-Ubuntu SMP Fri Feb 3 19:10:13 UTC 2023
   Machine: x86_64 (x86_64)
   Language: en_US.utf8 (charmap="UTF-8", collate="UTF-8")
   CPU 0: Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz (4988.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 1: Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz (4988.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   13:03:45 up 2 days, 16:37,  1 user,  load average: 0.19, 0.06, 0.01; runlevel 2023-09-26

------------------------------------------------------------------------
Benchmark Run: Fri Sep 29 2023 13:03:45 - 13:32:11
2 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       35098649.2 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     4219.0 MWIPS (11.8 s, 7 samples)
Execl Throughput                               3873.8 lps   (29.9 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        961393.9 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          268675.1 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       2282005.3 KBps  (30.0 s, 2 samples)
Pipe Throughput                             1866502.0 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                  56472.2 lps   (10.0 s, 7 samples)
Process Creation                               8202.4 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   8404.9 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   1453.7 lpm   (60.0 s, 2 samples)
System Call Overhead                        2619472.2 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   35098649.2   3007.6
Double-Precision Whetstone                       55.0       4219.0    767.1
Execl Throughput                                 43.0       3873.8    900.9
File Copy 1024 bufsize 2000 maxblocks          3960.0     961393.9   2427.8
File Copy 256 bufsize 500 maxblocks            1655.0     268675.1   1623.4
File Copy 4096 bufsize 8000 maxblocks          5800.0    2282005.3   3934.5
Pipe Throughput                               12440.0    1866502.0   1500.4
Pipe-based Context Switching                   4000.0      56472.2    141.2
Process Creation                                126.0       8202.4    651.0
Shell Scripts (1 concurrent)                     42.4       8404.9   1982.3
Shell Scripts (8 concurrent)                      6.0       1453.7   2422.9
System Call Overhead                          15000.0    2619472.2   1746.3
                                                                   ========
System Benchmarks Index Score                                        1351.9

------------------------------------------------------------------------
Benchmark Run: Fri Sep 29 2023 13:32:11 - 14:00:18
2 CPUs in system; running 2 parallel copies of tests

Dhrystone 2 using register variables       50900552.1 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     9060.8 MWIPS (9.6 s, 7 samples)
Execl Throughput                               5453.4 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       1266742.6 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          340873.5 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       2518505.5 KBps  (30.0 s, 2 samples)
Pipe Throughput                             2344043.1 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 341438.6 lps   (10.0 s, 7 samples)
Process Creation                              13558.7 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                  10554.9 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   1483.7 lpm   (60.1 s, 2 samples)
System Call Overhead                        3817568.8 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   50900552.1   4361.7
Double-Precision Whetstone                       55.0       9060.8   1647.4
Execl Throughput                                 43.0       5453.4   1268.2
File Copy 1024 bufsize 2000 maxblocks          3960.0    1266742.6   3198.8
File Copy 256 bufsize 500 maxblocks            1655.0     340873.5   2059.7
File Copy 4096 bufsize 8000 maxblocks          5800.0    2518505.5   4342.3
Pipe Throughput                               12440.0    2344043.1   1884.3
Pipe-based Context Switching                   4000.0     341438.6    853.6
Process Creation                                126.0      13558.7   1076.1
Shell Scripts (1 concurrent)                     42.4      10554.9   2489.4
Shell Scripts (8 concurrent)                      6.0       1483.7   2472.8
System Call Overhead                          15000.0    3817568.8   2545.0
                                                                   ========
System Benchmarks Index Score                                        2093.1

```
### intel 8700k

实验室老演员

```shell

========================================================================
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: hua-System-Product-Name: GNU/Linux
   OS: GNU/Linux -- 5.4.0-150-generic -- #167~18.04.1-Ubuntu SMP Wed May 24 00:51:42 UTC 2023
   Machine: x86_64 (x86_64)
   Language: en_US.utf8 (charmap="UTF-8", collate="UTF-8")
   CPU 0: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 1: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 2: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 3: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 4: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 5: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 6: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 7: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 8: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 9: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 10: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 11: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (7399.7 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   18:59:24 up 9 days, 49 min,  2 users,  load average: 0.00, 0.02, 0.00; runlevel 2023-09-21

------------------------------------------------------------------------
Benchmark Run: 五 9月 29 2023 18:59:24 - 19:29:46
12 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       59000503.5 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     3447.3 MWIPS (24.2 s, 7 samples)
Execl Throughput                               5629.3 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        674004.0 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          172074.9 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       2212394.7 KBps  (30.0 s, 2 samples)
Pipe Throughput                              836578.0 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 247693.9 lps   (10.0 s, 7 samples)
Process Creation                               3196.3 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   2934.7 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   7703.4 lpm   (60.0 s, 2 samples)
System Call Overhead                         484227.8 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   59000503.5   5055.7
Double-Precision Whetstone                       55.0       3447.3    626.8
Execl Throughput                                 43.0       5629.3   1309.1
File Copy 1024 bufsize 2000 maxblocks          3960.0     674004.0   1702.0
File Copy 256 bufsize 500 maxblocks            1655.0     172074.9   1039.7
File Copy 4096 bufsize 8000 maxblocks          5800.0    2212394.7   3814.5
Pipe Throughput                               12440.0     836578.0    672.5
Pipe-based Context Switching                   4000.0     247693.9    619.2
Process Creation                                126.0       3196.3    253.7
Shell Scripts (1 concurrent)                     42.4       2934.7    692.2
Shell Scripts (8 concurrent)                      6.0       7703.4  12839.1
System Call Overhead                          15000.0     484227.8    322.8
                                                                   ========
System Benchmarks Index Score                                        1195.1

------------------------------------------------------------------------
Benchmark Run: 五 9月 29 2023 19:29:46 - 20:00:59
12 CPUs in system; running 12 parallel copies of tests

Dhrystone 2 using register variables      407602326.5 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                    28285.9 MWIPS (28.6 s, 7 samples)
Execl Throughput                              39157.9 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       3983417.4 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks         1059454.6 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks      11032114.2 KBps  (30.0 s, 2 samples)
Pipe Throughput                             5722406.8 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                1394848.6 lps   (10.0 s, 7 samples)
Process Creation                              87978.6 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                  71985.1 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                  10631.5 lpm   (60.0 s, 2 samples)
System Call Overhead                        2975871.0 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0  407602326.5  34927.4
Double-Precision Whetstone                       55.0      28285.9   5142.9
Execl Throughput                                 43.0      39157.9   9106.5
File Copy 1024 bufsize 2000 maxblocks          3960.0    3983417.4  10059.1
File Copy 256 bufsize 500 maxblocks            1655.0    1059454.6   6401.5
File Copy 4096 bufsize 8000 maxblocks          5800.0   11032114.2  19020.9
Pipe Throughput                               12440.0    5722406.8   4600.0
Pipe-based Context Switching                   4000.0    1394848.6   3487.1
Process Creation                                126.0      87978.6   6982.4
Shell Scripts (1 concurrent)                     42.4      71985.1  16977.6
Shell Scripts (8 concurrent)                      6.0      10631.5  17719.2
System Call Overhead                          15000.0    2975871.0   1983.9
                                                                   ========
System Benchmarks Index Score                                        8457.5
```
### b550 5700G wsl2下


wsl2 最新的虚拟化，单核和直通的系统有点不太一样？

本来想测下wsl2和原生linux的性能差距，反而wsl2单核性能更好？


```shell
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: yxc: GNU/Linux
   OS: GNU/Linux -- 5.15.123.1-microsoft-standard-WSL2 -- #1 SMP Mon Aug 7 19:01:48 UTC 2023
   Machine: x86_64 (x86_64)
   Language: en_US.utf8 (charmap="ANSI_X3.4-1968", collate="ANSI_X3.4-1968")
   CPU 0: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 1: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 2: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 3: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 4: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 5: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 6: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 7: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 8: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 9: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 10: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 11: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 12: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 13: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 14: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   CPU 15: AMD Ryzen 7 5700G with Radeon Graphics (7599.8 bogomips)
          Hyper-Threading, x86-64, MMX, AMD MMX, Physical Address Ext, SYSENTER/SYSEXIT, AMD virtualization, SYSCALL/SYSRET
   09:24:55 up  9:55,  3 users,  load average: 0.14, 0.03, 0.01; runlevel Sep

------------------------------------------------------------------------
Benchmark Run: Sun Oct 01 2023 09:24:55 - 09:52:50
16 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       61853969.0 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                    11177.9 MWIPS (9.9 s, 7 samples)
Execl Throughput                               6475.9 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       2110978.2 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          564875.8 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       4874458.8 KBps  (30.0 s, 2 samples)
Pipe Throughput                             3276301.0 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                  36236.3 lps   (10.0 s, 7 samples)
Process Creation                              11761.4 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                  20254.4 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                  12190.0 lpm   (60.0 s, 2 samples)
System Call Overhead                        2646488.6 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   61853969.0   5300.3
Double-Precision Whetstone                       55.0      11177.9   2032.3
Execl Throughput                                 43.0       6475.9   1506.0
File Copy 1024 bufsize 2000 maxblocks          3960.0    2110978.2   5330.8
File Copy 256 bufsize 500 maxblocks            1655.0     564875.8   3413.1
File Copy 4096 bufsize 8000 maxblocks          5800.0    4874458.8   8404.2
Pipe Throughput                               12440.0    3276301.0   2633.7
Pipe-based Context Switching                   4000.0      36236.3     90.6
Process Creation                                126.0      11761.4    933.4
Shell Scripts (1 concurrent)                     42.4      20254.4   4777.0
Shell Scripts (8 concurrent)                      6.0      12190.0  20316.7
System Call Overhead                          15000.0    2646488.6   1764.3
                                                                   ========
System Benchmarks Index Score                                        2598.1

------------------------------------------------------------------------
Benchmark Run: Sun Oct 01 2023 09:52:50 - 10:20:50
16 CPUs in system; running 16 parallel copies of tests

Dhrystone 2 using register variables      590473955.9 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                   139722.6 MWIPS (10.1 s, 7 samples)
Execl Throughput                              54184.0 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks      12131383.2 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks         5702877.1 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       9266426.8 KBps  (30.0 s, 2 samples)
Pipe Throughput                            34270601.7 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                3933495.7 lps   (10.0 s, 7 samples)
Process Creation                              71638.0 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                 143127.1 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                  19927.1 lpm   (60.0 s, 2 samples)
System Call Overhead                       30878397.8 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0  590473955.9  50597.6
Double-Precision Whetstone                       55.0     139722.6  25404.1
Execl Throughput                                 43.0      54184.0  12600.9
File Copy 1024 bufsize 2000 maxblocks          3960.0   12131383.2  30634.8
File Copy 256 bufsize 500 maxblocks            1655.0    5702877.1  34458.5
File Copy 4096 bufsize 8000 maxblocks          5800.0    9266426.8  15976.6
Pipe Throughput                               12440.0   34270601.7  27548.7
Pipe-based Context Switching                   4000.0    3933495.7   9833.7
Process Creation                                126.0      71638.0   5685.6
Shell Scripts (1 concurrent)                     42.4     143127.1  33756.4
Shell Scripts (8 concurrent)                      6.0      19927.1  33211.8
System Call Overhead                          15000.0   30878397.8  20585.6
                                                                   ========
System Benchmarks Index Score                                       21491.6

```


### mac m1

mackbook air 没有风扇

8核心arm架构



```shell
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: computers-MacBook-Air.local: Darwin
   OS: Darwin -- 21.6.0 -- Darwin Kernel Version 21.6.0: Fri Sep 15 16:17:13 PDT 2023; root:xnu-8020.240.18.703.5~1/RELEASE_ARM64_T8101
   Machine: arm64 (unknown)
   Language: en_US.utf8 (charmap="US-ASCII", collate=)
   CPU 0: Apple M1 (0.0 bogomips)
          
   CPU 1: Apple M1 (0.0 bogomips)
          
   CPU 2: Apple M1 (0.0 bogomips)
          
   CPU 3: Apple M1 (0.0 bogomips)
          
   CPU 4: Apple M1 (0.0 bogomips)
          
   CPU 5: Apple M1 (0.0 bogomips)
          
   CPU 6: Apple M1 (0.0 bogomips)
          
   CPU 7: Apple M1 (0.0 bogomips)
          
   9:27  up 3 days, 50 mins, 2 users, load averages: 1.66 2.46 3.85; runlevel 3

------------------------------------------------------------------------
Benchmark Run: Sun Oct 01 2023 09:27:55 - 10:10:58
8 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       58221190.2 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     7080.7 MWIPS (10.0 s, 7 samples)
Execl Throughput                               1777.9 lps   (30.3 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       1013340.6 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          216960.9 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       2522712.6 KBps  (30.0 s, 2 samples)
Pipe Throughput                             1990599.9 lps   (10.1 s, 7 samples)
Pipe-based Context Switching                 316780.0 lps   (10.1 s, 7 samples)
Process Creation                               3780.0 lps   (30.5 s, 2 samples)
Shell Scripts (1 concurrent)                   7293.0 lpm   (60.5 s, 2 samples)
Shell Scripts (8 concurrent)                   2095.5 lpm   (60.0 s, 2 samples)
System Call Overhead                        1365623.6 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   58221190.2   4989.0
Double-Precision Whetstone                       55.0       7080.7   1287.4
Execl Throughput                                 43.0       1777.9    413.5
File Copy 1024 bufsize 2000 maxblocks          3960.0    1013340.6   2558.9
File Copy 256 bufsize 500 maxblocks            1655.0     216960.9   1310.9
File Copy 4096 bufsize 8000 maxblocks          5800.0    2522712.6   4349.5
Pipe Throughput                               12440.0    1990599.9   1600.2
Pipe-based Context Switching                   4000.0     316780.0    791.9
Process Creation                                126.0       3780.0    300.0
Shell Scripts (1 concurrent)                     42.4       7293.0   1720.0
Shell Scripts (8 concurrent)                      6.0       2095.5   3492.5
System Call Overhead                          15000.0    1365623.6    910.4
                                                                   ========
System Benchmarks Index Score                                        1441.6

------------------------------------------------------------------------
Benchmark Run: Sun Oct 01 2023 10:10:58 - 10:39:36
8 CPUs in system; running 8 parallel copies of tests

Dhrystone 2 using register variables      237588256.9 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                    38356.2 MWIPS (10.4 s, 7 samples)
Execl Throughput                               6868.2 lps   (29.8 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       1025127.1 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          329655.0 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       3314905.4 KBps  (30.0 s, 2 samples)
Pipe Throughput                             1623388.0 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 286685.3 lps   (10.0 s, 7 samples)
Process Creation                               9543.5 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   9640.0 lpm   (60.1 s, 2 samples)
Shell Scripts (8 concurrent)                   1129.0 lpm   (60.2 s, 2 samples)
System Call Overhead                        4956704.3 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0  237588256.9  20358.9
Double-Precision Whetstone                       55.0      38356.2   6973.9
Execl Throughput                                 43.0       6868.2   1597.3
File Copy 1024 bufsize 2000 maxblocks          3960.0    1025127.1   2588.7
File Copy 256 bufsize 500 maxblocks            1655.0     329655.0   1991.9
File Copy 4096 bufsize 8000 maxblocks          5800.0    3314905.4   5715.4
Pipe Throughput                               12440.0    1623388.0   1305.0
Pipe-based Context Switching                   4000.0     286685.3    716.7
Process Creation                                126.0       9543.5    757.4
Shell Scripts (1 concurrent)                     42.4       9640.0   2273.6
Shell Scripts (8 concurrent)                      6.0       1129.0   1881.7
System Call Overhead                          15000.0    4956704.3   3304.5
                                                                   ========
System Benchmarks Index Score                                        2524.3

```




### Jetson nano

jetson nano 买gpu送 cpu


四核心armv8

```shell
========================================================================
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: nano: GNU/Linux
   OS: GNU/Linux -- 4.9.253-tegra -- #1 SMP PREEMPT Sat Feb 19 08:59:22 PST 2022
   Machine: aarch64 (aarch64)
   Language: en_US.utf8 (charmap="UTF-8", collate="UTF-8")
   CPU 0: ARMv8 Processor rev 1 (v8l) (38.4 bogomips)
          
   CPU 1: ARMv8 Processor rev 1 (v8l) (38.4 bogomips)
          
   CPU 2: ARMv8 Processor rev 1 (v8l) (38.4 bogomips)
          
   CPU 3: ARMv8 Processor rev 1 (v8l) (38.4 bogomips)
          
   21:47:12 up  2:19,  2 users,  load average: 0.18, 0.10, 0.09; runlevel 2023-10-05

------------------------------------------------------------------------
Benchmark Run: Thu Oct 05 2023 21:47:12 - 22:15:18
4 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       13642217.5 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     1491.8 MWIPS (9.9 s, 7 samples)
Execl Throughput                               1151.5 lps   (29.9 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        190006.4 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks           53279.4 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks        536132.7 KBps  (30.0 s, 2 samples)
Pipe Throughput                              350309.5 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                  52986.9 lps   (10.0 s, 7 samples)
Process Creation                               1097.6 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   3557.3 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   1241.9 lpm   (60.0 s, 2 samples)
System Call Overhead                         358482.8 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   13642217.5   1169.0
Double-Precision Whetstone                       55.0       1491.8    271.2
Execl Throughput                                 43.0       1151.5    267.8
File Copy 1024 bufsize 2000 maxblocks          3960.0     190006.4    479.8
File Copy 256 bufsize 500 maxblocks            1655.0      53279.4    321.9
File Copy 4096 bufsize 8000 maxblocks          5800.0     536132.7    924.4
Pipe Throughput                               12440.0     350309.5    281.6
Pipe-based Context Switching                   4000.0      52986.9    132.5
Process Creation                                126.0       1097.6     87.1
Shell Scripts (1 concurrent)                     42.4       3557.3    839.0
Shell Scripts (8 concurrent)                      6.0       1241.9   2069.8
System Call Overhead                          15000.0     358482.8    239.0
                                                                   ========
System Benchmarks Index Score                                         399.1

------------------------------------------------------------------------
Benchmark Run: Thu Oct 05 2023 22:15:18 - 22:43:28
4 CPUs in system; running 4 parallel copies of tests

Dhrystone 2 using register variables       54128141.9 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     5912.8 MWIPS (9.8 s, 7 samples)
Execl Throughput                               4111.3 lps   (29.6 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        712611.1 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          199426.8 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       1978399.8 KBps  (30.0 s, 2 samples)
Pipe Throughput                             1393971.4 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 158461.3 lps   (10.0 s, 7 samples)
Process Creation                              13370.8 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   9162.8 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   1328.3 lpm   (60.1 s, 2 samples)
System Call Overhead                        1425165.6 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   54128141.9   4638.2
Double-Precision Whetstone                       55.0       5912.8   1075.1
Execl Throughput                                 43.0       4111.3    956.1
File Copy 1024 bufsize 2000 maxblocks          3960.0     712611.1   1799.5
File Copy 256 bufsize 500 maxblocks            1655.0     199426.8   1205.0
File Copy 4096 bufsize 8000 maxblocks          5800.0    1978399.8   3411.0
Pipe Throughput                               12440.0    1393971.4   1120.6
Pipe-based Context Switching                   4000.0     158461.3    396.2
Process Creation                                126.0      13370.8   1061.2
Shell Scripts (1 concurrent)                     42.4       9162.8   2161.0
Shell Scripts (8 concurrent)                      6.0       1328.3   2213.8
System Call Overhead                          15000.0    1425165.6    950.1
                                                                   ========
System Benchmarks Index Score                                        1433.8
```

### 5700G 直通

5700G 直通，跟上面的wsl2的是一个环境，单核跑分要比wsl2的差很多

原因待排查

```shell
Benchmark Run: 六 10月 07 2023 11:26:22 - 11:54:25                                                                                                                                                             
16 CPUs in system; running 1 parallel copy of tests                                                                                                                                                            
                                                                                                                                                                                                               
Dhrystone 2 using register variables       62539238.4 lps   (10.0 s, 7 samples)                                                                                                                                
Double-Precision Whetstone                    11078.2 MWIPS (9.9 s, 7 samples)                                                                                                                                 
Execl Throughput                               1696.9 lps   (30.0 s, 2 samples)                                                                                                                                
File Copy 1024 bufsize 2000 maxblocks        852880.3 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          217770.7 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       2604951.8 KBps  (30.0 s, 2 samples)
Pipe Throughput                             1282429.9 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 192002.2 lps   (10.0 s, 7 samples)
Process Creation                               6138.4 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   5645.1 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   4058.9 lpm   (60.0 s, 2 samples)
System Call Overhead                        1418272.6 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   62539238.4   5359.0
Double-Precision Whetstone                       55.0      11078.2   2014.2
Execl Throughput                                 43.0       1696.9    394.6
File Copy 1024 bufsize 2000 maxblocks          3960.0     852880.3   2153.7
File Copy 256 bufsize 500 maxblocks            1655.0     217770.7   1315.8
File Copy 4096 bufsize 8000 maxblocks          5800.0    2604951.8   4491.3
Pipe Throughput                               12440.0    1282429.9   1030.9
Pipe-based Context Switching                   4000.0     192002.2    480.0
Process Creation                                126.0       6138.4    487.2
Shell Scripts (1 concurrent)                     42.4       5645.1   1331.4
Shell Scripts (8 concurrent)                      6.0       4058.9   6764.9
System Call Overhead                          15000.0    1418272.6    945.5
                                                                   ========
System Benchmarks Index Score                                        1481.0

------------------------------------------------------------------------
Benchmark Run: 六 10月 07 2023 11:54:25 - 12:22:50
16 CPUs in system; running 16 parallel copies of tests


Dhrystone 2 using register variables      571147767.3 lps   (10.0 s, 7 samples)                                                                                                                                
Double-Precision Whetstone                   135200.1 MWIPS (9.9 s, 7 samples)
Execl Throughput                              16789.3 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       9397157.8 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks         2490354.1 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks      11401304.1 KBps  (30.0 s, 2 samples)
Pipe Throughput                            14631879.5 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                1739709.0 lps   (10.0 s, 7 samples)
Process Creation                              56859.5 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                  40181.3 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                   6847.6 lpm   (60.1 s, 2 samples)
System Call Overhead                       13243508.4 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0  571147767.3  48941.5
Double-Precision Whetstone                       55.0     135200.1  24581.8
Execl Throughput                                 43.0      16789.3   3904.5
File Copy 1024 bufsize 2000 maxblocks          3960.0    9397157.8  23730.2
File Copy 256 bufsize 500 maxblocks            1655.0    2490354.1  15047.5
File Copy 4096 bufsize 8000 maxblocks          5800.0   11401304.1  19657.4
Pipe Throughput                               12440.0   14631879.5  11762.0
Pipe-based Context Switching                   4000.0    1739709.0   4349.3
Process Creation                                126.0      56859.5   4512.7
Shell Scripts (1 concurrent)                     42.4      40181.3   9476.7
Shell Scripts (8 concurrent)                      6.0       6847.6  11412.7
System Call Overhead                          15000.0   13243508.4   8829.0
                                                                   ========
System Benchmarks Index Score                                       11797.4

```


### 双路 E5 2640

双路 E5-2640 v4

单颗 10核20线程

单核性能很差，但是胜在核心多


截至11月21日，淘宝单颗售价33元，普通双路x99 主板应该就可以跑起来
```shell
========================================================================
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: swarm02: GNU/Linux
   OS: GNU/Linux -- 3.10.0-1160.el7.x86_64 -- #1 SMP Mon Oct 19 16:18:59 UTC 2020
   Machine: x86_64 (x86_64)
   Language: en_US.utf8 (charmap="UTF-8", collate="UTF-8")
   CPU 0: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 1: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 2: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 3: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 4: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 5: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 6: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 7: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 8: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 9: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 10: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 11: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 12: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 13: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 14: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 15: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 16: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 17: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 18: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 19: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 20: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 21: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 22: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 23: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 24: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 25: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 26: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 27: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 28: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 29: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 30: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 31: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 32: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 33: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 34: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 35: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 36: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 37: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 38: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4794.4 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   CPU 39: Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz (4799.2 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET, Intel virtualization
   19:42:24 up 34 days, 11:21, 134 users,  load average: 0.85, 0.33, 0.39; runlevel 2023-09-03

------------------------------------------------------------------------
Benchmark Run: Sat Oct 07 2023 19:42:24 - 20:12:29
40 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       39685296.6 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     3025.5 MWIPS (19.6 s, 7 samples)
Execl Throughput                               1574.0 lps   (29.7 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        714999.7 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          187415.6 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks        548331.5 KBps  (30.0 s, 2 samples)
Pipe Throughput                             1043406.7 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 152907.6 lps   (10.0 s, 7 samples)
Process Creation                               6522.0 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   2038.3 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                    908.2 lpm   (60.0 s, 2 samples)
System Call Overhead                         858504.9 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   39685296.6   3400.6
Double-Precision Whetstone                       55.0       3025.5    550.1
Execl Throughput                                 43.0       1574.0    366.0
File Copy 1024 bufsize 2000 maxblocks          3960.0     714999.7   1805.6
File Copy 256 bufsize 500 maxblocks            1655.0     187415.6   1132.4
File Copy 4096 bufsize 8000 maxblocks          5800.0     548331.5    945.4
Pipe Throughput                               12440.0    1043406.7    838.8
Pipe-based Context Switching                   4000.0     152907.6    382.3
Process Creation                                126.0       6522.0    517.6
Shell Scripts (1 concurrent)                     42.4       2038.3    480.7
Shell Scripts (8 concurrent)                      6.0        908.2   1513.7
System Call Overhead                          15000.0     858504.9    572.3
                                                                   ========
System Benchmarks Index Score                                         819.3

------------------------------------------------------------------------
Benchmark Run: Sat Oct 07 2023 20:12:29 - 20:42:22
40 CPUs in system; running 40 parallel copies of tests

Dhrystone 2 using register variables      665567883.0 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                   153184.8 MWIPS (9.9 s, 7 samples)
Execl Throughput                              42868.6 lps   (29.4 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       5697936.7 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks         1415183.8 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       7202314.7 KBps  (30.0 s, 2 samples)
Pipe Throughput                            22925236.5 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                4833304.9 lps   (10.0 s, 7 samples)
Process Creation                             102372.2 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                   6951.8 lpm   (60.1 s, 2 samples)
Shell Scripts (8 concurrent)                    930.5 lpm   (61.6 s, 2 samples)
System Call Overhead                       20398924.7 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0  665567883.0  57032.4
Double-Precision Whetstone                       55.0     153184.8  27851.8
Execl Throughput                                 43.0      42868.6   9969.5
File Copy 1024 bufsize 2000 maxblocks          3960.0    5697936.7  14388.7
File Copy 256 bufsize 500 maxblocks            1655.0    1415183.8   8551.0
File Copy 4096 bufsize 8000 maxblocks          5800.0    7202314.7  12417.8
Pipe Throughput                               12440.0   22925236.5  18428.6
Pipe-based Context Switching                   4000.0    4833304.9  12083.3
Process Creation                                126.0     102372.2   8124.8
Shell Scripts (1 concurrent)                     42.4       6951.8   1639.6
Shell Scripts (8 concurrent)                      6.0        930.5   1550.9
System Call Overhead                          15000.0   20398924.7  13599.3
                                                                   ========
System Benchmarks Index Score                                       10351.5
```


### 双路 8369B


阿里云的服务器，双路 8369B，主要拿来搭配四卡gpu的

搜了下这个cpu甚至是阿里云定制？


```bash

   #    #  #    #  #  #    #          #####   ######  #    #   ####   #    #
   #    #  ##   #  #   #  #           #    #  #       ##   #  #    #  #    #
   #    #  # #  #  #    ##            #####   #####   # #  #  #       ######
   #    #  #  # #  #    ##            #    #  #       #  # #  #       #    #
   #    #  #   ##  #   #  #           #    #  #       #   ##  #    #  #    #
    ####   #    #  #  #    #          #####   ######  #    #   ####   #    #

   Version 5.1.3                      Based on the Byte Magazine Unix Benchmark

   Multi-CPU version                  Version 5 revisions by Ian Smith,
                                      Sunnyvale, CA, USA
   January 13, 2011                   johantheghost at yahoo period com

------------------------------------------------------------------------------
   Use directories for:
      * File I/O tests (named fs***) = /root/yxc/byte-unixbench/UnixBench/tmp
      * Results                      = /root/yxc/byte-unixbench/UnixBench/results
------------------------------------------------------------------------------


1 x Dhrystone 2 using register variables  1 2 3 4 5 6 7 8 9 10

1 x Double-Precision Whetstone  1 2 3 4 5 6 7 8 9 10

1 x Execl Throughput  1 2 3

1 x File Copy 1024 bufsize 2000 maxblocks  1 2 3

1 x File Copy 256 bufsize 500 maxblocks  1 2 3

1 x File Copy 4096 bufsize 8000 maxblocks  1 2 3

1 x Pipe Throughput  1 2 3 4 5 6 7 8 9 10

1 x Pipe-based Context Switching  1 2 3 4 5 6 7 8 9 10

1 x Process Creation  1 2 3

1 x System Call Overhead  1 2 3 4 5 6 7 8 9 10

1 x Shell Scripts (1 concurrent)  1 2 3

1 x Shell Scripts (8 concurrent)  1 2 3

128 x Dhrystone 2 using register variables  1 2 3 4 5 6 7 8 9 10

128 x Double-Precision Whetstone  1 2 3 4 5 6 7 8 9 10

128 x Execl Throughput  1 2 3

128 x File Copy 1024 bufsize 2000 maxblocks  1 2 3

128 x File Copy 256 bufsize 500 maxblocks  1 2 3

128 x File Copy 4096 bufsize 8000 maxblocks  1 2 3

128 x Pipe Throughput  1 2 3 4 5 6 7 8 9 10

128 x Pipe-based Context Switching  1 2 3 4 5 6 7 8 9 10

128 x Process Creation  1 2 3

128 x System Call Overhead  1 2 3 4 5 6 7 8 9 10

128 x Shell Scripts (1 concurrent)  1 2 3

128 x Shell Scripts (8 concurrent)  1 2 3

========================================================================
   BYTE UNIX Benchmarks (Version 5.1.3)

   System: PAI-AIS-A10x4: GNU/Linux
   OS: GNU/Linux -- 4.15.0-213-generic -- #224-Ubuntu SMP Mon Jun 19 13:30:12 UTC 2023
   Machine: x86_64 (x86_64)
   Language: en_US.utf8 (charmap="UTF-8", collate="UTF-8")
   CPU 0: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 1: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 2: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 3: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 4: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 5: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 6: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 7: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 8: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 9: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 10: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 11: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 12: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 13: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 14: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 15: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 16: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 17: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 18: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 19: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 20: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 21: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 22: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 23: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 24: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 25: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 26: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 27: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 28: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 29: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 30: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 31: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 32: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 33: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 34: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 35: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 36: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 37: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 38: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 39: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 40: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 41: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 42: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 43: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 44: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 45: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 46: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 47: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 48: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 49: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 50: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 51: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 52: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 53: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 54: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 55: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 56: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 57: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 58: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 59: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 60: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 61: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 62: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 63: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 64: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 65: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 66: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 67: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 68: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 69: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 70: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 71: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 72: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 73: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 74: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 75: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 76: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 77: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 78: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 79: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 80: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 81: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 82: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 83: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 84: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 85: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 86: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 87: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 88: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 89: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 90: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 91: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 92: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 93: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 94: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 95: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 96: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 97: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 98: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 99: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 100: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 101: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 102: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 103: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 104: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 105: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 106: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 107: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 108: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 109: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 110: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 111: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 112: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 113: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 114: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 115: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 116: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 117: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 118: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 119: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 120: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 121: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 122: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 123: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 124: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 125: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 126: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   CPU 127: Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz (5800.0 bogomips)
          Hyper-Threading, x86-64, MMX, Physical Address Ext, SYSENTER/SYSEXIT, SYSCALL/SYSRET
   09:51:22 up 8 days, 22:15,  6 users,  load average: 1.43, 1.36, 1.10; runlevel 2023-10-30

------------------------------------------------------------------------
Benchmark Run: Wed Nov 08 2023 09:51:22 - 10:20:13
128 CPUs in system; running 1 parallel copy of tests

Dhrystone 2 using register variables       45005117.2 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                     3980.6 MWIPS (14.5 s, 7 samples)
Execl Throughput                               4899.9 lps   (30.0 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks       1103110.7 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          310813.0 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       3054515.6 KBps  (30.0 s, 2 samples)
Pipe Throughput                             2679536.0 lps   (10.0 s, 7 samples)
Pipe-based Context Switching                 162531.8 lps   (10.0 s, 7 samples)
Process Creation                              10804.2 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                  14123.6 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                  11074.5 lpm   (60.0 s, 2 samples)
System Call Overhead                        3379002.3 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0   45005117.2   3856.5
Double-Precision Whetstone                       55.0       3980.6    723.8
Execl Throughput                                 43.0       4899.9   1139.5
File Copy 1024 bufsize 2000 maxblocks          3960.0    1103110.7   2785.6
File Copy 256 bufsize 500 maxblocks            1655.0     310813.0   1878.0
File Copy 4096 bufsize 8000 maxblocks          5800.0    3054515.6   5266.4
Pipe Throughput                               12440.0    2679536.0   2154.0
Pipe-based Context Switching                   4000.0     162531.8    406.3
Process Creation                                126.0      10804.2    857.5
Shell Scripts (1 concurrent)                     42.4      14123.6   3331.0
Shell Scripts (8 concurrent)                      6.0      11074.5  18457.4
System Call Overhead                          15000.0    3379002.3   2252.7
                                                                   ========
System Benchmarks Index Score                                        2137.7

------------------------------------------------------------------------
Benchmark Run: Wed Nov 08 2023 10:20:13 - 10:49:10
128 CPUs in system; running 128 parallel copies of tests

Dhrystone 2 using register variables     2850132582.9 lps   (10.0 s, 7 samples)
Double-Precision Whetstone                   553962.3 MWIPS (10.4 s, 7 samples)
Execl Throughput                              41916.2 lps   (29.9 s, 2 samples)
File Copy 1024 bufsize 2000 maxblocks        951799.7 KBps  (30.0 s, 2 samples)
File Copy 256 bufsize 500 maxblocks          237924.9 KBps  (30.0 s, 2 samples)
File Copy 4096 bufsize 8000 maxblocks       3834499.8 KBps  (30.0 s, 2 samples)
Pipe Throughput                           166378221.5 lps   (10.0 s, 7 samples)
Pipe-based Context Switching               10514069.2 lps   (10.0 s, 7 samples)
Process Creation                              96201.2 lps   (30.0 s, 2 samples)
Shell Scripts (1 concurrent)                 199633.9 lpm   (60.0 s, 2 samples)
Shell Scripts (8 concurrent)                  27003.2 lpm   (60.1 s, 2 samples)
System Call Overhead                      272779429.4 lps   (10.0 s, 7 samples)

System Benchmarks Index Values               BASELINE       RESULT    INDEX
Dhrystone 2 using register variables         116700.0 2850132582.9 244227.3
Double-Precision Whetstone                       55.0     553962.3 100720.4
Execl Throughput                                 43.0      41916.2   9747.9
File Copy 1024 bufsize 2000 maxblocks          3960.0     951799.7   2403.5
File Copy 256 bufsize 500 maxblocks            1655.0     237924.9   1437.6
File Copy 4096 bufsize 8000 maxblocks          5800.0    3834499.8   6611.2
Pipe Throughput                               12440.0  166378221.5 133744.6
Pipe-based Context Switching                   4000.0   10514069.2  26285.2
Process Creation                                126.0      96201.2   7635.0
Shell Scripts (1 concurrent)                     42.4     199633.9  47083.5
Shell Scripts (8 concurrent)                      6.0      27003.2  45005.3
System Call Overhead                          15000.0  272779429.4 181853.0
                                                                   ========
System Benchmarks Index Score                                       24894.5
```