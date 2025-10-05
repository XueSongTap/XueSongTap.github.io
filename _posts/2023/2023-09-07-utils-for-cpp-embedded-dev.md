---
layout: articles
title: 实车调试常用shell 命令
tags: shell
---



### 基础命令


#### 时间同步信息 chronyc sources -v

```shell
acs@iZf8zeytjbajfs6v6d0oxuZ:~$ chronyc sources -v
210 Number of sources = 15

  .-- Source mode  '^' = server, '=' = peer, '#' = local clock.
 / .- Source state '*' = current synced, '+' = combined , '-' = not combined,
| /   '?' = unreachable, 'x' = time may be in error, '~' = time too variable.
||                                                 .- xxxx [ yyyy ] +/- zzzz
||      Reachability register (octal) -.           |  xxxx = adjusted offset,
||      Log2(Polling interval) --.      |          |  yyyy = measured offset,
||                                \     |          |  zzzz = estimated error.
||                                 |    |           \
MS Name/IP address         Stratum Poll Reach LastRx Last sample
===============================================================================
^* 100.100.61.88                 1  10   377   959   +353us[ +416us] +/-   27ms
^+ 203.107.6.88                  2  10   377   952  +2201us[+2201us] +/-   33ms
^+ 120.25.115.20                 2  10   275   32m   +604us[ +684us] +/- 4093us
^? 10.143.33.49                  0  10     0     -     +0ns[   +0ns] +/-    0ns
^+ 100.100.3.1                   2  10   377   925   -398us[ -398us] +/-   16ms
^+ 100.100.3.2                   2  10   377   937  -1030us[-1030us] +/-   16ms
^+ 100.100.3.3                   2  10    57    83  -1291us[-1291us] +/-   18ms
^? 10.143.33.50                  0  10     0     -     +0ns[   +0ns] +/-    0ns
^? 10.143.33.51                  0  10     0     -     +0ns[   +0ns] +/-    0ns
^? 10.143.0.44                   0  10     0     -     +0ns[   +0ns] +/-    0ns
^? 10.143.0.45                   0  10     0     -     +0ns[   +0ns] +/-    0ns
^? 10.143.0.46                   0  10     0     -     +0ns[   +0ns] +/-    0ns
^+ 100.100.5.1                   2  10   377   950   -163us[ -163us] +/-   18ms
^- 100.100.5.2                   2  10   241   32m   -273us[ -193us] +/-   16ms
^+ 100.100.5.3                   2  10   377   973  -1503us[-1441us] +/-   16ms

```

#### 检查pcie信息 lspci



```shell
yxc@yxc-MS-7B89:~$ lspci
00:00.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1480
00:00.2 IOMMU: Advanced Micro Devices, Inc. [AMD] Device 1481
00:01.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1482
00:01.1 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 1483
00:01.3 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 1483
00:02.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1482
00:03.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1482
00:03.1 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 1483
00:04.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1482
00:05.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1482
00:07.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1482
00:07.1 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 1484
00:08.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1482
00:08.1 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 1484
00:08.2 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 1484
00:08.3 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 1484
00:14.0 SMBus: Advanced Micro Devices, Inc. [AMD] FCH SMBus Controller (rev 61)
00:14.3 ISA bridge: Advanced Micro Devices, Inc. [AMD] FCH LPC Bridge (rev 51)
00:18.0 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1440
00:18.1 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1441
00:18.2 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1442
00:18.3 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1443
00:18.4 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1444
00:18.5 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1445
00:18.6 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1446
00:18.7 Host bridge: Advanced Micro Devices, Inc. [AMD] Device 1447
01:00.0 Non-Volatile memory controller: Sandisk Corp Device 5009 (rev 01)
03:00.0 USB controller: Advanced Micro Devices, Inc. [AMD] Device 43d5 (rev 01)
03:00.1 SATA controller: Advanced Micro Devices, Inc. [AMD] Device 43c8 (rev 01)
03:00.2 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 43c6 (rev 01)
20:00.0 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 43c7 (rev 01)
20:01.0 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 43c7 (rev 01)
20:04.0 PCI bridge: Advanced Micro Devices, Inc. [AMD] Device 43c7 (rev 01)
22:00.0 Ethernet controller: Realtek Semiconductor Co., Ltd. RTL8111/8168/8411 PCI Express Gigabit Ethernet Controller (rev 15)
25:00.0 Non-Volatile memory controller: Sandisk Corp Device 5006
26:00.0 VGA compatible controller: NVIDIA Corporation Device 2206 (rev a1)
26:00.1 Audio device: NVIDIA Corporation Device 1aef (rev a1)
27:00.0 Non-Essential Instrumentation [1300]: Advanced Micro Devices, Inc. [AMD] Device 148a
28:00.0 Non-Essential Instrumentation [1300]: Advanced Micro Devices, Inc. [AMD] Device 1485
28:00.1 Encryption controller: Advanced Micro Devices, Inc. [AMD] Device 1486
28:00.3 USB controller: Advanced Micro Devices, Inc. [AMD] Device 149c
28:00.4 Audio device: Advanced Micro Devices, Inc. [AMD] Device 1487
30:00.0 SATA controller: Advanced Micro Devices, Inc. [AMD] FCH SATA Controller [AHCI mode] (rev 51)
31:00.0 SATA controller: Advanced Micro Devices, Inc. [AMD] FCH SATA Controller [AHCI mode] (rev 51)
```

#### 查看时间diff

```
clockdiff ip
```

嵌入式端可能设计自己编译

https://github.com/iputils/iputils


#### file-coredump 查看coredump由哪个生成

```shell
file core-01Ad0-4841-1680489979 
```



