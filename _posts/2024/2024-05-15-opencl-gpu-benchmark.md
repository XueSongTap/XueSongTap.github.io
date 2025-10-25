---
layout: article
title: OpenCL性能评测工具
tags: opencl gpu 
---

找到一款opencl-benchmark 工具

https://github.com/ProjectPhysX/OpenCL-Benchmark


双卡1080ti实测：

```bash
yxc@hua-System-Product-Name:~/code/3/OpenCL-Benchmark$ ./make.sh 
.-----------------------------------------------------------------------------.
|----------------.------------------------------------------------------------|
| Device ID    0 | NVIDIA GeForce GTX 1080 Ti                                 |
| Device ID    1 | NVIDIA GeForce GTX 1080 Ti                                 |
|----------------'------------------------------------------------------------|
|----------------.------------------------------------------------------------|
| Device ID      | 0                                                          |
| Device Name    | NVIDIA GeForce GTX 1080 Ti                                 |
| Device Vendor  | NVIDIA Corporation                                         |
| Device Driver  | 530.41.03 (Linux)                                          |
| OpenCL Version | OpenCL C 1.2                                               |
| Compute Units  | 28 at 1657 MHz (3584 cores, 11.877 TFLOPs/s)               |
| Memory, Cache  | 11169 MB, 1344 KB global / 48 KB local                     |
| Buffer Limits  | 2792 MB global, 64 KB constant                             |
|----------------'------------------------------------------------------------|
| Info: OpenCL C code successfully compiled.                                  |
| FP64  compute                                         0.427 TFLOPs/s (1/32) |
| FP32  compute                                        12.837 TFLOPs/s ( 1x ) |
| FP16  compute                                          not supported        |
| INT64 compute                                         1.530  TIOPs/s (1/8 ) |
| INT32 compute                                         4.505  TIOPs/s (1/3 ) |
| INT16 compute                                        12.740  TIOPs/s ( 1x ) |
| INT8  compute                                        13.012  TIOPs/s ( 1x ) |
| Memory Bandwidth ( coalesced read      )                        372.41 GB/s |
| Memory Bandwidth ( coalesced      write)                        429.60 GB/s |
| Memory Bandwidth (misaligned read      )                        163.06 GB/s |
| Memory Bandwidth (misaligned      write)                        100.27 GB/s |
| PCIe   Bandwidth (send                 )                         10.41 GB/s |
| PCIe   Bandwidth (   receive           )                         10.32 GB/s |
| PCIe   Bandwidth (        bidirectional)            (Gen4 x16)   10.02 GB/s |
|-----------------------------------------------------------------------------|
|----------------.------------------------------------------------------------|
| Device ID      | 1                                                          |
| Device Name    | NVIDIA GeForce GTX 1080 Ti                                 |
| Device Vendor  | NVIDIA Corporation                                         |
| Device Driver  | 530.41.03 (Linux)                                          |
| OpenCL Version | OpenCL C 1.2                                               |
| Compute Units  | 28 at 1632 MHz (3584 cores, 11.698 TFLOPs/s)               |
| Memory, Cache  | 11172 MB, 1344 KB global / 48 KB local                     |
| Buffer Limits  | 2793 MB global, 64 KB constant                             |
|----------------'------------------------------------------------------------|
| Info: OpenCL C code successfully compiled.                                  |
| FP64  compute                                         0.431 TFLOPs/s (1/24) |
| FP32  compute                                        12.888 TFLOPs/s ( 1x ) |
| FP16  compute                                          not supported        |
| INT64 compute                                         1.533  TIOPs/s (1/8 ) |
| INT32 compute                                         4.533  TIOPs/s (1/3 ) |
| INT16 compute                                        12.429  TIOPs/s ( 1x ) |
| INT8  compute                                        13.058  TIOPs/s ( 1x ) |
| Memory Bandwidth ( coalesced read      )                        374.60 GB/s |
| Memory Bandwidth ( coalesced      write)                        429.12 GB/s |
| Memory Bandwidth (misaligned read      )                        163.98 GB/s |
| Memory Bandwidth (misaligned      write)                        100.70 GB/s |
| PCIe   Bandwidth (send                 )                          1.58 GB/s |
| PCIe   Bandwidth (   receive           )                          1.63 GB/s |
| PCIe   Bandwidth (        bidirectional)            (Gen1 x16)    1.59 GB/s |
|-----------------------------------------------------------------------------|
|-----------------------------------------------------------------------------|
| Done. Press Enter to exit.                                                  |
'-----------------------------------------------------------------------------'

```



cpu 跑

```bash
.-----------------------------------------------------------------------------.
|----------------.------------------------------------------------------------|
| Device ID    0 | cpu-haswell-AMD Ryzen 5 5600G with Radeon Graphics         |
|----------------'------------------------------------------------------------|
|----------------.------------------------------------------------------------|
| Device ID      | 0                                                          |
| Device Name    | cpu-haswell-AMD Ryzen 5 5600G with Radeon Graphics         |
| Device Vendor  | AuthenticAMD                                               |
| Device Driver  | 5.0+debian (Linux)                                         |
| OpenCL Version | OpenCL C 3.0                                               |
| Compute Units  | 12 at 4465 MHz (6 cores, 0.857 TFLOPs/s)                   |
| Memory, Cache  | 58116 MB RAM, 16384 KB global / 512 KB local               |
| Buffer Limits  | 16384 MB global, 512 KB constant                           |
|----------------'------------------------------------------------------------|
| Info: OpenCL C code successfully compiled.                                  |
| FP64  compute                                         0.030 TFLOPs/s (1/32) |
| FP32  compute                                         0.027 TFLOPs/s (1/32) |
| FP16  compute                                          not supported        |
| INT64 compute                                         0.033  TIOPs/s (1/24) |
| INT32 compute                                         0.027  TIOPs/s (1/32) |
| INT16 compute                                         0.064  TIOPs/s (1/12) |
| INT8  compute                                         0.030  TIOPs/s (1/32) |
| Memory Bandwidth ( coalesced read      )                         24.39 GB/s |
| Memory Bandwidth ( coalesced      write)                         13.29 GB/s |
| Memory Bandwidth (misaligned read      )                         31.94 GB/s |
| Memory Bandwidth (misaligned      write)                         14.78 GB/s |
|-----------------------------------------------------------------------------|
'-----------------------------------------------------------------------------'
```


```bash
yxc@nas:~/OpenCL-Benchmark$ ./make.sh
.-----------------------------------------------------------------------------.
|----------------.------------------------------------------------------------|
| Device ID    0 | Intel(R) HD Graphics 610                                   |
|----------------'------------------------------------------------------------|
|----------------.------------------------------------------------------------|
| Device ID      | 0                                                          |
| Device Name    | Intel(R) HD Graphics 610                                   |
| Device Vendor  | Intel(R) Corporation                                       |
| Device Driver  | 24.31.30508.7 (Linux)                                      |
| OpenCL Version | OpenCL C 3.0                                               |
| Compute Units  | 12 at 1000 MHz (96 cores, 0.192 TFLOPs/s)                  |
| Memory, Cache  | 14397 MB RAM, 384 KB global / 64 KB local                  |
| Buffer Limits  | 4095 MB global, 4194296 KB constant                        |
|----------------'------------------------------------------------------------|
| Info: OpenCL C code successfully compiled.                                  |
| FP64  compute                                         0.044 TFLOPs/s (1/4 ) |
| FP32  compute                                         0.171 TFLOPs/s ( 1x ) |
| FP16  compute                                         0.315 TFLOPs/s ( 2x ) |
| INT64 compute                                         0.006  TIOPs/s (1/32) |
| INT32 compute                                         0.059  TIOPs/s (1/3 ) |
| INT16 compute                                         0.342  TIOPs/s ( 2x ) |
| INT8  compute                                         0.064  TIOPs/s (1/3 ) |
| Memory Bandwidth ( coalesced read      )                          9.54 GB/s |
| Memory Bandwidth ( coalesced      write)                         12.17 GB/s |
| Memory Bandwidth (misaligned read      )                         16.33 GB/s |
| Memory Bandwidth (misaligned      write)                         10.35 GB/s |
|-----------------------------------------------------------------------------|
'-----------------------------------------------------------------------------'
yxc@nas:~/OpenCL-Benchmark$
```