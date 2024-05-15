---
layout: articles
title: opencl benchmark 工具
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





