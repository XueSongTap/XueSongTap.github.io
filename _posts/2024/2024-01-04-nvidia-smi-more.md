---
layout: articles
title: NVIDIA-SMI命令完全指南：GPU监控与管理实用技巧
tags: nvidia-smi gpu
---

## nvidia-smi 技巧


### 查询nvlink互联拓扑：
```
nvidia-smi topo -m
```

以下是几个例子

#### 双卡1080ti，普通家用主板:
```bash
yxc@hua-System-Product-Name:~$ nvidia-smi topo -m
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

输出结果显示了两个GPU（GPU0和GPU1）的拓扑关系，以及它们与CPU的亲和性（CPU Affinity）。

GPU0 和 GPU1 列表示不同的GPU设备。

X 表示自己，即GPU0对应的是自己，不需要与自己建立连接。

PHB 表示连接是通过一个PCIe主机桥（PCIe Host Bridge）完成的，通常这个桥接器是CPU。这意味着两块GPU通过PCIe总线互联，且它们可能直接连接到CPU上的不同PCIe接口上。

CPU Affinity 列显示的是CPU的亲和性，即哪些CPU核心（逻辑处理器）与各个GPU相连。在这个例子中，GPU0和GPU1都与CPU上的核心0到11相连，意味着两个GPU可以被分配到任何这些CPU核上运行任
务。

NUMA Affinity 列显示了GPU与NUMA（Non-Uniform Memory Access）节点的亲和性，但在这里显示为 N/A（不适用），可能是因为使用的主板不支持NUMA或者CPU架构不是NUMA的。


又解释了Legend有哪些：

SYS：连接跨越PCIe以及NUMA节点之间的SMP互连（例如QPI/UPI）。

NODE：连接跨越PCIe以及一个NUMA节点内的PCIe主机桥之间的互连。


PHB：连接跨越PCIe以及一个PCIe主机桥。

PXB：连接跨越多个PCIe桥，但没有跨越PCIe主机桥。

PIX：连接最多跨越一个PCIe桥。

NV#：连接跨越一组绑定的#个NVLink通道。


#### 8卡A100，nvlink：

```bash
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    NIC6    NIC7    NIC8    NIC9    NIC10   NIC11   NIC12   NIC13   NIC14   NIC15   NIC16   CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV12    NV12    NV12    NV12    NV12    NV12    NV12    PXB     PXB     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-103   0               N/A
GPU1    NV12     X      NV12    NV12    NV12    NV12    NV12    NV12    PXB     PXB     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-103   0               N/A
GPU2    NV12    NV12     X      NV12    NV12    NV12    NV12    NV12    SYS     SYS     SYS     SYS     PXB     PXB     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-103   0               N/A
GPU3    NV12    NV12    NV12     X      NV12    NV12    NV12    NV12    SYS     SYS     SYS     SYS     PXB     PXB     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-103   0               N/A
GPU4    NV12    NV12    NV12    NV12     X      NV12    NV12    NV12    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     PXB     PXB     SYS     SYS     SYS     SYS     SYS     0-103   0               N/A
GPU5    NV12    NV12    NV12    NV12    NV12     X      NV12    NV12    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     PXB     PXB     SYS     SYS     SYS     SYS     SYS     0-103   0               N/A
GPU6    NV12    NV12    NV12    NV12    NV12    NV12     X      NV12    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     PXB     PXB     SYS     0-103   0               N/A
GPU7    NV12    NV12    NV12    NV12    NV12    NV12    NV12     X      SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     PXB     PXB     SYS     0-103   0               N/A
NIC0    PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS      X      PIX     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC1    PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     PIX      X      PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC2    PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB      X      PIX     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC3    PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     PIX      X      SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC4    SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS      X      PIX     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC5    SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PIX      X      PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC6    SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB      X      PIX     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC7    SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     PIX      X      SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC8    SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS      X      PIX     PXB     PXB     SYS     SYS     SYS     SYS     SYS
NIC9    SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PIX      X      PXB     PXB     SYS     SYS     SYS     SYS     SYS
NIC10   SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB      X      PIX     SYS     SYS     SYS     SYS     SYS
NIC11   SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     PIX      X      SYS     SYS     SYS     SYS     SYS
NIC12   SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS      X      PIX     PXB     PXB     SYS
NIC13   SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PIX      X      PXB     PXB     SYS
NIC14   SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB      X      PIX     SYS
NIC15   SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PXB     PXB     PIX      X      SYS
NIC16   SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_2
  NIC1: mlx5_3
  NIC2: mlx5_4
  NIC3: mlx5_5
  NIC4: mlx5_6
  NIC5: mlx5_7
  NIC6: mlx5_8
  NIC7: mlx5_9
  NIC8: mlx5_10
  NIC9: mlx5_11
  NIC10: mlx5_12
  NIC11: mlx5_13
  NIC12: mlx5_14
  NIC13: mlx5_15
  NIC14: mlx5_16
  NIC15: mlx5_17
  NIC16: mlx5_bond_0

```

GPU0 到 GPU7 列表示8个GPU设备。

NIC0 则表示网络接口卡，这些都是具体的网络设备。


NV12 表示两个GPU之间通过12条NVLink连接。每个GPU都与其他所有GPU通过NVLink相连，这表明了一个高密度、高速的GPU网络，非常适合执行并行计算密集型任务。

PXB 表示连接跨越了多个PCIe桥（但没有跨越PCIe主机桥），PIX 则表示连接最多跨越了一个PCIe桥。这些连接类型通常用于描述GPU与NICs之间的关系。

SYS 表示连接跨越了PCIe以及SMP互连，包括NUMA节点之间的连接（例如QPI/UPI）。SYS 连接类型通常表明连接是通过系统总线，可能涉及更长的物理距离和潜在的更高延迟。

CPU Affinity 列显示了所有GPU都与CPU的逻辑处理器0到103相连，表示这个系统有多核CPU，并且所有GPU都可以与任何CPU核心通信。

NUMA Affinity 列显示了所有GPU都与NUMA节点0相关联，并且 GPU NUMA ID 列为 N/A，这可能意味着系统可能不支持或没有启用NUMA或者所有的GPU都属于同一个NUMA节点。

NIC Legend 提供了关于网络接口卡的额外信息，说明每个NIC的型号。

总结来说，在这个8卡A100 GPU系统中，所有GPU都通过NVLink互连，并能够与CPU上的所有核心通信。网络接口卡通过PCIe连接到系统


#### 4卡A10, 无nvlink

```
$nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      SYS     SYS     SYS     SYS     SYS     0-127   0               N/A
GPU1    SYS      X      SYS     SYS     SYS     SYS     0-127   0               N/A
GPU2    SYS     SYS      X      SYS     SYS     SYS     0-127   0               N/A
GPU3    SYS     SYS     SYS      X      SYS     SYS     0-127   0               N/A
NIC0    SYS     SYS     SYS     SYS      X      PIX
NIC1    SYS     SYS     SYS     SYS     PIX      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
```

GPU0 到 GPU3 列分别表示系统中的四个GPU设备。

NIC0 和 NIC1 表示网络接口卡，这些是连接到系统网络的设备。在这个例子中，它们是Mellanox的网络接口卡（由NIC Legend说明）。

SYS 表示连接跨越了PCIe以及可能的NUMA节点之间的SMP互连（例如QPI/UPI）。在这个例子中，所有的GPU与其他GPU之间的连接都是 SYS 类型，这意味着它们都通过PCIe总线连接但没有使用高速NVLink。

NUMA Affinity 列显示了所有GPU都与NUMA节点0相关联。GPU NUMA ID 列显示为 N/A，这可能意味着系统可能不支持或没有启用NUMA，或者所有GPU都属于同一个NUMA节点。

NIC Legend 提供了有关网络接口卡的额外信息，mlx5_0 和 mlx5_1 表示两个Mellanox的网络接口卡。

总结来说，这个系统中的四块A10 GPU是通过系统的PCIe总线连接的，并且与CPU的所有核心都有通信能力。虽然它们之间没有NVLink，但这样的配置对于一些不需要密集GPU间通信的计算任务来说是足够的。


### 显示内存、gpu的占用率，动态显示成列表形式，并输出到csv表格中
lms是毫秒为粒度刷新 如果秒的话，参数设置为l就可以了

```bash
nvidia-smi -lms --query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv | tee gpu-log.csv 
```
```
$nvidia-smi -lms --query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv | tee gpu-log.csv 
timestamp, pstate, temperature.gpu, utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB]
2024/01/16 14:44:16.977, P8, 25, 0 %, 0 %, 23028 MiB, 22494 MiB, 21 MiB
2024/01/16 14:44:16.981, P8, 24, 0 %, 0 %, 23028 MiB, 22508 MiB, 7 MiB
2024/01/16 14:44:16.986, P8, 23, 0 %, 0 %, 23028 MiB, 22508 MiB, 7 MiB
2024/01/16 14:44:16.990, P8, 26, 0 %, 0 %, 23028 MiB, 22508 MiB, 7 MiB
2024/01/16 14:44:17.095, P8, 25, 0 %, 0 %, 23028 MiB, 22494 MiB, 21 MiB
```


### PCIE带宽

```bash
nvidia-smi dmon -i 0 -s mutc -d 1 -o TD
```
```bash
$nvidia-smi dmon -i 0 -s mutc -d 1 -o TD
#Date       Time        gpu     fb   bar1   ccpm     sm    mem    enc    dec    jpg    ofa  rxpci  txpci   mclk   pclk 
#YYYYMMDD   HH:MM:SS    Idx     MB     MB     MB      %      %      %      %      %      %   MB/s   MB/s    MHz    MHz 
 20240104   13:54:35      0    239      3      0      6      1      0      0      0      0      5     10   6250   1695 

 20240104   13:54:36      0     74     10      0      0      0      0      0      0      0    217    677   6250   1695 
 20240104   13:54:37      0      7      2      0      3      0      0      0      0      0      0      3   6250   1695 
 20240104   13:54:38      0    478      6      0     13      0      0      0      0      0     44     30   6250   1695 
 20240104   13:54:39      0     67     10      0      0      0      0      0      0      0     30     40   6250   1695 
 20240104   13:54:40      0      7      2      0      3      0      0      0      0      0      0      0   6250   1695 
```


如果需要监控其他的指标，可以用这个命令查看支持的属性
命令：`nvidia-smi --help-query-gpu` 


```bash
$nvidia-smi --help-query-gpu

List of valid properties to query for the switch "--query-gpu":

"timestamp"
The timestamp of when the query was made in format "YYYY/MM/DD HH:MM:SS.msec".

"driver_version"
The version of the installed NVIDIA display driver. This is an alphanumeric string.

Section about vgpu_driver_capability properties
Retrieves information about driver level caps.

"vgpu_driver_capability.heterogenous_multivGPU"
Whether heterogeneuos multi-vGPU is supported by driver.

"count"
The number of NVIDIA GPUs in the system.

"name" or "gpu_name"
The official product name of the GPU. This is an alphanumeric string. For all products.

"serial" or "gpu_serial"
This number matches the serial number physically printed on each board. It is a globally unique immutable alphanumeric value.

"uuid" or "gpu_uuid"
This value is the globally unique immutable alphanumeric identifier of the GPU. It does not correspond to any physical label on the board.

"pci.bus_id" or "gpu_bus_id"
PCI bus id as "domain:bus:device.function", in hex.

"pci.domain"
PCI domain number, in hex.

"pci.bus"
PCI bus number, in hex.

"pci.device"
PCI device number, in hex.

"pci.device_id"
PCI vendor device id, in hex

"pci.sub_device_id"
PCI Sub System id, in hex

Section about vgpu_device_capability properties
Retrieves information about device level caps.

"vgpu_device_capability.fractional_multiVgpu"
Fractional vGPU profiles on this GPU can be used in multi-vGPU configurations.

"vgpu_device_capability.heterogeneous_timeSlice_profile"
Supports concurrent execution of timesliced vGPU profiles of differing types.

"vgpu_device_capability.heterogeneous_timeSlice_sizes"
Supports concurrent execution of timesliced vGPU profiles of differing framebuffer sizes.

"pcie.link.gen.current"
The current PCI-E link generation. These may be reduced when the GPU is not in use. Deprecated, use pcie.link.gen.gpucurrent instead.

"pcie.link.gen.gpucurrent"
The current PCI-E link generation. These may be reduced when the GPU is not in use.

"pcie.link.gen.max"
The maximum PCI-E link generation possible with this GPU and system configuration. For example, if the GPU supports a higher PCIe generation than the system supports then this reports the system PCIe generation.

"pcie.link.gen.gpumax"
The maximum PCI-E link generation supported by this GPU.

"pcie.link.gen.hostmax"
The maximum PCI-E link generation supported by the root port corresponding to this GPU.

"pcie.link.width.current"
The current PCI-E link width. These may be reduced when the GPU is not in use.

"pcie.link.width.max"
The maximum PCI-E link width possible with this GPU and system configuration. For example, if the GPU supports a higher PCIe generation than the system supports then this reports the system PCIe generation.

"index"
Zero based index of the GPU. Can change at each boot.

"display_mode"
A flag that indicates whether a physical display (e.g. monitor) is currently connected to any of the GPU's connectors. "Enabled" indicates an attached display. "Disabled" indicates otherwise.

"display_active"
A flag that indicates whether a display is initialized on the GPU's (e.g. memory is allocated on the device for display). Display can be active even when no monitor is physically attached. "Enabled" indicates an active display. "Disabled" indicates otherwise.

"persistence_mode"
A flag that indicates whether persistence mode is enabled for the GPU. Value is either "Enabled" or "Disabled". When persistence mode is enabled the NVIDIA driver remains loaded even when no active clients, such as X11 or nvidia-smi, exist. This minimizes the driver load latency associated with running dependent apps, such as CUDA programs. Linux only.

"addressing_mode"
A flag that indicates the type of addressing mode enabled for the GPU. Value is either "HMM" or "ATS" or "None". When the mode is HMM, system allocated memory (malloc, mmap) is addressable from the device (GPU), via software-based mirroring of the CPU's page tables, on the GPU. When the mode is ATS, system allocated memory (malloc, mmap) is addressable from the device (GPU), via Address Translation Services. This means that there is (effectively) a single set of page tables, and the CPU and GPU both use them. The mode is None when neither HMM nor ATS is active. Linux only.

"accounting.mode"
A flag that indicates whether accounting mode is enabled for the GPU. Value is either "Enabled" or "Disabled". When accounting is enabled statistics are calculated for each compute process running on the GPU.Statistics can be queried during the lifetime or after termination of the process.The execution time of process is reported as 0 while the process is in running state and updated to actualexecution time after the process has terminated. See --help-query-accounted-apps for more info.

"accounting.buffer_size"
The size of the circular buffer that holds list of processes that can be queried for accounting stats. This is the maximum number of processes that accounting information will be stored for before information about oldest processes will get overwritten by information about new processes.

Section about driver_model properties
On Windows, the TCC and WDDM driver models are supported. The driver model can be changed with the (-dm) or (-fdm) flags. The TCC driver model is optimized for compute applications. I.E. kernel launch times will be quicker with TCC. The WDDM driver model is designed for graphics applications and is not recommended for compute applications. Linux does not support multiple driver models, and will always have the value of "N/A". Only for selected products. Please see feature matrix in NVML documentation.

"driver_model.current"
The driver model currently in use. Always "N/A" on Linux.

"driver_model.pending"
The driver model that will be used on the next reboot. Always "N/A" on Linux.

"vbios_version"
The BIOS of the GPU board.

Section about inforom properties
Version numbers for each object in the GPU board's inforom storage. The inforom is a small, persistent store of configuration and state data for the GPU. All inforom version fields are numerical. It can be useful to know these version numbers because some GPU features are only available with inforoms of a certain version or higher.

"inforom.img" or "inforom.image"
Global version of the infoROM image. Image version just like VBIOS version uniquely describes the exact version of the infoROM flashed on the board in contrast to infoROM object version which is only an indicator of supported features.

"inforom.oem"
Version for the OEM configuration data.

"inforom.ecc"
Version for the ECC recording data.

"inforom.pwr" or "inforom.power"
Version for the power management data.

Section about reset_status properties
GPU reset status information. Reports if there is a GPU reset required or drain and reset recommended to recover from a bad state. 'N/A' indicates that the field is not supported on the current device or device configuration. An error message indicates that retrieving the field failed.

"reset_status.reset_required"
Checks if a GPU reset is required.

"reset_status.drain_and_reset_recommended"
Checks if a GPU drain and reset is recommended.

Section about gom properties
GOM allows to reduce power usage and optimize GPU throughput by disabling GPU features. Each GOM is designed to meet specific user needs.
In "All On" mode everything is enabled and running at full speed.
The "Compute" mode is designed for running only compute tasks. Graphics operations are not allowed.
The "Low Double Precision" mode is designed for running graphics applications that don't require high bandwidth double precision.
GOM can be changed with the (--gom) flag.

"gom.current" or "gpu_operation_mode.current"
The GOM currently in use.

"gom.pending" or "gpu_operation_mode.pending"
The GOM that will be used on the next reboot.

"fan.speed"
The fan speed value is the percent of the product's maximum noise tolerance fan speed that the device's fan is currently intended to run at. This value may exceed 100% in certain cases. Note: The reported speed is the intended fan speed. If the fan is physically blocked and unable to spin, this output will not match the actual fan speed. Many parts do not report fan speeds because they rely on cooling via fans in the surrounding enclosure.

"pstate"
The current performance state for the GPU. States range from P0 (maximum performance) to P12 (minimum performance).

Section about clocks_event_reasons properties
Retrieves information about factors that are reducing the frequency of clocks. If all event reasons are returned as "Not Active" it means that clocks are running as high as possible.

"clocks_event_reasons.supported" or "clocks_throttle_reasons.supported"
Bitmask of supported clock event reasons. See nvml.h for more details.

"clocks_event_reasons.active" or "clocks_throttle_reasons.active"
Bitmask of active clock event reasons. See nvml.h for more details.

"clocks_event_reasons.gpu_idle" or "clocks_throttle_reasons.gpu_idle"
Nothing is running on the GPU and the clocks are dropping to Idle state. This limiter may be removed in a later release.

"clocks_event_reasons.applications_clocks_setting" or "clocks_throttle_reasons.applications_clocks_setting"
GPU clocks are limited by applications clocks setting. E.g. can be changed by nvidia-smi --applications-clocks=

"clocks_event_reasons.sw_power_cap" or "clocks_throttle_reasons.sw_power_cap"
SW Power Scaling algorithm is reducing the clocks below requested clocks because the GPU is consuming too much power. E.g. SW power cap limit can be changed with nvidia-smi --power-limit=

"clocks_event_reasons.hw_slowdown" or "clocks_throttle_reasons.hw_slowdown"
HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This is an indicator of:
 HW Thermal Slowdown: temperature being too high
 HW Power Brake Slowdown: External Power Brake Assertion is triggered (e.g. by the system power supply)
 * Power draw is too high and Fast Trigger protection is reducing the clocks
 * May be also reported during PState or clock change
 * This behavior may be removed in a later release

"clocks_event_reasons.hw_thermal_slowdown" or "clocks_throttle_reasons.hw_thermal_slowdown"
HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This is an indicator of temperature being too high

"clocks_event_reasons.hw_power_brake_slowdown" or "clocks_throttle_reasons.hw_power_brake_slowdown"
HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This is an indicator of External Power Brake Assertion being triggered (e.g. by the system power supply)

"clocks_event_reasons.sw_thermal_slowdown" or "clocks_throttle_reasons.sw_thermal_slowdown"
SW Thermal capping algorithm is reducing clocks below requested clocks because GPU temperature is higher than Max Operating Temp.

"clocks_event_reasons.sync_boost" or "clocks_throttle_reasons.sync_boost"
Sync Boost This GPU has been added to a Sync boost group with nvidia-smi or DCGM in
 * order to maximize performance per watt. All GPUs in the sync boost group
 * will boost to the minimum possible clocks across the entire group. Look at
 * the event reasons for other GPUs in the system to see why those GPUs are
 * holding this one at lower clocks.

Section about memory properties
On-board memory information. Reported total memory is affected by ECC state. If ECC is enabled the total available memory is decreased by several percent, due to the requisite parity bits. The driver may also reserve a small amount of memory for internal use, even without active work on the GPU.

"memory.total"
Total installed GPU memory.

"memory.reserved"
Total memory reserved by the NVIDIA driver and firmware.

"memory.used"
Total memory allocated by active contexts.

"memory.free"
Total free memory.

"compute_mode"
The compute mode flag indicates whether individual or multiple compute applications may run on the GPU.
"0: Default" means multiple contexts are allowed per device.
"1: Exclusive_Thread", deprecated, use Exclusive_Process instead
"2: Prohibited" means no contexts are allowed per device (no compute apps).
"3: Exclusive_Process" means only one context is allowed per device, usable from multiple threads at a time.

"compute_cap"
The CUDA Compute Capability, represented as Major DOT Minor.

Section about utilization properties
Utilization rates report how busy each GPU is over time, and can be used to determine how much an application is using the GPUs in the system.
Note: On MIG-enabled GPUs, querying the utilization of encoder, decoder, jpeg, ofa, gpu, and memory is not currently supported.

"utilization.gpu"
Percent of time over the past sample period during which one or more kernels was executing on the GPU.
The sample period may be between 1 second and 1/6 second depending on the product.

"utilization.memory"
Percent of time over the past sample period during which global (device) memory was being read or written.
The sample period may be between 1 second and 1/6 second depending on the product.

"utilization.encoder"
Percent of time over the past sample period during which one or more kernels was executing on the Encoder Engine.
The sample period may be between 1 second and 1/6 second depending on the product.

"utilization.decoder"
Percent of time over the past sample period during which one or more kernels was executing on the Decoder Engine.
The sample period may be between 1 second and 1/6 second depending on the product.

"utilization.jpeg"
Percent of time over the past sample period during which one or more kernels was executing on the Jpeg Engine.
The sample period may be between 1 second and 1/6 second depending on the product.

"utilization.ofa"
Percent of time over the past sample period during which one or more kernels was executing on the Optical Flow Accelerator Engine.
The sample period may be between 1 second and 1/6 second depending on the product.

Section about encoder.stats properties
Encoder stats report number of encoder sessions, average FPS and average latency in us for given GPUs in the system.

"encoder.stats.sessionCount"
Number of encoder sessions running on the GPU.

"encoder.stats.averageFps"
Average FPS of all sessions running on the GPU.

"encoder.stats.averageLatency"
Average latency in microseconds of all sessions running on the GPU.

Section about ecc.mode properties
A flag that indicates whether ECC support is enabled. May be either "Enabled" or "Disabled". Changes to ECC mode require a reboot. Requires Inforom ECC object version 1.0 or higher.

"ecc.mode.current"
The ECC mode that the GPU is currently operating under.

"ecc.mode.pending"
The ECC mode that the GPU will operate under after the next reboot.

Section about ecc.errors properties
NVIDIA GPUs can provide error counts for various types of ECC errors. Some ECC errors are either single or double bit, where single bit errors are corrected and double bit errors are uncorrectable. Texture memory errors may be correctable via resend or uncorrectable if the resend fails. These errors are available across two timescales (volatile and aggregate). Single bit ECC errors are automatically corrected by the HW and do not result in data corruption. Double bit errors are detected but not corrected. Please see the ECC documents on the web for information on compute application behavior when double bit errors occur. Volatile error counters track the number of errors detected since the last driver load. Aggregate error counts persist indefinitely and thus act as a lifetime counter.

"ecc.errors.corrected.volatile.device_memory"
Errors detected in global device memory.

"ecc.errors.corrected.volatile.dram"
Errors detected in global device memory.

"ecc.errors.corrected.volatile.register_file"
Errors detected in register file memory.

"ecc.errors.corrected.volatile.l1_cache"
Errors detected in the L1 cache.

"ecc.errors.corrected.volatile.l2_cache"
Errors detected in the L2 cache.

"ecc.errors.corrected.volatile.texture_memory"
Parity errors detected in texture memory.

"ecc.errors.corrected.volatile.cbu"
Parity errors detected in CBU.

"ecc.errors.corrected.volatile.sram"
Errors detected in global SRAMs.

"ecc.errors.corrected.volatile.total"
Total errors detected across entire chip.

"ecc.errors.corrected.aggregate.device_memory"
Errors detected in global device memory.

"ecc.errors.corrected.aggregate.dram"
Errors detected in global device memory.

"ecc.errors.corrected.aggregate.register_file"
Errors detected in register file memory.

"ecc.errors.corrected.aggregate.l1_cache"
Errors detected in the L1 cache.

"ecc.errors.corrected.aggregate.l2_cache"
Errors detected in the L2 cache.

"ecc.errors.corrected.aggregate.texture_memory"
Parity errors detected in texture memory.

"ecc.errors.corrected.aggregate.cbu"
Parity errors detected in CBU.

"ecc.errors.corrected.aggregate.sram"
Errors detected in global SRAMs.

"ecc.errors.corrected.aggregate.total"
Total errors detected across entire chip.

"ecc.errors.uncorrected.volatile.device_memory"
Errors detected in global device memory.

"ecc.errors.uncorrected.volatile.dram"
Errors detected in global device memory.

"ecc.errors.uncorrected.volatile.register_file"
Errors detected in register file memory.

"ecc.errors.uncorrected.volatile.l1_cache"
Errors detected in the L1 cache.

"ecc.errors.uncorrected.volatile.l2_cache"
Errors detected in the L2 cache.

"ecc.errors.uncorrected.volatile.texture_memory"
Parity errors detected in texture memory.

"ecc.errors.uncorrected.volatile.cbu"
Parity errors detected in CBU.

"ecc.errors.uncorrected.volatile.sram"
Errors detected in global SRAMs.

"ecc.errors.uncorrected.volatile.total"
Total errors detected across entire chip.

"ecc.errors.uncorrected.aggregate.device_memory"
Errors detected in global device memory.

"ecc.errors.uncorrected.aggregate.dram"
Errors detected in global device memory.

"ecc.errors.uncorrected.aggregate.register_file"
Errors detected in register file memory.

"ecc.errors.uncorrected.aggregate.l1_cache"
Errors detected in the L1 cache.

"ecc.errors.uncorrected.aggregate.l2_cache"
Errors detected in the L2 cache.

"ecc.errors.uncorrected.aggregate.texture_memory"
Parity errors detected in texture memory.

"ecc.errors.uncorrected.aggregate.cbu"
Parity errors detected in CBU.

"ecc.errors.uncorrected.aggregate.sram"
Errors detected in global SRAMs.

"ecc.errors.uncorrected.aggregate.total"
Total errors detected across entire chip.

Section about retired_pages properties
NVIDIA GPUs can retire pages of GPU device memory when they become unreliable. This can happen when multiple single bit ECC errors occur for the same page, or on a double bit ECC error. When a page is retired, the NVIDIA driver will hide it such that no driver, or application memory allocations can access it.

"retired_pages.single_bit_ecc.count" or "retired_pages.sbe"
The number of GPU device memory pages that have been retired due to multiple single bit ECC errors.

"retired_pages.double_bit.count" or "retired_pages.dbe"
The number of GPU device memory pages that have been retired due to a double bit ECC error.

"retired_pages.pending"
Checks if any GPU device memory pages are pending retirement on the next reboot. Pages that are pending retirement can still be allocated, and may cause further reliability issues.

"temperature.gpu"
 Core GPU temperature. in degrees C.

"temperature.gpu.tlimit"
 GPU T.Limit temperature. in degrees C.

"temperature.memory"
 HBM memory temperature. in degrees C.

"power.management"
A flag that indicates whether power management is enabled. Either "Supported" or "[Not Supported]". Requires Inforom PWR object version 3.0 or higher or Kepler device.

"power.draw"
The last measured power draw for the entire board, in watts. On Ampere or newer devices, returns average power draw over 1 sec. On older devices, returns instantaneous power draw. Only available if power management is supported. This reading is accurate to within +/- 5 watts.

"power.draw.average"
The last measured average power draw for the entire board, in watts. Only available if power management is supported and Ampere (except GA100) or newer devices. This reading is accurate to within +/- 5 watts.

"power.draw.instant"
The last measured instant power draw for the entire board, in watts. Only available if power management is supported. This reading is accurate to within +/- 5 watts.

"power.limit"
The software power limit in watts. Set by software like nvidia-smi. On Kepler devices Power Limit can be adjusted using [-pl | --power-limit=] switches.

"enforced.power.limit"
The power management algorithm's power ceiling, in watts. Total board power draw is manipulated by the power management algorithm such that it stays under this value. This value is the minimum of various power limiters.

"power.default_limit"
The default power management algorithm's power ceiling, in watts. Power Limit will be set back to Default Power Limit after driver unload.

"power.min_limit"
The minimum value in watts that power limit can be set to.

"power.max_limit"
The maximum value in watts that power limit can be set to.

"clocks.current.graphics" or "clocks.gr"
Current frequency of graphics (shader) clock.

"clocks.current.sm" or "clocks.sm"
Current frequency of SM (Streaming Multiprocessor) clock.

"clocks.current.memory" or "clocks.mem"
Current frequency of memory clock.

"clocks.current.video" or "clocks.video"
Current frequency of video encoder/decoder clock.

Section about clocks.applications properties
User specified frequency at which applications will be running at. Can be changed with [-ac | --applications-clocks] switches.

"clocks.applications.graphics" or "clocks.applications.gr"
User specified frequency of graphics (shader) clock.

"clocks.applications.memory" or "clocks.applications.mem"
User specified frequency of memory clock.

Section about clocks.default_applications properties
Default frequency at which applications will be running at. Application clocks can be changed with [-ac | --applications-clocks] switches. Application clocks can be set to default using [-rac | --reset-applications-clocks] switches.

"clocks.default_applications.graphics" or "clocks.default_applications.gr"
Default frequency of applications graphics (shader) clock.

"clocks.default_applications.memory" or "clocks.default_applications.mem"
Default frequency of applications memory clock.

Section about clocks.max properties
Maximum frequency at which parts of the GPU are design to run.

"clocks.max.graphics" or "clocks.max.gr"
Maximum frequency of graphics (shader) clock.

"clocks.max.sm" or "clocks.max.sm"
Maximum frequency of SM (Streaming Multiprocessor) clock.

"clocks.max.memory" or "clocks.max.mem"
Maximum frequency of memory clock.

Section about mig.mode properties
A flag that indicates whether MIG mode is enabled. May be either "Enabled" or "Disabled". Changes to MIG mode require a GPU reset.

"mig.mode.current"
The MIG mode that the GPU is currently operating under.

"mig.mode.pending"
The MIG mode that the GPU will operate under after reset.

Section about gsp.mode properties
A flag that indicates whether GSP firmware is enabled.May be either "Enabled" or "Disabled".

"gsp.mode.current"
The current status of GSP firmware.

"gsp.mode.default"
The default status of GSP firmware.

Section about protected_memory properties
On-board protected memory information.

"protected_memory.total"
Total installed GPU conf compute protected memory.

"protected_memory.used"
Total conf compute protected memory allocated by active contexts.

"protected_memory.free"
Total free conf compute protected memory.

"fabric.state"
Current state of GPU fabric registration process.

"fabric.status"
Error status, valid only if gpu fabric registration state is "completed"
```


### nvidia-smi dmon -s m
优点：显示如下，这个fb与nvidia-smi查询的内存占用数量是一样的，具体如下图，
```bash
[yxc01841111@NGIS-QTFCR20230280 /home/yxc01841111]
$nvidia-smi dmon -s m
# gpu     fb   bar1   ccpm 
# Idx     MB     MB     MB 
    0      4      1      0 
    1      4      1      0 
    2      4      1      0 
    3      4      1      0 
    4      4      1      0 
    5      4      1      0 
    6      4      1      0 
    7      4      1      0 
    0      4      1      0 
    1      4      1      0 
    2      4      1      0 
    3      4      1      0 
    4      4      1      0 
    5      4      1      0 
    6      4      1      0 
    7      4      1      0
```
缺点：1、最小粒度为秒，不能更细粒度显示 2、没有找到gpu利用率的参数，不能完全满足监控的需要

### nvidia-smi dmon其他参数解释
    指定显示哪些监控指标（默认为puc），其中：
        p：电源使用情况和温度（pwr：功耗，temp：温度）
        u：GPU使用率（sm：流处理器，mem：显存，enc：编码资源，dec：解码资源）
        c：GPU处理器和GPU内存时钟频率（mclk：显存频率，pclk：处理器频率）
        v：电源和热力异常
        m：FB内存和Bar1内存
        e：ECC错误和PCIe重显错误个数
        t：PCIe读写带宽
### nvidia-smi -a 查看所有参数的情况，

```bash
$nvidia-smi -a 

==============NVSMI LOG==============

Timestamp                                 : Thu Jan  4 14:41:13 2024
Driver Version                            : 535.129.03
CUDA Version                              : 12.2

Attached GPUs                             : 8
GPU 00000000:2F:00.0
    Product Name                          : NVIDIA A100-SXM4-40GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1323121035405
    GPU UUID                              : GPU-faae5314-b87e-87c0-400b-b7cfb9e7b310
    Minor Number                          : 0
    VBIOS Version                         : 92.00.36.00.04
    MultiGPU Board                        : No
    Board ID                              : 0x2f00
    Board Part Number                     : 692-2G506-0200-002
    GPU Part Number                       : 20B0-884-A1
    FRU Part Number                       : N/A
    Module ID                             : 4
    Inforom Version
        Image Version                     : G506.0200.00.04
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 535.129.03
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x2F
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B010DE
        Bus Id                            : 00000000:2F:00.0
        Sub System Id                     : 0x134F10DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 40960 MiB
        Reserved                          : 620 MiB
        Used                              : 4 MiB
        Free                              : 40334 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 22 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 40 C
        Memory Max Operating Temp         : 95 C
    GPU Power Readings
        Power Draw                        : 52.12 W
        Current Power Limit               : 400.00 W
        Requested Power Limit             : 400.00 W
        Default Power Limit               : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 1215 MHz
        Video                             : 585 MHz
    Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Default Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1215 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 706.250 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes                             : None

GPU 00000000:35:00.0
    Product Name                          : NVIDIA A100-SXM4-40GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1323121013540
    GPU UUID                              : GPU-20af7308-7522-d66f-fe01-fb6f0551a4b9
    Minor Number                          : 1
    VBIOS Version                         : 92.00.36.00.04
    MultiGPU Board                        : No
    Board ID                              : 0x3500
    Board Part Number                     : 692-2G506-0200-002
    GPU Part Number                       : 20B0-884-A1
    FRU Part Number                       : N/A
    Module ID                             : 2
    Inforom Version
        Image Version                     : G506.0200.00.04
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 535.129.03
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x35
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B010DE
        Bus Id                            : 00000000:35:00.0
        Sub System Id                     : 0x134F10DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 40960 MiB
        Reserved                          : 620 MiB
        Used                              : 4 MiB
        Free                              : 40334 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 22 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 22 C
        Memory Max Operating Temp         : 95 C
    GPU Power Readings
        Power Draw                        : 53.16 W
        Current Power Limit               : 400.00 W
        Requested Power Limit             : 400.00 W
        Default Power Limit               : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 1215 MHz
        Video                             : 585 MHz
    Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Default Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1215 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 712.500 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes                             : None

GPU 00000000:62:00.0
    Product Name                          : NVIDIA A100-SXM4-40GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1323121015220
    GPU UUID                              : GPU-55be19c4-22e1-4622-c7c5-a617a8001574
    Minor Number                          : 2
    VBIOS Version                         : 92.00.36.00.04
    MultiGPU Board                        : No
    Board ID                              : 0x6200
    Board Part Number                     : 692-2G506-0200-002
    GPU Part Number                       : 20B0-884-A1
    FRU Part Number                       : N/A
    Module ID                             : 8
    Inforom Version
        Image Version                     : G506.0200.00.04
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 535.129.03
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x62
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B010DE
        Bus Id                            : 00000000:62:00.0
        Sub System Id                     : 0x134F10DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 40960 MiB
        Reserved                          : 620 MiB
        Used                              : 4 MiB
        Free                              : 40334 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 21 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 38 C
        Memory Max Operating Temp         : 95 C
    GPU Power Readings
        Power Draw                        : 51.79 W
        Current Power Limit               : 400.00 W
        Requested Power Limit             : 400.00 W
        Default Power Limit               : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 1215 MHz
        Video                             : 585 MHz
    Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Default Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1215 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 712.500 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes                             : None

GPU 00000000:67:00.0
    Product Name                          : NVIDIA A100-SXM4-40GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1323121015264
    GPU UUID                              : GPU-8029ef2b-47a3-15f7-cca6-5777783049b4
    Minor Number                          : 3
    VBIOS Version                         : 92.00.36.00.04
    MultiGPU Board                        : No
    Board ID                              : 0x6700
    Board Part Number                     : 692-2G506-0200-002
    GPU Part Number                       : 20B0-884-A1
    FRU Part Number                       : N/A
    Module ID                             : 6
    Inforom Version
        Image Version                     : G506.0200.00.04
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 535.129.03
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x67
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B010DE
        Bus Id                            : 00000000:67:00.0
        Sub System Id                     : 0x134F10DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 40960 MiB
        Reserved                          : 620 MiB
        Used                              : 4 MiB
        Free                              : 40334 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 22 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 44 C
        Memory Max Operating Temp         : 95 C
    GPU Power Readings
        Power Draw                        : 52.33 W
        Current Power Limit               : 400.00 W
        Requested Power Limit             : 400.00 W
        Default Power Limit               : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 1215 MHz
        Video                             : 585 MHz
    Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Default Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1215 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 725.000 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes                             : None

GPU 00000000:9F:00.0
    Product Name                          : NVIDIA A100-SXM4-40GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1322921084044
    GPU UUID                              : GPU-a62346e8-6045-319e-845d-0bf10385caaf
    Minor Number                          : 4
    VBIOS Version                         : 92.00.36.00.04
    MultiGPU Board                        : No
    Board ID                              : 0x9f00
    Board Part Number                     : 692-2G506-0200-002
    GPU Part Number                       : 20B0-884-A1
    FRU Part Number                       : N/A
    Module ID                             : 7
    Inforom Version
        Image Version                     : G506.0200.00.04
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 535.129.03
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x9F
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B010DE
        Bus Id                            : 00000000:9F:00.0
        Sub System Id                     : 0x134F10DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 40960 MiB
        Reserved                          : 620 MiB
        Used                              : 4 MiB
        Free                              : 40334 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 23 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 22 C
        Memory Max Operating Temp         : 95 C
    GPU Power Readings
        Power Draw                        : 53.05 W
        Current Power Limit               : 400.00 W
        Requested Power Limit             : 400.00 W
        Default Power Limit               : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 1215 MHz
        Video                             : 585 MHz
    Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Default Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1215 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 706.250 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes                             : None

GPU 00000000:A5:00.0
    Product Name                          : NVIDIA A100-SXM4-40GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1323121035431
    GPU UUID                              : GPU-926d0f1b-0ac4-e9cb-ec83-520d975c12cf
    Minor Number                          : 5
    VBIOS Version                         : 92.00.36.00.04
    MultiGPU Board                        : No
    Board ID                              : 0xa500
    Board Part Number                     : 692-2G506-0200-002
    GPU Part Number                       : 20B0-884-A1
    FRU Part Number                       : N/A
    Module ID                             : 5
    Inforom Version
        Image Version                     : G506.0200.00.04
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 535.129.03
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0xA5
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B010DE
        Bus Id                            : 00000000:A5:00.0
        Sub System Id                     : 0x134F10DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 40960 MiB
        Reserved                          : 620 MiB
        Used                              : 4 MiB
        Free                              : 40334 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 22 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 22 C
        Memory Max Operating Temp         : 95 C
    GPU Power Readings
        Power Draw                        : 55.01 W
        Current Power Limit               : 400.00 W
        Requested Power Limit             : 400.00 W
        Default Power Limit               : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 1215 MHz
        Video                             : 585 MHz
    Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Default Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1215 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 712.500 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes                             : None

GPU 00000000:C8:00.0
    Product Name                          : NVIDIA A100-SXM4-40GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1323121035302
    GPU UUID                              : GPU-aac22a4c-e9dd-6fb5-46e5-0f14b9374044
    Minor Number                          : 6
    VBIOS Version                         : 92.00.36.00.04
    MultiGPU Board                        : No
    Board ID                              : 0xc800
    Board Part Number                     : 692-2G506-0200-002
    GPU Part Number                       : 20B0-884-A1
    FRU Part Number                       : N/A
    Module ID                             : 3
    Inforom Version
        Image Version                     : G506.0200.00.04
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 535.129.03
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0xC8
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B010DE
        Bus Id                            : 00000000:C8:00.0
        Sub System Id                     : 0x134F10DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 1000 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 40960 MiB
        Reserved                          : 620 MiB
        Used                              : 4 MiB
        Free                              : 40334 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 22 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 23 C
        Memory Max Operating Temp         : 95 C
    GPU Power Readings
        Power Draw                        : 52.17 W
        Current Power Limit               : 400.00 W
        Requested Power Limit             : 400.00 W
        Default Power Limit               : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 1215 MHz
        Video                             : 585 MHz
    Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Default Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1215 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 712.500 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes                             : None

GPU 00000000:CD:00.0
    Product Name                          : NVIDIA A100-SXM4-40GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1322921056183
    GPU UUID                              : GPU-1a011247-3834-ff88-6682-d0776a7d7a1e
    Minor Number                          : 7
    VBIOS Version                         : 92.00.36.00.04
    MultiGPU Board                        : No
    Board ID                              : 0xcd00
    Board Part Number                     : 692-2G506-0200-002
    GPU Part Number                       : 20B0-884-A1
    FRU Part Number                       : N/A
    Module ID                             : 1
    Inforom Version
        Image Version                     : G506.0200.00.04
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 535.129.03
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0xCD
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B010DE
        Bus Id                            : 00000000:CD:00.0
        Sub System Id                     : 0x134F10DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 40960 MiB
        Reserved                          : 620 MiB
        Used                              : 4 MiB
        Free                              : 40334 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 22 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 36 C
        Memory Max Operating Temp         : 95 C
    GPU Power Readings
        Power Draw                        : 53.10 W
        Current Power Limit               : 400.00 W
        Requested Power Limit             : 400.00 W
        Default Power Limit               : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 1215 MHz
        Video                             : 585 MHz
    Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Default Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1215 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 712.500 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes                             : None

```
5、nvidia-smi -l 按秒刷新追加显示



nvidia-smi 使用注意点：

nvidia-smi 只是统计 sm 的加权平均， 也就是gpu-utils 的统计

nvidia-smi 的采集粒度偏大


指定板卡id，查看gpu状态，nvidia-smi -i 0


查看gpu 的详细状态信息
nvidia-smi -q
nvidia-smi -i 0 -q

```bash
$nvidia-smi -i 0 -q

==============NVSMI LOG==============

Timestamp                                 : Thu Jan  4 14:45:46 2024
Driver Version                            : 535.129.03
CUDA Version                              : 12.2

Attached GPUs                             : 8
GPU 00000000:2F:00.0
    Product Name                          : NVIDIA A100-SXM4-40GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Ampere
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Disabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1323121035405
    GPU UUID                              : GPU-faae5314-b87e-87c0-400b-b7cfb9e7b310
    Minor Number                          : 0
    VBIOS Version                         : 92.00.36.00.04
    MultiGPU Board                        : No
    Board ID                              : 0x2f00
    Board Part Number                     : 692-2G506-0200-002
    GPU Part Number                       : 20B0-884-A1
    FRU Part Number                       : N/A
    Module ID                             : 4
    Inforom Version
        Image Version                     : G506.0200.00.04
        OEM Object                        : 2.0
        ECC Object                        : 6.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : N/A
        Latest Duration                   : N/A
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : 535.129.03
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x2F
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x20B010DE
        Bus Id                            : 00000000:2F:00.0
        Sub System Id                     : 0x134F10DE
        GPU Link Info
            PCIe Generation
                Max                       : 4
                Current                   : 4
                Device Current            : 4
                Device Max                : 4
                Host Max                  : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 0 KB/s
        Rx Throughput                     : 0 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : N/A
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 40960 MiB
        Reserved                          : 620 MiB
        Used                              : 4 MiB
        Free                              : 40334 MiB
    BAR1 Memory Usage
        Total                             : 65536 MiB
        Used                              : 1 MiB
        Free                              : 65535 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 640 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 22 C
        GPU T.Limit Temp                  : N/A
        GPU Shutdown Temp                 : 92 C
        GPU Slowdown Temp                 : 89 C
        GPU Max Operating Temp            : 85 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 40 C
        Memory Max Operating Temp         : 95 C
    GPU Power Readings
        Power Draw                        : 52.45 W
        Current Power Limit               : 400.00 W
        Requested Power Limit             : 400.00 W
        Default Power Limit               : 400.00 W
        Min Power Limit                   : 100.00 W
        Max Power Limit                   : 400.00 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 1215 MHz
        Video                             : 585 MHz
    Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Default Applications Clocks
        Graphics                          : 1095 MHz
        Memory                            : 1215 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1410 MHz
        SM                                : 1410 MHz
        Memory                            : 1215 MHz
        Video                             : 1290 MHz
    Max Customer Boost Clocks
        Graphics                          : 1410 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 706.250 mV
    Fabric
        State                             : N/A
        Status                            : N/A
    Processes                             : None
```

### 查看gpu 的编码器状态
```bash
nvidia-smi -q | grep -i enc

nvidia-smi -i 0 -q | grep -i enc
```
```bash
$nvidia-smi -i 0 -q | grep -i enc
    Persistence Mode                      : Disabled
        Encoder                           : 0 %
    Encoder Stats
        Average Latency                   : 0
        Average Latency                   : 0
```

### nvidia-smi dmon 相关

设备监控命令，以滚动条形式显示GPU设备统计信息
nvidia-smi dmon



GPU统计信息以一行的滚动格式显示，要监控的指标可以基于终端窗口的宽度进行调整。 监控所有的GPU
#### 附加选项：
nvidia-smi dmon -i xxx
用逗号分隔GPU索引，PCI总线ID或UUID

nvidia-smi dmon -d xxx
指定刷新时间（默认为1秒）

nvidia-smi dmon -c xxx
显示指定数目的统计信息并退出

nvidia-smi dmon -s xxx
指定显示哪些监控指标（默认为puc），其中：
p：电源使用情况和温度（pwr：功耗，temp：温度）
u：GPU使用率（sm：流处理器，mem：显存，enc：编码资源，dec：解码资源）
c：GPU处理器和GPU内存时钟频率（mclk：显存频率，pclk：处理器频率）
v：电源和热力异常
m：FB内存和Bar1内存
e：ECC错误和PCIe重显错误个数
t：PCIe读写带宽

nvidia-smi dmon –o D/T
指定显示的时间格式D：YYYYMMDD，THH:MM:SS

nvidia-smi dmon –f xxx
将查询的信息输出到具体的文件中，不在终端显示


### nvidia-smi pmon 相关
进程监控命令，以滚动条形式显示GPU进程状态信息。
nvidia-smi pmon
GPU进程统计信息以一行的滚动格式显示，此工具列出了GPU所有进程的统计信息。要监控的指标可以基于终端窗口的宽度进行调整。
附加选项：
nvidia-smi pmon -i xxx
用逗号分隔GPU索引，PCI总线ID或UUID

nvidia-smi pmon -d xxx
指定刷新时间（默认为1秒，最大为10秒）

nvidia-smi pmon -c xxx
显示指定数目的统计信息并退出

nvidia-smi pmon -s xxx
指定显示哪些监控指标（默认为u），其中：
u：GPU使用率
m：FB内存使用情况