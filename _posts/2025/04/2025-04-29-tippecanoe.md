---
layout: articles
title: tippecanoe 地图网格切分工具
tags: map
---


Tippecanoe 是一个命令行工具，用于从大型地理空间数据集构建矢量瓦片集。其主要目的是实现尺度无关的可视化，在所有缩放级别上保留原始数据的密度和细节，


简化（RDP）、索引（R-Tree）、裁剪（Clipping）、投影（Mercator）、合并（Union）等

## 概述


### 结构

<img src="/img/250429/tippecanoe_structure.png" alt="alt text" width="500">


#### 1. 命令行工具

* tippecanoe：主要工具，用于生成矢量瓦片
* tile-join：用于合并瓦片和连接 CSV 属性的工具
* tippecanoe-decode：将矢量瓦片转换回 GeoJSON 的工具
* tippecanoe-json-tool：用于操作 GeoJSON 的实用工具
* tippecanoe-enumerate：列出 MBTiles 文件中的瓦片



#### 2. 核心组件
* Input Parsers 输入解析器：处理各种输入格式（GeoJSON、Geobuf、CSV）
* Core Processing 核心处理：管理瓦片生成、几何操作、特征过滤
* Output Generation 输出生成：以 MBTiles 或目录格式创建矢量瓦片


### 数据处理流程 Data Processing Pipeline


Tippecanoe 的数据处理流程包括以下关键阶段：

<img src="/img/250429/tippecanoe_data_process_pipeline.png" alt="alt text" width="500">

1. 输入解析 Parse input：将输入数据转换为内部特征表示（serial_feature）
2. 坐标投影 project Coordinates：在地理坐标和瓦片坐标之间进行转换
3. 特征索引 index feature：按瓦片位置组织特征以进行高效处理
4. 缩放级别处理 zoom：处理多个缩放级别的特征表示
5. 特征处理 Feature Processing：
    - Simplify Geometries 简化几何形状同时保留重要细节
    - Filter Features 基于属性或表达式过滤特征
    - Coalesce Features 合并相似特征以减少冗余
    - Clip to Tile Boundaries 将特征裁剪到瓦片边界
6. 矢量瓦片编码 Encode as MVT：创建压缩的协议缓冲区瓦片
7. 输出生成：写入 MBTiles SQLite 数据库或目录结构



### 3.关键技术特性 Key Technical Features

尤其是针对于大型数据集，


#### 内存管理 Memory Management

- 对于大于可用 RAM 的数据集使用基于文件的中间存储
- 内存映射文件访问以实现高效 I/O
- 字符串池化以减少属性的冗余存储

#### 性能优化 Performance Optimizations

- 使用 -P 选项进行多线程处理
- 瓦片生成的并行处理
- 特征索引的基数Radix 排序
- 高效的几何算法

#### 特征选择策略 Feature Selection Strategies

提供了多种特征选择策略

- 基于密度的选择 (Density-based Selection)
`--drop-densest-as-needed`: 在密度最高的区域优先丢弃特征，以保持瓦片大小在限制之内。 README.md:100-103
- 基于大小的选择 (Size-based Selection)
--drop-smallest-as-needed: 优先丢弃物理尺寸最小的特征（最短的线或最小的多边形）。 README.md:479
- 合并特征 (Feature Coalescing)
--coalesce-densest-as-needed: 尝试将密度最高区域的特征合并到附近的特征中。 
--coalesce-smallest-as-needed: 尝试将最小的特征合并到附近的特征中。

- 基于比例的选择 (Fraction-based Selection)
--drop-fraction-as-needed: 动态丢弃每个缩放级别的一部分特征，以保持大型瓦片在大小限制之内。 README.md:478
- 基于伽马值的选择 (Gamma-based Selection)
-g (gamma): 控制基于密度的点丢弃。伽马值为2会将小于一个像素距离的点数量减少到原来的平方根。 README.md:488
- 聚类 (Clustering)
--cluster-densest-as-needed: 如果瓦片太大，通过增加特征之间的最小间距并从每个组中留下一个占位符特征来减小其大小。

#### 几何处理 Geometry Processing


Tippecanoe 包含几个几何处理流程，用于优化和简化地理数据：

- 线和多边形简化 (Line and Polygon Simplification)
使用 Douglas-Peucker 算法简化线和多边形几何形状。
可以通过 -S 参数控制简化的容差。 README.md:493-494
可以使用 `--no-line-simplification` 禁用简化。

- 共享边界处理 (Shared Boundary Processing)
`--detect-shared-borders`: 检测多个多边形之间共享的边界，并在每个多边形中以相同的方式简化它们。 
- 瓦片边界裁剪 (Tile Boundary Clipping)
控制特征如何被裁剪到瓦片边界。
可以设置缓冲区大小 (`--buffer`)，在相邻瓦片之间复制特征。
可以使用 `--no-clipping` 禁用裁剪。
- 小多边形处理 (Tiny Polygon Handling)
默认情况下，小于最小面积的多边形会被扩散，使得其中一些会被绘制为最小尺寸的正方形，而其他的则不会被绘制，以保持它们应该共同拥有的总面积。
可以使用 `--no-tiny-polygon-reduction` 禁用此功能。
- 特征合并 (Feature Coalescing)
`--coalesce`: 合并具有相同属性的连续特征。
`--reorder`: 重新排序特征，使具有相同属性的特征连续排列。
- 几何修复 (Geometry Correction)
`--detect-longitude-wraparound`: 检测特征内的连续点何时跳到世界的另一侧，并尝试修复几何形状。
处理多边形环方向的选项，用于区分内部和外部多边形环。


## 核心概念 Core Concepts

### 设计理念

Tippecanoe 的设计理念：

使数据具有尺度独立性视图，使得从整个世界到单个建筑物的任何级别，您都能看到数据的密度和纹理，而不是通过删除所谓不重要的特征或聚类或聚合它们而进行简化


### 矢量瓦片和坐标 Tile Structure and Coordinates


#### 瓦片金字塔 和 缩放层级

