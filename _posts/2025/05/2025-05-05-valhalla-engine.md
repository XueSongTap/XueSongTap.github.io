---
layout: articles
title: valhalla 算路引擎
tags: map
---


基于openstreetmap数据的算路引擎，括路径规划、等时线计算、高程采样、地图匹配和旅行优化


### 核心特性

Tiled Hierarchical Data Structure: Allows for small memory footprint on constrained devices, enables offline routing, and provides a means for regional extracts and partial updates. 支持在资源受限设备上使用较小内存，便于离线路由，并支持区域提取与局部更新。


Dynamic Runtime Costing: Edge and vertex costing via a plugin architecture, allowing for customized routing behavior and alternate route generation.通过插件架构实现边和节点的代价评估，支持自定义路由行为和备选路径生成。


Plugin-based Narrative Generation: Customizable turn-by-turn instruction generation that can be tailored to administrative areas or target locales.可定制的逐路口导航指令生成逻辑，支持按行政区或目标地区进行本地化。



Multi-modal Routing: Supports mixing auto, pedestrian, bike, and public transportation in the same route, along with time-constrained routing. 多模式路由：支持将汽车、步行、自行车和公共交通混合于同一路径中，同时支持时间约束的路径规划


### 核心组件 Core Components



<img src="/img/250505/valhalla-core-components.png" alt="alt text" width="500">


- **Midgard**: Basic geographic and geometric algorithms 地理/几何数据结构

- Baldr: Base data structures for accessing and caching tiled route data 路由数据结构/算法 

- Sif: 边缘/转换的成本计算 
- Skadi: 数字高程模型使用

- Tyr: API 接口层


- Loki: - 位置服务, 处理请求中的位置信息，将其与图形边缘关联 mkdocs.yml:90
- Thor - 路径计算,现路由算法，计算路径 mkdocs.yml:112-115
- Odin - 叙述生成, 生成路线导航指令和叙述 mkdocs.yml:92

- Meili - 地图匹配 mkdocs.yml:96-103

- Mjolnir - 路由图/瓦片生成 mkdocs.yml:104-111

