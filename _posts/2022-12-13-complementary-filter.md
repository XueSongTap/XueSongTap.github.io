---
layout: articles
title: 互补滤波的实现
tags: complementary_filter
---


## 概念

加速计 陀螺仪输出的参数

### 创造命名空间

通过ros的 nodehandler
创造命名空间

设置监听`ros::spin()`


### 读取陀螺仪、加速计读数

读取陀螺仪，加速计读书。进行数据update


### checksum 类似crc校验

### 调试参数进行滤波

主要是参数进行滤波



### ros topic 发布参数

延时问题
