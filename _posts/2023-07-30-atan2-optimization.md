---
layout: articles
title: atan2 优化
tags: c++ atan2 优化
---
## 性能分析


火焰图发现`std::atan2` 存在平顶

## 性能优化

之前用的`std::atan2` 性能不理想，项目中已经有opencv库，查阅资料`cv::fastAsan2` 更快，具体参考

https://blog.csdn.net/u014629875/article/details/97817442



https://blog.csdn.net/lien0906/article/details/49587759

## 性能对比

1. cv::fastAtan2比std::atan2快约2.6倍

2. 在cv::fastAtan2的基础上，使用neon加速3.2倍[带宽x4];

3. 相对原方法std::atan2优化8.5倍左右
## 精度对比