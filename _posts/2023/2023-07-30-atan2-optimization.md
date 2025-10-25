---
layout: article
title: C++三角函数优化实践-从std::atan2到NEON加速
tags: cpp
---
## 性能分析


火焰图发现`std::atan2` 存在平顶

## 性能优化


### cv::fastAsan2 函数替换
之前用的`std::atan2` 性能不理想，项目中已经有opencv库，查阅资料`cv::fastAsan2` 更快，具体参考

https://blog.csdn.net/u014629875/article/details/97817442



https://blog.csdn.net/lien0906/article/details/49587759

### cv::fastAsan2 实现原理

#### 核心实现：

```
static const float atan2_p1 = 0.9997878412794807f*(float)(180/CV_PI);
static const float atan2_p3 = -0.3258083974640975f*(float)(180/CV_PI);
static const float atan2_p5 = 0.1555786518463281f*(float)(180/CV_PI);
static const float atan2_p7 = -0.04432655554792128f*(float)(180/CV_PI);

float fastAtan2(float y, float x) {
    float ax = std::abs(x), ay = std::abs(y);
    float a, c, c2;
    
    // 第一步：计算0-90度内的角度
    if(ax >= ay) {
        c = ay/(ax + (float)DBL_EPSILON);
        c2 = c*c;
        a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    } else {
        c = ax/(ay + (float)DBL_EPSILON);
        c2 = c*c;
        a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
    }
    
    // 第二步：根据象限确定最终角度
    if(x < 0) a = 180.f - a;
    if(y < 0) a = 360.f - a;
    return a;
}
```
#### 优化原理

##### 多项式拟合
- 使用7次多项式拟合arctan函数
- 系数预计算并转换为角度制
- 避免直接计算三角函数

##### 化简计算步骤
- 先计算0-90度范围内的角度
- 通过象限判断完成360度映射
- 避免使用条件分支和查表
##### 精度控制
- 精度约为0.3度
- 在实际应用中足够使用
- 用精度换取性能提升
## 性能对比

1. cv::fastAtan2比std::atan2快约2.6倍

2. 在cv::fastAtan2的基础上，使用neon加速3.2倍[带宽x4];

3. 相对原方法std::atan2优化8.5倍左右
