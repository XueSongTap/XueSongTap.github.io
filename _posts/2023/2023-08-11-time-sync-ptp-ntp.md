---
layout: article
title: 时间同步技术全解析：从NTP到TSN
tags: time sync ntp ptp
---

## 1. 时间同步基础概念
### 1.1 为什么需要时间同步
- 分布式系统协调
- 多媒体同步播放
- 工业控制精确timing
- 传感器数据融合

## 2. AVB 体系/协议框架
### 2.1 AVB 简介
http://blog.coderhuo.tech/2020/03/22/AVB_summury/

AVB被称为时间敏感网络，它主要解决两个问题
- 网络传输问题：带宽预留
- 多媒体同步问题：时钟恢复与播放同步

### 2.2 AVB协议族中的gPTP协议
AVB域内的每一个节点都是一个时钟，由以下两个角色组成：
- 一个主时钟（Grandmaster Clock），它是标准时间的来源；
- 其他的都是从时钟（Slave Clock），它们必须把自己的时间和主时钟调整一致

http://blog.coderhuo.tech/2020/04/05/gptp_summury/#2-%e4%b8%bb%e6%97%b6%e9%92%9f%e9%80%89%e5%8f%96

### gPTP 和PTP关系

gPTP对PTPv2进行了简化，固定了特性选项的选择，gPTP相当于是PTPv2的一个特定profile。

## NTP相关
1）NTP通过软件实现，只需要一次握手

2）PTP通过软件和硬件实现，需要两次握手

https://www.jianshu.com/p/8bb29838ae1b

https://blog.srefan.com/2017/07/ntp-protocol/

## SPI 时间同步相关

## 以太网 autosar 时间同步相关

https://new.qq.com/rain/a/20220601A050DG00

## TSN 时间敏感网络相关

https://www.yisu.com/zixun/13613.html

## PPS/摄像头同步

主要是秒脉冲同步

## 相机同步 

秒脉冲/PWM(f=1Hz) 

主要目的是控制各个串解器从而控制多路相机同步曝光，保证多个相机同步出图

## 参考

https://zhuanlan.zhihu.com/p/288467842