---
layout: article
title: 实车调试
tags: 实车 调试
---

## 0 基本的硬件 实车信息

工控机，

ntpserver

域控芯片板子，

nas

通过工控机，连接到板子上，车辆上电需要10min，启动完成，交换机启动可能会很慢， 能从工控机ping 通板子说明交换机已经启动完成

## 1 时间同步

时间同步是最首要要检查的，时间不同步，感知软件无法启动

时间不同步的话，从板端向上逐级排查

主要顺序是： 板端 -> 工控机 -> ntp server


### 检查板端的时间同步

`data` 命令

`chronyc sources -v`命令

看到每个工控机和板端的时间差， delta 控制在3ms内
```shell
clockdiff 192.168.1.11
```
### 检查 ntpserver信号


ntp-server 依赖gps信号，需要在地面上启动（地库不行），gps 有5颗星信号会比较好

## 2 曝光同步

时间同步的基础上，camera出图曝光同步在5ms


## 3 CAN 数据

ifconfig  查看can0


## 4 软件部署

otaupdate

## 5 实车参数相关

camera 标定

lidar 标定

imu 标定

gnss的rtk账号

## 6 基本检查项目 check_list

### 每一路 数据的fps

### odometry 数据

直接看下实车轨迹，挂P挡可能不正常，需要动一下

### 标定检查

平面网格和地面的平行程度

### 传感器管理中心相关


#### gnss状态
rtk状态

#### camera输出帧率
查看fps


