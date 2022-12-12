---
layout: articles
title: uart 串口serial 读写，非阻塞
tags: c uart serial 串口 非阻塞
---

# 读写uart板卡，imu模块

## 打开串口

指定Baudrate，遍历寻找指定的波特率

`termios`库
termios是在POSIX规范中定义的标准接口，表示终端设备，包括虚拟终端、串口等。串口通过termios进行配置。

open函数尝试dev是否可以打开

`tcgetattr`获取终端相关的参数

`cfsetispeed` 设置波特率

`tcsetattr`参数设置

`tcflush`清除收到的数据

## 读取
存入buf指针指向的区域

