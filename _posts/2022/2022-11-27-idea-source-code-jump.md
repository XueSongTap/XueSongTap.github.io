---
layout: article
title: idea 查看源码不跳转.java .scala问题
tags: idea java
---

## 问题

idea ctrl+鼠标点击想查看源码只挑转到.class文件

## 分析

maven下载的时候没有下载source源码


## 解决

setting -> maven-> importing

download选项勾选source，maven再手动下载一次