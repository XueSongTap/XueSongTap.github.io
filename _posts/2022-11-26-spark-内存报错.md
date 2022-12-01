---
layout: articles
title: spark 报错Spark ：【error】System memory 259522560 must be at least 471859200 调整jvm
tags: hadoop spark
---


## 报错

Spark ：【error】System memory 259522560 must be at least 471859200

## 原因
jvm虚拟机 运行内存不足 


## 方法
idea里面run-configuration-vm options
设置`-Xms256m -Xmx1024m`