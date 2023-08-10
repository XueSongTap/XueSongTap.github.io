---
layout: articles
title: win下配准spark java.io.IOException: Could not locate executable null\bin\winutils.exe in the Hadoop binaries 报错
tags: hadoop spark
---



## 环境
win10 jdk1.8 scale spark2.11


## 代码

```java
package com.atguigu.spark.core.wc

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext};
object Spark01_WorldCount {
    def main(args:Array[String]) : Unit = {

        val sparConf = new SparkConf().setMaster("local").setAppName("WorldCount")
        val sc = new SparkContext(sparConf)

        // TODO 执行业务操作



        // TODO 关闭连接
        sc.stop()



    }
}
```

## 报错信息
```
java.io.IOException: Could not locate executable null\bin\winutils.exe in the Hadoop binaries
```
查了下winutils竟然是hadoop的东西

跑spark之前需要配置下

## 解决

### hadoop安装
下载hadoop bin文件

https://dlcdn.apache.org/hadoop/common/

解压缩到任意目录

win下因为比较特殊，需要添加winutils
https://github.com/cdarlint/winutils

直接下载文件添加到`hadoop/bin`目录

复制一份hadoop.dll放到C:\Windows\System32下。


### 环境变量配置

只在idea下用的话，可以直接配置idea的变量

在 IDEA 中配置 Run Configuration，添加 HADOOP_HOME 变量

![Run conf](/assets/images/idea_run_conf.png)

![Run conf](/assets/images/hadoop_home.png)