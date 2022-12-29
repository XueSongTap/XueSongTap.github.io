---
layout: articles
title: houmo.ai day1
tags: intern
---

## 入职


## 确定本周小目标


咱们定个这个星期的小目标：

1. 把apollo的环境搭建建立，编译运行，（代码在develop分支， 现在编译创建容器都是通过./houmoai.sh 这个脚本来执行）


2. 熟悉一下`mainboard、cyber_recored、cyber_monitor` 这几个命令，看看每个通道里面数据是啥样。

3. 跑一下视觉感知  `mainboard -d /apollo/modules/perception/production/dag/dag_streaming_perception_dev_kit_camera.dag`, 根据dag文件去找一下现在的配置文件，看看现在pipeline是啥样。



文档：
https://dig-into-apollo.readthedocs.io/en/latest/how_to_build/readme.html


## apollo 编译


参考：

https://zhuanlan.zhihu.com/p/403590569

https://huhuhang.com/post/machine-learning/baidu-apollo-install


cyber-rt文档
https://cyber-rt.readthedocs.io/en/latest/CyberRT_API_for_Developers.htm


apollo官网的文档

https://developer.apollo.auto/document_cn.html?target=/Apollo-Homepage-Document/Apollo_Doc_CN_6_0/

赵虚左课程
https://www.bilibili.com/video/BV16U4y1U75F/?p=8&spm_id_from=pageDriver&vd_source=7371452b85fe4d187885825b04f8393a

启动apollo docker容器
```bash
./docker/scripts/dev_start.sh
```

进入docker
```bash
./docker/scripts/dev_into.sh
```

这时候进入了当前用户空间的docker，管理权限用sudo

使用自定义的sh，其实也就是`./apollo.sh`魔改
```bash
#! cuda 编译
./apollo.sh build

#！cpu编译
./apollo.sh build_cpu
```
或者其他魔改脚本

启动dreamview仿真环境
```bash
bash scripts/bootstrap.sh
```


## docker权限问题

Docker 添加用户组
将登陆用户加入到docker用户组中
```bash
sudo usermod -aG docker ${USER}
```
更新用户组
```bash
newgrp docker
```