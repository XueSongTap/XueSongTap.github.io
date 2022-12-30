---
layout: articles
title: houmo.ai day3
tags: intern
---

## 尝试写dag

cyber-rt 模块

一个launch文件有多个module

每个module包含一个dag文件

一个dag文件对应一个或者多个components


### cyber_launch

```
<cyber>
    <module>
        <name>planning</name>   \\ module名称
        <dag_conf>/apollo/modules/planning/dag/planning.dag</dag_conf>   \\ module的dag文件
        <process_name>planning</process_name>   \\ 指定调度文件
    </module>
</cyber>
```

module 用于区分模块
name 模块名称，主要用来cyber_launch启动的时候显示名称
dag_conf module模块对应的dag文件
process_name 指定module的调度文件，如果找不到则会提示


### dag 文件


```
module_config {
    module_library : "/apollo/bazel-bin/modules/drivers/lidar/velodyne/driver/libvelodyne_driver_component.so"
    // 模块的so文件，用于加载到内存
    # 128
    components {  // 第1个组件
      class_name : "VelodyneDriverComponent"
      config {
        name : "velodyne_128_driver"   // 名称必须不一样
        // 配置文件
        config_file_path : "/apollo/modules/drivers/lidar/velodyne/conf/velodyne128_conf.pb.txt"
      }
    }
    # 16_front_up
    components {  // 第2个组件
      class_name : "VelodyneDriverComponent"
      config {
        name : "velodyne_16_front_up_driver"
        config_file_path : "/apollo/modules/drivers/lidar/velodyne/conf/velodyne16_front_up_conf.pb.txt"
      }
    }
}

module_config {
    module_library : "/apollo/bazel-bin/modules/drivers/lidar/velodyne/parser/libvelodyne_convert_component.so"
    // 模块的so文件，用于加载到内存
    # 128
    components {
      class_name : "VelodyneConvertComponent"
      config {
        name : "velodyne_128_convert"   
        config_file_path : "/apollo/modules/drivers/lidar/velodyne/conf/velodyne128_conf.pb.txt"
        readers {
          channel: "/apollo/sensor/lidar128/Scan"
        }
      }
    }
    # 16_front_up_center
    components {
      class_name : "VelodyneConvertComponent"
      config {
        name : "velodyne_16_front_up_convert"
        config_file_path : "/apollo/modules/drivers/lidar/velodyne/conf/velodyne16_front_up_conf.pb.txt"
        readers {
          channel: "/apollo/sensor/lidar16/front/up/Scan"
        }
      }
    }
}
```

dag文件有一个或者多个module_config，而每个module_config中对应一个或者多个components。

参考：https://zhuanlan.zhihu.com/p/350355878


## cyber框架
参考：https://zhuanlan.zhihu.com/p/91322837

cyber提供的功能概括起来包括2方面：

消息队列 - 主要作用是接收和发送各个节点的消息，涉及到消息的发布、订阅以及消息的buffer缓存等。

实时调度 - 主要作用是调度处理上述消息的算法模块，保证算法模块能够实时调度处理消息。

除了这2方面的工作，cyber还需要提供以下2部分的工作：

用户接口 - 提供灵活的用户接口

工具 - 提供一系列的工具，例如bag包播放，点云可视化，消息监控等


### cyber入口

cyber的入口在"cyber/mainboard"目录中：

```bash
├── mainboard.cc           // 主函数
├── module_argument.cc     // 模块输入参数
├── module_argument.h
├── module_controller.cc   // 模块加载，卸载
└── module_controller.h
```


## apollo感知模块配置文件
参考：https://zhuanlan.zhihu.com/p/515360049


## apollo 红绿灯感知模块为例子

参考 

https://cloud.tencent.com/developer/article/1795359


参考 https://zhuanlan.zhihu.com/p/476879068#: