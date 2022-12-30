---
layout: articles
title: houmo.ai day2
tags: intern
---


## 任务

2. 熟悉一下`mainboard、cyber_recored、cyber_monitor` 这几个命令，看看每个通道里面数据是啥样。


3. 跑一下视觉感知  `mainboard -d /apollo/modules/perception/production/dag/dag_streaming_perception_dev_kit_camera.dag`, 根据dag文件去找一下现在的配置文件，看看现在pipeline是啥样。


/apollo/modules/perception/production/dag/dag_streaming_perception_dev_kit_camera.dag
```xml
module_config {
  module_library : "/apollo/bazel-bin/modules/perception/onboard/component/libperception_component_camera.so"
  components {
    class_name : "FusionCameraDetectionComponent"
    config {
      name: "FusionCameraComponent"
      config_file_path: "/apollo/modules/perception/production/conf/perception/camera/fusion_camera_detection_component.pb.txt"
      flag_file_path: "/apollo/modules/perception/production/conf/perception/perception_common.flag"
    }
  }
}
```


"/apollo/modules/perception/production/conf/perception/camera/fusion_camera_detection_component.pb.txt"
```
camera_names: "front_6mm,front_12mm"
input_camera_channel_names : "/apollo/sensor/camera/front_6mm/image,/apollo/sensor/camera/front_12mm/image"
timestamp_offset : 0.0
camera_obstacle_perception_conf_dir : "/apollo/modules/perception/production/conf/perception/camera"
camera_obstacle_perception_conf_file : "obstacle.pt"
frame_capacity : 15
image_channel_num : 3
enable_undistortion : false
enable_visualization : false
output_final_obstacles : true
output_obstacles_channel_name : "/apollo/perception/obstacles"
camera_perception_viz_message_channel_name : "/perception/inner/camera_viz_msg"
prefused_channel_name : "/perception/inner/PrefusedObjects"
default_camera_pitch : 0.0
default_camera_height : 1.5
lane_calibration_working_sensor_name : "front_6mm"
calibrator_method : "LaneLineCalibrator"
calib_service_name : "OnlineCalibrationService"
run_calib_service : true
output_camera_debug_msg : false
camera_debug_channel_name : "/perception/camera_debug"
ts_diff : 0.1
visual_debug_folder : "/apollo/debug_output"
visual_camera : "front_6mm"
write_visual_img : false
enable_cipv : false
debug_level : 0
```

/apollo/modules/perception/production/conf/perception/perception_common.flag

```
###########################################################################
# Flags from sensor_manager

# SensorManager config file
# type: string
# default:
--obs_sensor_meta_path=./modules/perception/production/data/perception/common/sensor_meta.pt

# The intrinsics/extrinsics dir
# type: string
# default:
--obs_sensor_intrinsic_path=/apollo/modules/perception/data/params
```

./modules/perception/production/data/perception/common/sensor_meta.pt
```
sensor_meta {
    name: "velodyne128"
    type: VELODYNE_128
    orientation: PANORAMIC
}
sensor_meta {
    name: "velodyne64"
    type: VELODYNE_64
    orientation: PANORAMIC
}
sensor_meta {
    name: "velodyne16"
    type: VELODYNE_16
    orientation: FRONT
}

sensor_meta {
    name: "radar_front"
    type: LONG_RANGE_RADAR
    orientation: FRONT
}

sensor_meta {
    name: "radar_rear"
    type: LONG_RANGE_RADAR
    orientation: REAR
}

sensor_meta {
    name: "front_6mm"
    type: MONOCULAR_CAMERA
    orientation: FRONT
}

sensor_meta {
    name: "front_12mm"
    type: MONOCULAR_CAMERA
    orientation: FRONT
}

sensor_meta {
    name: "camera_front_left"
    type: MONOCULAR_CAMERA
    orientation: FRONT
}

sensor_meta {
    name: "camera_left_front"
    type: MONOCULAR_CAMERA
    orientation: LEFT
}

sensor_meta {
    name: "camera_left_rear"
    type: MONOCULAR_CAMERA
    orientation: LEFT
}

sensor_meta {
    name: "camera_right_front"
    type: MONOCULAR_CAMERA
    orientation: RIGHT
}

sensor_meta {
    name: "camera_right_rear"
    type: MONOCULAR_CAMERA
    orientation: RIGHT
}

sensor_meta {
    name: "neolix_multi_lidar"
    type: VELODYNE_16
    orientation: FRONT
}

#sensor_meta {
#    name: "onsemi_wide"
#    type: MONOCULAR_CAMERA
#    orientation: FRONT
#}

#sensor_meta {
#    name: "onsemi_traffic"
#    type: MONOCULAR_CAMERA
#    orientation: FRONT
#}
```

## vscode docker链接问题

vscode可以使用docker remote插件远程开发服务器山的docker

但是这时候连接的用户最好有docker权限，否则查看不了docker容器

我的用户是后赋予的docker权限，重新登陆vscode remote，还是无法查看所有的docker

这时候需要查找vscode 的进程全部杀掉，再重新登录

```bash
ps -ef | grep xiaochuan/.vscode | awk '{print $2}' | xargs kill -9
```

