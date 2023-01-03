---
layout: articles
title: apollo感知模块调试方法
tags: apollo debug perception
---

# apollo感知模块调试方法

## 准备运行环境

Git cone repo，并  git checkout develop分支

git clone "http://gerrit.houmo.ai/ai_solution/autopilot/apollo" && (cd "apollo" && mkdir -p .git/hooks && curl -Lo `git rev-parse --git-dir`/hooks/commit-msg http://gerrit.houmo.ai/tools/hooks/commit-msg; chmod +x `git rev-parse --git-dir`/hooks/commit-msg)


bash docker/script/dev_start.sh bash docker/script/dev_into.sh 

bash build.sh

## 调整配置文件

### 根据需要调试的内容和车型选择dag配置文件

例如

/apollo/modules/perception/production/dag/dag_streaming_perception_dev_kit_camera.dag
可以从这边文章中找到dreamview启动的对应dag文件 http://confluence.houmo.ai/pages/viewpage.action?pageId=14453086


### dag中找到相应配置文件

从 dag文件中，如config_file_path找到配置文件fusion_camera_detection_component.pb.txt

```
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

### 配置pb文件

对配置PB文件fusion_camera_detection_component.pb.txt中，enable_visualization，output_camera_debug_msg，visual_debug_folder ，write_visual_img ，如下设置。

```

camera_names: "front_6mm"
input_camera_channel_names : "/apollo/sensor/camera/front_6mm/image"
timestamp_offset : 0.0
camera_obstacle_perception_conf_dir : "/apollo/modules/perception/production/conf/perception/camera"
camera_obstacle_perception_conf_file : "obstacle.pt"
frame_capacity : 15
image_channel_num : 3
enable_undistortion : false
enable_visualization : true
output_final_obstacles : true
output_obstacles_channel_name : "/apollo/perception/obstacles2"
camera_perception_viz_message_channel_name : "/perception/inner/camera_viz_msg"
prefused_channel_name : "/perception/inner/PrefusedObjects"
default_camera_pitch : 0.0
default_camera_height : 1.5
lane_calibration_working_sensor_name : "front_6mm"
calibrator_method : "LaneLineCalibrator"
calib_service_name : "OnlineCalibrationService"
run_calib_service : true
output_camera_debug_msg : true
camera_debug_channel_name : "/perception/camera_debug"
ts_diff : 0.1
visual_debug_folder : "/apollo/debug_output"
visual_camera : "front_6mm"
write_visual_img : true
enable_cipv : true
debug_level : 4
```

### 增加中间结果输出

在上步的XXX_component.pb.txt中找到conf_dir和conf_file，修改对应的*.pt文件，如"obstacle.pt"

```
camera_obstacle_perception_conf_dir : "/apollo/modules/perception/production/conf/perception/camera"
camera_obstacle_perception_conf_file : "obstacle.pt"
```

在*.pt文件的末尾，增加debug_param, 这些中间结果就会记录在制定的路径中。

```
 debug_param{
  track_out_file: "/apollo/debug_output/track_out_file.txt"
  camera2world_out_file: "/apollo/debug_output/camera2world_out_file.txt"
  lane_out_dir: "/apollo/debug_output/lane_out"
  calibration_out_dir: "/apollo/debug_output/calibration_out"
  detection_out_dir: "/apollo/debug_output/detection_out"
  detect_feature_dir: "/apollo/debug_output/detect_feature"
  tracked_detection_out_dir:"/apollo/debug_output/tracked_detection_out"
}
```

## 3.回灌测试

### mainboard

```
mainboard -d XXX/dag_streaming_perception_dev_kit_camera.dag -r bag
```

其中-d后面跟的XXX.dag是前面对应的感知子系统运行component的bag文件， -r 跟的 bag用以表示在模拟回灌模式下测试运行。


### cyber_recorder play

播放回灌 cyber_recorder play -f /data  播放数据包（bag数据包），关闭感知输出

```
cyber_recorder play -f /data/apollo/data/bag/2022-03-09-16-01-29/20220309160129.record.* -k /apollo/perception/obstacles
cyber_recorder play -f /data/apollo/data/bag/2022-03-02-15-49-38/20220302154938.record.*  -k /apollo/perception/obstacles
```

## debug调试

### 设置日志打印级别
 可设置如下环境变量
  e
   (INFO)  (  export GLOG_minloglevel=1 WARN)
  export GLOG_alsologtostderr=1
  export GLOG_colorlogtostderr=1
  export GLOG_v=4
### 解析LIDAR点云输入

修改配置：modules/tools/record_parse_save/parser_params.yaml 中filepath地址为对应的recorder 文件地址

```
    records:
    filepath: /apollo/data/record_files/2019-04-22-14-27-05/2019-04-22-14-27-05_records/

parse: lidar
# use one of the following options or add more:
  # lidar
  # radar
  # camera
  # odometry

lidar:     # for velodyne vls-128 lidar
  channel_name: /sensor/velodyne16/back/PointCloud2
  out_folder_extn: neolix_hesai
  timestamp_file_extn: neolix_hesai_timestamp.txt

radar:     # for ARS-408 radar mounted in front
  channel_name: /apollo/sensor/radar/front
  out_folder_extn: _radar_conti408_front
  timestamp_file_extn: _radar_conti408_front_timestamp.txt

camera:   # for 6mm camera mounted in front
  channel_name: /sensor/camera/front_6mm/image  #/sensor/camera/front_6mm/image
  out_folder_extn: neolix_camera_6mm_front
  timestamp_file_extn: neolix_camera_6mm_front_timestamp.txt

odometry:
  channel_name: /sensor/novatel/Odometry
  out_folder_extn: _odometry
  timestamp_file_extn: _odometry_timestamp.txt

```

执行
```
执行 ./bazel-bin/modules/tools/record_parse_save/record_parse_save
```

### 解析视频输入
同上，保存为jpeg