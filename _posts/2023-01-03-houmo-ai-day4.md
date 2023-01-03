---
layout: articles
title: houmo.ai day4
tags: intern
---

# houmo.ai day4

## 传感器标定任务

1.掌握相机内参标定、以及与激光雷达联合标定的方法

​2.熟悉apollo中传感器标定文件的存放位置，传感器参数文件的作用。​


3.完成传感器标定及对应文档梳理。​采集的数据位置 ：​​10.64.32.51:/record_1216_left_front&record_1216_right_front​参考文档：​标定文档_v2​


## Apollo record file 解析


apollo使用不同于rosbag的`record.0000`后缀的record file

matlab的`rosbag`函数无法正常读取，估计ros系统下也无法正常读取

猜测是apollo用protobuf交换数据，需要使用专门的parse脚本，进行protobuf消息的反序列化才可以正常解析

查看record file 信息
```bash
[xiaochuan@in-dev-docker:/apollo]$ cyber_recorder info  /apollo/data/record_files/record_1216_left_front/20221216160933.record.00000 
```
输出结果：
```
record_file:    /apollo/data/record_files/record_1216_left_front/20221216160933.record.00000
version:        1.0
duration:       11.996691 Seconds
begin_time:     2022-12-16-16:09:33
end_time:       2022-12-16-16:09:45
size:           332581359 Bytes (317.174286 MB)
is_complete:    true
message_number: 232
channel_number: 2
channel_info:   
                /apollo/sensor/lslidarCH128X1/left/PointCloud2         121 messages: apollo.drivers.PointCloud
                /apollo/sensor/camera/front_left/image/compressed      111 messages: apollo.drivers.CompressedImage
```

## apollo record 提取

`/apollo/modules/tools/record_parse_save` 文件夹下有脚本用于进行record的提取

```
total 72
drwxr-xr-x  4 xiaochuan root 4096 Jan  3 14:09 ./
drwxr-xr-x 37 xiaochuan root 4096 Dec 28 15:08 ../
-rw-r--r--  1 xiaochuan root 2246 Dec 28 15:08 BUILD
-rw-r--r--  1 xiaochuan root 3306 Dec 28 14:14 README.md
drwxr-xr-x  2 xiaochuan root 4096 Jan  3 15:56 __pycache__/
-rw-r--r--  1 xiaochuan root 1599 Dec 28 15:08 display_odom.py
drwxr-xr-x  2 xiaochuan root 4096 Dec 28 14:14 images/
-rwxr-xr-x  1 xiaochuan root 2571 Dec 28 15:08 parse_best_pose.py*
-rw-r--r--  1 xiaochuan root 3145 Jan  3 15:52 parse_camera.py
-rwxr-xr-x  1 xiaochuan root 2441 Dec 28 15:08 parse_heading.py*
-rw-r--r--  1 xiaochuan root 3597 Jan  3 16:32 parse_lidar.py
-rwxr-xr-x  1 xiaochuan root 3359 Dec 28 15:08 parse_localization.py*
-rwxr-xr-x  1 xiaochuan root 3164 Dec 28 15:08 parse_odometry.py*
-rw-r--r--  1 xiaochuan root 4654 Dec 28 14:14 parse_radar.py
-rwxr-xr-x  1 xiaochuan root  928 Jan  3 16:29 parser_params.yaml*
-rw-r--r--  1 xiaochuan root 5360 Jan  3 16:32 record_parse_save.py
```

根据实际的record的info，主要对`parser_params.yaml`进行修改

```yaml
# record文件夹路径
records:
  filepath: /data/xiaochuan/record_files/record_1216_left_front/

# 选择需要的参数
parse: lidar
# use one of the following options or add more:
  # lidar
  # radar
  # camera
  # odometry

# 定义参数所在的channel
lidar:     
  channel_name: /apollo/sensor/lslidarCH128X1/left/PointCloud2
  out_folder_extn: _lidar_CH128X1
  timestamp_file_extn: _lidar_CH128X1_timestamp.txt

radar:     
  channel_name: /apollo/sensor/radar/front
  out_folder_extn: _radar_conti408_front
  timestamp_file_extn: _radar_conti408_front_timestamp.txt

# camare 可能有两种情况，compressed或者不压缩的，需要在record_parse_save.py里面选择使用相应的函数
camera:   
  channel_name: /apollo/sensor/camera/front_left/image/compressed  
  out_folder_extn: _camera_front_left
  timestamp_file_extn: _camera_front_left_timestamp.txt

odometry:
  channel_name: /sensor/novatel/Odometry
  out_folder_extn: _odometry
  timestamp_file_extn: _odometry_timestamp.txt
```

配置好yaml后，执行
```bash
./bazel-bin/modules/tools/record_parse_save/record_parse_save
```
可以得到不同数据解析成jpeg、pcd等



## 相机内参标定

### 相关概念
内参:相机自身特性相关的参数、比如相机的焦距、像素大小,相机内参矩阵反应了相机自身的属性

### 标定方法

https://ww2.mathworks.cn/help/vision/ug/using-the-single-camera-calibrator-app.html

命令行`cameraCalibrator`

### 标定流程

- A. 从工具栏选择Add Images， 添加多张图片，matlab会自动筛选出能用的图片，不合格的图片会被reject。
- B. calibrate之后，可以选择show undistorted，查看图片畸变矫正的效果从而间接测试矫正的效果。重投影误差大的图片，直接remove掉
- C. 点击Export Camera Params  可以把标定结果导出到workspace，然后再workspace中右键另存为mat文件 
- D. 测试标定出来的内参的验证方法：
    1.Show Undistorted 可以验证参数的好坏，主要看图片边缘的弯曲是否被正确的矫正成直线
    2.还可以把matlab标定出来的参数，送进python脚本测试畸变矫正效果
### 内参验证

```python
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

#自带的
#mtx =  np.array([[1996.241246, 0.000000, 953.758716],
#                 [0.000000, 1993.372273, 500.599692],
#                 [0.000000, 0.000000, 1.000000]])
# zidai
#dist = np.array([-0.571007, 0.319193, 0.000497, -0.002258, 0.000000])

#mtx =  np.array([[1851.321, 0.000000, 911.641],
#                 [0.000000, 1839.062, 581.810],
#                 [0.000000, 0.000000, 1.000000]])
#dist = np.array([-0.621481, 1.126058, -0.013302, 0.005903, 0.000000])


#mtx =  np.array([[4238.685, 0.000000, 958.099],
#                 [0.000000, 4274.294, 539.872],
#                 [0.000000, 0.000000, 1.000000]])
#dist = np.array([-2.595953, -5.390706, -0.033212, 0.019096, 0.000000])


mtx =  np.array([[3113.585, 0.000000, 995.8281],
                 [0.000000, 3140.730, 504.3034],
                 [0.000000, 0.000000, 1.000000]])
dist = np.array([-1.22554, 1.1119, 0, 0, 0.000000])





#mtx =  np.array([[960.956, 0.000000, 1315.005],
#                [0.000000, 967.572, 541.246],
#                 [0.000000, 0.000000, 1.000000]])
#dist = np.array([ -0.007988 , -0.085829 , 0.001559 , -0.024983, 0.000000])




# Define distortion coefficients





def test():
	"""
	read the pickle file on disk and implement undistor on image
	show the oringal/undistort image
	"""
	print("Reading the sample image...")
	#img = cv2.imread('D:\\szj_project\\apollo\\_camera_6mm_front_aj\\image_3659_288182.jpeg')
	img = cv2.imread('D:\\szj_project\\apollo\\_camera_6mm_front_aj\\2.jpeg')
	img_size = (img.shape[1],img.shape[0])
	print('img_size is',img_size)
	w = img_size[0]
	h = img_size[1]
	print("w is",w)
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1, (w,h))
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
	dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# Visualize undistortion
	print("Visulize the result...")
	f, (ax1,ax2) = plt.subplots(2,1, figsize=(20,20))
	ax1.imshow(img), ax1.set_title('Original Image', fontsize=15)
	ax2.imshow(dst), ax2.set_title('Undistored Image', fontsize=15)
	ax1.grid()
	ax2.grid()
	plt.show()
test()
```




## 激光雷达联合标定 
### 标定板制作
按下图所示，在KT板上，挖一个菱形的孔

## apollo 中传感器标定文件的存放位置    