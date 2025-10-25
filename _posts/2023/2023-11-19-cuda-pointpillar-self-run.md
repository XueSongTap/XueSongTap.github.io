---
layout: article
title: CUDA-Pointpillar 配置
tags: debian linux
---


## Unknown CMake command "cuda_add_library".报错

Unknown CMake command "cuda_add_library".

发现find_package(CUDA)没有被调用， 导致cuda_add_library()命令没有被识别。开启调用即可。

如果需要找到，正确链接 /usr/local/cuda 即可




##  报错，找不到libnvinfer_plugin.so.8

```
[11/18/2023-17:14:14] [E] Uncaught exception detected: Unable to open library: libnvinfer_plugin.so.8 due to libcublas.so.12: cannot open shared object file: No such file or directory
&&&& FAILED TensorRT.trtexec [TensorRT v8601] # /usr/src/tensorrt/bin/trtexec --onnx=./model/backbone.onnx --fp16 --plugins=build/libpointpillar_core.so --saveEngine=./model/backbone.plan --inputIOFormats=fp16:chw,int32:chw,int32:chw --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed
root@yxc-MS-7B89:/home/yxc/code/for_new_cuda_pointpillar/CUDA-PointPillars# export LD_LIBRARY_PATH^C
```

发现不是找不到 libnvinfer_plugin.so.8 而是找不到 libcublas.so.12

find 发现是 libcublas.so.12在 /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublas.so.12

cuda 路径下

拷贝过去即可：

```bash
cp /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublas.so.12 /usr/lib/x86_64-linux-gnu/

cp /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/lib/x86_64-linux-gnu/
```


## 成功build


```bash
[11/18/2023-17:33:59] [I] 
[11/18/2023-17:34:03] [I] 
[11/18/2023-17:34:03] [I] === Profile (1434 iterations ) ===
[11/18/2023-17:34:03] [I]    Time(ms)     Avg.(ms)   Median(ms)   Time(%)   Layer
[11/18/2023-17:34:03] [I]      486.12       0.3390       0.3389      21.0   {ForeignNode[771 + (Unnamed Layer* 1) [Shuffle]...dummy_gather_]}
[11/18/2023-17:34:03] [I]      130.88       0.0913       0.0911       5.7   PPScatter_0
[11/18/2023-17:34:03] [I]      109.12       0.0761       0.0758       4.7   Conv_235 + Relu_236
[11/18/2023-17:34:03] [I]       84.73       0.0591       0.0594       3.7   Conv_237 + Relu_238
[11/18/2023-17:34:03] [I]       85.48       0.0596       0.0594       3.7   Conv_239 + Relu_240
[11/18/2023-17:34:03] [I]       84.65       0.0590       0.0594       3.7   Conv_241 + Relu_242
[11/18/2023-17:34:03] [I]       50.08       0.0349       0.0348       2.2   Conv_261 + Relu_262
[11/18/2023-17:34:03] [I]      146.96       0.1025       0.1024       6.4   ConvTranspose_243 + BatchNormalization_244 + Relu_245
[11/18/2023-17:34:03] [I]       73.25       0.0511       0.0512       3.2   Conv_263 + Relu_264
[11/18/2023-17:34:03] [I]       70.12       0.0489       0.0492       3.0   Conv_265 + Relu_266
[11/18/2023-17:34:03] [I]       69.59       0.0485       0.0481       3.0   Conv_267 + Relu_268
[11/18/2023-17:34:03] [I]       69.79       0.0487       0.0489       3.0   Conv_269 + Relu_270
[11/18/2023-17:34:03] [I]       69.57       0.0485       0.0481       3.0   Conv_271 + Relu_272
[11/18/2023-17:34:03] [I]       40.04       0.0279       0.0276       1.7   Conv_291 + Relu_292
[11/18/2023-17:34:03] [I]       74.68       0.0521       0.0522       3.2   ConvTranspose_273 + BatchNormalization_274 + Relu_275
[11/18/2023-17:34:03] [I]       68.07       0.0475       0.0471       2.9   Conv_293 + Relu_294
[11/18/2023-17:34:03] [I]       65.36       0.0456       0.0453       2.8   Conv_295 + Relu_296
[11/18/2023-17:34:03] [I]       65.52       0.0457       0.0461       2.8   Conv_297 + Relu_298
[11/18/2023-17:34:03] [I]       65.31       0.0455       0.0452       2.8   Conv_299 + Relu_300
[11/18/2023-17:34:03] [I]       65.62       0.0458       0.0461       2.8   Conv_301 + Relu_302
[11/18/2023-17:34:03] [I]       96.28       0.0671       0.0675       4.2   ConvTranspose_303 + BatchNormalization_304 + Relu_305
[11/18/2023-17:34:03] [I]      117.43       0.0819       0.0819       5.1   Conv_311 || Conv_308 || Conv_307
[11/18/2023-17:34:03] [I]       15.43       0.0108       0.0113       0.7   Transpose_312
[11/18/2023-17:34:03] [I]        7.58       0.0053       0.0051       0.3   Reformatting CopyNode for Output Tensor 0 to Transpose_312
[11/18/2023-17:34:03] [I]       33.89       0.0236       0.0236       1.5   Transpose_310
[11/18/2023-17:34:03] [I]       31.68       0.0221       0.0225       1.4   Reformatting CopyNode for Output Tensor 0 to Transpose_310
[11/18/2023-17:34:03] [I]       20.70       0.0144       0.0143       0.9   Transpose_309
[11/18/2023-17:34:03] [I]       11.46       0.0080       0.0082       0.5   Reformatting CopyNode for Output Tensor 0 to Transpose_309
[11/18/2023-17:34:03] [I]     2309.39       1.6105       1.6100     100.0   Total
[11/18/2023-17:34:03] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8601] # /usr/src/tensorrt/bin/trtexec --onnx=./model/backbone.onnx --fp16 --plugins=build/libpointpillar_core.so --saveEngine=./model/backbone.plan --inputIOFormats=fp16:chw,int32:chw,int32:chw --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed
```

## 成功运行 

```bash
root@yxc-MS-7B89:/home/yxc/code/for_new_cuda_pointpillar/CUDA-PointPillars/build# ./pointpillar ../data/  . --timer

GPU has cuda devices: 1
----device id: 0 info----
  GPU : NVIDIA GeForce RTX 3080 
  Capbility: 8.6
  Global memory: 10006MB
  Const memory: 64KB
  SM in a block: 48KB
  warp size: 32
  threads in a block: 1024
  block dim: (1024,1024,64)
  grid dim: (2147483647,65535,65535)

Total 10
------------------------------------------------------
Lidar Backbone 🌱 is Static Shape model
Inputs: 3
        0.voxels : {10000 x 32 x 10} [Float16]
        1.voxel_idxs : {10000 x 4} [Int32]
        2.voxel_num : {1} [Int32]
Outputs: 3
        0.cls_preds : {1 x 248 x 216 x 18} [Float32]
        1.box_preds : {1 x 248 x 216 x 42} [Float32]
        2.dir_cls_preds : {1 x 248 x 216 x 12} [Float32]
------------------------------------------------------

<<<<<<<<<<<
Load file: ../data/000003.bin
Lidar points count: 18911
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.12493 ms
[⏰ Lidar Voxelization]:        0.07690 ms
[⏰ Lidar Backbone & Head]:     2.70435 ms
[⏰ Lidar Decoder + NMS]:       3.06573 ms
Total: 5.847 ms
=============================================
Detections after NMS: 5
Saved prediction in: .000003.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000000.bin
Lidar points count: 20285
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07651 ms
[⏰ Lidar Voxelization]:        0.03757 ms
[⏰ Lidar Backbone & Head]:     1.69472 ms
[⏰ Lidar Decoder + NMS]:       2.92045 ms
Total: 4.653 ms
=============================================
Detections after NMS: 8
Saved prediction in: .000000.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000006.bin
Lidar points count: 19473
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07597 ms
[⏰ Lidar Voxelization]:        0.04058 ms
[⏰ Lidar Backbone & Head]:     1.70496 ms
[⏰ Lidar Decoder + NMS]:       3.04742 ms
Total: 4.793 ms
=============================================
Detections after NMS: 18
Saved prediction in: .000006.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000008.bin
Lidar points count: 17238
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.06371 ms
[⏰ Lidar Voxelization]:        0.03725 ms
[⏰ Lidar Backbone & Head]:     1.69165 ms
[⏰ Lidar Decoder + NMS]:       2.99734 ms
Total: 4.726 ms
=============================================
Detections after NMS: 24
Saved prediction in: .000008.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000005.bin
Lidar points count: 19962
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07789 ms
[⏰ Lidar Voxelization]:        0.04506 ms
[⏰ Lidar Backbone & Head]:     1.70496 ms
[⏰ Lidar Decoder + NMS]:       3.01453 ms
Total: 4.765 ms
=============================================
Detections after NMS: 8
Saved prediction in: .000005.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000004.bin
Lidar points count: 19063
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.06710 ms
[⏰ Lidar Voxelization]:        0.04054 ms
[⏰ Lidar Backbone & Head]:     1.70189 ms
[⏰ Lidar Decoder + NMS]:       3.00029 ms
Total: 4.743 ms
=============================================
Detections after NMS: 15
Saved prediction in: .000004.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000002.bin
Lidar points count: 20210
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07466 ms
[⏰ Lidar Voxelization]:        0.03731 ms
[⏰ Lidar Backbone & Head]:     1.69472 ms
[⏰ Lidar Decoder + NMS]:       2.95325 ms
Total: 4.685 ms
=============================================
Detections after NMS: 14
Saved prediction in: .000002.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000007.bin
Lidar points count: 19423
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07462 ms
[⏰ Lidar Voxelization]:        0.04138 ms
[⏰ Lidar Backbone & Head]:     1.70496 ms
[⏰ Lidar Decoder + NMS]:       3.06074 ms
Total: 4.807 ms
=============================================
Detections after NMS: 12
Saved prediction in: .000007.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000009.bin
Lidar points count: 19411
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07024 ms
[⏰ Lidar Voxelization]:        0.04096 ms
[⏰ Lidar Backbone & Head]:     1.70291 ms
[⏰ Lidar Decoder + NMS]:       3.11091 ms
Total: 4.855 ms
=============================================
Detections after NMS: 13
Saved prediction in: .000009.txt
>>>>>>>>>>>

<<<<<<<<<<<
Load file: ../data/000001.bin
Lidar points count: 18630
==================PointPillars===================
[⏰ [NoSt] CopyLidar]:  0.07107 ms
[⏰ Lidar Voxelization]:        0.04045 ms
[⏰ Lidar Backbone & Head]:     1.70086 ms
[⏰ Lidar Decoder + NMS]:       3.13741 ms
Total: 4.879 ms
=============================================
Detections after NMS: 10
Saved prediction in: .000001.txt
>>>>>>>>>>>
root@yxc-MS-7B89:/home/yxc/code/for_new_cuda_pointpillar/CUDA-PointPillars/build# 
```


参考

https://www.cnblogs.com/zjutzz/p/12094762.html


https://zhuanlan.zhihu.com/p/657200184