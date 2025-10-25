---
layout: article
title: Docker BuildKit故障排查问题分析
tags: docker buildkit
---
## 问题
docker build 过程中 build kit报错
```
yxc@yxc-MS-7B89:~/code/3/vllm$ docker build -t yxc.vllm.cuda.1 .
Sending build context to Docker daemon  28.02MB
Step 1/54 : FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dev
 ---> 963712b8747f
Step 2/54 : RUN apt-get update -y     && apt-get install -y python3-pip git
 ---> Using cache
 ---> 50ac8ffcba92
Step 3/54 : RUN ldconfig /usr/local/cuda-12.1/compat/
 ---> Using cache
 ---> f52028037b08
Step 4/54 : WORKDIR /workspace
 ---> Using cache
 ---> f9e21e0763d2
Step 5/54 : COPY requirements-common.txt requirements-common.txt
 ---> Using cache
 ---> 48d7a48a1008
Step 6/54 : COPY requirements-cuda.txt requirements-cuda.txt
 ---> Using cache
 ---> 3d00035d3684
Step 7/54 : RUN --mount=type=cache,target=/root/.cache/pip     pip install -r requirements-cuda.txt
the --mount option requires BuildKit. Refer to https://docs.docker.com/go/buildkit/ to learn how to build images with BuildKit enabled
yxc@yxc-MS-7B89:~/code/3/vllm$ 
```


## 解决：

这个问题提，“the --mount option requires BuildKit”，意味着你使用了--mount选项，但是没有启用Docker的BuildKit功能。BuildKit是Docker的一个现代化镜像构建工具，它提供了一些增强功能，如构建缓存的管理。

Docker 18.09 开始，是自带，但是不是默认开启

19.03+ 开始可以 docker build x 启用

### 临时启用

```
DOCKER_BUILDKIT=1 docker build -t yxc.vllm.cuda.1 .
```

### 永久启用


修改Docker的配置文件来永久启用BuildKit。在Docker配置文件`/etc/docker/daemon.json`中添加以下内容（如果文件不存在，则需要创建它）：
   ```json
   {
     "features": {
       "buildkit": true
     }
   }
   ```
