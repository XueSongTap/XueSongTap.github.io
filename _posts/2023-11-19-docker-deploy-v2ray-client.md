---
layout: articles
title: 基于Docker容器化部署V2Ray：网络代理配置完整指南
tags: docker v2ray proxy
---


## docker 部署 v2ray 客户端

```bash
docker pull v2ray/official


mkdir /etc/v2ray
mkdir /var/log/v2ray


vi /etc/v2ray/config.json
```

这个json可以从gui的客户端比如win的客户端里直接转换出来


config.json中的，记得把listen参数进行修改.
    "listen":"127.0.0.1", 确保是 "0.0.0.0"或者删除这一行.

运行
```bash
docker run \
--restart=always \
--name=v2ray \
-v /etc/v2ray/:/etc/v2ray/ \
-v /var/log/v2ray:/var/log/v2ray \
-i -t -d \
-p 10808:10808 \
-p 10809:10809 \
v2ray/official:latest
```

v2ray的http 端口是socks 的+1 ,记得两个端口都放开

```bash

export http_proxy=http://127.0.0.1:10809
export https_proxy=$http_proxy
```

## 参考：

https://www.youtube.com/watch?v=ZJb9OVv4mRk&ab_channel=TinMark

https://toutyrater.github.io/app/docker-deploy-v2ray.html

https://einverne.github.io/post/2018/01/v2ray.html