---
layout: article
title: Dockerfile构建加速：使用国内apt、pip镜像源加速容器构建
tags: docker
---


修改dockerfile 用国内源：

替换ubuntu源：
```bash
RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu/g' /etc/apt/sources.list
```

替换pip源
```bash
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

